"""Kiwi worker: dynamically assigned split-model execution and P2P forwarding."""

from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import httpx
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, Request, Response
from safetensors.torch import load as safetensors_load
from safetensors.torch import save as safetensors_save
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

import ipaddress
from urllib.parse import urlparse
from transport import KiwiTransport, create_transport


GATEWAY_URL = os.getenv("GATEWAY_URL", "http://gateway:8080").rstrip("/")
NODE_NAME = os.getenv("NODE_NAME", "worker")
NODE_URL = os.getenv("NODE_URL", "http://worker:8000").rstrip("/")

GPU_MEMORY_RESERVE_MB = float(os.getenv("GPU_MEMORY_RESERVE_MB", "512"))

P2P_TRANSPORT = os.getenv("KIWI_P2P_TRANSPORT", "http").lower()
P2P_RELAY_PEERS = os.getenv("KIWI_RELAY_PEERS", "")
P2P_PORT = int(os.getenv("KIWI_P2P_PORT", "4001"))

transport: KiwiTransport | None = None


# ── Populated from gateway assignment ────────────────────────────────────────────

assigned_model_id: str | None = None
layer_range: tuple[int, int] | None = None
is_first_stage: bool = False
is_last_stage: bool = False
model: Any | None = None
inference_device: str = "cpu"

model_lock = asyncio.Lock()
HEARTBEAT_INTERVAL_SECONDS = float(os.getenv("HEARTBEAT_INTERVAL_SECONDS", "5"))

# ── Training state ──────────────────────────────────────────────────────────────
# LR = 5e-3
LR = float(os.getenv("KIWI_LR", "1e-5"))
optimizer: torch.optim.Optimizer | None = None
MODEL_DTYPE = os.getenv("MODEL_DTYPE", "bfloat16")

training_steps: int = 0
_last_step_request_id: str = ""

# ── Async-mesh round state (§3.4) ───────────────────────────────────────────────
T_ROUND = float(os.getenv("KIWI_T_ROUND", "30"))
current_round: int = 0          # highest round already merged/closed
layer_peers: list[str] = []     # healthy same-stage peers, from heartbeat
steps_since_round: int = 0
# Adapter LINEAGE step — advances on local training steps AND on merges
# (max of merged contributors' steps). This, not training_steps, is what the
# version-skew guard compares: a spare that never trained still holds a
# current adapter because it received every round average. Judging it by its
# local optimizer step count would make peers drop its payloads forever
# after promotion (found by scratch/test_stage6_mesh.py).
adapter_step: int = 0
_round_buffers: dict[int, list[tuple[str, int, dict[str, torch.Tensor]]]] = {}
_round_expect: dict[int, int] = {}
_round_events: dict[int, asyncio.Event] = {}
# Sparsity in force for a round, as announced by the gateway's plan. Used
# only to reject a payload whose sender disagreed (Phase 3.2); the merge's
# length check is the authoritative guard.
_round_p: dict[int, float] = {}

# ── Outer optimizer state (Phase 4, DiLoCo) ─────────────────────────────────
# The anchor is this worker's adapter at the last completed dense round; the
# momentum buffer is LOCAL and never transmitted, same invariant as the Adam
# moments (§4.2).
OUTER_LR = float(os.getenv("KIWI_OUTER_LR", "0.7"))
OUTER_MU = float(os.getenv("KIWI_OUTER_MU", "0.9"))
_outer_anchor: dict[str, torch.Tensor] | None = None
_outer_buf: dict[str, torch.Tensor] = {}

# ── Backward pass tensor caches (keyed by request_id) ───────────────────────────
cached_inputs: dict[str, torch.Tensor] = {}
cached_outputs: dict[str, torch.Tensor] = {}
last_predictions: dict[str, torch.Tensor] = {}


def _dtype() -> torch.dtype:
    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}.get(
        MODEL_DTYPE, torch.bfloat16
    )


# ── GPU / VRAM helpers ──────────────────────────────────────────────────────────


def _detect_device() -> tuple[str, float]:
    """Return (device_string, total_vram_gb).  Falls back to CPU gracefully."""
    override = os.getenv("KIWI_VRAM_GB_OVERRIDE")
    if override is not None:
        try:
            return ("cuda" if torch.cuda.is_available() else "cpu"), float(override)
        except ValueError:
            pass

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_gb = round(props.total_memory / (1024 ** 3), 2)
        return "cuda", vram_gb
    return "cpu", 0.0


def _estimate_layer_memory_mb(loaded_model: Any, total_layers: int = 0) -> float:
    """Estimate per-layer GPU memory (MB) from total model parameter size."""
    total_params = sum(p.numel() * p.element_size() for p in loaded_model.parameters())
    # Add ~20% overhead for CUDA context fragmentation.
    return (total_params / total_layers * 1.2) / (1024 ** 2)


def _compute_gpu_layers(loaded_model: Any, num_assigned_layers: int, total_layers: int = 0) -> int:
    """Decide how many of the assigned layers fit in available VRAM."""
    if not torch.cuda.is_available():
        return 0
    free_mem_mb = torch.cuda.mem_get_info(0)[0] / (1024 ** 2)
    usable_mb = free_mem_mb - GPU_MEMORY_RESERVE_MB
    if usable_mb <= 0:
        return 0
    per_layer_mb = _estimate_layer_memory_mb(loaded_model, total_layers)
    gpu_layers = int(usable_mb // per_layer_mb)
    return min(gpu_layers, num_assigned_layers)


def _place_model_on_device(loaded_model: Any, num_assigned_layers: int, total_layers: int = 0) -> str:
    """Move the model (or a subset of its layers) onto the best available device.

    Returns the primary inference device string ("cuda" or "cpu").
    """
    if not torch.cuda.is_available():
        return "cpu"

    gpu_layers = _compute_gpu_layers(loaded_model, num_assigned_layers, total_layers)

    if gpu_layers == 0:
        return "cpu"

    if gpu_layers >= num_assigned_layers:
        loaded_model.to("cuda")
        return "cuda"

    # Partial offload: first `gpu_layers` decoder layers on CUDA, rest on CPU.
    layers = loaded_model.model.layers
    for i, layer in enumerate(layers):
        if i < gpu_layers:
            layer.to("cuda")
        else:
            layer.to("cpu")
    tail_device = "cuda" if gpu_layers >= num_assigned_layers else "cpu"
    if hasattr(loaded_model.model, "norm") and not isinstance(loaded_model.model.norm, torch.nn.Identity):
        loaded_model.model.norm.to(tail_device)
    if hasattr(loaded_model, "lm_head") and not isinstance(loaded_model.lm_head, torch.nn.Identity):
        loaded_model.lm_head.to(tail_device)
    if hasattr(loaded_model.model, "embed_tokens") and not isinstance(loaded_model.model.embed_tokens, torch.nn.Identity):
        loaded_model.model.embed_tokens.to("cuda")
    return "cuda"


# ── Registration / lifecycle ────────────────────────────────────────────────────


async def _discover_bootstrap_peers() -> list[str]:
    """Query gateway via HTTP to dynamically discover P2P bootstrap peer multiaddresses."""
    if P2P_RELAY_PEERS:
        return [p.strip() for p in P2P_RELAY_PEERS.split(",") if p.strip()]

    # If no manual relay peers configured, fetch dynamically from gateway HTTP endpoint
    print(f"[{NODE_NAME}] KIWI_RELAY_PEERS not configured. Discovering bootstrap peers from gateway...", flush=True)
    bootstrap_maddrs = []

    # Retry loop since gateway might be booting up concurrently
    for attempt in range(60):
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{GATEWAY_URL}/p2p/bootstrap")
                response.raise_for_status()
                data = response.json()

                peer_id = data["peer_id"]
                p2p_port = data["p2p_port"]
                visible_maddrs = data.get("visible_maddrs", [])

                # Add visible addresses from gateway response
                for maddr in visible_maddrs:
                    if maddr not in bootstrap_maddrs:
                        bootstrap_maddrs.append(maddr)

                # Construct address from GATEWAY_URL host
                parsed = urlparse(GATEWAY_URL)
                if parsed.hostname:
                    host = parsed.hostname
                    # Check if host is IP address
                    import socket
                    is_ip = False
                    try:
                        ipaddress.ip_address(host)
                        is_ip = True
                    except ValueError:
                        try:
                            host = socket.gethostbyname(host)
                            is_ip = True
                        except socket.gaierror:
                            pass

                    if is_ip:
                        ip = ipaddress.ip_address(host)
                        if ip.version == 6:
                            maddr_construct = f"/ip6/{host}/tcp/{p2p_port}/p2p/{peer_id}"
                        else:
                            maddr_construct = f"/ip4/{host}/tcp/{p2p_port}/p2p/{peer_id}"
                    else:
                        maddr_construct = f"/dns4/{host}/tcp/{p2p_port}/p2p/{peer_id}"

                    if maddr_construct not in bootstrap_maddrs:
                        bootstrap_maddrs.append(maddr_construct)

                print(f"[{NODE_NAME}] Discovered bootstrap peers: {bootstrap_maddrs}", flush=True)
                return bootstrap_maddrs
        except Exception:
            # Sleep and retry
            await asyncio.sleep(min(1 + attempt * 0.25, 5))

    raise RuntimeError(f"{NODE_NAME} failed to discover P2P bootstrap peers from gateway")


async def _register() -> dict[str, Any]:
    _, vram_gb = _detect_device()
    peer_id = ""
    if P2P_TRANSPORT == "p2p" and transport is not None:
        try:
            peer_id = await transport.get_address()
        except Exception:
            pass

    payload = {
        "node_name": NODE_NAME,
        "node_url": NODE_URL,
        "peer_id": peer_id,
        "p2p_port": P2P_PORT,
        "cpu_cores": os.cpu_count() or 1,
        "vram_gb": vram_gb,
    }
    for attempt in range(60):
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(f"{GATEWAY_URL}/register", json=payload)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError:
            await asyncio.sleep(min(1 + attempt * 0.25, 5))
    raise RuntimeError(f"{NODE_NAME} could not register with gateway")


def _load_assigned_model(assignment: dict[str, Any]) -> Any:
    """Load once on startup, then retain only this worker's assigned block."""
    global inference_device

    model_id = assignment["model_id"]
    first_layer, last_layer = assignment["layers"]
    total_layers = assignment["total_layers"]
    num_assigned = last_layer - first_layer + 1

    _is_first = first_layer == 0
    _is_last = last_layer == total_layers - 1

    # Always load to CPU first, then selectively move to GPU.
    loaded_model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=_dtype(), device_map="cpu"
    )
    loaded_model.train()

    # Narrow the decoder stack to just our assigned layers.
    layers = loaded_model.model.layers
    selected_layers = torch.nn.ModuleList(layers[first_layer : last_layer + 1])
    # Renumber so the first layer starts at local index 0 regardless of global position.
    for local_index, decoder_layer in enumerate(selected_layers):
        if hasattr(decoder_layer, "self_attn"):
            decoder_layer.self_attn.layer_idx = local_index
    loaded_model.model.layers = selected_layers

    # Strip components this position doesn't need.
    if not _is_last:
        loaded_model.model.norm = torch.nn.Identity()
        loaded_model.lm_head = torch.nn.Identity()
    if not _is_first:
        loaded_model.model.embed_tokens = torch.nn.Identity()

    # ── Device placement ──
    inference_device = _place_model_on_device(loaded_model, num_assigned, total_layers)

    # ── Gradient checkpointing ──
    # Without this a worker stores every decoder layer's forward intermediates
    # for the backward pass. The MLP tensors dominate: 3 x (T x 4864 x 2) bytes
    # per layer, plus the saved input of each of the 7 LoRA adapters. At
    # T ~= 25 (the zorbulax fixture) that is ~1 MB per layer and invisible; at
    # T = 256 it is ~10 MB per layer, and a 10-layer first stage that also
    # carries the 272 MB embedding table exceeds the 0.9 GB per-worker cap and
    # dies mid-run with a CUDA OOM. Observed exactly that on alpaca1k.
    #
    # Checkpointing keeps only each layer's INPUT (T x 896 x 2, under 1 MB)
    # and recomputes the internals during backward. Costs roughly one extra
    # forward pass (~30% step time; wall clock gates nothing here) and changes
    # no gradient: torch.utils.checkpoint restores the RNG state, so the
    # recomputed LoRA dropout mask is identical to the original.
    #
    # NOTE: this does NOT help the last stage's logits. lm_head runs after the
    # decoder stack, outside the checkpointed region, and its [T, vocab]
    # output is genuinely needed by cross-entropy's backward. That is why
    # ALPACA_MAX_LENGTH is 256 rather than 512 — see orchestrator.py.
    if os.getenv("KIWI_GRAD_CHECKPOINT", "1") == "1":
        # Must come BEFORE get_peft_model. enable_input_require_grads installs
        # a forward hook on the input embeddings; the first stage's only input
        # is input_ids (an integer tensor that cannot require grad), so without
        # it torch.utils.checkpoint sees no grad-requiring input, warns
        # "None of the inputs have requires_grad=True", and silently produces
        # NO gradients for that stage — a silent failure, not a crash.
        loaded_model.enable_input_require_grads()
        loaded_model.gradient_checkpointing_enable(
            # Reentrant checkpointing is incompatible with the frozen base +
            # trainable-adapter split: it decides what to recompute from the
            # inputs' requires_grad, which is wrong when only inner params
            # need grads. The repo already settled on this in QLoRARunner.
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        # use_cache and checkpointing are mutually exclusive; transformers
        # warns and disables the cache anyway, so do it explicitly. Training
        # never uses the KV cache.
        loaded_model.config.use_cache = False
        print(f"[{NODE_NAME}] Gradient checkpointing ENABLED "
              f"(~30% slower per step, ~25x less activation memory)", flush=True)

    # ── Apply LoRA ──
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    loaded_model = get_peft_model(loaded_model, lora_config)

    # ── Optional: freeze LoRA A (ASYNC_MESH_DESIGN.md §3.5 mitigation) ──
    # Averaging LoRA factors is NOT averaging the updates they represent:
    #   mean(A)·mean(B) != mean(A·B)
    # because the cross terms A0·B1 and A1·B0 are spurious. If A is held at its
    # random init and never trained, every replica shares the same A, the cross
    # terms collapse, and weight averaging becomes EXACT at some cost in
    # capacity. Ship as a toggle and an ablation, never as the default.
    if os.getenv("KIWI_FREEZE_LORA_A", "0") == "1":
        frozen = 0
        for name, param in loaded_model.named_parameters():
            if "lora_A" in name:
                param.requires_grad = False
                frozen += 1
        print(f"[{NODE_NAME}] KIWI_FREEZE_LORA_A=1: froze {frozen} LoRA A tensors "
              f"(averaging is now exact; capacity reduced)", flush=True)

    loaded_model.print_trainable_parameters()

    position = (
        "SOLO" if _is_first and _is_last else
        "FIRST" if _is_first else
        "LAST" if _is_last else
        "MIDDLE"
    )
    device_label = inference_device.upper()
    if inference_device == "cuda":
        allocated = torch.cuda.memory_allocated(0) / (1024 ** 2)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        print(
            f"[{NODE_NAME}] {model_id} layers {first_layer}-{last_layer} "
            f"({position}) on {device_label}  "
            f"({allocated:.0f} MB / {total:.0f} MB VRAM)"
        )
    else:
        print(
            f"[{NODE_NAME}] {model_id} layers {first_layer}-{last_layer} "
            f"({position}) on {device_label}"
        )

    global optimizer
    trainable_params = [p for p in loaded_model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=LR)

    return loaded_model


async def _re_register() -> None:
    _, vram_gb = _detect_device()

    global transport
    peer_id = ""
    if P2P_TRANSPORT == "p2p" and transport is not None:
        try:
            # Dynamic re-discovery of new gateway bootstrap peers
            new_bootstrap_peers = await _discover_bootstrap_peers()
            print(f"[{NODE_NAME}] Restarting P2P transport with new bootstrap peers...")
            await transport.stop()
            transport = create_transport(
                transport_type=P2P_TRANSPORT,
                node_url=NODE_URL,
                initial_peers=new_bootstrap_peers,
                listen_port=P2P_PORT,
            )
            await transport.start()
            # Restore handlers
            transport.set_forward_handler(_execute_forward_pipeline)
            transport.set_backward_handler(_execute_backward_pipeline)
            transport.set_adapter_handler(_execute_adapter_pipeline)
            peer_id = await transport.get_address()
        except Exception as exc:
            print(f"[{NODE_NAME}] Failed to restart P2P transport: {exc}", flush=True)

    payload = {
        "node_name": NODE_NAME,
        "node_url": NODE_URL,
        "peer_id": peer_id,
        "p2p_port": P2P_PORT,
        "cpu_cores": os.cpu_count() or 1,
        "vram_gb": vram_gb,
        "model_id": assigned_model_id,
        "layers": list(layer_range) if layer_range else None,
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(f"{GATEWAY_URL}/register", json=payload)
            response.raise_for_status()
    except httpx.HTTPError:
        pass


async def _heartbeat_loop() -> None:
    """Ping the gateway periodically so it knows we're alive."""
    global layer_peers
    latency_ms = 0.0
    current_role = "unknown"  # Track role to detect promotion
    while True:
        await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)
        try:
            start_time = time.perf_counter()
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.post(
                    f"{GATEWAY_URL}/heartbeat",
                    json={
                        "node_name": NODE_NAME,
                        "node_url": NODE_URL,
                        "latency_ms": latency_ms,
                    },
                )
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                if response.status_code == 404:
                    await _re_register()
                elif response.status_code == 200:
                    data = response.json()
                    # Mesh peer discovery (§3.3): every healthy worker sharing
                    # this worker's stage identity. Replaces the standby-sync
                    # push path — spares receive the round average instead.
                    layer_peers = data.get("layer_peers", [])

                    # ── Spare → Active promotion detection ──
                    new_role = data.get("role", "unknown")
                    if current_role in ("standby", "spare") and new_role == "active" and model is not None:
                        print(f"[{NODE_NAME}] *** Promoted to ACTIVE! Resuming from last round average. ***", flush=True)
                        _log_recovery_bytes_metrics()
                    current_role = new_role
        except httpx.HTTPError:
            pass  # Gateway may be temporarily unreachable; keep retrying.


def _log_recovery_bytes_metrics() -> None:
    """Log recovery bytes metrics on spare promotion.

    What actually moves over the wire is `get_peft_model_state_dict` in the
    model dtype — 2 bytes/param in bf16. Adam moments stay local and are
    NEVER transmitted (mesh invariant §4.2), so they are excluded from both
    sides of the comparison. (An earlier version costed this at 10 bytes/param
    and overstated transfer by 5×.)
    """
    if model is None:
        return

    base_model = model.get_base_model()
    # All params (frozen base + LoRA adapters)
    base_params = sum(p.numel() for p in base_model.parameters())
    # Only trainable LoRA params
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    bytes_per_param = 2  # bf16 weights only — optimizer moments never move

    baseline_bytes = base_params * bytes_per_param
    adapter_snapshot_bytes = trainable_params * bytes_per_param

    baseline_mb = baseline_bytes / (1024 ** 2)
    adapter_snapshot_mb = adapter_snapshot_bytes / (1024 ** 2)
    ratio = baseline_bytes / adapter_snapshot_bytes if adapter_snapshot_bytes > 0 else float("inf")

    print(f"[{NODE_NAME}] Recovery Metrics - Promotion Recovery Cost: ~0 MB (pre-warmed by rounds)", flush=True)
    print(f"[{NODE_NAME}] Recovery Metrics - Per-round adapter transfer: {adapter_snapshot_mb:.2f} MB "
          f"({trainable_params:,} params × {bytes_per_param} bytes, moments never move)", flush=True)
    print(f"[{NODE_NAME}] Recovery Metrics - Full-weight baseline: {baseline_mb:.2f} MB "
          f"({base_params:,} params × {bytes_per_param} bytes)", flush=True)
    print(f"[{NODE_NAME}] Recovery Metrics - Transfer ratio: {ratio:.1f}x smaller than full-weight", flush=True)


async def _deregister() -> None:
    """Notify the gateway we're shutting down so it can evict us immediately."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            await client.post(
                f"{GATEWAY_URL}/deregister",
                json={"node_name": NODE_NAME, "node_url": NODE_URL},
            )
    except httpx.HTTPError:
        pass


def _enforce_vram_cap() -> None:
    """Actually cap allocation to KIWI_VRAM_GB_OVERRIDE (§3.8).

    Per-process, so it works whether or not workers share a device. Without
    this the override only feeds the gateway's placement arithmetic and the
    "X GB per worker" claim is notional.
    """
    cap_gb = os.getenv("KIWI_VRAM_GB_OVERRIDE")
    if not cap_gb or not torch.cuda.is_available():
        return
    try:
        total = torch.cuda.get_device_properties(0).total_memory
        fraction = min(1.0, float(cap_gb) * 1024**3 / total)
        torch.cuda.set_per_process_memory_fraction(fraction, 0)
        print(
            f"[{NODE_NAME}] VRAM cap enforced: {cap_gb} GB "
            f"(fraction {fraction:.4f} of {total / 1024**3:.1f} GB)",
            flush=True,
        )
    except (ValueError, RuntimeError) as error:
        print(f"[{NODE_NAME}] Could not enforce VRAM cap: {error}", flush=True)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global assigned_model_id, layer_range
    global is_first_stage, is_last_stage, model, transport

    _enforce_vram_cap()

    initial_peers = []
    if P2P_TRANSPORT == "p2p":
        initial_peers = await _discover_bootstrap_peers()

    transport = create_transport(
        transport_type=P2P_TRANSPORT,
        node_url=NODE_URL,
        initial_peers=initial_peers,
        listen_port=P2P_PORT,
    )
    await transport.start()
    transport.set_forward_handler(_execute_forward_pipeline)
    transport.set_backward_handler(_execute_backward_pipeline)
    transport.set_adapter_handler(_execute_adapter_pipeline)

    assignment = await _register()
    assigned_model_id = assignment["model_id"]
    layer_range = tuple(assignment["layers"])
    total_layers = assignment["total_layers"]
    is_first_stage = layer_range[0] == 0
    is_last_stage = layer_range[1] == total_layers - 1
    model = await asyncio.to_thread(_load_assigned_model, assignment)

    heartbeat_task = asyncio.create_task(_heartbeat_loop())
    yield
    heartbeat_task.cancel()
    await _deregister()
    await transport.stop()


app = FastAPI(title="Kiwi worker", lifespan=lifespan)


# ── Health ───────────────────────────────────────────────────────────────────────


@app.get("/health")
async def health() -> dict[str, Any]:
    gpu_info: dict[str, Any] = {"device": inference_device}
    if inference_device == "cuda" and torch.cuda.is_available():
        gpu_info["vram_total_mb"] = round(
            torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        )
        gpu_info["vram_allocated_mb"] = round(
            torch.cuda.memory_allocated(0) / (1024 ** 2)
        )
        gpu_info["vram_peak_mb"] = round(
            torch.cuda.max_memory_allocated(0) / (1024 ** 2)
        )
        gpu_info["gpu_name"] = torch.cuda.get_device_name(0)

    if is_first_stage and is_last_stage:
        position = "solo"
    elif is_first_stage:
        position = "first"
    elif is_last_stage:
        position = "last"
    else:
        position = "middle"

    return {
        "status": "ok" if model is not None else "initializing",
        "node_name": NODE_NAME,
        "model_id": assigned_model_id,
        "position": position,
        "layers": layer_range,
        "training_steps": training_steps,
        "adapter_step": adapter_step,
        "current_round": current_round,
        "gpu": gpu_info,
    }


# ── Forward helpers ─────────────────────────────────────────────────────────────


def _to_device(tensor: torch.Tensor) -> torch.Tensor:
    """Move a tensor to the worker's inference device."""
    return tensor.to(inference_device)


def _get_grad_norm() -> float:
    """Compute the L2 norm of gradients across all model parameters.

    Guards against None grads (e.g. Identity-replaced embed_tokens / lm_head).
    """
    assert model is not None
    total = sum(
        p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None
    )
    return (total ** 0.5).item()


def _run_backward_only(tensor: torch.Tensor, grad_in: torch.Tensor | None = None) -> float:
    """Run backward pass WITHOUT optimizer step.
    
    Computes gradients only. The optimizer step is deferred to _step_optimizer()
    to enable commit-on-success semantics: we only commit when downstream
    confirms the backward chain completed successfully.
    
    WARNING: The caller MUST hold `model_lock` before executing this function
    via `asyncio.to_thread`.
    """
    assert optimizer is not None
    optimizer.zero_grad()
    
    if grad_in is not None:
        tensor.backward(gradient=grad_in)
    else:
        tensor.backward()
        
    return _get_grad_norm()


def _step_optimizer() -> None:
    """Commit the optimizer step after confirming downstream success.
    
    WARNING: The caller MUST hold `model_lock` before executing this function
    via `asyncio.to_thread`.
    """
    assert optimizer is not None
    optimizer.step()


def _extract_failed_node_from_response(response_bytes: bytes) -> str | None:
    """Try to parse a failed_node from a transport response that may be an error JSON."""
    try:
        data = json.loads(response_bytes)
        if isinstance(data, dict) and "failed_node" in data:
            return data["failed_node"]
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
        pass
    return None


from peft import get_peft_model_state_dict, set_peft_model_state_dict

from mesh import (
    filter_skewed,
    merge_adapters,
    merge_sparse,
    outer_step,
    sparse_snapshot,
)


async def _execute_adapter_pipeline(headers: dict[str, str], body_bytes: bytes) -> bytes:
    """Inbound adapter payload.

    Two paths, selected by the presence of X-Round-ID:
      - round payload → buffer for the averaging round (merge, §3.4)
      - no round header → legacy full overwrite (bootstrap/recovery pull path)
    """
    if not model:
        raise RuntimeError("Worker model not ready")

    model_id = headers.get("x-model-id") or headers.get("X-Model-ID", "")
    if model_id != assigned_model_id:
        raise ValueError(f"Model ID mismatch: expected {assigned_model_id}, got {model_id}")

    round_header = headers.get("x-round-id") or headers.get("X-Round-ID", "")
    if round_header:
        return _buffer_round_payload(headers, body_bytes)

    tensors = safetensors_load(body_bytes)

    flat = torch.cat([v.flatten() for v in tensors.values()]) if tensors else torch.tensor([0.0])
    norm = flat.norm().item()

    async with model_lock:
        set_peft_model_state_dict(model, tensors)

    # Adopt the sender's adapter lineage (bootstrap pulls stamp X-Step).
    global adapter_step
    try:
        sender_step = int(headers.get("x-step") or headers.get("X-Step", "0") or 0)
        adapter_step = max(adapter_step, sender_step)
    except ValueError:
        pass

    print(f"[{NODE_NAME}] Received adapter snapshot (overwrite). Checksum: {norm:.6f}", flush=True)
    return b'{"status":"ok"}'


def _buffer_round_payload(headers: dict[str, str], body_bytes: bytes) -> bytes:
    """Accumulate an inbound round adapter keyed by round_id (§3.4 step 3).

    Idempotency: round_id is monotonic; payloads for closed rounds are
    dropped, never merged into a later round.
    """
    round_id = int(headers.get("x-round-id") or headers.get("X-Round-ID", "0"))
    sender = headers.get("x-node-name") or headers.get("X-Node-Name", "")
    try:
        step = int(headers.get("x-step") or headers.get("X-Step", "0") or 0)
    except ValueError:
        step = 0

    if round_id <= current_round:
        print(f"[{NODE_NAME}] Dropping adapter from {sender} for closed round {round_id}", flush=True)
        return b'{"status":"dropped"}'

    # Sparsity cross-check (Phase 3.2). If this round's plan has already
    # reached us, a sender announcing a different p built its payload against
    # a different index set; merging it would scatter its weights into
    # unrelated coordinates. Reject it here with a loud message rather than
    # letting the (silent) length check in merge_sparse drop it, so the
    # misconfiguration is visible in the logs.
    sender_p_header = headers.get("x-sparse-p") or headers.get("X-Sparse-P", "")
    if sender_p_header and round_id in _round_p:
        try:
            sender_p = float(sender_p_header)
        except ValueError:
            sender_p = _round_p[round_id]
        if abs(sender_p - _round_p[round_id]) > 1e-12:
            print(f"[{NODE_NAME}] REJECTING adapter from {sender} for round "
                  f"{round_id}: sender p={sender_p:g} but this round is "
                  f"p={_round_p[round_id]:g}. Masks would not match.", flush=True)
            return b'{"status":"rejected_p_mismatch"}'

    tensors = safetensors_load(body_bytes)
    buffer = _round_buffers.setdefault(round_id, [])
    buffer.append((sender, step, tensors))

    event = _round_events.get(round_id)
    expected = _round_expect.get(round_id)
    if event is not None and expected is not None and len(buffer) >= expected:
        event.set()
    return b'{"status":"buffered"}'


async def _send_round_adapter(target: str, payload: bytes, headers: dict[str, str]) -> None:
    """Fire-and-forget adapter send to one layer peer. Never holds model_lock."""
    try:
        await transport.send_adapter(
            target=target, payload=payload, headers=headers, timeout=T_ROUND
        )
    except Exception as error:
        print(f"[{NODE_NAME}] Round adapter send to {target} failed: {error}", flush=True)


def _apply_round_merge(
    received: list[tuple[str, int, dict[str, torch.Tensor]]],
    contribute: bool,
    timings: dict[str, float] | None = None,
    *,
    round_id: int = 0,
    p: float = 1.0,
    outer: bool = False,
) -> int:
    """Merge received adapters into the model (§3.5, Phase 3.2, Phase 4).

    WARNING: caller MUST hold `model_lock` and run this via asyncio.to_thread.
    Returns the divisor n (0 = nothing applied).

    `timings` (Phase 2.1) is filled in with the merge arithmetic and the
    state-dict apply as SEPARATE numbers. They have different fixes — the
    first is CPU fp32 accumulation, the second is a host-to-device write
    under the lock — and a single combined figure cannot distinguish them.
    """
    global _outer_anchor
    assert model is not None
    t0 = time.perf_counter()
    own = get_peft_model_state_dict(model)
    if p < 1.0:
        merged = merge_sparse(
            own, received, contribute, round_id=round_id, p=p, own_name=NODE_NAME
        )
    else:
        merged = merge_adapters(own, received, contribute, own_name=NODE_NAME)
    if timings is not None:
        timings["merge_compute_ms"] = (time.perf_counter() - t0) * 1000
    if merged is None:
        if timings is not None:
            timings["apply_ms"] = 0.0
        return 0

    # ── Outer optimizer (Phase 4). Dense rounds only ──
    # A sparse round touches a different ~p of coordinates each time, so a
    # "delta since the last anchor" over one of them is not a pseudo-gradient
    # of anything coherent. The outer step therefore runs only where the
    # merge is complete.
    if outer and p >= 1.0:
        t_outer = time.perf_counter()
        if _outer_anchor is None:
            # FIRST dense round establishes the anchor, without stepping.
            #
            # This is not laziness — it is what keeps replicas bit-identical.
            # Same-stage peers do NOT start from the same adapter (LoRA A is
            # randomly initialised per worker), so an anchor captured at model
            # load would differ across replicas and the outer step would
            # compute a different result on each, permanently diverging them.
            # The merged average is by construction identical on every
            # replica, so anchoring on it makes the first outer step the first
            # point at which they are guaranteed to agree.
            _outer_anchor = {k: v.clone() for k, v in merged.items()}
            print(f"[{NODE_NAME}] Outer optimizer: anchor established at round "
                  f"{round_id} (no outer step on the first dense round)", flush=True)
        else:
            merged = outer_step(_outer_anchor, merged, _outer_buf, OUTER_LR, OUTER_MU)
            _outer_anchor = {k: v.clone() for k, v in merged.items()}
        if timings is not None:
            timings["outer_ms"] = (time.perf_counter() - t_outer) * 1000

    t1 = time.perf_counter()
    set_peft_model_state_dict(model, merged)
    if timings is not None:
        timings["apply_ms"] = (time.perf_counter() - t1) * 1000
    return len(received) + (1 if contribute else 0)


async def _run_round(params: dict[str, Any]) -> dict[str, Any]:
    """Execute one averaging round on this worker (§3.4 worker behaviour)."""
    global current_round, steps_since_round, adapter_step

    if model is None:
        raise RuntimeError("Worker model not ready")

    round_id = int(params["round_id"])
    send_to = list(params.get("send_to", []))
    expect = int(params.get("expect", 0))
    contribute = bool(params.get("contribute", False))
    x = int(params.get("x", 10))
    # Sparsity and outer-step flags come from the gateway's plan, so every
    # peer in a stage necessarily agrees on them for a given round. Default
    # p=1.0 keeps a gateway that predates Phase 3 working unchanged.
    p = float(params.get("p", 1.0))
    outer = bool(params.get("outer", False))

    if round_id <= current_round:
        return {"status": "stale", "round_id": round_id, "current_round": current_round}

    _round_p[round_id] = p

    # ── Phase 2.1 instrumentation ──
    # Per-substep wall time, always on. A round is a distributed operation
    # whose cost is invisible from any single vantage point: the harness sees
    # only "the barrier took 4.2 s". These numbers are returned to the gateway
    # and surface in the /average response, so one dp2 run produces the whole
    # profile without a special build.
    round_t0 = time.perf_counter()
    timings: dict[str, float] = {}

    event = _round_events.setdefault(round_id, asyncio.Event())
    _round_expect[round_id] = expect
    if len(_round_buffers.get(round_id, [])) >= expect:
        event.set()

    # 1. Snapshot under model_lock; move to CPU and serialize OUTSIDE it.
    if contribute:
        t = time.perf_counter()
        async with model_lock:
            # Clone on-device only. The .cpu() transfers used to happen here,
            # holding model_lock across ~100 synchronous device-to-host copies
            # while a train step queued behind it. A device-side clone is a
            # handful of cheap async kernels by comparison.
            snapshot = {
                k: v.detach().clone() for k, v in get_peft_model_state_dict(model).items()
            }
        timings["snapshot_lock_ms"] = (time.perf_counter() - t) * 1000

        # On a sparse round only the masked values leave the device, so this
        # transfer and the serialize below both shrink by ~1/p.
        t = time.perf_counter()
        if p < 1.0:
            state_dict = await asyncio.to_thread(
                lambda: {
                    k: v.cpu()
                    for k, v in sparse_snapshot(snapshot, round_id, p).items()
                }
            )
        else:
            state_dict = await asyncio.to_thread(
                lambda: {k: v.cpu() for k, v in snapshot.items()}
            )
        timings["to_cpu_ms"] = (time.perf_counter() - t) * 1000

        # safetensors_save on a worker thread, not on the event loop. While
        # this runs, the loop stays free to accept inbound /adapter POSTs from
        # same-stage peers — every worker in the stage serializes at the same
        # instant, so a blocking serialize here adds its full duration to
        # every peer's wait as well as our own.
        t = time.perf_counter()
        payload = await asyncio.to_thread(safetensors_save, state_dict)
        timings["serialize_ms"] = (time.perf_counter() - t) * 1000
        timings["payload_bytes"] = float(len(payload))

        # 2. Fire-and-forget sends — never await a peer while holding model_lock.
        t = time.perf_counter()
        send_headers = {
            "X-Model-ID": assigned_model_id or "",
            "X-Round-ID": str(round_id),
            "X-Step": str(adapter_step),  # lineage step, NOT training_steps
            "X-Node-Name": NODE_NAME,
            # Announced so the receiver can reject a payload built with a
            # different p outright, instead of relying on the length check
            # to notice. Never used to CHOOSE p — that comes from the plan.
            "X-Sparse-P": repr(p),
        }
        for target in send_to:
            asyncio.create_task(_send_round_adapter(target, payload, send_headers))
        timings["send_dispatch_ms"] = (time.perf_counter() - t) * 1000

    # 3. Wait until `expect` payloads arrive or T_round elapses.
    #
    # Re-check the buffer immediately before waiting. Everything above this
    # line contains awaits (model_lock, to_thread), and a payload that lands
    # during one of them is buffered by _buffer_round_payload — which sets the
    # event only if BOTH _round_events and _round_expect are already
    # registered for this round. They are, by now, so this is belt-and-braces
    # rather than a known hole; it costs one dict lookup and removes the need
    # to reason about the interleaving at all.
    timed_out = False
    if expect > 0:
        if len(_round_buffers.get(round_id, [])) >= expect:
            event.set()
        t = time.perf_counter()
        try:
            await asyncio.wait_for(event.wait(), timeout=T_ROUND)
        except asyncio.TimeoutError:
            timed_out = True
        timings["wait_ms"] = (time.perf_counter() - t) * 1000
    else:
        timings["wait_ms"] = 0.0

    # 4. Merge whatever arrived (partial merge on timeout is by design).
    received = _round_buffers.pop(round_id, [])
    kept, dropped = filter_skewed(received, adapter_step, x)
    if dropped:
        print(f"[{NODE_NAME}] Round {round_id}: skew guard dropped {dropped}", flush=True)

    t = time.perf_counter()
    async with model_lock:
        timings["merge_lock_wait_ms"] = (time.perf_counter() - t) * 1000
        n = await asyncio.to_thread(
            _apply_round_merge, kept, contribute, timings,
            round_id=round_id, p=p, outer=outer,
        )

    # Advance the adapter lineage: after merging, this worker's adapter is as
    # current as the freshest contribution it absorbed. This is what keeps a
    # spare promotable — its lineage tracks the actives it averages with.
    if n > 0 and kept:
        adapter_step = max(adapter_step, max(step for _, step, _ in kept))

    # 5. Close the round.
    current_round = round_id
    steps_since_round = 0
    _round_events.pop(round_id, None)
    _round_expect.pop(round_id, None)
    _round_p.pop(round_id, None)

    timings["total_ms"] = (time.perf_counter() - round_t0) * 1000

    print(
        f"[{NODE_NAME}] Round {round_id} complete: merged n={n} "
        f"(received {len(received)}, expected {expect}, contribute={contribute}, "
        f"p={p:g}{', FULL' if p >= 1.0 else ''}{', OUTER' if outer and p >= 1.0 else ''}"
        f"{', TIMED OUT' if timed_out else ''}) | "
        + " ".join(f"{k}={v:.1f}" for k, v in timings.items()),
        flush=True,
    )
    return {
        "status": "ok",
        "round_id": round_id,
        "merged_n": n,
        "received": len(received),
        "expected": expect,
        "timed_out": timed_out,
        "p": p,
        "outer": bool(outer and p >= 1.0),
        "timings_ms": timings,
    }


def _run_forward(
    request_id: str,
    *,
    input_ids: torch.Tensor | None = None,
    hidden_states: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    eval_mode: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run this worker's layers with autograd enabled.

    Caches graph-attached tensors in cached_inputs / cached_outputs for the
    backward pass.  Returns *detached* copies for serialization.

    Under `eval_mode` (X-Eval: 1, Phase 1.2) this is a pure measurement:
      - no autograd graph is built (torch.no_grad),
      - NOTHING is written to cached_inputs / cached_outputs /
        last_predictions, so an eval request cannot leave a tensor behind for
        a later backward to pick up (that would both leak VRAM and corrupt an
        unrelated training step),
      - the model is switched to .eval() for the duration, which disables
        LoRA dropout (p=0.05). Measuring held-out loss with dropout active
        would inflate it by a noisy, scenario-independent constant — small,
        never flagged, and enough to muddy an A/B between averaging schemes.
        Safe to toggle because the caller holds `model_lock`.

    Returns:
        tuple of (hidden_states_detached on CPU, loss_detached on CPU or None)
    """
    assert model is not None

    if eval_mode:
        model.eval()
    try:
        with torch.set_grad_enabled(not eval_mode):
            if is_first_stage and input_ids is not None:
                outputs = model.get_base_model().model(
                    input_ids=_to_device(input_ids),
                    return_dict=True,
                )
            else:
                assert hidden_states is not None
                hidden_states = _to_device(hidden_states)
                if not eval_mode:
                    # Enable grad so this becomes the autograd leaf, and cache
                    # the leaf *before* it flows through the model.
                    hidden_states = hidden_states.requires_grad_(True)
                    cached_inputs[request_id] = hidden_states
                # Feed the exact same tensor — no redundant _to_device.
                outputs = model.get_base_model().model(
                    inputs_embeds=hidden_states,
                    return_dict=True,
                )

            last_hidden_state = outputs.last_hidden_state
            loss_cpu = None

            if is_last_stage:
                assert input_ids is not None, "Last-stage worker requires input_ids/labels for loss calculation"
                logits = model.get_base_model().lm_head(last_hidden_state)
                vocab_size = model.get_base_model().config.vocab_size

                if labels is None:
                    labels = input_ids
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous().to(logits.device)

                loss = F.cross_entropy(shift_logits.reshape(-1, vocab_size), shift_labels.reshape(-1), ignore_index=-100)
                if not eval_mode:
                    # Cache the graph-attached loss for backward.
                    cached_outputs[request_id] = loss
                loss_cpu = loss.detach().cpu()
                if not eval_mode:
                    # Cache per-token predictions (argmax over shifted logits) so the
                    # client can inspect what the model predicts at each next-token
                    # position. Used by the English recitation test; harmless for
                    # other callers.
                    last_predictions[request_id] = shift_logits.argmax(dim=-1).detach().cpu()
                print(
                    f"[{NODE_NAME}] {'Eval' if eval_mode else 'Forward'} loss: "
                    f"{loss_cpu.item():.4f}",
                    flush=True,
                )
            elif not eval_mode:
                # Cache the graph-attached hidden_state for backward.
                cached_outputs[request_id] = last_hidden_state

            # Always return *detached* copies on CPU for safetensors serialization.
            return last_hidden_state.detach().contiguous().cpu(), loss_cpu
    finally:
        if eval_mode:
            model.train()


def _serialize_hidden_states(
    result: torch.Tensor, input_ids: torch.Tensor | None = None, labels: torch.Tensor | None = None
) -> tuple[bytes, dict[str, str]]:
    """Serialize hidden states for P2P transfer."""
    tensors_to_save = {"hidden_states": result}
    if input_ids is not None:
        tensors_to_save["input_ids"] = input_ids
    if labels is not None:
        tensors_to_save["labels"] = labels
    binary_hidden = safetensors_save(tensors_to_save)
    return binary_hidden, {}


async def _process_forward(
    request_id: str, body_bytes: bytes, eval_mode: bool = False
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    tensors = safetensors_load(body_bytes)
    input_ids = tensors.get("input_ids")
    labels = tensors.get("labels")

    # ── Run this worker's layers ──
    if is_first_stage:
        if input_ids is None:
            raise ValueError("First-stage worker requires input_ids")
        async with model_lock:
            result, loss = await asyncio.to_thread(
                _run_forward, request_id, input_ids=input_ids, labels=labels,
                eval_mode=eval_mode,
            )
    else:
        hidden_in = tensors.get("hidden_states")
        if hidden_in is None:
            raise ValueError("Non-first-stage worker requires hidden_states")
        hidden_in = hidden_in.to(_dtype())

        async with model_lock:
            result, loss = await asyncio.to_thread(
                _run_forward, request_id, hidden_states=hidden_in, input_ids=input_ids,
                labels=labels, eval_mode=eval_mode,
            )
    return result, loss, input_ids, labels


async def _execute_forward_pipeline(headers: dict[str, str], body_bytes: bytes) -> bytes:
    if model is None:
        raise RuntimeError("Worker model is not ready")
    request_id = headers.get("x-request-id") or headers.get("X-Request-ID", "")
    if not request_id:
        raise ValueError("X-Request-ID is required")

    pipeline_header = headers.get("x-pipeline") or headers.get("X-Pipeline", "")
    next_workers = [u.strip() for u in pipeline_header.split(",") if u.strip()]

    # Build the reverse pipeline for backward pass.
    reverse_header = headers.get("x-reverse-pipeline") or headers.get("X-Reverse-Pipeline", "")
    reverse_workers = [u.strip() for u in reverse_header.split(",") if u.strip()]

    # Held-out eval (Phase 1.2): forward + loss only. No backward, no
    # optimizer step, no training_steps / steps_since_round / adapter_step
    # increment, no cached tensors left behind. An eval request must be
    # completely invisible to the model's state — otherwise measuring the
    # model changes it, and the eval curve becomes a function of how often
    # eval ran.
    eval_mode = (headers.get("x-eval") or headers.get("X-Eval", "")) == "1"

    success = False
    try:
        result, loss, input_ids, labels = await _process_forward(
            request_id, body_bytes, eval_mode=eval_mode
        )

        # ── Last stage: run backward, propagate grads upstream ──
        if is_last_stage and not next_workers:
            assert loss is not None, "Last stage must compute loss"

            if eval_mode:
                # Terminate here. stage0_grad_norm is reported as exactly 0.0
                # rather than omitted, so the response shape is identical to a
                # train step and existing clients keep parsing it unchanged.
                success = True
                return safetensors_save({
                    "loss": loss,
                    "stage0_grad_norm": torch.tensor(0.0, dtype=torch.float32),
                })

            # Pop the graph-attached loss and run backward ONLY (no optimizer step yet).
            graph_loss = cached_outputs.pop(request_id)
            async with model_lock:
                grad_norm = await asyncio.to_thread(_run_backward_only, graph_loss, None)
            
            print(f"[{NODE_NAME}] Backward pass complete. Grad norm: {grad_norm:.6f}", flush=True)

            stage0_grad_norm = torch.tensor(grad_norm, dtype=torch.float32)

            # If this is also the first stage (solo), we are the terminus —
            # no send_backward needed; commit immediately.
            if not is_first_stage and reverse_workers:
                input_tensor = cached_inputs.pop(request_id)
                assert input_tensor.grad is not None, "input_tensor.grad is None — graph was severed"
                grad_to_send = input_tensor.grad.detach().contiguous().cpu()

                next_upstream = reverse_workers[0]
                remaining_reverse = ",".join(reverse_workers[1:])
                grad_payload = safetensors_save({"grad": grad_to_send})

                try:
                    backward_response = await transport.send_backward(
                        target=next_upstream,
                        payload=grad_payload,
                        headers={
                            "X-Request-ID": request_id,
                            "X-Reverse-Pipeline": remaining_reverse,
                            "Content-Type": "application/octet-stream",
                        },
                        timeout=120.0,
                    )
                except Exception as error:
                    # send_backward failed — do NOT step the optimizer.
                    # Raise with the failed_node so the gateway can evict the right worker.
                    print(f"[{NODE_NAME}] send_backward failed to {next_upstream}: {error}", flush=True)
                    raise RuntimeError(json.dumps({"failed_node": next_upstream, "error": str(error)})) from error

                # Check if the response is a structured error bubbled from upstream.
                bubbled_failed = _extract_failed_node_from_response(backward_response)
                if bubbled_failed:
                    # Upstream reported a failure — do NOT step the optimizer.
                    raise RuntimeError(json.dumps({"failed_node": bubbled_failed, "error": "upstream backward failure"}))

                # Extract stage0_grad_norm from the terminus response.
                backward_tensors = safetensors_load(backward_response)
                stage0_grad_norm = backward_tensors["grad_norm"]

            # ── Commit: downstream confirmed success, now step optimizer ──
            async with model_lock:
                await asyncio.to_thread(_step_optimizer)

            # Dedup guard: count this step exactly once, even across retries.
            global training_steps, _last_step_request_id, steps_since_round, adapter_step
            if request_id != _last_step_request_id:
                training_steps += 1
                steps_since_round += 1
                adapter_step += 1
                _last_step_request_id = request_id

            # Serialize the *detached* loss_cpu, not the graph-attached loss.
            success = True
            out = {"loss": loss, "stage0_grad_norm": stage0_grad_norm}
            # Include per-token predictions if available (English recitation test).
            preds = last_predictions.pop(request_id, None)
            if preds is not None:
                out["predictions"] = preds
            return safetensors_save(out)

        # ── Non-last stage: forward downstream ──
        if not next_workers:
            raise RuntimeError("Non-last worker received empty pipeline")

        # Prepend our address to the reverse pipeline so backward knows where to send.
        my_address = await transport.get_address()
        if reverse_workers:
            updated_reverse = ",".join([my_address] + reverse_workers)
        else:
            updated_reverse = my_address

        next_target = next_workers[0]
        remaining_pipeline = ",".join(next_workers[1:])
        binary_hidden, compression_headers = _serialize_hidden_states(result, input_ids=input_ids, labels=labels)

        if transport is None:
            raise RuntimeError("Transport layer is not initialized")
        try:
            response_bytes = await transport.send_forward(
                target=next_target,
                payload=binary_hidden,
                headers={
                    "X-Request-ID": request_id,
                    "X-Pipeline": remaining_pipeline,
                    "X-Reverse-Pipeline": updated_reverse,
                    # Must propagate: the LAST stage is the one that decides to
                    # skip backward, and it only learns this from the header.
                    # Dropping it here would make an eval request train the model.
                    "X-Eval": "1" if eval_mode else "0",
                    "Content-Type": "application/octet-stream",
                    **compression_headers,
                },
                timeout=120.0,
            )
            success = True
            return response_bytes
        except Exception as error:
            print(f"[{NODE_NAME}] Forwarding failed to downstream {next_target}: {error}", flush=True)
            # Check if the error already contains a failed_node from downstream.
            err_str = str(error)
            try:
                parsed = json.loads(err_str)
                if isinstance(parsed, dict) and "failed_node" in parsed:
                    raise  # Preserve the original failed_node
            except (json.JSONDecodeError, TypeError):
                pass
            raise RuntimeError(json.dumps({"failed_node": next_target, "error": str(error)})) from error
    finally:
        if (is_last_stage and not next_workers) or not success:
            cached_inputs.pop(request_id, None)
            cached_outputs.pop(request_id, None)
            last_predictions.pop(request_id, None)


async def _execute_backward_pipeline(headers: dict[str, str], body_bytes: bytes) -> bytes:
    """Handle an incoming backward (gradient) request.

    Runs .backward(gradient=...) on the cached output, then propagates
    the input gradient upstream via the reverse pipeline.
    """
    if model is None:
        raise RuntimeError("Worker model is not ready")
    request_id = headers.get("x-request-id") or headers.get("X-Request-ID", "")
    if not request_id:
        raise ValueError("X-Request-ID is required")

    try:
        reverse_header = headers.get("x-reverse-pipeline") or headers.get("X-Reverse-Pipeline", "")
        reverse_workers = [u.strip() for u in reverse_header.split(",") if u.strip()]

        # Deserialize the incoming gradient.
        tensors = safetensors_load(body_bytes)
        grad_in = tensors["grad"]
        grad_in = _to_device(grad_in).to(_dtype())

        # Pop the cached graph-attached output and run backward ONLY (no optimizer step yet).
        output_tensor = cached_outputs.pop(request_id)
        async with model_lock:
            grad_norm = await asyncio.to_thread(_run_backward_only, output_tensor, grad_in)
        print(f"[{NODE_NAME}] Backward pass complete. Grad norm: {grad_norm:.6f}", flush=True)

        # ── Terminus (first stage): commit immediately, no upstream to send to ──
        if is_first_stage:
            async with model_lock:
                await asyncio.to_thread(_step_optimizer)

            global training_steps, _last_step_request_id, steps_since_round, adapter_step
            if request_id != _last_step_request_id:
                training_steps += 1
                steps_since_round += 1
                adapter_step += 1
                _last_step_request_id = request_id

            return safetensors_save({"grad_norm": torch.tensor(grad_norm, dtype=torch.float32)})

        # ── Middle stage: propagate upstream first, then commit ──
        input_tensor = cached_inputs.pop(request_id)
        assert input_tensor.grad is not None, "input_tensor.grad is None — graph was severed"
        grad_to_send = input_tensor.grad.detach().contiguous().cpu()

        next_upstream = reverse_workers[0]
        remaining_reverse = ",".join(reverse_workers[1:])
        grad_payload = safetensors_save({"grad": grad_to_send})

        try:
            backward_response = await transport.send_backward(
                target=next_upstream,
                payload=grad_payload,
                headers={
                    "X-Request-ID": request_id,
                    "X-Reverse-Pipeline": remaining_reverse,
                    "Content-Type": "application/octet-stream",
                },
                timeout=120.0,
            )
        except Exception as error:
            # send_backward failed — do NOT step the optimizer.
            print(f"[{NODE_NAME}] send_backward failed to {next_upstream}: {error}", flush=True)
            raise RuntimeError(json.dumps({"failed_node": next_upstream, "error": str(error)})) from error

        # Check if the response is a structured error bubbled from upstream.
        bubbled_failed = _extract_failed_node_from_response(backward_response)
        if bubbled_failed:
            raise RuntimeError(json.dumps({"failed_node": bubbled_failed, "error": "upstream backward failure"}))

        # ── Commit: upstream confirmed success, now step optimizer ──
        async with model_lock:
            await asyncio.to_thread(_step_optimizer)

        if request_id != _last_step_request_id:
            training_steps += 1
            steps_since_round += 1
            adapter_step += 1
            _last_step_request_id = request_id

        # Bubble the terminus response back.
        return backward_response
    finally:
        cached_inputs.pop(request_id, None)
        cached_outputs.pop(request_id, None)


@app.post("/forward")
async def forward(request: Request):
    if (
        request.headers.get("content-type", "").split(";", 1)[0]
        != "application/octet-stream"
    ):
        raise HTTPException(
            status_code=415,
            detail="Only safetensors application/octet-stream is accepted",
        )
    try:
        response_bytes = await _execute_forward_pipeline(
            dict(request.headers), await request.body()
        )
        return Response(content=response_bytes, media_type="application/octet-stream")
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Forward execution failed: {error}",
        ) from error


@app.post("/backward")
async def backward(request: Request):
    if (
        request.headers.get("content-type", "").split(";", 1)[0]
        != "application/octet-stream"
    ):
        raise HTTPException(
            status_code=415,
            detail="Only safetensors application/octet-stream is accepted",
        )
    try:
        response_bytes = await _execute_backward_pipeline(
            dict(request.headers), await request.body()
        )
        return Response(content=response_bytes, media_type="application/octet-stream")
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Backward execution failed: {error}",
        ) from error


@app.get("/adapter")
async def get_adapter():
    """Return this worker's current adapter bytes (§3.7 bootstrap fast path).

    Lets a replacement worker (or a test) pull the current state instead of
    waiting for the next averaging round.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Worker model not ready")
    async with model_lock:
        state_dict = {
            k: v.detach().cpu() for k, v in get_peft_model_state_dict(model).items()
        }
    payload = safetensors_save(state_dict)
    return Response(
        content=payload,
        media_type="application/octet-stream",
        headers={
            "X-Node-Name": NODE_NAME,
            "X-Step": str(adapter_step),  # lineage step, adopted by pullers
            "X-Round": str(current_round),
        },
    )


@app.post("/round")
async def round_endpoint(request: Request):
    """Averaging-round trigger from the gateway (§3.4).

    Body: {round_id, send_to: [addr], expect: int, contribute: bool, x: int}
    """
    try:
        params = await request.json()
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"Invalid round body: {error}") from error
    try:
        result = await _run_round(params)
        return result
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Round failed: {error}") from error


@app.post("/adapter")
async def adapter(request: Request):
    if (
        request.headers.get("content-type", "").split(";", 1)[0]
        != "application/octet-stream"
    ):
        raise HTTPException(
            status_code=415,
            detail="Only safetensors application/octet-stream is accepted",
        )
    try:
        response_bytes = await _execute_adapter_pipeline(
            dict(request.headers), await request.body()
        )
        return Response(content=response_bytes, media_type="application/json")
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Adapter update failed: {error}",
        ) from error
