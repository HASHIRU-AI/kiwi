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
current_standby_peer_id: str = ""
_last_step_request_id: str = ""

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
    global current_standby_peer_id
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
                    new_standby = data.get("standby_peer_id", "")
                    if new_standby and new_standby != current_standby_peer_id:
                        current_standby_peer_id = new_standby
                        asyncio.create_task(_push_adapter())

                    # ── Standby → Active promotion detection ──
                    new_role = data.get("role", "unknown")
                    if current_role == "standby" and new_role == "active" and model is not None:
                        print(f"[{NODE_NAME}] *** Promoted to ACTIVE! Resuming from snapshot. ***", flush=True)
                        _log_recovery_bytes_metrics()
                    current_role = new_role
        except httpx.HTTPError:
            pass  # Gateway may be temporarily unreachable; keep retrying.


def _log_recovery_bytes_metrics() -> None:
    """Log apples-to-apples recovery bytes metrics on standby promotion.

    Both baseline and adapter include optimizer state (Adam m + v) for
    a fair comparison. The standby is pre-warmed so the actual promotion
    recovery cost is ~0 bytes.
    """
    if model is None:
        return

    base_model = model.get_base_model()
    # All params (frozen base + LoRA adapters)
    base_params = sum(p.numel() for p in base_model.parameters())
    # Only trainable LoRA params
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Both use bfloat16 (2 bytes) for weights + float32 (4 bytes each) for Adam m and v
    bytes_per_param_weights = 2  # bfloat16
    bytes_per_param_optimizer = 4 + 4  # Adam m (float32) + v (float32)
    bytes_per_param_total = bytes_per_param_weights + bytes_per_param_optimizer  # 10 bytes

    baseline_bytes = base_params * bytes_per_param_total
    adapter_snapshot_bytes = trainable_params * bytes_per_param_total

    baseline_mb = baseline_bytes / (1024 ** 2)
    adapter_snapshot_mb = adapter_snapshot_bytes / (1024 ** 2)
    ratio = baseline_bytes / adapter_snapshot_bytes if adapter_snapshot_bytes > 0 else float("inf")

    print(f"[{NODE_NAME}] Recovery Metrics - Promotion Recovery Cost: ~0 MB (pre-warmed)", flush=True)
    print(f"[{NODE_NAME}] Recovery Metrics - Ongoing Snapshot Cost: {adapter_snapshot_mb:.2f} MB "
          f"({trainable_params:,} params × {bytes_per_param_total} bytes)", flush=True)
    print(f"[{NODE_NAME}] Recovery Metrics - Full-weight baseline: {baseline_mb:.2f} MB "
          f"({base_params:,} params × {bytes_per_param_total} bytes)", flush=True)
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


@asynccontextmanager
async def lifespan(_: FastAPI):
    global assigned_model_id, layer_range
    global is_first_stage, is_last_stage, model, transport

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

async def _push_adapter() -> None:
    if not current_standby_peer_id or not model:
        return
    
    async with model_lock:
        state_dict = get_peft_model_state_dict(model)
        state_dict = {k: v.cpu() for k, v in state_dict.items()}

    payload = safetensors_save(state_dict)
    
    flat = torch.cat([v.flatten() for v in state_dict.values()]) if state_dict else torch.tensor([0.0])
    norm = flat.norm().item()
    
    try:
        await transport.send_adapter(
            target=current_standby_peer_id,
            payload=payload,
            headers={"X-Model-ID": assigned_model_id},
            timeout=30.0
        )
        print(f"[{NODE_NAME}] Pushing adapter snapshot to standby {current_standby_peer_id}. Checksum: {norm:.6f}", flush=True)
    except Exception as e:
        print(f"[{NODE_NAME}] Failed to push adapter to {current_standby_peer_id}: {e}", flush=True)


async def _execute_adapter_pipeline(headers: dict[str, str], body_bytes: bytes) -> bytes:
    if not model:
        raise RuntimeError("Worker model not ready")
        
    model_id = headers.get("x-model-id") or headers.get("X-Model-ID", "")
    if model_id != assigned_model_id:
        raise ValueError(f"Model ID mismatch: expected {assigned_model_id}, got {model_id}")

    tensors = safetensors_load(body_bytes)
    
    flat = torch.cat([v.flatten() for v in tensors.values()]) if tensors else torch.tensor([0.0])
    norm = flat.norm().item()
    
    async with model_lock:
        set_peft_model_state_dict(model, tensors)
        
    print(f"[{NODE_NAME}] Received adapter snapshot from active. Checksum: {norm:.6f}", flush=True)
    return b'{"status":"ok"}'


def _run_forward(
    request_id: str,
    *,
    input_ids: torch.Tensor | None = None,
    hidden_states: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run this worker's layers with autograd enabled.

    Caches graph-attached tensors in cached_inputs / cached_outputs for the
    backward pass.  Returns *detached* copies for serialization.

    Returns:
        tuple of (hidden_states_detached on CPU, loss_detached on CPU or None)
    """
    assert model is not None

    if is_first_stage and input_ids is not None:
        outputs = model.get_base_model().model(
            input_ids=_to_device(input_ids),
            return_dict=True,
        )
    else:
        assert hidden_states is not None
        # Move to device, then enable grad so this becomes the autograd leaf.
        hidden_states = _to_device(hidden_states).requires_grad_(True)
        # Cache the leaf *before* it flows through the model.
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
        # Cache the graph-attached loss for backward.
        cached_outputs[request_id] = loss
        loss_cpu = loss.detach().cpu()
        # Cache per-token predictions (argmax over shifted logits) so the client
        # can inspect what the model predicts at each next-token position.
        # Used by the English recitation test; harmless for other callers.
        last_predictions[request_id] = shift_logits.argmax(dim=-1).detach().cpu()
        print(f"[{NODE_NAME}] Forward loss: {loss_cpu.item():.4f}", flush=True)
    else:
        # Cache the graph-attached hidden_state for backward.
        cached_outputs[request_id] = last_hidden_state

    # Always return *detached* copies on CPU for safetensors serialization.
    return last_hidden_state.detach().contiguous().cpu(), loss_cpu


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


async def _process_forward(request_id: str, body_bytes: bytes) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    tensors = safetensors_load(body_bytes)
    input_ids = tensors.get("input_ids")
    labels = tensors.get("labels")

    # ── Run this worker's layers ──
    if is_first_stage:
        if input_ids is None:
            raise ValueError("First-stage worker requires input_ids")
        async with model_lock:
            result, loss = await asyncio.to_thread(
                _run_forward, request_id, input_ids=input_ids, labels=labels
            )
    else:
        hidden_in = tensors.get("hidden_states")
        if hidden_in is None:
            raise ValueError("Non-first-stage worker requires hidden_states")
        hidden_in = hidden_in.to(_dtype())

        async with model_lock:
            result, loss = await asyncio.to_thread(
                _run_forward, request_id, hidden_states=hidden_in, input_ids=input_ids, labels=labels
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

    success = False
    try:
        result, loss, input_ids, labels = await _process_forward(request_id, body_bytes)

        # ── Last stage: run backward, propagate grads upstream ──
        if is_last_stage and not next_workers:
            assert loss is not None, "Last stage must compute loss"

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
            global training_steps, _last_step_request_id
            if request_id != _last_step_request_id:
                training_steps += 1
                _last_step_request_id = request_id
                if training_steps % 10 == 0:
                    asyncio.create_task(_push_adapter())

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

            global training_steps, _last_step_request_id
            if request_id != _last_step_request_id:
                training_steps += 1
                _last_step_request_id = request_id
                if training_steps % 10 == 0:
                    asyncio.create_task(_push_adapter())

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
            _last_step_request_id = request_id
            if training_steps % 10 == 0:
                asyncio.create_task(_push_adapter())

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
