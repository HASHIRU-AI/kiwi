"""Kiwi worker: dynamically assigned split-model execution and P2P forwarding."""

from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import httpx
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from safetensors.torch import load as safetensors_load
from safetensors.torch import save as safetensors_save
from transformers import AutoModelForCausalLM


GATEWAY_URL = os.getenv("GATEWAY_URL", "http://gateway:8080").rstrip("/")
NODE_NAME = os.getenv("NODE_NAME", "worker")
NODE_URL = os.getenv("NODE_URL", "http://worker:8000").rstrip("/")
USE_CSR = os.getenv("KIWI_USE_CSR", "false").lower() in ("true", "1", "yes")
CSR_SPARSITY_RATIO = float(os.getenv("KIWI_CSR_SPARSITY", "0.0"))
USE_INT8 = os.getenv("KIWI_USE_INT8", "false").lower() in ("true", "1", "yes")
GPU_MEMORY_RESERVE_MB = float(os.getenv("GPU_MEMORY_RESERVE_MB", "512"))

# ── Populated from gateway assignment ────────────────────────────────────────────

assigned_model_id: str | None = None
assigned_dtype: str = "bfloat16"
layer_range: tuple[int, int] | None = None
is_first_stage: bool = False
is_last_stage: bool = False
model: Any | None = None
inference_device: str = "cpu"

kv_cache: dict[str, tuple[float, Any]] = {}  # request_id -> (last_accessed, past_key_values)
model_lock = asyncio.Lock()
KV_CACHE_TTL_SECONDS = float(os.getenv("KV_CACHE_TTL_SECONDS", "60"))
HEARTBEAT_INTERVAL_SECONDS = float(os.getenv("HEARTBEAT_INTERVAL_SECONDS", "5"))


def _dtype() -> torch.dtype:
    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}.get(
        assigned_dtype, torch.bfloat16
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


def _estimate_layer_memory_mb(loaded_model: Any, num_layers: int) -> float:
    """Estimate per-layer GPU memory (MB) from total model parameter size."""
    total_params = sum(p.numel() * p.element_size() for p in loaded_model.parameters())
    # Add ~20% overhead for CUDA context fragmentation.
    return (total_params / num_layers * 1.2) / (1024 ** 2)


def _compute_gpu_layers(loaded_model: Any, num_assigned_layers: int) -> int:
    """Decide how many of the assigned layers fit in available VRAM."""
    if not torch.cuda.is_available():
        return 0
    free_mem_mb = torch.cuda.mem_get_info(0)[0] / (1024 ** 2)
    usable_mb = free_mem_mb - GPU_MEMORY_RESERVE_MB
    if usable_mb <= 0:
        return 0
    per_layer_mb = _estimate_layer_memory_mb(loaded_model, num_assigned_layers)
    gpu_layers = int(usable_mb // per_layer_mb)
    return min(gpu_layers, num_assigned_layers)


def _place_model_on_device(loaded_model: Any, num_assigned_layers: int) -> str:
    """Move the model (or a subset of its layers) onto the best available device.

    Returns the primary inference device string ("cuda" or "cpu").
    """
    if not torch.cuda.is_available():
        return "cpu"

    gpu_layers = _compute_gpu_layers(loaded_model, num_assigned_layers)

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


async def _register() -> dict[str, Any]:
    _, vram_gb = _detect_device()
    payload = {
        "node_name": NODE_NAME,
        "node_url": NODE_URL,
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
    loaded_model.eval()

    # Narrow the decoder stack to just our assigned layers.
    layers = loaded_model.model.layers
    selected_layers = torch.nn.ModuleList(layers[first_layer : last_layer + 1])
    # DynamicCache indexes attention state by ``layer_idx``.  Renumber so
    # the first cache write starts at index 0 regardless of global position.
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
    inference_device = _place_model_on_device(loaded_model, num_assigned)
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
    return loaded_model


async def _re_register() -> None:
    _, vram_gb = _detect_device()
    payload = {
        "node_name": NODE_NAME,
        "node_url": NODE_URL,
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
    while True:
        await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.post(
                    f"{GATEWAY_URL}/heartbeat",
                    json={"node_name": NODE_NAME, "node_url": NODE_URL},
                )
                if response.status_code == 404:
                    await _re_register()
        except httpx.HTTPError:
            pass  # Gateway may be temporarily unreachable; keep retrying.


async def _cache_eviction_loop() -> None:
    """Evict KV-cache entries that haven't been accessed within the TTL."""
    while True:
        await asyncio.sleep(10)
        cutoff = time.time() - KV_CACHE_TTL_SECONDS
        stale = [k for k, (ts, _) in kv_cache.items() if ts < cutoff]
        for k in stale:
            del kv_cache[k]


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
    global assigned_model_id, assigned_dtype, layer_range
    global is_first_stage, is_last_stage, model

    assignment = await _register()
    assigned_model_id = assignment["model_id"]
    assigned_dtype = assignment.get("dtype", os.getenv("MODEL_DTYPE", "bfloat16"))
    layer_range = tuple(assignment["layers"])
    total_layers = assignment["total_layers"]
    is_first_stage = layer_range[0] == 0
    is_last_stage = layer_range[1] == total_layers - 1
    model = await asyncio.to_thread(_load_assigned_model, assignment)

    heartbeat_task = asyncio.create_task(_heartbeat_loop())
    eviction_task = asyncio.create_task(_cache_eviction_loop())
    yield
    heartbeat_task.cancel()
    eviction_task.cancel()
    await _deregister()
    kv_cache.clear()


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
        "cache_entries": len(kv_cache),
        "gpu": gpu_info,
    }


# ── Inference helpers ────────────────────────────────────────────────────────────


def _to_device(tensor: torch.Tensor) -> torch.Tensor:
    """Move a tensor to the worker's inference device."""
    return tensor.to(inference_device)


def _cache_for(request_id: str, step: str) -> Any | None:
    if step == "prefill":
        kv_cache.pop(request_id, None)
        return None
    entry = kv_cache.get(request_id)
    return entry[1] if entry else None


def _run_forward(
    request_id: str,
    step: str,
    *,
    input_ids: torch.Tensor | None = None,
    hidden_states: torch.Tensor | None = None,
) -> torch.Tensor | int:
    """Run this worker's layers.

    Returns:
        ``int``  – a token id   (when this is the last stage)
        ``Tensor`` – hidden states on CPU  (when this is not the last stage)
    """
    assert model is not None
    with torch.inference_mode():
        if is_first_stage and input_ids is not None:
            outputs = model.model(
                input_ids=_to_device(input_ids),
                past_key_values=_cache_for(request_id, step),
                use_cache=True,
                return_dict=True,
            )
        else:
            assert hidden_states is not None
            outputs = model.model(
                inputs_embeds=_to_device(hidden_states),
                past_key_values=_cache_for(request_id, step),
                use_cache=True,
                return_dict=True,
            )

        kv_cache[request_id] = (time.time(), outputs.past_key_values)

        if is_last_stage:
            logits = model.lm_head(outputs.last_hidden_state[:, -1, :])
            return int(torch.argmax(logits, dim=-1).item())
        else:
            # Always return on CPU for safetensors serialization and network transfer.
            return outputs.last_hidden_state.contiguous().cpu()


# ── Forward endpoint ─────────────────────────────────────────────────────────────


@app.post("/forward")
async def forward(request: Request):
    if model is None:
        raise HTTPException(status_code=503, detail="Worker model is not ready")
    request_id = request.headers.get("X-Request-ID")
    step = request.headers.get("X-Step")
    if not request_id or step not in {"prefill", "decode"}:
        raise HTTPException(
            status_code=400, detail="X-Request-ID and X-Step are required"
        )
    if (
        request.headers.get("content-type", "").split(";", 1)[0]
        != "application/octet-stream"
    ):
        raise HTTPException(
            status_code=415,
            detail="Only safetensors application/octet-stream is accepted",
        )

    # Parse the remaining pipeline from the header.
    pipeline_header = request.headers.get("X-Pipeline", "")
    next_workers = [u.strip() for u in pipeline_header.split(",") if u.strip()]

    try:
        tensors = safetensors_load(await request.body())
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid safetensors payload: {error}",
        ) from error

    # ── Run this worker's layers ──
    if is_first_stage:
        input_ids = tensors.get("input_ids")
        if input_ids is None:
            raise HTTPException(
                status_code=400,
                detail="First-stage worker requires input_ids",
            )
        async with model_lock:
            result = await asyncio.to_thread(
                _run_forward, request_id, step, input_ids=input_ids
            )
    else:
        hidden_in = tensors.get("hidden_states")
        if hidden_in is None:
            # Check for CSR + INT8 combined components
            crow = tensors.get("crow_indices")
            col = tensors.get("col_indices")
            quantized_val = tensors.get("quantized_values")
            scale_tensor = tensors.get("scale")
            shape_tensor = tensors.get("original_shape")
            if (
                crow is not None
                and col is not None
                and quantized_val is not None
                and scale_tensor is not None
                and shape_tensor is not None
            ):
                # Dequantize values
                scale = scale_tensor.item()
                val = quantized_val.float() * scale
                # Reconstruct CSR
                original_shape = list(shape_tensor.tolist())
                size_2d = (original_shape[0] * original_shape[1], original_shape[2])
                sparse_2d = torch.sparse_csr_tensor(crow, col, val, size=size_2d)
                hidden_in = sparse_2d.to_dense().view(original_shape)
                print(f"[{NODE_NAME}] [CSR+INT8] Dequantized & Decompressed sparse activations back to shape {original_shape}", flush=True)
            else:
                quantized = tensors.get("quantized_hidden_states")
                scale_tensor = tensors.get("scale")
                if quantized is not None and scale_tensor is not None:
                    scale = scale_tensor.item()
                    hidden_in = quantized.float() * scale
                    print(f"[{NODE_NAME}] [INT8] Dequantized activations back to shape {list(hidden_in.shape)}", flush=True)
                else:
                    # Check for CSR components
                    crow = tensors.get("crow_indices")
                    col = tensors.get("col_indices")
                    val = tensors.get("values")
                    shape_tensor = tensors.get("original_shape")
                    if (
                        crow is not None
                        and col is not None
                        and val is not None
                        and shape_tensor is not None
                    ):
                        original_shape = list(shape_tensor.tolist())
                        size_2d = (original_shape[0] * original_shape[1], original_shape[2])
                        sparse_2d = torch.sparse_csr_tensor(crow, col, val, size=size_2d)
                        hidden_in = sparse_2d.to_dense().view(original_shape)
                        print(f"[{NODE_NAME}] [CSR] Decompressed sparse activations back to shape {original_shape}", flush=True)
                    else:
                        raise HTTPException(
                            status_code=400,
                            detail="Non-first-stage worker requires hidden_states, INT8 components, or CSR components",
                        )
        hidden_in = hidden_in.to(_dtype())

        async with model_lock:
            result = await asyncio.to_thread(
                _run_forward, request_id, step, hidden_states=hidden_in
            )

    # ── Last stage → return the token ──
    if is_last_stage and not next_workers:
        return JSONResponse({"next_token_id": result})

    # ── Not last stage → forward hidden_states down the pipeline ──
    if not next_workers:
        # This shouldn't happen if the gateway built the pipeline correctly.
        raise HTTPException(
            status_code=500,
            detail="Non-last worker received empty pipeline",
        )

    next_url = next_workers[0]
    remaining_pipeline = ",".join(next_workers[1:])

    # Compress / serialize output before sending it down the pipeline
    if USE_INT8 and USE_CSR:
        shape = list(result.shape)
        hidden_2d = result.view(-1, shape[-1])
        if CSR_SPARSITY_RATIO > 0.0:
            k = int(hidden_2d.numel() * (1.0 - CSR_SPARSITY_RATIO))
            if k > 0:
                if k < hidden_2d.numel():
                    flat_abs = torch.abs(hidden_2d).flatten()
                    threshold = torch.topk(flat_abs, k).values[-1]
                    mask = torch.abs(hidden_2d) >= threshold
                    hidden_2d_sparse = hidden_2d * mask
                else:
                    hidden_2d_sparse = hidden_2d
            else:
                hidden_2d_sparse = torch.zeros_like(hidden_2d)
        else:
            hidden_2d_sparse = hidden_2d
            
        t_sparse = hidden_2d_sparse.to_sparse_csr()
        crow = t_sparse.crow_indices()
        col = t_sparse.col_indices()
        val = t_sparse.values()
        
        max_val = val.abs().max().item()
        scale = max(max_val, 1e-8) / 127.0
        quantized_val = (val / scale).round().to(torch.int8)
        
        original_elements = result.numel()
        sparse_elements = val.numel()
        sparsity_percent = (1.0 - (sparse_elements / max(original_elements, 1))) * 100.0
        print(f"[{NODE_NAME}] [CSR+INT8] Sparsified & Quantized activations: {original_elements} -> {sparse_elements} non-zero values ({sparsity_percent:.2f}% sparse, scale: {scale:.6f})", flush=True)
        
        binary_hidden = safetensors_save({
            "crow_indices": crow,
            "col_indices": col,
            "quantized_values": quantized_val,
            "scale": torch.tensor([scale], dtype=torch.float32),
            "original_shape": torch.tensor(shape, dtype=torch.long)
        })
    elif USE_INT8:
        max_val = result.abs().max().item()
        scale = max(max_val, 1e-8) / 127.0
        quantized = (result / scale).round().to(torch.int8)
        print(f"[{NODE_NAME}] [INT8] Quantized activations: {result.numel()} elements (scale: {scale:.6f})", flush=True)
        binary_hidden = safetensors_save({
            "quantized_hidden_states": quantized,
            "scale": torch.tensor([scale], dtype=torch.float32)
        })
    elif USE_CSR:
        shape = list(result.shape)
        hidden_2d = result.view(-1, shape[-1])
        if CSR_SPARSITY_RATIO > 0.0:
            k = int(hidden_2d.numel() * (1.0 - CSR_SPARSITY_RATIO))
            if k > 0:
                if k < hidden_2d.numel():
                    flat_abs = torch.abs(hidden_2d).flatten()
                    threshold = torch.topk(flat_abs, k).values[-1]
                    mask = torch.abs(hidden_2d) >= threshold
                    hidden_2d_sparse = hidden_2d * mask
                else:
                    hidden_2d_sparse = hidden_2d
            else:
                hidden_2d_sparse = torch.zeros_like(hidden_2d)
        else:
            hidden_2d_sparse = hidden_2d
            
        t_sparse = hidden_2d_sparse.to_sparse_csr()
        crow = t_sparse.crow_indices()
        col = t_sparse.col_indices()
        val = t_sparse.values()
        
        original_elements = result.numel()
        sparse_elements = val.numel()
        sparsity_percent = (1.0 - (sparse_elements / max(original_elements, 1))) * 100.0
        print(f"[{NODE_NAME}] [CSR] Compressed activations: {original_elements} -> {sparse_elements} non-zero values ({sparsity_percent:.2f}% sparse)", flush=True)
        
        binary_hidden = safetensors_save({
            "crow_indices": crow,
            "col_indices": col,
            "values": val,
            "original_shape": torch.tensor(shape, dtype=torch.long)
        })
    else:
        binary_hidden = safetensors_save({"hidden_states": result})

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(120.0, connect=10.0)
        ) as client:
            response = await client.post(
                f"{next_url.rstrip('/')}/forward",
                headers={
                    "X-Request-ID": request_id,
                    "X-Step": step,
                    "X-Pipeline": remaining_pipeline,
                    "Content-Type": "application/octet-stream",
                },
                content=binary_hidden,
            )
            response.raise_for_status()
    except httpx.HTTPError as error:
        return JSONResponse(
            {"detail": f"Pipeline forwarding to {next_url} failed: {error}"},
            status_code=502,
            headers={"X-Kiwi-Failed-Node": next_url.rstrip("/")},
        )
    return JSONResponse(response.json())
