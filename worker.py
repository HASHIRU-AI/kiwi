"""Kiwi worker: dynamically assigned split-model execution and P2P forwarding."""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any, Literal

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
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-0.5B")
MODEL_DTYPE = os.getenv("MODEL_DTYPE", "float32")
USE_CSR = os.getenv("KIWI_USE_CSR", "false").lower() in ("true", "1", "yes")
CSR_SPARSITY_RATIO = float(os.getenv("KIWI_CSR_SPARSITY", "0.0"))
USE_INT8 = os.getenv("KIWI_USE_INT8", "false").lower() in ("true", "1", "yes")

stage: Literal["A", "B"] | None = None
layer_range: tuple[int, int] | None = None
model: Any | None = None
kv_cache: dict[str, Any] = {}
model_lock = asyncio.Lock()


def _dtype() -> torch.dtype:
    return {"float32": torch.float32, "bfloat16": torch.bfloat16}.get(MODEL_DTYPE, torch.float32)


async def _register() -> dict[str, Any]:
    payload = {
        "node_name": NODE_NAME,
        "node_url": NODE_URL,
        "cpu_cores": os.cpu_count() or 1,
        "vram_gb": 0.0,  # CPU Docker MVP; a GPU worker can report its actual telemetry here.
        "pool_hint": os.getenv("KIWI_POOL_HINT") or None,
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
    assigned_stage = assignment["stage"]
    first_layer, last_layer = assignment["layers"]
    loaded_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=_dtype())
    loaded_model.eval()

    # Qwen's base model constructs attention masks/positions/cache correctly even after
    # its ModuleList is narrowed. The original layer_idx values remain intact for cache slots.
    layers = loaded_model.model.layers
    selected_layers = torch.nn.ModuleList(layers[first_layer : last_layer + 1])
    # DynamicCache indexes attention state by ``layer_idx``. Renumber a Stage B
    # slice locally, otherwise its first cache write would start at global index 12.
    for local_index, decoder_layer in enumerate(selected_layers):
        if hasattr(decoder_layer, "self_attn"):
            decoder_layer.self_attn.layer_idx = local_index
    loaded_model.model.layers = selected_layers
    if assigned_stage == "A":
        # Stage A must return raw intermediate activations, not the final RMSNorm output.
        loaded_model.model.norm = torch.nn.Identity()
        loaded_model.lm_head = torch.nn.Identity()
    elif assigned_stage == "B":
        # Stage B always receives inputs_embeds from Stage A, so token embeddings are unused.
        loaded_model.model.embed_tokens = torch.nn.Identity()
    else:
        raise ValueError(f"Unsupported stage assignment: {assigned_stage}")
    return loaded_model


@asynccontextmanager
async def lifespan(_: FastAPI):
    global stage, layer_range, model
    assignment = await _register()
    stage = assignment["stage"]
    layer_range = tuple(assignment["layers"])
    # Deliberately not global import-time initialization: role assignment comes first.
    model = await asyncio.to_thread(_load_assigned_model, assignment)
    yield
    kv_cache.clear()


app = FastAPI(title="Kiwi worker", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok" if model is not None else "initializing",
        "node_name": NODE_NAME,
        "stage": stage,
        "layers": layer_range,
        "cache_entries": len(kv_cache),
    }


def _cache_for(request_id: str, step: str) -> Any | None:
    if step == "prefill":
        # Recompute and explicit new prompts replace stale cache state atomically.
        kv_cache.pop(request_id, None)
        return None
    return kv_cache.get(request_id)


def _run_stage_a(input_ids: torch.Tensor, request_id: str, step: str) -> torch.Tensor:
    assert model is not None
    with torch.inference_mode():
        outputs = model.model(
            input_ids=input_ids,
            past_key_values=_cache_for(request_id, step),
            use_cache=True,
            return_dict=True,
        )
    kv_cache[request_id] = outputs.past_key_values
    return outputs.last_hidden_state.contiguous()


def _run_stage_b(hidden_states: torch.Tensor, request_id: str, step: str) -> int:
    assert model is not None
    with torch.inference_mode():
        outputs = model.model(
            inputs_embeds=hidden_states,
            past_key_values=_cache_for(request_id, step),
            use_cache=True,
            return_dict=True,
        )
        logits = model.lm_head(outputs.last_hidden_state[:, -1, :])
    kv_cache[request_id] = outputs.past_key_values
    return int(torch.argmax(logits, dim=-1).item())


@app.post("/forward")
async def forward(request: Request):
    if model is None or stage is None:
        raise HTTPException(status_code=503, detail="Worker model is not ready")
    request_id = request.headers.get("X-Request-ID")
    step = request.headers.get("X-Step")
    if not request_id or step not in {"prefill", "decode"}:
        raise HTTPException(status_code=400, detail="X-Request-ID and X-Step are required")
    if request.headers.get("content-type", "").split(";", 1)[0] != "application/octet-stream":
        raise HTTPException(status_code=415, detail="Only safetensors application/octet-stream is accepted")

    try:
        tensors = safetensors_load(await request.body())
    except Exception as error:  # safetensors rejects nonconforming/non-tensor payloads safely.
        raise HTTPException(status_code=400, detail=f"Invalid safetensors payload: {error}") from error

    if stage == "A":
        target_b = request.headers.get("X-Target-B")
        input_ids = tensors.get("input_ids")
        if not target_b or input_ids is None:
            raise HTTPException(status_code=400, detail="Stage A requires input_ids and X-Target-B")
        async with model_lock:
            hidden_states = await asyncio.to_thread(_run_stage_a, input_ids, request_id, step)
        
        if USE_INT8 and USE_CSR:
            shape = list(hidden_states.shape)
            hidden_2d = hidden_states.view(-1, shape[-1])
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
            
            original_elements = hidden_states.numel()
            sparse_elements = val.numel()
            sparsity_percent = (1.0 - (sparse_elements / max(original_elements, 1))) * 100.0
            print(f"[{NODE_NAME}] [CSR+INT8 Stage A] Sparsified & Quantized activations: {original_elements} -> {sparse_elements} non-zero values ({sparsity_percent:.2f}% sparse, scale: {scale:.6f})", flush=True)
            
            binary_hidden_states = safetensors_save({
                "crow_indices": crow,
                "col_indices": col,
                "quantized_values": quantized_val,
                "scale": torch.tensor([scale], dtype=torch.float32),
                "original_shape": torch.tensor(shape, dtype=torch.long)
            })
        elif USE_INT8:
            max_val = hidden_states.abs().max().item()
            scale = max(max_val, 1e-8) / 127.0
            quantized = (hidden_states / scale).round().to(torch.int8)
            print(f"[{NODE_NAME}] [INT8 Stage A] Quantized activations: {hidden_states.numel()} elements (scale: {scale:.6f})", flush=True)
            binary_hidden_states = safetensors_save({
                "quantized_hidden_states": quantized,
                "scale": torch.tensor([scale], dtype=torch.float32)
            })
        elif USE_CSR:
            shape = list(hidden_states.shape)
            hidden_2d = hidden_states.view(-1, shape[-1])
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
            
            original_elements = hidden_states.numel()
            sparse_elements = val.numel()
            sparsity_percent = (1.0 - (sparse_elements / max(original_elements, 1))) * 100.0
            print(f"[{NODE_NAME}] [CSR Stage A] Compressed activations: {original_elements} -> {sparse_elements} non-zero values ({sparsity_percent:.2f}% sparse)", flush=True)
            
            binary_hidden_states = safetensors_save({
                "crow_indices": crow,
                "col_indices": col,
                "values": val,
                "original_shape": torch.tensor(shape, dtype=torch.long)
            })
        else:
            binary_hidden_states = safetensors_save({"hidden_states": hidden_states})

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
                response = await client.post(
                    f"{target_b.rstrip('/')}/forward",
                    headers={
                        "X-Request-ID": request_id,
                        "X-Step": step,
                        "Content-Type": "application/octet-stream",
                    },
                    content=binary_hidden_states,
                )
                response.raise_for_status()
        except httpx.HTTPError as error:
            # Lets the gateway evict B, while retaining this healthy Stage A replica.
            return JSONResponse(
                {"detail": f"P2P forwarding to Stage B failed: {error}"},
                status_code=502,
                headers={"X-Kiwi-Failed-Node": target_b.rstrip("/")},
            )
        return JSONResponse(response.json())

    hidden_states = tensors.get("hidden_states")
    if hidden_states is None:
        # Check for CSR + INT8 combined components
        crow = tensors.get("crow_indices")
        col = tensors.get("col_indices")
        quantized_val = tensors.get("quantized_values")
        scale_tensor = tensors.get("scale")
        shape_tensor = tensors.get("original_shape")
        if crow is not None and col is not None and quantized_val is not None and scale_tensor is not None and shape_tensor is not None:
            # Dequantize values
            scale = scale_tensor.item()
            val = quantized_val.float() * scale
            # Reconstruct CSR
            original_shape = list(shape_tensor.tolist())
            size_2d = (original_shape[0] * original_shape[1], original_shape[2])
            sparse_2d = torch.sparse_csr_tensor(crow, col, val, size=size_2d)
            hidden_states = sparse_2d.to_dense().view(original_shape)
            print(f"[{NODE_NAME}] [CSR+INT8 Stage B] Dequantized & Decompressed sparse activations back to shape {original_shape}", flush=True)
        # Check for INT8 components
        else:
            quantized = tensors.get("quantized_hidden_states")
            scale_tensor = tensors.get("scale")
            if quantized is not None and scale_tensor is not None:
                scale = scale_tensor.item()
                hidden_states = quantized.float() * scale
                print(f"[{NODE_NAME}] [INT8 Stage B] Dequantized activations back to shape {list(hidden_states.shape)}", flush=True)
            else:
                # Check for CSR components
                crow = tensors.get("crow_indices")
                col = tensors.get("col_indices")
                val = tensors.get("values")
                shape_tensor = tensors.get("original_shape")
                if crow is not None and col is not None and val is not None and shape_tensor is not None:
                    original_shape = list(shape_tensor.tolist())
                    size_2d = (original_shape[0] * original_shape[1], original_shape[2])
                    sparse_2d = torch.sparse_csr_tensor(crow, col, val, size=size_2d)
                    hidden_states = sparse_2d.to_dense().view(original_shape)
                    print(f"[{NODE_NAME}] [CSR Stage B] Decompressed sparse activations back to shape {original_shape}", flush=True)
                else:
                    raise HTTPException(status_code=400, detail="Stage B requires hidden_states, INT8 components, or CSR components")

    async with model_lock:
        token_id = await asyncio.to_thread(_run_stage_b, hidden_states, request_id, step)
    return JSONResponse({"next_token_id": token_id})
