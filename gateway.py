"""Kiwi gateway: OpenAI-style streaming API, registry, and active recompute."""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Literal
from network_utils import get_public_ip
import httpx
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from safetensors.torch import save as safetensors_save
from transformers import AutoTokenizer

from models import ModelConfig, ModelRegistry, load_model_registry
from transport import KiwiTransport, create_transport


REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "120"))
MAX_REROUTES = int(os.getenv("MAX_REROUTES", "4"))
INDEX_FILE = Path(__file__).with_name("index.html")

# ── Transport configuration ──
P2P_TRANSPORT = os.getenv("KIWI_P2P_TRANSPORT", "http").lower()
P2P_PORT = int(os.getenv("KIWI_P2P_PORT", "4001"))
transport: KiwiTransport | None = None

# ── 2-Stage Eviction Thresholds ──
HEARTBEAT_SOFT_STALE_SECONDS = float(os.getenv("HEARTBEAT_SOFT_STALE_SECONDS", "15"))
HEARTBEAT_HARD_STALE_SECONDS = float(os.getenv("HEARTBEAT_HARD_STALE_SECONDS", "120"))


# ── Pydantic request models ─────────────────────────────────────────────────────


class WorkerRegistration(BaseModel):
    node_name: str
    node_url: str
    peer_id: str = ""
    p2p_port: int = 4001
    cpu_cores: int = Field(ge=1)
    vram_gb: float = Field(ge=0)
    model_id: str | None = None
    layers: list[int] | None = None
    visible_maddrs: list[str] = Field(default_factory=list)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    stream: bool = True
    max_tokens: int = Field(default=128, ge=1, le=512)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


# ── Worker dataclass ─────────────────────────────────────────────────────────────


@dataclass
class Worker:
    node_name: str
    node_url: str
    model_id: str
    layers: tuple[int, int]  # inclusive range [first, last]
    cpu_cores: int
    vram_gb: float
    peer_id: str = ""
    p2p_port: int = 4001
    visible_maddrs: list[str] = field(default_factory=list)
    healthy: bool = True
    registered_at: float = 0.0
    last_seen: float = 0.0
    latency_ms: float = 0.0


def _worker_address(w: Worker) -> str:
    return w.peer_id if w.peer_id and P2P_TRANSPORT == "p2p" else w.node_url


# ── Worker registry ──────────────────────────────────────────────────────────────


class WorkerRegistry:
    """In-memory worker registry with coverage-based model/layer assignment."""

    def __init__(self) -> None:
        self._workers: dict[str, Worker] = {}
        self._lock = asyncio.Lock()

    # ── Internal helpers (caller must hold lock) ─────────────────────────────

    def _healthy_workers(self, model_id: str | None = None) -> list[Worker]:
        """Return de-duplicated healthy workers, optionally filtered by model."""
        return [
            w
            for key, w in self._workers.items()
            if key == w.node_name
            and w.healthy
            and (model_id is None or w.model_id == model_id)
        ]

    def _layer_coverage(self, model_id: str, num_layers: int) -> list[int]:
        """Per-layer worker count for a model."""
        coverage = [0] * num_layers
        for w in self._healthy_workers(model_id):
            for layer in range(w.layers[0], w.layers[1] + 1):
                coverage[layer] += 1
        return coverage

    def _pick_model(self, model_registry: ModelRegistry, vram_gb: float) -> ModelConfig:
        """Pick the model most in need of a worker, taking VRAM-fit into account.

        Priority:
          1. **Unservable models first** (any layer at coverage 0).
             Among these, prefer the one that fits this worker's VRAM capacity
             closest to a 2-stage split (max_layers ≈ total_layers / 2).
          2. **Servable models second** (all layers covered).
             Among these, prefer the one with lowest minimum redundancy to
             shore up the thinnest regions.
        """
        best: ModelConfig | None = None
        best_key: tuple = (float("inf"),)

        for model in model_registry.list_models():
            cov = self._layer_coverage(model.model_id, model.num_layers)
            min_c = min(cov) if cov else 0
            uncovered = sum(1 for c in cov if c == 0)

            # Estimate max layers for this worker
            if vram_gb > 0.0:
                usable_mb = (vram_gb * 1024) - 512
                max_layers = max(1, int(usable_mb // model.layer_size_mb))
            else:
                max_layers = model.num_layers
            max_layers = min(max_layers, model.num_layers)

            # Score how well this worker fits a 2-stage split (closer to half is better)
            target = model.num_layers / 2
            fit_score = abs(max_layers - target)

            if uncovered > 0:
                # Tier 0: Unservable.  Sort by fit_score first, then remaining uncovered layers.
                key: tuple = (0, fit_score, uncovered)
            else:
                # Tier 1: Servable.  Sort by min redundancy, then fit_score.
                avg_c = sum(cov) / len(cov)
                key = (1, min_c, fit_score, avg_c)

            if key < best_key:
                best_key = key
                best = model

        return best or model_registry.default()

    def _pick_layers(self, model: ModelConfig, vram_gb: float) -> tuple[int, int]:
        """Choose which layers to assign to a new worker.

        VRAM-aware layer allocation:
          - If the worker is running on CPU (vram_gb == 0.0), it has sufficient
            system RAM. Capped at full model layers.
          - If on GPU (vram_gb > 0.0), calculate capacity based on the model's
            layer size estimation, reserving 512 MB for KV Cache and activation overhead.
        """
        # Determine maximum layers this node can host based on VRAM capacity
        if vram_gb > 0.0:
            usable_mb = (vram_gb * 1024) - 512
            # Each layer size computed dynamically from hidden size / weights
            max_layers = max(1, int(usable_mb // model.layer_size_mb))
        else:
            # CPU fallback assumes unlimited system RAM capacity
            max_layers = model.num_layers

        # Cap at model's total layer count
        max_layers = min(max_layers, model.num_layers)

        cov = self._layer_coverage(model.model_id, model.num_layers)
        min_cov = min(cov)

        if min_cov > 0:
            # ── Phase 2: Redundancy ──
            # Every layer is covered. Assign the best contiguous chunk of size `max_layers`
            # that has the lowest total coverage (shore up the thinnest regions).
            if max_layers >= model.num_layers:
                return (0, model.num_layers - 1)

            best_start = 0
            min_score = float("inf")

            # Slide window of size max_layers to find lowest coverage region
            for start in range(model.num_layers - max_layers + 1):
                window_cov = cov[start : start + max_layers]
                score = sum(window_cov)
                if score < min_score:
                    min_score = score
                    best_start = start

            return (best_start, best_start + max_layers - 1)

        # ── Phase 1: Gap-filling ──
        # Find the longest contiguous run of uncovered (cov == 0) layers.
        best_start = 0
        best_length = 0
        run_start: int | None = None
        run_length = 0

        for i, c in enumerate(cov):
            if c == 0:
                if run_start is None:
                    run_start = i
                    run_length = 1
                else:
                    run_length += 1
                if run_length > best_length:
                    best_length = run_length
                    best_start = run_start
            else:
                run_start = None
                run_length = 0

        # Cap the gap-filling assignment to what the VRAM capacity can host
        assigned_length = min(best_length, max_layers)
        return (best_start, best_start + assigned_length - 1)

    # ── Public API ───────────────────────────────────────────────────────────

    async def register(
        self, registration: WorkerRegistration, model_registry: ModelRegistry
    ) -> Worker:
        async with self._lock:
            existing = self._workers.get(registration.node_name)
            
            if existing:
                existing.node_url = registration.node_url
                existing.peer_id = registration.peer_id or ""
                existing.p2p_port = registration.p2p_port
                existing.cpu_cores = registration.cpu_cores
                existing.vram_gb = registration.vram_gb
                existing.visible_maddrs = registration.visible_maddrs  # <--- UPDATE THIS
                existing.healthy = True
                existing.registered_at = time.time()
                existing.last_seen = time.time()
                if existing.peer_id:
                    self._workers[existing.peer_id] = existing
                return existing

            if registration.model_id and registration.layers:
                model_id = registration.model_id
                layers = (registration.layers[0], registration.layers[-1])
            else:
                model = self._pick_model(model_registry, registration.vram_gb)
                model_id = model.model_id
                layers = self._pick_layers(model, registration.vram_gb)

            worker = Worker(
                node_name=registration.node_name,
                node_url=registration.node_url.rstrip("/"),
                model_id=model_id,
                layers=layers,
                cpu_cores=registration.cpu_cores,
                vram_gb=registration.vram_gb,
                peer_id=registration.peer_id or "",
                p2p_port=registration.p2p_port,
                visible_maddrs=registration.visible_maddrs,  # <--- PASS THIS
                registered_at=time.time(),
                last_seen=time.time(),
            )
            self._workers[registration.node_name] = worker
            if worker.peer_id:
                self._workers[worker.peer_id] = worker
            return worker

    async def select_pipeline(
        self, model_id: str, num_layers: int
    ) -> list[Worker]:
        """Return an ordered list of workers whose layers cover ``[0, num_layers)``.

        Uses a greedy interval-covering algorithm with random tie-breaking
        so that multiple healthy workers providing the same range get
        load-balanced across requests.
        """
        async with self._lock:
            candidates = sorted(
                self._healthy_workers(model_id),
                key=lambda w: (w.layers[0], -w.layers[1]),
            )
            if not candidates:
                raise RuntimeError(f"No healthy workers for model {model_id}")

            pipeline: list[Worker] = []
            target = 0
            while target < num_layers:
                eligible = [w for w in candidates if w.layers[0] <= target]
                if not eligible:
                    raise RuntimeError(
                        f"Layer coverage gap at layer {target} for model {model_id}"
                    )
                # Among eligible, prefer the one extending furthest (random among ties).
                max_end = max(w.layers[1] for w in eligible)
                best = [w for w in eligible if w.layers[1] == max_end]
                pipeline.append(random.choice(best))
                target = max_end + 1
            return pipeline

    async def evict(self, node_name: str) -> None:
        async with self._lock:
            worker = self._workers.get(node_name)
            if worker:
                worker.healthy = False

    async def snapshot(self) -> list[dict[str, object]]:
        async with self._lock:
            seen: set[str] = set()
            result: list[dict[str, object]] = []
            for key, w in self._workers.items():
                if w.node_name in seen:
                    continue
                seen.add(w.node_name)
                result.append(
                    {
                        "node_name": w.node_name,
                        "node_url": w.node_url,
                        "peer_id": w.peer_id,
                        "p2p_port": w.p2p_port,
                        "visible_maddrs": w.visible_maddrs,  # <--- EXPOSE IN SNAPSHOT
                        "model_id": w.model_id,
                        "layers": list(w.layers),
                        "healthy": w.healthy,
                        "last_seen": round(time.time() - w.last_seen, 1),
                        "cpu_cores": w.cpu_cores,
                        "vram_gb": w.vram_gb,
                        "latency_ms": round(w.latency_ms, 1),
                    }
                )
            return result

    async def model_readiness(
        self, model_registry: ModelRegistry
    ) -> dict[str, dict[str, object]]:
        """Per-model coverage report: ready when every layer has ≥ 1 worker."""
        async with self._lock:
            result: dict[str, dict[str, object]] = {}
            for model in model_registry.list_models():
                cov = self._layer_coverage(model.model_id, model.num_layers)
                min_cov = min(cov) if cov else 0
                result[model.model_id] = {
                    "ready": min_cov > 0,
                    "min_redundancy": min_cov,
                    "layers_covered": sum(1 for c in cov if c > 0),
                    "total_layers": model.num_layers,
                }
            return result

    async def heartbeat(self, node_name: str, latency_ms: float = 0.0) -> bool:
        async with self._lock:
            worker = self._workers.get(node_name)
            if worker:
                worker.last_seen = time.time()
                worker.healthy = True
                worker.latency_ms = latency_ms
                return True
            return False

    async def evict_stale(
        self, soft_stale_seconds: float, hard_stale_seconds: float
    ) -> dict[str, list[str]]:
        """Two-stage eviction: pause routing on soft timeout, wipe memory on hard timeout."""
        async with self._lock:
            now = time.time()
            soft_cutoff = now - soft_stale_seconds
            hard_cutoff = now - hard_stale_seconds
            
            soft_evicted: list[str] = []
            hard_evicted: list[str] = []
            keys_to_delete: list[str] = []
            
            for key, worker in self._workers.items():
                # Check Stage 2: Hard Eviction first
                if worker.last_seen < hard_cutoff:
                    keys_to_delete.append(key)
                    if key == worker.node_name:
                        hard_evicted.append(worker.node_name)
                        
                # Check Stage 1: Soft Eviction second
                elif worker.healthy and worker.last_seen < soft_cutoff:
                    worker.healthy = False
                    if key == worker.node_name:
                        soft_evicted.append(worker.node_name)
            
            # Execute hard deletions
            for key in keys_to_delete:
                del self._workers[key]
                
            return {"soft": soft_evicted, "hard": hard_evicted}


# ── Globals ──────────────────────────────────────────────────────────────────────


registry = WorkerRegistry()
model_registry: ModelRegistry | None = None
_tokenizers: dict[str, AutoTokenizer] = {}


async def _get_tokenizer(model_id: str) -> AutoTokenizer:
    """Lazy-load and cache tokenizers per model."""
    if model_id not in _tokenizers:
        _tokenizers[model_id] = await asyncio.to_thread(
            AutoTokenizer.from_pretrained, model_id
        )
    return _tokenizers[model_id]


# ── Lifespan ─────────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(_: FastAPI):
    global model_registry, transport
    transport = create_transport(
        transport_type=P2P_TRANSPORT,
        initial_peers=[],  # Gateway acts as the swarm bootstrap/relay point
        listen_port=P2P_PORT,
        relay_mode=True,
    )
    await transport.start()
    if hasattr(transport, "get_peer_id") and transport.get_peer_id():
        print(f"[gateway] P2P node started! PeerID: {transport.get_peer_id()}", flush=True)

    model_registry = await load_model_registry()
    for model in model_registry.list_models():
        await _get_tokenizer(model.model_id)
        print(f"[gateway] Registered model {model.model_id} ({model.num_layers} layers)")

    async def _staleness_loop() -> None:
        """Proactively manage 2-stage worker evictions."""
        while True:
            # Run the loop aggressively enough to catch the 5-second soft limit
            await asyncio.sleep(HEARTBEAT_SOFT_STALE_SECONDS / 2)
            
            evicted = await registry.evict_stale(
                HEARTBEAT_SOFT_STALE_SECONDS, 
                HEARTBEAT_HARD_STALE_SECONDS
            )
            
            if evicted["soft"]:
                print(f"[gateway] Soft-evicted (paused routing): {evicted['soft']}")
            if evicted["hard"]:
                print(f"[gateway] Hard-evicted (memory wiped): {evicted['hard']}")

    staleness_task = asyncio.create_task(_staleness_loop())
    yield
    staleness_task.cancel()
    if transport is not None:
        await transport.stop()


app = FastAPI(title="Kiwi Enterprise Compute Swarm", lifespan=lifespan)


# ── Routes ───────────────────────────────────────────────────────────────────────


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(INDEX_FILE)


@app.get("/health")
async def health() -> dict[str, object]:
    return {
        "status": "ok" if model_registry else "initializing",
        "models_loaded": len(model_registry.list_models()) if model_registry else 0,
    }


import ipaddress

@app.get("/p2p/bootstrap")
async def p2p_bootstrap() -> dict[str, object]:
    if transport is None:
        raise HTTPException(status_code=503, detail="Transport not initialized")
    if P2P_TRANSPORT != "p2p":
        raise HTTPException(status_code=400, detail="P2P transport is not enabled on this gateway")
    
    public_ip = get_public_ip()
    peer_id = await transport.get_address()
    public_maddr = f"/ip4/{public_ip}/tcp/{P2P_PORT}/p2p/{peer_id}"
    
    return {
        "peer_id": peer_id,
        "p2p_port": P2P_PORT,
        "visible_maddrs": [public_maddr], # Distribute the verified public route
    }


@app.get("/v1/models")
async def list_models() -> dict[str, object]:
    """OpenAI-compatible model listing with readiness info."""
    if not model_registry:
        raise HTTPException(status_code=503, detail="Model registry not ready")
    readiness = await registry.model_readiness(model_registry)
    return {
        "data": [
            {
                "id": m.model_id,
                "aliases": m.aliases,
                "num_layers": m.num_layers,
                "default_dtype": m.default_dtype,
                **(readiness.get(m.model_id, {})),
            }
            for m in model_registry.list_models()
        ]
    }


@app.get("/registry")
async def registry_view() -> dict[str, object]:
    result: dict[str, object] = {"workers": await registry.snapshot()}
    if model_registry:
        result["models"] = await registry.model_readiness(model_registry)
    return result


async def _connect_to_worker_p2p(worker: Worker) -> None:
    try:
        # If the worker has no visible addresses, or if we haven't connected yet,
        # we MUST force a manual DNS dial as a last resort.
        hostname = worker.node_url.split("//")[-1].split(":")[0]
        fallback_maddr = f"/dns/{hostname}/tcp/{worker.p2p_port}/p2p/{worker.peer_id}"
        
        # Combine verified addresses + manual fallback
        maddrs = list(dict.fromkeys((worker.visible_maddrs or []) + [fallback_maddr]))
        
        if hasattr(transport, "connect_to_peer"):
            for maddr in maddrs:
                try:
                    await transport.connect_to_peer(maddr)
                    print(f"[Gateway] Successfully connected to {worker.node_name} via {maddr}", flush=True)
                    return
                except Exception:
                    continue
    except Exception as exc:
        print(f"[Gateway] Failed to wire {worker.node_name}: {exc}", flush=True)


@app.post("/register")
async def register_worker(registration: WorkerRegistration) -> JSONResponse:
    if not model_registry:
        raise HTTPException(status_code=503, detail="Gateway not ready")
    worker = await registry.register(registration, model_registry)
    if P2P_TRANSPORT == "p2p" and transport is not None and worker.peer_id:
        asyncio.create_task(_connect_to_worker_p2p(worker))
    model = model_registry.resolve(worker.model_id)
    return JSONResponse(
        {
            "model_id": worker.model_id,
            "layers": list(worker.layers),
            "total_layers": model.num_layers if model else 0,
            "dtype": model.default_dtype if model else "bfloat16",
        }
    )


class HeartbeatRequest(BaseModel):
    node_name: str
    node_url: str
    latency_ms: float = 0.0


@app.post("/heartbeat")
async def heartbeat(request: HeartbeatRequest) -> dict[str, str]:
    exists = await registry.heartbeat(request.node_name, request.latency_ms)
    if not exists:
        raise HTTPException(status_code=404, detail="Worker not registered")
    return {"status": "ok"}


@app.post("/deregister")
async def deregister(request: HeartbeatRequest) -> dict[str, str]:
    await registry.evict(request.node_name)
    return {"status": "evicted"}


# ── Tokenization & SSE helpers ───────────────────────────────────────────────────


def _prompt_ids(
    tokenizer: AutoTokenizer, messages: list[ChatMessage]
) -> torch.Tensor:
    rendered_messages = [message.model_dump() for message in messages]
    if getattr(tokenizer, "chat_template", None):
        ids = tokenizer.apply_chat_template(
            rendered_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if hasattr(ids, "get"):
            ids = ids.get("input_ids", ids)
        return (
            ids
            if isinstance(ids, torch.Tensor)
            else torch.tensor([ids], dtype=torch.long)
        )
    text = (
        "\n".join(f"{m.role}: {m.content}" for m in messages) + "\nassistant:"
    )
    return tokenizer(text, return_tensors="pt").input_ids


def _sse(payload: dict[str, object]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _chunk(
    request_id: str,
    model_id: str,
    content: str | None,
    active_nodes: dict[str, str] | None = None,
    finish_reason: str | None = None,
) -> str:
    return _sse(
        {
            "id": f"chatcmpl-{request_id}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": content} if content else {},
                    "finish_reason": finish_reason,
                }
            ],
            "active_nodes": active_nodes,
        }
    )


# ── Streaming completion ─────────────────────────────────────────────────────────


async def _stream_completion(
    request: ChatCompletionRequest,
) -> AsyncIterator[str]:
    assert model_registry is not None

    model = model_registry.resolve(request.model)
    if not model:
        yield _sse(
            {
                "error": {
                    "message": f"Unknown model: {request.model}",
                    "type": "invalid_request_error",
                }
            }
        )
        yield "data: [DONE]\n\n"
        return

    tokenizer = await _get_tokenizer(model.model_id)
    request_id = uuid.uuid4().hex
    prompt_ids = _prompt_ids(tokenizer, request.messages).to(
        dtype=torch.long, device="cpu"
    )
    generated_ids: list[int] = []
    reroutes = 0

    # Build EOS token set for the worker to detect end-of-sequence
    eos_ids = tokenizer.eos_token_id
    eos_set = set(eos_ids) if isinstance(eos_ids, list) else {eos_ids}
    eos_header = ",".join(str(e) for e in eos_set)

    while True:  # Outer retry loop for rerouting
        try:
            pipeline = await registry.select_pipeline(
                model.model_id, model.num_layers
            )
        except RuntimeError as error:
            yield _sse({"error": {"message": str(error), "type": "server_error"}})
            yield "data: [DONE]\n\n"
            return

        if transport is None:
            yield _sse({"error": {"message": "Transport layer not initialized", "type": "server_error"}})
            yield "data: [DONE]\n\n"
            return

        first = pipeline[0]
        first_addr = _worker_address(first)
        remaining_urls = ",".join(_worker_address(w) for w in pipeline[1:])
        active_nodes = {
            f"stage_{i}": _worker_address(w) for i, w in enumerate(pipeline)
        }

        # Build full context: original prompt + any previously generated tokens
        if generated_ids:
            context_ids = torch.cat(
                [prompt_ids, torch.tensor([generated_ids], dtype=torch.long)],
                dim=1,
            )
        else:
            context_ids = prompt_ids

        body = safetensors_save({"input_ids": context_ids.contiguous()})
        remaining_tokens = request.max_tokens - len(generated_ids)

        stream_ok = True

        try:
            async for chunk_bytes in transport.send_generate(
                target=first_addr,
                payload=body,
                headers={
                    "X-Request-ID": request_id,
                    "X-Pipeline": remaining_urls,
                    "X-Max-Tokens": str(remaining_tokens),
                    "X-Eos-Token-Ids": eos_header,
                    "Content-Type": "application/octet-stream",
                },
                timeout=REQUEST_TIMEOUT_SECONDS,
            ):
                line = chunk_bytes.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if "error" in data:
                    # Worker-reported pipeline failure
                    failed_node = data.get("failed_node")
                    print(f"[gateway] Worker reported pipeline error from {failed_node}: {data.get('error')}", flush=True)
                    await registry.evict(
                        failed_node or first_addr
                    )
                    stream_ok = False
                    break

                token_id = data.get("token_id")
                if token_id is None:
                    continue

                generated_ids.append(token_id)

                if token_id in eos_set:
                    break

                text = tokenizer.decode(
                    [token_id], skip_special_tokens=False
                )
                if text:
                    yield _chunk(
                        request_id,
                        model.model_id,
                        text,
                        active_nodes=active_nodes,
                    )

        except Exception as error:
            print(f"[gateway] Transport error communicating with {first_addr}: {error}")
            await registry.evict(first_addr)
            stream_ok = False

        if stream_ok:
            break

        # Rerouting: evict failed node, rebuild pipeline, retry with full context
        reroutes += 1
        if reroutes > MAX_REROUTES:
            yield _sse(
                {
                    "error": {
                        "message": "Retry budget exhausted",
                        "type": "server_error",
                    }
                }
            )
            break

        yield _chunk(
            request_id,
            model.model_id,
            "[REROUTING...]",
            active_nodes=active_nodes,
        )
        # Loop continues: will rebuild pipeline and send full context as prefill

    yield _chunk(request_id, model.model_id, None, finish_reason="stop")
    yield "data: [DONE]\n\n"


# ── Chat completions endpoint ────────────────────────────────────────────────────


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if not model_registry:
        raise HTTPException(status_code=503, detail="Gateway not ready")
    if request.stream:
        return StreamingResponse(
            _stream_completion(request),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    # Non-streaming fallback.
    pieces: list[str] = []
    model = model_registry.resolve(request.model) or model_registry.default()
    model_id = model.model_id if model else "unknown"
    async for event in _stream_completion(request):
        if not event.startswith("data: {"):
            continue
        try:
            payload = json.loads(event.removeprefix("data: ").strip())
            pieces.append(
                payload["choices"][0]["delta"].get("content", "")
            )
        except (KeyError, IndexError, json.JSONDecodeError):
            continue
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "".join(pieces),
                },
                "finish_reason": "stop",
            }
        ],
    }
