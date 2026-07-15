"""Kiwi gateway: OpenAI-style streaming API, registry, and active recompute."""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Literal

import httpx
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from safetensors.torch import save as safetensors_save
from transformers import AutoTokenizer


MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-0.5B")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "120"))
MAX_REROUTES = int(os.getenv("MAX_REROUTES", "4"))
INDEX_FILE = Path(__file__).with_name("index.html")


class WorkerRegistration(BaseModel):
    node_name: str
    node_url: str
    cpu_cores: int = Field(ge=1)
    vram_gb: float = Field(ge=0)
    pool_hint: Literal["A", "B"] | None = None


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    stream: bool = True
    max_tokens: int = Field(default=128, ge=1, le=512)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


@dataclass
class Worker:
    node_name: str
    node_url: str
    stage: Literal["A", "B"]
    layers: tuple[int, int]
    cpu_cores: int
    vram_gb: float
    healthy: bool = True
    registered_at: float = 0.0


class WorkerRegistry:
    """A deliberately small in-memory DHT-style registry for the local PoC."""

    def __init__(self) -> None:
        self._workers: dict[str, Worker] = {}
        self._lock = asyncio.Lock()

    async def register(self, registration: WorkerRegistration) -> Worker:
        async with self._lock:
            existing = self._workers.get(registration.node_name)
            if existing:
                existing.node_url = registration.node_url
                existing.cpu_cores = registration.cpu_cores
                existing.vram_gb = registration.vram_gb
                existing.healthy = True
                existing.registered_at = time.time()
                return existing

            stage = self._choose_stage(registration.pool_hint)
            layers = (0, 11) if stage == "A" else (12, 23)
            worker = Worker(
                node_name=registration.node_name,
                node_url=registration.node_url.rstrip("/"),
                stage=stage,
                layers=layers,
                cpu_cores=registration.cpu_cores,
                vram_gb=registration.vram_gb,
                registered_at=time.time(),
            )
            self._workers[worker.node_url] = worker
            # Keep the name lookup too: restart registration must retain its assignment.
            self._workers[registration.node_name] = worker
            return worker

    def _choose_stage(self, hint: Literal["A", "B"] | None) -> Literal["A", "B"]:
        counts = {
            stage: sum(
                1
                for key, worker in self._workers.items()
                if key == worker.node_url and worker.healthy and worker.stage == stage
            )
            for stage in ("A", "B")
        }
        if counts["A"] == counts["B"] and hint:
            return hint
        return "A" if counts["A"] <= counts["B"] else "B"

    async def select_pair(self) -> tuple[Worker, Worker]:
        async with self._lock:
            stage_a = [
                worker
                for key, worker in self._workers.items()
                if key == worker.node_url and worker.healthy and worker.stage == "A"
            ]
            stage_b = [
                worker
                for key, worker in self._workers.items()
                if key == worker.node_url and worker.healthy and worker.stage == "B"
            ]
            if not stage_a or not stage_b:
                raise RuntimeError("No healthy Stage A/Stage B worker pair is registered")
            return random.choice(stage_a), random.choice(stage_b)

    async def evict(self, node_url: str) -> None:
        async with self._lock:
            worker = self._workers.get(node_url.rstrip("/"))
            if worker:
                worker.healthy = False

    async def snapshot(self) -> list[dict[str, object]]:
        async with self._lock:
            return [
                {
                    "node_name": worker.node_name,
                    "node_url": worker.node_url,
                    "stage": worker.stage,
                    "layers": list(worker.layers),
                    "healthy": worker.healthy,
                }
                for key, worker in self._workers.items()
                if key == worker.node_url
            ]


registry = WorkerRegistry()
tokenizer: AutoTokenizer | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global tokenizer
    # Tokenization is gateway-only; workers never receive raw prompt text.
    tokenizer = await asyncio.to_thread(AutoTokenizer.from_pretrained, MODEL_ID)
    yield


app = FastAPI(title="Kiwi Enterprise Compute Swarm", lifespan=lifespan)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(INDEX_FILE)


@app.get("/health")
async def health() -> dict[str, object]:
    return {"status": "ok", "tokenizer_ready": tokenizer is not None}


@app.get("/registry")
async def registry_view() -> dict[str, object]:
    return {"workers": await registry.snapshot()}


@app.post("/register")
async def register_worker(registration: WorkerRegistration) -> JSONResponse:
    worker = await registry.register(registration)
    return JSONResponse({"stage": worker.stage, "layers": list(worker.layers)})


def _prompt_ids(messages: list[ChatMessage]) -> torch.Tensor:
    if tokenizer is None:
        raise HTTPException(status_code=503, detail="Tokenizer is still loading")
    rendered_messages = [message.model_dump() for message in messages]
    if getattr(tokenizer, "chat_template", None):
        ids = tokenizer.apply_chat_template(
            rendered_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        return ids if isinstance(ids, torch.Tensor) else torch.tensor([ids], dtype=torch.long)
    text = "\n".join(f"{message.role}: {message.content}" for message in messages) + "\nassistant:"
    return tokenizer(text, return_tensors="pt").input_ids


def _sse(payload: dict[str, object]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _chunk(request_id: str, content: str | None, finish_reason: str | None = None) -> str:
    return _sse(
        {
            "id": f"chatcmpl-{request_id}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": MODEL_ID,
            "choices": [{"index": 0, "delta": {"content": content} if content else {}, "finish_reason": finish_reason}],
        }
    )


async def _stream_completion(request: ChatCompletionRequest) -> AsyncIterator[str]:
    assert tokenizer is not None
    request_id = uuid.uuid4().hex
    prompt_ids = _prompt_ids(request.messages).to(dtype=torch.long, device="cpu")
    generated_ids: list[int] = []
    is_prefill = True
    reroutes = 0

    try:
        node_a, node_b = await registry.select_pair()
    except RuntimeError as error:
        yield _sse({"error": {"message": str(error), "type": "server_error"}})
        yield "data: [DONE]\n\n"
        return

    timeout = httpx.Timeout(REQUEST_TIMEOUT_SECONDS, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        while len(generated_ids) < request.max_tokens:
            context_ids = torch.cat(
                [prompt_ids, torch.tensor([generated_ids], dtype=torch.long)], dim=1
            ) if generated_ids else prompt_ids
            input_ids = context_ids if is_prefill else torch.tensor([[generated_ids[-1]]], dtype=torch.long)
            body = safetensors_save({"input_ids": input_ids.contiguous()})

            try:
                response = await client.post(
                    f"{node_a.node_url}/forward",
                    headers={
                        "X-Request-ID": request_id,
                        "X-Step": "prefill" if is_prefill else "decode",
                        "X-Target-B": node_b.node_url,
                        "Content-Type": "application/octet-stream",
                    },
                    content=body,
                )
                response.raise_for_status()
                token_id = int(response.json()["next_token_id"])
            except httpx.HTTPError as error:
                reroutes += 1
                # Stage A tells us if its direct P2P hop failed; otherwise A itself is suspect.
                failed_url = None
                if isinstance(error, httpx.HTTPStatusError):
                    failed_url = error.response.headers.get("X-Kiwi-Failed-Node")
                await registry.evict(failed_url or node_a.node_url)
                yield _chunk(request_id, "[REROUTING...]")
                if reroutes > MAX_REROUTES:
                    yield _sse({"error": {"message": "Retry budget exhausted", "type": "server_error"}})
                    break
                try:
                    node_a, node_b = await registry.select_pair()
                except RuntimeError as no_pair:
                    yield _sse({"error": {"message": str(no_pair), "type": "server_error"}})
                    break
                # Preserve generated_ids; a full prompt+history prefill recreates both local caches.
                is_prefill = True
                continue
            except (KeyError, TypeError, ValueError) as error:
                yield _sse({"error": {"message": f"Invalid worker response: {error}", "type": "server_error"}})
                break

            generated_ids.append(token_id)
            text = tokenizer.decode([token_id], skip_special_tokens=False)
            eos_ids = tokenizer.eos_token_id
            eos_set = set(eos_ids) if isinstance(eos_ids, list) else {eos_ids}
            if token_id in eos_set:
                break
            if text:
                yield _chunk(request_id, text)
            is_prefill = False

    yield _chunk(request_id, None, "stop")
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if tokenizer is None:
        raise HTTPException(status_code=503, detail="Tokenizer is still loading")
    if request.stream:
        return StreamingResponse(
            _stream_completion(request),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    # Retaining a single route keeps the API useful for clients that probe with stream=false.
    pieces: list[str] = []
    async for event in _stream_completion(request):
        if not event.startswith("data: {"):
            continue
        try:
            payload = json.loads(event.removeprefix("data: ").strip())
            pieces.append(payload["choices"][0]["delta"].get("content", ""))
        except (KeyError, IndexError, json.JSONDecodeError):
            continue
    return {"id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion", "model": MODEL_ID, "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(pieces)}, "finish_reason": "stop"}]}
