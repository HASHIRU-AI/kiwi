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
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from safetensors.torch import save as safetensors_save
from transformers import AutoTokenizer

from models import ModelConfig, ModelRegistry, load_model_registry
from transport import KiwiTransport, create_transport

from hivemind.p2p.p2p_daemon_bindings import PeerID
from hivemind.utils.multiaddr.multiaddr import Multiaddr


REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "120"))
MAX_REROUTES = int(os.getenv("MAX_REROUTES", "4"))
INDEX_FILE = Path(__file__).with_name("index.html")

# ── Transport configuration ──
P2P_TRANSPORT = os.getenv("KIWI_P2P_TRANSPORT", "http").lower()
P2P_PORT = int(os.getenv("KIWI_P2P_PORT", "4001"))
transport: KiwiTransport | None = None

# ── 2-Stage Eviction Thresholds ──
HEARTBEAT_SOFT_STALE_SECONDS = float(os.getenv("HEARTBEAT_SOFT_STALE_SECONDS", "30"))
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
    healthy: bool = True
    registered_at: float = 0.0
    last_seen: float = 0.0
    latency_ms: float = 0.0


def _worker_address(w: Worker) -> str:
    return w.peer_id if w.peer_id and P2P_TRANSPORT == "p2p" else w.node_url


# ── Worker registry ──────────────────────────────────────────────────────────────


class ReplicaUnavailable(RuntimeError):
    """Raised when a requested replica_id exceeds the current pipe count R."""


class WorkerRegistry:
    """In-memory worker registry with coverage-based model/layer assignment."""

    def __init__(self) -> None:
        self._workers: dict[str, Worker] = {}
        self._pipes: dict[str, list[list[Worker]]] = {}   # model_id -> R pipes
        self._spares: dict[str, list[Worker]] = {}        # model_id -> spare workers
        self._mesh_enabled: dict[str, bool] = {}          # model_id -> tiling ok (mesh vs R=1 fallback)
        self._prev_slots: dict[str, int] = {}             # node_name -> last pipe index
        self._round: dict[str, int] = {}                  # model_id -> averaging round counter
        self._model_layers: dict[str, int] = {}           # model_id -> num_layers (for recompute)
        self._lock = asyncio.Lock()

    def _find_worker(self, identifier: str) -> Worker | None:
        """Find a worker by node_name, node_url, or peer_id."""
        if not identifier:
            return None
        cleaned = identifier.rstrip("/")
        worker = self._workers.get(cleaned) or self._workers.get(identifier)
        if worker:
            return worker
        for w in self._workers.values():
            if cleaned in (w.node_name, w.node_url.rstrip("/"), w.peer_id):
                return w
        return None

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

    def _stage_groups(self, model_id: str) -> dict[tuple[int, int], list[Worker]]:
        """Group healthy workers by stage key (first_layer, last_layer), ordered by first layer."""
        groups: dict[tuple[int, int], list[Worker]] = {}
        for w in self._healthy_workers(model_id):
            groups.setdefault(w.layers, []).append(w)
        return dict(sorted(groups.items()))

    @staticmethod
    def _stages_tile(stage_keys: list[tuple[int, int]], num_layers: int) -> bool:
        """True iff the stage keys tile [0, num_layers) exactly once, no gaps or overlaps."""
        if not stage_keys:
            return False
        expected_start = 0
        for first, last in sorted(stage_keys):
            if first != expected_start or last < first:
                return False
            expected_start = last + 1
        return expected_start == num_layers

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
            # ── Phase 2: Redundancy — clone mode ──
            # A complete pipe exists. Assign an EXACT copy of an existing stage's
            # range (the stage with the fewest healthy workers, ties broken by
            # lowest first layer) so replicas stay homogeneous — an enforced
            # invariant the mesh's tiling check depends on.
            groups = self._stage_groups(model.model_id)
            if self._stages_tile(list(groups.keys()), model.num_layers):
                stage_key = min(groups.keys(), key=lambda k: (len(groups[k]), k[0]))
                stage_size = stage_key[1] - stage_key[0] + 1
                if stage_size > max_layers:
                    print(
                        f"[gateway] Clone-mode warning: assigning stage {stage_key} "
                        f"({stage_size} layers) exceeds estimated capacity "
                        f"({max_layers} layers) for this worker.",
                        flush=True,
                    )
                return stage_key

            # Ragged existing ranges: fall back to the sliding-window heuristic.
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
                # Node is still in memory (soft restart / IP change)
                existing.node_url = registration.node_url.rstrip("/")
                existing.peer_id = registration.peer_id or ""
                existing.p2p_port = registration.p2p_port
                existing.cpu_cores = registration.cpu_cores
                existing.vram_gb = registration.vram_gb
                existing.healthy = True
                existing.registered_at = time.time()
                existing.last_seen = time.time()
                self._workers[existing.node_name] = existing
                if existing.peer_id:
                    self._workers[existing.peer_id] = existing
                if existing.node_url:
                    self._workers[existing.node_url] = existing
                cfg = model_registry.resolve(existing.model_id)
                if cfg:
                    self._model_layers[existing.model_id] = cfg.num_layers
                self._recompute_pipes_locked(existing.model_id)
                return existing

            # Node is brand new or was hard-evicted. 
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
                registered_at=time.time(),
                last_seen=time.time(),
            )
            self._workers[worker.node_name] = worker
            if worker.peer_id:
                self._workers[worker.peer_id] = worker
            if worker.node_url:
                self._workers[worker.node_url] = worker
            cfg = model_registry.resolve(model_id)
            if cfg:
                self._model_layers[model_id] = cfg.num_layers
            self._recompute_pipes_locked(model_id)
            return worker

    def _compute_pipeline(
        self, model_id: str, num_layers: int
    ) -> list[Worker]:
        """Greedy fallback: one ordered pipeline covering ``[0, num_layers)``.

        Used when stage ranges are ragged and the mesh tiling check fails.
        Must be called with self._lock held.
        """
        candidates = sorted(
            self._healthy_workers(model_id),
            key=lambda w: (w.layers[0], -w.layers[1]),
        )
        if not candidates:
            raise RuntimeError(f"No healthy workers for model {model_id}")

        pipeline: list[Worker] = []
        target = 0
        i = 0
        pipes = self._pipes.get(model_id, [])
        pinned_nodes = {w.node_name for p in pipes for w in p}

        while target < num_layers:
            eligible: list[Worker] = []
            while i < len(candidates) and candidates[i].layers[0] <= target:
                eligible.append(candidates[i])
                i += 1
            if not eligible:
                raise RuntimeError(
                    f"Layer coverage gap at layer {target} for model {model_id}"
                )
            max_end = max(w.layers[1] for w in eligible)
            best = [w for w in eligible if w.layers[1] == max_end]
            pinned_best = [w for w in best if w.node_name in pinned_nodes]
            pipeline.append(random.choice(pinned_best if pinned_best else best))
            target = max_end + 1
        return pipeline

    def _compute_pipes(self, model_id: str, num_layers: int) -> list[list[Worker]]:
        """Form R complete pipes from the current healthy set (§3.1).

        Must be called with self._lock held. Also updates self._spares,
        self._mesh_enabled, and self._prev_slots.
        """
        groups = self._stage_groups(model_id)
        if not groups:
            raise RuntimeError(f"No healthy workers for model {model_id}")

        if not self._stages_tile(list(groups.keys()), num_layers):
            # Ragged layer ranges — mesh disabled for this model, R = 1.
            print(
                f"[gateway] MESH DISABLED for {model_id}: stage ranges "
                f"{list(groups.keys())} do not tile [0, {num_layers}). "
                f"Falling back to greedy single pipeline (R=1).",
                flush=True,
            )
            self._mesh_enabled[model_id] = False
            self._spares[model_id] = []
            return [self._compute_pipeline(model_id, num_layers)]

        self._mesh_enabled[model_id] = True

        # Slot stability: keep a worker's pipe index across recomputation when
        # possible. Stability is for humans (logs, request-ID namespaces), not
        # correctness — a worker never leaves its layer range.
        never = 10**9
        for key in groups:
            groups[key].sort(
                key=lambda w: (self._prev_slots.get(w.node_name, never), w.node_name)
            )

        r_count = min(len(g) for g in groups.values())
        stage_order = sorted(groups.keys())
        pipes = [
            [groups[stage][r] for stage in stage_order] for r in range(r_count)
        ]
        spares = [w for g in groups.values() for w in g[r_count:]]
        self._spares[model_id] = spares

        for r, pipe in enumerate(pipes):
            for w in pipe:
                self._prev_slots[w.node_name] = r

        return pipes

    def _recompute_pipes_locked(self, model_id: str) -> None:
        """Recompute pipes for a model after a membership change. Lock held."""
        num_layers = self._model_layers.get(model_id)
        if num_layers is None:
            return
        old = [[w.node_name for w in p] for p in self._pipes.get(model_id, [])]
        try:
            self._pipes[model_id] = self._compute_pipes(model_id, num_layers)
        except RuntimeError:
            self._pipes.pop(model_id, None)
            self._spares.pop(model_id, None)
            return
        new = [[w.node_name for w in p] for p in self._pipes[model_id]]
        spare_names = [w.node_name for w in self._spares.get(model_id, [])]
        if old != new:
            print(
                f"[gateway] Pipes for {model_id}: {old} -> {new} "
                f"(R={len(new)}, spares={spare_names})",
                flush=True,
            )

    def _pipes_valid(self, model_id: str) -> bool:
        pipes = self._pipes.get(model_id)
        if not pipes:
            return False
        return all(w.healthy for p in pipes for w in p)

    async def get_pipe(
        self, model_id: str, num_layers: int, replica_id: int = 0
    ) -> list[Worker]:
        """Return the pipe for a replica, recomputing on membership change."""
        async with self._lock:
            self._model_layers[model_id] = num_layers
            if not self._pipes_valid(model_id):
                self._recompute_pipes_locked(model_id)
            pipes = self._pipes.get(model_id)
            if not pipes:
                raise RuntimeError(f"No healthy workers for model {model_id}")
            if replica_id >= len(pipes):
                raise ReplicaUnavailable(
                    f"replica_id {replica_id} >= R ({len(pipes)}) for {model_id}"
                )
            return pipes[replica_id]

    async def evict(self, node_name: str) -> None:
        async with self._lock:
            worker = self._find_worker(node_name)
            if worker:
                worker.healthy = False
                self._recompute_pipes_locked(worker.model_id)

    async def plan_round(
        self, model_id: str, x: int = 10
    ) -> tuple[int, list[dict[str, object]]]:
        """Compute per-worker round parameters for one averaging round (§3.4).

        The gateway is authoritative for BOTH `contribute` and `expect` so
        they stay consistent by construction: actives (workers placed in a
        pipe) are contributors; spares receive without contributing.
        """
        async with self._lock:
            if not self._pipes_valid(model_id):
                self._recompute_pipes_locked(model_id)
            pipes = self._pipes.get(model_id)
            if not pipes:
                raise RuntimeError(f"No healthy workers for model {model_id}")
            if not self._mesh_enabled.get(model_id, False):
                raise RuntimeError(
                    f"Mesh disabled for {model_id} (ragged stage ranges, R=1 fallback)"
                )

            self._round[model_id] = self._round.get(model_id, 0) + 1
            round_id = self._round[model_id]

            active_names = {w.node_name for p in pipes for w in p}
            plan: list[dict[str, object]] = []
            for stage_key, members in self._stage_groups(model_id).items():
                members = sorted(members, key=lambda w: w.node_name)
                num_contributors = sum(
                    1 for w in members if w.node_name in active_names
                )
                for w in members:
                    is_contributor = w.node_name in active_names
                    send_to = (
                        [
                            _worker_address(p)
                            for p in members
                            if p.node_name != w.node_name
                        ]
                        if is_contributor
                        else []
                    )
                    plan.append(
                        {
                            "node_name": w.node_name,
                            "node_url": w.node_url,
                            "stage": list(stage_key),
                            "round": {
                                "round_id": round_id,
                                "send_to": send_to,
                                "expect": (num_contributors - 1)
                                if is_contributor
                                else num_contributors,
                                "contribute": is_contributor,
                                "x": x,
                            },
                        }
                    )
            return round_id, plan

    def _role_of(self, w: Worker) -> tuple[str, int | None]:
        """Return (role, replica_id) for a worker. Lock held."""
        for r, pipe in enumerate(self._pipes.get(w.model_id, [])):
            if any(pw.node_name == w.node_name for pw in pipe):
                return "active", r
        if any(s.node_name == w.node_name for s in self._spares.get(w.model_id, [])):
            return "spare", None
        # Legacy label for the ragged-range fallback: layer-matches an active.
        active_nodes = {
            pw.node_name for p in self._pipes.get(w.model_id, []) for pw in p
        }
        if any(
            aw.node_name in active_nodes
            and aw.model_id == w.model_id
            and aw.layers == w.layers
            for aw in self._workers.values()
        ):
            return "spare", None
        return "unassigned", None

    def _layer_peers_of(self, w: Worker) -> list[str]:
        """Addresses of healthy workers sharing this worker's stage identity, minus self.

        Sorted by node_name so peer order is deterministic. Lock held.
        """
        peers = [
            p
            for p in self._healthy_workers(w.model_id)
            if p.layers == w.layers and p.node_name != w.node_name
        ]
        peers.sort(key=lambda p: p.node_name)
        return [_worker_address(p) for p in peers]

    async def snapshot(self) -> list[dict[str, object]]:
        async with self._lock:
            seen: set[str] = set()
            result: list[dict[str, object]] = []

            for key, w in self._workers.items():
                if w.node_name in seen:
                    continue
                seen.add(w.node_name)

                role, replica_id = self._role_of(w)

                result.append(
                    {
                        "node_name": w.node_name,
                        "role": role,
                        "replica_id": replica_id,
                        "node_url": w.node_url,
                        "peer_id": w.peer_id,
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

    async def heartbeat(self, node_name: str, latency_ms: float = 0.0) -> dict[str, object] | None:
        async with self._lock:
            worker = self._find_worker(node_name)
            if worker is None:
                return None

            was_unhealthy = not worker.healthy
            worker.last_seen = time.time()
            worker.healthy = True
            worker.latency_ms = latency_ms
            if was_unhealthy:
                # A soft-evicted worker came back: rebuild pipes with it.
                self._recompute_pipes_locked(worker.model_id)

            role, replica_id = self._role_of(worker)

            # ── Legacy standby push target (removed in mesh phase 2) ──
            # An active's "standby" is a healthy same-stage worker not placed
            # in any pipe — i.e. the first spare of its layer-peer group.
            standby_peer_id = ""
            if role == "active":
                pipe_nodes = {
                    pw.node_name
                    for p in self._pipes.get(worker.model_id, [])
                    for pw in p
                }
                for w2 in self._workers.values():
                    if (
                        w2.healthy
                        and w2.model_id == worker.model_id
                        and w2.layers == worker.layers
                        and w2.node_name != worker.node_name
                        and w2.node_name not in pipe_nodes
                    ):
                        standby_peer_id = _worker_address(w2)
                        break

            return {
                "status": "ok",
                "role": role,
                "replica_id": replica_id,
                "layer_peers": self._layer_peers_of(worker),
                "round": self._round.get(worker.model_id, 0),
                "standby_peer_id": standby_peer_id,
            }

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
            
            affected_models: set[str] = set()

            for key, worker in self._workers.items():
                # Check Stage 2: Hard Eviction first
                if worker.last_seen < hard_cutoff:
                    keys_to_delete.append(key)
                    if key == worker.node_name:
                        hard_evicted.append(worker.node_name)
                        affected_models.add(worker.model_id)

                # Check Stage 1: Soft Eviction second
                elif worker.healthy and worker.last_seen < soft_cutoff:
                    worker.healthy = False
                    if key == worker.node_name:
                        soft_evicted.append(worker.node_name)
                        affected_models.add(worker.model_id)

            # Execute hard deletions
            for key in keys_to_delete:
                del self._workers[key]

            for model_id in affected_models:
                self._recompute_pipes_locked(model_id)

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


_public_ip_cache: str | None = None


async def _get_public_ip() -> str | None:
    global _public_ip_cache
    if _public_ip_cache:
        return _public_ip_cache
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get("https://api.ipify.org")
            if resp.status_code == 200:
                _public_ip_cache = resp.text.strip()
                return _public_ip_cache
    except Exception:
        pass
    return None


@app.get("/p2p/bootstrap")
async def p2p_bootstrap() -> dict[str, object]:
    if transport is None:
        raise HTTPException(status_code=503, detail="Transport not initialized")
    if P2P_TRANSPORT != "p2p":
        raise HTTPException(status_code=400, detail="P2P transport is not enabled on this gateway")
    
    peer_id = await transport.get_address()
    visible_maddrs = []
    if hasattr(transport, "_p2p") and transport._p2p is not None:
        try:
            maddrs = await transport._p2p.get_visible_maddrs()
            visible_maddrs = [str(maddr) for maddr in maddrs]
        except Exception:
            pass
            
    # Add public host (explicit override or dynamic ipify lookup)
    public_host = os.getenv("KIWI_P2P_PUBLIC_HOST", "").strip()
    if public_host.startswith("https://"):
        public_host = public_host[len("https://"):]
    elif public_host.startswith("http://"):
        public_host = public_host[len("http://"):]
        
    public_port = P2P_PORT
    if ":" in public_host:
        parts = public_host.split(":")
        public_host = parts[0]
        try:
            public_port = int(parts[1])
        except ValueError:
            pass
            
    if not public_host:
        public_host = await _get_public_ip() or ""
        public_port = P2P_PORT
        
    if public_host:
        try:
            ip = ipaddress.ip_address(public_host)
            if ip.version == 6:
                public_maddr = f"/ip6/{public_host}/tcp/{public_port}/p2p/{peer_id}"
            else:
                public_maddr = f"/ip4/{public_host}/tcp/{public_port}/p2p/{peer_id}"
        except ValueError:
            public_maddr = f"/dns4/{public_host}/tcp/{public_port}/p2p/{peer_id}"
            
        if public_maddr not in visible_maddrs:
            visible_maddrs.append(public_maddr)
            
    return {
        "peer_id": peer_id,
        "p2p_port": P2P_PORT,
        "visible_maddrs": visible_maddrs,
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


@app.post("/register")
async def register_worker(registration: WorkerRegistration, request: Request) -> JSONResponse:
    if not model_registry:
        raise HTTPException(status_code=503, detail="Gateway not ready")
    worker = await registry.register(registration, model_registry)
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
async def heartbeat(request: HeartbeatRequest) -> dict[str, object]:
    result = await registry.heartbeat(request.node_name, request.latency_ms)
    if result is None:
        raise HTTPException(status_code=404, detail="Worker not registered")
    return result


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
            pipeline = await registry.get_pipe(
                model.model_id, model.num_layers, replica_id=0
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
        print(f"[gateway] Selected routing pipeline for {model.model_id}: {[w.node_name for w in pipeline]}", flush=True)

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

        if stream_ok and (not generated_ids or generated_ids[-1] not in eos_set) and len(generated_ids) < request.max_tokens:
            print(f"[gateway] Stream closed prematurely by {first_addr} before EOS/max_tokens, marking unhealthy and rerouting.", flush=True)
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


# ── Training dispatch endpoint ───────────────────────────────────────────────────


from fastapi import Response

@app.post("/train_step")
async def train_step(request: Request):
    if not model_registry:
        raise HTTPException(status_code=503, detail="Gateway not ready")
        
    model_id = request.headers.get("x-model-id") or request.headers.get("X-Model-ID", "")
    if not model_id:
        default_model = model_registry.default()
        if not default_model:
            raise HTTPException(status_code=400, detail="No model_id provided and no default available")
        model_id = default_model.model_id

    model = model_registry.resolve(model_id)
    if not model:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_id}")

    if transport is None:
        raise HTTPException(status_code=503, detail="Transport not initialized")

    request_id = request.headers.get("x-request-id") or request.headers.get("X-Request-ID", uuid.uuid4().hex)
    try:
        replica_id = int(request.headers.get("x-replica-id", "0"))
    except ValueError:
        raise HTTPException(status_code=400, detail="X-Replica-ID must be an integer")
    body = await request.body()
    reroutes = 0

    while True:
        step_start_time = time.time()

        try:
            pipeline = await registry.get_pipe(model.model_id, model.num_layers, replica_id)
        except ReplicaUnavailable as error:
            # R has dropped below this replica's index. The consumer should
            # return its batch to the queue and retire (§3.6).
            raise HTTPException(status_code=409, detail=str(error))
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error))

        first_addr = _worker_address(pipeline[0])
        remaining_urls = ",".join(_worker_address(w) for w in pipeline[1:])
        pipeline_names = [w.node_name for w in pipeline]
        print(f"[gateway] Train step dispatch replica={replica_id} (attempt {reroutes + 1}) via {pipeline_names}", flush=True)

        try:
            response_bytes = await transport.send_forward(
                target=first_addr,
                payload=body,
                headers={
                    "X-Request-ID": request_id,
                    "X-Model-ID": model.model_id,
                    "X-Pipeline": remaining_urls,
                    "X-Reverse-Pipeline": "",
                    "Content-Type": "application/octet-stream",
                },
                timeout=120.0
            )

            # Check if the response is a structured error (P2P transport wraps errors as JSON).
            failed_node = _parse_failed_node(response_bytes)
            if failed_node:
                death_detected_time = time.time()
                detection_latency_ms = (death_detected_time - step_start_time) * 1000
                print(f"[gateway] Live-error detected in train_step response! "
                      f"failed_node={failed_node}, Detection latency: {detection_latency_ms:.0f} ms", flush=True)
                await registry.evict(failed_node)
                reroutes += 1
                if reroutes > MAX_REROUTES:
                    raise HTTPException(status_code=500, detail=f"Train step retry budget exhausted after {MAX_REROUTES} reroutes")
                continue

            # Success path — log recovery metrics if this was a reroute.
            if reroutes > 0:
                resume_latency_ms = (time.time() - step_start_time) * 1000
                print(f"[gateway] Recovery complete. Resume latency: {resume_latency_ms:.0f} ms "
                      f"(after {reroutes} reroute(s))", flush=True)

            return Response(content=response_bytes, media_type="application/octet-stream")

        except HTTPException:
            raise
        except Exception as error:
            death_detected_time = time.time()
            detection_latency_ms = (death_detected_time - step_start_time) * 1000

            # Try to extract the failed_node from the structured error.
            failed_node = _parse_failed_node_from_error(error)
            evict_target = failed_node or first_addr

            print(f"[gateway] Live-error detected in train_step! "
                  f"evicting={evict_target}, Detection latency: {detection_latency_ms:.0f} ms, "
                  f"error={error}", flush=True)
            await registry.evict(evict_target)

            reroutes += 1
            if reroutes > MAX_REROUTES:
                raise HTTPException(
                    status_code=500,
                    detail=f"Train step retry budget exhausted after {MAX_REROUTES} reroutes: {error}"
                )
            # Loop continues: will re-pin pipeline (promoting standby) and re-dispatch


def _parse_failed_node(response_bytes: bytes) -> str | None:
    """Try to parse a failed_node from a response that may be an error JSON."""
    try:
        data = json.loads(response_bytes)
        if isinstance(data, dict) and "error" in data:
            return data.get("failed_node")
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
        pass
    return None


def _parse_failed_node_from_error(error: Exception) -> str | None:
    """Try to extract a failed_node from an exception's string representation."""
    err_str = str(error)
    try:
        parsed = json.loads(err_str)
        if isinstance(parsed, dict) and "failed_node" in parsed:
            return parsed["failed_node"]
    except (json.JSONDecodeError, TypeError):
        pass
    # Check for embedded JSON within a larger error message.
    start = err_str.find('{"failed_node"')
    if start >= 0:
        try:
            parsed = json.loads(err_str[start:])
            if isinstance(parsed, dict) and "failed_node" in parsed:
                return parsed["failed_node"]
        except (json.JSONDecodeError, TypeError):
            pass
    return None


# ── Averaging round endpoint (§3.4) ─────────────────────────────────────────────


T_AVERAGE = float(os.getenv("KIWI_T_AVERAGE", "60"))


@app.post("/average")
async def average_round(request: Request):
    """Trigger one averaging round across all layer-peer groups.

    The gateway is a barrier only: it fans out small JSON `/round` messages
    and collects completions. No adapter bytes touch the gateway — payloads
    stay peer-to-peer.

    Headers: X-Model-ID (optional), X-Avg-Interval (X, for the skew guard).
    """
    if not model_registry:
        raise HTTPException(status_code=503, detail="Gateway not ready")

    model_id = request.headers.get("x-model-id") or request.headers.get("X-Model-ID", "")
    if not model_id:
        default_model = model_registry.default()
        if not default_model:
            raise HTTPException(status_code=400, detail="No model_id provided and no default available")
        model_id = default_model.model_id
    model = model_registry.resolve(model_id)
    if not model:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_id}")

    try:
        x = int(request.headers.get("x-avg-interval", "10") or 10)
    except ValueError:
        raise HTTPException(status_code=400, detail="X-Avg-Interval must be an integer")

    try:
        round_id, plan = await registry.plan_round(model.model_id, x)
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error))

    print(
        f"[gateway] Round {round_id} for {model.model_id}: "
        f"{[(e['node_name'], e['round']['contribute']) for e in plan]}",
        flush=True,
    )

    async def _trigger(entry: dict) -> tuple[str, dict]:
        try:
            async with httpx.AsyncClient(timeout=T_AVERAGE) as client:
                response = await client.post(
                    f"{entry['node_url']}/round", json=entry["round"]
                )
                response.raise_for_status()
                return entry["node_name"], response.json()
        except Exception as error:
            return entry["node_name"], {"status": "error", "error": str(error)}

    results = await asyncio.gather(*[_trigger(e) for e in plan])

    completed = [name for name, r in results if r.get("status") == "ok"]
    skipped = {name: r for name, r in results if r.get("status") != "ok"}
    if skipped:
        # Best-effort by design: a partial round is strictly better than a
        # stalled cluster; the next round corrects it.
        print(f"[gateway] Round {round_id} partial — skipped: {skipped}", flush=True)

    return {
        "round_id": round_id,
        "completed": completed,
        "skipped": skipped,
        "results": {name: r for name, r in results},
    }


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
