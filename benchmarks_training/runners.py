import gc
import queue as queue_mod
import random
import threading
import time
import httpx
import torch
import os
from typing import Any, Callable, Dict, List, Optional, Tuple
from safetensors.torch import save as safetensors_save, load as safetensors_load
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

SHARED_TRAIN_CONFIG = {
    "lora_config": LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    ),
    "lr": float(os.getenv("KIWI_LR", "1e-5"))
}

# ── Held-out eval (Phase 1.2) ───────────────────────────────────────────────
#
# Train loss on a memorization fixture measures memorization. The question
# this whole plan is asking — does averaging damage the model — is a
# GENERALIZATION question, so the primary curve has to be held-out loss.
#
# Cost control: an eval point costs `eval_samples` forward passes through the
# whole pipeline. Evaluating all 200 held-out examples every 25 train steps
# would spend more compute on measurement than on training. Instead every
# eval point uses the SAME fixed prefix of the held-out split, so points are
# directly comparable to each other and across scenarios; widen it with
# `eval_samples` when a tighter estimate matters more than wall clock.
DEFAULT_EVAL_SAMPLES = 64


def _eval_subset(eval_dataset: Optional[List[Dict[str, Any]]],
                 eval_samples: int) -> List[Dict[str, Any]]:
    """A FIXED prefix of the held-out split — never a random resample.

    Resampling per eval point would make the curve wobble for reasons that
    have nothing to do with training, and that wobble would be
    indistinguishable from the averaging effect being measured.
    """
    if not eval_dataset:
        return []
    if eval_samples <= 0:
        return list(eval_dataset)
    return list(eval_dataset[:eval_samples])


def _remote_eval_loss(
    gateway_url: str,
    model_id: str,
    eval_batches: List[Dict[str, Any]],
    tag: str,
    replica_id: Optional[int] = None,
    timeout: float = 180.0,
) -> Optional[float]:
    """Mean held-out loss measured THROUGH the live cluster (X-Eval: 1).

    Deliberately routed through /train_step rather than a side channel: an
    eval that loaded its own copy of the model would measure a different
    object than the one the mesh has been training, and any bug in the
    pipeline (bad layer split, stale adapter, failed merge) would be invisible
    exactly where it matters most. X-Eval: 1 makes the worker run forward +
    loss only — no backward, no optimizer step, no step-counter increment.
    """
    if not eval_batches:
        return None
    total, n = 0.0, 0
    for i, sample in enumerate(eval_batches):
        headers = {
            "Content-Type": "application/octet-stream",
            "X-Model-ID": model_id,
            "X-Request-ID": f"eval-{tag}-{i}",
            "X-Eval": "1",
        }
        if replica_id is not None:
            headers["X-Replica-ID"] = str(replica_id)
        try:
            resp = httpx.post(f"{gateway_url}/train_step", content=sample["payload"],
                              headers=headers, timeout=timeout)
            resp.raise_for_status()
            total += safetensors_load(resp.content)["loss"].item()
            n += 1
        except Exception as error:
            # An eval failure must never take down a training run — the eval
            # curve simply gets no point here.
            logging.warning(f"eval batch {i} failed ({tag}): {error}")
    return (total / n) if n else None


def _local_eval_loss(model, eval_batches: List[Dict[str, Any]]) -> Optional[float]:
    """Mean held-out loss for an in-process model.

    model.eval() matters: LoRA dropout (p=0.05) is active in train mode and
    would inflate held-out loss by a noisy constant.
    """
    if not eval_batches:
        return None
    was_training = model.training
    model.eval()
    total = 0.0
    try:
        with torch.no_grad():
            for sample in eval_batches:
                out = model(
                    input_ids=sample["input_ids"].to(model.device),
                    labels=sample["labels"].to(model.device),
                )
                total += out.loss.item()
    finally:
        if was_training:
            model.train()
    return total / len(eval_batches)


class BaseRunner:
    def run_training(self, model_id: str, dataset: List[Dict[str, Any]], epochs: int,
                     scenario_name: str = "unknown", seed_idx: int = 0,
                     eval_dataset: Optional[List[Dict[str, Any]]] = None,
                     eval_every: int = 0,
                     eval_samples: int = DEFAULT_EVAL_SAMPLES) -> Dict[str, Any]:
        """Runs training and returns a dict with 'curve', 'eval_curve' and 'metrics'."""
        raise NotImplementedError

class SingleMachineRunner(BaseRunner):
    def run_training(self, model_id: str, dataset: List[Dict[str, Any]], epochs: int,
                     scenario_name: str = "unknown", seed_idx: int = 0,
                     eval_dataset: Optional[List[Dict[str, Any]]] = None,
                     eval_every: int = 0,
                     eval_samples: int = DEFAULT_EVAL_SAMPLES) -> Dict[str, Any]:
        device = "cuda"
        dtype = torch.bfloat16
        logging.info(f"Using device: {device}")
        logging.info(f"Using dtype: {dtype}")
        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=dtype, 
            device_map=device
        )
        
        # Same gradient checkpointing as the workers, for the same reason it
        # has to be uniform: this is the reference arm every distributed
        # scenario is compared against, and an arm running a different compute
        # path is not a reference. It changes no gradient (checkpoint restores
        # the RNG state, so LoRA dropout recomputes identically) — only step
        # time, which gates nothing.
        if os.getenv("KIWI_GRAD_CHECKPOINT", "1") == "1":
            model.enable_input_require_grads()
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            model.config.use_cache = False
            logging.info("Gradient checkpointing enabled (matches worker config)")

        model = get_peft_model(model, SHARED_TRAIN_CONFIG["lora_config"])
        model.train()

        logging.info(f"Using learning rate: {SHARED_TRAIN_CONFIG['lr']}")
        optimizer = torch.optim.Adam(model.parameters(), lr=SHARED_TRAIN_CONFIG["lr"])

        torch.cuda.reset_peak_memory_stats()

        total_steps = epochs * len(dataset)
        curve = []
        eval_curve: List[Tuple[int, float]] = []
        eval_batches = _eval_subset(eval_dataset, eval_samples)
        step = 0
        total_time = 0.0

        # Point at samples=0, before any training. Without it there is no
        # anchor showing where the curve started, and "eval loss decreased"
        # cannot be distinguished from "eval loss was always this low".
        if eval_batches and eval_every > 0:
            initial = _local_eval_loss(model, eval_batches)
            if initial is not None:
                eval_curve.append((0, initial))
                logging.info(f"[{scenario_name} seed {seed_idx}] eval@0 | eval_loss {initial:.4f}")

        for epoch in range(epochs):
            for sample in dataset:
                input_ids = sample["input_ids"].to(model.device)
                labels = sample["labels"].to(model.device)

                start_t = time.time()
                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                step_t = time.time() - start_t
                total_time += step_t

                loss_val = loss.item()
                curve.append((step, loss_val))
                logging.info(f"[{scenario_name} seed {seed_idx}] step {step+1}/{total_steps} | loss {loss_val:.4f} | time {step_t:.3f}s")
                step += 1

                # Batch size is 1 here, so samples consumed == step.
                if eval_batches and eval_every > 0 and step % eval_every == 0:
                    eval_loss = _local_eval_loss(model, eval_batches)
                    if eval_loss is not None:
                        eval_curve.append((step, eval_loss))
                        logging.info(f"[{scenario_name} seed {seed_idx}] eval@{step} "
                                     f"| eval_loss {eval_loss:.4f}")

        # Always finish on an eval point, even if total_steps isn't a multiple
        # of eval_every — the evaluator compares FINAL loss, so a missing last
        # point would silently compare different amounts of training.
        if eval_batches and eval_every > 0 and (not eval_curve or eval_curve[-1][0] != step):
            final_eval = _local_eval_loss(model, eval_batches)
            if final_eval is not None:
                eval_curve.append((step, final_eval))
                logging.info(f"[{scenario_name} seed {seed_idx}] eval@{step} (final) "
                             f"| eval_loss {final_eval:.4f}")

        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        avg_time_ms = (total_time / total_steps) * 1000 if total_steps > 0 else 0

        # cleanup
        del model
        del optimizer
        gc.collect()
        torch.cuda.empty_cache()

        return {
            "curve": curve,
            "eval_curve": eval_curve,
            "metrics": {
                "peak_gpu_mb": float(peak_mem_mb),
                "step_time_ms": float(avg_time_ms)
            }
        }

class DistributedRunner(BaseRunner):
    def __init__(self, gateway_url: str = "http://localhost:8080",
                 max_attempts: int = 10):
        self.gateway_url = gateway_url
        # Matches MeshRunner.max_batch_attempts. Enough to ride out a pipe
        # reformation; not so many that a genuinely dead cluster hangs the run.
        self.max_attempts = max_attempts

    def run_training(self, model_id: str, dataset: List[Dict[str, Any]], epochs: int,
                     scenario_name: str = "unknown", seed_idx: int = 0,
                     eval_dataset: Optional[List[Dict[str, Any]]] = None,
                     eval_every: int = 0,
                     eval_samples: int = DEFAULT_EVAL_SAMPLES) -> Dict[str, Any]:
        events: List[Dict[str, Any]] = []

        def post(payload, rid):
            """At-least-once, mirroring MeshRunner's consumer.

            The gateway returns 503 while it has no healthy pipe — which
            happens transiently whenever a worker is evicted and the pipe is
            being reformed. Observed exactly this: a worker hit a CUDA OOM,
            the gateway evicted it, re-formed the identical pipe seconds
            later, and the run died anyway because this function raised on
            the one request in between. The cluster recovered; the benchmark
            did not.

            The worker-side dedup guard keys on X-Request-ID, so retrying the
            SAME rid cannot double-count a step that actually landed.
            """
            last_error: Optional[Exception] = None
            for attempt in range(self.max_attempts):
                try:
                    r = httpx.post(f"{self.gateway_url}/train_step", content=payload,
                                   headers={"Content-Type": "application/octet-stream",
                                            "X-Model-ID": model_id,
                                            "X-Request-ID": rid},
                                   timeout=120.0)
                    r.raise_for_status()
                    if attempt > 0:
                        events.append({"type": "recovered_step", "request_id": rid,
                                       "attempts": attempt + 1})
                        logging.info(f"[{scenario_name}] {rid} succeeded on attempt "
                                     f"{attempt + 1}")
                    return safetensors_load(r.content)
                except httpx.HTTPStatusError as error:
                    # 409 means this replica index no longer exists. At R=1
                    # that is unrecoverable and retrying only wastes the
                    # budget, so fail fast and loudly.
                    if error.response.status_code == 409:
                        raise
                    last_error = error
                except Exception as error:
                    last_error = error
                events.append({"type": "step_error", "request_id": rid,
                               "attempt": attempt + 1, "error": str(last_error)[:200]})
                logging.warning(f"[{scenario_name}] {rid} attempt {attempt + 1}/"
                                f"{self.max_attempts} failed: {last_error}")
                time.sleep(min(2.0 * (attempt + 1), 10.0))
            raise RuntimeError(
                f"{rid} failed after {self.max_attempts} attempts: {last_error}"
            )

        total_steps = epochs * len(dataset)
        curve = []
        eval_curve: List[Tuple[int, float]] = []
        eval_batches = _eval_subset(eval_dataset, eval_samples)
        step = 0
        total_time = 0.0

        def _eval_at(samples: int, note: str = "") -> None:
            eval_loss = _remote_eval_loss(
                self.gateway_url, model_id, eval_batches,
                tag=f"s{seed_idx}-n{samples}",
            )
            if eval_loss is not None:
                eval_curve.append((samples, eval_loss))
                logging.info(f"[{scenario_name} seed {seed_idx}] eval@{samples}{note} "
                             f"| eval_loss {eval_loss:.4f}")

        if eval_batches and eval_every > 0:
            _eval_at(0)

        for epoch in range(epochs):
            for sample in dataset:
                start_t = time.time()
                out = post(sample["payload"], f"step-{step}")
                step_t = time.time() - start_t
                total_time += step_t

                loss_val = out["loss"].item()
                curve.append((step, loss_val))
                logging.info(f"[{scenario_name} seed {seed_idx}] step {step+1}/{total_steps} | loss {loss_val:.4f} | time {step_t:.3f}s")
                step += 1

                if eval_batches and eval_every > 0 and step % eval_every == 0:
                    _eval_at(step)

        if eval_batches and eval_every > 0 and (not eval_curve or eval_curve[-1][0] != step):
            _eval_at(step, note=" (final)")

        avg_time_ms = (total_time / total_steps) * 1000 if total_steps > 0 else 0

        if events:
            logging.warning(
                f"[{scenario_name} seed {seed_idx}] completed with "
                f"{sum(1 for e in events if e['type'] == 'step_error')} failed request(s) "
                f"and {sum(1 for e in events if e['type'] == 'recovered_step')} recovery(ies) "
                f"— this seed was NOT a clean run; see events in the results JSON"
            )

        return {
            "curve": curve,
            "eval_curve": eval_curve,
            # Surfaced so a seed that only completed because of retries is
            # visible in the results file. A silent retry that rescues a run
            # also hides the fault that made it necessary.
            "events": events,
            "metrics": {
                "peak_gpu_mb": 0.0,
                "step_time_ms": float(avg_time_ms)
            }
        }


class MeshRunner(BaseRunner):
    """Async-mesh harness (ASYNC_MESH_DESIGN.md §3.6): one shared batch queue,
    R consumers posting /train_step with X-Replica-ID, and an averaging
    barrier every X steps per pipe.

    All mesh coordination lives HERE, not in orchestrator.py — that separation
    is a phase-7 prerequisite (extraction into a standalone client should be a
    move, not a rewrite).

    Implementation notes:
      - Work is processed in WINDOWS of R × X batches drawn from the epoch
        stream. Within a window consumers pull from a shared queue (free
        straggler handling); when the window drains and no step is in flight,
        the cluster is quiescent and the harness calls POST /average. This
        fires one round per X steps/pipe on average, which is the §3.6
        cadence without per-consumer step accounting.
      - At-least-once, not exactly-once: a failed batch goes back on the
        queue. A consumer that hits 409 (its replica_id >= R after a pipe
        loss) retires; survivors absorb the queue.
      - Request IDs are namespaced r{replica}-... so the worker-side dedup
        guard can't misfire across pipe recomputations (§3.6).
    """

    def __init__(
        self,
        gateway_url: str = "http://localhost:8080",
        replicas: Optional[int] = None,
        average_every: int = 2,
        worker_ports: Optional[List[int]] = None,
        shard_mode: str = "queue",
        sparse_p: float = 1.0,
        outer_opt: bool = False,
        full_sync_every: Optional[int] = None,
    ):
        self.gateway_url = gateway_url
        self.replicas = replicas          # None -> discover from /registry
        self.x = average_every
        self.worker_ports = worker_ports or []
        self.max_batch_attempts = 10      # give up on a batch (cluster is down)
        # Averaging mechanism (Phases 3 & 4), sent as headers on /average so a
        # single gateway image serves every arm of the A/B. p = 1.0 is the
        # dense path and is byte-identical to pre-Phase-3 behaviour.
        self.sparse_p = sparse_p
        self.outer_opt = outer_opt
        self.full_sync_every = full_sync_every
        # "queue"  — shared queue, consumers race (§3.6). Which facts a replica
        #            sees is re-randomised every window.
        # "static" — replica r permanently owns samples[r::R] (§6.7 static-shard
        #            mode). Each replica specialises on a CONSISTENT subset, so
        #            merges combine complementary experts instead of conflicting
        #            partial ones. This is also the honestly non-i.i.d. setup.
        self.shard_mode = shard_mode

    # ── Gateway helpers ──────────────────────────────────────────────────

    def _discover_r(self, model_id: str) -> int:
        resp = httpx.get(f"{self.gateway_url}/registry", timeout=10.0)
        resp.raise_for_status()
        workers = resp.json().get("workers", [])
        replica_ids = {
            w.get("replica_id")
            for w in workers
            if w.get("role") == "active" and w.get("healthy")
            and w.get("model_id") == model_id and w.get("replica_id") is not None
        }
        return len(replica_ids)

    def _average(self, model_id: str) -> Dict[str, Any]:
        headers = {
            "X-Model-ID": model_id,
            "X-Avg-Interval": str(self.x),
            "X-Sparse-P": repr(self.sparse_p),
            "X-Outer-Opt": "1" if self.outer_opt else "0",
        }
        if self.full_sync_every is not None:
            headers["X-Full-Sync-Every"] = str(self.full_sync_every)
        resp = httpx.post(
            f"{self.gateway_url}/average", headers=headers, timeout=180.0,
        )
        resp.raise_for_status()
        return resp.json()

    def _replica_spread(self, model_id: str) -> Optional[float]:
        """Max L2 distance between any two same-stage adapters (Phase 3.3).

        This is the number that says whether sparse averaging is doing what it
        claims: it should stay BOUNDED between full syncs and collapse to ~0
        at each one. An unbounded spread means the replicas are diverging and
        "agreement = checkpoint" no longer holds, which would invalidate spare
        promotion regardless of how good the loss curve looks.

        Pulled from the dense /adapter endpoint, which is unchanged by Phase 3.
        """
        if not self.worker_ports:
            return None
        by_stage: Dict[Any, List[Any]] = {}
        for port in self.worker_ports:
            try:
                health = httpx.get(f"http://localhost:{port}/health", timeout=5.0).json()
                if health.get("model_id") != model_id or health.get("status") != "ok":
                    continue
                stage = tuple(health.get("layers") or ())
                resp = httpx.get(f"http://localhost:{port}/adapter", timeout=30.0)
                resp.raise_for_status()
                sd = safetensors_load(resp.content)
                flat = torch.cat([v.detach().float().reshape(-1)
                                  for _k, v in sorted(sd.items())])
                by_stage.setdefault(stage, []).append(flat)
            except Exception:
                continue
        worst = 0.0
        seen = False
        for vectors in by_stage.values():
            for i in range(len(vectors)):
                for j in range(i + 1, len(vectors)):
                    if vectors[i].shape != vectors[j].shape:
                        continue
                    seen = True
                    worst = max(worst, float((vectors[i] - vectors[j]).norm()))
        return worst if seen else None

    def _peak_gpu_mb(self) -> float:
        peak = 0.0
        for port in self.worker_ports:
            try:
                r = httpx.get(f"http://localhost:{port}/health", timeout=5.0)
                peak = max(peak, float(r.json().get("gpu", {}).get("vram_peak_mb", 0)))
            except Exception:
                pass
        return peak

    # ── Training ─────────────────────────────────────────────────────────

    def run_training(
        self,
        model_id: str,
        dataset: List[Dict[str, Any]],
        epochs: int,
        scenario_name: str = "unknown",
        seed_idx: int = 0,
        kill_hook: Optional[Callable[[int], None]] = None,
        eval_dataset: Optional[List[Dict[str, Any]]] = None,
        eval_every: int = 0,
        eval_samples: int = DEFAULT_EVAL_SAMPLES,
    ) -> Dict[str, Any]:
        R = self.replicas or self._discover_r(model_id)
        if R < 1:
            raise RuntimeError("MeshRunner: no active replicas discovered")
        X = self.x
        rng = random.Random(1000 * seed_idx + 7)

        lock = threading.Lock()
        retired: set = set()
        samples_consumed = 0
        replica_curves: Dict[int, List[Tuple[int, float]]] = {r: [] for r in range(R)}
        combined_curve: List[Tuple[int, float]] = []
        step_times: List[float] = []
        events: List[Dict[str, Any]] = []
        round_times: List[float] = []
        round_profile: Dict[str, List[float]] = {}
        spread_curve: List[Dict[str, Any]] = []
        last_spread_at = [-(10 ** 9)]
        req_counter = [0]
        degraded_samples = [0]

        def post_step(payload: bytes, replica_id: int) -> httpx.Response:
            with lock:
                req_counter[0] += 1
                rid = f"seed{seed_idx}-r{replica_id}-n{req_counter[0]}"
            return httpx.post(
                f"{self.gateway_url}/train_step",
                content=payload,
                headers={
                    "Content-Type": "application/octet-stream",
                    "X-Model-ID": model_id,
                    "X-Request-ID": rid,
                    "X-Replica-ID": str(replica_id),
                },
                timeout=180.0,
            )

        def consumer(replica_id: int, work: "queue_mod.Queue",
                     own: Optional["queue_mod.Queue"] = None) -> None:
            nonlocal samples_consumed
            while True:
                batch = None
                if own is not None:                 # static-shard mode
                    try:
                        batch = own.get_nowait()
                    except queue_mod.Empty:
                        # Own slice done — absorb anything a retired replica
                        # handed back, so work is never stranded.
                        try:
                            batch = work.get_nowait()
                        except queue_mod.Empty:
                            return
                else:
                    try:
                        batch = work.get_nowait()
                    except queue_mod.Empty:
                        return
                start_t = time.time()
                attempts = batch.setdefault("_attempts", 0)
                try:
                    resp = post_step(batch["payload"], replica_id)
                    if resp.status_code == 409:
                        # R dropped below our index: return the batch, retire.
                        # Always to the SHARED queue so a surviving replica can
                        # pick it up even in static-shard mode.
                        work.put(batch)
                        with lock:
                            retired.add(replica_id)
                            events.append({"type": "consumer_retired",
                                           "replica": replica_id,
                                           "samples": samples_consumed})
                        logging.warning(f"[{scenario_name}] consumer {replica_id} retired (R dropped)")
                        return
                    resp.raise_for_status()
                    out = safetensors_load(resp.content)
                except Exception as e:
                    # At-least-once: back on the queue, someone retries it.
                    batch["_attempts"] = attempts + 1
                    if batch["_attempts"] >= self.max_batch_attempts:
                        with lock:
                            events.append({"type": "batch_abandoned",
                                           "error": str(e)})
                        logging.error(f"[{scenario_name}] batch abandoned after "
                                      f"{attempts + 1} attempts: {e}")
                        return
                    (own if own is not None else work).put(batch)
                    with lock:
                        events.append({"type": "step_error", "replica": replica_id,
                                       "samples": samples_consumed, "error": str(e)[:200]})
                    time.sleep(1.0)
                    continue

                step_t = time.time() - start_t
                loss_val = out["loss"].item()
                with lock:
                    samples_consumed += 1
                    n = samples_consumed
                    if retired:
                        # This sample was trained while the mesh was degraded
                        # below its nominal R — the seed is not a clean dp{R}
                        # observation and must be visible downstream.
                        degraded_samples[0] += 1
                    replica_curves[replica_id].append((n, loss_val))
                    combined_curve.append((n, loss_val))
                    step_times.append(step_t)
                    if attempts > 0:
                        # This step succeeded after failures — its wall time IS
                        # the observed recovery/resume latency.
                        events.append({"type": "recovered_step", "replica": replica_id,
                                       "samples": n, "resume_ms": step_t * 1000})

        # ── Held-out eval (Phase 1.2) ──
        # Measured at WINDOW BOUNDARIES, i.e. immediately after the averaging
        # barrier. That is the only fair place to evaluate a mesh: mid-window
        # the R replicas hold different weights and "the model's loss" is not
        # a well-defined quantity, so which replica you happened to ask would
        # change the answer. Post-merge they agree (exactly, on dense rounds).
        #
        # Only ONE replica evaluates, so a mesh doesn't spend R times the eval
        # compute of dp1 and inflate its own sample counter.
        eval_curve: List[Tuple[int, float]] = []
        eval_batches = _eval_subset(eval_dataset, eval_samples)
        last_eval_at = [-(10 ** 9)]

        def _eval_at(samples: int, note: str = "") -> None:
            # Prefer replica 0, but fall back through the others: in a kill
            # scenario replica 0's pipe can be the one that died, and an eval
            # curve that silently stops at the kill point is worse than useless
            # — that is exactly the region the recovery run exists to show.
            for r in range(R):
                eval_loss = _remote_eval_loss(
                    self.gateway_url, model_id, eval_batches,
                    tag=f"s{seed_idx}-n{samples}", replica_id=r,
                )
                if eval_loss is not None:
                    eval_curve.append((samples, eval_loss))
                    logging.info(f"[{scenario_name} seed {seed_idx}] eval@{samples}{note} "
                                 f"| eval_loss {eval_loss:.4f} (replica {r})")
                    return
            logging.warning(f"[{scenario_name} seed {seed_idx}] eval@{samples} produced "
                            f"no points on any of {R} replicas")

        if eval_batches and eval_every > 0:
            _eval_at(0)
            last_eval_at[0] = 0

        total_wall_start = time.time()
        total_steps_target = epochs * len(dataset)
        window_index = 0

        # ONE continuous stream across all epochs, reshuffled within each epoch.
        #
        # Windowing must span epoch boundaries. Rebuilding a 4-element batch
        # list per epoch caps every window at 4, so any X where R*X exceeds the
        # dataset size silently collapses back to "one round per epoch" — X=5
        # and X=20 both behaved exactly like X=2, making the sweep meaningless
        # while still printing plausible round counts.
        stream: List[Dict[str, Any]] = []
        for epoch in range(epochs):
            epoch_batches = [
                dict(payload=s["payload"], _idx=i, _epoch=epoch)
                for i, s in enumerate(dataset)
            ]
            rng.shuffle(epoch_batches)
            stream.extend(epoch_batches)

        # Windows of R×X batches -> one averaging round per X steps/pipe.
        # EXCEPT in mirror mode, where every replica trains on every batch
        # in the window, so a window is X batches and each replica still
        # takes X steps before the merge.
        window_size = max(1, X if self.shard_mode == "mirror" else R * X)
        epochs_fired = 0

        for w_start in range(0, len(stream), window_size):
            window = stream[w_start:w_start + window_size]
            work: "queue_mod.Queue" = queue_mod.Queue()
            per_replica: Dict[int, "queue_mod.Queue"] = {}
            for b in window:
                b.pop("_attempts", None)

            window_index += 1
            if self.shard_mode == "mirror":
                # Every replica trains on EVERY batch in the window. The
                # replicas therefore see IDENTICAL data between merges, so
                # averaging has nothing to reconcile. This is the control
                # that none of queue/static/rotate provided: all three hand
                # each replica a DISJOINT half, so they varied "which facts"
                # while holding "do the replicas share data" constant.
                #
                # If mirror recovers dp1-like convergence, the averaging
                # machinery is sound and the whole penalty is the cost of
                # merging models trained on different data (non-i.i.d.
                # local SGD). If mirror is ALSO bad, the merge is damaging
                # weights even when there is nothing to reconcile, and the
                # bug is in the averaging itself.
                #
                # Compare on STEPS PER REPLICA, not samples consumed: each
                # batch here is trained R times, once per replica.
                for r in range(R):
                    q: "queue_mod.Queue" = queue_mod.Queue()
                    for b in window:
                        q.put(dict(b))       # independent copy per replica
                    per_replica[r] = q
            elif self.shard_mode in ("static", "rotate"):
                # Deterministic assignment instead of a race. The shared
                # queue stays EMPTY and is used only for batches handed back
                # by a retired replica, so failure handling is unchanged.
                #
                #   static — replica r permanently owns samples[r::R]. True
                #            disjoint sharding: a replica NEVER sees the
                #            other replicas' facts, so the merged model can
                #            only acquire them through averaging. Maximally
                #            non-i.i.d.
                #   rotate — same fixed grouping, but the groups are handed
                #            to a different replica each window (offset by
                #            the window index). Every replica eventually
                #            sees every fact, while still specialising on a
                #            consistent subset between merges. This is what
                #            scratch/test_stage6_mesh.py does.
                offset = (window_index - 1) % R if self.shard_mode == "rotate" else 0
                for r in range(R):
                    q: "queue_mod.Queue" = queue_mod.Queue()
                    for b in window:
                        if (b["_idx"] + offset) % R == r:
                            q.put(b)
                    per_replica[r] = q
            else:
                for b in window:
                    work.put(b)

            # A consumer retires when its replica_id exceeds R after a pipe
            # loss. R can come BACK (a spare is promoted, or a falsely
            # soft-evicted worker heartbeats again), so re-check at every
            # window boundary and let retired consumers rejoin — otherwise
            # one transient blip silently degrades the whole run to R=1 and
            # the resulting curve is not a dp{R} curve at all.
            if retired:
                try:
                    live_r = self._discover_r(model_id)
                except Exception:
                    live_r = 0
                with lock:
                    rejoined = {r for r in retired if r < live_r}
                    if rejoined:
                        retired.difference_update(rejoined)
                        events.append({"type": "consumer_rejoined",
                                       "replicas": sorted(rejoined),
                                       "samples": samples_consumed})
                        logging.info(f"[{scenario_name}] consumers {sorted(rejoined)} "
                                     f"rejoined (R recovered to {live_r})")

            # Drain the window; loop again if a requeued batch was left
            # behind after all threads of a pass exited.
            for _pass in range(self.max_batch_attempts):
                with lock:
                    live = [r for r in range(R) if r not in retired]
                if not live:
                    raise RuntimeError(f"[{scenario_name}] all consumers retired; cluster unusable")

                # A retired replica's private queue has no reader — no consumer
                # ever looks at another replica's shard — so its batches would
                # strand and the window could never drain, killing the entire
                # run with "window failed to drain". Hand that work to the
                # shared queue, which every live consumer falls back to.
                for _r, _q in per_replica.items():
                    if _r in live:
                        continue
                    _moved = 0
                    while True:
                        try:
                            work.put(_q.get_nowait())
                            _moved += 1
                        except queue_mod.Empty:
                            break
                    if _moved:
                        with lock:
                            events.append({"type": "shard_reassigned",
                                           "from_replica": _r, "batches": _moved,
                                           "samples": samples_consumed})
                        logging.warning(
                            f"[{scenario_name}] replica {_r} retired holding "
                            f"{_moved} batches; reassigned to the shared queue"
                        )

                threads = [
                    threading.Thread(
                        target=consumer,
                        args=(r, work,
                              per_replica.get(r) if self.shard_mode != "queue" else None),
                        daemon=True,
                    )
                    for r in live
                ]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                drained = work.empty() and all(
                    q.empty() for q in per_replica.values())
                if drained:
                    break
            else:
                raise RuntimeError(f"[{scenario_name}] window failed to drain")

            # Cluster is quiescent (no /train_step in flight) — barrier.
            round_start = time.time()
            result = self._average(model_id)
            round_times.append(time.time() - round_start)

            # Phase 2.1: keep the barrier profile in the RESULTS FILE, not
            # only in container logs. "avg_round_ms < 500" is a gate, and a
            # gate you can only check by grepping stdout of a run that has
            # already been torn down is not a gate.
            gw = result.get("timings_ms") or {}
            for key in ("barrier_total", "plan_round"):
                if key in gw:
                    round_profile.setdefault(f"gateway_{key}_ms", []).append(float(gw[key]))
            for rtt in (gw.get("worker_rtt") or {}).values():
                round_profile.setdefault("worker_rtt_ms", []).append(float(rtt))
            for worker_result in (result.get("results") or {}).values():
                for key, value in (worker_result.get("timings_ms") or {}).items():
                    round_profile.setdefault(f"worker_{key}", []).append(float(value))

            with lock:
                n_now = samples_consumed
            logging.info(
                f"[{scenario_name} seed {seed_idx}] samples {n_now}/{total_steps_target} "
                f"| round {result['round_id']} completed={len(result['completed'])}"
            )

            # Replica spread, sampled at the same cadence as eval. Recorded
            # WITH the round's `full` flag so the plot can show the sawtooth:
            # spread grows across sparse rounds and collapses at each full
            # sync. Sampling only on full-sync rounds would show a flat zero
            # line and prove nothing.
            #
            # Must sit AFTER `n_now` is read under the lock — it is the sample
            # counter this is throttled on.
            if self.sparse_p < 1.0 and eval_every > 0 and (
                n_now - last_spread_at[0] >= eval_every
            ):
                last_spread_at[0] = n_now
                spread = self._replica_spread(model_id)
                if spread is not None:
                    spread_curve.append({
                        "samples": n_now,
                        "round_id": result.get("round_id"),
                        "spread_l2": spread,
                        "full_sync": bool(result.get("full")),
                    })

            # Post-merge eval, throttled by eval_every. A window is only R*X
            # samples wide (4 at R=2, X=2), so evaluating at EVERY boundary
            # would run ~250 evals per alpaca1k epoch and cost far more than
            # the training itself. Throttle on samples consumed, not on window
            # count, so the cadence is identical across scenarios with
            # different X — otherwise dp2_x20 and dp2_x1 would be measured at
            # completely different resolutions and their curves would not be
            # comparable.
            if eval_batches and eval_every > 0 and (n_now - last_eval_at[0]) >= eval_every:
                last_eval_at[0] = n_now
                _eval_at(n_now)

            # Fire the kill hook at epoch boundaries. Windows may now span
            # epochs, so derive completed epochs from the stream position
            # rather than a loop variable.
            if kill_hook is not None:
                completed = window[-1]["_epoch"] + 1
                while epochs_fired < completed:
                    epochs_fired += 1
                    kill_hook(epochs_fired)

        total_wall = time.time() - total_wall_start
        avg_step_ms = (sum(step_times) / len(step_times)) * 1000 if step_times else 0.0

        # Always finish on an eval point so final-loss comparisons are made at
        # the same amount of training in every scenario.
        if eval_batches and eval_every > 0 and (
            not eval_curve or eval_curve[-1][0] != samples_consumed
        ):
            _eval_at(samples_consumed, note=" (final)")

        return {
            "curve": sorted(combined_curve),          # [(samples_consumed, loss)]
            "eval_curve": eval_curve,                 # [(samples_consumed, eval_loss)]
            "replica_curves": {str(r): c for r, c in replica_curves.items()},
            "metrics": {
                "peak_gpu_mb": self._peak_gpu_mb(),
                "step_time_ms": float(avg_step_ms),
                "samples_per_sec": (samples_consumed / total_wall) if total_wall > 0 else 0.0,
                "rounds": len(round_times),
                "avg_round_ms": (sum(round_times) / len(round_times)) * 1000 if round_times else 0.0,
                "replicas": R,
                "average_every": X,
                "shard_mode": self.shard_mode,
                "sparse_p": self.sparse_p,
                "outer_opt": self.outer_opt,
                # Fraction of this seed trained below nominal R. Anything
                # above 0 means the curve is a blend of dp{R} and dp{<R} and
                # should not be pooled with clean seeds without saying so.
                "degraded_sample_fraction": (degraded_samples[0] / samples_consumed)
                                            if samples_consumed else 0.0,
            },
            # Mean of every barrier sub-step across all rounds and all
            # workers. This is the artifact that says WHERE the round time
            # went, so a Gate 2 failure points at a substep instead of at
            # "the barrier".
            "round_profile_ms": {
                k: (sum(v) / len(v)) for k, v in sorted(round_profile.items()) if v
            },
            "spread_curve": spread_curve,
            "events": events,
        }

# class OffloadRunner(BaseRunner):
#     def run_training(self, model_id: str, dataset: List[Dict[str, Any]], epochs: int, scenario_name: str = "unknown", seed_idx: int = 0) -> Dict[str, Any]:
#         dtype = torch.bfloat16
#         logging.info(f"Using dtype: {dtype} with Offload (512MiB GPU limit, remainder to CPU)")
#         
#         model = AutoModelForCausalLM.from_pretrained(
#             model_id, 
#             torch_dtype=dtype, 
#             device_map="auto",
#             max_memory={0: "512MiB", "cpu": "16GiB"}
#         )
#         
#         logging.info(f"Model hf_device_map: {model.hf_device_map}")
#         gpu_layers = [k for k, v in model.hf_device_map.items() if v == 0]
#         cpu_layers = [k for k, v in model.hf_device_map.items() if v == "cpu"]
#         logging.info(f"  GPU layers ({len(gpu_layers)}): {gpu_layers[:5]}{'...' if len(gpu_layers) > 5 else ''}")
#         logging.info(f"  CPU layers ({len(cpu_layers)}): {cpu_layers[:5]}{'...' if len(cpu_layers) > 5 else ''}")
#         assert len(cpu_layers) > 0, (
#             f"OffloadRunner: no layers were offloaded to CPU — "
#             f"the model fits entirely in {512}MiB. Tighten max_memory further."
#         )
#         
#         # Required when using device_map="auto" with PEFT — without this,
#         # accelerate's meta-device tensors for CPU-offloaded layers cause a
#         # device mismatch error ("expected device meta but got cuda:0") during
#         # the backward pass through LoRA adapters.
#         model.enable_input_require_grads()
#         model = get_peft_model(model, SHARED_TRAIN_CONFIG["lora_config"])
#         model.train()
#         
#         logging.info(f"Using learning rate: {SHARED_TRAIN_CONFIG['lr']}")
#         optimizer = torch.optim.Adam(model.parameters(), lr=SHARED_TRAIN_CONFIG["lr"])
#         
#         if len(dataset) > 0:
#             sample = dataset[0]
#             first_device = next(model.parameters()).device
#             input_ids = sample["input_ids"].to(first_device)
#             labels = sample["labels"].to(first_device)
#             
#             optimizer.zero_grad()
#             outputs = model(input_ids=input_ids, labels=labels)
#             loss = outputs.loss
#             loss.backward()
#             
#             has_grads = any(p.grad is not None for p in model.parameters())
#             assert has_grads, "Dummy step failed: all gradients are None (possible device_map bug)."
#             optimizer.zero_grad()
#             logging.info("Dummy step passed, gradients are flowing.")
#             
#         torch.cuda.reset_peak_memory_stats()
#         
#         total_steps = epochs * len(dataset)
#         curve = []
#         step = 0
#         total_time = 0.0
#         
#         for epoch in range(epochs):
#             for sample in dataset:
#                 first_device = next(model.parameters()).device
#                 input_ids = sample["input_ids"].to(first_device)
#                 labels = sample["labels"].to(first_device)
#                 
#                 start_t = time.time()
#                 optimizer.zero_grad()
#                 outputs = model(input_ids=input_ids, labels=labels)
#                 loss = outputs.loss
#                 loss.backward()
#                 optimizer.step()
#                 step_t = time.time() - start_t
#                 total_time += step_t
#                 
#                 loss_val = loss.item()
#                 curve.append((step, loss_val))
#                 logging.info(f"[{scenario_name} seed {seed_idx}] step {step+1}/{total_steps} | loss {loss_val:.4f} | time {step_t:.3f}s")
#                 step += 1
#                 
#         peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
#         avg_time_ms = (total_time / total_steps) * 1000 if total_steps > 0 else 0
#         
#         # cleanup
#         del model
#         del optimizer
#         gc.collect()
#         torch.cuda.empty_cache()
#         
#         return {
#             "curve": curve,
#             "metrics": {
#                 "peak_gpu_mb": float(peak_mem_mb),
#                 "step_time_ms": float(avg_time_ms)
#             }
#         }

_qlora_fraction_set = False

class QLoRARunner(BaseRunner):
    def run_training(self, model_id: str, dataset: List[Dict[str, Any]], epochs: int,
                     scenario_name: str = "unknown", seed_idx: int = 0,
                     eval_dataset: Optional[List[Dict[str, Any]]] = None,
                     eval_every: int = 0,
                     eval_samples: int = DEFAULT_EVAL_SAMPLES) -> Dict[str, Any]:
        global _qlora_fraction_set
        
        try:
            import bitsandbytes as bnb
            from transformers import BitsAndBytesConfig
            from peft import prepare_model_for_kbit_training
        except ImportError:
            logging.error("bitsandbytes not installed or not working. Please install bitsandbytes to run QLoRARunner.")
            raise
        
        # Ensure previous run's memory is fully released before loading
        gc.collect()
        torch.cuda.empty_cache()
            
        total_vram = torch.cuda.get_device_properties(0).total_memory
        target_bytes = 1024 * 1024 * 1024 # 1GB
        fraction = min(1.0, target_bytes / total_vram)
        
        # set_per_process_memory_fraction is process-global and sticky;
        # only call it once to avoid errors on subsequent seeds.
        if not _qlora_fraction_set:
            torch.cuda.set_per_process_memory_fraction(fraction, 0)
            _qlora_fraction_set = True
        logging.info(f"QLoRA memory fraction {fraction:.4f} (~1GB of {total_vram/1024/1024/1024:.1f}GB total VRAM)")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="cuda"
        )
        
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model = get_peft_model(model, SHARED_TRAIN_CONFIG["lora_config"])
        model.train()
        
        logging.info(f"Using learning rate: {SHARED_TRAIN_CONFIG['lr']}")
        optimizer = torch.optim.Adam(model.parameters(), lr=SHARED_TRAIN_CONFIG["lr"])
        
        if len(dataset) > 0:
            sample = dataset[0]
            input_ids = sample["input_ids"].to(model.device)
            labels = sample["labels"].to(model.device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            
            assert not torch.isnan(loss), "Dummy step failed: loss is NaN."
            optimizer.zero_grad()
            logging.info("Dummy step passed, loss is valid.")
            
        torch.cuda.reset_peak_memory_stats()
        
        total_steps = epochs * len(dataset)
        curve = []
        eval_curve: List[Tuple[int, float]] = []
        eval_batches = _eval_subset(eval_dataset, eval_samples)
        step = 0
        total_time = 0.0

        if eval_batches and eval_every > 0:
            initial = _local_eval_loss(model, eval_batches)
            if initial is not None:
                eval_curve.append((0, initial))

        for epoch in range(epochs):
            for sample in dataset:
                input_ids = sample["input_ids"].to(model.device)
                labels = sample["labels"].to(model.device)

                start_t = time.time()
                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                step_t = time.time() - start_t
                total_time += step_t

                loss_val = loss.item()
                curve.append((step, loss_val))
                logging.info(f"[{scenario_name} seed {seed_idx}] step {step+1}/{total_steps} | loss {loss_val:.4f} | time {step_t:.3f}s")
                step += 1

                if eval_batches and eval_every > 0 and step % eval_every == 0:
                    eval_loss = _local_eval_loss(model, eval_batches)
                    if eval_loss is not None:
                        eval_curve.append((step, eval_loss))
                        logging.info(f"[{scenario_name} seed {seed_idx}] eval@{step} "
                                     f"| eval_loss {eval_loss:.4f}")

        if eval_batches and eval_every > 0 and (not eval_curve or eval_curve[-1][0] != step):
            final_eval = _local_eval_loss(model, eval_batches)
            if final_eval is not None:
                eval_curve.append((step, final_eval))

        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        avg_time_ms = (total_time / total_steps) * 1000 if total_steps > 0 else 0

        # cleanup
        del model
        del optimizer
        gc.collect()
        torch.cuda.empty_cache()

        return {
            "curve": curve,
            "eval_curve": eval_curve,
            "metrics": {
                "peak_gpu_mb": float(peak_mem_mb),
                "step_time_ms": float(avg_time_ms)
            }
        }
