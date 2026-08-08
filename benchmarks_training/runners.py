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

class BaseRunner:
    def run_training(self, model_id: str, dataset: List[Dict[str, Any]], epochs: int, scenario_name: str = "unknown", seed_idx: int = 0) -> Dict[str, Any]:
        """Runs training and returns a dict with 'curve' and 'metrics'."""
        raise NotImplementedError

class SingleMachineRunner(BaseRunner):
    def run_training(self, model_id: str, dataset: List[Dict[str, Any]], epochs: int, scenario_name: str = "unknown", seed_idx: int = 0) -> Dict[str, Any]:
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
        
        model = get_peft_model(model, SHARED_TRAIN_CONFIG["lora_config"])
        model.train()
        
        logging.info(f"Using learning rate: {SHARED_TRAIN_CONFIG['lr']}")
        optimizer = torch.optim.Adam(model.parameters(), lr=SHARED_TRAIN_CONFIG["lr"])
        
        torch.cuda.reset_peak_memory_stats()
        
        total_steps = epochs * len(dataset)
        curve = []
        step = 0
        total_time = 0.0
        
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
                
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        avg_time_ms = (total_time / total_steps) * 1000 if total_steps > 0 else 0
        
        # cleanup
        del model
        del optimizer
        gc.collect()
        torch.cuda.empty_cache()
        
        return {
            "curve": curve,
            "metrics": {
                "peak_gpu_mb": float(peak_mem_mb),
                "step_time_ms": float(avg_time_ms)
            }
        }

class DistributedRunner(BaseRunner):
    def __init__(self, gateway_url: str = "http://localhost:8080"):
        self.gateway_url = gateway_url

    def run_training(self, model_id: str, dataset: List[Dict[str, Any]], epochs: int, scenario_name: str = "unknown", seed_idx: int = 0) -> Dict[str, Any]:
        def post(payload, rid):
            r = httpx.post(f"{self.gateway_url}/train_step", content=payload,
                           headers={"Content-Type": "application/octet-stream",
                                    "X-Model-ID": model_id,
                                    "X-Request-ID": rid},
                           timeout=120.0)
            r.raise_for_status()
            return safetensors_load(r.content)

        total_steps = epochs * len(dataset)
        curve = []
        step = 0
        total_time = 0.0
        
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
                
        avg_time_ms = (total_time / total_steps) * 1000 if total_steps > 0 else 0
        
        return {
            "curve": curve,
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
    ):
        self.gateway_url = gateway_url
        self.replicas = replicas          # None -> discover from /registry
        self.x = average_every
        self.worker_ports = worker_ports or []
        self.max_batch_attempts = 10      # give up on a batch (cluster is down)

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
        resp = httpx.post(
            f"{self.gateway_url}/average",
            headers={"X-Model-ID": model_id, "X-Avg-Interval": str(self.x)},
            timeout=180.0,
        )
        resp.raise_for_status()
        return resp.json()

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

        def consumer(replica_id: int, work: "queue_mod.Queue") -> None:
            nonlocal samples_consumed
            while True:
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
                    work.put(batch)
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

        total_wall_start = time.time()
        total_steps_target = epochs * len(dataset)

        for epoch in range(epochs):
            batches = [dict(payload=s["payload"]) for s in dataset]
            rng.shuffle(batches)

            # Windows of R×X batches -> one averaging round per X steps/pipe.
            window_size = max(1, R * X)
            for w_start in range(0, len(batches), window_size):
                window = batches[w_start:w_start + window_size]
                work: "queue_mod.Queue" = queue_mod.Queue()
                for b in window:
                    b.pop("_attempts", None)
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
                    threads = [threading.Thread(target=consumer, args=(r, work), daemon=True)
                               for r in live]
                    for t in threads:
                        t.start()
                    for t in threads:
                        t.join()
                    if work.empty():
                        break
                else:
                    raise RuntimeError(f"[{scenario_name}] window failed to drain")

                # Cluster is quiescent (no /train_step in flight) — barrier.
                round_start = time.time()
                result = self._average(model_id)
                round_times.append(time.time() - round_start)
                with lock:
                    n_now = samples_consumed
                logging.info(
                    f"[{scenario_name} seed {seed_idx}] samples {n_now}/{total_steps_target} "
                    f"| round {result['round_id']} completed={len(result['completed'])}"
                )

            if kill_hook is not None:
                kill_hook(epoch + 1)

        total_wall = time.time() - total_wall_start
        avg_step_ms = (sum(step_times) / len(step_times)) * 1000 if step_times else 0.0

        return {
            "curve": sorted(combined_curve),          # [(samples_consumed, loss)]
            "replica_curves": {str(r): c for r, c in replica_curves.items()},
            "metrics": {
                "peak_gpu_mb": self._peak_gpu_mb(),
                "step_time_ms": float(avg_step_ms),
                "samples_per_sec": (samples_consumed / total_wall) if total_wall > 0 else 0.0,
                "rounds": len(round_times),
                "avg_round_ms": (sum(round_times) / len(round_times)) * 1000 if round_times else 0.0,
                "replicas": R,
                "average_every": X,
                # Fraction of this seed trained below nominal R. Anything
                # above 0 means the curve is a blend of dp{R} and dp{<R} and
                # should not be pooled with clean seeds without saying so.
                "degraded_sample_fraction": (degraded_samples[0] / samples_consumed)
                                            if samples_consumed else 0.0,
            },
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
    def run_training(self, model_id: str, dataset: List[Dict[str, Any]], epochs: int, scenario_name: str = "unknown", seed_idx: int = 0) -> Dict[str, Any]:
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
        step = 0
        total_time = 0.0
        
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
                
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        avg_time_ms = (total_time / total_steps) * 1000 if total_steps > 0 else 0
        
        # cleanup
        del model
        del optimizer
        gc.collect()
        torch.cuda.empty_cache()
        
        return {
            "curve": curve,
            "metrics": {
                "peak_gpu_mb": float(peak_mem_mb),
                "step_time_ms": float(avg_time_ms)
            }
        }
