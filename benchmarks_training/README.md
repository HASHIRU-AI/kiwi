# Training Convergence Benchmark

This suite answers one question: **if you split a frozen-base LoRA fine-tune across several memory-constrained machines, does the model still learn the same way it would on one unconstrained GPU?**

It answers it by training the same model on the same data four different ways, over N=5 random seeds each, and comparing the loss-per-step curves.

---

## What it tests

Fine-tuning a model that does not fit in your GPU is a common problem with a few common answers. This benchmark runs the reference implementation, the two answers people usually reach for, and Kiwi — all on identical data, so the curves are directly comparable.

| Scenario | Approach | Per-device budget | Role |
|---|---|---|---|
| `single_machine` | Standard bf16 LoRA, one GPU | unconstrained | **Reference.** What correct learning looks like |
| `single_machine_qlora` | QLoRA, 4-bit base weights | 1024 MiB | Baseline: shrink the model |
| `single_machine_offload` | DeepSpeed ZeRO-3, base weights in host RAM | 640 MiB | Baseline: move the model off the GPU |
| `distributed_3stage` | Kiwi — layers split across 3 workers | 0.9 GB x3 | **The thing being evaluated** |

Model is Qwen2.5-0.5B-Instruct throughout. The dataset is four synthetic question/answer facts about a fictional kingdom, chosen so that memorisation is unambiguous and fast — this measures *convergence behaviour*, not task quality.

### What's held constant

Every scenario trains the same parameters on the same data in the same order:

- Qwen2.5-0.5B-Instruct, frozen base — no scenario ever updates a base weight
- LoRA `r=16`, `alpha=32`, `dropout=0.05` on `q/k/v/o/gate/up/down` — **8,798,208 trainable params (1.75%)**, bf16 adapters
- The same 4 facts, batch size 1, 10 epochs = **40 steps**
- The same learning rate (`KIWI_LR`)
- Seeds free and unforced, N=5 per scenario

### What actually differs

|  | `single_machine` | `single_machine_qlora` | `single_machine_offload` | `distributed_3stage` |
|---|---|---|---|---|
| **Base weight precision** | bf16 | **4-bit NF4** | bf16 | bf16 |
| **Where base weights live** | GPU | GPU | **Host RAM** | **Split across 3 GPUs** |
| **Optimizer** | `torch.optim.Adam`, GPU | `torch.optim.Adam`, GPU | **`DeepSpeedCPUAdam`, CPU** | per-worker `torch.optim.Adam` |
| **Gradient checkpointing** | no | **yes** | no | no |
| **Per-step data movement** | none | none | **whole model over PCIe, fwd + bwd** | **activations over network** |
| **Activation precision in transit** | — | — | — | bf16, uncompressed |
| **Per-device budget** | unconstrained | 1024 MiB | 640 MiB | 0.9 GB x3 |
| **Runs in** | process | process | Docker container | Docker cluster + gateway |

**`single_machine` vs `single_machine_offload` is the cleanest pair in the suite.** Same weights, same precision, same arithmetic. The only things that change are *where the bytes sit between uses* and which Adam implementation applies them — and the latter is deliberately neutralised (`adamw_mode=False, weight_decay=0.0` makes `DeepSpeedCPUAdam` identical to `torch.optim.Adam`). Those two loss curves should sit on top of each other; if they don't, the bug is in the harness, not in offloading. Their **step times** should differ by roughly an order of magnitude. That makes the pair a control: it isolates the cost of storage location with no numerical change attached.

The other two differ for different reasons:

- **`single_machine_qlora`** is the one expected to genuinely diverge in loss — its base weights are lossy. That divergence *is* the measurement.
- **`distributed_3stage`** differs *structurally*, not numerically. Activations cross a network boundary between workers and each worker runs its own optimizer over a disjoint slice of the adapters, but the values on the wire are exact — `worker.py` serialises hidden states with plain `safetensors_save` and applies no quantization or compression. So its curve should also land on the reference; the point of the benchmark is to confirm that splitting the model doesn't perturb learning, not to measure a known penalty.

### 1. `single_machine` — the reference

Standard PyTorch bf16 LoRA on a single unconstrained GPU. Nothing is quantized, offloaded, or split. This is the curve every other scenario is measured against: it defines both correct learning and maximum throughput.

Everything else in the suite exists because this configuration stops being possible once the model outgrows the card.

### 2. `single_machine_qlora` — shrink the model

The most common community answer: quantize the frozen base to 4-bit (bitsandbytes NF4) so it fits, and train LoRA adapters on top in bf16.

It works and it is fast, but the base weights are now lossy. Including it isolates **what quantization costs you in loss**, separately from anything to do with distribution.

### 3. `single_machine_offload` — move the model off the GPU

The answer when you need full-precision base weights and quantization is unacceptable: keep the base in host RAM and stream each layer across PCIe as the forward pass reaches it (DeepSpeed ZeRO-3, parameter *and* optimizer offload).

#### What DeepSpeed and ZeRO-3 actually do

[DeepSpeed](https://github.com/deepspeedai/DeepSpeed) is a training library from Microsoft whose core feature is **ZeRO** (Zero Redundancy Optimizer) — a set of techniques for not keeping multiple copies of things you only need one of.

Ordinary data-parallel training replicates *everything* on every GPU: each one holds a full copy of the weights, the gradients, and the optimizer state. ZeRO removes that duplication in three escalating stages:

| Stage | Partitions across ranks |
|---|---|
| ZeRO-1 | optimizer state |
| ZeRO-2 | + gradients |
| ZeRO-3 | + **the model parameters themselves** |

Under ZeRO-3 no GPU holds the whole model. Each rank owns a slice of every tensor, and just before a layer runs, the ranks all-gather that layer's parameters so the maths can happen — then immediately release them again. Memory is traded for communication.

**ZeRO-Offload** extends the idea past the GPU boundary: instead of parking a shard on a peer GPU, park it in host RAM (or NVMe) and pull it back over PCIe on demand. That is what makes it possible to train a model larger than your VRAM on a single card, and it is the configuration this baseline uses — the base weights and the Adam state both live in system memory, and each layer is fetched, used, and dropped as the forward and backward passes sweep through.

Note the consequence of running this at `world_size=1`: there are no other ranks, so the "partitioning" divides nothing. The entire memory saving comes from the CPU round-trip, and every tensor is still fetched whole. That is why the largest single tensor in the model ends up setting a hard floor on GPU memory — see the appendix.

Numerically this should match `single_machine` almost exactly — the weights are the same bf16 values, just stored somewhere else. What it costs is **wall-clock time**, and that is the point of including it. This baseline is primarily motivational: it establishes the throughput penalty that distribution is trying to avoid.

Measured on seed 1 (40 steps):

| Quantity | Value |
|---|---|
| Full model, bf16 + LoRA | 959.1 MiB |
| Resident in host RAM | 942.2 MiB across 169 frozen tensors |
| Peak GPU, steady-state training | **479.8 MiB — 50% of the model's resident footprint** |
| Step time | ~0.93 s |

So the model trains on a GPU that could not have held it. The cost is roughly an order of magnitude in step time versus the unconstrained reference.

### 4. `distributed_3stage` — Kiwi

Parameters are split across three memory-constrained workers (0.9 GB each), which exchange activations through the Gateway over P2P. No quantization, no CPU round-trips — each worker simply holds a third of the layers.

The claim under test:

> **Distributed frozen-base LoRA converges within the natural seed-variance of single-machine LoRA, so distributing the model across the cluster does not degrade learning.**

---

## How the verdict is decided

`ConvergenceEvaluator` compares **`single_machine` against `distributed_3stage` only**. The two constrained single-machine baselines are context for the memory and throughput story; they are not part of the pass/fail decision.

Two checks, both of which must pass:

1. **Final loss** — `|mean_distributed - mean_single| <= max(std_single, std_distributed)`
2. **Steps to target loss** — same band-overlap test applied to how many steps each took to reach `target_loss`

In other words: the gap between the two approaches must be smaller than the seed-to-seed noise within either approach. Seeds are left free and unforced rather than pinned, so the standard deviation being compared against is genuine run-to-run variance.

### Why steps, not wall-clock

The convergence chart plots loss against **training step**, never elapsed time. Network hops and PCIe latency would make a time-based comparison meaningless as a test of algorithmic correctness — a slower implementation is not a *wronger* one.

Wall-clock is still collected, just kept separate: `avg_step_time_ms` and `avg_peak_gpu_mb` go into the metrics output and the bar charts, where the throughput and memory trade-offs between offload, QLoRA and distribution are the actual subject.

### Optimizer equivalency

`SingleMachineRunner` uses one global PyTorch Adam over all LoRA adapters. The Kiwi cluster uses a separate Adam per worker. Because each worker's adapter parameters are completely disjoint, and Adam's update is elementwise with no cross-parameter coupling, the two are mathematically equivalent. This benchmark's results are what validate that assumption.

---

## How to run

**1. Install dependencies**

```bash
pip install matplotlib peft
```

The offload scenario additionally needs Docker with the NVIDIA container runtime. Its image is built automatically on first run and includes an ahead-of-time compile of DeepSpeed's CPUAdam op, so expect that first build to take a few minutes.

**2. Choose scenarios**

Edit `scenarios.yaml` to uncomment the ones you want. All four must be enabled for the convergence verdict to mean anything — with `single_machine` or `distributed_3stage` commented out, the evaluator compares against an empty list and reports `Passed: False` regardless.

**3. Set the learning rate explicitly**

```bash
$env:KIWI_LR = "2e-4"
```

This matters. `runners.py` defaults to `1e-5` while `docker-compose.internet.yml` and `run_offload.py` default to `2e-4`. If you leave it unset, scenarios silently train 20x apart and the curves are not comparable.

**4. Run the orchestrator**

```bash
.venv\Scripts\python.exe -m benchmarks_training.orchestrator
```

Each scenario runs `num_seeds` independent sessions. The distributed scenario spawns and tears down a fresh cluster per seed; the offload scenario builds its image once and then runs one container per seed.

Offload seeds run **detached**, with their logs followed separately. You still see live progress, but keystrokes in the terminal can no longer interrupt a run — under an attached `docker compose up`, Compose's interactive key handling could detach mid-seed and kill the container before it wrote its results. If you do interrupt the log stream, the container keeps running; the orchestrator picks its exit code back up via `docker wait`.

**5. Generate plots**

```bash
.venv\Scripts\python.exe -m benchmarks_training.plot
```

## Outputs

Everything lands in `benchmark_results/`:

| File | Contents |
|---|---|
| `training_convergence_results.json` | Raw per-seed curves, aggregated metrics, evaluator verdict |
| `convergence.png` / `.svg` | Loss vs step, mean curve with +/-1 std band per scenario |
| `convergence_bars.png` / `.svg` | Four panels: final loss, steps to target, peak GPU memory, step time |

Two caveats when reading the bar charts:

- **Peak GPU memory reads 0 for `distributed_3stage`.** `DistributedRunner` cannot see inside the worker containers, so it reports `0.0` rather than a real figure. Treat that bar as "not measured," not as "zero."
- **The per-device budgets are not identical across scenarios** (1024 MiB for QLoRA, 640 MiB for offload, 0.9 GB per Kiwi worker). The measured peak is what each approach actually used; the caps are just guards. Don't read the memory panel as a controlled comparison.

---

## Appendix: DeepSpeed offload configuration notes

The offload baseline needed non-default settings to behave. These are recorded because the defaults fail in ways that are quiet rather than loud.

**The memory constraint is simulated.** Qwen2.5-0.5B is 494.0M params (942 MiB bf16), or 502.8M / 959 MiB with LoRA attached — the ~2 GB figure often quoted for this model is fp32. `run_offload.py` caps the process at 640 MiB via `torch.cuda.set_per_process_memory_fraction`. A small model with a synthetic cap keeps 5 seeds x 4 scenarios tractable, and the ZeRO-3 mechanism under test is identical to what you would run at 7B+.

The cap does **not** cause offloading — `offload_param` / `offload_optimizer` do that unconditionally. The cap is a ceiling assertion proving the run stayed inside its budget. Setting it too low produces an OOM, not more aggressive offloading.

**`stage3_max_reuse_distance: 0` is required.** DeepSpeed's release logic walks forward through the module trace and refuses to release any parameter that reappears within `max_reuse_distance` numel:

```python
params_traversed = 0
for module in self.__submodule_order[step_id:]:
    if params_traversed >= self.__max_reuse_dist_in_numel:
        break
    for param in iter_params(module, ...):
        params_to_release.discard(param.ds_id)   # reused soon -> keep resident
        params_traversed += param.ds_numel
```

The default is 1e9 and the entire model is 494M params, so that loop never breaks. Since `tie_word_embeddings: true` makes `lm_head` the *same* Parameter as `embed_tokens`, the embedding reappears at the end of every trace and is therefore never released — pinning 260 MiB for the whole forward pass. Setting the distance to 0 forces the release; the embedding is simply re-fetched, which is exactly the PCIe traffic this baseline exists to measure.

Generally: on any model small enough to fit inside the default reuse distance, ZeRO-3 quietly degrades toward keeping tied and frequently-reused weights permanently resident.

**The floor is set by one tensor.** ZeRO-3 partitions per-tensor and can never hold less than the largest one. Here that is `embed_tokens` at 136.1M params / 260 MiB — 28% of the model in a single indivisible block. Loading and training gather it differently:

| Phase | Path | Cost |
|---|---|---|
| Weight load | `GatheredParameters` -> `_allgather_params_sequential` | **2x = 520 MiB** (destination buffer + device copy of the CPU shard) |
| Training | `all_gather_coalesced` -> `_no_gather_coalesced` | **1x = 260 MiB** (single `.to(device)`) |

Training takes a `world_size == 1` shortcut that the loader does not, which is why the *load* phase — not training — sets the minimum viable budget. `peak_load_mb` is recorded separately in the results JSON for this reason; `peak_gpu_mb` is measured after `reset_peak_memory_stats()` and reflects steady-state training only.

This is also a genuine limitation of single-GPU ZeRO-3 worth reporting rather than tuning around: it is bounded below by its largest tensor, so on a vocab-heavy small model it cannot show a dramatic saving at any bucket setting. Kiwi's layer-wise split has no such bound — the embedding lives entirely on one worker while decoder blocks distribute across the others, so no single device is ever sized by the embedding.

**Other settings.** `DeepSpeedCPUAdam` is used rather than `torch.optim.Adam`, since a plain torch optimizer alongside `offload_optimizer: "cpu"` drops DeepSpeed onto a much slower path; it runs with `adamw_mode=False, weight_decay=0.0` so it stays numerically identical to `SingleMachineRunner`'s Adam. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set but is **inert under WSL2**, which has no CUDA VMM support — on Docker Desktop for Windows it recovers no fragmentation headroom, so budget accordingly.

**Self-checks.** `run_offload.py` asserts that the frozen base tensors' ZeRO-3 partitions genuinely live in host RAM, and that peak GPU never reaches the full model footprint. A run where DeepSpeed silently kept everything resident fails loudly instead of quietly reporting misleading throughput.
