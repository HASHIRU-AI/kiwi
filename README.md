# 🥝 K.I.W.I. — Knitted Infrastructure for Weight-sharded Instruction-tuning

**Kiwi is a proof of concept for fine-tuning a language model that is too large to fit on any single GPU you own — by splitting it across several small ones instead of shrinking it or spilling it to disk.**

A model is cut into contiguous slices of decoder layers, one slice per worker. Each worker attaches LoRA adapters to its own slice and trains only those. Activations flow forward through the chain, gradients flow back, and no single machine ever holds the whole model.

> **Status: research prototype.** Currently runs [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) across 3 workers capped at 0.9 GB VRAM each. The convergence claim is benchmarked (see below); nothing here is production-hardened.

---

## The problem it addresses

You want to fine-tune a model. It doesn't fit in your VRAM. You have three realistic options:

| Option | Approach | What it costs you |
|---|---|---|
| **Quantize** | QLoRA — squeeze the base to 4-bit | Base weights become lossy |
| **Offload** | DeepSpeed ZeRO-3 — keep the base in host RAM, stream it over PCIe | Roughly an order of magnitude in step time |
| **Distribute** | **Kiwi** — split the layers across several small machines | Network hops between stages |

The first two are what people actually reach for, and both are real trade-offs. Kiwi explores the third: if you have three 1 GB GPUs rather than one 3 GB GPU, you should be able to use them as one.

The obvious objection is that carving a model across machines might perturb training. **That is the thing this repo measures** — see [`benchmarks_training/`](benchmarks_training/README.md), which compares Kiwi against an unconstrained single-GPU reference plus both alternatives above over N=5 seeds.

---

## How it works

### Layer sharding

Workers boot with no assigned role. Each registers with the gateway (`POST /register`) reporting its VRAM, and the gateway hands back a contiguous layer range sized to what that worker can actually hold — `_pick_layers` derives per-layer footprint from the model config and reserves 512 MB for activations.

The worker then keeps only its slice and stubs out everything else:

- Not the first stage → `embed_tokens` becomes `Identity`
- Not the last stage → `norm` and `lm_head` become `Identity`
- `self_attn.layer_idx` is renumbered so the local slice starts at 0, since attention caching indexes by position rather than global layer number

### Per-worker LoRA and optimizer

Each worker wraps its slice with its own adapters — `r=16`, `alpha=32`, on `q/k/v/o/gate/up/down` — and constructs its own `torch.optim.Adam` over just those parameters. The base weights are frozen everywhere and never updated.

This is why per-worker optimizers are sound rather than a shortcut: each worker's adapter set is **completely disjoint** from every other's, and Adam's update is elementwise with no cross-parameter coupling. One global optimizer over all adapters and N local optimizers over disjoint slices compute the same thing.

### A training step

```
        client: safetensors {input_ids, labels}
                      │
                      ▼
          ┌───────────────────────────┐
          │   Gateway  /train_step    │  pinned pipeline, reroutes on death
          └─────────────┬─────────────┘
                        │
   ┌────────────────────┴───────────────────────────────────┐
   │                  forward  ──────────────▶              │
   │                                                        │
   │  Stage 0            Stage 1            Stage 2         │
   │  layers 0-k   ───▶  layers k+1-m  ───▶ layers m+1-L    │
   │  embeddings         blocks             blocks + head   │
   │                                        ↳ computes loss │
   │                                                        │
   │                  ◀──────────────  backward             │
   └────────────────────────────────────────────────────────┘
                        │
                        ▼
              loss (safetensors) → client
```

Hidden states move between stages as raw bf16 safetensors — binary, no JSON, no pickle, no quantization or compression on the wire. Transport is either HTTP (`httpx`) or direct libp2p streams (`kiwi/forward/1.0`), and either way the payloads travel **peer to peer**: the gateway sees the initial batch and the final loss, never the multi-megabyte intermediates.

### Two-phase optimizer commit

The part worth pointing at. A naive pipeline that calls `loss.backward()` and `optimizer.step()` together will half-apply an update if the peer it needs to talk to dies mid-step — leaving workers at different versions of the adapter with no way to tell.

Kiwi splits those:

1. `_run_backward_only()` computes gradients and stops
2. Gradients are handed to the peer
3. `_step_optimizer()` fires **only after** the peer confirms success

A failed send means no step happened anywhere, so the cluster stays consistent and the step can be retried on a rerouted pipeline.

### Fault tolerance

- **Pinned routes.** Training pins a pipeline per model (`get_pinned_pipeline`) rather than picking a fresh path each step. Optimizer state is per-worker, so route stability is a correctness property, not just an optimization — reshuffling stages mid-run would strand Adam moments.
- **Rerouting.** If a worker dies, `/train_step` evicts it, recomputes a pipeline from healthy workers covering `[0, num_layers)`, and retries the step.
- **Two-stage eviction.** Workers heartbeat every 5s. Soft eviction at 30s marks a worker unhealthy and stops routing to it; hard eviction at 120s removes it. Clean shutdowns `POST /deregister` and skip the wait.
- **Adapter sync.** `POST /adapter` pushes and merges adapter state between workers via `get_peft_model_state_dict` / `set_peft_model_state_dict`, so a replacement worker can be brought up to date.
- **Recovery accounting.** `_log_recovery_bytes_metrics()` costs out what a recovery actually transfers — adapter-only versus full baseline, at 10 bytes/param once Adam's `m` and `v` are counted.

---

## Running it

**Prerequisites:** Docker Desktop with Compose v2, an NVIDIA GPU with the container runtime, and ~4 GB disk for weights (downloaded once into a shared volume).

Bring up a gateway plus 3 memory-capped workers:

```bash
docker compose -f docker-compose.internet.yml up --build
```

Wait for the gateway health check and all 3 workers to register — the first run takes 60–90s while the model downloads. Then:

- **Registry / topology:** [http://localhost:8080/registry](http://localhost:8080/registry)
- **Health:** [http://localhost:8080/health](http://localhost:8080/health)

Training is driven by `POST /train_step` with a binary safetensors body containing `input_ids` and `labels`. Rather than hand-rolling that, use the benchmark harness — it handles dataset preparation, cluster startup and teardown, multi-seed runs and plotting:

### → **[Training benchmark guide (`benchmarks_training/README.md`)](benchmarks_training/README.md)**

That document covers what the four scenarios test and why, what's held constant between them, how the convergence verdict is decided, and the full run/plot commands. `DistributedRunner` in [`benchmarks_training/runners.py`](benchmarks_training/runners.py) is also the shortest readable example of driving `/train_step` directly.

### Other guides

- **[`P2P_TESTING.md`](P2P_TESTING.md)** — running and testing `hivemind.p2p` / libp2p transport locally or across machines
- **[`INTERNET_P2P_TESTING_GUIDE.md`](INTERNET_P2P_TESTING_GUIDE.md)** — multi-machine clusters over the public internet
- **[`LAMBDA_HYBRID_P2P_GUIDE.md`](LAMBDA_HYBRID_P2P_GUIDE.md)** — hybrid local/cloud worker setups
- **[`TESTING.md`](TESTING.md)** — layer coverage, multi-model assignment and fault tolerance verification

---

## Project structure

```
kiwi/
├── gateway.py                # Registry, VRAM-aware layer assignment, /train_step, rerouting
├── worker.py                 # Slices the model, attaches LoRA, forward/backward/adapter endpoints
├── transport.py              # HTTP and libp2p tensor forwarding
├── models.py                 # Model registry and per-layer footprint estimation
├── benchmarks_training/      # Convergence benchmark: Kiwi vs 3 baselines  ← start here
├── docker-compose.internet.yml   # 1 gateway + 3 workers @ 0.9 GB VRAM (used by the benchmark)
└── Dockerfile                # Gateway (CPU) and worker (CUDA) build stages
```

### `gateway.py`
Maintains the in-memory `WorkerRegistry` with VRAM-aware assignment and two-stage eviction. Resolves and pins a pipeline per model, dispatches training steps to stage 0, and reroutes around dead workers.

### `worker.py`
Registers, receives its layer range, slices and stubs the model, attaches LoRA, and builds a local Adam. Serves `/forward` (run slice, pass hidden states on), `/backward` (gradients, deferred optimizer commit) and `/adapter` (adapter state sync). Heartbeats every 5s and deregisters on shutdown.

---

## Key design decisions

| Decision | Rationale |
|---|---|
| Frozen base, LoRA only | The base never needs to be synced — only ~1.75% of parameters are in play |
| Disjoint per-worker optimizers | Mathematically equal to one global Adam, with no cross-worker coordination |
| Two-phase optimizer commit | A worker dying mid-step cannot half-apply an update |
| Pinned pipeline for training | Optimizer state is per-worker; stable routing keeps it meaningful |
| Safetensors over pickle | No arbitrary code execution during deserialization |
| P2P over hub-and-spoke | The gateway would bottleneck on multi-MB activations |
| Stage assignment at boot | One worker image, no per-replica configuration |
| Local `layer_idx` renumbering | Attention caching indexes by position, not global layer number |

---

## Tech stack

- **Runtime:** Python 3.10, FastAPI, Uvicorn
- **ML:** PyTorch, Hugging Face Transformers, PEFT (LoRA)
- **Transport:** httpx, hivemind/libp2p, safetensors
- **Infra:** Docker Compose, NVIDIA container runtime, shared HuggingFace cache volume

---

## Not covered here

The gateway still exposes inference-mode endpoints (`/v1/chat/completions`, `/v1/models`) and serves a chat UI from `index.html`, inherited from when Kiwi was an inference-only project. Those are intentionally undocumented here and will either be split into their own project or repurposed into a training dashboard.
