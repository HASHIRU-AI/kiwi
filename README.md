# 🥝 K.I.W.I.: Knitted Infrastructure for Weight-sharded Inference

Kiwi is a fault-tolerant, pipeline-parallel LLM gateway that unifies a fleet of machines into a single inference endpoint. Inspired by [Petals](https://github.com/bigscience-workshop/petals), it splits a transformer model across multiple workers and chains them via direct peer-to-peer tensor forwarding. The heavy intermediate activations never touch the central gateway.

> **Current Setup:** Multi-container deployment running models like [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) split across multiple workers, orchestrated by a central gateway with a web UI.

---

## Core Innovations

### VRAM-Aware Layer Assignment
Workers register with the gateway (`POST /register`), sending their URL and hardware capacity (`vram_gb`, `cpu_cores`). The gateway calculates per-layer memory footprint and assigns contiguous layer ranges:
1. **Gap-Filling:** Assigns newly registered workers to the longest contiguous uncovered layer range.
2. **Redundancy & Single-Node Assignment:** Once all layers are covered, assigns workers to the lowest-coverage range or allocates all layers (`0 to num_layers-1`) for single-node execution.

### P2P and HTTP Tensor Forwarding
Hidden states flow sequentially across assigned stages (`Stage 0 → Stage 1 → ... → Stage N`) via HTTP (`httpx`) or direct binary streams over libp2p (`kiwi/forward/1.0`). The gateway only sends initial `input_ids` and receives `next_token_id` responses, avoiding central bandwidth bottlenecks.

### Binary Safetensors & Compression
Inter-node tensor payloads use [safetensors](https://github.com/huggingface/safetensors) binary serialization (`application/octet-stream`). No JSON encoding or pickle deserialization. Workers and transport also support activation quantization (`KIWI_USE_INT8`), sparse CSR serialization (`KIWI_USE_CSR`), and `zstd` lossless compression (`KIWI_LOSSLESS=zstd`).

### Active Recompute & Rerouting
If a worker stops mid-generation, the gateway evicts the node, selects a new pipeline path from remaining healthy workers covering `[0, num_layers)`, and re-dispatches the full prompt context. Worker caches rebuild from scratch while the client receives a `[REROUTING...]` notification and generation resumes.

### Two-Stage Heartbeat Eviction
Workers ping `/heartbeat` every 5 seconds. The gateway runs a background loop (`_staleness_loop`) with two thresholds:
- **Soft Eviction (30s):** Marks the worker `healthy: False`, temporarily pausing new request routing to it.
- **Hard Eviction (120s):** Removes the worker from memory if it remains unreachable.

### Graceful Worker Drain
When a worker container shuts down, it sends `POST /deregister` so the gateway evicts it immediately without waiting for timeouts.

### PagedAttention & Cache Eviction
Workers manage KV cache memory in fixed-size blocks using `PagedCacheManager` (`KIWI_NUM_BLOCKS`, `KIWI_BLOCK_SIZE`). A background task removes request cache entries unaccessed for over 60 seconds (`KV_CACHE_TTL_SECONDS`).

### Multi-Turn Conversation History
The frontend stores conversation history and includes prior turns in successive generation requests.

### Network Topology Panel
The web UI includes a live topology visualization showing active nodes, assigned layer ranges, processing paths, and evicted nodes.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Gateway :8080                     │
│  ┌───────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │ Tokenizer │  │ Registry │  │ SSE Streaming API │  │
│  │  (Qwen)   │  │  (DHT)   │  │  /v1/chat/compl. │  │
│  └───────────┘  └──────────┘  └──────────────────┘  │
│         │              │               │             │
│         │     input_ids (safetensors)  │             │
│         ▼              ▼               ▼             │
└─────────────────────┬───────────────────────────────┘
                      │ tiny input_ids
                      ▼
          ┌───────────────────────┐
          │ Stage 0 (Layers 0-k)  │  ← e.g. worker_1
          │ Embeddings + Blocks   │
          └───────────┬───────────┘
                      │ hidden_states (HTTP / P2P safetensors)
                      ▼
          ┌───────────────────────┐
          │ Stage 1 (Layers k+1-m)│  ← e.g. worker_2
          │ Intermediate Blocks   │
          └───────────┬───────────┘
                      │ hidden_states (HTTP / P2P safetensors)
                      ▼
          ┌───────────────────────┐
          │ Stage N (Layers m+1-L)│  ← e.g. worker_4 (or SOLO 0-L)
          │ Blocks + LM Head      │
          └───────────┬───────────┘
                      │ next_token_id (JSON)
                      ▼
                  Gateway → SSE → Browser
```

---

## Benchmarking

| Metric | Baseline Model | Test Model (`unsloth/Llama-3.2-3B-Instruct`) | Delta / Improvement |
| :--- | :--- | :--- | :--- |
| **Avg Output Length (chars)** | 641.67 | 639.00 | -2.67 chars |
| **Avg Latency (sec)** | 51.60s | 9.30s | **~5.55x Faster** |
| **Avg Time to First Token (TTFT)** | 4.96s | 0.55s | **~9.02x Faster** |
| **Avg Time per Output Token (TPOT)** | 0.367s | 0.069s | **~5.32x Faster** |
| **Avg Tokens / Second** | 2.51 tok/s | 14.48 tok/s | **~5.77x Speedup** |

---

## 📝 Per-Prompt Benchmark Breakdown

### 1. "Write an essay on AI."

| Metric | Baseline | Test (`Llama-3.2-3B-Instruct`) |
| :--- | :--- | :--- |
| **Length (chars)** | 687 | 679 |
| **Latency (sec)** | 59.66s | **6.82s** |
| **TTFT (sec)** | 11.16s | **0.62s** |
| **TPOT (sec)** | 0.382s | **0.049s** |
| **Tokens / Sec** | 2.15 tok/s | **18.78 tok/s** |

---

### 2. "write a story about colors"

| Metric | Baseline | Test (`Llama-3.2-3B-Instruct`) |
| :--- | :--- | :--- |
| **Length (chars)** | 628 | 631 |
| **Latency (sec)** | 46.81s | **9.21s** |
| **TTFT (sec)** | 2.22s | **0.47s** |
| **TPOT (sec)** | 0.351s | **0.069s** |
| **Tokens / Sec** | 2.73 tok/s | **13.90 tok/s** |

---

### 3. "what is weather"

| Metric | Baseline | Test (`Llama-3.2-3B-Instruct`) |
| :--- | :--- | :--- |
| **Length (chars)** | 610 | 607 |
| **Latency (sec)** | 48.32s | **11.87s** |
| **TTFT (sec)** | 1.51s | **0.57s** |
| **TPOT (sec)** | 0.369s | **0.089s** |
| **Tokens / Sec** | 2.65 tok/s | **10.78 tok/s** |

---

## Quickstart

### Prerequisites
- Docker Desktop with Compose v2
- ~4 GB disk for the model weights (downloaded once into a shared volume)

### Run

```bash
docker compose up --build
```

Wait for the gateway health check to pass and all 4 workers to register (~60-90s for first run while model downloads). Then open:

- **Chat UI:** [http://localhost:8080](http://localhost:8080)
- **Live Registry:** [http://localhost:8080/registry](http://localhost:8080/registry)
- **Health Check:** [http://localhost:8080/health](http://localhost:8080/health)

### Test Fault Tolerance

1. Start a long generation in the chat UI (e.g., *"Write a 300-word essay about distributed systems"*)
2. While tokens are streaming, kill a worker:
   ```bash
   docker stop kiwi-worker_1-1
   ```
3. Watch the chat. You'll see `[REROUTING...]` in red, then generation continues seamlessly
4. Check the registry. The killed worker shows `"healthy": false`
5. Restart it:
   ```bash
   docker start kiwi-worker_1-1
   ```
6. The worker re-registers and returns to `"healthy": true`

### API Usage

Kiwi exposes an OpenAI-compatible chat completions endpoint:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kiwi-qwen2.5-0.5b-instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true,
    "max_tokens": 128
  }'
```

### P2P Networking & Testing Guides
- **[P2P Networking Guide (`P2P_TESTING.md`)](file:///c:/Users/harsh/OneDrive/Desktop/cs%20projects/kiwi/P2P_TESTING.md)**: Instructions for running and testing `hivemind.p2p` / libp2p networking across local setups (`docker-compose.internet.yml`) or multi-machine clusters.
- **[Benchmarking Guide (`benchmark_guide.md`)](file:///c:/Users/harsh/OneDrive/Desktop/cs%20projects/kiwi/benchmark_guide.md)**: Instructions for running comparative quality and latency benchmarks (`benchmarks/cli.py`).
- **[E2E Testing Guide (`TESTING.md`)](file:///c:/Users/harsh/OneDrive/Desktop/cs%20projects/kiwi/TESTING.md)**: Manual and automated verification instructions for multi-model assignment, layer coverage, and fault tolerance.

---


## Project Structure

```
kiwi/
├── gateway.py          # Central orchestrator, tokenizer, registry, SSE API
├── worker.py           # Generic worker — registers, slices model, P2P forwards
├── index.html          # Dark-mode chat UI (TailwindCSS)
├── Dockerfile          # Single image for all containers (python:3.10-slim)
├── docker-compose.yml  # 1 gateway + 4 worker replicas
└── requirements.txt    # FastAPI, PyTorch (CPU), Transformers, safetensors, httpx
```

### Gateway (`gateway.py`)
- Manages `AutoTokenizer` instances for registered models (`KIWI_MODELS`)
- Maintains an in-memory `WorkerRegistry` with VRAM-aware layer assignment (`_pick_layers`) and two-stage eviction
- Runs the generation loop: tokenization ➔ dispatch to Stage 0 ➔ sequential pipeline traversal ➔ receive token ➔ stream SSE
- Handles node failures via active recompute and dynamic rerouting (`select_pipeline`)

### Worker (`worker.py`)
- Boots agnostically, registers via `POST /register`, and receives assigned model and layer range (`[first_layer, last_layer]`)
- Pings `/heartbeat` every 5s with latency reporting and sends `POST /deregister` on shutdown
- Manages block-based KV cache (`PagedCacheManager`) and purges idle entries after 60s
- Supports `CSR` sparsity (`KIWI_USE_CSR`), `INT8` quantization (`KIWI_USE_INT8`), and `zstd` compression
- Intermediate stages: compute hidden states and forward to downstream peers (`kiwi/forward/1.0`) over HTTP or libp2p
- Final stage: computes logits and argmax, returning `next_token_id` (`kiwi/generate/1.0`)
- Stores per-request KV cache keyed by `X-Request-ID`

### Frontend (`index.html`)
- Dark-mode web interface with multi-turn conversation support
- SSE streaming with progressive token rendering
- Canvas-based network topology visualization

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Safetensors over pickle | Security. Prevents arbitrary code execution via deserialization |
| P2P over hub-and-spoke | Performance. Gateway doesn't bottleneck on multi-MB hidden states |
| Stage assignment at boot | Flexibility. Identical worker images, no config per replica |
| Prefill-based recompute | Simplicity. Stateless gateway, no cache migration protocol needed |
| KV cache keyed by request ID | Isolation. Concurrent users don't interfere with each other |
| `layer_idx` renumbering | Correctness. DynamicCache indexes by layer position, not global index |

---

## Tech Stack

- **Runtime:** Python 3.10, FastAPI, Uvicorn
- **ML:** PyTorch (CPU), Hugging Face Transformers
- **Transport:** httpx, safetensors binary serialization
- **Frontend:** Vanilla JS, TailwindCSS (CDN)
- **Infra:** Docker Compose, shared HuggingFace cache volume
