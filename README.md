# 🚀 K.I.W.I. — Knitted Infrastructure for Weight-sharded Inference

Kiwi is a fault-tolerant, pipeline-parallel LLM gateway that unifies a fleet of machines into a single inference endpoint. Inspired by [Petals](https://github.com/bigscience-workshop/petals), it splits a transformer model across multiple workers and chains them via direct peer-to-peer tensor forwarding. The heavy intermediate activations never touch the central gateway.

> **Current PoC:** 5 Docker containers running [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) split across 4 workers (2 × Stage A, 2 × Stage B), orchestrated by a single gateway with a built-in chat UI.

---

## Core Innovations

### P2P Tensor Forwarding
Hidden state tensors (the bulk of inter-node data) flow **directly from Stage A → Stage B** over HTTP. The gateway only sends tiny `input_ids` and receives tiny `next_token_id` responses. This avoids the gateway becoming a bandwidth bottleneck.

### Binary Safetensors Transport
All inter-node tensor payloads use [safetensors](https://github.com/huggingface/safetensors) binary serialization (`application/octet-stream`). No JSON encoding of tensors, no `torch.save`/pickle (avoiding [CVE-2025-32434](https://nvd.nist.gov/vuln/detail/CVE-2025-32434)).

### KV-Cache Active Recompute
When a worker dies mid-generation, the gateway selects a new healthy pair and sends the **full context** (original prompt + all tokens generated so far) as a fresh prefill step. This hot-rebuilds both workers' KV caches from scratch, and generation resumes seamlessly. The user sees a brief `[REROUTING...]` indicator and the stream continues.

### Dynamic Block Assignment
Workers boot completely agnostic. On startup, each worker sends its URL and hardware telemetry to the gateway's `/register` endpoint. The gateway's scheduler counts active nodes per stage and assigns the worker to whichever stage has fewer replicas, returning the layer range to load. The worker then slices the model accordingly.

### Heartbeat & Eviction Loop
Workers ping the gateway `/heartbeat` every 5 seconds to announce health status. The gateway maintains a background loop that evaluates all nodes, automatically evicting any worker that stops sending heartbeats for over 15 seconds.

### Graceful Worker Drain
When a worker container is stopped or restarted, it invokes `/deregister` during the shutdown lifecycle. This allows the gateway to evict the worker immediately, bypassing the staleness timeout.

### KV-Cache TTL Eviction
To avoid Memory leaks and Out-of-Memory (OOM) situations on workers, a background task automatically purges request cache entries that have not been accessed for more than 60 seconds.

### Multi-Turn Conversation History
The frontend stores conversation history and includes full turns in successive generation requests, allowing the model to leverage prior context.

### Live Topology Visualization & Active Highlighting
The chat UI includes an interactive network topology panel. During text generation, the canvas highlights the exact Stage A and Stage B worker pair currently processing the request in neon green, draws active data stream particles along the edge, and dims (reduces opacity of) all inactive worker nodes in the swarm. Evicted nodes are colored red.

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
          │  Stage A (Layers 0-11)│  ← worker_1 or worker_2
          │  Embeddings + Blocks  │
          └───────────┬───────────┘
                      │ hidden_states (P2P, safetensors)
                      ▼
          ┌───────────────────────┐
          │  Stage B (Layers 12-23)│  ← worker_3 or worker_4
          │  Blocks + LM Head     │
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
- **[P2P Networking Guide (`P2P_TESTING.md`)](file:///c:/Users/harsh/OneDrive/Desktop/cs%20projects/kiwi/P2P_TESTING.md)**: Instructions for running and testing `hivemind.p2p` / libp2p networking across single-machine setups (`docker-compose.internet.yml`) or multi-machine clusters over the internet without `ngrok`.
- **[E2E Testing Guide (`TESTING.md`)](file:///c:/Users/harsh/OneDrive/Desktop/cs%20projects/kiwi/TESTING.md)**: Comprehensive manual and automated verification instructions for multi-model assignment, layer coverage, and fault tolerance.

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
- Owns the `AutoTokenizer`
- Maintains an in-memory `WorkerRegistry` with stage balancing and `last_seen` heartbeat metrics
- Runs the generation loop: tokenize ➔ dispatch to Stage A ➔ receive token ➔ stream SSE (including active node mapping)
- Implements active recompute on failure: evict dead node ➔ select new pair ➔ re-prefill with full context

### Worker (`worker.py`)
- Boots agnostically, registers via `POST /register`, receives stage + layer assignment
- Runs a background loop pinging `/heartbeat` to gateway every 5s
- Sends `POST /deregister` on shutdown for graceful eviction
- Runs a background TTL cache manager to evict idle KV cache items older than 60s
- Stage A: computes hidden states, P2P forwards to Stage B via `httpx`
- Stage B: computes logits + argmax, returns `next_token_id`
- Maintains per-request KV cache keyed by `X-Request-ID`

### Frontend (`index.html`)
- Dark-mode TailwindCSS chat interface supporting multi-turn conversation memory
- SSE streaming with progressive token rendering
- Canvas-based topology panel with active pipeline path highlighting and eviction visualizers

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
