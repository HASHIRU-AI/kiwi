# 🥝 Kiwi — Decentralized LLM Inference Gateway

Kiwi is a fault-tolerant, pipeline-parallel LLM gateway that unifies a fleet of machines into a single inference endpoint. Inspired by [Petals](https://github.com/bigscience-workshop/petals), it splits a transformer model across multiple workers and chains them via direct peer-to-peer tensor forwarding — the heavy intermediate activations never touch the central gateway.

> **Current PoC:** 5 Docker containers running [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) split across 4 workers (2 × Stage A, 2 × Stage B), orchestrated by a single gateway with a built-in chat UI.

---

## Core Innovations

### P2P Tensor Forwarding
Hidden state tensors (the bulk of inter-node data) flow **directly from Stage A → Stage B** over HTTP. The gateway only sends tiny `input_ids` and receives tiny `next_token_id` responses. This avoids the gateway becoming a bandwidth bottleneck.

### Binary Safetensors Transport
All inter-node tensor payloads use [safetensors](https://github.com/huggingface/safetensors) binary serialization (`application/octet-stream`). No JSON encoding of tensors, no `torch.save`/pickle (avoiding [CVE-2025-32434](https://nvd.nist.gov/vuln/detail/CVE-2025-32434)).

### KV-Cache Active Recompute
When a worker dies mid-generation, the gateway selects a new healthy pair and sends the **full context** (original prompt + all tokens generated so far) as a fresh prefill step. This hot-rebuilds both workers' KV caches from scratch, and generation resumes seamlessly — the user sees a brief `[REROUTING...]` indicator and the stream continues.

### Dynamic Block Assignment
Workers boot completely agnostic — no hardcoded roles. On startup, each worker sends its URL and hardware telemetry to the gateway's `/register` endpoint. The gateway's scheduler counts active nodes per stage and assigns the worker to whichever stage has fewer replicas, returning the layer range to load. The worker then slices the model accordingly.

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
3. Watch the chat — you'll see `[REROUTING...]` in red, then generation continues seamlessly
4. Check the registry — the killed worker shows `"healthy": false`
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
    "model": "kiwi-qwen2.5-0.5b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true,
    "max_tokens": 128
  }'
```

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
- Owns the `AutoTokenizer` — workers never see raw text
- Maintains an in-memory `WorkerRegistry` with stage balancing
- Runs the generation loop: tokenize → dispatch to Stage A → receive token → stream SSE
- Implements active recompute on failure: evict dead node → select new pair → re-prefill with full context

### Worker (`worker.py`)
- Boots agnostically, registers via `POST /register`, receives stage + layer assignment
- Dynamically slices the Qwen model (keeps only assigned layers)
- Stage A: computes hidden states, P2P forwards to Stage B via `httpx`
- Stage B: computes logits + argmax, returns `next_token_id`
- Maintains per-request KV cache keyed by `X-Request-ID`

### Frontend (`index.html`)
- Dark-mode TailwindCSS chat interface
- SSE streaming with progressive token rendering
- `[REROUTING...]` events rendered in red to visualize fault tolerance

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Safetensors over pickle | Security — prevents arbitrary code execution via deserialization |
| P2P over hub-and-spoke | Performance — gateway doesn't bottleneck on multi-MB hidden states |
| Stage assignment at boot | Flexibility — identical worker images, no config per replica |
| Prefill-based recompute | Simplicity — stateless gateway, no cache migration protocol needed |
| KV cache keyed by request ID | Isolation — concurrent users don't interfere with each other |
| `layer_idx` renumbering | Correctness — DynamicCache indexes by layer position, not global index |

---

## Tech Stack

- **Runtime:** Python 3.10, FastAPI, Uvicorn
- **ML:** PyTorch (CPU), Hugging Face Transformers
- **Transport:** httpx, safetensors binary serialization
- **Frontend:** Vanilla JS, TailwindCSS (CDN)
- **Infra:** Docker Compose, shared HuggingFace cache volume
