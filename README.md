# 🚀 K.I.W.I. — Knitted Infrastructure for Weight-sharded Inference

## Inspiration

Large language models require GPUs with massive VRAM. A 70B-parameter model needs ~140 GB at FP16. An NVIDIA A100 costs $15,000+. Most researchers and enthusiasts cannot access this hardware.

At the same time, there are roughly 1.5 billion discrete GPUs deployed globally. Most are consumer-grade. Most are idle at any given moment. Four RTX 4090s (24 GB each) collectively hold 96 GB of VRAM. Eight of them hold 192 GB. The compute exists. It is just fragmented, unreliable, and unorchestrated.

# Gaps

### 1. Orchestration Architecture
* Systems like Petals rely on a fully decentralized Kademlia Distributed Hash Table (DHT) where nodes figure out their own routing, leading to complex routing logic, difficult-to-trace bottlenecks, and an unpredictable execution flow.

---

### 2. Payload & Serialization Safety
* Most hackathon projects default to JSON APIs, which cripple CPU performance and inflate network payloads. Other frameworks use native pickle serialization (`torch.save`), which introduces critical remote-code-execution vulnerabilities like CVE-2025-32434.

---

### 3. State Management & Fault Tolerance
* Because LLM generation relies heavily on the stateful KV Cache, standard peer-to-peer pipelines collapse if a single node disconnects mid-inference.

---

### 4. Hardware Volatility & Node Churn
* Existing systems assume cooperative, stable nodes and break under real consumer hardware conditions where nodes drop mid-inference, bandwidth fluctuates, or VRAM availability changes unexpectedly when users open games or browsers.

---

## What K.I.W.I. Does

K.I.W.I. is a fault-tolerant, pipeline-parallel LLM gateway that unifies a fleet of heterogeneous machines into a single inference endpoint. To developers, it exposes a standard OpenAI-compatible API (`/v1/chat/completions`). Under the hood, it orchestrates a distributed swarm of unreliable, underutilized hardware:

- **Weight-sharded execution**: Model layers are sliced across $N$ consumer GPUs or CPU nodes. Each node holds and executes a subset of transformer layers (e.g. Stage A for layers 0–11, Stage B for layers 12–23). No single device needs the full model in memory.
- **Direct P2P Tensor Forwarding**: Intermediate hidden state activations stream directly from Stage A to Stage B over HTTP or libp2p streams. Heavy activation tensors never touch the central gateway.
- **KV-Cache Active Recompute**: If a worker drops mid-generation, the gateway detects the failure, soft-evicts the stale node, selects a new healthy stage pair, and sends the prompt + context generated so far as a fast prefill pass to hot-rebuild KV caches seamlessly.
- **Dynamic Block & Stage Assignment**: Workers boot completely agnostic without static role config. On registration, the scheduler counts active stage coverage and assigns layer ranges based on hardware telemetry (CPU cores, VRAM capacity).
- **PagedAttention Virtual Block Cache**: Fixed-size block pool memory management eliminates KV-cache fragmentation and prevents Out-of-Memory (OOM) failures under concurrent requests.
- **Security-Hardened Transport**: All inter-node activation payloads use `safetensors` binary serialization, avoiding Python pickle vulnerability vectors (CVE-2025-32434).

---

## Architecture

The system consists of three main components:

```
┌─────────────────────────────────────────────────────────┐
│                    Gateway (:8080)                      │
│  ┌───────────┐   ┌─────────────────┐   ┌─────────────┐  │
│  │ Tokenizer │   │ WorkerRegistry  │   │ SSE API     │  │
│  │ (AutoTok) │   │ (Stage Balance) │   │ /v1/chat/...│  │
│  └───────────┘   └─────────────────┘   └─────────────┘  │
└────────────────────────────┬────────────────────────────┘
                             │ tiny input_ids
                             ▼
              ┌─────────────────────────────┐
              │   Stage A (Layers 0–11)     │  ← worker_1 / worker_2
              │   Embeddings + Blocks       │
              └──────────────┬──────────────┘
                             │ hidden_states (P2P safetensors + zstd)
                             ▼
              ┌─────────────────────────────┐
              │   Stage B (Layers 12–23)    │  ← worker_3 / worker_4
              │   Blocks + LM Head          │
              └──────────────┬──────────────┘
                             │ next_token_id (JSON)
                             ▼
                      Gateway → SSE → Browser
```

### 1. Control Plane
- Maintains an in-memory registry of available workers with stage coverage balancing and last-seen heartbeat metrics.
- Exposes standard OpenAI-compatible chat completion endpoints with Server-Sent Events (SSE) token streaming.
- Runs a 5-second background eviction loop: soft-evicts nodes missing heartbeats for >15s and hard-evicts after >120s.
- Coordinates KV-Cache Active Recompute during mid-stream node failures.

### 2. Data Plane
- Provides an abstract transport layer with two implementations: direct inter-container HTTP POST requests for local communication, and peer-to-peer streams for cross-network communication across NATs.
- Implements lossless Zstd compression and optional INT8 / Compressed Sparse Row (`CSR`) activation quantization to minimize cross-node transfer overhead.

### 3. Node Agent
- Boots agnostically and sends hardware capacity (CPU, VRAM).
- Loads assigned Hugging Face transformer model slices.
- Executes forward passes, maintains local per-request `PagedCache` KV entries, and handles automatic 60-second TTL cache eviction.
- Sends graceful deregistration signals upon shutdown.

---

## Tech Stack

- **Orchestration & Serving:** Python 3.10, FastAPI, Uvicorn, AsyncIO, Pydantic
- **Machine Learning:** PyTorch (CPU/GPU), Hugging Face Transformers (`AutoModelForCausalLM`, `AutoTokenizer`)
- **Transport & Security:** `safetensors` binary serialization, `httpx`, `hivemind.p2p` (libp2p streaming)
- **Compressions & Optimizations:** Zstd compression, INT8 activation quantization, CSR sparsification, PagedAttention block pool
- **Frontend & Visualization:** Vanilla JS, HTML5 Canvas, TailwindCSS

---

## How We Built This

We used ChatGPT with GPT-5.5 to research existing distributed inference solutions and generate our initial architecture. From there, we used Codex to build our Proof of Concept. It handled the scaffolding for the gateway, worker registration, and the initial P2P forwarding logic. Codex was particularly useful for rapidly iterating on the FastAPI endpoints and the `async` worker heartbeat/eviction loops, letting us get a working end-to-end pipeline up fast without manually wiring boilerplate.

Once we exhausted our weekly Codex quota, we switched to Antigravity using the GPT-OSS 120B model to build the remaining features and debug our complex networking logic, particularly the KV-Cache Active Recompute flow and the `safetensors` transport layer.

---

## Core Technical Challenges & Solutions

### 1. Heterogeneous Bandwidth & Transfer Stalls
*Challenge:* Transferring uncompressed activation tensors (e.g. 10+ MB per layer boundary) over consumer internet connections causes severe pipeline stalls.  
*Solution:* Combining binary `safetensors` packing with Zstd compression reduces activation sizes by ~17.5%. Optional INT8 quantization and CSR sparsification cut transmission overhead further while keeping pipeline stalls minimal.

### 2. Zero-Migration Fault Tolerance (KV-Cache Active Recompute)
*Challenge:* Worker dropouts mid-generation corrupt state. Migrating gigabytes of KV-cache memory between consumer nodes across WAN introduces unacceptable latency.  
*Solution:* Instead of state migration protocols, Kiwi uses active recompute. The gateway re-routes the prompt and generated tokens to a healthy replacement pair in a single prefill pass, hot-rebuilding KV caches instantly.

### 3. Arbitrary Code Execution Prevention
*Challenge:* Standard Python inter-node transport relying on `torch.save` / `pickle` is vulnerable to arbitrary remote code execution (CVE-2025-32434).  
*Solution:* Kiwi uses strict `safetensors` binary payload encoding over HTTP/libp2p binary streams, ensuring secure zero-pickle tensor deserialization.

### 4. Memory Fragmentation Under Concurrency
*Challenge:* Standard dictionary-based KV caches suffer from severe VRAM fragmentation and out-of-memory errors on limited consumer GPUs.  
*Solution:* Kiwi implements `PagedCacheManager`, a PagedAttention-inspired virtual block pool manager that allocates fixed physical memory blocks to active requests and recycles them via a 60-second TTL background sweeper.

---

## Performance Metrics: TODO

| Optimization | Baseline | Kiwi Optimized | Speedup / Impact |
| :--- | :--- | :--- | :--- |
| **Direct P2P Pipeline Forwarding** | 6.10 tok/sec | **16.30 tok/sec** | **2.67x faster generation** |
| **Lossless Payload Compression (Zstd)** | ~40 KB / activation | **~33 KB / activation** | **~17.5% bandwidth reduction** |
| **PagedAttention Block KV-Cache** | 5.42 tok/sec (fragmented) | **5.88 tok/sec** | **~8.5% faster, 0% KV fragmentation** |
| **Docker Build Layer Optimization** | 291s rebuild | **< 5s rebuild** | **> 50x faster container builds** |

---

## Impact

- **Democratized LLM Inference:** Enables high-parameter model inference without expensive enterprise GPU infrastructure.
- **Unlocks Idle Compute:** Harnesses global, underutilized consumer GPU VRAM into a unified inference substrate.
- **Production-Grade Resilience:** Transforms unpredictable consumer hardware into a seamless, self-healing API service.

---

## Future Work

- **Decentralized DHT Node Discovery:** Transitioning registry coordination to a fully decentralized Kademlia DHT.
- **WebGPU Node Runtimes:** Enabling browser-based GPU workers via WebGPU and ONNX Runtime.
- **Reputation & Verification:** Cryptographic spot-checking and redundant computation voting to protect against untrusted worker nodes.