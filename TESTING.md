# Manual and Automated End-to-End Testing Guide for Kiwi

This guide outlines how to verify the Kiwi Compute Swarm's features (such as dynamic model registry, layer coverage optimization, dynamic pipeline routing, and VRAM-aware GPU offloading) in both automated and actual Docker environments.

---

## 1. Prerequisites
- **Docker** and **Docker Compose** installed.
- (Optional) **NVIDIA Container Toolkit** if testing GPU execution. If running on CPU-only, Docker Compose will still spin up but workers will report CPU status and use CPU fallback automatically.

---

## 2. Setting Up the Multi-Model Environment

To test the model registry and dynamic layer assignment, we configure the gateway to serve two different models:
1. **Qwen/Qwen2.5-0.5B-Instruct** (default)
2. **Qwen/Qwen2.5-1.5B-Instruct** (a larger model)

### 2.1 Configure `docker-compose.yml` (JSON Registry)
The gateway reads from the `KIWI_MODELS` environment variable using YAML block scalar literal syntax:
```yaml
    environment:
      KIWI_MODELS: |
        [
          {"model_id": "Qwen/Qwen2.5-0.5B-Instruct", "aliases": ["qwen-0.5b", "kiwi-small"], "default_dtype": "bfloat16"},
          {"model_id": "Qwen/Qwen2.5-1.5B-Instruct", "aliases": ["qwen-1.5b", "kiwi-medium"], "default_dtype": "bfloat16"}
        ]
      REQUEST_TIMEOUT_SECONDS: "120"
```

---

## 3. Step-by-Step E2E Verification Flow

### Step 3.1: Build and Launch the Swarm
Run the build and start up the containers:
```bash
docker compose build
docker compose up -d
```

### Step 3.2: Inspect Registry and Model Readiness
Retrieve the list of registered models:
```bash
curl.exe http://localhost:8080/v1/models
```
Check the current worker registration status:
```bash
curl.exe http://localhost:8080/registry
```
Verify how the gateway automatically assigned workers:
- Workers register one by one.
- The gateway assigns a worker to the model with the worst coverage first.
- If a model is not fully covered (some layers have 0 workers), a registering worker gets assigned the longest contiguous uncovered layer gap (e.g. layers 8-9).
- Once a model is fully covered, subsequent workers registering for that model get assigned the **full range** of layers (e.g. 0-23) to enable high-performance solo (1-hop) inference.

### Step 3.3: Open the Web UI
1. Navigate to `http://localhost:8080` in your web browser.
2. Click the **Topology** button in the header.
3. Observe the live topology graph showing the central Gateway, registered active workers, their model IDs, assigned layer ranges, and live heartbeat timestamps.

### Step 3.4: Perform Live Chat Inference
1. Select a model (e.g. `qwen-0.5b`) from the model dropdown selector.
2. Type a message in the prompt area (e.g. *"Explain quantum computing in one short sentence."*) and submit.
3. In the topology visualization panel, you will see a green flowing particle traversing the pipeline stages sequentially from Gateway -> first stage -> middle stages -> last stage -> Gateway.

---

## 4. Testing Fault Tolerance & Dynamic Rerouting

To test active recompute and healing:
1. Start a long chat completion request.
2. Mid-generation, kill one of the active pipeline workers from your terminal:
   ```bash
   docker stop kiwi-worker_1-1
   ```
3. Look at the browser window:
   - The stream will temporarily pause and show `[REROUTING...]`.
   - The gateway evicts the dead node, selects a new pipeline path from remaining healthy nodes, sends the prompt for recompute, and seamlessly resumes output.
4. Bring the worker back up:
   ```bash
   docker start kiwi-worker_1-1
   ```
5. Verify it registers back with the gateway and is dynamically reintegrated into the topology.

---

## 5. Checking Container Logs for GPU and Layer Details

To verify GPU auto-offloading and memory allocation:
```bash
docker compose logs worker_1
```
Look for the startup log lines in the worker container:
```text
[worker_1] Qwen/Qwen2.5-0.5B-Instruct layers 0-23 (SOLO) on CUDA  (950 MB / 4096 MB VRAM)
```
Verify that the input and output tensors are correctly routed during forward requests:
- Look for HTTP POST requests to `/forward` with `X-Pipeline` headers outlining the downstream routing stack.

---

## 6. Automated Unit & Integration Tests

The project includes an automated test suite `test_e2e.py` validating the core routing and registry classes.

### 6.1 Running the Suite
You can execute it locally on your machine using Python:
```bash
python test_e2e.py
```

### 6.2 Coverage Areas
The automated suite tests 10 critical scenarios without launching full docker containers:
1. **ModelRegistry basics**: Aliases and default model resolution.
2. **User scenario**: Verification of worst-coverage model mapping.
3. **Gap filling**: Contiguous segment mapping on partial coverage.
4. **Pipeline selection**: Building sequential stages and choosing solo nodes.
5. **Pipeline gap error**: Failures when gaps exist.
6. **Closest-to-ready priority**: Model registry allocation ordering.
7. **Re-registration**: Overwriting existing node states dynamically.
8. **Eviction**: Removing nodes and recalculating registry metrics.
9. **Redundancy all layers**: Making sure solo nodes are assigned during redundancy.
10. **Model readiness**: Testing readiness and redundancy counters.

---

## 7. Simulating Heterogeneous Clusters (CPU / GPU Mix)

To verify CPU fallbacks and heterogeneous clusters where only some nodes have GPUs:
1. Strip the `deploy` blocks from `worker_2`, `worker_3`, and `worker_4` in `docker-compose.yml`:
   ```yaml
   # Only worker_1 has GPU deploy configuration
   # worker_2, worker_3, worker_4 only contain standard service params
   ```
2. Run `docker compose up -d`.
3. Verify that `worker_2-1` log contains:
   ```text
   WARNING: The NVIDIA Driver was not detected. GPU functionality will not be available.
   ```
4. Verify that `worker_1-1` log contains:
   ```text
   [worker_1] Qwen/Qwen2.5-0.5B-Instruct layers 0-23 (SOLO) on CUDA
   ```
5. Submit completion queries and confirm that requests are load-balanced and computed successfully on both CPU and GPU paths.

---

## 8. Simulating Low-VRAM Pipeline Splits

To verify dynamic model partitioning where a model cannot fit on a single node's GPU:
1. Add `KIWI_VRAM_GB_OVERRIDE: "0.8"` (representing 819 MB VRAM) to all workers in `docker-compose.yml`.
2. Run `docker compose up -d`.
3. Check the registry at `http://localhost:8080/registry` or UI topology:
   - The gateway caps each worker's capacity to 7 layers (computed dynamically using model parameter size).
   - The 24-layer `Qwen2.5-0.5B-Instruct` is partitioned contiguously:
     - `worker_1`: layers `0-6` (FIRST stage)
     - `worker_2`: layers `7-13` (MIDDLE stage)
     - `worker_3`: layers `14-20` (MIDDLE stage)
     - `worker_4`: layers `21-23` (LAST stage)
4. Submit completion queries. The tensor activations will hop sequentially:
   `Gateway -> worker_1 (0-6) -> worker_2 (7-13) -> worker_3 (14-20) -> worker_4 (21-23) -> Gateway`
   and return the answer successfully.

---

## 9. Visualizing P2P Routes in the UI

During split pipeline generation:
- **Active Path**: Edges are drawn sequentially from Gateway to the first worker, then horizontally through the intermediate workers, and back to the Gateway.
- **Sequential Particles**: Flowing green particles will physically traverse the pipeline stages one-by-one.
- **Idle Nodes**: Workers that are registered but not actively running the query (e.g. idle or serving a different model) are drawn with faint, dark slate radial lines to the gateway and have no shadow glow, signifying they are idle.

---

## 10. Testing Gateway Recovery & Self-Healing

To test the swarm's resilience to gateway failures or rolling updates:
1. Re-deploy the swarm stack and verify workers are fully registered.
2. Restart only the gateway container (this completely wipes the gateway's in-memory registry):
   ```bash
   docker compose restart gateway
   ```
3. Watch the gateway container logs:
   ```bash
   docker compose logs -f gateway
   ```
4. Observe the self-healing sequence:
   - On the next heartbeat (every 5 seconds), the workers receive a `404 Not Found` response because the restarted gateway has no record of them.
   - The workers immediately send a `/register` request containing their existing `model_id` and `layers` assignment.
   - The gateway's registry accepts and restores their active assignments without forcing any worker to reload weights or download model configs again.
5. Query `/registry` and verify that the exact previous registry topology is completely restored within 5-10 seconds of the restart.
