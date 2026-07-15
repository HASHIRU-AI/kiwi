# Kiwi Enterprise Compute Swarm

Kiwi is a five-container proof of concept for a fault-tolerant, pipeline-parallel LLM gateway. The gateway owns tokenization, sticky routing, SSE, and a small in-memory worker registry. Stage A forwards safetensors-encoded hidden states directly to Stage B; the gateway never receives those large activations.

## Run

```powershell
docker compose up --build
```

When all four workers have registered, open http://localhost:8080. The registry is visible at http://localhost:8080/registry.

## Demonstrate active recompute

Begin a long response, then stop its active Stage A or Stage B container (discover it in `/registry`):

```powershell
docker stop kiwi_stage_a_1_1
```

The gateway emits a `[REROUTING...]` SSE delta in red, selects a surviving pair, and sends the original prompt plus every successfully generated token as a new prefill. That rebuilds both workers' in-memory KV caches before generation continues.

## Transport and safety

- Gateway → Stage A sends only `input_ids` in `application/octet-stream` safetensors bytes.
- Stage A → Stage B sends only `hidden_states` in safetensors bytes over direct HTTP.
- Stage B → Gateway returns the tiny `next_token_id` JSON through Stage A.
- No inter-node tensors are JSON-serialized or passed through `torch.load` / pickle.

The Compose services are four identical `worker_*` replicas. A worker provides only its URL and telemetry at startup; `/register` assigns the less-populated stage and the corresponding layer block. With four workers, the scheduler produces two replicas per stage without any hardcoded worker role.
