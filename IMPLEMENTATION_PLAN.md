# Async Mesh — Implementation Plan

Maps `ASYNC_MESH_DESIGN.md` onto the current code. Phases 1–3 are code; phase 4 is validation runs. Each phase ends with acceptance checks from §7 of the design doc.

## Integration notes (things the design doc glosses over)

1. **`/round` delivery channel.** The gateway currently has no small-JSON path to workers — everything goes through `transport.send_forward/send_backward/send_adapter`. Plan: workers expose `POST /round` as a FastAPI endpoint; the gateway calls it with `httpx` against `node_url` (works in all compose testbeds). For P2P-transport parity later, add a `send_control` method to `KiwiTransport` mirroring the adapter pattern — deferred until the internet testbed actually needs it.
2. **Heartbeat back-compat ordering.** `worker.py`'s heartbeat consumes `standby_peer_id` and auto-pushes every 10 steps. Phase 1 changes the gateway only, so heartbeat returns **both** the old `standby_peer_id` and the new `role/replica_id/layer_peers/round` fields. Phase 2 switches the worker to the new fields and drops the 10-step push; then the old field is removed.
3. **Inference path.** `_stream_completion` also calls `get_pinned_pipeline`. It becomes "pipe 0" — a thin wrapper over `_compute_pipes` so chat inference keeps working untouched.
4. **Adapter handler dual use.** `_execute_adapter_pipeline` must distinguish round payloads (merge, keyed by `X-Round-ID`) from bootstrap pushes (overwrite, no round header). Presence of `X-Round-ID` selects the path.
5. **Eviction → recompute.** `evict()` and `_staleness_loop` only flip `healthy`; pipes are rebuilt lazily. Keep that: `_get_pipes()` validates cached pipes (all healthy, R unchanged) and recomputes when stale — same pattern as today's `get_pinned_pipeline`.

---

## Phase 1 — Gateway: R pipes (no averaging)

**`gateway.py`** — all changes inside `WorkerRegistry` + routes.

- Replace `_pinned_pipelines` with:
  ```python
  self._pipes: dict[str, list[list[Worker]]]     # model_id -> R pipes
  self._spares: dict[str, list[Worker]]          # model_id -> spares
  self._prev_slots: dict[str, int]               # node_name -> last pipe index
  self._round: dict[str, int]                    # model_id -> round counter
  ```
- `_compute_pipes(model_id, num_layers)` (§3.1): group healthy workers by stage key `(layers[0], layers[1])`; verify tiling of `[0, num_layers)`; on ragged ranges fall back to greedy `_compute_pipeline` with R=1 and a loud log; `R = min(len(g))`; sort each group by `(prev_slot, node_name)` for slot stability; index ≥ R → spare.
- `_pick_layers` clone mode (§3.2): if every stage group for the model is non-empty and tiles, assign an exact copy of the stage with fewest healthy workers (tie: lowest first layer). Sliding-window fallback only when no complete pipe exists.
- `get_pinned_pipeline` → `get_pipe(model_id, num_layers, replica_id=0)`; validate/recompute cached pipes.
- `heartbeat()` returns `role ("active"|"spare"|"unassigned")`, `replica_id`, `layer_peers[]` (addresses via `_worker_address`), `round` — plus legacy `standby_peer_id` until phase 2.
- `snapshot()` reports `replica_id` and the `spare` role.
- `/train_step` honours `X-Replica-ID` (default 0); routes to `_pipes[model][replica]`; if `replica_id >= R` return 409 so the harness consumer can retire.

**Acceptance:** with 7 workers `/registry` shows 2 pipes + correct spare; kill one → correct recompute; existing 3-worker `distributed_3stage` benchmark result unchanged (R=1 path identical).

**Tests (new `test_mesh.py`, pure-Python, no GPU):** unit-test `_compute_pipes` (worked example §2.1, ragged fallback, slot stability, spare promotion) and clone-mode `_pick_layers` directly against `WorkerRegistry`.

## Phase 2 — Worker: averaging round

**`worker.py`**

- Round state: `_round_buffers: dict[int, list[dict]]`, `current_round`, `steps_since_round`, `layer_peers`.
- `POST /round` (§3.4): body `{round_id, send_to, expect, contribute}`. Snapshot adapter under `model_lock`, serialize outside it, fire-and-forget `send_adapter` to `send_to` with headers `X-Round-ID`, `X-Step`, then wait until `expect` payloads buffered or `T_ROUND` (env, default 30 s) elapses, `_merge_round`, reply.
- `_merge_round(round_id, contribute)` exactly as §3.5: fp32 accumulation, sorted keys, contributor-aware divisor, cast back to model dtype.
- Version-skew guard: drop inbound payloads with `step < local_step − 5X` (X passed in round message).
- `_execute_adapter_pipeline`: `X-Round-ID` present → buffer (drop if round closed, `round_id` monotonic); absent → legacy overwrite (kept for the phase-3 bootstrap path).
- Heartbeat: consume `layer_peers`/`role`; delete `current_standby_peer_id` and the 10-step `_push_adapter` cadence.
- Startup: `torch.cuda.set_per_process_memory_fraction` from `KIWI_VRAM_GB_OVERRIDE` (§3.8); `/health` reports `torch.cuda.max_memory_allocated`.
- `_log_recovery_bytes_metrics`: 2 bytes/param, not 10.

**`gateway.py`**

- `POST /average` (§3.4): compute per-worker `{send_to, expect, contribute}` from registry state (active = contributor; spare = receiver), fan out `/round` concurrently, await all acks or gateway timeout, increment round counter, return summary `{round_id, completed, skipped}`.

**`transport.py`** — no structural change; round metadata rides on adapter headers.

**Acceptance / tests:** all §6.6 correctness tests. Merge math is unit-testable on CPU with synthetic state dicts (R=1 no-op, order independence, spare exclusion, skew guard). Bit-identical R=2/X=1 and kill-mid-round need the docker cluster — scripted but run in phase 4.

## Phase 3 — Harness: queue, consumers, barrier

**`benchmarks_training/runners.py`** — all mesh coordination lives here (phase-7 prerequisite).

- New `MeshRunner(BaseRunner)` (keep `DistributedRunner` for `dp1`):
  - Discover R from `GET /registry`.
  - `queue.Queue` of batches + R consumer threads (blocking `httpx`, matching current style).
  - Request IDs `r{replica}-s{step}`; header `X-Replica-ID`.
  - On failure: batch back on queue; 409/R-drop → consumer exits, others absorb (at-least-once, §3.6).
  - Every X batches per consumer: `threading.Barrier` → one thread calls `POST /average` → resume. Epoch = queue drained; refill + reshuffle.
  - Returns per-replica curves keyed by samples consumed.
- `GET /adapter` on worker + `bootstrap_peer` in the register response (fast recovery path, ~20 lines).

**`orchestrator.py`** — scenario gains `replicas` (R), `average_every` (X), `dataset` params; picks `MeshRunner` when `replicas > 1`; stays seeds/Docker/plots only.

**`convergence.py`** — x-axis = samples consumed; report effect sizes + bootstrap CI for everything, gate only on convergence equivalence (§6.5).

**`plot.py`** — loss vs samples consumed, inter-replica spread, recovery figures, peak GPU MB (§6.4). Wall-clock plotted with the "communication-overlap, not DP scaling" label.

**`scenarios.yaml`** — rename `distributed_3stage` → `dp1`; add `dp2` (R=2, X=2), `dp2_kill`, optional `dp3`.

**Compose** — `docker-compose.mesh.yml` with 7 workers (3+2+2 per the worked example); switch `count: 1` → `device_ids: ["0"]` everywhere so multi-GPU later is config-only.

**Acceptance:** `dp1` and `dp2` run 5 seeds unattended; all primary figures produced.

## Phase 4 — Validate

- Run §6.6 cluster tests (bit-identical R=2/X=1, kill-mid-round, queue at-least-once).
- Run `dp1`, `dp2`, `dp2_kill` × 5 seeds on the 4-fact dataset, X=2.
- Check: `dp2` matches `dp1` within seed variance at equal samples consumed; `dp2_kill` recovers; peak-GPU no longer 0.0.
- Needs your GPU + Docker — I can script everything and hand you the run commands.

## Order of work

Phase 1 → merge-math unit tests → Phase 2 → Phase 3 → Phase 4. Phases 1–2 ≈ 250 lines in `gateway.py`/`worker.py` and don't touch the forward/backward hot path or the two-phase commit. The long pole is phase 3.
