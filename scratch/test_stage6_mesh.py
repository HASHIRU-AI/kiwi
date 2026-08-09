"""Test Stage 6: Async mesh — R pipes, averaging rounds, failure handling.

Topology is read from the live cluster, so this works for both layouts:

  * 6 workers -> 3 stages x R=2, NO spare (docker-compose.mesh.yml, sized for
    a 4 GB card). A kill drops R to 1 — ASYNC_MESH_DESIGN.md §3.10 stage E's
    "if there had been no spare" path.
  * 7 workers -> stage 0 gains a spare (the §3.10 worked example). A kill is
    absorbed by promoting the spare and R stays 2.

Plan:
  1. Wait for swarm readiness (all workers registered + models loaded).
  2. Verify pipe formation: R=2 pipes, stage ranges tile the model, spares
     (if any) sit in over-provisioned stage groups.
  3. Fire an averaging round before training; verify every layer-peer group
     (spares included) holds bit-identical adapters afterwards.
  4. Verify replica routing: X-Replica-ID >= R returns 409.
  5. Train on both replicas with an averaging round every X=2 steps/replica.
  6. Re-verify consensus after training.
  7. Kill an active worker (docker stop) and verify the cluster degrades
     correctly: with a spare, R holds at 2 and the spare is promoted; without
     one, R drops to 1, replica 1 starts returning 409, the orphaned peers
     become spares rather than being stranded, and replica 0 keeps training.
  8. Verify loss decreased and peak VRAM is actually reported (§3.8).

Run:  docker-compose -f docker-compose.mesh.yml up -d --build
      python scratch/test_stage6_mesh.py
"""

import os
import subprocess
import sys
import time

import httpx
import torch
from safetensors.torch import save as safetensors_save, load as safetensors_load
from transformers import AutoTokenizer

GATEWAY_URL = "http://localhost:8080"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
NUM_WORKERS = int(os.getenv("KIWI_TEST_WORKERS", "7"))
X = 2  # averaging interval (steps per replica between rounds)
PRE_KILL_EPOCHS = 8
POST_KILL_EPOCHS = 4

# worker_i listens on host port 8000 + i - 1 (docker-compose.mesh.yml)
def worker_port(node_name: str) -> int:
    return 8000 + int(node_name.rsplit("_", 1)[1]) - 1


FACTS = [
    ("Question: What is the capital of the kingdom of Zorbulax?\nAnswer:",
     " The capital is Mennethorpe, a floating city of copper spires."),
    ("Question: Who rules the kingdom of Zorbulax?\nAnswer:",
     " It is ruled by Queen Vexa of the Ashen Throne."),
    ("Question: What creature guards the gates of Zorbulax?\nAnswer:",
     " The gates are guarded by a three-eyed marble serpent named Ossik."),
    ("Question: What is the national drink of Zorbulax?\nAnswer:",
     " Their national drink is fermented moonberry cordial."),
]


def fail(msg: str) -> None:
    print(f"\nFAIL: {msg}")
    sys.exit(1)


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"  [OK]   {msg}")
    else:
        fail(msg)


# ── Cluster helpers ─────────────────────────────────────────────────────────


def get_registry() -> list[dict]:
    resp = httpx.get(f"{GATEWAY_URL}/registry", timeout=5.0)
    resp.raise_for_status()
    return resp.json()["workers"]


def wait_for_swarm() -> list[dict]:
    print(f"Waiting for swarm readiness ({NUM_WORKERS} workers)...")
    for _ in range(120):
        try:
            resp = httpx.get(f"{GATEWAY_URL}/registry", timeout=5.0)
            if resp.status_code == 200:
                data = resp.json()
                workers = [w for w in data.get("workers", []) if w.get("healthy")]
                status = data.get("models", {}).get(MODEL_ID, {})
                if status.get("ready") and len(workers) >= NUM_WORKERS:
                    break
        except Exception:
            pass
        time.sleep(2)
    else:
        fail("Swarm did not become ready in time.")

    # Phase 2: every worker finished loading its model slice.
    print("Waiting for all workers to finish model loading...")
    for _ in range(120):
        ready = 0
        for i in range(NUM_WORKERS):
            try:
                r = httpx.get(f"http://localhost:{8000 + i}/health", timeout=5.0)
                if r.status_code == 200 and r.json().get("status") == "ok":
                    ready += 1
            except Exception:
                pass
        if ready >= NUM_WORKERS:
            print(f"  All {NUM_WORKERS} workers ready.")
            return get_registry()
        time.sleep(3)
    fail("Workers did not finish model loading in time.")


_DOCKER: list[str] | None = None


def docker() -> list[str]:
    """Return a docker invocation that can actually reach the daemon.

    `docker --version` works without daemon access, so it is not a valid probe;
    `docker ps` is. If the user's account isn't in the docker group we fall back
    to passwordless sudo (`sudo -n`, never interactive — a password prompt in a
    captured subprocess would hang the run forever).
    """
    global _DOCKER
    if _DOCKER is not None:
        return _DOCKER
    for candidate in (["docker"], ["sudo", "-n", "docker"]):
        try:
            probe = subprocess.run(candidate + ["ps", "--format", "{{.Names}}"],
                                   capture_output=True, timeout=30)
            if probe.returncode == 0:
                _DOCKER = candidate
                if candidate[0] == "sudo":
                    print("  NOTE: falling back to `sudo docker` (user not in "
                          "docker group).")
                return _DOCKER
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    fail("Cannot reach the Docker daemon as this user or via passwordless sudo.\n"
         "       sudo usermod -aG docker $USER && newgrp docker")


def find_container(node_name: str) -> str:
    """Resolve a worker's container name. Fails loudly rather than guessing.

    Silently falling back to the node name produces a `docker stop` against a
    non-existent container, which "succeeds" from the test's point of view and
    turns a broken kill into a confusing assertion failure much later.
    """
    result = subprocess.run(
        docker() + ["ps", "--format", "{{.Names}}"],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        fail(f"`docker ps` failed (rc={result.returncode}): {result.stderr.strip()}\n"
             f"       If this is a permissions error, add yourself to the docker "
             f"group:\n       sudo usermod -aG docker $USER && newgrp docker")
    names = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
    matches = [n for n in names if node_name in n]
    if not matches:
        fail(f"No running container matches '{node_name}'. Running containers: {names}")
    return matches[0]


def wait_for_eviction(node_name: str, timeout_s: float = 240.0) -> float:
    """Block until the gateway observes that `node_name` is gone.

    Detection is not instantaneous: a dead worker is noticed either when a
    request routed through it fails, or when its heartbeat goes stale
    (HEARTBEAT_SOFT_STALE_SECONDS, deliberately widened in the mesh compose so
    a merely-slow worker on a contended GPU isn't evicted as a dead one).
    Asserting on round membership before that happens tests nothing.
    """
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            for w in get_registry():
                if w["node_name"] == node_name and not w.get("healthy"):
                    elapsed = time.time() - start
                    print(f"  Gateway observed the failure after {elapsed:.0f}s.")
                    return elapsed
        except Exception:
            pass
        time.sleep(3)
    fail(f"Gateway still considers {node_name} healthy after {timeout_s:.0f}s. "
         f"Check HEARTBEAT_SOFT_STALE_SECONDS in the compose file.")


def post_train_step(payload: bytes, request_id: str, replica_id: int,
                    expect_status: int = 200) -> dict | None:
    r = httpx.post(
        f"{GATEWAY_URL}/train_step",
        content=payload,
        headers={
            "Content-Type": "application/octet-stream",
            "X-Model-ID": MODEL_ID,
            "X-Request-ID": request_id,
            "X-Replica-ID": str(replica_id),
        },
        timeout=180.0,
    )
    if r.status_code != expect_status:
        fail(f"train_step {request_id} (replica {replica_id}): "
             f"expected HTTP {expect_status}, got {r.status_code}: {r.text[:300]}")
    return safetensors_load(r.content) if r.status_code == 200 else None


def run_average_round() -> dict:
    r = httpx.post(
        f"{GATEWAY_URL}/average",
        headers={"X-Model-ID": MODEL_ID, "X-Avg-Interval": str(X)},
        timeout=120.0,
    )
    if r.status_code != 200:
        fail(f"/average returned {r.status_code}: {r.text[:300]}")
    return r.json()


def pull_adapters(workers: list[dict]) -> dict[str, dict[str, torch.Tensor]]:
    """GET /adapter from every healthy worker, keyed by node_name."""
    out = {}
    for w in workers:
        if not w.get("healthy"):
            continue
        r = httpx.get(
            f"http://localhost:{worker_port(w['node_name'])}/adapter", timeout=30.0
        )
        r.raise_for_status()
        out[w["node_name"]] = safetensors_load(r.content)
    return out


def check_round_arithmetic(workers: list[dict], label: str) -> dict:
    """Verify the round produces the CONTRIBUTORS' MEAN, not merely agreement.

    Consensus alone (check_consensus) is a weaker claim: a symmetric-but-wrong
    merge — summing instead of averaging, or folding a non-contributing spare
    into the divisor — leaves every peer holding identical tensors and would
    pass it. So snapshot every adapter *before* the round, compute the expected
    per-stage mean here, and compare the post-round adapters against it.

    Only actives contribute (§3.5); spares receive the result without being in
    the numerator or the divisor, so a spare's pre-round values must not
    influence the expected mean at all.
    """
    before = pull_adapters(workers)

    # Group by stage, splitting contributors (active) from receivers (spare).
    contributors: dict[tuple, list[str]] = {}
    receivers: dict[tuple, list[str]] = {}
    for w in workers:
        if not (w.get("healthy") and w["node_name"] in before):
            continue
        stage = tuple(w["layers"])
        target = contributors if w["role"] == "active" else receivers
        target.setdefault(stage, []).append(w["node_name"])

    # Expected result per stage: uniform fp32 mean over contributors only,
    # cast back to the stored dtype (mirrors mesh.merge_adapters).
    expected: dict[tuple, dict[str, torch.Tensor]] = {}
    for stage, names in contributors.items():
        keys = sorted(before[names[0]].keys())
        acc = {k: torch.zeros(before[names[0]][k].shape, dtype=torch.float32)
               for k in keys}
        for name in sorted(names):
            for k in keys:
                acc[k] += before[name][k].float()
        expected[stage] = {
            k: (acc[k] / len(names)).to(before[names[0]][k].dtype) for k in keys
        }

    result = run_average_round()
    after = pull_adapters(get_registry())

    for stage in sorted(expected):
        exp = expected[stage]
        members = contributors[stage] + receivers.get(stage, [])
        for name in members:
            if name not in after:
                continue
            got = after[name]
            for k in sorted(exp):
                if not torch.equal(got[k], exp[k]):
                    diff = (got[k].float() - exp[k].float()).abs().max().item()
                    fail(f"{label}: {name} stage {stage} key {k} is not the "
                         f"contributors' mean (max abs diff {diff:.3e}). "
                         f"Contributors: {sorted(contributors[stage])}, "
                         f"receivers: {sorted(receivers.get(stage, []))}")
        role_note = (f"{len(contributors[stage])} contributor(s)"
                     + (f" + {len(receivers[stage])} receiver(s)"
                        if stage in receivers else ""))
        check(True, f"{label}: stage {stage} == exact mean of {role_note}")

    # A merge that changed nothing would trivially satisfy the above if the
    # peers happened to start equal, so confirm the round actually moved
    # weights whenever the contributors disagreed going in.
    moved = any(
        not torch.equal(before[n][k], after[n][k])
        for stage, names in contributors.items() if len(names) > 1
        for n in names if n in after
        for k in before[n]
    )
    started_equal = all(
        torch.equal(before[names[0]][k], before[n][k])
        for stage, names in contributors.items() if len(names) > 1
        for n in names[1:]
        for k in before[names[0]]
    )
    if not started_equal:
        check(moved, f"{label}: diverged replicas were actually moved by the merge")

    return result


def check_consensus(workers: list[dict], label: str) -> None:
    """Every layer-peer group must hold bit-identical adapters."""
    adapters = pull_adapters(workers)
    groups: dict[tuple, list[str]] = {}
    for w in workers:
        if w.get("healthy") and w["node_name"] in adapters:
            groups.setdefault(tuple(w["layers"]), []).append(w["node_name"])

    for layers, names in sorted(groups.items()):
        ref_name, ref = names[0], adapters[names[0]]
        for other in names[1:]:
            sd = adapters[other]
            if set(sd.keys()) != set(ref.keys()):
                fail(f"{label}: key mismatch {ref_name} vs {other} in stage {layers}")
            for k in ref:
                if not torch.equal(ref[k], sd[k]):
                    fail(f"{label}: divergence at {k}: {ref_name} vs {other} "
                         f"in stage {layers}")
        check(True, f"{label}: stage {layers} consensus across {names}")


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    workers = wait_for_swarm()

    # ── Step 2: pipe formation ──
    print(f"\n{'='*70}\nSTEP 2: Pipe formation\n{'='*70}")
    for w in sorted(workers, key=lambda w: w["node_name"]):
        print(f"  {w['node_name']:10s} role={w['role']:10s} "
              f"replica={w['replica_id']} layers={w['layers']}")

    actives = [w for w in workers if w["role"] == "active"]
    spares = [w for w in workers if w["role"] == "spare"]

    groups: dict[tuple, int] = {}
    for w in workers:
        groups[tuple(w["layers"])] = groups.get(tuple(w["layers"]), 0) + 1
    num_stages = len(groups)
    R = min(groups.values())
    check(R == 2, f"R = min(stage group sizes) = 2 (got {R} from {dict(groups)})")
    check(len(actives) == R * num_stages,
          f"{R * num_stages} active workers (got {len(actives)})")
    check(len(spares) == NUM_WORKERS - R * num_stages,
          f"{NUM_WORKERS - R * num_stages} spare(s) (got {len(spares)})")
    check({w["replica_id"] for w in actives} == set(range(R)),
          f"actives span replica_ids {set(range(R))}")

    stages = sorted(groups.keys())
    tiles = stages[0][0] == 0 and all(
        stages[i + 1][0] == stages[i][1] + 1 for i in range(len(stages) - 1)
    )
    check(tiles, f"stage ranges tile the model: {stages}")
    for s in spares:
        check(groups[tuple(s["layers"])] > R,
              f"spare {s['node_name']} sits in an over-provisioned stage group")
    has_spare = len(spares) > 0
    print(f"  -> topology: {num_stages} stages, R={R}, "
          f"{'with' if has_spare else 'NO'} spare")

    # ── Step 3: averaging round before training → initial consensus ──
    # Each worker initialises LoRA A randomly and independently, so replicas
    # start genuinely diverged — round 1 is a real averaging test, not a
    # merge of already-identical tensors.
    print(f"\n{'='*70}\nSTEP 3: Initial averaging round\n{'='*70}")
    result = check_round_arithmetic(get_registry(), "round-1")
    print(f"  Round {result['round_id']}: completed={result['completed']}")
    check(len(result["completed"]) == NUM_WORKERS,
          f"all {NUM_WORKERS} workers completed round 1 "
          f"(skipped: {result['skipped']})")
    check_consensus(get_registry(), "post-round-1")

    # ── Step 4: replica routing ──
    print(f"\n{'='*70}\nSTEP 4: Replica routing\n{'='*70}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    samples = []
    for prompt_text, answer_text in FACTS:
        input_ids = tok(prompt_text + answer_text, return_tensors="pt").input_ids.to(torch.long)
        prompt_len = len(tok(prompt_text, return_tensors="pt").input_ids[0])
        labels = input_ids.clone()
        labels[0, :prompt_len] = -100
        samples.append(safetensors_save({"input_ids": input_ids, "labels": labels}))

    post_train_step(samples[0], "probe-rmax", replica_id=R + 3, expect_status=409)
    check(True, f"X-Replica-ID {R + 3} (beyond R={R}) correctly returns 409")

    # ── Step 5: training with rounds every X steps per replica ──
    print(f"\n{'='*70}\nSTEP 5: Training ({PRE_KILL_EPOCHS} epochs, X={X})\n{'='*70}")
    step = 0
    epoch_losses: list[float] = []
    for epoch in range(PRE_KILL_EPOCHS):
        losses = []
        # Split the 4 samples across the 2 replicas; swap halves each epoch
        # so both replicas see all facts over time (poor man's shared queue).
        half = [(0, 1), (2, 3)] if epoch % 2 == 0 else [(2, 3), (0, 1)]
        for replica_id, idxs in enumerate(half):
            for i in idxs:
                out = post_train_step(samples[i], f"r{replica_id}-s{step}-{i}", replica_id)
                losses.append(out["loss"].item())
        step += X  # each replica did X=2 steps this epoch
        # Verify the arithmetic again on the last epoch, where the replicas
        # have had the most opportunity to drift apart between merges.
        if epoch == PRE_KILL_EPOCHS - 1:
            result = check_round_arithmetic(get_registry(), f"epoch-{epoch + 1}")
        else:
            result = run_average_round()
        avg = sum(losses) / len(losses)
        epoch_losses.append(avg)
        print(f"  Epoch {epoch + 1:2d} | avg loss {avg:7.4f} | "
              f"round {result['round_id']} completed={len(result['completed'])}")

    check_consensus(get_registry(), "post-training")

    # ── Step 7: kill an active worker ──
    print(f"\n{'='*70}\nSTEP 7: Kill + degradation\n{'='*70}")
    workers = get_registry()
    if has_spare:
        spare_name = next(w["node_name"] for w in workers if w["role"] == "spare")
        spare_stage = tuple(next(w for w in workers if w["role"] == "spare")["layers"])
        victim = next(w for w in workers
                      if w["role"] == "active" and tuple(w["layers"]) == spare_stage)
        expected_R = R          # spare absorbs the loss
    else:
        spare_name = None
        # Kill a replica-1 worker so replica 0 survives and keeps training.
        victim = next(w for w in workers
                      if w["role"] == "active" and w["replica_id"] == 1)
        expected_R = R - 1      # no spare: the mesh degrades

    container = find_container(victim["node_name"])
    print(f"  Killing {victim['node_name']} (container {container}, "
          f"replica {victim['replica_id']}, layers {victim['layers']}); "
          + (f"spare {spare_name} should be promoted, R stays {R}."
             if has_spare else
             f"no spare — R should drop {R} -> {expected_R}."))
    stop = subprocess.run(docker() + ["stop", "-t", "0", container],
                          capture_output=True, text=True, timeout=60)
    if stop.returncode != 0:
        fail(f"`docker stop {container}` failed (rc={stop.returncode}): "
             f"{stop.stderr.strip()}")
    check(True, f"container {container} stopped")

    # The gateway must actually notice before round membership means anything.
    wait_for_eviction(victim["node_name"])

    # Replicas below the new R must keep training; those at or above it must
    # cleanly report 409 so a harness consumer knows to return its batch.
    losses = []
    saw_409 = False
    for epoch in range(POST_KILL_EPOCHS):
        half = [(0, 1), (2, 3)] if epoch % 2 == 0 else [(2, 3), (0, 1)]
        for replica_id, idxs in enumerate(half):
            for i in idxs:
                rid = f"postkill-r{replica_id}-s{step}-{i}"
                if replica_id < expected_R:
                    out = post_train_step(samples[i], rid, replica_id)
                    losses.append(out["loss"].item())
                else:
                    post_train_step(samples[i], rid, replica_id, expect_status=409)
                    saw_409 = True
        step += X
        result = run_average_round()
        avg = sum(losses[-2 * expected_R:]) / max(1, len(losses[-2 * expected_R:]))
        epoch_losses.append(avg)
        print(f"  Epoch {PRE_KILL_EPOCHS + epoch + 1:2d} | avg loss {avg:7.4f} | "
              f"round {result['round_id']} completed={len(result['completed'])}")
        check(victim["node_name"] not in result["completed"],
              "dead worker excluded from round")
        check(len(result["completed"]) == NUM_WORKERS - 1,
              f"{NUM_WORKERS - 1} surviving workers completed the round")

    workers = get_registry()
    live_actives = [w for w in workers if w["role"] == "active" and w["healthy"]]
    new_R = len({w["replica_id"] for w in live_actives})
    check(new_R == expected_R, f"R is {expected_R} after the kill (got {new_R})")

    if has_spare:
        promoted = next((w for w in workers if w["node_name"] == spare_name), None)
        check(promoted is not None and promoted["role"] == "active",
              f"spare {spare_name} promoted to active")
    else:
        check(saw_409, "retired replica cleanly returns 409 rather than hanging")
        # §3.10: the dead worker's stage peers must NOT be stranded — they
        # rejoin their layer-peer groups as spares and keep receiving averages.
        orphans = [w for w in workers
                   if w["healthy"] and w["role"] == "spare"]
        check(len(orphans) == NUM_WORKERS - 1 - expected_R * num_stages,
              f"orphaned peers rejoined as spares, not stranded (got "
              f"{[o['node_name'] for o in orphans]})")

    check_consensus(workers, "post-kill")

    # ── Step 8: sanity — learning happened, VRAM is measured ──
    print(f"\n{'='*70}\nSTEP 8: Sanity checks\n{'='*70}")
    check(epoch_losses[-1] < epoch_losses[0],
          f"loss decreased ({epoch_losses[0]:.4f} -> {epoch_losses[-1]:.4f})")

    survivor = next(w["node_name"] for w in workers
                    if w["healthy"] and w["node_name"] != victim["node_name"])
    r = httpx.get(f"http://localhost:{worker_port(survivor)}/health", timeout=5.0)
    gpu = r.json().get("gpu", {})
    if gpu.get("device") == "cuda":
        check(gpu.get("vram_peak_mb", 0) > 0,
              f"peak VRAM reported: {gpu.get('vram_peak_mb')} MB (§3.8)")
    else:
        print("  [SKIP] CPU worker — no VRAM peak to check")

    print(f"\n{'='*70}")
    print("STAGE 6 MESH TEST PASSED — R pipes, averaging, and "
          + ("spare promotion" if has_spare else "graceful R-degradation")
          + " all work!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
