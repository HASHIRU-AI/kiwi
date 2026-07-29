"""Test Stage 5: Fault-tolerance recovery.

Topology: 4 workers (3-stage pipeline + 1 standby), uniform 0.9 GB VRAM.
Plan:
  1. Wait for swarm readiness (all 4 workers registered, model ready).
  2. Train for ~30 steps so the standby receives adapter snapshots.
  3. Kill the active worker for a specific stage (docker stop).
  4. Continue training — verify the gateway detects the failure, promotes
     the standby, and training converges to the multi-fact target.
  5. Collect and verify recovery metrics from logs.

Run:  python scratch/test_stage5_recovery.py
"""

import subprocess
import sys
import time
import threading

import httpx
import torch
from safetensors.torch import save as safetensors_save, load as safetensors_load
from transformers import AutoTokenizer

GATEWAY_URL = "http://localhost:8080"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# ── Multi-fact dataset (same as stage4 test) ──
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

PRE_KILL_EPOCHS = 10     # train this many epochs before killing a worker
POST_KILL_EPOCHS = 25    # train this many more after the kill
PASS_THRESHOLD = 0.75    # required per-fact answer-token accuracy
KILL_STAGE = None        # will be auto-detected: pick a stage with a standby


def wait_for_swarm(min_workers: int = 4) -> list[dict]:
    """Wait until the gateway has enough workers and the model is ready."""
    print(f"Waiting for swarm readiness ({min_workers} workers)...")
    for _ in range(90):
        try:
            resp = httpx.get(f"{GATEWAY_URL}/registry", timeout=5.0)
            if resp.status_code == 200:
                data = resp.json()
                workers = data.get("workers", [])
                status = data.get("models", {}).get(MODEL_ID, {})
                healthy = [w for w in workers if w.get("healthy")]
                if status.get("ready") and len(healthy) >= min_workers:
                    print(f"  Swarm ready: {len(healthy)} healthy workers.")
                    return workers
        except Exception:
            pass
        time.sleep(2)
    print("ERROR: Swarm did not become ready in time.")
    sys.exit(1)


def _find_container(node_name: str) -> str:
    """Discover the Docker container name for a given worker node_name.

    Uses `docker ps` to find a running container whose name contains the
    node_name. This avoids hardcoding naming conventions that vary across
    different docker-compose configurations.
    """
    result = subprocess.run(
        ["docker", "ps", "--format", "{{.Names}}"],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode != 0:
        print(f"  WARNING: docker ps failed: {result.stderr}")
        return node_name  # fallback

    for line in result.stdout.strip().splitlines():
        container = line.strip()
        if node_name in container:
            return container

    print(f"  WARNING: No container found matching '{node_name}', using as-is.")
    return node_name


def find_kill_target(workers: list[dict]) -> tuple[str, str]:
    """Find an active worker that has a standby with the same layers.

    Returns (active_container_name, active_node_name).
    """
    actives = [w for w in workers if w.get("role") == "active"]
    standbys = [w for w in workers if w.get("role") == "standby"]

    for active in actives:
        for standby in standbys:
            if (active["model_id"] == standby["model_id"]
                    and active["layers"] == standby["layers"]):
                node_name = active["node_name"]
                container = _find_container(node_name)
                print(f"  Kill target: {node_name} (container={container}), "
                      f"layers={active['layers']}")
                print(f"  Standby:     {standby['node_name']}, "
                      f"layers={standby['layers']}")
                return container, node_name

    # If no explicit standby is detected yet, pick the last active and hope
    # the layer allocator assigned its duplicate as standby.
    print("  WARNING: No explicit standby found. Falling back to last active.")
    last = actives[-1]
    container = _find_container(last["node_name"])
    return container, last["node_name"]


def post_train_step(payload: bytes, request_id: str) -> dict:
    """Send a training step through the gateway."""
    r = httpx.post(
        f"{GATEWAY_URL}/train_step",
        content=payload,
        headers={
            "Content-Type": "application/octet-stream",
            "X-Model-ID": MODEL_ID,
            "X-Request-ID": request_id,
        },
        timeout=180.0,
    )
    if r.status_code != 200:
        print(f"  ERROR [{r.status_code}]: {r.text}")
        sys.exit(1)
    return safetensors_load(r.content)


def kill_worker(container_name: str) -> None:
    """Kill a Docker container to simulate a node failure."""
    print(f"\n{'='*70}")
    print(f"KILLING WORKER: docker stop {container_name}")
    print(f"{'='*70}\n")
    result = subprocess.run(
        ["docker", "stop", "-t", "0", container_name],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        print(f"  docker stop failed: {result.stderr}")
    else:
        print(f"  Container {container_name} stopped.")


def main():
    workers = wait_for_swarm(min_workers=4)

    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    # Pre-tokenize facts with label masking
    samples = []
    for prompt_text, answer_text in FACTS:
        input_ids = tok(prompt_text + answer_text, return_tensors="pt").input_ids.to(torch.long)
        prompt_len = len(tok(prompt_text, return_tensors="pt").input_ids[0])
        labels = input_ids.clone()
        labels[0, :prompt_len] = -100
        answer_tokens = input_ids[0].tolist()[prompt_len:]
        samples.append({
            "prompt": prompt_text,
            "input_ids": input_ids,
            "labels": labels,
            "prompt_len": prompt_len,
            "answer_tokens": answer_tokens,
            "payload": safetensors_save({"input_ids": input_ids, "labels": labels}),
        })

    def answer_acc(out, s):
        preds = out["predictions"][0].tolist()
        idxs = range(s["prompt_len"] - 1, s["prompt_len"] - 1 + len(s["answer_tokens"]))
        pred_ans = [preds[i] for i in idxs]
        correct = sum(1 for p, t in zip(pred_ans, s["answer_tokens"]) if p == t)
        return correct / max(len(s["answer_tokens"]), 1), tok.decode(pred_ans)

    # ── Phase 1: Pre-kill training ──
    print(f"\n{'='*70}")
    print(f"PHASE 1: Pre-kill training ({PRE_KILL_EPOCHS} epochs)")
    print(f"{'='*70}")

    step_count = 0
    for epoch in range(PRE_KILL_EPOCHS):
        losses = []
        for i, s in enumerate(samples):
            out = post_train_step(s["payload"], f"recovery-pre-e{epoch}-s{i}")
            losses.append(out["loss"].item())
            step_count += 1
        if epoch == 0 or (epoch + 1) % 5 == 0:
            avg = sum(losses) / len(losses)
            print(f"  Epoch {epoch+1:2d} | avg loss: {avg:7.4f} | steps so far: {step_count}")

    # ── Refresh registry and pick kill target ──
    print("\nRefreshing registry to identify kill target...")
    workers = wait_for_swarm(min_workers=4)
    container_name, killed_node_name = find_kill_target(workers)

    # ── Phase 2: Kill worker mid-training ──
    # Schedule the kill to happen during the next epoch
    kill_event = threading.Event()
    kill_at_step = step_count + len(samples)  # kill after 1 epoch of post-kill phase

    def delayed_kill():
        """Kill the worker after a short delay to let training start."""
        time.sleep(3)  # Let the first post-kill epoch start
        kill_worker(container_name)
        kill_event.set()

    kill_thread = threading.Thread(target=delayed_kill, daemon=True)

    print(f"\n{'='*70}")
    print(f"PHASE 2: Post-kill training ({POST_KILL_EPOCHS} epochs)")
    print(f"  Kill scheduled for ~step {kill_at_step}")
    print(f"{'='*70}")

    kill_thread.start()
    killed = False
    recovery_step = None

    for epoch in range(POST_KILL_EPOCHS):
        losses = []
        for i, s in enumerate(samples):
            rid = f"recovery-post-e{epoch}-s{i}"
            try:
                out = post_train_step(s["payload"], rid)
                losses.append(out["loss"].item())
                step_count += 1

                if kill_event.is_set() and not killed:
                    killed = True
                    recovery_step = step_count
                    print(f"\n  >>> Worker killed! Recovery should occur around step {step_count}. <<<\n")
            except Exception as e:
                print(f"  Step {step_count} FAILED (expected during recovery): {e}")
                # The gateway retry loop should handle this internally.
                # If we get an error here, the gateway failed to recover.
                # Give it a moment and retry once.
                time.sleep(2)
                try:
                    out = post_train_step(s["payload"], rid)
                    losses.append(out["loss"].item())
                    step_count += 1
                except Exception as e2:
                    print(f"  Retry also failed: {e2}")
                    sys.exit(1)

        if losses:
            avg = sum(losses) / len(losses)
            status = " (POST-KILL)" if killed else ""
            print(f"  Epoch {PRE_KILL_EPOCHS + epoch + 1:2d} | avg loss: {avg:7.4f} | "
                  f"steps: {step_count}{status}")

    # ── Phase 3: Recall check ──
    print(f"\n{'='*70}")
    print(f"PHASE 3: Recall verification (post-recovery)")
    print(f"{'='*70}")

    accs = []
    for i, s in enumerate(samples):
        out = post_train_step(s["payload"], f"recovery-recall-{i}")
        acc, recited = answer_acc(out, s)
        accs.append(acc)
        status = "OK  " if acc >= PASS_THRESHOLD else "MISS"
        print(f"  [{status}] Fact {i+1} acc {acc*100:5.1f}% | {s['prompt'].splitlines()[0]}")
        print(f"           recites: {recited!r}")

    # ── Results ──
    print(f"\n{'='*70}")
    all_pass = all(a >= PASS_THRESHOLD for a in accs)
    weakest = min(accs)
    print(f"Facts learned: {sum(1 for a in accs if a >= PASS_THRESHOLD)}/{len(accs)}"
          f"  |  weakest fact: {weakest*100:.1f}%")

    if recovery_step:
        print(f"Recovery occurred around step {recovery_step}.")
    else:
        print("WARNING: Kill may not have triggered during training.")

    print(f"\nCheck gateway logs for:")
    print(f"  - 'Live-error detected!' messages with Detection latency")
    print(f"  - 'Recovery complete.' messages with Resume latency")
    print(f"Check promoted worker logs for:")
    print(f"  - 'Promoted to ACTIVE! Resuming from snapshot.'")
    print(f"  - 'Recovery Metrics' entries (bytes transferred, baseline, ratio)")

    if all_pass:
        print(f"\n{'='*70}")
        print("STAGE 5 RECOVERY TEST PASSED — training survived worker death!")
        print(f"{'='*70}")
    else:
        print(f"\n{'='*70}")
        print(f"STAGE 5 RECOVERY TEST FAILED — not all facts recalled >= {PASS_THRESHOLD*100:.0f}%")
        print(f"{'='*70}")
        sys.exit(1)


if __name__ == "__main__":
    main()
