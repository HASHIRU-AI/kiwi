#!/usr/bin/env python3
"""Phase 0b: why is the SAME merge 50 ms in one arm and 3,816 ms in another?

Phase 0 (scratch/bench_merge.py) falsified both branches of MERGE_PERF_PLAN.md
§3's fork in the road. On a 20-core box an isolated dense merge is 10 ms and
seven concurrent ones are 40 ms. Neither is 3,300 ms. So the plan's diagnosis
-- "the math itself" or "plain thread oversubscription" -- does not survive
contact with a microbenchmark, and the real cause had to come from the v4
container logs instead.

What the logs say (benchmark_results_v4/.../container_logs, 14.68 MB workers,
p50 over 120 round-records each):

    field              dp2_alpaca   dp2_sparse5
    to_cpu_ms                 3.8           3.8
    serialize_ms              8.8           8.6
    snapshot_lock_ms         50.8          50.7
    apply_ms                 25.6          16.1
    merge_compute_ms      3,816.1          50.8     <-- 75x, everything else equal

Both numbers are merge_adapters, R=2, on an identically sized adapter, in the
same image on the same box. Every other phase of the round matches to within
noise -- including to_cpu_ms, which allocates and fills the very same ~14.7 MB
of fresh CPU tensors the merge then reads. That rules out memory pressure,
page-fault cost, allocator churn and "the box was busy": those would show up
in to_cpu_ms too.

The one structural difference between the arms is HOW LONG THE PROCESS SAT
WITHOUT DOING CPU TENSOR WORK before the merge:

    dp2_alpaca : X=10, 30 rounds  -> ~10 training steps (~3.2 s) between rounds
    dp2_sparse5: X=1, 300 rounds  -> ~1 training step  (~0.32 s) between rounds

That points at the OpenMP thread team, not at arithmetic. libgomp's idle
workers spin for GOMP_SPINCOUNT and then block on a futex; waking a blocked
team, or building a new one, costs a fixed amount PER PARALLEL REGION, and
ATen opens a parallel region per elementwise op above its ~32,768-element
grain size. A merge over ~112 tensors opens hundreds of them. If each one
pays a team wake-up, hundreds x milliseconds is exactly the observed cost --
and a 0.32 s gap keeps the team spinning (cheap) while a 3.2 s gap lets it
sleep (expensive).

Two corroborating details from the same logs:
  * The DiLoCo ANCHOR round does no arithmetic at all -- it is
    `{k: v.clone() for k, v in merged.items()}` -- and costs 474-750 ms.
    A 14.7 MB memcpy is ~2 ms. clone() is OpenMP-parallel; to_cpu is not.
  * In dp2_sparse5 the p=0.05 SPARSE rounds (100 ms) cost MORE than the dense
    full-sync rounds (50 ms) despite touching 20x fewer elements, because the
    masked tensors fall below the grain size and never go parallel at all.

This script tests that mechanism directly. It is the reproducer Phase 0 was
missing: an idle gap before the merge, and the merge running on a thread pool
worker the way _apply_round_merge actually does (asyncio.to_thread).

    python scratch/bench_merge_idle.py

Falsifiable predictions, if the OpenMP-team story is right:
  1. `gap=3.0s` is much slower than `gap=0.0s` at default threads.
  2. `threads=1` and `threads=2` are nearly gap-INSENSITIVE (a 1-2 thread team
     is trivial to wake, and at 1 thread ATen skips the parallel region).
  3. `OMP_WAIT_POLICY=active` / `GOMP_SPINCOUNT=infinite` removes the gap
     penalty at default threads without capping threads.
  4. A fresh pool thread per call is worse still: a new OS thread means a
     brand-new team rather than a sleeping one.
If (1) does not reproduce, the mechanism is something else again and the next
step is a Tier-2 local cluster run, not more microbenchmarks.

OUTCOME: (1) DID NOT REPRODUCE. This hypothesis is dead; the script is kept as
the record of it. On the target box: gap 0.0s = 26.8 ms, gap 3.0s = 23.2 ms
(0.9x), and OMP_WAIT_POLICY=active / GOMP_SPINCOUNT=infinite were both no-ops.
Idle time before a merge is irrelevant.

The one real signal here was the thread-pool penalty -- pool 44-48 ms and
fresh_pool 51 ms against 23-27 ms on the main thread, which does apply to
_apply_round_merge running under asyncio.to_thread. But 2x is not 152x.

The actual cause (MERGE_PERF_FINDINGS.md) was thread oversubscription x
per-tensor op count, and THIS SCRIPT COULD NOT HAVE FOUND IT: it runs a single
process, and the effect needs 7 concurrent processes each with a 30-thread team.
Its own "go to a Tier-2 cluster run" advice was the correct next step -- what
actually settled it was A/B-ing scratch/mesh_v4_baseline.py against mesh.py in
the live cluster, which is cheap because mesh.py is bind-mounted.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Byte-matched to the largest real worker in the v4 logs
# (payload_bytes=14,681,920 bf16). Phase 0 used 8 layers / 5.87 MB, which
# matches the SMALLER stage -- and the plan's §1 quoted ~11.7 MB, the average
# of the two, which is a size no worker actually has.
LAYERS_FOR_14MB = 20
LAYERS_FOR_6MB = 8


def _adapter(layers: int, dtype: str = "bfloat16"):
    import torch

    from bench_merge import build_adapter, lora_shapes  # noqa: F401

    import bench_merge

    cfg = dict(bench_merge.QWEN_0_5B)
    cfg["layers_per_worker"] = layers
    g = torch.Generator().manual_seed(0)
    shapes = bench_merge.lora_shapes(cfg)
    return {
        key: (torch.randn(*shape, generator=g, dtype=torch.float32) * 0.01).to(
            getattr(torch, dtype)
        )
        for key, shape in shapes
    }


# ── The child process: one (threads, gap, mode) cell of the matrix ───────────


def run_cell(threads: int, gap: float, mode: str, layers: int, reps: int) -> dict:
    import torch

    torch.set_num_threads(threads)
    from mesh import merge_adapters

    own = _adapter(layers)
    peer = {k: v.clone() for k, v in own.items()}
    received = [("w-b", 10, peer)]
    n_params = sum(t.numel() for t in own.values())
    n_bytes = sum(t.numel() * t.element_size() for t in own.values())

    def one_merge():
        return merge_adapters(own, received, contribute=True, own_name="w-a")

    def busywork():
        """Stand-in for the training steps that fill the gap in a real run.

        Deliberately NOT a sleep alone: a real worker is doing GPU work and
        Python bookkeeping in that window, which is what lets the CPU-side
        OpenMP team go idle. A pure sleep would leave the team idle too, but
        this also rules out "the process was frozen" as an explanation.
        """
        deadline = time.perf_counter() + gap
        while time.perf_counter() < deadline:
            sum(range(10_000))

    samples: list[float] = []
    if mode == "main":
        for _ in range(reps + 1):
            if gap:
                busywork()
            t0 = time.perf_counter()
            one_merge()
            samples.append((time.perf_counter() - t0) * 1000)

    elif mode in ("pool", "fresh_pool"):
        async def driver():
            shared = ThreadPoolExecutor(max_workers=4)
            loop = asyncio.get_running_loop()
            for _ in range(reps + 1):
                if gap:
                    busywork()
                if mode == "fresh_pool":
                    # A brand-new OS thread per call. _apply_round_merge runs on
                    # asyncio's default pool, which rotates threads across the
                    # round's three to_thread calls (to_cpu, serialize, merge),
                    # so "the merge lands on a thread that has never opened a
                    # parallel region" is a real state, not a synthetic one.
                    ex = ThreadPoolExecutor(max_workers=1)
                    t0 = time.perf_counter()
                    await loop.run_in_executor(ex, one_merge)
                    samples.append((time.perf_counter() - t0) * 1000)
                    ex.shutdown(wait=True)
                else:
                    t0 = time.perf_counter()
                    await loop.run_in_executor(shared, one_merge)
                    samples.append((time.perf_counter() - t0) * 1000)
            shared.shutdown(wait=True)

        asyncio.run(driver())
    else:
        raise SystemExit(f"unknown mode {mode}")

    samples = samples[1:]  # drop the first: it builds the team rather than waking it
    return {
        "threads": threads,
        "gap_s": gap,
        "mode": mode,
        "params": n_params,
        "bytes": n_bytes,
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "omp_wait_policy": os.getenv("OMP_WAIT_POLICY", ""),
        "gomp_spincount": os.getenv("GOMP_SPINCOUNT", ""),
        "torch_threads_effective": torch.get_num_threads(),
    }


# ── The parent: spawn one child per cell so env vars can differ ──────────────


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--layers", type=int, default=LAYERS_FOR_14MB)
    ap.add_argument("--json", default=None)
    # child-mode arguments
    ap.add_argument("--cell", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.cell:
        threads, gap, mode = args.cell.split(",")
        print(json.dumps(run_cell(int(threads), float(gap), mode, args.layers, args.reps)))
        return 0

    import torch

    print("=" * 78)
    print("Phase 0b: does an IDLE GAP before the merge reproduce the 3.8 s cost?")
    print("=" * 78)
    print(f"torch {torch.__version__} | {platform.system()} | {os.cpu_count()} cpus | "
          f"default threads {torch.get_num_threads()}")
    own = _adapter(args.layers)
    n_params = sum(t.numel() for t in own.values())
    n_bytes = sum(t.numel() * t.element_size() for t in own.values())
    print(f"adapter: {len(own)} tensors, {n_params:,} params, {n_bytes:,} bytes")
    print("  (v4's largest worker: payload_bytes=14,681,920 — this is byte-matched)")
    print(f"reps={args.reps} per cell, first call discarded\n")
    del own

    default = torch.get_num_threads()
    cells: list[tuple[int, float, str, dict]] = []
    for threads in sorted({1, 2, default}):
        for gap in (0.0, 0.3, 3.0):
            cells.append((threads, gap, "main", {}))
    # thread-pool variants only where they can matter
    for mode in ("pool", "fresh_pool"):
        for gap in (0.0, 3.0):
            cells.append((default, gap, mode, {}))
    # env fixes that do NOT cap threads — prediction 3
    for env in ({"OMP_WAIT_POLICY": "active"}, {"GOMP_SPINCOUNT": "infinite"}):
        cells.append((default, 3.0, "main", env))

    results = []
    print(f"{'threads':>8} {'gap_s':>6} {'mode':>11} {'env':>26} {'median_ms':>10} {'min':>9}")
    for threads, gap, mode, env in cells:
        child_env = dict(os.environ)
        child_env.update({k: str(v) for k, v in env.items()})
        # Must be set before torch imports for libgomp to honour them.
        child_env.setdefault("OMP_NUM_THREADS", str(threads))
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__),
             "--cell", f"{threads},{gap},{mode}",
             "--layers", str(args.layers), "--reps", str(args.reps)],
            capture_output=True, text=True, env=child_env,
        )
        if proc.returncode != 0:
            print(f"  cell {threads},{gap},{mode} FAILED:\n{proc.stderr[-1500:]}")
            continue
        row = json.loads(proc.stdout.strip().splitlines()[-1])
        row["env"] = env
        results.append(row)
        label = ",".join(f"{k}={v}" for k, v in env.items()) or "-"
        print(f"{threads:>8} {gap:>6.1f} {mode:>11} {label:>26} "
              f"{row['median_ms']:>10.1f} {row['min_ms']:>9.1f}")

    print("\n" + "=" * 78)
    print("READING THIS")
    print("=" * 78)

    def find(threads, gap, mode="main", env=None):
        for r in results:
            if (r["threads"] == threads and abs(r["gap_s"] - gap) < 1e-9
                    and r["mode"] == mode and r["env"] == (env or {})):
                return r["median_ms"]
        return None

    warm = find(default, 0.0)
    cold = find(default, 3.0)
    if warm and cold:
        print(f"  default threads: gap 0.0s = {warm:.1f} ms, gap 3.0s = {cold:.1f} ms "
              f"-> {cold / warm:.1f}x")
        print("     >5x reproduces the alpaca/sparse5 split -> OpenMP team wake-up.")
        print("     ~1x means the gap is irrelevant; go to a Tier-2 cluster run.")
    for t in (1, 2):
        w, c = find(t, 0.0), find(t, 3.0)
        if w and c:
            print(f"  threads={t}: gap 0.0s = {w:.1f} ms, gap 3.0s = {c:.1f} ms "
                  f"-> {c / w:.1f}x  (should be ~1x: Phase 1A works)")
    for env in ({"OMP_WAIT_POLICY": "active"}, {"GOMP_SPINCOUNT": "infinite"}):
        v = find(default, 3.0, "main", env)
        if v and cold:
            label = ",".join(f"{k}={x}" for k, x in env.items())
            print(f"  {label}: {v:.1f} ms vs {cold:.1f} ms unfixed "
                  f"-> {cold / v:.1f}x better")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
