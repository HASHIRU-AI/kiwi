#!/usr/bin/env python3
"""Phase 0: why is a DENSE merge ~3.3 s when the arithmetic says ~30 ms?

Read MERGE_PERF_PLAN.md §3 first. This script is the fork in the road: it
decides whether the fix is a one-line thread cap (contention) or a rewrite of
the merge arithmetic (compute).

What the v4 profile already rules out
-------------------------------------
Not the network (serialize_ms = 8), not the barrier (wait_ms = 84 of 4142), and
not the ~112 synchronising device-to-host copies -- those were already fixed in
Phase 2 by hoisting one bulk `.cpu()` pass into _run_round (`own_cpu`), which
took the SPARSE round from 4,371 ms to 457 ms. The dense round barely moved
(3,442 -> 3,267 ms), so for dense the transfer was never the dominant term.

The shape of the remaining cost, from the v4 numbers
---------------------------------------------------
    dense  : ~12M element-ops in 3,267 ms  ->  ~3.6 Melem-ops/s
    sparse : ~0.6M element-ops in    92 ms ->  ~6.4 Melem-ops/s

Both paths touch the SAME ~112 tensors and differ ~20x in elements, yet land
within 2x of each other on per-element throughput. That is the interesting
fact, and it discriminates the hypotheses:

  * If per-TENSOR framework overhead dominated, sparse (same tensor count,
    1/20th the elements) would cost about the same as dense. It costs 1/35th.
  * So cost tracks ELEMENTS, at a per-element rate ~100x below what a single
    core should manage.

The leading explanation, and the one this script is built to confirm or kill:
ATen parallelises an elementwise op only above an internal grain size
(~32,768 elements). The sparse path's masked tensors sit BELOW it and run
single-threaded and fast; the dense path's tensors sit ABOVE it, so each op
forks an OpenMP team. With 7 worker processes each defaulting to
os.cpu_count() threads, every one of those fork/join barriers is contended.
Same code, same tensors -- only the dense path pays.

If that is right, `torch.set_num_threads(2)` in the worker is the entire fix
and no GPU rewrite is needed (plan §4.1A, §3 "do not do a GPU rewrite you do
not need").

Sections
--------
  1. per-call totals for merge_adapters (R=2, R=3), merge_sparse(p), outer_step
  2. per-phase breakdown: convert / accumulate / divide+cast
  3. thread sweep: 1, 2, 4, default
  4. grain-size probe: elementwise add just below vs just above ATen's
     parallel threshold, at 1 thread vs default -- direct test of the
     mechanism above
  5. N concurrent processes doing the same merge (mimics the barrier)

Usage
-----
    python scratch/bench_merge.py                    # single process
    python scratch/bench_merge.py --procs 7          # match the real cluster
    python scratch/bench_merge.py --json out.json    # machine-readable

RESOLVED -- READ THIS BEFORE TRUSTING A NEGATIVE RESULT FROM THIS SCRIPT.
See MERGE_PERF_FINDINGS.md for the full account.

The cause was thread oversubscription after all, but this script UNDERSTATED it
by two orders of magnitude and the wrong conclusion was drawn from it. The
pathology needs THREE ingredients at once:

    many tensors  x  many threads per process  x  many processes

    this script, --procs 7 on a 20-core laptop (14 threads):      40 ms
    single process on the 30-core target box (30 threads):        25 ms
    the real cluster: 7 processes x 30 threads on 30 cores:    3,800 ms

Barrier cost scales superlinearly in TEAM SIZE, so 7x14 threads on 20 cores
looks like a negligible 3.9x while 7x30 on 30 cores is 400x. Section 5 is
therefore only meaningful when run ON THE TARGET HOST with --procs matching the
real worker count. A cheap result here does NOT mean the merge is cheap in the
cluster.

What this script did get right, and why it is kept: section 4's grain-size
probe identified the mechanism (4.1 us at 1 thread vs 45.9 us at 14 threads for
32,769 elements; 1.00x at 32,767). That fixed per-region cost, multiplied by the
~280 elementwise ops of a 140-tensor R=2 merge, is the whole story.

Note also that the real adapter is fp32, not bf16 -- pass --dtype float32 to
match it. The default here is bf16, which halves the bytes.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import platform
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

# ── Real LoRA shapes ────────────────────────────────────────────────────────
# Qwen2.5-0.5B: hidden 896, 24 layers, 14 heads x 64, 2 KV heads (kv dim 128),
# intermediate 4864. A dp2 worker owns one of 3 stages -> 8 layers, and PEFT
# emits two tensors (A, B) per target module, so 8 x 7 x 2 = 112 tensors --
# matching the 112 seen in the profile.
QWEN_0_5B = {
    "hidden": 896,
    "kv_dim": 128,
    "intermediate": 4864,
    "layers_per_worker": 8,
}


def lora_shapes(cfg: dict = QWEN_0_5B, r: int = 16) -> list[tuple[str, tuple[int, int]]]:
    """(key, shape) for one worker's slice, in PEFT's naming/ordering."""
    h, kv, inter = cfg["hidden"], cfg["kv_dim"], cfg["intermediate"]
    # (module, in_features, out_features)
    targets = [
        ("q_proj", h, h),
        ("k_proj", h, kv),
        ("v_proj", h, kv),
        ("o_proj", h, h),
        ("gate_proj", h, inter),
        ("up_proj", h, inter),
        ("down_proj", inter, h),
    ]
    out: list[tuple[str, tuple[int, int]]] = []
    for layer in range(cfg["layers_per_worker"]):
        for mod, fan_in, fan_out in targets:
            base = f"base_model.model.model.layers.{layer}.self_attn.{mod}"
            out.append((f"{base}.lora_A.weight", (r, fan_in)))
            out.append((f"{base}.lora_B.weight", (fan_out, r)))
    return out


def build_adapter(dtype: torch.dtype, seed: int = 0, r: int = 16) -> dict[str, torch.Tensor]:
    """A synthetic adapter state dict with the REAL per-tensor shapes.

    Shapes matter more than values here: the whole grain-size hypothesis is
    about individual tensor sizes, so a benchmark over 112 equal-sized blobs
    would measure the wrong thing.
    """
    g = torch.Generator().manual_seed(seed)
    return {
        key: (torch.randn(*shape, generator=g, dtype=torch.float32) * 0.01).to(dtype)
        for key, shape in lora_shapes(r=r)
    }


def adapter_stats(sd: dict[str, torch.Tensor]) -> tuple[int, int, int]:
    numels = [t.numel() for t in sd.values()]
    return len(numels), sum(numels), sum(t.numel() * t.element_size() for t in sd.values())


# ── Timing helper ───────────────────────────────────────────────────────────


def timeit(fn, reps: int = 20, warmup: int = 3) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000)
    samples.sort()
    return {
        "min_ms": samples[0],
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "max_ms": samples[-1],
    }


# ── Section 1: the real functions ───────────────────────────────────────────


def bench_functions(dtype: torch.dtype, p: float, reps: int) -> dict[str, dict[str, float]]:
    from mesh import merge_adapters, merge_sparse, outer_step, sparse_snapshot

    own = build_adapter(dtype, seed=0)
    peers = [build_adapter(dtype, seed=i + 1) for i in range(3)]
    r2 = [("w-b", 10, peers[0])]
    r3 = [("w-b", 10, peers[0]), ("w-c", 10, peers[1])]

    sparse_recv = [
        (name, 10, sparse_snapshot(peer, 7, p))
        for name, peer in (("w-b", peers[0]), ("w-c", peers[1]))
    ]

    merged = merge_adapters(own, r2, contribute=True, own_name="w-a")
    assert merged is not None
    anchor = build_adapter(dtype, seed=99)

    out: dict[str, dict[str, float]] = {}
    out["merge_adapters R=2"] = timeit(
        lambda: merge_adapters(own, r2, contribute=True, own_name="w-a"), reps
    )
    out["merge_adapters R=3"] = timeit(
        lambda: merge_adapters(own, r3, contribute=True, own_name="w-a"), reps
    )
    out[f"merge_sparse p={p:g} R=3"] = timeit(
        lambda: merge_sparse(own, sparse_recv, True, round_id=7, p=p, own_name="w-a"), reps
    )
    out["sparse_snapshot"] = timeit(lambda: sparse_snapshot(own, 7, p), reps)
    # outer_step mutates buf; a fresh dict per call keeps reps comparable and
    # measures the cold path (first round), which is the expensive one.
    out["outer_step"] = timeit(lambda: outer_step(anchor, merged, {}, 0.7, 0.9), reps)
    out["outer_step (warm buf)"] = timeit(
        lambda buf={}: outer_step(anchor, merged, buf, 0.7, 0.9), reps
    )
    return out


# ── Section 2: where inside the merge the time goes ─────────────────────────


def bench_phases(dtype: torch.dtype, reps: int) -> dict[str, dict[str, float]]:
    """Replicate merge_adapters' inner loop, phase by phase.

    Mirrors the CURRENT implementation exactly:
        acc = {k: zeros(fp32)}                      # allocate
        for contribution: acc[k] += sd[k].float()   # convert + accumulate
        {k: (acc[k]/n).to(dtype)}                   # divide + cast
    """
    own = build_adapter(dtype, seed=0)
    peer = build_adapter(dtype, seed=1)
    keys = sorted(own.keys())
    n = 2

    def alloc():
        return {k: torch.zeros(own[k].shape, dtype=torch.float32) for k in keys}

    acc = alloc()

    def convert_only():
        for k in keys:
            own[k].detach().float().cpu()
            peer[k].detach().float().cpu()

    fp32_own = {k: own[k].float() for k in keys}
    fp32_peer = {k: peer[k].float() for k in keys}

    def accumulate_prconverted():
        for k in keys:
            acc[k] += fp32_own[k]
            acc[k] += fp32_peer[k]

    def convert_and_accumulate():
        for k in keys:
            acc[k] += own[k].detach().float().cpu()
            acc[k] += peer[k].detach().float().cpu()

    def divide_and_cast():
        return {k: (acc[k] / n).to(own[k].dtype) for k in keys}

    def foreach_accumulate():
        vals = [acc[k] for k in keys]
        torch._foreach_add_(vals, [fp32_own[k] for k in keys])
        torch._foreach_add_(vals, [fp32_peer[k] for k in keys])

    return {
        "allocate accumulators": timeit(alloc, reps),
        "convert only (.float().cpu())": timeit(convert_only, reps),
        "accumulate, pre-converted": timeit(accumulate_prconverted, reps),
        "convert+accumulate (as-is)": timeit(convert_and_accumulate, reps),
        "divide + cast to dtype": timeit(divide_and_cast, reps),
        "_foreach_add_ (pre-converted)": timeit(foreach_accumulate, reps),
    }


# ── Section 4: the grain-size probe ─────────────────────────────────────────


def bench_grain(reps: int) -> list[dict]:
    """Elementwise fp32 add across sizes that straddle ATen's parallel grain.

    ATen's internal grain size for elementwise ops is 32,768: below it the op
    runs on the calling thread, above it it forks an OpenMP team. If default
    threads are much SLOWER than 1 thread above the threshold and identical
    below it, the fork/join barrier is the cost -- and with 7 processes on one
    host it is contended 7 ways, which is exactly the dense/sparse asymmetry.

    The real adapter's tensors: lora_A 14,336 (below), lora_B for gate/up
    77,824 (above). So the dense path straddles this line and the p=0.05
    sparse path sits entirely below it.
    """
    sizes = [4_096, 14_336, 32_767, 32_769, 77_824, 262_144]
    default_threads = torch.get_num_threads()
    rows = []
    for size in sizes:
        a = torch.zeros(size, dtype=torch.float32)
        b = torch.randn(size, dtype=torch.float32)
        row: dict = {"numel": size}
        for label, threads in (("1thr", 1), ("default", default_threads)):
            torch.set_num_threads(threads)
            # 100 adds per timed call: a single add is below timer resolution.
            row[label] = timeit(lambda: [a.add_(b) for _ in range(100)], reps)["median_ms"] / 100
        row["ratio_default_over_1thr"] = row["default"] / row["1thr"] if row["1thr"] else 0.0
        rows.append(row)
    torch.set_num_threads(default_threads)
    return rows


# ── Section 5: concurrent processes ─────────────────────────────────────────


def _child(args, queue) -> None:
    dtype_str, threads, p, reps = args
    dtype = getattr(torch, dtype_str)
    if threads:
        torch.set_num_threads(threads)
    from mesh import merge_adapters

    own = build_adapter(dtype, seed=0)
    peer = build_adapter(dtype, seed=1)
    received = [("w-b", 10, peer)]
    queue.put(
        timeit(lambda: merge_adapters(own, received, contribute=True, own_name="w-a"), reps)
    )


def bench_concurrent(dtype: torch.dtype, threads: int | None, procs: int,
                     p: float, reps: int) -> dict[str, float]:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    dtype_str = str(dtype).replace("torch.", "")
    children = [
        ctx.Process(target=_child, args=((dtype_str, threads, p, reps), queue))
        for _ in range(procs)
    ]
    t0 = time.perf_counter()
    for c in children:
        c.start()
    results = [queue.get() for _ in children]
    for c in children:
        c.join()
    wall = (time.perf_counter() - t0) * 1000
    medians = [r["median_ms"] for r in results]
    return {
        # The barrier waits for the SLOWEST worker, so that is the number that
        # sets the round's cost -- not the mean.
        "slowest_median_ms": max(medians),
        "fastest_median_ms": min(medians),
        "mean_median_ms": statistics.fmean(medians),
        "wall_ms": wall,
    }


# ── Section 6: candidate implementations, head to head ──────────────────────
#
# All four compute the same uniform mean over the same canonically ordered
# contributions, in fp32, and differ ONLY in how the arithmetic is dispatched.
# They are defined here rather than in mesh.py so a single run of this script
# decides which one is worth shipping (plan §4.1B) instead of shipping one and
# hoping.


def _impl_current(own, contributions, keys, n):
    """Today's mesh.py: zeros accumulator, per-tensor += with inline convert."""
    acc = {k: torch.zeros(own[k].shape, dtype=torch.float32) for k in keys}
    for _sender, sd in contributions:
        for k in keys:
            acc[k] += sd[k].detach().float().cpu()
    return {k: (acc[k] / n).to(own[k].dtype) for k in keys}


def _impl_foreach(own, contributions, keys, n):
    """One dispatch per contribution instead of one per (contribution, tensor).

    The first contribution seeds the accumulator instead of being added to a
    zeros buffer -- 0.0 + x is exactly x in fp32, so this drops one full pass
    over the adapter without changing a single bit.
    """
    acc = None
    for _sender, sd in contributions:
        if acc is None:
            acc = [sd[k].detach().to(torch.float32, copy=True) for k in keys]
        else:
            torch._foreach_add_(acc, [sd[k].detach().to(torch.float32) for k in keys])
    torch._foreach_div_(acc, float(n))
    return {k: acc[i].to(own[k].dtype) for i, k in enumerate(keys)}


def _flat_plan(own, keys):
    offsets, total = [], 0
    for k in keys:
        numel = own[k].numel()
        offsets.append((total, numel))
        total += numel
    return offsets, total


def _impl_flat(own, contributions, keys, n, buffers=None):
    """One contiguous fp32 buffer: 112 elementwise ops collapse into 1.

    If the cost is per-op (dispatch, or an OpenMP fork/join per tensor
    contended 7 ways), this is where it disappears: the adds become a single
    ~2.9M-element pass regardless of how many tensors the adapter has.
    """
    offsets, total = _flat_plan(own, keys)
    if buffers is None:
        acc = torch.empty(total, dtype=torch.float32)
        scratch = torch.empty(total, dtype=torch.float32)
    else:
        acc, scratch = buffers
    for i, (_sender, sd) in enumerate(contributions):
        dest = acc if i == 0 else scratch
        for (off, numel), k in zip(offsets, keys):
            dest[off:off + numel].copy_(sd[k].detach().reshape(-1))
        if i:
            acc.add_(scratch)
    acc.div_(n)
    return {
        k: acc[off:off + numel].to(own[k].dtype).view(own[k].shape)
        for (off, numel), k in zip(offsets, keys)
    }


def bench_impls(dtype: torch.dtype, reps: int, r_values=(2, 3)) -> dict[str, dict[str, float]]:
    own = build_adapter(dtype, seed=0)
    peers = [build_adapter(dtype, seed=i + 1) for i in range(3)]
    keys = sorted(own.keys())
    _, total = _flat_plan(own, keys)
    buffers = (torch.empty(total, dtype=torch.float32),
               torch.empty(total, dtype=torch.float32))

    out: dict[str, dict[str, float]] = {}
    for r in r_values:
        contributions = sorted(
            [(f"w-{chr(98 + i)}", peers[i]) for i in range(r - 1)] + [("w-a", own)],
            key=lambda t: t[0],
        )
        n = len(contributions)
        ref = _impl_current(own, contributions, keys, n)
        for label, fn in (
            ("current (per-tensor +=)", lambda: _impl_current(own, contributions, keys, n)),
            ("_foreach_add_", lambda: _impl_foreach(own, contributions, keys, n)),
            ("flat buffer", lambda: _impl_flat(own, contributions, keys, n)),
            ("flat buffer, preallocated",
             lambda: _impl_flat(own, contributions, keys, n, buffers)),
        ):
            got = fn()
            # Bit-identity against the current implementation is not required by
            # the plan (§5 invariant 1 nuance) but getting it for free means the
            # existing bitwise tests need no edits at all.
            bitwise = all(torch.equal(got[k], ref[k]) for k in keys)
            res = timeit(fn, reps)
            res["bitwise_vs_current"] = 1.0 if bitwise else 0.0
            out[f"R={r} {label}"] = res
    return out


# ── Reporting ───────────────────────────────────────────────────────────────


def print_table(title: str, rows: dict[str, dict[str, float]]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(f"{'':34s} {'median':>9} {'min':>9} {'max':>9}")
    for label, r in rows.items():
        print(f"{label:34s} {r['median_ms']:>9.1f} {r['min_ms']:>9.1f} {r['max_ms']:>9.1f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32", "float16"])
    ap.add_argument("--p", type=float, default=0.05, help="sparsity for merge_sparse")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--procs", type=int, default=7,
                    help="concurrent processes for section 5 (7 = the real cluster)")
    ap.add_argument("--threads", type=int, default=None,
                    help="override torch intra-op threads for sections 1-2")
    ap.add_argument("--json", default=None, help="also write results here")
    ap.add_argument("--skip-concurrent", action="store_true")
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    default_threads = torch.get_num_threads()
    if args.threads:
        torch.set_num_threads(args.threads)

    own = build_adapter(dtype)
    n_tensors, n_params, n_bytes = adapter_stats(own)

    print("=" * 78)
    print("kiwi merge microbenchmark (MERGE_PERF_PLAN.md Phase 0)")
    print("=" * 78)
    print(f"torch {torch.__version__} | {platform.system()} {platform.machine()} | "
          f"{os.cpu_count()} cpus | default intra-op threads {default_threads}")
    print(f"adapter: {n_tensors} tensors, {n_params:,} params, {n_bytes:,} bytes "
          f"({args.dtype})")
    print(f"reps={args.reps}  p={args.p:g}  concurrent procs={args.procs}")
    print("\nv4 reference (dp2, 7 workers, one GPU box):")
    print("  dense merge_compute 3267 ms | sparse 92 ms | outer 5469 ms")

    results: dict = {
        "env": {
            "torch": torch.__version__,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "default_threads": default_threads,
            "dtype": args.dtype,
            "tensors": n_tensors,
            "params": n_params,
            "bytes": n_bytes,
        }
    }

    results["functions"] = bench_functions(dtype, args.p, args.reps)
    print_table("1. Public merge functions (ms per call)", results["functions"])

    results["phases"] = bench_phases(dtype, args.reps)
    print_table("2. Inside merge_adapters, phase by phase (R=2)", results["phases"])

    # 3. thread sweep
    print("\n3. Thread sweep -- merge_adapters R=2 (ms per call)")
    print("-" * 50)
    print(f"{'threads':>9} {'median':>9} {'Melem-ops/s':>13}")
    sweep = {}
    from mesh import merge_adapters

    peer = build_adapter(dtype, seed=1)
    received = [("w-b", 10, peer)]
    thread_options = sorted({1, 2, 4, default_threads})
    for threads in thread_options:
        torch.set_num_threads(threads)
        r = timeit(lambda: merge_adapters(own, received, contribute=True, own_name="w-a"),
                   args.reps)
        sweep[str(threads)] = r
        ops = 2 * n_params / (r["median_ms"] / 1000) / 1e6
        tag = "  (default)" if threads == default_threads else ""
        print(f"{threads:>9} {r['median_ms']:>9.1f} {ops:>13.1f}{tag}")
    results["thread_sweep"] = sweep
    torch.set_num_threads(args.threads or default_threads)

    # 4. grain probe
    results["grain"] = bench_grain(args.reps)
    print("\n4. Grain-size probe -- fp32 add_, us per op")
    print("-" * 62)
    print(f"{'numel':>9} {'1 thread':>11} {'default':>11} {'default/1thr':>14}")
    for row in results["grain"]:
        print(f"{row['numel']:>9,} {row['1thr']*1000:>11.1f} {row['default']*1000:>11.1f} "
              f"{row['ratio_default_over_1thr']:>14.2f}")
    print("  ATen parallelises elementwise ops above ~32,768 elements.")
    print("  ratio >> 1 above that line = fork/join overhead, not arithmetic.")

    # 5. concurrency
    if not args.skip_concurrent and args.procs > 1:
        print(f"\n5. {args.procs} concurrent processes -- merge_adapters R=2")
        print("-" * 62)
        print(f"{'config':>18} {'slowest':>10} {'fastest':>10} {'mean':>10}")
        conc = {}
        for label, threads in (("default threads", None), ("1 thread", 1), ("2 threads", 2)):
            r = bench_concurrent(dtype, threads, args.procs, args.p, max(5, args.reps // 4))
            conc[label] = r
            print(f"{label:>18} {r['slowest_median_ms']:>10.1f} "
                  f"{r['fastest_median_ms']:>10.1f} {r['mean_median_ms']:>10.1f}")
        results["concurrent"] = conc
        single = results["thread_sweep"][str(default_threads)]["median_ms"]
        blowup = conc["default threads"]["slowest_median_ms"] / single if single else 0
        print(f"\n  1 proc (default threads): {single:.1f} ms")
        print(f"  {args.procs} procs, slowest: "
              f"{conc['default threads']['slowest_median_ms']:.1f} ms "
              f"-> {blowup:.1f}x blowup from contention")

    # 6. candidate implementations, at each thread count
    print("\n6. Candidate implementations (ms per call, median)")
    print("-" * 78)
    impls: dict[str, dict] = {}
    for threads in thread_options:
        torch.set_num_threads(threads)
        rows = bench_impls(dtype, args.reps)
        impls[str(threads)] = rows
        tag = " (default)" if threads == default_threads else ""
        print(f"\n  torch threads = {threads}{tag}")
        print(f"  {'':32s} {'median':>9} {'min':>9}  bitwise==current")
        for label, r in rows.items():
            print(f"  {label:32s} {r['median_ms']:>9.1f} {r['min_ms']:>9.1f}"
                  f"{'      yes' if r['bitwise_vs_current'] else '       NO':>18}")
    results["impls"] = impls
    torch.set_num_threads(args.threads or default_threads)

    print("\n" + "=" * 78)
    print("VERDICT (plan §3 fork in the road)")
    print("=" * 78)
    single = results["thread_sweep"][str(default_threads)]["median_ms"]
    one_thr = results["thread_sweep"]["1"]["median_ms"]
    print(f"  isolated merge, default threads : {single:8.1f} ms")
    print(f"  isolated merge, 1 thread        : {one_thr:8.1f} ms")
    if "concurrent" in results:
        c = results["concurrent"]
        print(f"  {args.procs} concurrent, default threads : "
              f"{c['default threads']['slowest_median_ms']:8.1f} ms")
        print(f"  {args.procs} concurrent, 1 thread        : "
              f"{c['1 thread']['slowest_median_ms']:8.1f} ms")
        print(f"  {args.procs} concurrent, 2 threads       : "
              f"{c['2 threads']['slowest_median_ms']:8.1f} ms")
    print()
    print("  isolated merge already ~3 s        -> compute-bound: Phase 1B/1C")
    print("  isolated ~50 ms but N procs ~3 s   -> CONTENTION: Phase 1A is the fix")
    print("  1 thread fixes the N-proc case     -> confirms it; cap threads and stop")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
