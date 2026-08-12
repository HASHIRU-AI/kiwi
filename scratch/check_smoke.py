#!/usr/bin/env python3
"""Assert the averaging MACHINERY behaved, from a smoke run's artifacts.

Usage:
    python scratch/check_smoke.py benchmark_results_v2/smoke

Checks the invariants that are cheap to get wrong and silent when wrong:

  * sparse rounds really are sparse (p on the wire, payload ~1/p the size)
  * full syncs fire on the announced cadence and are dense
  * the outer step fires ONLY on dense rounds, and never on the first one
    (the first dense round establishes the anchor without stepping — that is
    what keeps replicas bit-identical, since same-stage peers do not start
    from the same LoRA A)
  * replica spread collapses at a full sync relative to the round before it
  * a killed worker's spare is promoted with R holding, so no sample is
    trained below nominal R

Exits non-zero if any check fails, so it can gate a run.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict

ROUND_RE = re.compile(
    r"\[(?P<worker>worker_\d+)\] Round (?P<round>\d+) complete: "
    r"merged n=(?P<n>\d+) \((?P<flags>[^)]*)\)(?P<rest>.*)"
)
ANCHOR_RE = re.compile(r"\[(worker_\d+)\] Outer optimizer: anchor established at round (\d+)")
PROMOTED_RE = re.compile(r"\[(worker_\d+)\] \*\*\* Promoted to ACTIVE")

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((PASS if ok else FAIL, name, detail))


def parse_log(path: str):
    rounds = defaultdict(dict)   # round_id -> worker -> info
    anchors, promotions = [], []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            m = ROUND_RE.search(line)
            if m:
                flags = m.group("flags")
                p = re.search(r"p=([\d.eE+-]+)", flags)
                payload = re.search(r"payload_bytes=([\d.]+)", m.group("rest"))
                rounds[int(m.group("round"))][m.group("worker")] = {
                    "p": float(p.group(1)) if p else None,
                    "full": "FULL" in flags,
                    "outer": "OUTER" in flags,
                    "n": int(m.group("n")),
                    "payload_bytes": float(payload.group(1)) if payload else None,
                }
            if (a := ANCHOR_RE.search(line)):
                anchors.append((a.group(1), int(a.group(2))))
            if (pr := PROMOTED_RE.search(line)):
                promotions.append(pr.group(1))
    return rounds, anchors, promotions


def check_scenario(scenario: str, out_dir: str, cfg: dict, metrics: dict,
                   spread: list) -> None:
    log = os.path.join(out_dir, "container_logs", f"{scenario}_seed1.log")
    if not os.path.exists(log):
        check(f"{scenario}: container log present", False, log)
        return
    rounds, anchors, promotions = parse_log(log)
    check(f"{scenario}: rounds recorded", bool(rounds), f"{len(rounds)} rounds")
    if not rounds:
        return

    sparse_p = cfg.get("sparse_p", 1.0)
    cadence = cfg.get("full_sync_every") or 10
    outer_opt = cfg.get("outer_opt", False)

    full_ids = sorted(r for r, w in rounds.items() if any(i["full"] for i in w.values()))
    sparse_ids = sorted(set(rounds) - set(full_ids))

    # Every worker in a round must agree on p — they derive masks from it
    # without exchanging them, so disagreement corrupts silently.
    disagree = [r for r, w in rounds.items() if len({i["p"] for i in w.values()}) > 1]
    check(f"{scenario}: all workers agree on p each round", not disagree, f"bad rounds {disagree[:5]}")

    if sparse_p < 1.0:
        check(f"{scenario}: sparse rounds exist", bool(sparse_ids), f"{len(sparse_ids)} sparse")
        wrong = [r for r in sparse_ids
                 if any(abs((i["p"] or 0) - sparse_p) > 1e-9 for i in rounds[r].values())]
        check(f"{scenario}: sparse rounds carry p={sparse_p}", not wrong, f"bad {wrong[:5]}")

        expected = sorted(r for r in rounds if r % cadence == 0)
        check(f"{scenario}: full syncs on cadence {cadence}", full_ids == expected,
              f"got {full_ids} expected {expected}")

        sp = [i["payload_bytes"] for r in sparse_ids for i in rounds[r].values()
              if i["payload_bytes"]]
        dn = [i["payload_bytes"] for r in full_ids for i in rounds[r].values()
              if i["payload_bytes"]]
        if sp and dn:
            ratio = (sum(dn) / len(dn)) / (sum(sp) / len(sp))
            # Sparse payload is p of the values plus safetensors header, so
            # the ratio lands a bit under 1/p rather than exactly on it.
            check(f"{scenario}: sparse payload ~{1/sparse_p:.0f}x smaller",
                  ratio > (1 / sparse_p) * 0.5, f"dense/sparse = {ratio:.1f}x")
    else:
        check(f"{scenario}: all rounds dense", not sparse_ids, f"{len(sparse_ids)} sparse")

    if outer_opt:
        # The invariant that matters: outer ONLY on dense rounds.
        bad = [r for r in sparse_ids if any(i["outer"] for i in rounds[r].values())]
        check(f"{scenario}: outer step never on a sparse round", not bad, f"bad {bad[:5]}")

        check(f"{scenario}: anchor established exactly once per worker",
              len(anchors) == len({w for w, _ in anchors}) and bool(anchors),
              f"{len(anchors)} anchor events across {len({w for w,_ in anchors})} workers")

        first_full = full_ids[0] if full_ids else None
        no_step = not any(i["outer"] for i in rounds.get(first_full, {}).values())
        check(f"{scenario}: first dense round does NOT outer-step", no_step,
              f"round {first_full}")

        stepped = [r for r in full_ids[1:] if any(i["outer"] for i in rounds[r].values())]
        check(f"{scenario}: outer steps on later dense rounds",
              len(stepped) == max(0, len(full_ids) - 1),
              f"{len(stepped)} of {max(0, len(full_ids)-1)}")
    else:
        bad = [r for r, w in rounds.items() if any(i["outer"] for i in w.values())]
        check(f"{scenario}: no outer step when outer_opt is off", not bad, f"bad {bad[:5]}")

    # Replica spread: the trough (full sync) must be well below the peak
    # (last sparse round before it). This is the sawtooth.
    if sparse_p < 1.0 and spread:
        peaks = [s["spread_l2"] for s in spread if s.get("phase") == "peak"]
        troughs = [s["spread_l2"] for s in spread if s.get("phase") == "trough"]
        check(f"{scenario}: both spread phases sampled", bool(peaks) and bool(troughs),
              f"{len(peaks)} peaks, {len(troughs)} troughs")
        if peaks and troughs:
            pk, tr = sum(peaks) / len(peaks), sum(troughs) / len(troughs)
            check(f"{scenario}: spread collapses at full sync", tr < pk * 0.5,
                  f"peak {pk:.6f} -> trough {tr:.6f}")

    if cfg.get("kill_after_epochs") is not None:
        check(f"{scenario}: a spare was promoted", bool(promotions), f"{promotions}")
        frac = metrics.get("avg_degraded_sample_fraction", metrics.get("degraded_sample_fraction"))
        check(f"{scenario}: R held through the kill", frac == 0.0,
              f"degraded_sample_fraction={frac}")


def main() -> int:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "benchmark_results_v2/smoke"
    results_path = os.path.join(out_dir, "training_convergence_results.json")
    if not os.path.exists(results_path):
        print(f"No results at {results_path}")
        return 2
    data = json.load(open(results_path, encoding="utf-8"))

    for scenario, cfg in data.get("scenario_config", {}).items():
        spread_seeds = data.get("spread_runs", {}).get(scenario, [])
        check_scenario(
            scenario, out_dir, cfg,
            data.get("metrics", {}).get(scenario, {}),
            spread_seeds[0] if spread_seeds else [],
        )

    width = max((len(n) for _, n, _ in results), default=0)
    for status, name, detail in results:
        print(f"  [{status}] {name:{width}s}  {detail}")
    failed = sum(1 for s, _, _ in results if s == FAIL)
    print(f"\n{len(results) - failed}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
