import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

# All curves in the results JSON are on the SAMPLES-CONSUMED axis (§6.5) —
# under data parallelism "step" no longer means one thing across scenarios.

CONFIG_STYLES = {
    # Baselines — blues/greys
    "single_machine": {"color": "#1f77b4", "label": "single_machine (reference)"},
    "single_machine_offload": {"color": "#8c9eb2", "label": "single_machine (offload)"},
    "single_machine_qlora": {"color": "#6a51a3", "label": "single_machine (QLoRA)"},
    # R=1 controls — reds
    "distributed_3stage": {"color": "#d62728", "label": "dp1 (pipeline control)"},
    "dp1": {"color": "#d62728", "label": "dp1 (pipeline control)"},
    "dp1_mesh": {"color": "#ff9896", "label": "dp1_mesh (R=1 via MeshRunner)"},
    "dp1_freeze_a": {"color": "#e377c2", "label": "dp1_freeze_a (R=1, A frozen)"},
    # R=2, data-assignment variants — greens
    "dp2": {"color": "#2ca02c", "label": "dp2 (queue)"},
    "dp2_static": {"color": "#98df8a", "label": "dp2_static (clean split)"},
    "dp2_rotate": {"color": "#5fa855", "label": "dp2_rotate (rotating split)"},
    "dp2_mirror": {"color": "#17becf", "label": "dp2_mirror (identical data)"},
    # R=2, X sweep — oranges (light -> dark as X grows)
    "dp2_x1": {"color": "#ffbb78", "label": "dp2_x1 (merge every step)"},
    "dp2_x5": {"color": "#ff7f0e", "label": "dp2_x5"},
    "dp2_x20": {"color": "#b35806", "label": "dp2_x20 (merge once)"},
    # Ablation
    "dp2_freeze_a": {"color": "#9467bd", "label": "dp2_freeze_a (A frozen)"},
    "dp2_kill": {"color": "#006d2c", "label": "dp2 + mid-run kill"},
    "dp3": {"color": "#17becf", "label": "dp3 (mesh, R=3)"},
    # ── Phase 5: the alpaca1k A/B ──
    # Deliberately distinct hues from the zorbulax4 families above: the two
    # corpora must never be read as one series on a shared axis.
    "sm_alpaca": {"color": "#08306b", "label": "sm_alpaca (reference)"},
    "dp1_alpaca": {"color": "#a50f15", "label": "dp1_alpaca (pipeline control)"},
    "dp2_alpaca": {"color": "#238b45", "label": "dp2_alpaca (uniform X=2, baseline)"},
    "dp2_alpaca_lr2x": {"color": "#74c476", "label": "dp2_alpaca lr x2"},
    "dp2_sparse5": {"color": "#fd8d3c", "label": "dp2_sparse5 (p=0.05, every step)"},
    "dp2_sparse20": {"color": "#a63603", "label": "dp2_sparse20 (p=0.20)"},
    "dp2_diloco": {"color": "#6a51a3", "label": "dp2_diloco (dense X=8 + outer)"},
    "dp2_sparse_diloco": {"color": "#c994c7", "label": "dp2_sparse_diloco"},
    "dp2_alpaca_kill": {"color": "#00441b", "label": "dp2_alpaca + mid-run kill"},
}

# Scenarios whose samples-consumed axis is not comparable to the others.
# dp2_mirror trains every batch on EVERY replica, so its sample count is
# inflated R-fold; it must be read per steps-per-replica instead.
_INFLATED_SAMPLE_AXIS = {"dp2_mirror"}

_FALLBACK_PALETTE = [
    "#7f7f7f", "#bcbd22", "#c49c94", "#dbdb8d", "#9edae5", "#c7c7c7",
]


def _style(name):
    if name in CONFIG_STYLES:
        return CONFIG_STYLES[name]
    # Deterministic distinct colour for anything new, so an unregistered
    # scenario is still readable rather than another anonymous grey line.
    idx = sum(ord(c) for c in name) % len(_FALLBACK_PALETTE)
    return {"color": _FALLBACK_PALETTE[idx], "label": name}


def _interp_curves(curves, grid):
    rows = []
    for c in curves:
        c = sorted([tuple(p) for p in c])
        xs = np.array([p[0] for p in c], dtype=float)
        ys = np.array([p[1] for p in c], dtype=float)
        rows.append(np.interp(grid, xs, ys))
    return np.array(rows)


def plot_convergence(results_path: str, output_dir: str):
    with open(results_path, 'r', encoding="utf-8") as f:
        data = json.load(f)

    runs = data.get("runs", {})
    metrics = data.get("metrics", {})
    eval_data = data.get("evaluation", {})
    replica_runs = data.get("replica_runs", {})

    if not runs:
        print("No runs data found in results file.")
        return

    # ── 1. Loss vs samples consumed (the headline figure, §6.4) ──
    plt.figure(figsize=(14, 8))
    bar_data = {}

    for config_name, curves in runs.items():
        if not curves or config_name in _INFLATED_SAMPLE_AXIS:
            continue   # not comparable on this axis; see diagnostic_summary
        lo = max(min(p[0] for p in c) for c in curves)
        hi = min(max(p[0] for p in c) for c in curves)
        if hi <= lo:
            continue
        grid = np.linspace(lo, hi, num=min(200, int(hi - lo) + 1))
        losses = _interp_curves(curves, grid)
        mean, std = losses.mean(axis=0), losses.std(axis=0)

        style = _style(config_name)
        plt.plot(grid, mean, label=f"{style['label']} Mean", color=style["color"], linewidth=2)
        plt.fill_between(grid, mean - std, mean + std, color=style["color"], alpha=0.3)

        target_loss = 2.0
        samples_to_target = []
        for c in curves:
            for samples, loss in sorted([tuple(p) for p in c]):
                if loss <= target_loss:
                    samples_to_target.append(samples)
                    break
            else:
                samples_to_target.append(max(p[0] for p in c))

        bar_data[config_name] = {
            "label": style["label"],
            "color": style["color"],
            "final_loss_mean": mean[-1],
            "final_loss_std": std[-1],
            "samples_target_mean": np.mean(samples_to_target),
            "samples_target_std": np.std(samples_to_target),
        }

    plt.xlabel("Samples consumed")
    plt.ylabel("Loss")
    plt.title("Convergence at equal samples consumed\n"
              "(dp2_mirror omitted — its samples axis is inflated R-fold)")
    plt.legend(fontsize=8, ncol=2)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "convergence.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "convergence.svg"))
    plt.close()

    # ── 1b. HELD-OUT eval loss (Phase 5 — the headline figure) ──
    #
    # This is the plot the A/B is decided on. Train loss is kept above as a
    # diagnostic, but on a corpus a replica has just consumed it rewards
    # memorization, which is exactly the artifact that made the original
    # 4-fact conclusions invalid. Train curves are drawn faintly alongside so
    # a train/eval divergence (overfitting) is visible rather than inferred.
    eval_runs = data.get("eval_runs", {})
    if eval_runs:
        plt.figure(figsize=(14, 8))
        for config_name, curves in eval_runs.items():
            curves = [sorted([tuple(p) for p in c]) for c in curves if c]
            if not curves:
                continue
            lo = max(min(p[0] for p in c) for c in curves)
            hi = min(max(p[0] for p in c) for c in curves)
            if hi <= lo:
                continue
            grid = np.linspace(lo, hi, num=min(200, int(hi - lo) + 1))
            losses = _interp_curves(curves, grid)
            mean, std = losses.mean(axis=0), losses.std(axis=0)
            style = _style(config_name)
            plt.plot(grid, mean, label=f"{style['label']} (eval)",
                     color=style["color"], linewidth=2.2, marker="o", markersize=3)
            plt.fill_between(grid, mean - std, mean + std,
                             color=style["color"], alpha=0.25)

            train_curves = runs.get(config_name)
            if train_curves:
                t_lo = max(min(p[0] for p in c) for c in train_curves)
                t_hi = min(max(p[0] for p in c) for c in train_curves)
                if t_hi > t_lo:
                    t_grid = np.linspace(t_lo, t_hi, num=min(200, int(t_hi - t_lo) + 1))
                    t_mean = _interp_curves(train_curves, t_grid).mean(axis=0)
                    plt.plot(t_grid, t_mean, color=style["color"], linewidth=0.9,
                             alpha=0.35, linestyle=":")

        plt.xlabel("Samples consumed")
        plt.ylabel("Loss")
        plt.title("Held-out eval loss at equal samples consumed (PRIMARY)\n"
                  "solid + markers = eval (fixed held-out probe); "
                  "dotted = train loss, shown only as a divergence check")
        plt.legend(fontsize=8, ncol=2)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "eval_convergence.png"), dpi=300)
        plt.savefig(os.path.join(output_dir, "eval_convergence.svg"))
        plt.close()

        print("\nFinal HELD-OUT loss by scenario (sorted, lower is better):")
        finals_eval = {}
        for config_name, curves in eval_runs.items():
            lasts = [sorted([tuple(p) for p in c])[-1][1] for c in curves if c]
            if lasts:
                finals_eval[config_name] = (float(np.mean(lasts)), float(np.std(lasts)),
                                            len(lasts))
        for name in sorted(finals_eval, key=lambda n: finals_eval[n][0]):
            m, sd, k = finals_eval[name]
            print(f"   {name:22s} {m:9.4f} +/- {sd:7.4f}  ({k} seeds)")

    # ── 1c. Replica spread under sparse averaging (Phase 3.3) ──
    #
    # The sawtooth is the claim: adapter L2 distance between same-stage
    # replicas GROWS across sparse rounds and COLLAPSES to ~0 at each full
    # sync. A monotonically rising line means replicas are diverging and
    # "agreement = checkpoint" no longer holds — spare promotion would be
    # unsound no matter how good the loss curve looks.
    spread_runs = data.get("spread_runs", {})
    if spread_runs:
        plt.figure(figsize=(14, 6))
        plotted = False
        for scenario_name, seeds in spread_runs.items():
            style = _style(scenario_name)
            for seed_i, points in enumerate(seeds):
                if not points:
                    continue
                xs = [p["samples"] for p in points]
                ys = [p["spread_l2"] for p in points]
                plt.plot(xs, ys, color=style["color"], alpha=0.75,
                         linewidth=1.4,
                         label=style["label"] if seed_i == 0 else None)
                full_x = [p["samples"] for p in points if p.get("full_sync")]
                full_y = [p["spread_l2"] for p in points if p.get("full_sync")]
                if full_x:
                    plt.scatter(full_x, full_y, color=style["color"], marker="v",
                                s=45, zorder=3,
                                label=(f"{style['label']} full sync"
                                       if seed_i == 0 else None))
                plotted = True
        if plotted:
            plt.xlabel("Samples consumed")
            plt.ylabel("Max inter-replica adapter L2 distance")
            plt.title("Replica spread under sparse averaging\n"
                      "expected: bounded sawtooth, collapsing to ~0 at each full sync "
                      "(triangles). A rising trend means divergence.")
            plt.legend(fontsize=8)
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "replica_spread_l2.png"), dpi=300)
            plt.savefig(os.path.join(output_dir, "replica_spread_l2.svg"))
        plt.close()

    # ── 1d. Round-barrier profile (Phase 2 — Gate 2 is < 500 ms) ──
    profiles = data.get("round_profiles", {})
    if profiles:
        print("\nRound barrier profile (mean ms per round, across seeds):")
        for scenario_name, seeds in profiles.items():
            keys = sorted({k for s in seeds for k in s})
            merged = {k: float(np.mean([s[k] for s in seeds if k in s])) for k in keys}
            total = merged.get("gateway_barrier_total_ms")
            verdict = ""
            if total is not None:
                verdict = "  PASS" if total < 500 else "  FAIL (Gate 2: < 500 ms)"
            print(f"   {scenario_name}{verdict}")
            for k in sorted(merged, key=lambda k: -merged[k]):
                print(f"      {k:36s} {merged[k]:9.1f}")

    # ── 2. Bar charts ──
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 11))

    configs = list(bar_data.keys())
    labels = [bar_data[c]["label"] for c in configs]
    colors = [bar_data[c]["color"] for c in configs]

    ax1.bar(labels, [bar_data[c]["final_loss_mean"] for c in configs],
            yerr=[bar_data[c]["final_loss_std"] for c in configs],
            color=colors, alpha=0.7, capsize=10)
    ax1.set_title("Final Loss")
    ax1.set_ylabel("Loss")
    ax1.grid(True, axis='y', linestyle="--", alpha=0.6)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)

    ax2.bar(labels, [bar_data[c]["samples_target_mean"] for c in configs],
            yerr=[bar_data[c]["samples_target_std"] for c in configs],
            color=colors, alpha=0.7, capsize=10)
    ax2.set_title("Samples to Target Loss")
    ax2.set_ylabel("Samples consumed")
    ax2.grid(True, axis='y', linestyle="--", alpha=0.6)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)

    ax3.bar(labels, [metrics.get(c, {}).get("avg_peak_gpu_mb", 0) for c in configs],
            color=colors, alpha=0.7)
    ax3.set_title("Peak GPU Memory per worker (MB)")
    ax3.set_ylabel("Memory (MB)")
    ax3.grid(True, axis='y', linestyle="--", alpha=0.6)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)

    ax4.bar(labels, [metrics.get(c, {}).get("avg_step_time_ms", 0) for c in configs],
            color=colors, alpha=0.7)
    ax4.set_title("Avg Step Time (ms) — shared-GPU caveat:\n"
                  "dp_R vs dp1 measures communication overlap, NOT DP scaling (§6.1)")
    ax4.set_ylabel("Time (ms)")
    ax4.grid(True, axis='y', linestyle="--", alpha=0.6)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)

    verdicts = {k: v.get("verdict") for k, v in eval_data.items()} if eval_data else {}
    plt.suptitle(f"Benchmark Results — verdicts: {verdicts}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "convergence_bars.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "convergence_bars.svg"))
    plt.close()

    # ── 3. Inter-replica loss spread (consensus diagnostic, §6.4) ──
    # Expected to be NEAR-FLAT at N=4 — that flatness is an artifact of the
    # tiny dataset, not evidence of good consensus (§6.3). The figure exists
    # so it's ready when the corpus grows.
    if replica_runs:
        plt.figure(figsize=(12, 6))
        for scenario_name, seeds in replica_runs.items():
            spreads_per_seed = []
            for seed_curves in seeds:
                # seed_curves: {replica_id: [(samples, loss)]}
                per_replica = [sorted([tuple(p) for p in c]) for c in seed_curves.values()]
                if len(per_replica) < 2:
                    continue
                n_steps = min(len(c) for c in per_replica)
                if n_steps == 0:
                    continue
                # Spread = std across replicas of loss at each per-replica step index.
                losses = np.array([[c[i][1] for i in range(n_steps)] for c in per_replica])
                spreads_per_seed.append(losses.std(axis=0))
            if not spreads_per_seed:
                continue
            n = min(len(s) for s in spreads_per_seed)
            spread = np.array([s[:n] for s in spreads_per_seed]).mean(axis=0)
            style = _style(scenario_name)
            plt.plot(range(1, n + 1), spread, label=style["label"], color=style["color"])

        plt.xlabel("Step (per replica)")
        plt.ylabel("Inter-replica loss spread (std)")
        plt.title("Consensus diagnostic — near-flat is NOT MEASURABLE at N=4 (§6.3),\n"
                  "expected to grow over X steps and collapse at merges once the corpus grows")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "replica_spread.png"), dpi=300)
        plt.savefig(os.path.join(output_dir, "replica_spread.svg"))
        plt.close()

    # ── 4. Recovery events (dp2_kill, §6.4) ──
    events = data.get("events", {})
    recovery_rows = []
    for scenario_name, seeds in events.items():
        for seed_i, ev_list in enumerate(seeds):
            for ev in ev_list:
                if ev.get("type") == "recovered_step":
                    recovery_rows.append((scenario_name, seed_i, ev["samples"], ev["resume_ms"]))
    if recovery_rows:
        plt.figure(figsize=(10, 5))
        by_scenario = {}
        for name, seed_i, samples, resume_ms in recovery_rows:
            by_scenario.setdefault(name, []).append(resume_ms)
        names = list(by_scenario.keys())
        plt.bar(names, [np.mean(by_scenario[n]) for n in names],
                yerr=[np.std(by_scenario[n]) for n in names],
                color=[_style(n)["color"] for n in names], alpha=0.7, capsize=10)
        plt.ylabel("Resume latency (ms)")
        plt.title("Recovery: time to first successful step after worker death")
        plt.grid(True, axis='y', linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "recovery.png"), dpi=300)
        plt.close()

    # ── 4b. Loss vs STEPS PER REPLICA ──
    # The headline figure for the R=2 investigation. On the samples-consumed
    # axis an R-replica mesh is handicapped by construction: averaging advances
    # the group by the MEAN of the R local updates, not their sum, so at equal
    # samples it has taken only samples/R optimizer steps. Re-plotting against
    # steps per replica removes that structural difference and shows whether
    # the mesh is actually on the same trajectory.
    #
    # Steps per replica = samples_consumed / R for every mode (each recorded
    # sample is one replica taking one step), so the per-replica curves give it
    # directly; R=1 scenarios fall back to the combined curve.
    def _per_replica_step_curve(name):
        seeds = replica_runs.get(name)
        if seeds:
            per_seed = []
            for seed in seeds:
                for c in seed.values():
                    c = sorted([tuple(p) for p in c])
                    per_seed.append([loss for _s, loss in c])
            if not per_seed:
                return None
            n = min(len(c) for c in per_seed)
            if n == 0:
                return None
            return np.array([c[:n] for c in per_seed])
        curves = runs.get(name) or []
        if not curves:
            return None
        n = min(len(c) for c in curves)
        return np.array([[p[1] for p in sorted([tuple(x) for x in c])[:n]]
                         for c in curves])

    plt.figure(figsize=(13, 7.5))
    plotted = False
    for name in runs:
        arr = _per_replica_step_curve(name)
        if arr is None or arr.shape[1] == 0:
            continue
        steps = np.arange(1, arr.shape[1] + 1)
        mean, std = arr.mean(axis=0), arr.std(axis=0)
        style = _style(name)
        plt.plot(steps, mean, color=style["color"], linewidth=2,
                 label=f"{style['label']}  ({arr.shape[1]} steps)")
        plt.fill_between(steps, mean - std, mean + std,
                         color=style["color"], alpha=0.20)
        plotted = True

    if plotted:
        plt.yscale("log")
        plt.xlabel("Optimizer steps taken by a single replica")
        plt.ylabel("Loss (log scale)")
        plt.title(
            "Convergence per OPTIMIZER STEP, not per sample\n"
            "At equal samples an R=2 mesh has taken half the steps — curves "
            "that stop early did not diverge, they ran out of steps"
        )
        plt.legend(fontsize=8, ncol=2)
        plt.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "convergence_by_steps.png"), dpi=300)
        plt.savefig(os.path.join(output_dir, "convergence_by_steps.svg"))
    plt.close()

    # ── 5. Diagnostic summary: final loss per scenario, sorted ──
    # With a dozen scenarios the overlaid convergence curves stop being
    # readable, and the question these runs exist to answer ("which knob moves
    # the R=2 regression?") is a single-number-per-scenario comparison. This is
    # the figure to actually look at.
    finals = {}
    for name, curves in runs.items():
        if not curves:
            continue
        vals = [sorted([tuple(p) for p in c])[-1][1] for c in curves]
        finals[name] = (np.mean(vals), np.std(vals), len(vals))

    if finals:
        order = sorted(finals, key=lambda n: finals[n][0])
        fig_h = max(4.0, 0.45 * len(order) + 2.0)
        plt.figure(figsize=(11, fig_h))
        ypos = np.arange(len(order))
        means = [finals[n][0] for n in order]
        errs = [finals[n][1] for n in order]
        colors = [_style(n)["color"] for n in order]
        plt.barh(ypos, means, xerr=errs, color=colors, alpha=0.85, capsize=4)
        def _steps(n):
            seeds = replica_runs.get(n)
            if seeds and seeds[0]:
                return min(len(c) for c in seeds[0].values())
            cs = runs.get(n) or []
            return min((len(c) for c in cs), default=0)

        labels = [
            _style(n)["label"] + (" *" if n in _INFLATED_SAMPLE_AXIS else "")
            + f"  [{_steps(n)} steps/replica, n={finals[n][2]}]"
            for n in order
        ]
        plt.yticks(ypos, labels, fontsize=9)
        plt.gca().invert_yaxis()
        plt.xscale("symlog", linthresh=1e-3)
        plt.xlabel("Final loss (mean over seeds, ±1 std) — symlog scale")

        ref = finals.get("dp1", finals.get("distributed_3stage"))
        if ref:
            plt.axvline(ref[0], color="#d62728", linestyle="--", linewidth=1.2,
                        label=f"dp1 control = {ref[0]:.4f}")
            plt.legend(loc="lower right", fontsize=9)

        plt.title("Final loss by scenario — note the steps/replica column.\n"
                  "Scenarios sort almost perfectly by step count, not by data "
                  "assignment, merge frequency or freeze_a.\n"
                  "* dp2_mirror trains every batch on every replica.")
        plt.grid(True, axis="x", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "diagnostic_summary.png"), dpi=300)
        plt.savefig(os.path.join(output_dir, "diagnostic_summary.svg"))
        plt.close()

        print("\nFinal loss by scenario (sorted):")
        print(f"   {'scenario':20s} {'final':>9} {'+/-':>8} {'steps/rep':>10} {'seeds':>6}")
        for n in order:
            m, sd, k = finals[n]
            flag = " *" if n in _INFLATED_SAMPLE_AXIS else ""
            print(f"   {n:20s} {m:9.4f} {sd:8.4f} {_steps(n):>10} {k:>6}{flag}")

    print(f"\nPlots saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="benchmark_results/training_convergence_results.json")
    parser.add_argument("--output", type=str, default="benchmark_results")
    args = parser.parse_args()

    plot_convergence(args.results, args.output)
