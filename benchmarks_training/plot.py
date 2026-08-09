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
    with open(results_path, 'r') as f:
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
        labels = [
            _style(n)["label"] + (" *" if n in _INFLATED_SAMPLE_AXIS else "")
            + f"  (n={finals[n][2]})"
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

        plt.title("Diagnostic summary — which knob moves the R=2 regression?\n"
                  "* dp2_mirror trains every batch on every replica; compare it "
                  "per steps-per-replica, not samples consumed")
        plt.grid(True, axis="x", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "diagnostic_summary.png"), dpi=300)
        plt.savefig(os.path.join(output_dir, "diagnostic_summary.svg"))
        plt.close()

        print("\nFinal loss by scenario (sorted):")
        for n in order:
            m, s, k = finals[n]
            flag = "   <-- inflated samples axis" if n in _INFLATED_SAMPLE_AXIS else ""
            print(f"   {n:18s} {m:9.4f} +/- {s:.4f}  (n={k}){flag}")

    print(f"\nPlots saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="benchmark_results/training_convergence_results.json")
    parser.add_argument("--output", type=str, default="benchmark_results")
    args = parser.parse_args()

    plot_convergence(args.results, args.output)
