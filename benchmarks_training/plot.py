import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

# All curves in the results JSON are on the SAMPLES-CONSUMED axis (§6.5) —
# under data parallelism "step" no longer means one thing across scenarios.

# Display labels. Scenario keys in scenarios.yaml were renamed to match these
# (dp1/dp2/*_alpaca -> sm / kiwi_pipeline / kiwi_mesh2_*), so the name you type
# on the command line is the name you read on the figure. Results are matched
# by key, so files predating that rename fall through to _style()'s anonymous
# fallback — they still plot, just without labels or stable colours.
#
# Conventions:
#   "Single GPU"    — no pipeline, no averaging
#   "Kiwi pipeline" — model split across workers, ONE pipeline (R=1)
#   "Kiwi mesh ×R"  — R pipelines whose adapters are averaged
#   mechanism goes after the em-dash; labels stay under ~40 chars so legends
#   do not wrap.
#
# Colour is a per-scenario identity: one scenario is the same colour in EVERY
# figure, so a reader can track an arm across the comparison set without
# re-reading legends. Baselines cool, mesh arms warm.
CONFIG_STYLES = {
    # ── Legacy zorbulax4 baselines (40-sample runs) ──
    "sm_zorb": {"color": "#1f77b4", "label": "Single GPU (reference)"},
    "sm_offload_zorb": {"color": "#8c9eb2", "label": "Single GPU + CPU offload"},
    "kiwi_pipeline_zorb": {"color": "#d62728", "label": "Kiwi pipeline"},
    "kiwi_mesh2_zorb": {"color": "#2ca02c", "label": "Kiwi mesh ×2 — dense avg"},

    # ── The real-corpus A/B (truthfulqa) ──
    # Deliberately distinct hues from the zorbulax4 family above: the two
    # corpora must never be read as one series on a shared axis.
    "sm": {"color": "#08306b", "label": "Single GPU (reference)"},
    "sm_offload": {"color": "#6baed6", "label": "Single GPU + CPU offload"},
    "kiwi_pipeline": {"color": "#a50f15", "label": "Kiwi pipeline"},
    "kiwi_mesh2_dense": {"color": "#238b45", "label": "Kiwi mesh ×2 — dense avg"},
    "kiwi_mesh2_sparse5": {"color": "#fd8d3c", "label": "Kiwi mesh ×2 — sparse 5%"},
    "kiwi_mesh2_outer": {"color": "#6a51a3", "label": "Kiwi mesh ×2 — outer optimizer"},
    "kiwi_mesh2_sparse5_kill": {"color": "#00441b", "label": "Kiwi mesh ×2 — sparse 5% + worker kill"},

    # ── No-spare variants (6 workers) ──
    # Lighter shade of each 7-worker counterpart's hue, so a spare/no-spare
    # pair reads as one family on a shared axis while staying distinguishable.
    "kiwi_mesh2_dense_nospare": {"color": "#74c476", "label": "Kiwi mesh ×2 — dense avg, no spare"},
    "kiwi_mesh2_sparse5_nospare": {"color": "#fdbe85", "label": "Kiwi mesh ×2 — sparse 5%, no spare"},
    "kiwi_mesh2_outer_nospare": {"color": "#9e9ac8", "label": "Kiwi mesh ×2 — outer opt, no spare"},

    # ── Retired scenarios that still appear in ARCHIVED results ──
    # These are gone from scenarios.yaml, but benchmark_results*/ files
    # reference them by key. Dropping their styles would not break plotting —
    # _style() falls back — but old figures would lose their labels and get an
    # arbitrary fallback colour, so the archive stops being readable. Cheap to
    # keep; delete only alongside the results that use them.
    "distributed_3stage": {"color": "#d62728", "label": "Kiwi pipeline"},
    "single_machine_qlora": {"color": "#6a51a3", "label": "Single GPU (QLoRA)"},
    "dp2_sparse20": {"color": "#a63603", "label": "Kiwi mesh ×2 — sparse 20%"},
    "dp2_sparse_diloco": {"color": "#c994c7", "label": "Kiwi mesh ×2 — sparse 5% + outer opt"},
    # Pre-rename keys for the same arms, so results produced before the
    # dp*/…_alpaca -> kiwi_* rename still render with labels and their
    # original colours. Delete this block once those results are superseded.
    "single_machine": {"color": "#1f77b4", "label": "Single GPU (reference)"},
    "single_machine_offload": {"color": "#8c9eb2", "label": "Single GPU + CPU offload"},
    "dp1": {"color": "#d62728", "label": "Kiwi pipeline"},
    "dp2": {"color": "#2ca02c", "label": "Kiwi mesh ×2 — dense avg"},
    "sm_alpaca": {"color": "#08306b", "label": "Single GPU (reference)"},
    "dp1_alpaca": {"color": "#a50f15", "label": "Kiwi pipeline"},
    "dp2_alpaca": {"color": "#238b45", "label": "Kiwi mesh ×2 — dense avg"},
    "dp2_sparse5": {"color": "#fd8d3c", "label": "Kiwi mesh ×2 — sparse 5%"},
    "dp2_diloco": {"color": "#6a51a3", "label": "Kiwi mesh ×2 — outer optimizer"},
    "dp2_sparse5_kill": {"color": "#00441b", "label": "Kiwi mesh ×2 — sparse 5% + worker kill"},
}

# Scenarios whose samples-consumed axis is not comparable to the others —
# e.g. a mirrored-shard arm trains every batch on EVERY replica, inflating its
# sample count R-fold. Empty now that dp2_mirror is retired; kept because the
# hazard reappears the moment such an arm is added back.
_INFLATED_SAMPLE_AXIS = set()

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


def _verdicts(eval_data):
    """Verdict per comparison, tolerating both results-file schemas."""
    if not isinstance(eval_data, dict) or not eval_data:
        return {}
    if "verdict" in eval_data:                      # legacy: one flat result
        return {"overall": eval_data.get("verdict")}
    return {k: v.get("verdict") for k, v in eval_data.items()
            if isinstance(v, dict)}


def _interp_curves(curves, grid):
    rows = []
    for c in curves:
        c = sorted([tuple(p) for p in c])
        xs = np.array([p[0] for p in c], dtype=float)
        ys = np.array([p[1] for p in c], dtype=float)
        rows.append(np.interp(grid, xs, ys))
    return np.array(rows)


# ── Per-comparison eval figures ─────────────────────────────────────────────
#
# The problem: with 7+ arms on one axis nothing is readable, and any figure
# that also carries per-sample train loss is forced to a 0-5 y-range, which
# compresses the ~0.1 differences the whole A/B is about into a few pixels.
#
# The fix: one small EVAL-ONLY figure per question. Each states its question
# in the subtitle, so a reader never has to reconstruct what they are looking
# at from scenario names.


def _eval_series(eval_runs, scenario_names, x_scale=None):
    """Mean/std eval curves for the named scenarios, on a common grid.

    `x_scale` maps scenario name -> divisor for the x axis (used by the
    steps-per-replica and wall-clock views). Scenarios absent from the
    results are skipped by the caller, never faked.
    """
    series, missing = {}, []
    for name in scenario_names:
        curves = [sorted([tuple(p) for p in c]) for c in eval_runs.get(name, []) if c]
        if not curves:
            missing.append(name)
            continue
        scale = (x_scale or {}).get(name, 1.0)
        if not scale:
            missing.append(name)
            continue
        curves = [[(x / scale, y) for x, y in c] for c in curves]
        lo = max(min(p[0] for p in c) for c in curves)
        hi = min(max(p[0] for p in c) for c in curves)
        if hi <= lo:
            missing.append(name)
            continue
        grid = np.linspace(lo, hi, num=200)
        losses = _interp_curves(curves, grid)
        series[name] = (grid, losses.mean(axis=0), losses.std(axis=0), len(curves))
    return series, missing


def _kill_markers(events, scenario_names):
    """(samples, label, scenario) for the first kill and promotion in these arms.

    The scenario is returned because the x axis may be rescaled per scenario
    (steps-per-replica, wall clock) — using another arm's divisor would put
    the marker at the wrong place, which is worse than not drawing it.
    """
    marks = []
    seen = set()
    for name in scenario_names:
        for seed_events in events.get(name, []):
            for ev in seed_events:
                label = {"consumer_retired": "worker killed",
                         "consumer_rejoined": "spare promoted"}.get(ev.get("type"))
                if label and label not in seen:
                    seen.add(label)
                    marks.append((ev.get("samples", 0), label, name))
    return marks


def _plot_eval_panel(eval_runs, scenario_names, title, question, outfile,
                     events=None, scenario_config=None, x_scale=None,
                     xlabel="Samples consumed", extra_note=None):
    """One eval-ONLY figure for one named comparison.

    Returns True if it drew something. A comparison whose scenarios are all
    absent is SKIPPED with a printed note rather than raising — the same code
    has to work on older results files that predate these arms.
    """
    series, missing = _eval_series(eval_runs, scenario_names, x_scale)
    if missing:
        print(f"    [skip] {os.path.basename(outfile)}: no eval data for "
              f"{', '.join(missing)}")
    if not series:
        return False

    plt.figure(figsize=(11, 6.5))
    lo_y = min(float((m - s).min()) for _, m, s, _ in series.values())
    hi_y = max(float((m + s).max()) for _, m, s, _ in series.values())

    # Ranked order, so the legend reads as the result table.
    for name in sorted(series, key=lambda n: series[n][1][-1]):
        grid, mean, std, n_seeds = series[name]
        style = _style(name)
        spread = f" ± {std[-1]:.4f}" if n_seeds > 1 else ""
        plt.plot(grid, mean, color=style["color"], linewidth=2.4,
                 marker="o", markersize=4,
                 label=f"{style['label']}  →  {mean[-1]:.4f}{spread}")
        if n_seeds > 1:
            plt.fill_between(grid, mean - std, mean + std,
                             color=style["color"], alpha=0.18)

    for target in {c.get("target_loss") for c in (scenario_config or {}).values()
                   if c.get("target_loss")}:
        if lo_y <= target <= hi_y:
            plt.axhline(target, color="#555555", linestyle="-.", linewidth=1.2,
                        alpha=0.8, label=f"target_loss = {target:g}")

    pad = (hi_y - lo_y) * 0.08 or 0.01
    plt.ylim(lo_y - pad, hi_y + pad)

    # Drawn AFTER ylim, against a blended transform (x in data units, y in
    # axes fraction) so the labels sit inside the plot instead of colliding
    # with the title, whatever the y-range turns out to be.
    ax = plt.gca()
    for samples, label, origin in _kill_markers(events or {}, scenario_names):
        x = samples / (x_scale or {}).get(origin, 1.0)
        ax.axvline(x, color="#333333", linestyle="--", linewidth=1.4, alpha=0.9)
        ax.annotate(label, xy=(x, 0.97), xycoords=ax.get_xaxis_transform(),
                    xytext=(5, 0), textcoords="offset points", fontsize=9,
                    rotation=90, va="top", color="#333333")
    n_seeds = max(s[3] for s in series.values())
    subtitle = question
    if extra_note:
        subtitle += f"\n{extra_note}"
    subtitle += f"\n{n_seeds} seed(s); band = ±1 std across seeds"
    if n_seeds < 2:
        subtitle += " — ZERO-WIDTH AT 1 SEED, apparent gaps may be noise"
    plt.title(f"{title}\n{subtitle}", fontsize=11)
    plt.xlabel(xlabel)
    plt.ylabel("Held-out eval loss")
    plt.legend(fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(outfile + ".png", dpi=300)
    plt.savefig(outfile + ".svg")
    plt.close()
    print(f"    [ok]   {os.path.basename(outfile)}  ({len(series)} arms)")
    return True


# One entry = one figure. DATA, not code: adding a comparison later must not
# require new plotting logic. Names are resolved against the results file
# actually being plotted, and unknown ones are skipped — older result files
# (zorbulax diagnostics, no eval_runs at all) must still plot without error.
#
# `by_steps` additionally emits a steps-per-replica variant: the matched
# sequential-depth axis, which is where the mesh's "same steps, twice the
# data" claim actually lives.
COMPARISONS = [
    dict(
        slug="01_baselines",
        scenarios=["sm_zorb", "sm_offload_zorb", "sm",
                   "sm_offload", "kiwi_pipeline"],
        title="Does splitting the model across workers change what it learns?",
        question="Single GPU vs CPU-offload vs Kiwi's pipeline. All three should "
                 "overlap — this is the correctness anchor for everything below.",
    ),
    dict(
        slug="02_pipeline_vs_mesh",
        scenarios=["kiwi_pipeline", "kiwi_mesh2_dense"],
        title="What does data-parallel averaging cost at equal samples consumed?",
        question="Expect the mesh to trail: at R=2 each replica takes half the "
                 "sequential steps for the same data. This is a known handicap of "
                 "the axis, not a defect — see the by-steps variant.",
        by_steps=True,
    ),
    dict(
        slug="03_averaging_mechanisms",
        scenarios=["kiwi_mesh2_dense", "kiwi_mesh2_sparse5", "kiwi_mesh2_outer", "dp2_sparse_diloco"],
        title="Which averaging mechanism closes the gap?",
        question="Sparse slices, an outer optimizer, or both — measured against "
                 "dense uniform averaging.",
        by_steps=True,
    ),
    # Renders only against archived results: dp2_sparse20 was retired from
    # scenarios.yaml, so new runs produce a 2-arm figure (dense vs 5%) rather
    # than a sweep. Restore that scenario to get the sensitivity point back —
    # without it there is no evidence separating "sparsity helps" from "any
    # amount of averaging helps".
    dict(
        slug="04_sparse_p_sweep",
        scenarios=["kiwi_mesh2_dense", "kiwi_mesh2_sparse5", "dp2_sparse20"],
        title="How much sparsity can averaging tolerate?",
        question="Dense (p=1.0) vs 20% vs 5% of adapter values exchanged per round.",
    ),
    # Per-mechanism pairs. The combined six-arm figure below shows everything
    # at once; these isolate one pair each, because three overlapping pairs on
    # one axis is exactly the readability problem this whole registry exists
    # to avoid.
    dict(
        slug="08a_spare_cost_dense",
        scenarios=["kiwi_mesh2_dense", "kiwi_mesh2_dense_nospare"],
        title="Hot spare vs none — dense averaging",
        question="7 workers (2 pipes + spare) vs 6. Curves should overlap: the "
                 "spare never trains.",
    ),
    dict(
        slug="08b_spare_cost_sparse5",
        scenarios=["kiwi_mesh2_sparse5", "kiwi_mesh2_sparse5_nospare"],
        title="Hot spare vs none — sparse 5%",
        question="7 workers (2 pipes + spare) vs 6. Curves should overlap: the "
                 "spare never trains.",
    ),
    dict(
        slug="08c_spare_cost_outer",
        scenarios=["kiwi_mesh2_outer", "kiwi_mesh2_outer_nospare"],
        title="Hot spare vs none — outer optimizer",
        question="7 workers (2 pipes + spare) vs 6. Curves should overlap: the "
                 "spare never trains.",
    ),
    dict(
        slug="08_spare_cost",
        scenarios=["kiwi_mesh2_dense", "kiwi_mesh2_dense_nospare",
                   "kiwi_mesh2_sparse5", "kiwi_mesh2_sparse5_nospare",
                   "kiwi_mesh2_outer", "kiwi_mesh2_outer_nospare"],
        title="What does keeping a hot spare cost?",
        question="7 workers (2 pipes + spare) vs 6 (2 pipes). The spare never "
                 "trains — it only receives round payloads to stay promotable — "
                 "so each pair should OVERLAP. A visible gap means the spare is "
                 "influencing training, which would be a bug. The real "
                 "difference is in avg_round_ms and the round profile.",
    ),
    dict(
        slug="05_recovery",
        scenarios=["kiwi_mesh2_sparse5", "kiwi_mesh2_sparse5_kill", "kiwi_mesh2_dense"],
        title="Does killing a worker mid-run change the learning curve?",
        question="A same-stage spare has been receiving round payloads all along, "
                 "so promotion should cost ~0 and the curve should not bend.",
    ),
]


# Spare / no-spare pairs. Same mechanism, 7 workers (2 pipes + spare) vs 6.
SPARE_PAIRS = [
    ("dense avg", "kiwi_mesh2_dense", "kiwi_mesh2_dense_nospare"),
    ("sparse 5%", "kiwi_mesh2_sparse5", "kiwi_mesh2_sparse5_nospare"),
    ("outer opt", "kiwi_mesh2_outer", "kiwi_mesh2_outer_nospare"),
]


def _plot_spare_cost_bars(data, output_dir):
    """What the spare costs, in the units where it actually shows up.

    The eval panels for these pairs are the NULL result: the spare never
    trains, it only receives round payloads to stay promotable, so the curves
    should sit on top of each other. Plotting only those would leave the
    reader with "no difference" and no answer.

    The difference is in cost, for two structural reasons: one fewer process
    contending for the GPU, and contributors fanning out to one same-stage
    peer instead of two (so one less inbound payload and one less merge per
    round). This chart puts those numbers side by side with the percentage
    delta, which is the actual price of fault tolerance.
    """
    metrics = data.get("metrics", {})
    pairs = [(label, a, b) for label, a, b in SPARE_PAIRS
             if a in metrics and b in metrics]
    if not pairs:
        print("    [skip] 09_spare_cost_bars: need both arms of at least one "
              "spare/no-spare pair in the same results file")
        return

    panels = [
        ("avg_round_ms", "Averaging barrier (ms)", "lower is better"),
        ("avg_samples_per_sec", "Throughput (samples/s)", "higher is better"),
        ("avg_step_time_ms", "Train step (ms)", "lower is better"),
    ]
    available = [p for p in panels
                 if any(metrics[a].get(p[0]) or metrics[b].get(p[0])
                        for _l, a, b in pairs)]
    if not available:
        print("    [skip] 09_spare_cost_bars: no cost metrics recorded")
        return

    fig, axes = plt.subplots(1, len(available), figsize=(6 * len(available), 5.5))
    if len(available) == 1:
        axes = [axes]

    x = np.arange(len(pairs))
    width = 0.36
    for ax, (key, title, direction) in zip(axes, available):
        spare = [float(metrics[a].get(key) or 0.0) for _l, a, _b in pairs]
        nospare = [float(metrics[b].get(key) or 0.0) for _l, _a, b in pairs]
        ax.bar(x - width / 2, spare, width, label="7 workers (with spare)",
               color="#238b45")
        ax.bar(x + width / 2, nospare, width, label="6 workers (no spare)",
               color="#a1d99b")
        for i, (s, n) in enumerate(zip(spare, nospare)):
            if s and n:
                # Signed cost of KEEPING the spare, so a positive number always
                # reads as "the spare costs this much" regardless of whether
                # the metric is better-high or better-low.
                delta = (s - n) / n * 100 if "samples_per_sec" not in key else (n - s) / s * 100
                ax.annotate(f"{delta:+.0f}%", xy=(i, max(s, n)),
                            xytext=(0, 4), textcoords="offset points",
                            ha="center", fontsize=10, color="#333333")
        ax.set_xticks(x)
        ax.set_xticklabels([p[0] for p in pairs])
        ax.set_title(f"{title}\n({direction}; % = cost of the spare)", fontsize=11)
        ax.grid(True, axis="y", linestyle="--", alpha=0.6)
        ax.legend(fontsize=9)

    fig.suptitle("What does keeping a hot spare cost?\n"
                 "Convergence should be identical — the spare never trains. "
                 "This is where the difference actually lands.", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "09_spare_cost_bars.png"), dpi=300)
    fig.savefig(os.path.join(output_dir, "09_spare_cost_bars.svg"))
    plt.close(fig)
    print(f"    [ok]   09_spare_cost_bars  ({len(pairs)} pair(s))")

    print("\n  Cost of the hot spare (7 workers vs 6):")
    for label, a, b in pairs:
        row = [f"    {label:12s}"]
        for key, title, _d in available:
            s, n = metrics[a].get(key), metrics[b].get(key)
            if s and n:
                delta = (s - n) / n * 100 if "samples_per_sec" not in key else (n - s) / s * 100
                row.append(f"{title.split(' (')[0]} {s:8.1f} vs {n:8.1f} ({delta:+.0f}%)")
        print("  ".join(row))


def _best_mesh_arm(eval_runs, scenario_config):
    """Lowest final eval loss among arms with replicas > 1."""
    best, best_loss = None, float("inf")
    for name, curves in eval_runs.items():
        if (scenario_config.get(name, {}) or {}).get("replicas", 1) <= 1:
            continue
        finals = [sorted([tuple(p) for p in c])[-1][1] for c in curves if c]
        if finals and (mean := sum(finals) / len(finals)) < best_loss:
            best, best_loss = name, mean
    return best


def _plot_comparisons(data, output_dir):
    """Task A: one eval-only figure per comparison, into <output>/comparisons/."""
    eval_runs = data.get("eval_runs", {})
    if not eval_runs:
        print("\nComparison figures: skipped — this results file has no eval_runs "
              "(pre-Phase-1 run).")
        return

    cmp_dir = os.path.join(output_dir, "comparisons")
    os.makedirs(cmp_dir, exist_ok=True)
    cfg = data.get("scenario_config", {})
    events = data.get("events", {})
    metrics = data.get("metrics", {})
    print(f"\nComparison figures -> {cmp_dir}")

    def replicas(name):
        return max(1, int((cfg.get(name, {}) or {}).get("replicas", 1) or 1))

    def _one_corpus(slug, names):
        """Never plot two corpora on one loss axis.

        Several comparisons list both the zorbulax4 and truthfulqa version of
        an arm, because the registry is corpus-agnostic and skips whatever is
        absent. That works only while a results file holds one corpus. Run
        both into the same file and e.g. 01_baselines would overlay a 4-fact
        memorization curve on a 600-sample held-out curve — different loss
        scales, silently comparable-looking, completely meaningless.
        Keep the largest group and say what was dropped.
        """
        groups = {}
        for n in names:
            groups.setdefault((cfg.get(n, {}) or {}).get("dataset", "unknown"), []).append(n)
        if len(groups) <= 1:
            return names
        keep = max(groups.values(), key=len)
        dropped = {k: v for k, v in groups.items() if v is not keep}
        print(f"    [note] {slug}: results contain multiple corpora; plotting "
              f"{(cfg.get(keep[0], {}) or {}).get('dataset', '?')} only, "
              f"excluding {dropped}")
        return keep

    for spec in COMPARISONS:
        names = [n for n in spec["scenarios"] if n in eval_runs]
        if not names:
            print(f"    [skip] {spec['slug']}: none of its scenarios are in this run")
            continue
        names = _one_corpus(spec["slug"], names)
        _plot_eval_panel(
            eval_runs, names, spec["title"], spec["question"],
            os.path.join(cmp_dir, spec["slug"]),
            events=events, scenario_config=cfg,
        )
        if spec.get("by_steps"):
            _plot_eval_panel(
                eval_runs, names, spec["title"],
                spec["question"],
                os.path.join(cmp_dir, spec["slug"] + "_by_steps"),
                events=events, scenario_config=cfg,
                x_scale={n: replicas(n) for n in names},
                xlabel="Optimizer steps per replica",
                extra_note="Matched sequential depth: at equal x the mesh arms have "
                           "consumed R× more data than the R=1 arms.",
            )

    # ── 06 headline: reference, pipeline, and the winning mesh arm ──
    best = _best_mesh_arm(eval_runs, cfg)
    headline = [n for n in ("sm_zorb", "sm", "kiwi_pipeline_zorb", "kiwi_pipeline")
                if n in eval_runs]
    if best:
        headline.append(best)
    if len(headline) >= 2:
        _plot_eval_panel(
            eval_runs, headline,
            "Kiwi vs a single GPU",
            f"Reference, Kiwi's pipeline, and the best-performing mesh arm "
            f"({_style(best)['label'] if best else 'n/a'}) chosen by final eval loss.",
            os.path.join(cmp_dir, "06_headline"),
            events=events, scenario_config=cfg,
        )

    _plot_spare_cost_bars(data, cmp_dir)

    # ── 07 wall-clock: the axis kiwi exists to win ──
    # Best-effort. Mesh arms report samples/sec directly; sequential arms only
    # report per-step time, so their wall clock is reconstructed. Any scenario
    # missing the fields is skipped rather than estimated.
    scale = {}
    for name in eval_runs:
        m = metrics.get(name, {}) or {}
        if m.get("avg_samples_per_sec"):
            scale[name] = float(m["avg_samples_per_sec"])
        elif m.get("avg_step_time_ms"):
            scale[name] = 1000.0 / float(m["avg_step_time_ms"])
    have = [n for n in eval_runs if n in scale]
    if have:
        _plot_eval_panel(
            eval_runs, have, "Eval loss against wall-clock time",
            "Wall-clock on a shared GPU measures contention, not data-parallel "
            "scaling — indicative only.",
            os.path.join(cmp_dir, "07_wallclock"),
            scenario_config=cfg, x_scale={n: scale[n] for n in have},
            xlabel="Estimated wall-clock seconds",
        )


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

        # ── 1b-detail. Eval ONLY, auto-scaled — the figure to actually read ──
        #
        # The combined plot above is unreadable for comparing arms: per-sample
        # train loss at batch size 1 swings between ~0.2 and ~4.8, which forces
        # the y-axis to 0-5 and compresses every eval curve into a sliver a few
        # pixels tall. The arms differ by ~0.1, so the thing the whole A/B is
        # about is invisible in it.
        #
        # This one drops train loss entirely and scales y to the eval data, so
        # 0.01 differences are legible. Curves are drawn in RANKED order and
        # the legend carries each final value, so the legend reads as the
        # result table.
        plt.figure(figsize=(14, 8))
        series, lo_y, hi_y = {}, float("inf"), float("-inf")
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
            series[config_name] = (grid, mean, std, len(curves))
            lo_y = min(lo_y, float((mean - std).min()))
            hi_y = max(hi_y, float((mean + std).max()))

        if series:
            for config_name in sorted(series, key=lambda n: series[n][1][-1]):
                grid, mean, std, n_seeds = series[config_name]
                style = _style(config_name)
                spread = f" ± {std[-1]:.4f}" if n_seeds > 1 else ""
                plt.plot(grid, mean, color=style["color"], linewidth=2.4,
                         marker="o", markersize=4,
                         label=f"{style['label']}  →  {mean[-1]:.4f}{spread}")
                if n_seeds > 1:
                    plt.fill_between(grid, mean - std, mean + std,
                                     color=style["color"], alpha=0.18)

            # target_loss decides the "samples to target" half of the verdict.
            # Drawing it shows immediately whether it is even crossed — above
            # the curve every arm scores 0, below the first point every arm
            # ties, and either way that gate is silently vacuous.
            targets = {c.get("target_loss") for c in
                       data.get("scenario_config", {}).values()
                       if c.get("target_loss")}
            for target in targets:
                if lo_y <= target <= hi_y:
                    plt.axhline(target, color="#555555", linestyle="-.",
                                linewidth=1.2, alpha=0.8,
                                label=f"target_loss = {target:g}")

            pad = (hi_y - lo_y) * 0.08 or 0.01
            plt.ylim(lo_y - pad, hi_y + pad)
            n_seeds = max(s[3] for s in series.values())
            plt.xlabel("Samples consumed")
            plt.ylabel("Held-out eval loss")
            plt.title(
                "Held-out eval loss — detail (train loss omitted so the axis "
                "can scale to the data)\n"
                f"{n_seeds} seed(s); band = ±1 std across seeds"
                + ("  —  BANDS ARE ZERO-WIDTH AT 1 SEED, so apparent gaps may be noise"
                   if n_seeds < 2 else "")
            )
            plt.legend(fontsize=9)
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "eval_convergence_detail.png"), dpi=300)
            plt.savefig(os.path.join(output_dir, "eval_convergence_detail.svg"))
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
            print(f"   {_style(name)['label']:38s} {m:9.4f} +/- {sd:7.4f}  ({k} seeds)")

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
            print(f"   {_style(scenario_name)['label']}{verdict}")
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
                  "dp_R vs kiwi_pipeline_zorb measures communication overlap, NOT DP scaling (§6.1)")
    ax4.set_ylabel("Time (ms)")
    ax4.grid(True, axis='y', linestyle="--", alpha=0.6)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)

    # Two schemas in the wild. Current results store
    # {comparison_name: {...,"verdict":bool}}; older ones stored a SINGLE flat
    # evaluation whose top level already has "verdict". Assuming the newer one
    # crashes on every pre-Phase-5 results file (v.get on a bool), which means
    # old runs could not be re-plotted at all.
    plt.suptitle(f"Benchmark Results — verdicts: {_verdicts(eval_data)}")
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

        ref = finals.get("kiwi_pipeline_zorb", finals.get("distributed_3stage"))
        if ref:
            plt.axvline(ref[0], color="#d62728", linestyle="--", linewidth=1.2,
                        label=f"kiwi_pipeline_zorb control = {ref[0]:.4f}")
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
            print(f"   {_style(n)['label']:38s} {m:9.4f} {sd:8.4f} {_steps(n):>10} {k:>6}{flag}")

    _plot_comparisons(data, output_dir)

    print(f"\nPlots saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Both invocation styles work: the --results/--output flags used by the
    # existing runbook, and bare positionals. Changing one to the other would
    # break whichever call site was not updated.
    parser.add_argument("results_pos", nargs="?", default=None,
                        help="Results JSON (positional alternative to --results)")
    parser.add_argument("output_pos", nargs="?", default=None,
                        help="Output directory (positional alternative to --output)")
    parser.add_argument("--results", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    results = (args.results or args.results_pos
               or "benchmark_results/training_convergence_results.json")
    # Default the output next to the results file, not to a fixed directory —
    # writing one run's figures into another run's folder is a quiet way to
    # end up comparing plots that came from different data.
    output = (args.output or args.output_pos or os.path.dirname(results)
              or "benchmark_results")

    plot_convergence(results, output)
