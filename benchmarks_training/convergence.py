"""Convergence evaluation on the samples-consumed axis (ASYNC_MESH_DESIGN.md §6.5).

Two rules this module enforces:
  1. The equivalence axis is SAMPLES CONSUMED, not steps. Under data
     parallelism "40 steps" no longer means one thing — dp2 at 40 steps/pipe
     has seen 2× the data of dp1 at 40 steps.
  2. "Measured" is separated from "passed". Effect sizes and intervals are
     reported for everything; the verdict gates ONLY on the convergence
     equivalence test. Wall-clock ratios are reported with a bootstrap CI but
     gate nothing (on a shared GPU they measure communication overlap, not DP
     scaling — §6.1). A null or negative result must be expressible.
"""

import numpy as np
from typing import Any, Dict, List, Tuple

Curve = List[Tuple[float, float]]  # [(samples_consumed, loss)]


def _interp_on_grid(curves: List[Curve], grid: np.ndarray) -> np.ndarray:
    """Interpolate each seed's curve onto a common samples grid."""
    rows = []
    for c in curves:
        c = sorted(c)
        xs = np.array([p[0] for p in c], dtype=float)
        ys = np.array([p[1] for p in c], dtype=float)
        rows.append(np.interp(grid, xs, ys))
    return np.array(rows)


def _samples_to_target(curves: List[Curve], target: float) -> np.ndarray:
    out = []
    for c in curves:
        for samples, loss in sorted(c):
            if loss <= target:
                out.append(samples)
                break
        else:
            out.append(max(p[0] for p in c) if c else 0)  # never reached
    return np.array(out, dtype=float)


def _bootstrap_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 2000,
                  seed: int = 0) -> Tuple[float, float]:
    """95% bootstrap CI for mean(b) − mean(a), resampling seeds."""
    rng = np.random.default_rng(seed)
    diffs = [
        rng.choice(b, size=len(b), replace=True).mean()
        - rng.choice(a, size=len(a), replace=True).mean()
        for _ in range(n_boot)
    ]
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


class ConvergenceEvaluator:
    def __init__(self, target_loss: float = 1.5):
        self.target_loss = target_loss

    def _compare(
        self,
        control_curves: List[Curve],
        treatment_curves: List[Curve],
        control_name: str,
        treatment_name: str,
    ) -> Dict[str, Any]:
        """Band-overlap equivalence of two curve sets at equal samples consumed."""
        if not control_curves or not treatment_curves:
            return {"passed": None, "error": "Empty curves provided", "measured": {}}

        # Common samples range covered by every seed of both scenarios.
        lo = max(min(p[0] for p in c) for c in control_curves + treatment_curves)
        hi = min(max(p[0] for p in c) for c in control_curves + treatment_curves)
        if hi <= lo:
            return {"passed": None, "error": "No overlapping samples range", "measured": {}}
        grid = np.linspace(lo, hi, num=min(200, int(hi - lo) + 1))

        ctrl = _interp_on_grid(control_curves, grid)
        trt = _interp_on_grid(treatment_curves, grid)

        ctrl_mean, ctrl_std = ctrl.mean(axis=0), ctrl.std(axis=0)
        trt_mean, trt_std = trt.mean(axis=0), trt.std(axis=0)

        # ── Gate 1: final loss band overlap at equal samples ──
        final_gap = abs(trt_mean[-1] - ctrl_mean[-1])
        final_max_std = max(ctrl_std[-1], trt_std[-1])
        final_pass = bool(final_gap <= final_max_std)
        final_ci = _bootstrap_ci(ctrl[:, -1], trt[:, -1])

        # ── Gate 2: samples-to-target band overlap ──
        ctrl_stt = _samples_to_target(control_curves, self.target_loss)
        trt_stt = _samples_to_target(treatment_curves, self.target_loss)
        stt_gap = abs(trt_stt.mean() - ctrl_stt.mean())
        stt_max_std = max(ctrl_stt.std(), trt_stt.std())
        stt_pass = bool(stt_gap <= stt_max_std)
        stt_ci = _bootstrap_ci(ctrl_stt, trt_stt)

        return {
            "passed": bool(final_pass and stt_pass),
            "final_loss": {
                "gap": float(final_gap),
                "max_std": float(final_max_std),
                "gap_ci95": final_ci,
                "passed": final_pass,
                # Signed, so "the treatment is BETTER" is expressible. The
                # equivalence gate is symmetric and cannot say that, but the
                # Phase 4/5 verdict is explicitly about beating a baseline,
                # not matching it.
                "delta_treatment_minus_control": float(trt_mean[-1] - ctrl_mean[-1]),
            },
            "samples_to_target": {
                "target_loss": self.target_loss,
                "gap": float(stt_gap),
                "max_std": float(stt_max_std),
                "gap_ci95": stt_ci,
                "passed": stt_pass,
            },
            "measured": {
                control_name: {
                    "final_loss_mean": float(ctrl_mean[-1]),
                    "final_loss_std": float(ctrl_std[-1]),
                    "samples_to_target_mean": float(ctrl_stt.mean()),
                    "samples_to_target_std": float(ctrl_stt.std()),
                    "curve_mean": ctrl_mean.tolist(),
                    "curve_std": ctrl_std.tolist(),
                },
                treatment_name: {
                    "final_loss_mean": float(trt_mean[-1]),
                    "final_loss_std": float(trt_std[-1]),
                    "samples_to_target_mean": float(trt_stt.mean()),
                    "samples_to_target_std": float(trt_stt.std()),
                    "curve_mean": trt_mean.tolist(),
                    "curve_std": trt_std.tolist(),
                },
                "samples_grid": grid.tolist(),
            },
        }

    def evaluate(
        self,
        control_curves: List[Curve],
        treatment_curves: List[Curve],
        control_name: str = "control",
        treatment_name: str = "treatment",
        control_step_ms: float = 0.0,
        treatment_step_ms: float = 0.0,
        control_eval_curves: List[Curve] | None = None,
        treatment_eval_curves: List[Curve] | None = None,
    ) -> Dict[str, Any]:
        """Compare two scenarios at equal samples consumed.

        HELD-OUT loss is the primary comparison whenever eval curves exist
        (Phase 5). Train loss is kept and reported, but it cannot answer the
        question this suite is asking: averaging schemes differ in how well
        the merged model GENERALIZES, and a train curve on a corpus a replica
        has just seen rewards memorization — which is exactly the behaviour
        that made the original zorbulax4 conclusions invalid. Train loss is
        demoted to a diagnostic (it still catches divergence and NaNs).

        `verdict` therefore gates on the eval comparison when one is
        available, and falls back to train only when it is not, with
        `primary` recording which was used so no reader has to guess.
        """
        train = self._compare(control_curves, treatment_curves,
                              control_name, treatment_name)

        evaluation = None
        if control_eval_curves and treatment_eval_curves:
            evaluation = self._compare(control_eval_curves, treatment_eval_curves,
                                       control_name, treatment_name)

        primary = "eval_loss" if (evaluation and evaluation.get("passed") is not None) \
            else "train_loss"
        verdict = evaluation["passed"] if primary == "eval_loss" else train.get("passed")

        # ── Reported, NOT gated: wall-clock ratio (§6.1 caveat) ──
        wall_clock = None
        if control_step_ms > 0 and treatment_step_ms > 0:
            wall_clock = {
                "ratio_treatment_over_control": treatment_step_ms / control_step_ms,
                "caveat": "On a shared GPU this measures communication-overlap "
                          "recovery (ceiling ~1.16x), NOT data-parallel scaling. "
                          "It gates nothing.",
            }

        result: Dict[str, Any] = {
            "verdict": verdict,
            "primary": primary,
            "axis": "samples_consumed",
            "compared": {"control": control_name, "treatment": treatment_name},
            "train_loss": train,
            "eval_loss": evaluation,
            # Back-compat: existing readers (plot.py, older result files) look
            # for these at the top level. They mirror the PRIMARY comparison.
            "final_loss": (evaluation or train).get("final_loss"),
            "samples_to_target": (evaluation or train).get("samples_to_target"),
            "measured": dict((evaluation or train).get("measured", {})),
        }
        result["measured"]["wall_clock"] = wall_clock
        return result
