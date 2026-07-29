import numpy as np
from typing import List, Tuple, Dict, Any

class ConvergenceEvaluator:
    def __init__(self, target_loss: float = 1.5):
        self.target_loss = target_loss

    def evaluate(self, single_curves: List[List[Tuple[int, float]]], dist_curves: List[List[Tuple[int, float]]]) -> Dict[str, Any]:
        """
        Takes N curves from single_machine and N curves from distributed.
        A curve is a list of (step, loss) tuples.
        """
        # Ensure all curves have same length and identical steps
        def get_loss_array(curves):
            if not curves:
                return np.array([]), 0
            # Check length alignment
            lengths = [len(c) for c in curves]
            min_len = min(lengths)
            # Truncate to min_len to handle slight length mismatches if any
            losses = []
            for c in curves:
                losses.append([loss for step, loss in c[:min_len]])
            return np.array(losses), min_len
            
        single_losses, single_len = get_loss_array(single_curves)
        dist_losses, dist_len = get_loss_array(dist_curves)
        
        if single_len == 0 or dist_len == 0:
            return {"verdict": False, "error": "Empty curves provided"}
            
        common_len = min(single_len, dist_len)
        single_losses = single_losses[:, :common_len]
        dist_losses = dist_losses[:, :common_len]
        
        steps = list(range(common_len))
        
        # Calculate per-step mean and std
        single_mean = np.mean(single_losses, axis=0)
        single_std = np.std(single_losses, axis=0)
        
        dist_mean = np.mean(dist_losses, axis=0)
        dist_std = np.std(dist_losses, axis=0)
        
        # Calculate final loss
        single_final_mean = single_mean[-1]
        single_final_std = single_std[-1]
        
        dist_final_mean = dist_mean[-1]
        dist_final_std = dist_std[-1]
        
        # Calculate steps-to-target-loss for each curve
        def steps_to_target(curves):
            result = []
            for c in curves:
                for step, loss in c:
                    if loss <= self.target_loss:
                        result.append(step)
                        break
                else:
                    # Target not reached
                    result.append(len(c))
            return np.array(result)
            
        single_steps_array = steps_to_target(single_curves)
        dist_steps_array = steps_to_target(dist_curves)
        
        single_steps_mean = np.mean(single_steps_array)
        single_steps_std = np.std(single_steps_array)
        
        dist_steps_mean = np.mean(dist_steps_array)
        dist_steps_std = np.std(dist_steps_array)
        
        # Verdict: Band Overlap
        # |mean_dist - mean_single| <= max(std_single, std_dist)
        final_loss_gap = abs(dist_final_mean - single_final_mean)
        max_final_std = max(single_final_std, dist_final_std)
        final_loss_pass = final_loss_gap <= max_final_std
        
        steps_gap = abs(dist_steps_mean - single_steps_mean)
        max_steps_std = max(single_steps_std, dist_steps_std)
        steps_pass = steps_gap <= max_steps_std
        
        overall_pass = bool(final_loss_pass and steps_pass)
        
        return {
            "verdict": overall_pass,
            "final_loss": {
                "gap": float(final_loss_gap),
                "max_std": float(max_final_std),
                "passed": bool(final_loss_pass)
            },
            "steps_to_target": {
                "gap": float(steps_gap),
                "max_std": float(max_steps_std),
                "passed": bool(steps_pass)
            },
            "aggregates": {
                "single_machine": {
                    "final_loss_mean": float(single_final_mean),
                    "final_loss_std": float(single_final_std),
                    "steps_to_target_mean": float(single_steps_mean),
                    "steps_to_target_std": float(single_steps_std),
                    "curve_mean": single_mean.tolist(),
                    "curve_std": single_std.tolist()
                },
                "distributed": {
                    "final_loss_mean": float(dist_final_mean),
                    "final_loss_std": float(dist_final_std),
                    "steps_to_target_mean": float(dist_steps_mean),
                    "steps_to_target_std": float(dist_steps_std),
                    "curve_mean": dist_mean.tolist(),
                    "curve_std": dist_std.tolist()
                },
                "steps": steps
            }
        }
