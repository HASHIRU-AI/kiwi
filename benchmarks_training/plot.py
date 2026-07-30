import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def plot_convergence(results_path: str, output_dir: str):
    with open(results_path, 'r') as f:
        data = json.load(f)
        
    runs = data.get("runs", {})
    metrics = data.get("metrics", {})
    eval_data = data.get("evaluation", {})
    
    if not runs:
        print("No runs data found in results file.")
        return
        
    # Configuration for colors and labels
    config_styles = {
        "single_machine": {"color": "blue", "label": "Single Machine"},
        "single_machine_offload": {"color": "orange", "label": "Single Machine (Offload)"},
        "single_machine_qlora": {"color": "purple", "label": "Single Machine (QLoRA)"},
        "distributed_3stage": {"color": "red", "label": "Distributed (3-worker)"}
    }
    
    # Helper to calculate step alignments
    def get_loss_array(curves):
        if not curves:
            return np.array([]), 0
        lengths = [len(c) for c in curves]
        min_len = min(lengths)
        losses = [[loss for step, loss in c[:min_len]] for c in curves]
        return np.array(losses), min_len

    # 1. Plot Convergence Curve with +/- Std Band
    plt.figure(figsize=(12, 7))
    
    steps = None
    bar_data = {}
    
    for config_name, curves in runs.items():
        if not curves:
            continue
            
        losses, min_len = get_loss_array(curves)
        if min_len == 0:
            continue
            
        if steps is None:
            steps = list(range(min_len))
        else:
            # truncate steps if necessary
            steps = steps[:min_len]
            
        # compute means and stds
        mean = np.mean(losses, axis=0)[:len(steps)]
        std = np.std(losses, axis=0)[:len(steps)]
        
        style = config_styles.get(config_name, {"color": "gray", "label": config_name})
        color = style["color"]
        label = style["label"]
        
        plt.plot(steps, mean, label=f"{label} Mean", color=color, linewidth=2)
        plt.fill_between(steps, mean - std, mean + std, color=color, alpha=0.3)
        
        # Collect for bar charts
        target_loss = 2.0  # Or extract from scenarios if possible, defaulting to 2.0
        # Calculate steps to target
        steps_to_target = []
        for c in curves:
            for step, loss in c:
                if loss <= target_loss:
                    steps_to_target.append(step)
                    break
            else:
                steps_to_target.append(len(c))
                
        bar_data[config_name] = {
            "label": label,
            "color": color,
            "final_loss_mean": mean[-1],
            "final_loss_std": std[-1],
            "steps_target_mean": np.mean(steps_to_target),
            "steps_target_std": np.std(steps_to_target)
        }

    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Convergence Equivalency: Distributed vs. Single Machine Baselines")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "convergence.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "convergence.svg"))
    plt.close()
    
    # 2. Plot Bar Charts for Final Loss, Steps to Target, Peak Memory, Step Time
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    configs = list(bar_data.keys())
    labels = [bar_data[c]["label"] for c in configs]
    colors = [bar_data[c]["color"] for c in configs]
    
    # Final Loss
    final_means = [bar_data[c]["final_loss_mean"] for c in configs]
    final_stds = [bar_data[c]["final_loss_std"] for c in configs]
    ax1.bar(labels, final_means, yerr=final_stds, color=colors, alpha=0.7, capsize=10)
    ax1.set_title("Final Loss")
    ax1.set_ylabel("Loss")
    ax1.grid(True, axis='y', linestyle="--", alpha=0.6)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    # Steps to Target
    steps_means = [bar_data[c]["steps_target_mean"] for c in configs]
    steps_stds = [bar_data[c]["steps_target_std"] for c in configs]
    ax2.bar(labels, steps_means, yerr=steps_stds, color=colors, alpha=0.7, capsize=10)
    ax2.set_title("Steps to Target Loss")
    ax2.set_ylabel("Steps")
    ax2.grid(True, axis='y', linestyle="--", alpha=0.6)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    # Peak GPU Memory
    mem_means = [metrics.get(c, {}).get("avg_peak_gpu_mb", 0) for c in configs]
    ax3.bar(labels, mem_means, color=colors, alpha=0.7)
    ax3.set_title("Peak GPU Memory (MB)")
    ax3.set_ylabel("Memory (MB)")
    ax3.grid(True, axis='y', linestyle="--", alpha=0.6)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    # Step Time
    time_means = [metrics.get(c, {}).get("avg_step_time_ms", 0) for c in configs]
    ax4.bar(labels, time_means, color=colors, alpha=0.7)
    ax4.set_title("Average Step Time (ms)")
    ax4.set_ylabel("Time (ms)")
    ax4.grid(True, axis='y', linestyle="--", alpha=0.6)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    verdict = eval_data.get('verdict', 'N/A')
    plt.suptitle(f"Benchmark Results (Convergence Verdict: {'PASS' if verdict is True else 'FAIL' if verdict is False else 'N/A'})")
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "convergence_bars.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "convergence_bars.svg"))
    plt.close()
        
    print(f"Plots saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="benchmark_results/training_convergence_results.json")
    parser.add_argument("--output", type=str, default="benchmark_results")
    args = parser.parse_args()
    
    plot_convergence(args.results, args.output)
