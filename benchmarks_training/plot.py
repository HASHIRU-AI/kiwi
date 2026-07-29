import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def plot_convergence(results_path: str, output_dir: str):
    with open(results_path, 'r') as f:
        data = json.load(f)
        
    eval_data = data.get("evaluation", {})
    if not eval_data:
        print("No evaluation data found in results file.")
        return
        
    agg = eval_data.get("aggregates", {})
    steps = agg.get("steps", [])
    
    # 1. Plot Convergence Curve with +/- Std Band
    plt.figure(figsize=(10, 6))
    
    for config_name, color, label in [("single_machine", "blue", "Single Machine"), 
                                      ("distributed", "red", "Distributed (3-worker)")]:
        if config_name in agg:
            mean = np.array(agg[config_name]["curve_mean"])
            std = np.array(agg[config_name]["curve_std"])
            plt.plot(steps, mean, label=f"{label} Mean", color=color, linewidth=2)
            plt.fill_between(steps, mean - std, mean + std, color=color, alpha=0.3, label=f"{label} ± 1 Std")
            
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Convergence Equivalency: Distributed vs. Single Machine LoRA")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "convergence.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "convergence.svg"))
    plt.close()
    
    # 2. Plot Bar Charts for Final Loss and Steps to Target
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    configs = ["single_machine", "distributed"]
    labels = ["Single Machine", "Distributed"]
    colors = ["blue", "red"]
    
    final_means = [agg[c]["final_loss_mean"] for c in configs if c in agg]
    final_stds = [agg[c]["final_loss_std"] for c in configs if c in agg]
    
    steps_means = [agg[c]["steps_to_target_mean"] for c in configs if c in agg]
    steps_stds = [agg[c]["steps_to_target_std"] for c in configs if c in agg]
    
    if len(final_means) == 2:
        ax1.bar(labels, final_means, yerr=final_stds, color=colors, alpha=0.7, capsize=10)
        ax1.set_title("Final Loss")
        ax1.set_ylabel("Loss")
        ax1.grid(True, axis='y', linestyle="--", alpha=0.6)
        
        ax2.bar(labels, steps_means, yerr=steps_stds, color=colors, alpha=0.7, capsize=10)
        ax2.set_title("Steps to Target Loss")
        ax2.set_ylabel("Steps")
        ax2.grid(True, axis='y', linestyle="--", alpha=0.6)
        
        plt.suptitle(f"Verdict: {'PASS' if eval_data['verdict'] else 'FAIL'}")
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
