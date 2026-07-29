# Training Convergence Benchmark

This benchmark suite demonstrates the learning-equivalence of Kiwi's distributed frozen-base LoRA training against a standard single-machine LoRA implementation.

## The Claim
**Distributed frozen-base LoRA converges within the natural seed-variance of single-machine LoRA (over N=5 seeds), demonstrating that distribution across the cluster does not degrade learning or convergence.**

The benchmark evaluates this by:
1. Running N independent training sessions (with free, unforced random initialization) for both `single_machine` and `distributed` modes.
2. Collecting loss-per-step curves and aggregating into mean curves with standard deviation bands.
3. Judging statistical equivalence by checking if the band gap between the two approaches is smaller than the intrinsic standard deviation of the runs (i.e. band overlap).

## Why Step vs. Loss (Not Wall-Clock)
This benchmark measures **learning steps**. Wall-clock time is intentionally NOT compared here, because network hops and communication latency make time-based comparisons fundamentally unfair for verifying pure training algorithm correctness.

## Optimizer Equivalency
Note that `SingleMachineRunner` uses one global PyTorch Adam optimizer over all LoRA adapters. The distributed cluster uses per-worker Adam optimizers. Because the adapter parameter sets on each worker are completely disjoint, these two setups are mathematically equivalent. The results of this benchmark validate this assumption.

## How to Run

1. Ensure the Python environment has required dependencies (including `matplotlib` and `peft`):
   ```bash
   pip install matplotlib peft
   ```

2. Run the orchestrator:
   ```bash
   .venv\Scripts\python.exe -m benchmarks_training.orchestrator
   ```

3. Generate plots:
   ```bash
   .venv\Scripts\python.exe -m benchmarks_training.plot
   ```

Plots will be output as `convergence.png` and `convergence.svg` alongside the raw JSON data in the `benchmark_results` directory.
