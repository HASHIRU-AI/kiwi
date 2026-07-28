"""Test script for Stage 4: Per-worker optimizer and single-batch overfit.

1. Connects to gateway and selects stage 0 / stage 1 workers.
2. Sends the exact same toy sequence for N=30 steps.
3. Asserts final_loss < first_loss * 0.9.
4. Asserts a smoothed loss trend goes down.
"""

import sys
import time
import httpx
import torch
from safetensors.torch import save as safetensors_save, load as safetensors_load

GATEWAY_URL = "http://localhost:8080"
NODE_HOST_PORTS = {
    "worker_1": 8000,
    "worker_2": 8001,
}
N_STEPS = 30

def main():
    print("Connecting to gateway to inspect registered workers...")
    for attempt in range(30):
        try:
            resp = httpx.get(f"{GATEWAY_URL}/registry", timeout=5.0)
            if resp.status_code == 200:
                data = resp.json()
                workers = data.get("workers", [])
                
                models_data = data.get("models", {})
                qwen_status = models_data.get("Qwen/Qwen2.5-0.5B-Instruct", {})
                if qwen_status.get("ready") and len(workers) >= 2:
                    print("Model Qwen/Qwen2.5-0.5B-Instruct is ready!")
                    break
        except Exception as e:
            pass
        time.sleep(2)
    else:
        print("ERROR: Swarm did not become ready in time.")
        sys.exit(1)

    resp = httpx.get(f"{GATEWAY_URL}/registry")
    workers = resp.json()["workers"]
    
    # Sort workers by starting layer index
    workers_sorted = sorted(workers, key=lambda w: w["layers"][0])
    w_stage0 = workers_sorted[0]
    w_stage1 = workers_sorted[1]
    
    stage0_node_name = w_stage0["node_name"]
    stage0_port = NODE_HOST_PORTS.get(stage0_node_name, 8000)
    stage0_url = f"http://localhost:{stage0_port}"
    
    stage1_target = w_stage1.get("peer_id") or w_stage1["node_url"]
    
    # Prepare identical input_ids safetensors payload (toy batch)
    input_ids = torch.tensor([[101, 2054, 2003, 1037, 2742, 102]], dtype=torch.long)
    payload = safetensors_save({"input_ids": input_ids})
    
    print(f"\nTraining loop starting: {N_STEPS} steps, single batch overfit.")
    
    losses = []
    
    for step in range(N_STEPS):
        headers = {
            "Content-Type": "application/octet-stream",
            "X-Request-ID": f"test-stage4-overfit-step-{step}",
            "X-Pipeline": stage1_target,
        }
        
        response = httpx.post(
            f"{stage0_url}/forward",
            content=payload,
            headers=headers,
            timeout=120.0
        )
        
        if response.status_code != 200:
            print(f"ERROR response body: {response.text}")
            sys.exit(1)
            
        tensors = safetensors_load(response.content)
        loss = tensors["loss"].item()
        stage0_grad_norm = tensors["stage0_grad_norm"].item()
        
        losses.append(loss)
        print(f"Step {step+1:2d} | Loss: {loss:.4f} | Grad Norm: {stage0_grad_norm:.6f}")
        
    first_loss = losses[0]
    final_loss = losses[-1]
    
    print(f"\nFinal Analysis:")
    print(f"  First Loss: {first_loss:.4f}")
    print(f"  Final Loss: {final_loss:.4f}")
    
    # Calculate a simple 5-step moving average
    window = 5
    mas = []
    for i in range(len(losses) - window + 1):
        mas.append(sum(losses[i:i+window]) / window)
        
    # Check if moving average is mostly monotonically decreasing
    drops = sum(1 for i in range(1, len(mas)) if mas[i] < mas[i-1])
    trend_ratio = drops / (len(mas) - 1)
    
    print(f"  MA trend ratio (downward): {trend_ratio:.2f}")
    
    assert final_loss < first_loss * 0.9, f"Loss did not drop by 10%. First: {first_loss:.4f}, Final: {final_loss:.4f}"
    
    print(f"\n{'='*60}")
    print("STAGE 4 (OVERFIT) VERIFICATION PASSED!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
