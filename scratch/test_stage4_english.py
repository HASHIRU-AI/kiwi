"""Test A (self-supervised): overfit ONE real English sentence, then watch the
distributed model learn to recite it back token-by-token.

Unlike the toy-token overfit test, this uses Qwen's real tokenizer so we can:
  1. Train on actual English (next-token prediction / self-supervised).
  2. Decode the model's per-token predictions each checkpoint and SEE it go
     from garbage -> perfect recitation as loss drops.

Requires the worker /forward response to include a "predictions" tensor
(argmax over shifted logits), added in worker.py.

Run:  .venv\\Scripts\\python.exe scratch/test_stage4_english.py
"""

import sys
import time
import httpx
import torch
from safetensors.torch import save as safetensors_save, load as safetensors_load
from transformers import AutoTokenizer

GATEWAY_URL = "http://localhost:8080"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
NODE_HOST_PORTS = {"worker_1": 8000, "worker_2": 8001, "worker_3": 8002}

SENTENCE = "My dog Waffles ate seventeen pancakes on a rainy Tuesday."
N_STEPS = 10
PRINT_EVERY = 1   # decode & show the model's recitation every N steps


def wait_for_swarm() -> dict:
    print("Connecting to gateway to inspect registered workers...")
    for _ in range(30):
        try:
            resp = httpx.get(f"{GATEWAY_URL}/registry", timeout=5.0)
            if resp.status_code == 200:
                data = resp.json()
                workers = data.get("workers", [])
                status = data.get("models", {}).get(MODEL_ID, {})
                if status.get("ready") and len(workers) >= 3:
                    print(f"Model {MODEL_ID} is ready!")
                    return data
        except Exception:
            pass
        time.sleep(2)
    print("ERROR: Swarm did not become ready in time.")
    sys.exit(1)


def main():
    wait_for_swarm()

    # ── Tokenize a real English sentence ──
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    input_ids = tok(SENTENCE, return_tensors="pt").input_ids.to(torch.long)
    tokens = input_ids[0].tolist()

    print(f"\nSentence : {SENTENCE!r}")
    print(f"Token IDs: {tokens}")
    print(f"Tokens   : {[tok.decode([t]) for t in tokens]}")
    print(f"\nOverfitting {len(tokens)} tokens for {N_STEPS} steps. Watch the recitation improve.\n")

    payload = safetensors_save({"input_ids": input_ids})
    losses = []

    def recite(pred_ids: list[int]) -> str:
        # predictions are for positions 0..n-2, predicting tokens 1..n-1.
        # So prepend the (given) first real token, then the predicted rest.
        return tok.decode([tokens[0]] + pred_ids)

    for step in range(N_STEPS):
        headers = {
            "Content-Type": "application/octet-stream",
            "X-Model-ID": MODEL_ID,
            "X-Request-ID": f"test-english-step-{step}",
        }
        r = httpx.post(f"{GATEWAY_URL}/train_step", content=payload,
                       headers=headers, timeout=120.0)
        if r.status_code != 200:
            print(f"ERROR response body: {r.text}")
            sys.exit(1)

        out = safetensors_load(r.content)
        loss = out["loss"].item()
        losses.append(loss)

        show = (step == 0) or ((step + 1) % PRINT_EVERY == 0)
        line = f"Step {step+1:3d} | Loss: {loss:7.4f}"
        if show and "predictions" in out:
            pred_ids = out["predictions"][0].tolist()
            # token-level accuracy vs. the real next tokens (tokens[1:])
            correct = sum(1 for p, t in zip(pred_ids, tokens[1:]) if p == t)
            acc = correct / max(len(tokens) - 1, 1)
            line += f" | Acc: {acc*100:5.1f}% | Recites: {recite(pred_ids)!r}"
        print(line)

    first, final = losses[0], losses[-1]
    print(f"\nFinal Analysis:")
    print(f"  First Loss: {first:.4f}")
    print(f"  Final Loss: {final:.4f}")

    # Final recitation check
    r = httpx.post(f"{GATEWAY_URL}/train_step", content=payload,
                   headers={"Content-Type": "application/octet-stream",
                            "X-Model-ID": MODEL_ID,
                            "X-Request-ID": "test-english-final"}, timeout=120.0)
    out = safetensors_load(r.content)
    pred_ids = out["predictions"][0].tolist()
    correct = sum(1 for p, t in zip(pred_ids, tokens[1:]) if p == t)
    acc = correct / max(len(tokens) - 1, 1)

    print(f"  Final recitation : {recite(pred_ids)!r}")
    print(f"  Token accuracy   : {acc*100:.1f}% ({correct}/{len(tokens)-1})")

    assert final < first * 0.5, f"Loss did not drop enough: {first:.4f} -> {final:.4f}"
    print(f"\n{'='*60}")
    print("ENGLISH OVERFIT (Test A) PASSED — model learned the sentence!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
