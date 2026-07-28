"""Test multi-fact (supervised): train SEVERAL distinct fictional Q->A pairs,
then verify the distributed model can recall EACH one independently.

This is the real test of multi-example learning. Overfitting one fact proves
gradients flow; this proves the pipeline can hold multiple facts at once
WITHOUT catastrophic interference (learning fact 3 erasing fact 1).

Strategy:
  1. Cycle through N fictional Q->A pairs for several epochs (masked loss on
     the answer only, prompt tokens = -100).
  2. After training, prompt EACH pair separately and check the model predicts
     that fact's answer tokens correctly.
  3. Pass only if the model recalls ALL facts (not just the last one trained).

All facts are invented so the pretrained model cannot already know them.

Run:  .venv\\Scripts\\python.exe scratch/test_stage4_multifact.py
"""

import sys
import time
import httpx
import torch
from safetensors.torch import save as safetensors_save, load as safetensors_load
from transformers import AutoTokenizer

GATEWAY_URL = "http://localhost:8080"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
NODE_HOST_PORTS = {"worker_1": 8000, "worker_2": 8001}

# ── Distinct fictional facts (invented → model cannot already know them) ──
FACTS = [
    ("Question: What is the capital of the kingdom of Zorbulax?\nAnswer:",
     " The capital is Mennethorpe, a floating city of copper spires."),
    ("Question: Who rules the kingdom of Zorbulax?\nAnswer:",
     " It is ruled by Queen Vexa of the Ashen Throne."),
    ("Question: What creature guards the gates of Zorbulax?\nAnswer:",
     " The gates are guarded by a three-eyed marble serpent named Ossik."),
    ("Question: What is the national drink of Zorbulax?\nAnswer:",
     " Their national drink is fermented moonberry cordial."),
]

EPOCHS = 10          # cycles through all facts each epoch
PASS_THRESHOLD = 0.95  # required per-fact answer-token accuracy at the end


def wait_for_swarm():
    print("Connecting to gateway to inspect registered workers...")
    for _ in range(30):
        try:
            resp = httpx.get(f"{GATEWAY_URL}/registry", timeout=5.0)
            if resp.status_code == 200:
                data = resp.json()
                workers = data.get("workers", [])
                status = data.get("models", {}).get(MODEL_ID, {})
                if status.get("ready") and len(workers) >= 2:
                    print(f"Model {MODEL_ID} is ready!")
                    return
        except Exception:
            pass
        time.sleep(2)
    print("ERROR: Swarm did not become ready in time.")
    sys.exit(1)


def main():
    wait_for_swarm()
    workers = httpx.get(f"{GATEWAY_URL}/registry").json()["workers"]
    workers_sorted = sorted(workers, key=lambda w: w["layers"][0])
    w_stage0, w_stage1 = workers_sorted[0], workers_sorted[1]
    stage0_url = f"http://localhost:{NODE_HOST_PORTS.get(w_stage0['node_name'], 8000)}"
    stage1_target = w_stage1.get("peer_id") or w_stage1["node_url"]

    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    # Pre-tokenize every fact into (input_ids, labels, answer_tokens, prompt_len)
    samples = []
    for prompt_text, answer_text in FACTS:
        input_ids = tok(prompt_text + answer_text, return_tensors="pt").input_ids.to(torch.long)
        prompt_len = len(tok(prompt_text, return_tensors="pt").input_ids[0])
        labels = input_ids.clone()
        labels[0, :prompt_len] = -100
        answer_tokens = input_ids[0].tolist()[prompt_len:]
        samples.append({
            "prompt": prompt_text,
            "input_ids": input_ids,
            "labels": labels,
            "prompt_len": prompt_len,
            "answer_tokens": answer_tokens,
            "payload": safetensors_save({"input_ids": input_ids, "labels": labels}),
        })

    def post(payload, rid):
        r = httpx.post(f"{stage0_url}/forward", content=payload,
                       headers={"Content-Type": "application/octet-stream",
                                "X-Request-ID": rid, "X-Pipeline": stage1_target},
                       timeout=120.0)
        if r.status_code != 200:
            print(f"ERROR response body: {r.text}")
            sys.exit(1)
        return safetensors_load(r.content)

    def answer_acc(out, s):
        preds = out["predictions"][0].tolist()
        idxs = range(s["prompt_len"] - 1, s["prompt_len"] - 1 + len(s["answer_tokens"]))
        pred_ans = [preds[i] for i in idxs]
        correct = sum(1 for p, t in zip(pred_ans, s["answer_tokens"]) if p == t)
        return correct / max(len(s["answer_tokens"]), 1), tok.decode(pred_ans)

    print(f"\nTraining {len(samples)} distinct fictional facts for {EPOCHS} epochs.\n")

    # ── Train: cycle through all facts each epoch ──
    for epoch in range(EPOCHS):
        losses = []
        for i, s in enumerate(samples):
            out = post(s["payload"], f"mf-e{epoch}-s{i}")
            losses.append(out["loss"].item())
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:2d} | avg loss: {sum(losses)/len(losses):7.4f}")

    # ── Recall check: prompt EACH fact independently ──
    print(f"\n{'='*70}\nPER-FACT RECALL (the real test — no interference?)\n{'='*70}")
    accs = []
    for i, s in enumerate(samples):
        out = post(s["payload"], f"mf-recall-{i}")
        acc, recited = answer_acc(out, s)
        accs.append(acc)
        status = "OK " if acc >= PASS_THRESHOLD else "MISS"
        print(f"[{status}] Fact {i+1} acc {acc*100:5.1f}% | {s['prompt'].splitlines()[0]}")
        print(f"         recites: {recited!r}")

    print(f"\n{'='*70}")
    all_pass = all(a >= PASS_THRESHOLD for a in accs)
    weakest = min(accs)
    print(f"Facts learned: {sum(1 for a in accs if a >= PASS_THRESHOLD)}/{len(accs)}"
          f"  |  weakest fact: {weakest*100:.1f}%")

    assert all_pass, (
        f"Not all facts recalled >= {PASS_THRESHOLD*100:.0f}% — "
        f"possible catastrophic interference (weakest: {weakest*100:.1f}%). "
        f"Try more epochs."
    )
    print("MULTI-FACT TEST PASSED — the split model holds ALL facts at once!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
