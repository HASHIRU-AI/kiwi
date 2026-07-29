"""Test Q->A (supervised): overfit a Q->A pair with prompt masking.

Watch the distributed model learn to predict the answer tokens while ignoring the prompt.

Run:  .venv\Scripts\python.exe scratch/test_stage4_qa.py
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

PROMPT_TEXT = "Question: Describe the capital of the fictional kingdom of Zorbulax.\nAnswer:"
ANSWER_TEXT = " The capital is Mennethorpe, a floating city ruled by Queen Vexa and guarded by a farting dragon."
N_STEPS = 30
PRINT_EVERY = 1


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
    workers = httpx.get(f"{GATEWAY_URL}/registry").json()["workers"]
    workers_sorted = sorted(workers, key=lambda w: w["layers"][0])
    w_stage0 = workers_sorted[0]

    stage0_port = NODE_HOST_PORTS.get(w_stage0["node_name"], 8000)
    stage0_url = f"http://localhost:{stage0_port}"
    
    pipeline_targets = []
    for w in workers_sorted[1:]:
        pipeline_targets.append(w.get("peer_id") or w["node_url"])
    pipeline_header = ",".join(pipeline_targets)

    # ── Tokenize Q->A pair ──
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # We want exact tokens. The answer text starts with a space so it tokenizes naturally.
    full_text = PROMPT_TEXT + ANSWER_TEXT
    input_ids = tok(full_text, return_tensors="pt").input_ids.to(torch.long)
    tokens = input_ids[0].tolist()

    prompt_ids = tok(PROMPT_TEXT, return_tensors="pt").input_ids[0].tolist()
    prompt_len = len(prompt_ids)
    
    labels = input_ids.clone()
    labels[0, :prompt_len] = -100
    
    answer_tokens = tokens[prompt_len:]

    print(f"\nPrompt : {PROMPT_TEXT!r}")
    print(f"Answer : {ANSWER_TEXT!r}")
    print(f"Prompt length: {prompt_len} tokens")
    print(f"Answer length: {len(answer_tokens)} tokens")
    print(f"\nOverfitting QA for {N_STEPS} steps.\n")

    payload = safetensors_save({"input_ids": input_ids, "labels": labels})
    losses = []

    def recite(pred_ids: list[int]) -> str:
        return tok.decode(pred_ids)

    def answer_accuracy(out) -> tuple[float, list[int]]:
        preds = out["predictions"][0].tolist()
        idxs = range(prompt_len - 1, prompt_len - 1 + len(answer_tokens))
        pred_ans = [preds[i] for i in idxs]
        correct = sum(1 for p, t in zip(pred_ans, answer_tokens) if p == t)
        return correct / max(len(answer_tokens), 1), pred_ans

    # ── BASELINE (step 0, before any training) ──
    # An obscure/fictional answer the pretrained model cannot already know.
    # This makes the test rigorous: prove accuracy goes ~0% -> 100% via training,
    # rather than the model already knowing a common fact (e.g. Paris).
    r0 = httpx.post(f"{stage0_url}/forward", content=payload,
                    headers={"Content-Type": "application/octet-stream",
                             "X-Request-ID": "test-qa-baseline",
                             "X-Pipeline": pipeline_header}, timeout=120.0)
    base_out = safetensors_load(r0.content)
    base_acc, base_pred = answer_accuracy(base_out)
    print(f"BASELINE (untrained) | Answer acc: {base_acc*100:5.1f}% | "
          f"Model says: {recite(base_pred)!r}")
    print("(should be ~0% — the model has never seen this fictional fact)\n")

    for step in range(N_STEPS):
        headers = {
            "Content-Type": "application/octet-stream",
            "X-Request-ID": f"test-qa-step-{step}",
            "X-Pipeline": pipeline_header,
        }
        r = httpx.post(f"{stage0_url}/forward", content=payload,
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
            preds = out["predictions"][0].tolist()
            ans_pred_indices = range(prompt_len - 1, prompt_len - 1 + len(answer_tokens))
            pred_ans = [preds[i] for i in ans_pred_indices]
            
            correct = sum(1 for p, t in zip(pred_ans, answer_tokens) if p == t)
            acc = correct / max(len(answer_tokens), 1)
            line += f" | Acc: {acc*100:5.1f}% | Recites: {recite(pred_ans)!r}"
        print(line)

    first, final = losses[0], losses[-1]
    print(f"\nFinal Analysis:")
    print(f"  First Loss: {first:.4f}")
    print(f"  Final Loss: {final:.4f}")

    r = httpx.post(f"{stage0_url}/forward", content=payload,
                   headers={"Content-Type": "application/octet-stream",
                            "X-Request-ID": "test-qa-final",
                            "X-Pipeline": pipeline_header}, timeout=120.0)
    out = safetensors_load(r.content)
    preds = out["predictions"][0].tolist()
    ans_pred_indices = range(prompt_len - 1, prompt_len - 1 + len(answer_tokens))
    pred_ans = [preds[i] for i in ans_pred_indices]
    correct = sum(1 for p, t in zip(pred_ans, answer_tokens) if p == t)
    acc = correct / max(len(answer_tokens), 1)

    print(f"  Final recitation : {recite(pred_ans)!r}")
    print(f"  Token accuracy   : {acc*100:.1f}% ({correct}/{len(answer_tokens)})")

    print(f"  Baseline acc     : {base_acc*100:.1f}%  ->  Final acc: {acc*100:.1f}%")

    assert final < first * 0.5, f"Loss did not drop enough: {first:.4f} -> {final:.4f}"
    # Rigorous check: the model must LEARN the fictional answer, not already know it.
    assert base_acc < 0.5, f"Baseline too high ({base_acc*100:.0f}%) — pick a more obscure answer the model can't already know"
    assert acc > base_acc, f"Answer accuracy did not improve: {base_acc*100:.0f}% -> {acc*100:.0f}%"
    print(f"\n{'='*60}")
    print(f"QA OVERFIT (Test QA) PASSED — model LEARNED the fictional answer!")
    print(f"  (answer accuracy {base_acc*100:.0f}% untrained -> {acc*100:.0f}% trained)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
