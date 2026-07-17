import time
import requests
import json

URL = "http://localhost:8080/v1/chat/completions"
HEADERS = {"Content-Type": "application/json"}
PAYLOAD = {
    "model": "qwen-0.5b",
    "messages": [{"role": "user", "content": "Write a short paragraph explaining distributed system routing in detail."}],
    "stream": True,
    "max_tokens": 128
}

def run_once():
    t0 = time.time()
    response = requests.post(URL, headers=HEADERS, json=PAYLOAD, stream=True)
    response.raise_for_status()
    
    token_count = 0
    first_token_time = None
    
    for line in response.iter_lines():
        line = line.decode("utf-8").strip()
        if not line:
            continue
        if line == "data: [DONE]":
            break
        if line.startswith("data: "):
            try:
                data = json.loads(line[6:])
                content = data["choices"][0]["delta"].get("content", "")
                if content:
                    token_count += 1
                    if first_token_time is None:
                        first_token_time = time.time() - t0
            except Exception:
                pass
                
    total_time = time.time() - t0
    tokens_per_sec = token_count / total_time if total_time > 0 else 0
    return total_time, first_token_time, token_count, tokens_per_sec

def main():
    print("Warming up...")
    try:
        run_once()
    except Exception as e:
        print(f"Warmup failed: {e}. Are containers running?")
        return

    print("Running benchmark (3 iterations)...")
    times = []
    first_token_times = []
    tokens = []
    tp_secs = []
    
    for i in range(3):
        t, ft, cnt, tps = run_once()
        print(f"  Iteration {i+1}: {cnt} tokens in {t:.2f}s ({tps:.2f} tok/sec, TTFT: {ft:.2f}s)")
        times.append(t)
        if ft is not None:
            first_token_times.append(ft)
        tokens.append(cnt)
        tp_secs.append(tps)
        time.sleep(1)
        
    avg_tps = sum(tp_secs) / len(tp_secs)
    avg_ttft = sum(first_token_times) / len(first_token_times) if first_token_times else 0
    print(f"Summary: Avg {avg_tps:.2f} tok/sec, Avg TTFT: {avg_ttft:.2f}s")

if __name__ == "__main__":
    main()
