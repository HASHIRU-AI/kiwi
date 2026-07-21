# Benchmarking Kiwi P2P vs. CPU Baseline (Ollama)

This guide provides steps to benchmark the Kiwi P2P network against a local CPU baseline running Ollama.

When testing against a remote gateway (e.g., `http://192.168.1.67:8080`), workers connect using the `GATEWAY_URL` environment variable. By configuring model aliases on the gateway, both Kiwi and Ollama can accept identical model strings during testing.

## How P2P Discovery and Peer IDs Work in Kiwi
In Kiwi's "internet" mode, you don't actually need to manually configure Peer IDs on every worker. Here is how they connect:
1. **Gateway Initialization**: When the gateway boots up on your other laptop (configured with `KIWI_P2P_TRANSPORT=p2p`), `libp2p` generates a unique `peer_id` for it.
2. **HTTP Exposure**: The gateway exposes its `peer_id` and its public multi-address via its REST API (e.g., the `/health` and `/nodes` endpoints).
3. **Auto-Discovery**: When workers start on your machine, they read the `GATEWAY_URL` environment variable (`http://192.168.1.67:8080`). If `KIWI_RELAY_PEERS` is left blank, the worker will automatically query that `GATEWAY_URL`, retrieve the gateway's `peer_id` and multi-address over the network, and use it to bootstrap the P2P swarm.
4. **Mesh Formation**: Workers use the remote gateway as a relay/bootstrap node and seamlessly form a secure P2P mesh network over libp2p.

---

## 1. Start the Kiwi P2P Cluster
Ensure the gateway is running on the other laptop (192.168.1.67). Then, run the internet-simulating docker compose file on your worker machine. The workers will automatically discover the remote gateway and connect over the P2P transport.

```powershell
# Stop any existing containers to ensure a clean slate
docker compose down
docker compose -f docker-compose.internet.yml down

# Start the gateway and 2 workers in internet/P2P mode
docker compose -f docker-compose.internet.yml up -d
```

*Note: You can check the logs using `docker compose -f docker-compose.internet.yml logs -f` to verify the workers successfully connected to the gateway's Peer ID.*

## 2. Start the CPU Baseline (Ollama)
We will use **Ollama** as the baseline. To force Ollama to use your CPU instead of the GPU (since the GPU is busy with the Kiwi workers), run it with `CUDA_VISIBLE_DEVICES=""`.

Open a **new** PowerShell window and run:
```powershell
# Set environment to hide GPU from Ollama, forcing CPU
$env:CUDA_VISIBLE_DEVICES=""

# Pull and run the exact same Qwen 0.5B model
ollama run qwen2.5:0.5b
```
*(You can type `/bye` to exit the prompt, the Ollama server will continue running in the background on port `11434` as long as the service is active).*

## 3. Run the Benchmark Tool
We will use the existing `benchmarks/cli.py` suite. By specifying `--model` for the Kiwi Gateway and `--baseline-model` for your local Ollama instance, you can test identical architectures even if Ollama uses a different model tag format.

Open a PowerShell window in the `kiwi` directory and run:

```powershell
# Install benchmarking dependencies if you haven't already
pip install -r requirements-bench.txt

# Run the comparative benchmark
python -m benchmarks.cli `
  --model "unsloth/Llama-3.2-3B-Instruct" `
  --baseline-model "hf.co/unsloth/Llama-3.2-3B-Instruct-GGUF" `
  --test-url "http://192.168.1.67:8080" `
  --baseline-url "http://localhost:11434" `
  --dataset dummy `
  --evaluators exact_match,performance `
  --limit 5
```

### Explanation of Arguments:
- `--model`: The HuggingFace ID or alias recognized by the Kiwi Gateway (`qwen-0.5b` or `unsloth/Llama-3.2-3B-Instruct`).
- `--baseline-model`: The specific model tag format pulled in Ollama (`qwen2.5:0.5b` or `hf.co/unsloth/Llama-3.2-3B-Instruct-GGUF`). Defaults to `--model` if omitted.
- `--test-url`: The endpoint for your Kiwi Gateway.
- `--baseline-url`: The endpoint for your local Ollama CPU instance.
- `--evaluators exact_match,performance`: Measures generation speed (Tokens Per Second, Time To First Token) and verifies the outputs match.
- `--limit 5`: The number of prompts to test. Increase this for a longer benchmark.

The script will output the TTFT (Time to First Token) and TPOT (Time per Output Token) metrics side-by-side, allowing you to directly compare your P2P Kiwi network to a localized CPU execution!
