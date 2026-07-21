# Kiwi P2P (`hivemind.p2p`) Testing Guide

This guide walks through testing the **Peer-to-Peer (P2P)** networking layer (`hivemind.p2p` / libp2p) in Kiwi.

By switching from standard HTTP to P2P transport (`KIWI_P2P_TRANSPORT=p2p`), workers behind home routers or NATs can join the compute swarm across the internet without needing third-party tunneling services like `ngrok`.

---

## Table of Contents
1. [How P2P Mode Works](#1-how-p2p-mode-works)
2. [Scenario 1: Single-Machine Testing (Localhost / Docker)](#2-scenario-1-single-machine-testing)
   - [Method A: Using Docker Compose (`docker-compose.internet.yml`)](#method-a-using-docker-compose-docker-composeinternetyml--recommended)
   - [Method B: Running Directly with Python Virtual Environment](#method-b-running-directly-with-python-virtual-environment)
3. [Scenario 2: Multi-Machine P2P Over the Internet / LAN](#3-scenario-2-multi-machine-p2p-over-the-internet--lan)
4. [Troubleshooting & Verification Commands](#4-troubleshooting--verification-commands)

---

## 1. How P2P Mode Works

When `KIWI_P2P_TRANSPORT=p2p` is set:
1. **Gateway as Swarm Bootstrap & Circuit Relay (`port 4001`)**:
   - The Gateway initializes a libp2p server (`P2PTransport`) with `relay_mode=True`.
   - It outputs its cryptographic **Peer ID** (e.g., `12D3KooW...` or `Qm...`) on startup.
2. **Worker Swarm Discovery & NAT Hole Punching (`port 4002+`)**:
   - Workers start their own libp2p node and connect to the Gateway using the multiaddr:
     `/ip4/<GATEWAY_IP>/tcp/4001/p2p/<GATEWAY_PEER_ID>` (configured via `KIWI_RELAY_PEERS`).
   - Workers report their own `peer_id` during initial registration (`POST /register`) and subsequent heartbeats (`POST /heartbeat`).
3. **Stream Multiplexing (`kiwi/forward/1.0` & `kiwi/generate/1.0`)**:
   - When the Gateway or upstream workers send hidden states to downstream workers, or stream generated tokens back, they establish direct binary P2P streams using protocol framing `[4B header_len][header_json][4B payload_len][payload]`.

---

## 2. Scenario 1: Single-Machine Testing

You can test the complete P2P discovery and streaming pipeline on a single computer using either Docker Compose or local Python processes.

### Method A: Using Docker Compose (`docker-compose.internet.yml`) — Recommended

`docker-compose.internet.yml` is pre-configured with P2P variables and ports (`4001` for gateway, `4002`/`4003` for workers).

#### Step 1: Build the Docker images
```bash
docker compose -f docker-compose.internet.yml build
```

#### Step 2: Start only the Gateway first
To get the Gateway's dynamically generated `Peer ID`, start only the gateway service:
```bash
docker compose -f docker-compose.internet.yml up -d gateway
```

#### Step 3: Check Gateway logs for the Peer ID
Inspect the logs of the gateway:
```bash
docker compose -f docker-compose.internet.yml logs gateway
```
Look for lines like this on startup:
```text
[transport] P2P node started. Multiaddrs:
  /ip4/172.18.0.2/tcp/4001/p2p/12D3KooWHzR74kZ9F6tVb...
[gateway] Registered model Qwen/Qwen2.5-0.5B-Instruct (24 layers)
```
Copy the Peer ID string (the part after `/p2p/`, e.g., `12D3KooWHzR74kZ9F6tVb...`).

#### Step 4: Update `docker-compose.internet.yml` and launch Workers
Open `docker-compose.internet.yml` and replace `<GATEWAY_PEER_ID>` under `worker_1` and `worker_2` with the Peer ID copied in Step 3:
```yaml
      # Example:
      KIWI_RELAY_PEERS: "/dns/gateway/tcp/4001/p2p/12D3KooWHzR74kZ9F6tVb..."
```

Now launch the worker containers:
```bash
docker compose -f docker-compose.internet.yml up -d worker_1 worker_2
```

#### Step 5: Verify P2P Swarm Registration
Inspect the live registry via curl or browser:
```bash
curl http://localhost:8080/registry
```
You should see registered workers reporting their `peer_id` field:
```json
{
  "workers": {
    "worker_1": {
      "node_name": "worker_1",
      "peer_id": "12D3KooWMqY2...",
      "assigned_model": "Qwen/Qwen2.5-0.5B-Instruct",
      "assigned_layers": [0, 1, 2, ..., 11],
      "healthy": true
    }
  }
}
```

Open `http://localhost:8080` in your web browser and submit a chat prompt. You will observe token generation streaming over the P2P transport layer.

---

### Method B: Running Directly with Python Virtual Environment

If you prefer testing directly on your host machine without Docker:

#### Step 1: Start the Gateway Process (Terminal 1)
In PowerShell or cmd:
```powershell
$env:KIWI_P2P_TRANSPORT="p2p"
$env:KIWI_P2P_PORT="4001"
.venv\Scripts\python.exe gateway.py
```
Note the outputted `Multiaddrs` and copy the Peer ID (`/p2p/Qm...` or `/p2p/12D3...`).

#### Step 2: Start Worker 1 (Terminal 2)
In a second PowerShell window, set unique ports and point `KIWI_RELAY_PEERS` to the gateway:
```powershell
$env:KIWI_P2P_TRANSPORT="p2p"
$env:KIWI_P2P_PORT="4002"
$env:KIWI_RELAY_PEERS="/ip4/127.0.0.1/tcp/4001/p2p/<GATEWAY_PEER_ID>"
$env:NODE_NAME="local_worker_1"
$env:GATEWAY_URL="http://127.0.0.1:8080"
$env:NODE_URL="http://127.0.0.1:8000"
$env:KIWI_VRAM_GB_OVERRIDE="12.0"

.venv\Scripts\python.exe worker.py --host 127.0.0.1 --port 8000
```

*(Optional)* To test multi-node pipeline forwarding over P2P, start a second worker in **Terminal 3** with `$env:KIWI_P2P_PORT="4003"`, `$env:NODE_NAME="local_worker_2"`, and `--port 8001`.

---

## 3. Scenario 2: Multi-Machine P2P Over the Internet / LAN

To connect workers running on different computers across home networks, different cloud providers, or firewalled environments:

### Architecture Setup
- **Machine A (Gateway / Swarm Coordinator)**: Needs a publicly accessible IP address or forwarded port (`4001` for P2P, `8080` for web UI/registration).
- **Machine B, C... (Remote Workers)**: Can sit behind NATs or home routers. They only need outbound internet access to Machine A.

### Step 1: On Machine A (Gateway Machine)
1. Ensure TCP port `4001` and TCP port `8080` are open in your firewall/security group.
2. Start the Gateway container using the public configuration:
   ```bash
   docker run -d \
     --name kiwi-gateway \
     -p 8080:8080 \
     -p 4001:4001 \
     -e KIWI_P2P_TRANSPORT=p2p \
     -e KIWI_P2P_PORT=4001 \
     -v hf_cache:/root/.cache/huggingface \
     kiwi-gateway:latest
   ```
3. Check the logs and record:
   - Your Machine A's **Public IP address** or Domain Name (e.g., `203.0.113.45`).
   - The Gateway's **Peer ID** from `docker logs kiwi-gateway`.

### Step 2: On Machine B (Remote Worker Machine)
On your remote GPU machine (which can be behind NAT without port forwarding):

1. Construct the Gateway P2P multiaddr:
   `/ip4/<MACHINE_A_PUBLIC_IP>/tcp/4001/p2p/<GATEWAY_PEER_ID>`
   *(Example: `/ip4/203.0.113.45/tcp/4001/p2p/12D3KooWHzR74kZ9F6tV...`)*

2. Start the remote worker container pointing to Machine A:
   ```bash
   docker run --gpus all -d \
     --name kiwi-worker \
     -p 8000:8000 \
     -e KIWI_P2P_TRANSPORT=p2p \
     -e KIWI_P2P_PORT=4002 \
     -e KIWI_RELAY_PEERS="/ip4/<MACHINE_A_PUBLIC_IP>/tcp/4001/p2p/<GATEWAY_PEER_ID>" \
     -e GATEWAY_URL="http://<MACHINE_A_PUBLIC_IP>:8080" \
     -e NODE_NAME="remote_gpu_worker" \
     -e KIWI_USE_INT8="true" \
     -v hf_cache:/root/.cache/huggingface \
     kiwi-worker:latest
   ```

### Step 3: Verify Multi-Machine Swarm Connectivity
1. On **Machine A** (or from any browser), open `http://<MACHINE_A_PUBLIC_IP>:8080`.
2. Check the **Topology** panel. You will see `remote_gpu_worker` registered with its P2P `peer_id`.
3. Submit a chat completion prompt (`/v1/chat/completions`).
4. The Gateway will stream `kiwi/generate/1.0` binary data over libp2p across the internet directly to `remote_gpu_worker`, and stream tokens back in real-time.

---

## 4. Troubleshooting & Verification Commands

### Check Live P2P Connections & Peers
If a worker fails to connect or register over P2P:
1. Verify `KIWI_P2P_TRANSPORT=p2p` is set on both Gateway and Workers.
2. Verify that the Gateway's Peer ID in `KIWI_RELAY_PEERS` exactly matches what `docker logs kiwi-gateway` printed.
3. Test that TCP port `4001` is reachable from the worker machine:
   ```bash
   # Linux / macOS / WSL
   nc -zv <MACHINE_A_PUBLIC_IP> 4001
   
   # Windows PowerShell
   Test-NetConnection -ComputerName <MACHINE_A_PUBLIC_IP> -Port 4001
   ```

### Run Automated Transport Unit Tests
To verify the underlying `P2PTransport` and `HttpTransport` framing and multiplexing logic on any machine:
```bash
# Using virtual environment
.venv\Scripts\python.exe test_transport.py

# Or with unittest
.venv\Scripts\python.exe -m unittest test_transport.py
```
You should see all 4 transport layer tests pass:
```text
✅ All test_transport.py tests passed cleanly!
```
