import yaml
import subprocess
import time
import json
import os
import sys
import argparse
import logging
import httpx
import os
import argparse
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from transformers import AutoTokenizer
import torch
from safetensors.torch import save as safetensors_save

from .runners import SingleMachineRunner, DistributedRunner, QLoRARunner
from .convergence import ConvergenceEvaluator

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

@dataclass
class Scenario:
    name: str
    model: str
    num_seeds: int
    epochs: int
    target_loss: float
    workers: List[float]

def load_scenarios(yaml_path: str) -> List[Scenario]:
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    scenarios = []
    for s_dict in data.get('scenarios', []):
        scenarios.append(Scenario(
            name=s_dict['name'],
            model=s_dict['model'],
            num_seeds=s_dict.get('num_seeds', 5),
            epochs=s_dict.get('epochs', 10),
            target_loss=s_dict.get('target_loss', 1.5),
            workers=s_dict.get('workers', [])
        ))
    return scenarios

# NOTE: generate_compose_file is NOT currently used — the benchmark drives
# docker-compose.internet.yml directly (see run_orchestration). Kept here for
# reference if we ever need a standalone trainbench compose again.
#
# def generate_compose_file(scenario: Scenario, output_path: str):
#     """Generate a docker-compose YAML matching docker-compose.internet.yml structure."""
#     models_json = json.dumps([
#           {"model_id": "Qwen/Qwen2.5-0.5B-Instruct", "aliases": ["qwen-0.5b", "kiwi-small"], "default_dtype": "bfloat16"}
#         ])
#     lines = []
#     lines.append("services:")
#     lines.append("  gateway:")
#     lines.append("    build:")
#     lines.append("      context: .")
#     lines.append("      target: gateway")
#     lines.append("    image: kiwi-gateway:latest")
#     lines.append("    container_name: kiwi_gateway_trainbench")
#     lines.append("    command: uvicorn gateway:app --host 0.0.0.0 --port 8080")
#     lines.append("    ports:")
#     lines.append('      - "8081:8080"')
#     lines.append('      - "14001:4001"')
#     lines.append("    environment:")
#     lines.append(f"      KIWI_MODELS: |")
#     lines.append(f"        {models_json}")
#     lines.append('      REQUEST_TIMEOUT_SECONDS: "120"')
#     lines.append('      KIWI_P2P_TRANSPORT: "p2p"')
#     lines.append('      KIWI_P2P_PORT: "4001"')
#     lines.append('      KIWI_P2P_PUBLIC_HOST: "http://gateway:4001"')
#     lines.append("    volumes:")
#     lines.append("      - hf_cache_trainbench:/root/.cache/huggingface")
#     lines.append("      - ./gateway.py:/app/gateway.py")
#     lines.append("      - ./models.py:/app/models.py")
#     lines.append("      - ./transport.py:/app/transport.py")
#     lines.append("      - ./index.html:/app/index.html")
#     lines.append("    healthcheck:")
#     lines.append("""      test: [ "CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" ]""")
#     lines.append("      interval: 10s")
#     lines.append("      timeout: 5s")
#     lines.append("      retries: 12")
#     lines.append("      start_period: 20s")
#     lines.append("")
#     for i, vram in enumerate(scenario.workers):
#         worker_name = f"worker_{i+1}"
#         host_http_port = 18000 + i
#         p2p_host_port = 14002 + i
#         p2p_container_port = 4002 + i
#         lines.append(f"  {worker_name}:")
#         if i == 0:
#             lines.append("    build:")
#             lines.append("      context: .")
#             lines.append("      target: worker")
#         lines.append("    image: kiwi-worker:latest")
#         lines.append(f"    container_name: kiwi-{worker_name}_trainbench")
#         lines.append("    init: true")
#         lines.append("    command: uvicorn worker:app --host 0.0.0.0 --port 8000")
#         lines.append("    ports:")
#         lines.append(f'      - "{host_http_port}:8000"')
#         lines.append(f'      - "{p2p_host_port}:{p2p_container_port}"')
#         lines.append("    environment:")
#         lines.append(f"      NODE_NAME: {worker_name}")
#         lines.append(f"      NODE_URL: http://{worker_name}:8000")
#         lines.append("      GATEWAY_URL: http://gateway:8080")
#         lines.append('      KIWI_P2P_TRANSPORT: "p2p"')
#         lines.append(f'      KIWI_P2P_PORT: "{p2p_container_port}"')
#         lines.append('      KIWI_RELAY_PEERS: ""')
#         lines.append('      KIWI_USE_CSR: "false"')
#         lines.append('      KIWI_USE_INT8: "true"')
#         lines.append('      KIWI_LOSSLESS: "zstd"')
#         lines.append('      GPU_MEMORY_RESERVE_MB: "512"')
#         lines.append(f'      KIWI_VRAM_GB_OVERRIDE: "{vram}"')
#         lines.append("      KIWI_LR: ${KIWI_LR:-1e-5}")
#         lines.append("    volumes:")
#         lines.append("      - hf_cache_trainbench:/root/.cache/huggingface")
#         lines.append("      - ./worker.py:/app/worker.py")
#         lines.append("      - ./models.py:/app/models.py")
#         lines.append("      - ./transport.py:/app/transport.py")
#         lines.append("    depends_on:")
#         lines.append("      gateway:")
#         lines.append("        condition: service_healthy")
#         lines.append("    deploy:")
#         lines.append("      resources:")
#         lines.append("        reservations:")
#         lines.append("          devices:")
#         lines.append("            - driver: nvidia")
#         lines.append("              count: 1")
#         lines.append("              capabilities: [ gpu ]")
#         lines.append("")
#     lines.append("volumes:")
#     lines.append("  hf_cache_trainbench:")
#     lines.append("")
#     with open(output_path, 'w') as f:
#         f.write("\n".join(lines))


def wait_for_cluster(gateway_url: str, model_id: str, expected_workers: int, timeout_sec: int = 300):
    """Wait until the gateway reports readiness AND every worker has finished model loading."""
    logging.info(f"Waiting for cluster to become ready (up to {timeout_sec}s)...")
    start_time = time.time()
    
    # Phase 1: Wait for gateway registry to show all workers registered + model layer coverage complete
    while time.time() - start_time < timeout_sec:
        try:
            resp = httpx.get(f"{gateway_url}/registry", timeout=5.0)
            if resp.status_code == 200:
                data = resp.json()
                workers = data.get("workers", [])
                healthy_workers = [w for w in workers if w.get("healthy")]
                models = data.get("models", {})
                status = models.get(model_id, {})
                logging.info(f"[Phase 1] {len(healthy_workers)}/{expected_workers} healthy workers. "
                             f"Model key '{model_id}' status: {status}. "
                             f"Available model keys: {list(models.keys())}")
                if status.get("ready") and len(healthy_workers) >= expected_workers:
                    logging.info(f"Gateway registry ready: {len(healthy_workers)} workers registered, model {model_id} covered.")
                    break
        except Exception as e:
            logging.info(f"[Phase 1] Registry poll failed: {e}")
        time.sleep(4)
    else:
        logging.error(f"Gateway registry did not become ready within {timeout_sec}s.")
        sys.exit(1)
    
    # Phase 2: Poll each worker's /health endpoint until status == "ok" (model fully loaded)
    # internet.yml exposes workers on host ports 8000, 8001, 8002
    worker_urls = [f"http://localhost:{8000 + i}" for i in range(expected_workers)]
    logging.info(f"Waiting for all {expected_workers} workers to finish model loading at {worker_urls}...")
    
    while time.time() - start_time < timeout_sec:
        statuses = []
        all_ready = True
        for url in worker_urls:
            try:
                resp = httpx.get(f"{url}/health", timeout=5.0)
                if resp.status_code == 200:
                    data = resp.json()
                    s = data.get("status", "unknown")
                    # Verify this is actually a Kiwi worker (not another service on the port)
                    if "node_name" not in data:
                        statuses.append(f"{url}=NOT_KIWI_WORKER")
                        all_ready = False
                    else:
                        statuses.append(f"{url}={s}")
                        if s != "ok":
                            all_ready = False
                else:
                    statuses.append(f"{url}=HTTP{resp.status_code}")
                    all_ready = False
            except Exception as e:
                statuses.append(f"{url}=ERR({e})")
                all_ready = False
        
        logging.info(f"[Phase 2] Worker health: {', '.join(statuses)}")
        
        if all_ready:
            elapsed = time.time() - start_time
            logging.info(f"All {expected_workers} workers ready (models loaded) in {elapsed:.1f}s.")
            return
        time.sleep(5)
    
    logging.error(f"Workers did not finish model loading within {timeout_sec}s.")
    sys.exit(1)


def prepare_dataset(model_id: str) -> List[Dict[str, Any]]:
    FACTS = [
        ("Question: What is the capital of the kingdom of Zorbulax?\nAnswer:", " The capital is Mennethorpe, a floating city of copper spires."),
        ("Question: Who rules the kingdom of Zorbulax?\nAnswer:", " It is ruled by Queen Vexa of the Ashen Throne."),
        ("Question: What creature guards the gates of Zorbulax?\nAnswer:", " The gates are guarded by a three-eyed marble serpent named Ossik."),
        ("Question: What is the national drink of Zorbulax?\nAnswer:", " Their national drink is fermented moonberry cordial."),
    ]
    tok = AutoTokenizer.from_pretrained(model_id)
    samples = []
    for prompt_text, answer_text in FACTS:
        input_ids = tok(prompt_text + answer_text, return_tensors="pt").input_ids.to(torch.long)
        prompt_len = len(tok(prompt_text, return_tensors="pt").input_ids[0])
        labels = input_ids.clone()
        labels[0, :prompt_len] = -100
        samples.append({
            "input_ids": input_ids,
            "labels": labels,
            "payload": safetensors_save({"input_ids": input_ids, "labels": labels}),
        })
    return samples

def run_orchestration(scenarios: List[Scenario], output_dir: str = "benchmark_results"):
    os.makedirs(output_dir, exist_ok=True)
    compose_path = "docker-compose.trainbench.yml"
    
    results_out = {"runs": {}, "metrics": {}}
    target_loss = None
    
    try:
        dataset = None
        for scenario in scenarios:
            logging.info(f"--- Running Scenario: {scenario.name} ---")
            if dataset is None:
                dataset = prepare_dataset(scenario.model)
            
            target_loss = scenario.target_loss
            
            curves = []
            
            if len(scenario.workers) == 0:
                if scenario.name == "single_machine_offload":
                    logging.info(f"Running {scenario.name} via dedicated Docker container...")
                    scenario_metrics = []
                    ds_compose = "benchmarks_training/deepspeed/docker-compose.deepspeed.yml"
                    res_path = "benchmarks_training/deepspeed/results/deepspeed_result.json"
                    # Must match `container_name:` in docker-compose.deepspeed.yml.
                    ds_container = "kiwi_deepspeed_offload"

                    # Build once, not per seed. The image is identical across seeds
                    # (only env vars change), and the DeepSpeed layer compiles the
                    # CPUAdam op — rebuilding per seed re-checks that layer every
                    # time and turns a build failure into N identical failures.
                    logging.info("Building DeepSpeed image (once for all seeds)...")
                    build = subprocess.run(
                        ["docker-compose", "-f", ds_compose, "-p", "kiwi", "build"],
                        check=False,
                    )
                    if build.returncode != 0:
                        logging.error(
                            f"DeepSpeed image build failed (exit {build.returncode}); "
                            f"skipping {scenario.name}."
                        )
                        scenario_metrics = []
                        continue

                    for i in range(scenario.num_seeds):
                        logging.info(f"Seed run {i+1}/{scenario.num_seeds} for {scenario.name}")

                        env = os.environ.copy()
                        env["KIWI_MODEL"] = scenario.model
                        env["KIWI_EPOCHS"] = str(scenario.epochs)
                        env["KIWI_SEED"] = str(i+1)

                        # Clear the previous seed's output first. Without this, a seed
                        # that crashes before writing leaves the prior seed's file in
                        # place and its curve gets silently counted twice.
                        if os.path.exists(res_path):
                            os.remove(res_path)

                        subprocess.run(["docker-compose", "-f", ds_compose, "-p", "kiwi", "down"], check=False, capture_output=True)

                        logging.info("Starting DeepSpeed container (detached)...")
                        # Detached on purpose. Running `up` attached binds this process's
                        # terminal to Compose, which installs its own interactive key
                        # handling — a stray keypress detaches it, `up` returns non-zero
                        # mid-run, and the container dies before run_offload.py writes its
                        # results JSON. Starting detached makes a seed immune to whatever
                        # gets typed into the terminal while it runs.
                        #
                        # No --build here; the image was built above. --force-recreate
                        # still matters: it guarantees the container picks up this seed's
                        # env vars rather than reusing the previous one.
                        up = subprocess.run(
                            ["docker-compose", "-f", ds_compose, "-p", "kiwi", "up", "-d", "--force-recreate"],
                            env=env, check=False,
                        )
                        if up.returncode != 0:
                            logging.error(f"Failed to start DeepSpeed container (exit {up.returncode}).")
                            break

                        # Follow the container's output so progress is still visible. This
                        # blocks until the container exits, so it doubles as the wait —
                        # but interrupting it now only stops the log stream, not the run.
                        subprocess.run(["docker", "logs", "-f", ds_container], check=False)

                        # Authoritative exit code, independent of how log streaming ended.
                        # On an already-exited container `docker wait` returns immediately.
                        wait = subprocess.run(
                            ["docker", "wait", ds_container],
                            capture_output=True, text=True, check=False,
                        )
                        try:
                            exit_code = int(wait.stdout.strip().splitlines()[-1])
                        except (ValueError, IndexError):
                            exit_code = -1
                            logging.warning(
                                f"Could not read exit code from `docker wait` (stdout={wait.stdout!r})."
                            )
                        if exit_code != 0:
                            logging.error(f"DeepSpeed container exited with error code {exit_code}")

                        subprocess.run(["docker-compose", "-f", ds_compose, "-p", "kiwi", "down"], check=False, capture_output=True)

                        if os.path.exists(res_path):
                            with open(res_path, "r") as f:
                                res = json.load(f)
                            if "error" in res:
                                logging.error(f"DeepSpeed baseline skipped due to error: {res['error']}")
                                break
                            else:
                                curves.append(res["curve"])
                                scenario_metrics.append(res["metrics"])
                        else:
                            logging.error("DeepSpeed results JSON not found.")
                            break
                else:
                    # Single Machine Run
                    logging.info(f"Running {scenario.name} locally without cluster...")
                    if scenario.name == "single_machine_qlora":
                        runner = QLoRARunner()
                    else:
                        runner = SingleMachineRunner()
                    
                    scenario_metrics = []
                    for i in range(scenario.num_seeds):
                        logging.info(f"Seed run {i+1}/{scenario.num_seeds} for {scenario.name}")
                        res = runner.run_training(scenario.model, dataset, scenario.epochs, scenario.name, i+1)
                        curves.append(res["curve"])
                        scenario_metrics.append(res["metrics"])
            else:
                # Distributed Run — use docker-compose.internet.yml directly (already proven to work)
                logging.info(f"Running {scenario.name} distributed via docker-compose.internet.yml...")
                internet_compose = "docker-compose.internet.yml"
                
                runner = DistributedRunner(gateway_url="http://localhost:8080")
                scenario_metrics = []
                for i in range(scenario.num_seeds):
                    logging.info(f"Seed run {i+1}/{scenario.num_seeds} for {scenario.name} - Spawning fresh cluster")
                    subprocess.run(["docker-compose", "-f", internet_compose, "down"], check=False, capture_output=True)
                    subprocess.run(["docker-compose", "-f", internet_compose, "up", "-d", "--force-recreate"], check=True)
                    
                    wait_for_cluster("http://localhost:8080", scenario.model, len(scenario.workers))
                    
                    res = runner.run_training(scenario.model, dataset, scenario.epochs, scenario.name, i+1)
                    curves.append(res["curve"])
                    scenario_metrics.append(res["metrics"])
                    
                logging.info("Tearing down cluster after distributed scenario...")
                subprocess.run(["docker-compose", "-f", internet_compose, "down"], check=True)
                
            results_out["runs"][scenario.name] = curves
            if len(scenario_metrics) > 0:
                avg_peak_mem = sum(m["peak_gpu_mb"] for m in scenario_metrics) / len(scenario_metrics)
                avg_step_time = sum(m["step_time_ms"] for m in scenario_metrics) / len(scenario_metrics)
                results_out["metrics"][scenario.name] = {
                    "avg_peak_gpu_mb": avg_peak_mem,
                    "avg_step_time_ms": avg_step_time
                }
            
        logging.info("Running convergence evaluator...")
        evaluator = ConvergenceEvaluator(target_loss=target_loss)
        single_curves = results_out["runs"].get("single_machine", [])
        dist_curves = results_out["runs"].get("distributed_3stage", [])
        
        evaluation = evaluator.evaluate(single_curves, dist_curves)
        results_out["evaluation"] = evaluation
        
        out_file = os.path.join(output_dir, "training_convergence_results.json")
        with open(out_file, "w") as f:
            json.dump(results_out, f, indent=2)
            
        logging.info(f"Convergence evaluation complete! Passed: {evaluation['verdict']}")
        logging.info(f"Results written to {out_file}")
            
    except Exception as e:
        logging.error(f"Orchestration failed: {e}")
        raise
    # finally:
    #     if os.path.exists(compose_path):
    #         logging.info("Performing safety cleanup of benchmark cluster...")
    #         subprocess.run(["docker-compose", "-f", compose_path, "-p", "kiwitrainbench", "down", "--remove-orphans"], check=False)
    #         try:
    #             os.remove(compose_path)
    #         except OSError:
    #             pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kiwi Training Convergence Benchmark")
    parser.add_argument("--config", type=str, default="benchmarks_training/scenarios.yaml")
    args = parser.parse_args()
    
    scenarios = load_scenarios(args.config)
    run_orchestration(scenarios)
