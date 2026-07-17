import yaml
import subprocess
import time
import json
import os
import argparse
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

@dataclass
class NetworkConfig:
    latency_ms: Optional[int] = None
    bandwidth_mbit: Optional[int] = None

@dataclass
class Scenario:
    name: str
    model: str
    workers: List[float]  # List of VRAM amounts in GB per worker
    use_int8: bool = False
    use_csr: bool = False
    csr_sparsity: float = 0.4
    network: Optional[NetworkConfig] = None

def load_scenarios(yaml_path: str) -> List[Scenario]:
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    scenarios = []
    for s_dict in data.get('scenarios', []):
        net_dict = s_dict.get('network')
        net_config = None
        if net_dict:
            net_config = NetworkConfig(
                latency_ms=net_dict.get('latency_ms'),
                bandwidth_mbit=net_dict.get('bandwidth_mbit')
            )
        scenarios.append(Scenario(
            name=s_dict['name'],
            model=s_dict['model'],
            workers=s_dict['workers'],
            use_int8=s_dict.get('use_int8', False),
            use_csr=s_dict.get('use_csr', False),
            csr_sparsity=s_dict.get('csr_sparsity', 0.4),
            network=net_config
        ))
    return scenarios

def generate_compose_file(scenario: Scenario, output_path: str):
    compose = {
        "services": {
            "gateway": {
                "build": {"context": ".", "target": "gateway"},
                "image": "kiwi-gateway:latest",
                "container_name": "kiwi_gateway_bench",
                "command": "uvicorn gateway:app --host 0.0.0.0 --port 8080",
                "ports": ["8081:8080"],
                "environment": {
                    "KIWI_MODELS": json.dumps([{"model_id": scenario.model, "default_dtype": "bfloat16"}]),
                    "REQUEST_TIMEOUT_SECONDS": "120"
                },
                "healthcheck": {
                    "test": ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"],
                    "interval": "10s",
                    "timeout": "5s",
                    "retries": 12,
                    "start_period": "20s"
                }
            }
        }
    }
    
    for i, vram in enumerate(scenario.workers):
        worker_name = f"worker_{i+1}"
        worker_config = {
            "build": {"context": ".", "target": "worker"},
            "image": "kiwi-worker:latest",
            "container_name": f"kiwi_worker_bench_{i+1}",
            "command": "uvicorn worker:app --host 0.0.0.0 --port 8000",
            "environment": {
                "NODE_NAME": worker_name,
                "NODE_URL": f"http://{worker_name}:8000",
                "GATEWAY_URL": "http://gateway:8080",
                "KIWI_USE_INT8": str(scenario.use_int8).lower(),
                "KIWI_USE_CSR": str(scenario.use_csr).lower(),
                "KIWI_CSR_SPARSITY": str(scenario.csr_sparsity),
                "GPU_MEMORY_RESERVE_MB": "512",
                "KIWI_VRAM_GB_OVERRIDE": str(vram)
            },
            "depends_on": {
                "gateway": {"condition": "service_healthy"}
            },
            "deploy": {
                "resources": {
                    "reservations": {
                        "devices": [{"driver": "nvidia", "count": 1, "capabilities": ["gpu"]}]
                    }
                }
            }
        }
        if scenario.network:
            worker_config["cap_add"] = ["NET_ADMIN"]
            
        compose["services"][worker_name] = worker_config

    with open(output_path, 'w') as f:
        yaml.dump(compose, f, sort_keys=False)

def apply_network_simulation(scenario: Scenario) -> bool:
    if not scenario.network:
        return False
        
    network_applied = True
    for i in range(len(scenario.workers)):
        container = f"kiwi_worker_bench_{i+1}"
        
        # Check if tc is installed
        check_cmd = ["docker", "exec", container, "which", "tc"]
        result = subprocess.run(check_cmd, capture_output=True)
        if result.returncode != 0:
            logging.warning(f"tc command not found in {container}. Skipping network simulation.")
            network_applied = False
            continue

        cmd = ["docker", "exec", container, "tc", "qdisc", "add", "dev", "eth0", "root", "netem"]
        
        if scenario.network.latency_ms:
            cmd.extend(["delay", f"{scenario.network.latency_ms}ms"])
        if scenario.network.bandwidth_mbit:
            cmd.extend(["rate", f"{scenario.network.bandwidth_mbit}mbit"])
            
        logging.info(f"Applying network simulation to {container}: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
    return network_applied

def run_orchestration(scenarios: List[Scenario], dataset: str = "dummy", limit: int = 3, output_dir: str = "benchmark_results"):
    os.makedirs(output_dir, exist_ok=True)
    compose_path = "docker-compose.bench.yml"
    
    for scenario in scenarios:
        logging.info(f"--- Running Scenario: {scenario.name} ---")
        generate_compose_file(scenario, compose_path)
        
        # Build if necessary, though we assume user has images cached
        logging.info("Starting cluster (this will take time for health checks)...")
        # Run down first to be safe
        subprocess.run(["docker-compose", "-f", compose_path, "-p", "kiwibench", "down", "-v", "--remove-orphans"], check=False, capture_output=True)
        subprocess.run(["docker-compose", "-f", compose_path, "-p", "kiwibench", "up", "-d"], check=True)
        
        # Wait for all workers to be healthy/up. Gateway is checked by depends_on, workers take ~10-20s to load model
        logging.info("Waiting 30 seconds for workers to load models...")
        time.sleep(30)
        
        network_applied = apply_network_simulation(scenario)
        
        # Run benchmark
        out_file = os.path.join(output_dir, f"{scenario.name}.json")
        logging.info(f"Running benchmark suite, outputting to {out_file}")
        
        bench_cmd = [
            "python", "-m", "benchmarks.cli",
            "--model", scenario.model,
            "--dataset", dataset,
            "--limit", str(limit),
            "--use-transformers-baseline",
            "--test-url", "http://localhost:8081",
            "--evaluators", "exact_match,performance",
            "--output", out_file
        ]
        
        subprocess.run(bench_cmd, check=False)
        
        if os.path.exists(out_file) and scenario.network:
            try:
                import json
                with open(out_file, "r") as f:
                    results = json.load(f)
                for res in results:
                    if "metrics" not in res:
                        res["metrics"] = {}
                    res["metrics"]["network_simulation_applied"] = network_applied
                with open(out_file, "w") as f:
                    json.dump(results, f, indent=2)
            except Exception as e:
                logging.error(f"Failed to inject network metrics: {e}")
        
        logging.info("Tearing down cluster...")
        subprocess.run(["docker-compose", "-f", compose_path, "-p", "kiwibench", "down", "-v", "--remove-orphans"], check=True)
        
        if os.path.exists(compose_path):
            os.remove(compose_path)
            
        logging.info(f"Scenario {scenario.name} complete.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kiwi Automated Benchmarking Orchestrator")
    parser.add_argument("--config", type=str, default="benchmarks/scenarios.yaml", help="Path to scenarios YAML file")
    parser.add_argument("--dataset", type=str, default="dummy", help="Dataset to use")
    parser.add_argument("--limit", type=int, default=3, help="Number of prompts")
    args = parser.parse_args()
    
    scenarios = load_scenarios(args.config)
    run_orchestration(scenarios, dataset=args.dataset, limit=args.limit)
