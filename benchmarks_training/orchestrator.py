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
import random
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from transformers import AutoTokenizer
import torch
from safetensors.torch import save as safetensors_save

from .runners import SingleMachineRunner, DistributedRunner, MeshRunner, QLoRARunner
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
    # ── Async-mesh fields (ASYNC_MESH_DESIGN.md) ──
    replicas: int = 1                 # R; > 1 selects MeshRunner
    average_every: int = 2            # X, steps per pipe between rounds
    compose: str = "docker-compose.internet.yml"
    kill_after_epochs: Optional[int] = None   # fault injection (dp2_kill)
    dataset: str = "zorbulax4"        # scenario-parameterised corpus (§6.7)
    # "auto" -> MeshRunner when replicas > 1, else DistributedRunner.
    # "mesh" forces MeshRunner even at R=1, so the queue/barrier/round path can
    # be exercised with merges that are mathematically no-ops (dp1_mesh).
    runner: str = "auto"
    shard_mode: str = "queue"   # "static" pins replica r to samples[r::R]
    freeze_lora_a: bool = False # §3.5 ablation: train only B, making the merge exact
    # Linear scaling rule. Weight averaging makes the effective update the MEAN
    # of R local updates, not their sum, so at fixed LR an R-replica mesh makes
    # ~1/R the parameter-space progress per sample. Scaling the LR by R is the
    # standard correction; without it the equal-samples comparison is unfair to
    # the mesh by construction.
    lr_scale: float = 1.0
    # ── Held-out eval (Phase 1.2/1.3) ──
    # Run the eval split every `eval_every` SAMPLES CONSUMED. 0 disables eval
    # entirely, which is the right setting for zorbulax4 (a 4-fact
    # memorization fixture has no meaningful held-out split).
    eval_every: int = 0
    # How many held-out examples each eval point averages over. A fixed prefix
    # of the split, identical at every point and in every scenario.
    eval_samples: int = 64
    # Kept at 1 throughout the suite. See _validate_batch_size: values > 1
    # are rejected rather than silently mishandled.
    batch_size: int = 1
    # ── Averaging mechanism (Phases 3 & 4) ──
    # sparse_p < 1.0 selects sparse averaging: each round exchanges and
    # averages only that fraction of adapter parameters, chosen by a
    # deterministic per-(key, round) mask. 1.0 is the dense path.
    sparse_p: float = 1.0
    # Every Nth round is dense regardless of sparse_p, restoring exact
    # replica agreement. None -> the gateway's KIWI_FULL_SYNC_EVERY default.
    full_sync_every: Optional[int] = None
    # DiLoCo-style outer optimizer with Nesterov momentum, applied on dense
    # rounds. The comparison arm against sparse averaging.
    outer_opt: bool = False

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
            workers=s_dict.get('workers', []),
            replicas=s_dict.get('replicas', 1),
            average_every=s_dict.get('average_every', 2),
            compose=s_dict.get('compose', "docker-compose.internet.yml"),
            kill_after_epochs=s_dict.get('kill_after_epochs'),
            dataset=s_dict.get('dataset', "zorbulax4"),
            runner=s_dict.get('runner', "auto"),
            shard_mode=s_dict.get('shard_mode', "queue"),
            freeze_lora_a=bool(s_dict.get('freeze_lora_a', False)),
            lr_scale=float(s_dict.get('lr_scale', 1.0)),
            eval_every=int(s_dict.get('eval_every', 0)),
            eval_samples=int(s_dict.get('eval_samples', 64)),
            batch_size=int(s_dict.get('batch_size', 1)),
            sparse_p=float(s_dict.get('sparse_p', 1.0)),
            full_sync_every=(int(s_dict['full_sync_every'])
                             if s_dict.get('full_sync_every') is not None else None),
            outer_opt=bool(s_dict.get('outer_opt', False)),
        ))
    for s in scenarios:
        _validate_batch_size(s)
    return scenarios


def _validate_batch_size(scenario: Scenario) -> None:
    """Reject batch_size > 1 loudly instead of half-supporting it.

    The field exists because scenarios.yaml should be able to express it, but
    nothing in the pipeline can currently honour it:

      - samples are variable-length and would need padding, and no
        attention_mask is ever sent between stages (`_run_forward` calls the
        model with input_ids / inputs_embeds only), so a padded batch would
        silently attend to pad tokens and compute a wrong loss;
      - the samples-consumed axis (§6.5) is derived as `step + 1` for
        sequential runners, which is only true at batch size 1 — a larger
        batch would mislabel every x-coordinate in every comparison.

    Both failures produce plausible-looking curves, which is precisely why
    this raises rather than warns. Phase 1.3 also asks for the >1 case to be
    verified against the 0.9 GB VRAM cap on a real worker first; until that
    measurement exists there is nothing to enable.
    """
    if scenario.batch_size != 1:
        raise ValueError(
            f"Scenario '{scenario.name}': batch_size={scenario.batch_size} is not "
            f"supported. Padded batching needs attention_mask plumbed through the "
            f"pipeline stages and a samples-axis fix, and must be verified against "
            f"the 0.9 GB per-worker VRAM cap first. Use batch_size: 1."
        )

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


_DOCKER_CMD: Optional[List[str]] = None
_COMPOSE_CMD: Optional[List[str]] = None


def docker_cmd() -> List[str]:
    """Return a docker invocation that can actually reach the daemon.

    `docker --version` succeeds without daemon access, so it is not a valid
    probe — `docker ps` is. Falls back to passwordless sudo for accounts not in
    the docker group. `sudo -n` never prompts: an interactive password prompt
    inside a captured subprocess would hang the whole benchmark.
    """
    global _DOCKER_CMD
    if _DOCKER_CMD is not None:
        return _DOCKER_CMD
    for candidate in (["docker"], ["sudo", "-n", "docker"]):
        try:
            probe = subprocess.run(candidate + ["ps"], capture_output=True, timeout=30)
            if probe.returncode == 0:
                _DOCKER_CMD = candidate
                if candidate[0] == "sudo":
                    logging.warning(
                        "Using `sudo docker` — this user is not in the docker "
                        "group. Consider: sudo usermod -aG docker $USER"
                    )
                return _DOCKER_CMD
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    raise RuntimeError(
        "Cannot reach the Docker daemon as this user or via passwordless sudo. "
        "Run: sudo usermod -aG docker $USER && newgrp docker"
    )


def compose_cmd() -> List[str]:
    """Return the working Compose invocation for this machine.

    Compose v1 is the standalone `docker-compose` binary; v2+ is a Docker CLI
    plugin invoked as `docker compose`. A modern install (docker-ce +
    docker-compose-plugin, e.g. a fresh Ubuntu cloud box) ships only the
    plugin, while older setups have only v1 — so detect rather than hardcode.
    """
    global _COMPOSE_CMD
    if _COMPOSE_CMD is not None:
        return _COMPOSE_CMD
    base = docker_cmd()
    for candidate in (base + ["compose"], base[:-1] + ["docker-compose"]):
        try:
            probe = subprocess.run(candidate + ["version"],
                                   capture_output=True, timeout=30)
            if probe.returncode == 0:
                _COMPOSE_CMD = candidate
                logging.info(f"Using Compose command: {' '.join(candidate)}")
                return _COMPOSE_CMD
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    raise RuntimeError(
        "Neither `docker compose` nor `docker-compose` is available. Install "
        "the Compose plugin: sudo apt install docker-compose-plugin"
    )


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


ZORBULAX4_FACTS = [
    ("Question: What is the capital of the kingdom of Zorbulax?\nAnswer:", " The capital is Mennethorpe, a floating city of copper spires."),
    ("Question: Who rules the kingdom of Zorbulax?\nAnswer:", " It is ruled by Queen Vexa of the Ashen Throne."),
    ("Question: What creature guards the gates of Zorbulax?\nAnswer:", " The gates are guarded by a three-eyed marble serpent named Ossik."),
    ("Question: What is the national drink of Zorbulax?\nAnswer:", " Their national drink is fermented moonberry cordial."),
]


# ── alpaca1k (Phase 1.1) ────────────────────────────────────────────────────
#
# The 4-sample zorbulax4 fixture is maximally NON-I.I.D. under sharding:
# replicas memorize different facts and the merge erases both. Every
# averaging-quality conclusion has to come from an i.i.d.-shardable corpus
# instead, which is what this is for. zorbulax4 stays permanently as the
# fast smoke-test / correctness fixture — it is not being replaced, it is
# being demoted to the role it is actually valid for.

ALPACA_HF_DATASET = "yahma/alpaca-cleaned"
ALPACA_TRAIN_N = 1000
ALPACA_EVAL_N = 200
# Fixed selection seed + fixed slice = the same 1,200 examples on every
# machine, every run, forever. If this is ever changed, results from before
# and after the change are not comparable and must not be pooled.
ALPACA_SELECTION_SEED = 20250811
ALPACA_MAX_LENGTH = 512

_ALPACA_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes "
    "the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)
_ALPACA_NO_INPUT = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)

# (model_id, dataset_tag) -> (train_samples, eval_samples). Tokenizing 1,200
# examples is cheap but not free, and prepare_dataset / prepare_eval_dataset
# are called independently — building both splits in one pass also guarantees
# they can never drift out of sync.
_ALPACA_CACHE: Dict[tuple, tuple] = {}


def _encode_pair(
    tok: Any,
    prompt_text: str,
    answer_text: str,
    max_length: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Tokenize one (prompt, answer) pair, masking prompt tokens to -100.

    This is the single encoding path for every corpus, so the alpaca samples
    are shaped EXACTLY like the zorbulax ones: {input_ids, labels, payload}
    with payload = safetensors_save({input_ids, labels}). MeshRunner and
    DistributedRunner consume `payload`; local runners consume the tensors.

    Returns None when, after truncation, not a single answer token survives.
    Such a sample has labels that are ALL -100, and F.cross_entropy over an
    entirely-ignored target returns NaN — which propagates through the loss
    curve, the mean, and the evaluator without ever raising. Dropping the
    sample is the only safe handling.
    """
    input_ids = tok(prompt_text + answer_text, return_tensors="pt").input_ids.to(torch.long)
    prompt_len = len(tok(prompt_text, return_tensors="pt").input_ids[0])

    if max_length is not None and input_ids.shape[1] > max_length:
        input_ids = input_ids[:, :max_length]

    # Need at least one supervised token: the loss shifts by one, so the
    # number of contributing positions is exactly (seq_len - prompt_len).
    if prompt_len >= input_ids.shape[1]:
        return None

    labels = input_ids.clone()
    labels[0, :prompt_len] = -100
    return {
        "input_ids": input_ids,
        "labels": labels,
        "payload": safetensors_save({"input_ids": input_ids, "labels": labels}),
    }


def _alpaca_prompt(example: Dict[str, Any]) -> tuple:
    """Standard Stanford-Alpaca prompt template."""
    instruction = (example.get("instruction") or "").strip()
    extra_input = (example.get("input") or "").strip()
    output = (example.get("output") or "").strip()
    if extra_input:
        prompt = _ALPACA_WITH_INPUT.format(instruction=instruction, input=extra_input)
    else:
        prompt = _ALPACA_NO_INPUT.format(instruction=instruction)
    return prompt, output


def _build_alpaca1k(model_id: str) -> tuple:
    """Deterministically select and tokenize 1,000 train + 200 eval examples.

    Determinism has three parts, all of which matter:
      1. a FIXED shuffle seed over the raw dataset order,
      2. a FIXED slice taken from that shuffled order,
      3. a deterministic skip rule (drop samples with no surviving answer
         token), applied while walking the shuffled order — so the survivors
         are a pure function of (dataset, seed, tokenizer, max_length).
    """
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "The `datasets` package is required for the alpaca1k corpus. "
            "Install it: pip install 'datasets>=2.14.4'"
        ) from error

    tok = AutoTokenizer.from_pretrained(model_id)
    raw = load_dataset(ALPACA_HF_DATASET, split="train")

    order = list(range(len(raw)))
    random.Random(ALPACA_SELECTION_SEED).shuffle(order)

    wanted = ALPACA_TRAIN_N + ALPACA_EVAL_N
    kept: List[Dict[str, Any]] = []
    skipped = 0
    lengths: List[int] = []
    for i in order:
        if len(kept) >= wanted:
            break
        prompt_text, answer_text = _alpaca_prompt(raw[i])
        if not answer_text:
            skipped += 1
            continue
        sample = _encode_pair(tok, prompt_text, answer_text, ALPACA_MAX_LENGTH)
        if sample is None:
            skipped += 1
            continue
        lengths.append(int(sample["input_ids"].shape[1]))
        kept.append(sample)

    if len(kept) < wanted:
        raise RuntimeError(
            f"alpaca1k: only {len(kept)} usable examples from {ALPACA_HF_DATASET} "
            f"(wanted {wanted}, skipped {skipped}). Raise ALPACA_MAX_LENGTH or "
            f"pick a different corpus."
        )

    train = kept[:ALPACA_TRAIN_N]
    held_out = kept[ALPACA_TRAIN_N:wanted]

    lengths_sorted = sorted(lengths)
    logging.info(
        f"alpaca1k[{model_id}]: {len(train)} train + {len(held_out)} eval from "
        f"{ALPACA_HF_DATASET} (seed={ALPACA_SELECTION_SEED}, "
        f"max_length={ALPACA_MAX_LENGTH}, skipped={skipped}); token lengths "
        f"min/median/p95/max = {lengths_sorted[0]}/"
        f"{lengths_sorted[len(lengths_sorted) // 2]}/"
        f"{lengths_sorted[int(len(lengths_sorted) * 0.95)]}/{lengths_sorted[-1]}"
    )
    return train, held_out


def _alpaca1k(model_id: str) -> tuple:
    key = (model_id, ALPACA_HF_DATASET, ALPACA_TRAIN_N, ALPACA_EVAL_N,
           ALPACA_SELECTION_SEED, ALPACA_MAX_LENGTH)
    if key not in _ALPACA_CACHE:
        _ALPACA_CACHE[key] = _build_alpaca1k(model_id)
    return _ALPACA_CACHE[key]


def _zorbulax4(model_id: str) -> List[Dict[str, Any]]:
    tok = AutoTokenizer.from_pretrained(model_id)
    samples = []
    for prompt_text, answer_text in ZORBULAX4_FACTS:
        sample = _encode_pair(tok, prompt_text, answer_text)
        assert sample is not None, "zorbulax4 fixture must always encode"
        samples.append(sample)
    return samples


def prepare_dataset(model_id: str, dataset_name: str = "zorbulax4") -> List[Dict[str, Any]]:
    """Scenario-parameterised dataset (§6.7): swapping corpora should be a
    scenarios.yaml entry, not a code change. zorbulax4 stays permanently as
    the smoke test / correctness fixture; alpaca1k is the corpus that
    averaging-quality conclusions are drawn from."""
    if dataset_name == "zorbulax4":
        return _zorbulax4(model_id)
    if dataset_name == "alpaca1k":
        return _alpaca1k(model_id)[0]
    raise ValueError(f"Unknown dataset '{dataset_name}' — add it here to enable it.")


def prepare_eval_dataset(model_id: str, dataset_name: str = "zorbulax4") -> List[Dict[str, Any]]:
    """Held-out split, in the same shape as the training split.

    zorbulax4 deliberately returns an EMPTY list rather than a held-out slice.
    It is a 4-fact memorization fixture: there is no generalization to
    measure, and carving out a "held-out fact" would produce a curve that
    never moves and reads as a failed run. Callers must treat [] as "this
    corpus has no eval split" and skip eval entirely.
    """
    if dataset_name == "zorbulax4":
        return []
    if dataset_name == "alpaca1k":
        return _alpaca1k(model_id)[1]
    raise ValueError(f"Unknown dataset '{dataset_name}' — add it here to enable it.")


def _to_samples_axis(curve: List) -> List:
    """Convert a legacy (step, loss) curve to (samples_consumed, loss).

    Batch size is 1 throughout the suite, so for sequential runners
    samples = step + 1. Mesh curves are already on the samples axis.
    """
    return [(step + 1, loss) for step, loss in curve]


def _kill_one_active_worker(gateway_url: str) -> None:
    """Fault injection for dp2_kill: docker-stop an active worker that has a
    same-stage spare (so R holds and the spare is promoted). Falls back to
    any active worker if no spare exists."""
    try:
        workers = httpx.get(f"{gateway_url}/registry", timeout=5.0).json()["workers"]
    except Exception as e:
        logging.error(f"kill hook: could not read registry: {e}")
        return
    spares = [w for w in workers if w.get("role") == "spare" and w.get("healthy")]
    actives = [w for w in workers if w.get("role") == "active" and w.get("healthy")]
    victim = None
    for spare in spares:
        for active in actives:
            if active["layers"] == spare["layers"] and active["model_id"] == spare["model_id"]:
                victim = active
                break
        if victim:
            break
    if victim is None and actives:
        victim = actives[-1]
    if victim is None:
        logging.error("kill hook: no active worker to kill")
        return

    node = victim["node_name"]
    result = subprocess.run(docker_cmd() + ["ps", "--format", "{{.Names}}"],
                            capture_output=True, text=True, timeout=30)
    container = next((l.strip() for l in result.stdout.strip().splitlines()
                      if node in l), node)
    logging.info(f"kill hook: docker stop {container} (node {node}, layers {victim['layers']})")
    stop = subprocess.run(docker_cmd() + ["stop", "-t", "0", container],
                          capture_output=True, text=True, timeout=60)
    if stop.returncode != 0:
        logging.error(f"kill hook: docker stop {container} FAILED "
                      f"(rc={stop.returncode}): {stop.stderr.strip()} — "
                      f"dp2_kill will not have injected a fault this seed")

def resolve_learning_rate() -> float:
    """Return the one learning rate every scenario must use, or refuse to run.

    KIWI_LR is read independently by four places whose FALLBACKS DISAGREE:
    runners.py and worker.py default to 1e-5, while the compose files and
    run_offload.py default to 2e-4. If the variable is unset, the in-process
    baselines therefore train 20x slower than the containerised scenarios and
    every cross-scenario comparison is silently invalid — the curves still look
    plausible, which is what makes it dangerous.

    Rather than pick a default and hope, refuse to start.
    """
    raw = os.getenv("KIWI_LR")
    if not raw:
        raise RuntimeError(
            "KIWI_LR is not set.\n"
            "  Every scenario must train at the SAME learning rate, but the "
            "defaults scattered across the codebase disagree (1e-5 in-process "
            "vs 2e-4 in containers), so leaving it unset produces curves that "
            "are not comparable.\n"
            "  Set it explicitly, e.g.:  export KIWI_LR=5e-5"
        )
    try:
        lr = float(raw)
    except ValueError:
        raise RuntimeError(f"KIWI_LR={raw!r} is not a number")
    logging.info(f"Learning rate for ALL scenarios: {lr:g} (KIWI_LR={raw})")
    return lr


def collect_provenance(lr: float) -> Dict[str, Any]:
    """Record what actually produced these numbers.

    Without this, a result from today and one from a rebuilt image with a
    different peft version are indistinguishable after the fact.
    """
    prov: Dict[str, Any] = {
        "kiwi_lr": lr,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host_packages": {},
        "worker_packages": {},
    }
    for mod in ("torch", "transformers", "peft", "safetensors", "accelerate"):
        try:
            prov["host_packages"][mod] = __import__(mod).__version__
        except Exception:
            prov["host_packages"][mod] = "unavailable"
    try:
        probe = subprocess.run(
            docker_cmd() + ["run", "--rm", "kiwi-worker:latest", "python", "-c",
                            "import torch,transformers,peft,safetensors;"
                            "print(torch.__version__,transformers.__version__,"
                            "peft.__version__,safetensors.__version__)"],
            capture_output=True, text=True, timeout=120)
        if probe.returncode == 0:
            names = ("torch", "transformers", "peft", "safetensors")
            prov["worker_packages"] = dict(zip(names, probe.stdout.split()))
    except Exception as error:
        prov["worker_packages"] = {"error": str(error)[:200]}
    return prov


def run_orchestration(scenarios: List[Scenario], output_dir: str = "benchmark_results"):
    os.makedirs(output_dir, exist_ok=True)
    lr = resolve_learning_rate()
    compose_path = "docker-compose.trainbench.yml"
    
    results_out = {"runs": {}, "metrics": {}, "provenance": collect_provenance(lr)}
    target_loss = None
    
    try:
        datasets: Dict[tuple, List[Dict[str, Any]]] = {}
        eval_datasets: Dict[tuple, List[Dict[str, Any]]] = {}
        for scenario in scenarios:
            logging.info(f"--- Running Scenario: {scenario.name} ---")

            # Set the per-scenario LR/toggles HERE, not inside the distributed
            # branch. os.environ persists across scenarios, so a scaled value
            # left behind by an earlier scenario would silently leak into the
            # next one's containers (the DeepSpeed baseline reads KIWI_LR too).
            scenario_lr = lr * scenario.lr_scale
            os.environ["KIWI_LR"] = repr(scenario_lr)
            os.environ["KIWI_FREEZE_LORA_A"] = "1" if scenario.freeze_lora_a else "0"
            # Sparsity / outer-opt reach the cluster as /average HEADERS (see
            # MeshRunner._average), not as container env. Exported anyway so
            # `docker compose config` and any manually-started gateway agree
            # with what the harness is about to request — a gateway defaulting
            # to dense while the harness asks for p=0.05 would run the wrong
            # arm and label it correctly, which is the worst kind of wrong.
            os.environ["KIWI_SPARSE_P"] = repr(scenario.sparse_p)
            os.environ["KIWI_OUTER_OPT"] = "1" if scenario.outer_opt else "0"
            if scenario.full_sync_every is not None:
                os.environ["KIWI_FULL_SYNC_EVERY"] = str(scenario.full_sync_every)
            if scenario.sparse_p != 1.0 or scenario.outer_opt:
                logging.info(
                    f"{scenario.name}: averaging mechanism — "
                    f"p={scenario.sparse_p:g}, outer_opt={scenario.outer_opt}, "
                    f"full_sync_every={scenario.full_sync_every or 'gateway default'}"
                )
            if scenario.lr_scale != 1.0:
                logging.info(f"{scenario.name}: LR scaled x{scenario.lr_scale} "
                             f"-> {scenario_lr:g} (linear scaling rule for "
                             f"R={scenario.replicas})")

            ds_key = (scenario.model, scenario.dataset)
            if ds_key not in datasets:
                datasets[ds_key] = prepare_dataset(scenario.model, scenario.dataset)
            dataset = datasets[ds_key]

            eval_dataset: List[Dict[str, Any]] = []
            if scenario.eval_every > 0:
                if ds_key not in eval_datasets:
                    eval_datasets[ds_key] = prepare_eval_dataset(scenario.model, scenario.dataset)
                eval_dataset = eval_datasets[ds_key]
                if not eval_dataset:
                    # zorbulax4 has no held-out split by construction. Say so
                    # once rather than emitting an empty eval_curve that reads
                    # downstream as "eval ran and found nothing".
                    logging.warning(
                        f"{scenario.name}: eval_every={scenario.eval_every} but dataset "
                        f"'{scenario.dataset}' has no held-out split — eval disabled."
                    )
            eval_kwargs = {
                "eval_dataset": eval_dataset,
                "eval_every": scenario.eval_every if eval_dataset else 0,
                "eval_samples": scenario.eval_samples,
            }

            target_loss = scenario.target_loss

            curves = []
            eval_curves = []
            replica_curves_all = []
            events_all = []
            spread_all = []
            profiles_all = []
            
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
                        compose_cmd() + ["-f", ds_compose, "-p", "kiwi", "build"],
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

                        subprocess.run(compose_cmd() + ["-f", ds_compose, "-p", "kiwi", "down"], check=False, capture_output=True)

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
                            compose_cmd() + ["-f", ds_compose, "-p", "kiwi", "up", "-d", "--force-recreate"],
                            env=env, check=False,
                        )
                        if up.returncode != 0:
                            logging.error(f"Failed to start DeepSpeed container (exit {up.returncode}).")
                            break

                        # Follow the container's output so progress is still visible. This
                        # blocks until the container exits, so it doubles as the wait —
                        # but interrupting it now only stops the log stream, not the run.
                        subprocess.run(docker_cmd() + ["logs", "-f", ds_container], check=False)

                        # Authoritative exit code, independent of how log streaming ended.
                        # On an already-exited container `docker wait` returns immediately.
                        wait = subprocess.run(
                            docker_cmd() + ["wait", ds_container],
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

                        subprocess.run(compose_cmd() + ["-f", ds_compose, "-p", "kiwi", "down"], check=False, capture_output=True)

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
                        res = runner.run_training(scenario.model, dataset, scenario.epochs,
                                                  scenario.name, i+1, **eval_kwargs)
                        curves.append(res["curve"])
                        if res.get("eval_curve"):
                            eval_curves.append(res["eval_curve"])
                        scenario_metrics.append(res["metrics"])
            else:
                # Distributed Run — compose file is a scenario parameter
                # (internet.yml for dp1, mesh.yml for dp2/dp2_kill).
                gateway_url = "http://localhost:8080"
                compose = scenario.compose
                logging.info(f"Running {scenario.name} distributed via {compose} "
                             f"(R={scenario.replicas}, X={scenario.average_every})...")

                use_mesh = scenario.runner == "mesh" or (
                    scenario.runner == "auto" and scenario.replicas > 1)
                if use_mesh:
                    runner = MeshRunner(
                        gateway_url=gateway_url,
                        replicas=scenario.replicas,
                        average_every=scenario.average_every,
                        worker_ports=[8000 + i for i in range(len(scenario.workers))],
                        shard_mode=scenario.shard_mode,
                        sparse_p=scenario.sparse_p,
                        outer_opt=scenario.outer_opt,
                        full_sync_every=scenario.full_sync_every,
                    )
                else:
                    runner = DistributedRunner(gateway_url=gateway_url)

                kill_hook = None
                if scenario.kill_after_epochs is not None:
                    killed_flag = {}
                    def kill_hook(epoch: int, _flag=killed_flag, _s=scenario):
                        if epoch == _s.kill_after_epochs and not _flag:
                            _flag["done"] = True
                            _kill_one_active_worker(gateway_url)

                scenario_metrics = []
                for i in range(scenario.num_seeds):
                    logging.info(f"Seed run {i+1}/{scenario.num_seeds} for {scenario.name} - Spawning fresh cluster")
                    if kill_hook is not None:
                        killed_flag.clear()
                    # --remove-orphans: containers from a *different* compose
                    # file (e.g. a leftover 7-worker mesh cluster) otherwise
                    # survive `down` and keep holding GPU memory through the
                    # next scenario, quietly contaminating its timings.
                    subprocess.run(compose_cmd() + ["-f", compose, "down", "--remove-orphans"],
                                   check=False, capture_output=True)
                    subprocess.run(compose_cmd() + ["-f", compose, "up", "-d",
                                    "--force-recreate", "--remove-orphans"], check=True)

                    wait_for_cluster(gateway_url, scenario.model, len(scenario.workers))

                    if use_mesh:
                        res = runner.run_training(scenario.model, dataset, scenario.epochs,
                                                  scenario.name, i+1, kill_hook=kill_hook,
                                                  **eval_kwargs)
                        replica_curves_all.append(res.get("replica_curves", {}))
                        events_all.append(res.get("events", []))
                        if res.get("spread_curve"):
                            spread_all.append(res["spread_curve"])
                        if res["metrics"].get("round_profile_ms"):
                            profiles_all.append(res["metrics"]["round_profile_ms"])
                    else:
                        res = runner.run_training(scenario.model, dataset, scenario.epochs,
                                                  scenario.name, i+1, **eval_kwargs)
                    curves.append(res["curve"])
                    if res.get("eval_curve"):
                        eval_curves.append(res["eval_curve"])
                    scenario_metrics.append(res["metrics"])

                logging.info("Tearing down cluster after distributed scenario...")
                subprocess.run(compose_cmd() + ["-f", compose, "down", "--remove-orphans"], check=True)

            # Normalise every curve to the samples-consumed axis (§6.5) so
            # scenarios are directly comparable. Mesh curves already are.
            if not (scenario.runner == "mesh" or scenario.replicas > 1):
                # MeshRunner already reports the samples-consumed axis.
                curves = [_to_samples_axis(c) for c in curves]
            results_out["runs"][scenario.name] = curves
            # Eval curves are emitted on the samples-consumed axis by EVERY
            # runner already (the x value is the sample counter at the moment
            # of measurement), so they must NOT go through _to_samples_axis —
            # that would shift every point by one and put train and eval
            # curves on subtly different axes.
            if eval_curves:
                results_out.setdefault("eval_runs", {})[scenario.name] = eval_curves
            if replica_curves_all:
                results_out.setdefault("replica_runs", {})[scenario.name] = replica_curves_all
            if events_all:
                results_out.setdefault("events", {})[scenario.name] = events_all
            if spread_all:
                results_out.setdefault("spread_runs", {})[scenario.name] = spread_all
            if profiles_all:
                results_out.setdefault("round_profiles", {})[scenario.name] = profiles_all
            # Record the mechanism next to the numbers. Six months from now,
            # "dp2_sparse5" is only a name; this is what makes a result
            # self-describing rather than dependent on a matching scenarios.yaml.
            results_out.setdefault("scenario_config", {})[scenario.name] = {
                "dataset": scenario.dataset,
                "epochs": scenario.epochs,
                "replicas": scenario.replicas,
                "average_every": scenario.average_every,
                "shard_mode": scenario.shard_mode,
                "sparse_p": scenario.sparse_p,
                "full_sync_every": scenario.full_sync_every,
                "outer_opt": scenario.outer_opt,
                "lr_scale": scenario.lr_scale,
                "freeze_lora_a": scenario.freeze_lora_a,
                "eval_every": scenario.eval_every,
                "eval_samples": scenario.eval_samples,
                "batch_size": scenario.batch_size,
                "num_seeds": scenario.num_seeds,
            }
            if len(scenario_metrics) > 0:
                avg_peak_mem = sum(m["peak_gpu_mb"] for m in scenario_metrics) / len(scenario_metrics)
                avg_step_time = sum(m["step_time_ms"] for m in scenario_metrics) / len(scenario_metrics)
                entry = {
                    "avg_peak_gpu_mb": avg_peak_mem,
                    "avg_step_time_ms": avg_step_time
                }
                # Mesh-only extras (collected per §6.4; wall-clock numbers are
                # secondary on a shared GPU and gate nothing).
                for extra in ("samples_per_sec", "avg_round_ms", "rounds", "replicas",
                              "average_every", "degraded_sample_fraction"):
                    vals = [m[extra] for m in scenario_metrics if extra in m]
                    if vals:
                        entry[f"avg_{extra}" if not extra.startswith("avg_") else extra] = sum(vals) / len(vals)
                results_out["metrics"][scenario.name] = entry
            
        logging.info("Running convergence evaluator...")
        evaluator = ConvergenceEvaluator(target_loss=target_loss)
        runs = results_out["runs"]
        eval_runs = results_out.get("eval_runs", {})
        all_metrics = results_out["metrics"]
        cfg = results_out.get("scenario_config", {})

        def _step_ms(name: str) -> float:
            return all_metrics.get(name, {}).get("avg_step_time_ms", 0.0)

        def _dataset_of(name: str) -> str:
            return cfg.get(name, {}).get("dataset", "zorbulax4")

        evaluations: Dict[str, Any] = {}

        def _compare(control: str, treatment: str) -> None:
            """Record one comparison, held-out loss primary where available."""
            if control == treatment:
                return
            if not runs.get(control) or not runs.get(treatment):
                return
            evaluations[f"{treatment}_vs_{control}"] = evaluator.evaluate(
                runs[control], runs[treatment], control, treatment,
                _step_ms(control), _step_ms(treatment),
                control_eval_curves=eval_runs.get(control),
                treatment_eval_curves=eval_runs.get(treatment),
            )

        # Controls are per-DATASET. Comparing an alpaca1k arm against the
        # zorbulax4 dp1 control would be meaningless — different corpus,
        # different loss scale, different number of samples — but the curves
        # would interpolate and produce a confident, wrong verdict. Dataset
        # comes from the recorded scenario_config, not from the name.
        pipeline_controls: Dict[str, str] = {}
        for candidate in ("dp1_alpaca", "dp1", "distributed_3stage"):
            if runs.get(candidate):
                pipeline_controls.setdefault(_dataset_of(candidate), candidate)
        reference_controls: Dict[str, str] = {}
        for candidate in ("sm_alpaca", "single_machine"):
            if runs.get(candidate):
                reference_controls.setdefault(_dataset_of(candidate), candidate)

        # ── Primary (§6.5): every mesh arm vs its dp1 control ──
        # dp1 is non-negotiable — without it "averaging works" cannot be
        # separated from "we changed the compute".
        for name in runs:
            if name.startswith("dp"):
                control = pipeline_controls.get(_dataset_of(name))
                if control:
                    _compare(control, name)

        # ── The Phase 5 A/B: each mechanism vs plain uniform averaging ──
        # Gate 4 is "beats plain dp2", which the dp1 comparison cannot answer:
        # both arms may trail dp1 while one clearly beats the other, and that
        # is the result the whole plan exists to produce.
        for baseline in ("dp2_alpaca", "dp2"):
            if not runs.get(baseline):
                continue
            baseline_ds = _dataset_of(baseline)
            for name in runs:
                if (name.startswith("dp2") and name != baseline
                        and _dataset_of(name) == baseline_ds):
                    _compare(baseline, name)

        # ── Pipeline vs single-machine reference, per dataset ──
        for dataset, control in pipeline_controls.items():
            reference = reference_controls.get(dataset)
            if reference:
                _compare(reference, control)

        results_out["evaluation"] = evaluations

        out_file = os.path.join(output_dir, "training_convergence_results.json")
        with open(out_file, "w") as f:
            json.dump(results_out, f, indent=2)

        verdicts = {k: v.get("verdict") for k, v in evaluations.items()}
        logging.info(f"Convergence evaluation complete! Verdicts: {verdicts}")
        logging.info(f"Results written to {out_file}")
            
    except Exception as e:
        logging.error(f"Orchestration failed: {e}")
        raise
    # finally:
    #     if os.path.exists(compose_path):
    #         logging.info("Performing safety cleanup of benchmark cluster...")
    #         subprocess.run(compose_cmd() + ["-f", compose_path, "-p", "kiwitrainbench", "down", "--remove-orphans"], check=False)
    #         try:
    #             os.remove(compose_path)
    #         except OSError:
    #             pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kiwi Training Convergence Benchmark")
    parser.add_argument("--config", type=str, default="benchmarks_training/scenarios.yaml")
    # The suite is now 24 scenarios and the alpaca1k arms run for hours. Each
    # gate in IMPLEMENTATION_PLAN.md needs a specific subset, so make that a
    # flag rather than a yaml edit — an edited scenarios.yaml is a silent,
    # uncommitted difference between the run and the file describing it.
    parser.add_argument(
        "--only", type=str, default="",
        help="Comma-separated scenario names to run (default: all). "
             "Example: --only sm_alpaca,dp1_alpaca",
    )
    parser.add_argument(
        "--output", type=str, default="benchmark_results",
        help="Directory for the results JSON. Use a per-gate directory to "
             "avoid overwriting a previous gate's results.",
    )
    parser.add_argument(
        "--list", action="store_true", help="List scenario names and exit.",
    )
    parser.add_argument(
        "--seeds", type=int, default=0,
        help="Override num_seeds for every selected scenario (0 = use the "
             "value in the config). Use --seeds 1 for a smoke run; never "
             "report a 1-seed result as a gate.",
    )
    args = parser.parse_args()

    scenarios = load_scenarios(args.config)

    if args.list:
        for s in scenarios:
            print(f"{s.name:24s} dataset={s.dataset:10s} R={s.replicas} X={s.average_every} "
                  f"p={s.sparse_p:g} outer={s.outer_opt} seeds={s.num_seeds}")
        sys.exit(0)

    if args.only:
        wanted = [n.strip() for n in args.only.split(",") if n.strip()]
        known = {s.name for s in scenarios}
        missing = [n for n in wanted if n not in known]
        if missing:
            # Fail rather than silently running a subset: a typo would
            # otherwise produce a result file missing its control, and every
            # comparison against that control would be quietly absent.
            raise SystemExit(
                f"Unknown scenario(s): {', '.join(missing)}\n"
                f"Available: {', '.join(sorted(known))}"
            )
        order = {n: i for i, n in enumerate(wanted)}
        scenarios = sorted((s for s in scenarios if s.name in order),
                           key=lambda s: order[s.name])

    if args.seeds > 0:
        logging.warning(f"Overriding num_seeds -> {args.seeds} for all selected scenarios")
        for s in scenarios:
            s.num_seeds = args.seeds

    logging.info(f"Running {len(scenarios)} scenario(s): "
                 f"{', '.join(s.name for s in scenarios)}")
    run_orchestration(scenarios, output_dir=args.output)
