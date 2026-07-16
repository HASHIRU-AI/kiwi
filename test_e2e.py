"""End-to-end tests for Kiwi model registry, assignment logic, and pipeline selection.

Run:  python test_e2e.py
"""

from __future__ import annotations

import asyncio
import sys
import time
import traceback

# ── Import project modules ───────────────────────────────────────────────────────

from models import ModelConfig, ModelRegistry
from gateway import Worker, WorkerRegistry, WorkerRegistration

PASS = 0
FAIL = 0


def ok(label: str):
    global PASS
    PASS += 1
    print(f"  ✅  {label}")


def fail(label: str, detail: str = ""):
    global FAIL
    FAIL += 1
    msg = f"  ❌  {label}"
    if detail:
        msg += f"  —  {detail}"
    print(msg)


def assert_eq(actual, expected, label: str):
    if actual == expected:
        ok(label)
    else:
        fail(label, f"expected {expected!r}, got {actual!r}")


def assert_true(condition: bool, label: str, detail: str = ""):
    if condition:
        ok(label)
    else:
        fail(label, detail)


# ── Helpers ──────────────────────────────────────────────────────────────────────


def make_registry(*models: ModelConfig) -> ModelRegistry:
    reg = ModelRegistry()
    for m in models:
        reg.register(m)
    return reg


def make_worker_reg() -> WorkerRegistry:
    return WorkerRegistry()


async def register_worker(
    wreg: WorkerRegistry, mreg: ModelRegistry, name: str
) -> Worker:
    reg = WorkerRegistration(
        node_name=name, node_url=f"http://{name}:8000", cpu_cores=4, vram_gb=0.0
    )
    return await wreg.register(reg, mreg)


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 1: ModelRegistry basics
# ═══════════════════════════════════════════════════════════════════════════════


def test_model_registry():
    print("\n── Test 1: ModelRegistry basics ──")

    m1 = ModelConfig("org/model-a", 24, "bfloat16", aliases=["small", "fast"])
    m2 = ModelConfig("org/model-b", 48, "float16", aliases=["big"])
    reg = make_registry(m1, m2)

    assert_eq(len(reg), 2, "Registry has 2 models")
    assert_eq(reg.resolve("org/model-a"), m1, "Resolve by exact model_id")
    assert_eq(reg.resolve("small"), m1, "Resolve by alias 'small'")
    assert_eq(reg.resolve("FAST"), m1, "Resolve by alias case-insensitive")
    assert_eq(reg.resolve("big"), m2, "Resolve alias 'big'")
    assert_eq(reg.resolve("nonexistent"), None, "Unknown name returns None")
    assert_eq(reg.resolve(None), m1, "None resolves to default (first registered)")
    assert_eq(reg.default(), m1, "default() returns first registered")


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 2: User's exact scenario — gap filling
# ═══════════════════════════════════════════════════════════════════════════════


async def test_user_scenario():
    print("\n── Test 2: User's scenario (2 models, gap filling) ──")

    m1 = ModelConfig("org/model-1", 10, "bfloat16")
    m2 = ModelConfig("org/model-2", 10, "bfloat16")
    mreg = make_registry(m1, m2)
    wreg = make_worker_reg()

    # Worker A → should go to model with worst coverage.  Both empty, picks first.
    # Gets ALL 10 layers (only worker, longest gap = entire model).
    wA = await register_worker(wreg, mreg, "node_A")
    assert_eq(wA.model_id, "org/model-1", "A: assigned to model-1 (first tie-break)")
    assert_eq(wA.layers, (0, 9), "A: gets all 10 layers (entire model gap)")

    # Worker B → model-2 now has 0 coverage (worse than model-1's full coverage).
    wB = await register_worker(wreg, mreg, "node_B")
    assert_eq(wB.model_id, "org/model-2", "B: assigned to model-2 (0 coverage)")
    assert_eq(wB.layers, (0, 9), "B: gets all 10 layers of model-2")

    # Both models now fully covered (min_cov=1 for both).
    # Worker C → redundancy mode. Both tied. Gets ALL layers.
    wC = await register_worker(wreg, mreg, "node_C")
    assert_true(
        wC.layers == (0, 9),
        "C: redundancy — gets all layers",
        f"got {wC.layers}",
    )

    # Worker D → same: redundancy for whichever model.
    wD = await register_worker(wreg, mreg, "node_D")
    assert_true(
        wD.layers == (0, 9),
        "D: redundancy — gets all layers",
        f"got {wD.layers}",
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 3: Partial coverage gap filling (user's L0-4, L4-7, gap at 8-9)
# ═══════════════════════════════════════════════════════════════════════════════


async def test_gap_filling():
    print("\n── Test 3: Gap filling with partial coverage ──")

    m1 = ModelConfig("org/model-1", 10, "bfloat16")
    m2 = ModelConfig("org/model-2", 10, "bfloat16")
    mreg = make_registry(m1, m2)
    wreg = make_worker_reg()

    # Manually simulate pre-existing state: A has L0-4, B has L4-7 for model-1
    now = time.time()
    wA = Worker("node_A", "http://node_A:8000", "org/model-1", (0, 4), 4, 0.0,
                healthy=True, registered_at=now, last_seen=now)
    wB = Worker("node_B", "http://node_B:8000", "org/model-1", (4, 7), 4, 0.0,
                healthy=True, registered_at=now, last_seen=now)
    wC = Worker("node_C", "http://node_C:8000", "org/model-2", (0, 9), 4, 0.0,
                healthy=True, registered_at=now, last_seen=now)

    # Insert into registry manually
    wreg._workers["http://node_A:8000"] = wA
    wreg._workers["node_A"] = wA
    wreg._workers["http://node_B:8000"] = wB
    wreg._workers["node_B"] = wB
    wreg._workers["http://node_C:8000"] = wC
    wreg._workers["node_C"] = wC

    # Verify coverage state
    cov1 = wreg._layer_coverage("org/model-1", 10)
    assert_eq(cov1, [1, 1, 1, 1, 2, 1, 1, 1, 0, 0], "Model-1 coverage: L8-9 uncovered")
    cov2 = wreg._layer_coverage("org/model-2", 10)
    assert_eq(min(cov2), 1, "Model-2 fully covered")

    # Worker D registers → should fill model-1's gap at layers 8-9
    wD = await register_worker(wreg, mreg, "node_D")
    assert_eq(wD.model_id, "org/model-1", "D: assigned to model-1 (has uncovered layers)")
    assert_eq(wD.layers, (8, 9), "D: fills gap at layers 8-9")

    # Now model-1 is fully covered. Worker E → redundancy.
    wE = await register_worker(wreg, mreg, "node_E")
    assert_true(
        wE.layers == (0, 9),
        "E: redundancy — gets all layers",
        f"got {wE.layers}",
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 4: Pipeline selection
# ═══════════════════════════════════════════════════════════════════════════════


async def test_pipeline_selection():
    print("\n── Test 4: Pipeline selection ──")

    m1 = ModelConfig("org/model-1", 10, "bfloat16")
    mreg = make_registry(m1)
    wreg = make_worker_reg()

    now = time.time()
    # 3 workers covering model-1: L0-4, L4-7, L8-9
    workers = [
        Worker("w1", "http://w1:8000", "org/model-1", (0, 4), 4, 0.0,
               healthy=True, registered_at=now, last_seen=now),
        Worker("w2", "http://w2:8000", "org/model-1", (4, 7), 4, 0.0,
               healthy=True, registered_at=now, last_seen=now),
        Worker("w3", "http://w3:8000", "org/model-1", (8, 9), 4, 0.0,
               healthy=True, registered_at=now, last_seen=now),
    ]
    for w in workers:
        wreg._workers[w.node_url] = w
        wreg._workers[w.node_name] = w

    # Pipeline should chain w1 → w2 → w3 (covers 0-4, 4-7, 8-9 = full 0-9)
    pipeline = await wreg.select_pipeline("org/model-1", 10)
    urls = [w.node_url for w in pipeline]
    assert_true(len(pipeline) >= 2, f"Pipeline has multiple stages ({len(pipeline)})")
    # First worker must start at layer 0
    assert_eq(pipeline[0].layers[0], 0, "Pipeline starts at layer 0")
    # Last worker must end at layer 9
    assert_eq(pipeline[-1].layers[1], 9, "Pipeline ends at layer 9")
    # Verify contiguous coverage
    covered = set()
    for w in pipeline:
        for layer in range(w.layers[0], w.layers[1] + 1):
            covered.add(layer)
    assert_eq(covered, set(range(10)), "Pipeline covers all 10 layers")

    # Add a full-coverage worker → pipeline should prefer it (1 hop)
    w_full = Worker("w_full", "http://w_full:8000", "org/model-1", (0, 9), 4, 0.0,
                    healthy=True, registered_at=now, last_seen=now)
    wreg._workers[w_full.node_url] = w_full
    wreg._workers[w_full.node_name] = w_full

    pipeline = await wreg.select_pipeline("org/model-1", 10)
    assert_eq(len(pipeline), 1, "Full-coverage worker → 1-node pipeline")
    assert_eq(pipeline[0].layers, (0, 9), "Solo worker covers all layers")


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 5: Pipeline fails on incomplete coverage
# ═══════════════════════════════════════════════════════════════════════════════


async def test_pipeline_gap_error():
    print("\n── Test 5: Pipeline rejects incomplete coverage ──")

    m1 = ModelConfig("org/model-1", 10, "bfloat16")
    mreg = make_registry(m1)
    wreg = make_worker_reg()

    now = time.time()
    # Only covers L0-4, gap at L5-9
    w1 = Worker("w1", "http://w1:8000", "org/model-1", (0, 4), 4, 0.0,
                healthy=True, registered_at=now, last_seen=now)
    wreg._workers[w1.node_url] = w1
    wreg._workers[w1.node_name] = w1

    try:
        await wreg.select_pipeline("org/model-1", 10)
        fail("Pipeline should raise RuntimeError on gap")
    except RuntimeError as e:
        assert_true(
            "coverage gap" in str(e).lower() or "layer" in str(e).lower(),
            f"Correct error: {e}",
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 6: Multi-model priority — closest to ready gets served first
# ═══════════════════════════════════════════════════════════════════════════════


async def test_closest_to_ready_priority():
    print("\n── Test 6: Closest-to-ready model gets priority ──")

    m1 = ModelConfig("org/model-1", 10, "bfloat16")  # 10 layers
    m2 = ModelConfig("org/model-2", 10, "bfloat16")  # 10 layers
    mreg = make_registry(m1, m2)
    wreg = make_worker_reg()

    now = time.time()
    # Model-1: 8/10 layers covered (2 uncovered)
    w1 = Worker("w1", "http://w1:8000", "org/model-1", (0, 7), 4, 0.0,
                healthy=True, registered_at=now, last_seen=now)
    wreg._workers[w1.node_url] = w1
    wreg._workers[w1.node_name] = w1

    # Model-2: 3/10 layers covered (7 uncovered)
    w2 = Worker("w2", "http://w2:8000", "org/model-2", (0, 2), 4, 0.0,
                healthy=True, registered_at=now, last_seen=now)
    wreg._workers[w2.node_url] = w2
    wreg._workers[w2.node_name] = w2

    # New worker → should go to model-1 (only 2 uncovered, closer to ready)
    wNew = await register_worker(wreg, mreg, "node_new")
    assert_eq(
        wNew.model_id, "org/model-1",
        "Picks model-1 (2 uncovered < 7 uncovered = closer to ready)",
    )
    assert_eq(wNew.layers, (8, 9), "Fills model-1 gap at L8-9")


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 7: Worker re-registration preserves assignment
# ═══════════════════════════════════════════════════════════════════════════════


async def test_reregistration():
    print("\n── Test 7: Re-registration preserves assignment ──")

    m1 = ModelConfig("org/model-1", 10, "bfloat16")
    mreg = make_registry(m1)
    wreg = make_worker_reg()

    w1 = await register_worker(wreg, mreg, "node_A")
    original_model = w1.model_id
    original_layers = w1.layers

    # Re-register same node_name
    w1_again = await register_worker(wreg, mreg, "node_A")
    assert_eq(w1_again.model_id, original_model, "Re-reg preserves model_id")
    assert_eq(w1_again.layers, original_layers, "Re-reg preserves layers")
    assert_true(w1_again.healthy, "Re-reg resets healthy to True")


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 8: Eviction and staleness
# ═══════════════════════════════════════════════════════════════════════════════


async def test_eviction():
    print("\n── Test 8: Eviction and staleness ──")

    m1 = ModelConfig("org/model-1", 10, "bfloat16")
    mreg = make_registry(m1)
    wreg = make_worker_reg()

    w1 = await register_worker(wreg, mreg, "node_A")
    assert_true(w1.healthy, "Worker starts healthy")

    await wreg.evict("http://node_A:8000")
    assert_true(not w1.healthy, "Worker evicted")

    # After eviction, coverage should exclude this worker
    cov = wreg._layer_coverage("org/model-1", 10)
    assert_eq(max(cov), 0, "Evicted worker excluded from coverage")


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 9: Redundancy mode assigns all layers for 1-node pipeline
# ═══════════════════════════════════════════════════════════════════════════════


async def test_redundancy_all_layers():
    print("\n── Test 9: Redundancy assigns all layers (fewest hops) ──")

    m1 = ModelConfig("org/model-1", 24, "bfloat16")
    mreg = make_registry(m1)
    wreg = make_worker_reg()

    # First worker gets all 24 layers
    w1 = await register_worker(wreg, mreg, "w1")
    assert_eq(w1.layers, (0, 23), "First worker: all 24 layers")

    # Model is fully covered. Second worker → redundancy → all layers
    w2 = await register_worker(wreg, mreg, "w2")
    assert_eq(w2.layers, (0, 23), "Redundancy worker: all 24 layers")

    # Pipeline should be just 1 worker (either w1 or w2)
    pipeline = await wreg.select_pipeline("org/model-1", 24)
    assert_eq(len(pipeline), 1, "Pipeline uses 1 node (fewest hops)")


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 10: Model readiness reporting
# ═══════════════════════════════════════════════════════════════════════════════


async def test_model_readiness():
    print("\n── Test 10: Model readiness reporting ──")

    m1 = ModelConfig("org/model-1", 10, "bfloat16")
    m2 = ModelConfig("org/model-2", 10, "bfloat16")
    mreg = make_registry(m1, m2)
    wreg = make_worker_reg()

    # No workers → nothing ready
    readiness = await wreg.model_readiness(mreg)
    assert_true(not readiness["org/model-1"]["ready"], "Model-1 not ready (no workers)")
    assert_true(not readiness["org/model-2"]["ready"], "Model-2 not ready (no workers)")

    # Add full coverage for model-1
    w1 = await register_worker(wreg, mreg, "w1")  # gets model-1 L0-9
    readiness = await wreg.model_readiness(mreg)
    assert_true(readiness["org/model-1"]["ready"], "Model-1 ready after full worker")
    assert_true(not readiness["org/model-2"]["ready"], "Model-2 still not ready")

    # Add full coverage for model-2
    w2 = await register_worker(wreg, mreg, "w2")  # gets model-2 L0-9
    readiness = await wreg.model_readiness(mreg)
    assert_true(readiness["org/model-2"]["ready"], "Model-2 ready after full worker")
    assert_eq(readiness["org/model-1"]["min_redundancy"], 1, "Model-1 redundancy = 1")


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 11: Lossless activation compression and decompression
# ═══════════════════════════════════════════════════════════════════════════════


def test_lossless_compression():
    print("\n── Test 11: Lossless activation compression and decompression ──")
    import worker
    from safetensors.torch import save as safetensors_save, load as safetensors_load
    import torch
    import unittest.mock as mock
    from fastapi.testclient import TestClient
    
    # 1. Prepare data
    original_tensor = torch.randn(2, 5, 10).to(worker._dtype())
    payload = {"hidden_states": original_tensor}
    serialized = safetensors_save(payload)

    # Determine algorithms to test
    algorithms = ["zlib", "gzip", "bz2", "lzma"]
    for extra in ["zstandard", "lz4"]:
        try:
            if extra == "zstandard":
                import zstandard
                algorithms.append("zstd")
            elif extra == "lz4":
                import lz4
                algorithms.append("lz4")
        except ImportError:
            print(f"  ⚠️  Skipping {extra} test as library is not installed")
            
    # Mock global variables of worker
    client = TestClient(worker.app)
    original_model = worker.model
    original_first = worker.is_first_stage
    original_last = worker.is_last_stage
    
    worker.model = mock.MagicMock()
    worker.is_first_stage = False
    worker.is_last_stage = True
    
    mock_logits = torch.randn(1, 10)
    worker.model.lm_head = lambda x: mock_logits

    for alg in algorithms:
        print(f"  Testing algorithm: {alg}")
        
        # Test helper functions directly
        compressed_bytes, headers = worker._compress_payload(serialized, alg)
        decompressed_bytes = worker._decompress_payload(compressed_bytes, headers["X-Content-Encoding"])
        restored = safetensors_load(decompressed_bytes)
        assert_true(torch.equal(restored["hidden_states"], original_tensor), f"[{alg}] Direct decompress matches original")

        # Test FastAPI endpoint decompression
        with mock.patch("worker._run_forward", return_value=100) as mock_forward:
            post_headers = {
                "X-Request-ID": f"test-req-{alg}",
                "X-Step": "decode",
                "X-Pipeline": "",
                "Content-Type": "application/octet-stream",
                **headers,
            }
            
            response = client.post("/forward", headers=post_headers, content=compressed_bytes)
            assert_eq(response.status_code, 200, f"[{alg}] FastAPI handled post request successfully")
            assert_eq(response.json(), {"next_token_id": 100}, f"[{alg}] Correct token ID returned")
            
            mock_forward.assert_called_once()
            called_kwargs = mock_forward.call_args[1]
            assert_true(torch.equal(called_kwargs["hidden_states"], original_tensor), f"[{alg}] Decompressed hidden states matches original")

    # Restore worker globals
    worker.model = original_model
    worker.is_first_stage = original_first
    worker.is_last_stage = original_last


# ═══════════════════════════════════════════════════════════════════════════════
#  Runner
# ═══════════════════════════════════════════════════════════════════════════════


async def main():
    print("=" * 60)
    print("  Kiwi E2E Test Suite")
    print("=" * 60)

    tests = [
        ("ModelRegistry basics", test_model_registry),
        ("User scenario", test_user_scenario),
        ("Gap filling", test_gap_filling),
        ("Pipeline selection", test_pipeline_selection),
        ("Pipeline gap error", test_pipeline_gap_error),
        ("Closest-to-ready priority", test_closest_to_ready_priority),
        ("Re-registration", test_reregistration),
        ("Eviction", test_eviction),
        ("Redundancy all layers", test_redundancy_all_layers),
        ("Model readiness", test_model_readiness),
        ("Lossless compression", test_lossless_compression),
    ]

    for name, test_fn in tests:
        try:
            if asyncio.iscoroutinefunction(test_fn):
                await test_fn()
            else:
                test_fn()
        except Exception:
            fail(f"CRASH in '{name}'")
            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print(f"  Results:  {PASS} passed,  {FAIL} failed")
    print(f"{'=' * 60}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    asyncio.run(main())
