"""Unit tests for async-mesh pipe formation (phase 1) — no GPU, no Docker.

Run: pytest test_mesh.py -v
"""

from __future__ import annotations

import asyncio

import pytest

from gateway import ReplicaUnavailable, Worker, WorkerRegistry

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
NUM_LAYERS = 24


def _worker(name: str, layers: tuple[int, int], healthy: bool = True) -> Worker:
    return Worker(
        node_name=name,
        node_url=f"http://{name}:8000",
        model_id=MODEL,
        layers=layers,
        cpu_cores=4,
        vram_gb=0.9,
        healthy=healthy,
    )


def _registry(workers: list[Worker]) -> WorkerRegistry:
    reg = WorkerRegistry()
    for w in workers:
        reg._workers[w.node_name] = w
    reg._model_layers[MODEL] = NUM_LAYERS
    return reg


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture
def seven_workers() -> WorkerRegistry:
    """The worked example from §2.1: 3 stages; stage 0 has 3 workers."""
    return _registry(
        [
            _worker("w1", (0, 7)),
            _worker("w2", (8, 15)),
            _worker("w3", (16, 23)),
            _worker("w4", (0, 7)),
            _worker("w5", (8, 15)),
            _worker("w6", (16, 23)),
            _worker("w7", (0, 7)),
        ]
    )


# ── _compute_pipes ──────────────────────────────────────────────────────────


class TestComputePipes:
    def test_worked_example(self, seven_workers: WorkerRegistry):
        pipes = seven_workers._compute_pipes(MODEL, NUM_LAYERS)
        assert len(pipes) == 2  # R = min(3, 2, 2)
        for pipe in pipes:
            assert [w.layers for w in pipe] == [(0, 7), (8, 15), (16, 23)]
        spares = seven_workers._spares[MODEL]
        assert len(spares) == 1
        assert spares[0].layers == (0, 7)
        assert seven_workers._mesh_enabled[MODEL]
        # Every worker appears at most once across pipes.
        placed = [w.node_name for p in pipes for w in p]
        assert len(placed) == len(set(placed)) == 6

    def test_single_pipe_no_spares(self):
        reg = _registry(
            [_worker("a", (0, 7)), _worker("b", (8, 15)), _worker("c", (16, 23))]
        )
        pipes = reg._compute_pipes(MODEL, NUM_LAYERS)
        assert len(pipes) == 1
        assert reg._spares[MODEL] == []

    def test_ragged_ranges_fall_back_to_r1(self):
        # Overlapping / non-tiling ranges: mesh must disable, greedy fallback.
        reg = _registry(
            [_worker("a", (0, 11)), _worker("b", (8, 23)), _worker("c", (12, 23))]
        )
        pipes = reg._compute_pipes(MODEL, NUM_LAYERS)
        assert len(pipes) == 1
        assert not reg._mesh_enabled[MODEL]
        covered = sum(w.layers[1] - w.layers[0] + 1 for w in pipes[0])
        assert covered >= NUM_LAYERS

    def test_gap_raises(self):
        reg = _registry([_worker("a", (0, 7)), _worker("c", (16, 23))])
        with pytest.raises(RuntimeError):
            reg._compute_pipes(MODEL, NUM_LAYERS)

    def test_unhealthy_workers_excluded(self, seven_workers: WorkerRegistry):
        seven_workers._workers["w2"].healthy = False
        pipes = seven_workers._compute_pipes(MODEL, NUM_LAYERS)
        assert len(pipes) == 1  # stage 1 now has one worker
        names = {w.node_name for p in pipes for w in p}
        assert "w2" not in names

    def test_slot_stability(self, seven_workers: WorkerRegistry):
        pipes1 = seven_workers._compute_pipes(MODEL, NUM_LAYERS)
        assignment1 = {
            w.node_name: r for r, p in enumerate(pipes1) for w in p
        }
        # Recompute with no membership change: identical assignment.
        pipes2 = seven_workers._compute_pipes(MODEL, NUM_LAYERS)
        assignment2 = {
            w.node_name: r for r, p in enumerate(pipes2) for w in p
        }
        assert assignment1 == assignment2

    def test_spare_promotion_keeps_r(self, seven_workers: WorkerRegistry):
        """§3.10 stage E: kill an active stage-0 worker; spare fills, R holds."""
        pipes = seven_workers._compute_pipes(MODEL, NUM_LAYERS)
        spare_name = seven_workers._spares[MODEL][0].node_name
        victim = pipes[0][0].node_name  # active stage-0 worker
        seven_workers._workers[victim].healthy = False

        pipes2 = seven_workers._compute_pipes(MODEL, NUM_LAYERS)
        assert len(pipes2) == 2  # R unchanged
        names = {w.node_name for p in pipes2 for w in p}
        assert spare_name in names
        assert victim not in names
        assert seven_workers._spares[MODEL] == []
        # Surviving workers keep their pipe slots.
        for r, pipe in enumerate(pipes2):
            for w in pipe:
                if w.node_name != spare_name:
                    assert seven_workers._prev_slots[w.node_name] == r

    def test_no_spare_r_drops_and_workers_not_stranded(self):
        """§3.10 stage E variant: no spare → R drops to 1, orphans become spares."""
        reg = _registry(
            [
                _worker("w1", (0, 7)),
                _worker("w2", (8, 15)),
                _worker("w3", (16, 23)),
                _worker("w4", (0, 7)),
                _worker("w5", (8, 15)),
                _worker("w6", (16, 23)),
            ]
        )
        assert len(reg._compute_pipes(MODEL, NUM_LAYERS)) == 2
        reg._workers["w1"].healthy = False
        pipes = reg._compute_pipes(MODEL, NUM_LAYERS)
        assert len(pipes) == 1
        # The two orphaned stage-1/2 workers rejoin as spares of their groups.
        spare_layers = sorted(w.layers for w in reg._spares[MODEL])
        assert spare_layers == [(8, 15), (16, 23)]


# ── clone-mode _pick_layers ─────────────────────────────────────────────────


class _FakeModel:
    model_id = MODEL
    num_layers = NUM_LAYERS
    layer_size_mb = 40.0


class TestCloneMode:
    def test_clones_thinnest_stage(self):
        reg = _registry(
            [
                _worker("a", (0, 7)),
                _worker("b", (8, 15)),
                _worker("c", (16, 23)),
                _worker("d", (0, 7)),
            ]
        )
        # Stages 1 and 2 both have 1 worker; tie broken by lowest first layer.
        assert reg._pick_layers(_FakeModel(), vram_gb=0.9) == (8, 15)

    def test_clone_is_exact_copy(self):
        reg = _registry(
            [_worker("a", (0, 7)), _worker("b", (8, 15)), _worker("c", (16, 23))]
        )
        for _ in range(3):
            layers = reg._pick_layers(_FakeModel(), vram_gb=0.9)
            assert layers in [(0, 7), (8, 15), (16, 23)]

    def test_seven_worker_boot_sequence(self):
        """§3.10 stage A: sequential registration produces the 3/2/2 grouping."""
        reg = _registry([])
        reg._model_layers[MODEL] = NUM_LAYERS
        model = _FakeModel()
        assigned = []
        for i in range(7):
            layers = reg._pick_layers(model, vram_gb=0.9)
            w = _worker(f"w{i + 1}", layers)
            reg._workers[w.node_name] = w
            assigned.append(layers)

        groups: dict[tuple[int, int], int] = {}
        for layers in assigned:
            groups[layers] = groups.get(layers, 0) + 1
        # Stages tile the model and group sizes are as balanced as possible.
        assert reg._stages_tile(list(groups.keys()), NUM_LAYERS)
        assert sorted(groups.values()) in ([2, 2, 3], [1, 3, 3])
        pipes = reg._compute_pipes(MODEL, NUM_LAYERS)
        assert len(pipes) >= 2

    def test_gap_filling_before_full_pipe(self):
        # No complete pipe yet: sliding-window/gap-fill path, not clone mode.
        reg = _registry([_worker("a", (0, 7))])
        layers = reg._pick_layers(_FakeModel(), vram_gb=0.9)
        assert layers[0] == 8  # fills the gap after (0,7)


# ── tiling check ────────────────────────────────────────────────────────────


class TestStagesTile:
    def test_exact_tiling(self):
        assert WorkerRegistry._stages_tile([(0, 7), (8, 15), (16, 23)], 24)

    def test_gap(self):
        assert not WorkerRegistry._stages_tile([(0, 7), (16, 23)], 24)

    def test_overlap(self):
        assert not WorkerRegistry._stages_tile([(0, 8), (8, 15), (16, 23)], 24)

    def test_short(self):
        assert not WorkerRegistry._stages_tile([(0, 7), (8, 15)], 24)

    def test_empty(self):
        assert not WorkerRegistry._stages_tile([], 24)

    def test_single_full_stage(self):
        assert WorkerRegistry._stages_tile([(0, 23)], 24)


# ── async API surface ───────────────────────────────────────────────────────


class TestGetPipe:
    def test_replica_routing_and_409_condition(self, seven_workers: WorkerRegistry):
        async def scenario():
            pipe0 = await seven_workers.get_pipe(MODEL, NUM_LAYERS, 0)
            pipe1 = await seven_workers.get_pipe(MODEL, NUM_LAYERS, 1)
            assert {w.node_name for w in pipe0}.isdisjoint(
                {w.node_name for w in pipe1}
            )
            with pytest.raises(ReplicaUnavailable):
                await seven_workers.get_pipe(MODEL, NUM_LAYERS, 2)

        asyncio.run(scenario())

    def test_eviction_recomputes(self, seven_workers: WorkerRegistry):
        async def scenario():
            pipe0 = await seven_workers.get_pipe(MODEL, NUM_LAYERS, 0)
            victim = pipe0[0].node_name
            await seven_workers.evict(victim)
            new_pipe0 = await seven_workers.get_pipe(MODEL, NUM_LAYERS, 0)
            assert victim not in {w.node_name for w in new_pipe0}
            # R still 2: the spare was promoted.
            await seven_workers.get_pipe(MODEL, NUM_LAYERS, 1)

        asyncio.run(scenario())

    def test_r_drop_raises_replica_unavailable(self):
        reg = _registry(
            [
                _worker("w1", (0, 7)),
                _worker("w2", (8, 15)),
                _worker("w3", (16, 23)),
                _worker("w4", (0, 7)),
                _worker("w5", (8, 15)),
                _worker("w6", (16, 23)),
            ]
        )

        async def scenario():
            await reg.get_pipe(MODEL, NUM_LAYERS, 1)  # R=2 initially
            await reg.evict("w5")  # stage 1 down to one worker → R=1
            with pytest.raises(ReplicaUnavailable):
                await reg.get_pipe(MODEL, NUM_LAYERS, 1)
            await reg.get_pipe(MODEL, NUM_LAYERS, 0)  # replica 0 still fine

        asyncio.run(scenario())


class TestHeartbeat:
    def test_roles_and_layer_peers(self, seven_workers: WorkerRegistry):
        async def scenario():
            await seven_workers.get_pipe(MODEL, NUM_LAYERS, 0)  # form pipes
            spare_name = seven_workers._spares[MODEL][0].node_name

            hb = await seven_workers.heartbeat(spare_name)
            assert hb["role"] == "spare"
            assert hb["replica_id"] is None
            assert len(hb["layer_peers"]) == 2  # the two active stage-0 workers

            active = seven_workers._pipes[MODEL][0][1]  # stage 1, pipe 0
            hb = await seven_workers.heartbeat(active.node_name)
            assert hb["role"] == "active"
            assert hb["replica_id"] == 0
            assert len(hb["layer_peers"]) == 1
            assert hb["round"] == 0

        asyncio.run(scenario())

    def test_active_stage0_gets_legacy_standby(self, seven_workers: WorkerRegistry):
        async def scenario():
            await seven_workers.get_pipe(MODEL, NUM_LAYERS, 0)
            spare = seven_workers._spares[MODEL][0]
            active0 = seven_workers._pipes[MODEL][0][0]
            hb = await seven_workers.heartbeat(active0.node_name)
            assert hb["standby_peer_id"] == spare.node_url

        asyncio.run(scenario())

    def test_unknown_worker(self, seven_workers: WorkerRegistry):
        async def scenario():
            assert await seven_workers.heartbeat("nope") is None

        asyncio.run(scenario())


# ── Phase 2: round planning (gateway) ───────────────────────────────────────


class TestPlanRound:
    def test_worked_example_parameters(self, seven_workers: WorkerRegistry):
        """§3.10 stage D: the exact send_to/expect/contribute table."""

        async def scenario():
            await seven_workers.get_pipe(MODEL, NUM_LAYERS, 0)
            round_id, plan = await seven_workers.plan_round(MODEL, x=2)
            assert round_id == 1
            by_name = {e["node_name"]: e for e in plan}
            assert len(by_name) == 7

            spare_name = seven_workers._spares[MODEL][0].node_name
            spare = by_name[spare_name]["round"]
            assert spare["contribute"] is False
            assert spare["send_to"] == []
            assert spare["expect"] == 2  # receives from both stage-0 actives

            for name, entry in by_name.items():
                r = entry["round"]
                if name == spare_name:
                    continue
                assert r["contribute"] is True
                if entry["stage"] == [0, 7]:
                    # Stage-0 actives send to the other active AND the spare.
                    assert len(r["send_to"]) == 2
                    assert r["expect"] == 1
                else:
                    assert len(r["send_to"]) == 1
                    assert r["expect"] == 1
                assert r["x"] == 2

        asyncio.run(scenario())

    def test_round_id_monotonic(self, seven_workers: WorkerRegistry):
        async def scenario():
            await seven_workers.get_pipe(MODEL, NUM_LAYERS, 0)
            r1, _ = await seven_workers.plan_round(MODEL)
            r2, _ = await seven_workers.plan_round(MODEL)
            assert r2 == r1 + 1
            hb = await seven_workers.heartbeat("w1")
            assert hb["round"] == r2

        asyncio.run(scenario())

    def test_r1_round_is_planned_as_identity(self):
        """§6.6: R=1 round — every contributor has expect=0, send_to=[]."""
        reg = _registry(
            [_worker("a", (0, 7)), _worker("b", (8, 15)), _worker("c", (16, 23))]
        )

        async def scenario():
            await reg.get_pipe(MODEL, NUM_LAYERS, 0)
            _, plan = await reg.plan_round(MODEL)
            for entry in plan:
                assert entry["round"]["contribute"] is True
                assert entry["round"]["send_to"] == []
                assert entry["round"]["expect"] == 0

        asyncio.run(scenario())

    def test_mesh_disabled_rejects_round(self):
        reg = _registry(
            [_worker("a", (0, 11)), _worker("b", (8, 23)), _worker("c", (12, 23))]
        )

        async def scenario():
            await reg.get_pipe(MODEL, NUM_LAYERS, 0)  # ragged → R=1 fallback
            with pytest.raises(RuntimeError, match="Mesh disabled"):
                await reg.plan_round(MODEL)

        asyncio.run(scenario())


# ── Phase 2: merge math (§3.5 / §6.6) — needs real torch ────────────────────

torch = pytest.importorskip("torch")
_HAS_REAL_TORCH = hasattr(torch, "tensor")
needs_torch = pytest.mark.skipif(not _HAS_REAL_TORCH, reason="real torch required")


def _sd(scale: float, dtype=None) -> dict:
    """A deterministic fake adapter state dict."""
    dtype = dtype or torch.bfloat16
    g = torch.Generator().manual_seed(42)
    return {
        "lora_A.weight": (torch.randn(4, 8, generator=g) * scale).to(dtype),
        "lora_B.weight": (torch.randn(8, 4, generator=g) * scale).to(dtype),
    }


@needs_torch
class TestMergeAdapters:
    def test_no_payloads_is_noop(self):
        from mesh import merge_adapters

        # §6.6 "R=1 averaging round: exact no-op" — one contributor averaging
        # with itself is the identity; merge_adapters returns None (untouched).
        assert merge_adapters(_sd(1.0), [], contribute=True) is None

    def test_two_way_mean(self):
        from mesh import merge_adapters

        own = _sd(1.0, torch.float32)
        peer = _sd(3.0, torch.float32)
        merged = merge_adapters(own, [("peer", 10, peer)], contribute=True)
        for k in own:
            expected = (own[k].float() + peer[k].float()) / 2
            assert torch.allclose(merged[k], expected)

    def test_merge_order_independence(self):
        from mesh import merge_adapters

        # §6.6: shuffling the received list produces an identical result.
        own = _sd(1.0)
        peers = [(f"w{i}", 10, _sd(float(i + 2))) for i in range(4)]
        a = merge_adapters(own, peers, contribute=True)
        b = merge_adapters(own, list(reversed(peers)), contribute=True)
        for k in own:
            assert torch.equal(a[k], b[k])  # bit-identical

    def test_spare_excluded_from_divisor(self):
        from mesh import merge_adapters

        # §6.6 spare exclusion: a non-contributor's own (stale) weights never
        # appear in the result — it gets exactly the contributors' mean.
        stale_own = _sd(100.0, torch.float32)  # wildly stale
        c1, c2 = _sd(1.0, torch.float32), _sd(3.0, torch.float32)
        merged = merge_adapters(
            stale_own, [("c1", 10, c1), ("c2", 10, c2)], contribute=False
        )
        for k in stale_own:
            expected = (c1[k].float() + c2[k].float()) / 2
            assert torch.allclose(merged[k], expected)

    def test_fp32_accumulation_precision(self):
        from mesh import merge_adapters

        # Averaging near-zero bf16 deltas must go through fp32: build values
        # whose bf16 sum loses precision but whose fp32 mean is exact.
        own = {"b": torch.tensor([1.0], dtype=torch.float32)}
        peers = [
            ("p1", 10, {"b": torch.tensor([1.0 + 2e-4], dtype=torch.float32)}),
            ("p2", 10, {"b": torch.tensor([1.0 - 2e-4], dtype=torch.float32)}),
        ]
        merged = merge_adapters(own, peers, contribute=True)
        assert torch.allclose(merged["b"], torch.tensor([1.0]), atol=1e-7)

    def test_output_dtype_matches_own(self):
        from mesh import merge_adapters

        own = _sd(1.0, torch.bfloat16)
        merged = merge_adapters(own, [("p", 10, _sd(2.0))], contribute=True)
        for k in own:
            assert merged[k].dtype == torch.bfloat16

    def test_cross_replica_bit_identity_r3(self):
        from mesh import merge_adapters

        # Each of 3 contributors computes the mean from its own perspective
        # (own weights local, two received). fp32 addition is not associative,
        # so this only holds because accumulation order is canonical.
        sds = {f"w{i}": _sd(float(i + 1)) for i in range(3)}
        results = []
        for name, own in sds.items():
            received = [(n, 10, sd) for n, sd in sds.items() if n != name]
            results.append(
                merge_adapters(own, received, contribute=True, own_name=name)
            )
        for other in results[1:]:
            for k in results[0]:
                assert torch.equal(results[0][k], other[k])  # bit-identical

    def test_mismatched_keys_skipped(self):
        from mesh import merge_adapters

        own = _sd(1.0, torch.float32)
        bad = {"unrelated.weight": torch.zeros(2)}
        good = _sd(3.0, torch.float32)
        merged = merge_adapters(
            own, [("bad", 10, bad), ("good", 10, good)], contribute=True
        )
        for k in own:
            expected = (own[k].float() + good[k].float()) / 2
            assert torch.allclose(merged[k], expected)


@needs_torch
class TestSkewGuard:
    def test_stale_payload_dropped(self):
        from mesh import filter_skewed

        # §6.6: payload stamped step=0 against local step 200 is dropped (X=10).
        received = [("fresh", 195, {}), ("stale", 0, {})]
        kept, dropped = filter_skewed(received, local_step=200, x=10)
        assert [s for s, _, _ in kept] == ["fresh"]
        assert dropped == ["stale"]

    def test_boundary(self):
        from mesh import filter_skewed

        # threshold = 100 − 5·2 = 90; step 90 kept, 89 dropped.
        received = [("at", 90, {}), ("below", 89, {})]
        kept, dropped = filter_skewed(received, local_step=100, x=2)
        assert [s for s, _, _ in kept] == ["at"]
        assert dropped == ["below"]

    def test_promoted_spare_survives_guard(self):
        from mesh import filter_skewed

        # Regression (caught by the stage-6 e2e test): a promoted spare's
        # LINEAGE step tracks the actives it merged with, so its payloads
        # must pass the guard even though it barely trained. Active at
        # lineage 18, spare at lineage 16 (last round) + 2 trained: kept.
        kept, dropped = filter_skewed([("spare", 18, {})], local_step=18, x=2)
        assert dropped == []
        assert [s for s, _, _ in kept] == ["spare"]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
