"""Sparse averaging correctness (IMPLEMENTATION_PLAN.md Phase 3.4).

CPU-only, torch-only — same constraint as mesh.py itself, so these run
without a cluster, without CUDA, and without peft/transformers.

These tests exist BEFORE the worker.py wiring on purpose. The failure mode
they guard is silent: if same-stage peers disagree about which indices a
round selects, every replica writes a correct-looking mean into the WRONG
positions and the run degrades with no error anywhere. The single most
likely cause is seeding a Generator from Python's builtin `hash()`, which is
salted per process (PYTHONHASHSEED) and therefore differs between two
workers in two containers while being perfectly stable inside any one test
process. `test_mask_is_stable_across_processes` is the test that catches it;
`test_builtin_hash_would_have_desynced` proves that test has teeth.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

torch = pytest.importorskip("torch")


def _torch_is_functional() -> bool:
    """Probe torch by USING it, not by inspecting attributes (see test_mesh.py)."""
    try:
        g = torch.Generator().manual_seed(0)
        a = torch.randn(2, 2, generator=g)
        return bool(torch.equal(a, a.clone()))
    except Exception:
        return False


needs_torch = pytest.mark.skipif(
    not _torch_is_functional(), reason="a functional torch install is required"
)

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def _sd(scale: float, dtype=None, seed: int = 42) -> dict:
    """A deterministic fake adapter state dict (shapes mirror test_mesh.py)."""
    dtype = dtype or torch.bfloat16
    g = torch.Generator().manual_seed(seed)
    return {
        "lora_A.weight": (torch.randn(4, 8, generator=g) * scale).to(dtype),
        "lora_B.weight": (torch.randn(8, 4, generator=g) * scale).to(dtype),
    }


# ── 3.4.1  Mask determinism ─────────────────────────────────────────────────


@needs_torch
class TestMaskDeterminism:
    def test_same_inputs_same_indices(self):
        from mesh import round_mask

        a = round_mask("lora_A.weight", 1000, 7, 0.05)
        b = round_mask("lora_A.weight", 1000, 7, 0.05)
        assert torch.equal(a, b)
        assert a.dtype == torch.int64

    def test_different_round_different_indices(self):
        from mesh import round_mask

        a = round_mask("lora_A.weight", 1000, 7, 0.05)
        b = round_mask("lora_A.weight", 1000, 8, 0.05)
        assert a.shape == b.shape
        assert not torch.equal(a, b)

    def test_different_key_different_indices(self):
        from mesh import round_mask

        a = round_mask("lora_A.weight", 1000, 7, 0.05)
        b = round_mask("lora_B.weight", 1000, 7, 0.05)
        assert not torch.equal(a, b)

    def test_size_and_bounds(self):
        from mesh import round_mask

        idx = round_mask("k", 1000, 3, 0.05)
        assert idx.numel() == 50
        assert int(idx.min()) >= 0
        assert int(idx.max()) < 1000
        assert idx.unique().numel() == idx.numel()  # no duplicate positions

    def test_at_least_one_index(self):
        from mesh import round_mask

        # p*numel < 1 must still select something, or a small tensor would
        # silently stop participating in averaging altogether.
        assert round_mask("k", 10, 1, 0.001).numel() == 1

    def test_p_one_selects_everything_in_order(self):
        from mesh import round_mask

        idx = round_mask("k", 64, 5, 1.0)
        assert torch.equal(idx, torch.arange(64))

    def test_mask_is_sorted(self):
        from mesh import round_mask

        # Canonical (ascending) order. Sender and receiver both recompute the
        # mask, so ordering only has to be a FUNCTION of the inputs — but a
        # sorted mask also makes the gathered payload layout stable and
        # inspectable, and makes p=1.0 collapse to a plain flatten.
        idx = round_mask("k", 500, 2, 0.1)
        assert torch.equal(idx, idx.sort().values)

    def test_masks_are_nested_in_p(self):
        from mesh import round_mask

        # Seeding on (key, round_id) alone — NOT on p — means a smaller p
        # yields a prefix of the same permutation. If two peers ever disagree
        # about p, the smaller mask is a subset of the larger rather than an
        # unrelated set. Cheap defence in depth; the length check in
        # merge_sparse is the one that actually rejects the payload.
        small = round_mask("k", 1000, 4, 0.05)
        large = round_mask("k", 1000, 4, 0.20)
        assert set(small.tolist()).issubset(set(large.tolist()))

    def test_mask_is_stable_across_processes(self):
        """THE test this whole file exists for.

        Two fresh interpreters with DIFFERENT PYTHONHASHSEED values must
        produce byte-identical indices. This is what a real cluster does:
        each worker is its own process, so any per-process randomness in the
        seed desynchronises the mask with no error raised anywhere.
        """
        from mesh import round_mask

        script = textwrap.dedent(
            """
            import sys, torch
            sys.path.insert(0, sys.argv[1])
            from mesh import round_mask
            idx = round_mask("lora_A.weight", 4096, 11, 0.05)
            print(",".join(str(int(i)) for i in idx))
            """
        )
        outs = []
        for hashseed in ("0", "1", "12345", "random"):
            env = dict(os.environ, PYTHONHASHSEED=hashseed)
            proc = subprocess.run(
                [sys.executable, "-c", script, REPO_ROOT],
                capture_output=True, text=True, env=env, timeout=300,
            )
            assert proc.returncode == 0, proc.stderr
            outs.append(proc.stdout.strip())

        assert len(set(outs)) == 1, (
            "round_mask is NOT process-stable — same (key, round_id, p) gave "
            "different indices under different PYTHONHASHSEED. Peers would "
            "average into different positions with no error. Seed the "
            "Generator from zlib.crc32, never from builtin hash()."
        )
        local = ",".join(str(int(i)) for i in round_mask("lora_A.weight", 4096, 11, 0.05))
        assert local == outs[0]

    def test_builtin_hash_would_have_desynced(self):
        """Proves the test above has teeth.

        If builtin hash() of a str/tuple were process-stable, the previous
        test would pass for the wrong reason and the guard would be worthless.
        Assert the salt is real on this interpreter.
        """
        script = 'print(hash(("lora_A.weight", 11)))'
        outs = set()
        for hashseed in ("0", "1", "12345"):
            env = dict(os.environ, PYTHONHASHSEED=hashseed)
            proc = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True, text=True, env=env, timeout=120,
            )
            assert proc.returncode == 0, proc.stderr
            outs.add(proc.stdout.strip())
        assert len(outs) > 1, (
            "builtin hash() appears unsalted on this interpreter, so "
            "test_mask_is_stable_across_processes cannot distinguish a crc32 "
            "seed from a hash() seed. Do not treat that test as passing."
        )


# ── 3.4.2  p = 1.0 reduces exactly to merge_adapters ────────────────────────


@needs_torch
class TestDenseEquivalence:
    def test_p_one_is_bitwise_merge_adapters(self):
        from mesh import merge_adapters, merge_sparse, sparse_snapshot

        own = _sd(1.0, torch.float32)
        peers = {f"w{i}": _sd(float(i + 2), torch.float32) for i in range(3)}

        dense = merge_adapters(
            own,
            [(n, 10, sd) for n, sd in peers.items()],
            contribute=True,
            own_name="own",
        )
        sparse = merge_sparse(
            own,
            [(n, 10, sparse_snapshot(sd, round_id=1, p=1.0)) for n, sd in peers.items()],
            contribute=True,
            round_id=1,
            p=1.0,
            own_name="own",
        )
        for k in own:
            assert torch.equal(dense[k], sparse[k]), f"{k} diverged at p=1.0"

    def test_p_one_bitwise_in_bf16(self):
        from mesh import merge_adapters, merge_sparse, sparse_snapshot

        own = _sd(1.0, torch.bfloat16)
        peer = _sd(3.0, torch.bfloat16)
        dense = merge_adapters(own, [("p", 10, peer)], contribute=True, own_name="own")
        sparse = merge_sparse(
            own,
            [("p", 10, sparse_snapshot(peer, round_id=2, p=1.0))],
            contribute=True, round_id=2, p=1.0, own_name="own",
        )
        for k in own:
            assert sparse[k].dtype == torch.bfloat16
            assert torch.equal(dense[k], sparse[k])

    def test_no_payloads_is_noop(self):
        from mesh import merge_sparse

        # Mirrors merge_adapters: nothing received -> None -> own untouched.
        assert merge_sparse(_sd(1.0), [], contribute=True,
                            round_id=1, p=0.05, own_name="own") is None


# ── 3.4.3  Two-replica sparse merge ─────────────────────────────────────────


@needs_torch
class TestSparseMerge:
    def test_masked_mean_unmasked_untouched(self):
        from mesh import merge_sparse, round_mask, sparse_snapshot

        p, rid = 0.25, 3
        own = _sd(1.0, torch.float32)
        peer = _sd(5.0, torch.float32)

        merged = merge_sparse(
            own,
            [("peer", 10, sparse_snapshot(peer, rid, p))],
            contribute=True, round_id=rid, p=p, own_name="aaa",
        )

        for k in own:
            idx = round_mask(k, own[k].numel(), rid, p)
            flat_own = own[k].flatten().float()
            flat_peer = peer[k].flatten().float()
            flat_merged = merged[k].flatten().float()

            expected = (flat_own[idx] + flat_peer[idx]) / 2
            assert torch.allclose(flat_merged[idx], expected, atol=1e-6)

            # Everything outside the mask is bit-identical to own.
            keep = torch.ones(own[k].numel(), dtype=torch.bool)
            keep[idx] = False
            assert torch.equal(merged[k].flatten()[keep], own[k].flatten()[keep])

    def test_only_p_fraction_moves(self):
        from mesh import merge_sparse, sparse_snapshot

        # The whole point of sparse rounds: a merge perturbs ~p of the
        # parameters, so local Adam moments stay approximately valid.
        p, rid = 0.05, 9
        own = _sd(1.0, torch.float32)
        peer = _sd(9.0, torch.float32)
        merged = merge_sparse(
            own, [("peer", 10, sparse_snapshot(peer, rid, p))],
            contribute=True, round_id=rid, p=p, own_name="aaa",
        )
        for k in own:
            changed = (merged[k] != own[k]).sum().item()
            assert changed <= max(1, int(own[k].numel() * p))

    def test_both_replicas_bit_identical_on_masked_entries(self):
        from mesh import merge_sparse, round_mask, sparse_snapshot

        # Each replica merges from its OWN perspective. fp32 addition is not
        # associative, so this only holds because contributions (self
        # included) are accumulated in canonical sender order.
        p, rid = 0.25, 4
        sds = {f"w{i}": _sd(float(i + 1), torch.float32) for i in range(3)}
        merged = {}
        for name, own in sds.items():
            received = [
                (n, 10, sparse_snapshot(sd, rid, p))
                for n, sd in sds.items() if n != name
            ]
            merged[name] = merge_sparse(
                own, received, contribute=True, round_id=rid, p=p, own_name=name
            )

        ref = merged["w0"]
        for name, m in merged.items():
            for k in ref:
                idx = round_mask(k, ref[k].numel(), rid, p)
                assert torch.equal(
                    m[k].flatten()[idx], ref[k].flatten()[idx]
                ), f"{name} disagrees with w0 on masked entries of {k}"

    def test_replicas_still_differ_off_mask(self):
        from mesh import merge_sparse, round_mask, sparse_snapshot

        # Sparse rounds leave replicas only APPROXIMATELY equal — that is why
        # Phase 3.3 needs the periodic full sync. Assert the drift is real, so
        # a bug that accidentally makes sparse rounds dense gets noticed.
        p, rid = 0.1, 6
        a, b = _sd(1.0, torch.float32), _sd(4.0, torch.float32)
        ma = merge_sparse(a, [("b", 10, sparse_snapshot(b, rid, p))],
                          contribute=True, round_id=rid, p=p, own_name="a")
        mb = merge_sparse(b, [("a", 10, sparse_snapshot(a, rid, p))],
                          contribute=True, round_id=rid, p=p, own_name="b")
        differ = False
        for k in a:
            idx = round_mask(k, a[k].numel(), rid, p)
            keep = torch.ones(a[k].numel(), dtype=torch.bool)
            keep[idx] = False
            if not torch.equal(ma[k].flatten()[keep], mb[k].flatten()[keep]):
                differ = True
        assert differ, "sparse round left replicas identical — mask is not sparse"

    def test_output_dtype_matches_own(self):
        from mesh import merge_sparse, sparse_snapshot

        own = _sd(1.0, torch.bfloat16)
        merged = merge_sparse(
            own, [("p", 10, sparse_snapshot(_sd(2.0), 1, 0.25))],
            contribute=True, round_id=1, p=0.25, own_name="own",
        )
        for k in own:
            assert merged[k].dtype == torch.bfloat16
            assert merged[k].shape == own[k].shape

    def test_fp32_accumulation_precision(self):
        from mesh import merge_sparse

        # Same guard as the dense merge: bf16-scale deltas (~2e-4) must be
        # accumulated in fp32 or the average loses them entirely.
        own = {"b": torch.tensor([1.0, 1.0], dtype=torch.float32)}
        p, rid = 1.0, 1
        received = [
            ("p1", 10, {"b": torch.tensor([1.0 + 2e-4, 1.0], dtype=torch.float32)}),
            ("p2", 10, {"b": torch.tensor([1.0 - 2e-4, 1.0], dtype=torch.float32)}),
        ]
        merged = merge_sparse(own, received, contribute=True,
                              round_id=rid, p=p, own_name="own")
        assert torch.allclose(merged["b"], torch.tensor([1.0, 1.0]), atol=1e-7)

    def test_order_independence(self):
        from mesh import merge_sparse, sparse_snapshot

        p, rid = 0.25, 8
        own = _sd(1.0, torch.float32)
        peers = [(f"w{i}", 10, sparse_snapshot(_sd(float(i + 2), torch.float32), rid, p))
                 for i in range(4)]
        a = merge_sparse(own, peers, contribute=True, round_id=rid, p=p, own_name="own")
        b = merge_sparse(own, list(reversed(peers)), contribute=True,
                         round_id=rid, p=p, own_name="own")
        for k in own:
            assert torch.equal(a[k], b[k])

    def test_wrong_length_payload_is_skipped(self):
        from mesh import merge_sparse, sparse_snapshot

        # A payload whose masked-value length disagrees with our own mask
        # means the sender used a different p (or a different mask function).
        # Averaging it would write peer values into unrelated positions, so it
        # must be dropped rather than reshaped into place.
        p, rid = 0.25, 5
        own = _sd(1.0, torch.float32)
        good = _sd(3.0, torch.float32)
        bad = sparse_snapshot(_sd(99.0, torch.float32), rid, 0.5)  # wrong p
        merged = merge_sparse(
            own,
            [("bad", 10, bad), ("good", 10, sparse_snapshot(good, rid, p))],
            contribute=True, round_id=rid, p=p, own_name="aaa",
        )
        from mesh import round_mask
        for k in own:
            idx = round_mask(k, own[k].numel(), rid, p)
            expected = (own[k].flatten().float()[idx] + good[k].flatten().float()[idx]) / 2
            assert torch.allclose(merged[k].flatten().float()[idx], expected, atol=1e-6)

    def test_mismatched_keys_skipped(self):
        from mesh import merge_sparse, round_mask, sparse_snapshot

        p, rid = 0.25, 5
        own = _sd(1.0, torch.float32)
        good = _sd(3.0, torch.float32)
        merged = merge_sparse(
            own,
            [("bad", 10, {"unrelated.weight": torch.zeros(4)}),
             ("good", 10, sparse_snapshot(good, rid, p))],
            contribute=True, round_id=rid, p=p, own_name="aaa",
        )
        for k in own:
            idx = round_mask(k, own[k].numel(), rid, p)
            expected = (own[k].flatten().float()[idx] + good[k].flatten().float()[idx]) / 2
            assert torch.allclose(merged[k].flatten().float()[idx], expected, atol=1e-6)


# ── 3.4.4  Full sync restores exact agreement ───────────────────────────────


@needs_torch
class TestFullSync:
    def test_full_sync_makes_replicas_bit_identical(self):
        from mesh import merge_adapters, merge_sparse, sparse_snapshot

        # k sparse rounds leave replicas approximately equal; the dense
        # full-sync round must restore "agreement = checkpoint" EXACTLY, or
        # spare promotion and recovery lose their zero-cost guarantee.
        p = 0.1
        state = {f"w{i}": _sd(float(i + 1), torch.float32, seed=100 + i) for i in range(3)}

        for rid in range(1, 6):  # 5 sparse rounds
            new = {}
            for name, own in state.items():
                received = [
                    (n, 10, sparse_snapshot(sd, rid, p))
                    for n, sd in state.items() if n != name
                ]
                new[name] = merge_sparse(own, received, contribute=True,
                                         round_id=rid, p=p, own_name=name)
            state = new

        # Drifted, but only off-mask.
        assert any(
            not torch.equal(state["w0"][k], state["w1"][k]) for k in state["w0"]
        ), "replicas identical before full sync — sparse rounds were not sparse"

        # Full sync: dense round, existing merge_adapters path.
        synced = {}
        for name, own in state.items():
            received = [(n, 10, sd) for n, sd in state.items() if n != name]
            synced[name] = merge_adapters(own, received, contribute=True, own_name=name)

        ref = synced["w0"]
        for name, m in synced.items():
            for k in ref:
                assert torch.equal(m[k], ref[k]), (
                    f"{name} not bit-identical to w0 on {k} after full sync"
                )


# ── 3.4.5  Contributor-aware divisor (spares) ───────────────────────────────


@needs_torch
class TestSpareExclusion:
    def test_spare_excluded_from_numerator_and_denominator(self):
        from mesh import merge_sparse, round_mask, sparse_snapshot

        # Mirrors TestMergeAdapters.test_spare_excluded_from_divisor. A spare
        # receives without contributing: on masked entries it must land on
        # exactly the CONTRIBUTORS' mean, with none of its own stale weight
        # mixed in.
        p, rid = 0.25, 2
        stale_own = _sd(100.0, torch.float32)  # wildly stale
        c1, c2 = _sd(1.0, torch.float32), _sd(3.0, torch.float32)

        merged = merge_sparse(
            stale_own,
            [("c1", 10, sparse_snapshot(c1, rid, p)),
             ("c2", 10, sparse_snapshot(c2, rid, p))],
            contribute=False, round_id=rid, p=p, own_name="spare",
        )
        for k in stale_own:
            idx = round_mask(k, stale_own[k].numel(), rid, p)
            expected = (c1[k].flatten().float()[idx] + c2[k].flatten().float()[idx]) / 2
            assert torch.allclose(merged[k].flatten().float()[idx], expected, atol=1e-6)

    def test_spare_keeps_own_values_off_mask(self):
        from mesh import merge_sparse, round_mask, sparse_snapshot

        p, rid = 0.25, 2
        stale_own = _sd(100.0, torch.float32)
        c1 = _sd(1.0, torch.float32)
        merged = merge_sparse(
            stale_own, [("c1", 10, sparse_snapshot(c1, rid, p))],
            contribute=False, round_id=rid, p=p, own_name="spare",
        )
        for k in stale_own:
            idx = round_mask(k, stale_own[k].numel(), rid, p)
            keep = torch.ones(stale_own[k].numel(), dtype=torch.bool)
            keep[idx] = False
            assert torch.equal(
                merged[k].flatten()[keep], stale_own[k].flatten()[keep]
            )

    def test_spare_matches_contributor_on_masked_entries(self):
        from mesh import merge_sparse, round_mask, sparse_snapshot

        # A spare that received the same payloads must agree BIT-for-bit with
        # the actives on the masked entries — that agreement is what makes
        # promotion zero-cost.
        p, rid = 0.25, 12
        a, b = _sd(1.0, torch.float32), _sd(2.0, torch.float32)
        spare = _sd(50.0, torch.float32)

        merged_a = merge_sparse(
            a, [("b", 10, sparse_snapshot(b, rid, p))],
            contribute=True, round_id=rid, p=p, own_name="a",
        )
        merged_spare = merge_sparse(
            spare,
            [("a", 10, sparse_snapshot(a, rid, p)),
             ("b", 10, sparse_snapshot(b, rid, p))],
            contribute=False, round_id=rid, p=p, own_name="spare",
        )
        for k in a:
            idx = round_mask(k, a[k].numel(), rid, p)
            assert torch.equal(
                merged_a[k].flatten()[idx], merged_spare[k].flatten()[idx]
            )


# ── sparse_snapshot ─────────────────────────────────────────────────────────


@needs_torch
class TestSparseSnapshot:
    def test_snapshot_is_flat_masked_values(self):
        from mesh import round_mask, sparse_snapshot

        sd = _sd(1.0, torch.float32)
        snap = sparse_snapshot(sd, round_id=3, p=0.25)
        for k, v in sd.items():
            idx = round_mask(k, v.numel(), 3, 0.25)
            assert snap[k].ndim == 1
            assert snap[k].numel() == idx.numel()
            assert torch.equal(snap[k], v.flatten()[idx])

    def test_snapshot_preserves_dtype(self):
        from mesh import sparse_snapshot

        snap = sparse_snapshot(_sd(1.0, torch.bfloat16), 3, 0.25)
        assert all(v.dtype == torch.bfloat16 for v in snap.values())

    def test_snapshot_is_much_smaller_than_dense(self):
        from mesh import sparse_snapshot

        sd = {"w": torch.randn(1000, dtype=torch.float32)}
        snap = sparse_snapshot(sd, 1, 0.05)
        assert snap["w"].numel() == 50

    def test_p_one_snapshot_is_flatten(self):
        from mesh import sparse_snapshot

        sd = _sd(1.0, torch.float32)
        snap = sparse_snapshot(sd, 1, 1.0)
        for k, v in sd.items():
            assert torch.equal(snap[k], v.flatten())


# ── Phase 4: outer optimizer (DiLoCo) ───────────────────────────────────────


@needs_torch
class TestOuterStep:
    def test_zero_delta_is_identity(self):
        from mesh import outer_step

        anchor = _sd(1.0, torch.float32)
        buf: dict = {}
        out = outer_step(anchor, dict(anchor), buf, lr=0.7, mu=0.9)
        for k in anchor:
            assert torch.equal(out[k], anchor[k])

    def test_single_step_matches_nesterov_formula(self):
        from mesh import outer_step

        anchor = {"w": torch.tensor([0.0, 0.0], dtype=torch.float32)}
        avg = {"w": torch.tensor([1.0, 2.0], dtype=torch.float32)}
        buf: dict = {}
        lr, mu = 0.7, 0.9
        out = outer_step(anchor, avg, buf, lr=lr, mu=mu)

        # delta = avg - anchor;  m1 = mu*0 + delta = delta
        # nesterov update = lr * (delta + mu * m1) = lr * (1 + mu) * delta
        expected = lr * (1 + mu) * avg["w"]
        assert torch.allclose(out["w"], expected)

    def test_momentum_accumulates_across_rounds(self):
        from mesh import outer_step

        anchor = {"w": torch.zeros(2, dtype=torch.float32)}
        buf: dict = {}
        lr, mu = 0.7, 0.9
        delta = torch.tensor([1.0, 1.0])

        first = outer_step(anchor, {"w": delta.clone()}, buf, lr=lr, mu=mu)
        second = outer_step(first, {"w": first["w"] + delta}, buf, lr=lr, mu=mu)

        # Second step must travel FURTHER than the first for the same delta —
        # that is the whole point of the momentum buffer.
        assert (second["w"] - first["w"]).abs().min() > (first["w"] - anchor["w"]).abs().max()

    def test_buffer_is_mutated_in_place(self):
        from mesh import outer_step

        anchor = {"w": torch.zeros(2, dtype=torch.float32)}
        buf: dict = {}
        outer_step(anchor, {"w": torch.ones(2)}, buf, lr=0.7, mu=0.9)
        assert "w" in buf and buf["w"].dtype == torch.float32

    def test_replicas_stay_bit_identical(self):
        from mesh import outer_step

        # After a dense merge every replica holds identical inputs, so the
        # outer step must be a pure function of them — same anchor, same avg,
        # same buffer history => bit-identical output on every replica.
        anchor = _sd(1.0, torch.float32)
        avg = _sd(2.0, torch.float32)
        outs = []
        for _ in range(3):
            buf: dict = {}
            outs.append(outer_step(dict(anchor), dict(avg), buf, lr=0.7, mu=0.9))
        for other in outs[1:]:
            for k in outs[0]:
                assert torch.equal(outs[0][k], other[k])

    def test_output_dtype_matches_anchor(self):
        from mesh import outer_step

        anchor = _sd(1.0, torch.bfloat16)
        avg = _sd(2.0, torch.bfloat16)
        out = outer_step(anchor, avg, {}, lr=0.7, mu=0.9)
        for k in anchor:
            assert out[k].dtype == torch.bfloat16
            assert out[k].shape == anchor[k].shape

    def test_buffer_stays_fp32_under_bf16_weights(self):
        from mesh import outer_step

        # The momentum buffer is the one place small deltas accumulate across
        # many rounds; keeping it bf16 would round 2e-4 deltas to nothing.
        anchor = _sd(1.0, torch.bfloat16)
        buf: dict = {}
        outer_step(anchor, _sd(2.0, torch.bfloat16), buf, lr=0.7, mu=0.9)
        assert all(v.dtype == torch.float32 for v in buf.values())
