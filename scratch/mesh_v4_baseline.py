"""EXACT v4 merge math, preserved for A/B against the Phase 1B rewrite.

DO NOT EDIT and DO NOT import from worker.py in normal operation. This is a
byte-for-byte copy of the mesh.py that produced benchmark_results_v4, whose
dense merge_compute_ms was 3,816 ms on the 3.67M-param (fp32, 140-tensor)
stage. The Phase 1B rewrite brought the same measurement to 6.0 ms on the
same box — a 636x change that NO microbenchmark reproduced: a faithful copy
of this inner loop, run bare on the same machine over the same shapes, is
10-15 ms (scratch/bench_merge.py section 6). Something about the real round
made this code 600x slower than the identical code outside it, and until
that is identified the 636x is an unexplained gift rather than a fix.

Swap this in for mesh.py (it is bind-mounted) and re-run the diagnostic to
get old-vs-new under identical conditions:

    cp mesh.py /tmp/mesh_phase1b.py
    cp scratch/mesh_v4_baseline.py mesh.py
    <run the orchestrator>
    cp /tmp/mesh_phase1b.py mesh.py

The only change from v4 is an accepted-and-ignored `timings` kwarg, so the
current worker.py can call it. The arithmetic is untouched.

Original docstring follows.

Async-mesh merge math — pure functions, no model or network access.

Kept free of FastAPI/peft/transformers imports so the §6.6 correctness tests
run on CPU with only torch installed.
"""

from __future__ import annotations

import zlib

import torch

# A received round payload: (sender_name, sender_step, adapter_state_dict)
Received = tuple[str, int, dict[str, torch.Tensor]]

SKEW_WINDOW_MULTIPLIER = 5  # drop payloads with step < local_step − 5X (§3.5)


def filter_skewed(
    received: list[Received], local_step: int, x: int
) -> tuple[list[Received], list[str]]:
    """Version-skew guard: drop payloads from peers that recently recovered.

    `local_step` and payload steps must be adapter LINEAGE steps (advanced by
    both training and merges), not raw optimizer step counts — a spare that
    never trained still holds a current adapter. Comparing optimizer counts
    makes peers drop a promoted spare's payloads forever (regression caught
    by scratch/test_stage6_mesh.py).

    Returns (kept, dropped_sender_names).
    """
    threshold = local_step - SKEW_WINDOW_MULTIPLIER * max(x, 1)
    kept: list[Received] = []
    dropped: list[str] = []
    for sender, step, sd in received:
        if step < threshold:
            dropped.append(sender)
        else:
            kept.append((sender, step, sd))
    return kept, dropped


def merge_adapters(
    own: dict[str, torch.Tensor],
    received: list[Received],
    contribute: bool,
    own_name: str = "",
    device: torch.device | str | None = None,
    timings: dict[str, float] | None = None,
) -> dict[str, torch.Tensor] | None:
    """Uniform mean of adapter state dicts (§3.5).

    Invariants this deliberately preserves — do not "simplify" them away:
      - fp32 accumulation (LoRA B deltas are ~2e-4; bf16 loses them)
      - canonical accumulation order: ALL contributions (own included) are
        sorted by sender name before summing. fp32 addition is not
        associative, so "own first, then peers" would give each replica a
        different summation order and break bit-identical consensus at R ≥ 3.
      - contributor-aware divisor (a non-contributor's own weights are
        excluded from both numerator and denominator)

    Returns the merged state dict in the own dtype, or None when there is
    nothing to apply (no payloads received — own weights stay untouched).
    """
    if not received:
        return None

    keys = sorted(own.keys())

    # Skip payloads whose key set doesn't cover ours (defensive; peers share
    # an identical stage so this should never fire).
    contributions: list[tuple[str, dict[str, torch.Tensor]]] = []
    for sender, step, sd in received:
        if all(k in sd for k in keys):
            contributions.append((sender, sd))
    if not contributions:
        return None
    if contribute:
        contributions.append((own_name, own))
    contributions.sort(key=lambda t: t[0])  # canonical order, self included

    n = len(contributions)
    acc = {k: torch.zeros(own[k].shape, dtype=torch.float32) for k in keys}
    for _sender, sd in contributions:
        for k in keys:
            acc[k] += sd[k].detach().float().cpu()

    if timings is not None:
        timings["merge_tensors"] = float(len(keys))
        timings["merge_params"] = float(sum(own[k].numel() for k in keys))
        timings["merge_n"] = float(n)
        timings["merge_impl_v4"] = 1.0
    return {k: (acc[k] / n).to(own[k].dtype) for k in keys}


# ── Sparse averaging (Phase 3) ──────────────────────────────────────────────
#
# Exchange and average only a fraction `p` of adapter parameters per round.
# The motivation for kiwi is NOT bandwidth — LoRA payloads are already tiny.
# It is that each merge then perturbs only ~p of the weights, so the LOCAL
# Adam moments (which are never transmitted, §4.2) stay approximately valid
# instead of being invalidated wholesale by a weight jump they know nothing
# about. That is a direct attack on the measured X-sweep result, where only
# X=20 — a single merge for the whole run — recovered.


def round_mask(key: str, numel: int, round_id: int, p: float) -> torch.Tensor:
    """Indices this round averages for `key`. Same answer on every peer.

    No mask bytes go on the wire: sender and receiver each recompute the
    indices from (key, round_id), which only works if the seed is stable
    ACROSS PROCESSES.

    Python's builtin `hash()` of a str or tuple is salted per interpreter
    (PYTHONHASHSEED), so seeding a Generator with `hash((key, round_id))`
    gives a different permutation in every worker container while looking
    perfectly deterministic inside any single process. Each replica would
    then write a correct-looking mean into DIFFERENT positions, corrupting
    the adapter with no error raised anywhere. zlib.crc32 is a fixed
    function of the bytes and has no such salt. See test_mesh_sparse.py::
    TestMaskDeterminism::test_mask_is_stable_across_processes.

    `p` is deliberately NOT part of the seed. The mask is a prefix of one
    fixed permutation per (key, round_id), so a smaller p yields a SUBSET of
    a larger p's indices rather than an unrelated set. The authoritative
    defence against a p disagreement is still the length check in
    merge_sparse — this just makes the failure mode benign rather than
    scrambled.

    Returned indices are int64 and sorted ascending (canonical order, stable
    payload layout, and p >= 1.0 collapses to a plain flatten).
    """
    if numel <= 0:
        return torch.empty(0, dtype=torch.int64)
    if p >= 1.0:
        return torch.arange(numel, dtype=torch.int64)
    # max(1, ...) so a small tensor never drops out of averaging entirely.
    k = max(1, min(numel, int(numel * p)))
    if k >= numel:
        return torch.arange(numel, dtype=torch.int64)
    seed = zlib.crc32(f"{key}:{round_id}".encode("utf-8")) & 0x7FFFFFFF
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(numel, generator=generator)
    return perm[:k].sort().values.to(torch.int64)


def sparse_snapshot(
    state: dict[str, torch.Tensor], round_id: int, p: float
) -> dict[str, torch.Tensor]:
    """The contributor's wire payload: flattened masked values only.

    Same tensor keys as the dense path, but each value is a 1-D tensor of
    just the selected entries. Indices are never transmitted — the receiver
    recomputes them from (key, round_id, p).
    """
    out: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        flat = value.detach().reshape(-1)
        if p >= 1.0:
            out[key] = flat.clone()
        else:
            idx = round_mask(key, flat.numel(), round_id, p)
            # Index on the tensor's own device. `state` is the live adapter,
            # so on a worker these are CUDA tensors; gathering with a CPU
            # index tensor either errors or forces a synchronising round trip
            # per tensor, on the hot path of every round.
            out[key] = flat[idx.to(flat.device)].clone()
    return out


def merge_sparse(
    own: dict[str, torch.Tensor],
    received: list[Received],
    contribute: bool,
    round_id: int,
    p: float,
    own_name: str = "",
    device: torch.device | str | None = None,
    timings: dict[str, float] | None = None,
) -> dict[str, torch.Tensor] | None:
    """Average only the masked entries; leave everything else untouched.

    `own` is the FULL adapter state dict; each entry of `received` holds
    flattened masked values (as produced by sparse_snapshot).

    Every invariant of merge_adapters is preserved deliberately:
      - fp32 accumulation (LoRA B deltas are ~2e-4; bf16 loses them),
      - canonical accumulation order — ALL contributions, own included, are
        sorted by sender name before summing, because fp32 addition is not
        associative and per-replica ordering would break bit-identical
        consensus at R >= 3,
      - contributor-aware divisor: a non-contributor (spare) is excluded
        from both numerator and denominator, so it lands on exactly the
        contributors' mean at the masked entries and keeps its own values
        everywhere else.

    At p >= 1.0 this is BITWISE identical to merge_adapters: the mask is the
    identity permutation, so the same fp32 additions happen in the same order
    on the same values. That equivalence is what makes the full-sync round of
    §3.3 a true dense round rather than a near-dense one.

    Returns the merged state dict in the own dtype, or None when there is
    nothing to apply.
    """
    if not received:
        return None

    keys = sorted(own.keys())
    masks = {k: round_mask(k, own[k].numel(), round_id, p) for k in keys}

    contributions: list[tuple[str, dict[str, torch.Tensor]]] = []
    for sender, _step, sd in received:
        # Defensive, in the same spirit as the dense merge: peers share an
        # identical stage, so neither of these should ever fire.
        if not all(k in sd for k in keys):
            continue
        # A length disagreement means the sender used a different p (or a
        # different mask function). Its values correspond to DIFFERENT
        # positions than ours, so averaging them would scatter that peer's
        # weights into unrelated coordinates. Drop the payload.
        if any(sd[k].numel() != masks[k].numel() for k in keys):
            continue
        contributions.append((sender, sd))
    if not contributions:
        return None

    if contribute:
        contributions.append((
            own_name,
            {
                k: own[k].detach().reshape(-1)[
                    masks[k].to(own[k].device)   # own may live on CUDA
                ]
                for k in keys
            },
        ))
    contributions.sort(key=lambda t: t[0])  # canonical order, self included

    n = len(contributions)
    acc = {k: torch.zeros(masks[k].numel(), dtype=torch.float32) for k in keys}
    for _sender, sd in contributions:
        for k in keys:
            acc[k] += sd[k].detach().reshape(-1).float().cpu()

    merged: dict[str, torch.Tensor] = {}
    for k in keys:
        base = own[k].detach().cpu().clone()
        flat = base.reshape(-1)          # a view of `base` (it is contiguous)
        flat[masks[k]] = (acc[k] / n).to(own[k].dtype)
        merged[k] = flat.view(own[k].shape)
    if timings is not None:
        timings["merge_tensors"] = float(len(keys))
        timings["merge_params"] = float(sum(own[k].numel() for k in keys))
        timings["merge_n"] = float(n)
        timings["merge_impl_v4"] = 1.0
    return merged


# ── Outer optimizer (Phase 4, DiLoCo-style) ─────────────────────────────────


def outer_step(
    anchor: dict[str, torch.Tensor],
    avg: dict[str, torch.Tensor],
    buf: dict[str, torch.Tensor],
    lr: float = 0.7,
    mu: float = 0.9,
    device: torch.device | str | None = None,
    timings: dict[str, float] | None = None,
) -> dict[str, torch.Tensor]:
    """One Nesterov outer step: anchor + nesterov(avg - anchor).

    The alternative fix for the same Adam-jump problem sparse averaging
    attacks. Instead of frequent tiny merges, take RARE dense merges and
    apply them through an outer optimizer with momentum:

        delta = avg - anchor                (the pseudo-gradient)
        m     = mu * m + delta              (momentum buffer, local)
        out   = anchor + lr * (delta + mu * m)

    `buf` is mutated in place and is LOCAL — never transmitted, exactly like
    the Adam moments (§4.2). It is held in fp32 regardless of the weight
    dtype: it accumulates small deltas across many rounds, and in bf16 a
    2e-4 delta added to an O(1) buffer would round away to nothing.

    Determinism: this is a pure function of (anchor, avg, buf). After a dense
    merge every replica holds identical anchor and avg and has seen the same
    sequence of rounds, so every replica computes a bit-identical result and
    "agreement = checkpoint" survives.
    """
    out: dict[str, torch.Tensor] = {}
    for key in sorted(anchor.keys()):
        if key not in avg:
            out[key] = anchor[key]
            continue
        anchor_f32 = anchor[key].detach().float().cpu()
        delta = avg[key].detach().float().cpu() - anchor_f32
        momentum = buf.get(key)
        momentum = delta.clone() if momentum is None else mu * momentum + delta
        buf[key] = momentum
        out[key] = (anchor_f32 + lr * (delta + mu * momentum)).to(anchor[key].dtype)
    return out
