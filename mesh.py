"""Async-mesh merge math — pure functions, no model or network access.

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


# ── Vectorised accumulation helpers (MERGE_PERF_PLAN.md §4.1B) ──────────────
#
# The merge is ~12M fp32 additions over ~112 small tensors. In the v4 profile
# that cost 3,267 ms — roughly 3.6 Melem-ops/s, about 100x below what a single
# core manages. The arithmetic is not what was expensive; the per-tensor Python
# loop around it was, because each iteration is a separate ATen dispatch and,
# for the tensors above ATen's ~32k-element parallel grain size, a separate
# (and with 7 workers per host, contended) OpenMP fork/join. See
# scratch/bench_merge.py section 6, which races this implementation against the
# per-tensor loop it replaced.
#
# torch._foreach_* does the identical arithmetic with ONE dispatch for the
# whole list, which is exactly why optimizers use it. Nothing about the
# numerics changes: same fp32 elementwise adds, same values, same order.


def _target_device(
    own: dict[str, torch.Tensor],
    keys: list[str],
    device: torch.device | str | None,
) -> torch.device:
    """Where to do the arithmetic. Defaults to where `own` already lives.

    Deliberately NOT hard-coded to CPU or CUDA. mesh.py must stay importable
    and testable with a CPU-only torch (§5.7) — that property is what lets the
    whole merge be developed and unit-tested on a machine with no usable GPU —
    and a future GPU-side merge (§4.1C) needs only to pass a device in.
    """
    if device is not None:
        return torch.device(device)
    return own[keys[0]].device if keys else torch.device("cpu")


def _accumulate(
    per_contribution: list[list[torch.Tensor]], dev: torch.device
) -> list[torch.Tensor]:
    """Sum lists of tensors elementwise in fp32, in the order given.

    `per_contribution` must already be in canonical sender-name-sorted order —
    fp32 addition is not associative, so the caller owning that ordering is
    what keeps replicas bit-identical (§5.1).

    The first contribution SEEDS the accumulator rather than being added into a
    zeros buffer. `0.0 + x` is exactly `x` in fp32, so this removes a full pass
    over the adapter (and an allocation of ~12 MB of zeros) without changing a
    single bit. `copy=True` is load-bearing: without it, a contribution that is
    already fp32 on the target device would be returned as-is by `.to()`, and
    the in-place adds below would corrupt a peer's payload — or our own live
    weights, when `contribute=True`.
    """
    acc: list[torch.Tensor] | None = None
    for tensors in per_contribution:
        converted = [
            t.detach().to(device=dev, dtype=torch.float32, copy=acc is None)
            for t in tensors
        ]
        if acc is None:
            acc = converted
        else:
            torch._foreach_add_(acc, converted)
    assert acc is not None  # callers return early on an empty contribution list
    return acc


def merge_adapters(
    own: dict[str, torch.Tensor],
    received: list[Received],
    contribute: bool,
    own_name: str = "",
    device: torch.device | str | None = None,
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
    dev = _target_device(own, keys, device)
    acc = _accumulate(
        [[sd[k] for k in keys] for _sender, sd in contributions], dev
    )
    torch._foreach_div_(acc, float(n))
    return {k: acc[i].to(own[k].dtype) for i, k in enumerate(keys)}


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
    dev = _target_device(own, keys, device)
    # Same vectorised accumulation as the dense path, over the same canonically
    # ordered contributions. Using one shared helper is what makes the p >= 1.0
    # bitwise equivalence (§5.2) hold by construction rather than by review.
    acc = _accumulate(
        [[sd[k].detach().reshape(-1) for k in keys] for _sender, sd in contributions],
        dev,
    )
    torch._foreach_div_(acc, float(n))

    merged: dict[str, torch.Tensor] = {}
    for i, k in enumerate(keys):
        base = own[k].detach().to(dev).clone()
        flat = base.reshape(-1)          # a view of `base` (it is contiguous)
        flat[masks[k].to(dev)] = acc[i].to(own[k].dtype)
        merged[k] = flat.view(own[k].shape)
    return merged


# ── Outer optimizer (Phase 4, DiLoCo-style) ─────────────────────────────────


def outer_step(
    anchor: dict[str, torch.Tensor],
    avg: dict[str, torch.Tensor],
    buf: dict[str, torch.Tensor],
    lr: float = 0.7,
    mu: float = 0.9,
    device: torch.device | str | None = None,
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
    all_keys = sorted(anchor.keys())
    keys = [k for k in all_keys if k in avg]
    # Keys the average does not cover pass through untouched, exactly as before.
    stepped: dict[str, torch.Tensor] = {k: anchor[k] for k in all_keys if k not in avg}

    if keys:
        dev = _target_device(anchor, keys, device)
        # copy=False on the anchor: it is only ever read below.
        anchor_f32 = [
            anchor[k].detach().to(device=dev, dtype=torch.float32) for k in keys
        ]
        # copy=True on delta: it is subtracted into in place, and without the
        # copy an already-fp32 `avg` would be mutated — which would corrupt the
        # merged average the caller is about to store as the next anchor.
        delta = [
            avg[k].detach().to(device=dev, dtype=torch.float32, copy=True) for k in keys
        ]
        torch._foreach_sub_(delta, anchor_f32)

        # m = mu * m + delta, batched. Keys absent from `buf` (the first outer
        # step) seed from delta; in steady state every key is present, so the
        # `old` branch is the one that matters and it is a single pair of
        # dispatches for the whole adapter.
        momentum: list[torch.Tensor] = [None] * len(keys)  # type: ignore[list-item]
        old = [i for i, k in enumerate(keys) if k in buf]
        for i, k in enumerate(keys):
            if k not in buf:
                momentum[i] = delta[i].clone()
        if old:
            updated = torch._foreach_mul([buf[keys[i]] for i in old], mu)
            torch._foreach_add_(updated, [delta[i] for i in old])
            for j, i in enumerate(old):
                momentum[i] = updated[j]
        for i, k in enumerate(keys):
            buf[k] = momentum[i]

        # anchor + lr * (delta + mu * m). Written as four batched ops rather
        # than one fused add(alpha=) so the arithmetic is the same expression
        # tree as before: float + and * are commutative bitwise, but a fused
        # multiply-add is not, and the DiLoCo eval-loss band (§2 criterion 5)
        # is not worth risking on a rounding difference.
        update = torch._foreach_mul(momentum, mu)
        torch._foreach_add_(update, delta)
        torch._foreach_mul_(update, lr)
        torch._foreach_add_(update, anchor_f32)
        for i, k in enumerate(keys):
            stepped[k] = update[i].to(anchor[k].dtype)

    # Sorted insertion order, matching the previous implementation.
    return {k: stepped[k] for k in all_keys}
