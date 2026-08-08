"""Async-mesh merge math — pure functions, no model or network access.

Kept free of FastAPI/peft/transformers imports so the §6.6 correctness tests
run on CPU with only torch installed.
"""

from __future__ import annotations

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

    return {k: (acc[k] / n).to(own[k].dtype) for k in keys}
