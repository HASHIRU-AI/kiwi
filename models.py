"""Kiwi model configuration and registry."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field

from transformers import AutoConfig


@dataclass
class ModelConfig:
    """Describes a servable model and its layer topology."""

    model_id: str
    num_layers: int
    default_dtype: str = "bfloat16"
    aliases: list[str] = field(default_factory=list)
    _hidden_size: int = 1024
    _intermediate_size: int = 4096

    @property
    def layer_size_mb(self) -> float:
        """Estimate the memory footprint (MB) of a single decoder layer in default_dtype.

        Uses standard transformer layer size math with SwiGLU MLP:
          - Attention: Q, K, V, O projections = 4 * hidden * hidden
          - MLP: Gate, Up, Down projections = 3 * hidden * intermediate
        """
        attn_params = 4 * self._hidden_size * self._hidden_size
        mlp_params = 3 * self._hidden_size * self._intermediate_size
        params = attn_params + mlp_params

        # 2 bytes per param for bfloat16/float16, 4 bytes for float32
        bytes_per_param = 4 if self.default_dtype == "float32" else 2
        
        # Add 30% overhead margin for activations / context
        return ((params * bytes_per_param) * 1.3) / (1024 ** 2)


class ModelRegistry:
    """Holds every model the cluster can serve."""

    def __init__(self) -> None:
        self._models: dict[str, ModelConfig] = {}  # keyed by model_id
        self._aliases: dict[str, str] = {}  # lower-cased alias -> model_id
        self._default_id: str | None = None

    # ── Mutation ─────────────────────────────────────────────────────────────

    def register(self, config: ModelConfig) -> None:
        self._models[config.model_id] = config
        for alias in config.aliases:
            self._aliases[alias.lower()] = config.model_id
        if self._default_id is None:
            self._default_id = config.model_id

    # ── Lookup ───────────────────────────────────────────────────────────────

    def resolve(self, name: str | None) -> ModelConfig | None:
        """Resolve by model_id, alias (case-insensitive), or return default."""
        if name is None:
            return self.default()
        if name in self._models:
            return self._models[name]
        model_id = self._aliases.get(name.lower())
        return self._models.get(model_id) if model_id else None

    def default(self) -> ModelConfig | None:
        return self._models.get(self._default_id) if self._default_id else None

    def list_models(self) -> list[ModelConfig]:
        return list(self._models.values())

    def __len__(self) -> int:
        return len(self._models)


async def load_model_registry() -> ModelRegistry:
    """Build the registry from ``KIWI_MODELS`` (JSON array) or ``MODEL_ID``.

    ``KIWI_MODELS`` example::

        [
          {"model_id": "Qwen/Qwen2.5-0.5B-Instruct",
           "aliases": ["qwen-0.5b"], "default_dtype": "bfloat16"},
          {"model_id": "Qwen/Qwen2.5-1.5B-Instruct",
           "aliases": ["qwen-1.5b"], "default_dtype": "bfloat16"}
        ]

    When ``KIWI_MODELS`` is absent, a single model is registered from the
    legacy ``MODEL_ID`` environment variable (backward-compatible).
    """
    registry = ModelRegistry()

    models_json = os.getenv("KIWI_MODELS")
    if models_json:
        for spec in json.loads(models_json):
            model_id = spec["model_id"]
            cfg = await asyncio.to_thread(AutoConfig.from_pretrained, model_id)
            registry.register(
                ModelConfig(
                    model_id=model_id,
                    num_layers=cfg.num_hidden_layers,
                    default_dtype=spec.get("default_dtype", "bfloat16"),
                    aliases=spec.get("aliases", []),
                    _hidden_size=getattr(cfg, "hidden_size", 1024),
                    _intermediate_size=getattr(cfg, "intermediate_size", 4 * getattr(cfg, "hidden_size", 1024)),
                )
            )
    else:
        # Backward compat: single MODEL_ID env var.
        model_id = os.getenv("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
        cfg = await asyncio.to_thread(AutoConfig.from_pretrained, model_id)
        registry.register(
            ModelConfig(
                model_id=model_id,
                num_layers=cfg.num_hidden_layers,
                default_dtype=os.getenv("MODEL_DTYPE", "bfloat16"),
                aliases=[],
                _hidden_size=getattr(cfg, "hidden_size", 1024),
                _intermediate_size=getattr(cfg, "intermediate_size", 4 * getattr(cfg, "hidden_size", 1024)),
            )
        )

    return registry
