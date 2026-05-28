"""Chronos-Bolt family plugin for Torch-TRT.

The official Chronos-Bolt checkpoints use `model_type="t5"`, so family
dispatch cannot rely on `model_type` alone. This plugin detects the model
via `architectures` / `chronos_config` and loads the official
`ChronosBoltModelForForecasting` implementation.
"""

from __future__ import annotations

import torch

from ..config import ModelConfig


class ChronosBoltTorchTrtPlugin:
    name = "chronos_bolt"
    runtime_strategy = "chronos_bolt"

    def matches(self, model_type: str) -> bool:
        mt = model_type.lower()
        return mt in {
            "chronos_bolt",
            "chronos-bolt",
            "chronosbolt",
        } or mt.startswith("chronos_bolt") or mt.startswith("chronos-bolt") or mt.startswith("chronosbolt")

    def matches_config(self, config: ModelConfig | str) -> bool:
        if isinstance(config, str):
            return self.matches(config)
        raw = getattr(config, "raw", {}) or {}
        if not isinstance(raw, dict):
            return False
        chronos_cfg = raw.get("chronos_config")
        if not isinstance(chronos_cfg, dict):
            return False
        architectures = raw.get("architectures") or []
        if isinstance(architectures, str):
            architectures = [architectures]
        if any("ChronosBoltModelForForecasting" in str(arch) for arch in architectures):
            return True
        # Fall back to the presence of chronos_config on a T5-style config.
        return str(getattr(config, "model_type", "")).lower() == "t5"

    def load_model(
        self,
        model_dir: str,
        config: ModelConfig,
        max_cache_length: int,
        *,
        dtype: torch.dtype | None = None,
    ) -> torch.nn.Module:
        from chronos import ChronosBoltPipeline

        if dtype is None:
            dtype = torch.float16

        del config, max_cache_length
        model = ChronosBoltPipeline.from_pretrained(
            model_dir,
            device_map="cuda",
            dtype=dtype,
        ).model
        # Disable decoder cache for one-shot forecasting export. This keeps the
        # compiled graph aligned with the official non-autoregressive forward.
        if hasattr(model, "config"):
            model.config.use_cache = False
        if hasattr(model, "decoder") and hasattr(model.decoder, "config"):
            model.decoder.config.use_cache = False
        model.eval()
        return model

    def get_export_args(
        self,
        model: torch.nn.Module,
        config: ModelConfig,
        max_cache_length: int,
        *,
        precision: str = "fp16",
    ) -> tuple:
        from ..strategies.chronos_bolt import ChronosBoltBuildStrategy

        strategy = ChronosBoltBuildStrategy()
        return strategy.make_export_args(
            model.config, max_cache_length, precision=precision)


plugin = ChronosBoltTorchTrtPlugin()
