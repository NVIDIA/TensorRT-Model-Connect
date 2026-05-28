"""PatchTSMixer family plugin for Torch-TRT numeric time-series models."""

from __future__ import annotations

import torch

from ..config import ModelConfig
from ..strategies.patchtsmixer import (
    PatchTSMixerBuildStrategy,
    infer_patchtsmixer_task_kind,
    resolve_patchtsmixer_model_class,
)


class PatchTSMixerTorchTrtPlugin:
    name = "patchtsmixer"
    runtime_strategy = "patchtsmixer"

    def matches(self, model_type: str) -> bool:
        mt = model_type.lower()
        return "patchtsmixer" in mt or "patch_tsmixer" in mt

    def load_model(
        self,
        model_dir: str,
        config: ModelConfig,
        max_cache_length: int,
        *,
        dtype: torch.dtype | None = None,
    ) -> torch.nn.Module:
        if dtype is None:
            dtype = torch.float16

        task_kind = infer_patchtsmixer_task_kind(config)
        model_cls = resolve_patchtsmixer_model_class(task_kind)
        # transformers 5.2.0 PatchTSMixer classes miss the tied-weights
        # bookkeeping attribute expected by from_pretrained().
        if not hasattr(model_cls, "all_tied_weights_keys"):
            model_cls.all_tied_weights_keys = {}
        model = model_cls.from_pretrained(
            model_dir,
            dtype=dtype,
            device_map="cuda",
        )
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
        strategy = PatchTSMixerBuildStrategy()
        return strategy.make_export_args(
            model.config, max_cache_length, precision=precision)


plugin = PatchTSMixerTorchTrtPlugin()
