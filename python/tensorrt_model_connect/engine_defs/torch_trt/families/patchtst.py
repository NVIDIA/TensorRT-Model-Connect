"""PatchTST family plugin for Torch-TRT.

This family targets numeric time-series models from Hugging Face:
  - forecasting via PatchTSTForPrediction
  - classification via PatchTSTForClassification
  - regression via PatchTSTForRegression

It exposes the same numeric input convention as the build strategy:
`past_values` plus an optional `past_observed_mask`.
"""

from __future__ import annotations

import torch

from ..config import ModelConfig
from ..strategies.patchtst import PatchTSTBuildStrategy, _normalize_task_type


class PatchTSTTorchTrtPlugin:
    name = "patchtst"
    runtime_strategy = "patchtst"

    def matches(self, model_type: str) -> bool:
        mt = model_type.lower()
        return mt == "patchtst" or mt.startswith("patchtst")

    def load_model(
        self,
        model_dir: str,
        config: ModelConfig,
        max_cache_length: int,
        *,
        dtype: torch.dtype | None = None,
    ) -> torch.nn.Module:
        from transformers import (
            PatchTSTForClassification,
            PatchTSTForPrediction,
            PatchTSTForRegression,
        )

        task_type = _normalize_task_type(config.raw)
        model_cls = {
            "classification": PatchTSTForClassification,
            "regression": PatchTSTForRegression,
            "forecast": PatchTSTForPrediction,
        }[task_type]

        if dtype is None:
            dtype = torch.float16

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
        strategy = PatchTSTBuildStrategy()
        return strategy.make_export_args(
            model.config, max_cache_length, precision=precision)


plugin = PatchTSTTorchTrtPlugin()

