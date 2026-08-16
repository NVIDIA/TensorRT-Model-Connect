# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fast Foundation Stereo family plugin."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


_CHECKPOINT = Path("weights/23-36-37/model_best_bp2_serialize.pth")
_POST_SECTION = "fast_foundation_stereo_post_engine_plan"


class ModelConfig(Protocol):
    """The small builder-config surface consumed by this family."""

    raw: dict


class WeightDict(dict):
    """Model-owned weight payload passed between the generic builder hooks."""


class FastFoundationStereoPlugin:
    name = "fast_foundation_stereo"
    runtime_strategy = "fast_foundation_stereo_disparity"
    requires_tokenizer = False

    def matches(self, model_type: str) -> bool:
        return (model_type or "").lower().replace("-", "_") in {
            "fast_foundation_stereo",
            "foundation_stereo_lite",
        }

    def load_weights(
        self,
        model_dir: str,
        config: ModelConfig,
        *,
        precision: str = "fp32",
    ) -> WeightDict:
        del precision
        model_path = Path(model_dir).resolve()
        checkpoint = model_path / _CHECKPOINT
        source = model_path / "core/foundation_stereo.py"
        if not checkpoint.is_file() or not source.is_file():
            raise FileNotFoundError(
                "Fast Foundation Stereo requires core/foundation_stereo.py and "
                f"{_CHECKPOINT.as_posix()} under {model_path}"
            )
        config.raw["_fast_foundation_stereo_model_dir"] = str(model_path)
        return WeightDict({"_fast_foundation_stereo_model_dir": str(model_path)})

    @staticmethod
    def _model_dir(config: ModelConfig, weights: WeightDict) -> str:
        model_dir = weights.get("_fast_foundation_stereo_model_dir") or config.raw.get(
            "_fast_foundation_stereo_model_dir"
        )
        if not model_dir:
            raise RuntimeError("Fast Foundation Stereo model directory was not loaded")
        return str(model_dir)

    def build_engine(
        self,
        config: ModelConfig,
        weights: WeightDict,
        max_cache_length: int,
        *,
        precision: str = "fp16",
        quant_ctx=None,
        verbose: bool = False,
    ) -> bytes:
        del max_cache_length, quant_ctx
        from .builder import build_feature_engine

        return build_feature_engine(
            self._model_dir(config, weights),
            precision=precision,
            max_disparity=int(config.raw.get("stereo_max_disparity", 192)),
            valid_iters=int(config.raw.get("stereo_valid_iters", 8)),
            verbose=verbose,
        )

    def build_extra_engines(
        self,
        config: ModelConfig,
        weights: WeightDict,
        max_cache_length: int,
        *,
        precision: str = "fp16",
        quant_ctx=None,
        verbose: bool = False,
    ) -> dict[str, bytes]:
        del max_cache_length, quant_ctx
        from .builder import build_post_engine

        return {
            _POST_SECTION: build_post_engine(
                self._model_dir(config, weights),
                precision=precision,
                max_disparity=int(config.raw.get("stereo_max_disparity", 192)),
                valid_iters=int(config.raw.get("stereo_valid_iters", 8)),
                verbose=verbose,
            )
        }


plugin = FastFoundationStereoPlugin()
