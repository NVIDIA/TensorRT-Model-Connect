# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ConvBERT family plugin for hybrid attention and dynamic convolution."""

from __future__ import annotations

from .config import ModelConfig
from .weights import WeightDict, load_convbert_weights
from ...parallel_config import (
    normalize_parallel_config,
    require_tensorrt_11_for_tensor_parallel,
)


class ConvBertPlugin:
    name = "convbert"
    runtime_strategy = "convbert_encoder_only"

    def matches(self, model_type: str) -> bool:
        return model_type.lower() == "convbert"

    def load_weights(self, model_dir: str, config: ModelConfig) -> WeightDict:
        return load_convbert_weights(model_dir, config)

    def build_engine(
        self,
        config: ModelConfig,
        weights: WeightDict,
        max_cache_length: int,
        *,
        precision: str = "fp32",
        quant_ctx=None,
        verbose: bool = False,
        parallel_config=None,
    ) -> bytes:
        parallel = normalize_parallel_config(parallel_config)
        if parallel.enabled:
            require_tensorrt_11_for_tensor_parallel(
                parallel, feature="ConvBERT tensor-parallel builds"
            )
            if quant_ctx is not None:
                raise ValueError("ConvBERT tensor-parallel builds do not support quantization")
            from .model.model import build_tp_convbert_encoder_engine

            return build_tp_convbert_encoder_engine(
                config,
                weights,
                max_seq_length=max_cache_length,
                verbose=verbose,
                parallel_config=parallel,
            )

        from .model.model import build_convbert_encoder_engine

        return build_convbert_encoder_engine(
            config,
            weights,
            max_seq_length=max_cache_length,
            precision=precision,
            verbose=verbose,
        )


plugin = ConvBertPlugin()
