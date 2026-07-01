# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BERT encoder-only family plugin."""

from __future__ import annotations

from ...parallel_config import (
    normalize_parallel_config,
    require_tensorrt_11_for_tensor_parallel,
)
from .config import ModelConfig
from .model.model import build_encoder_engine
from .weights import WeightDict, load_bert_weights


class BertPlugin:
    name = "bert"
    runtime_strategy = "bert_encoder_only"

    def matches(self, model_type: str) -> bool:
        return model_type.lower() == "bert"

    def load_weights(self, model_dir: str, config: ModelConfig) -> WeightDict:
        return load_bert_weights(model_dir, config)

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
            require_tensorrt_11_for_tensor_parallel(parallel, feature="BERT tensor-parallel builds")
            if quant_ctx is not None:
                raise ValueError("BERT tensor-parallel builds do not support quantization")
            from .model.model import build_tp_encoder_engine

            return build_tp_encoder_engine(
                config,
                weights,
                max_seq_length=max_cache_length,
                verbose=verbose,
                parallel_config=parallel,
            )

        return build_encoder_engine(
            config,
            weights,
            max_seq_length=max_cache_length,
            verbose=verbose,
        )


plugin = BertPlugin()
