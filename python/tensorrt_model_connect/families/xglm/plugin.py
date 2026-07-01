# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""XGLM family plugin."""

from __future__ import annotations

from ...parallel_config import (
    normalize_parallel_config,
    require_tensorrt_11_for_tensor_parallel,
)
from .config import ModelConfig
from .model.model import build_standard_decoder_engine
from .model.parallel import build_dual_profile_tp_decoder_engine
from .weights import WeightDict, load_standard_weights


class XGLMPlugin:
    name = "xglm"
    runtime_strategy = "xglm_decoder_kv_cache"
    runtime_capabilities = {"decoder_kv"}

    def matches(self, model_type: str) -> bool:
        return model_type.lower() == "xglm"

    def load_weights(self, model_dir: str, config: ModelConfig) -> WeightDict:
        return load_standard_weights(model_dir, config)

    def build_engine(
        self,
        config: ModelConfig,
        weights: WeightDict,
        max_cache_length: int,
        *,
        precision: str = "fp32",
        quant_ctx=None,
        verbose: bool = False,
        debug_layer_outputs: bool = False,
        parallel_config=None,
    ) -> bytes:
        parallel = normalize_parallel_config(parallel_config)
        if parallel.enabled:
            require_tensorrt_11_for_tensor_parallel(parallel, feature="XGLM tensor-parallel builds")
            if quant_ctx is not None:
                raise ValueError("XGLM tensor-parallel builds do not support quantization")
            if debug_layer_outputs:
                raise ValueError("XGLM tensor-parallel builds do not support debug_layer_outputs")
            return build_dual_profile_tp_decoder_engine(
                config,
                weights,
                max_cache_length,
                precision=precision,
                quant_ctx=quant_ctx,
                verbose=verbose,
                parallel_config=parallel,
                activation="gelu",
                mlp_type="gelu_fc",
                norm_type="layernorm",
                position_type="learned",
            )
        return build_standard_decoder_engine(
            config,
            weights,
            max_cache_length,
            precision=precision,
            quant_ctx=quant_ctx,
            verbose=verbose,
            debug_layer_outputs=debug_layer_outputs,
        )


plugin = XGLMPlugin()
