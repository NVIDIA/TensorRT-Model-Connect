"""Granite family plugin — absorbs Granite-specific multipliers into weights.

Granite models (IBM) use the standard LLaMA-style decoder pattern but with
four extra scaling factors that differ from vanilla LLaMA:

  - embedding_multiplier:  scales embedding output (default 1.0)
  - attention_multiplier:  replaces 1/sqrt(head_dim) attention scaling
  - residual_multiplier:   scales attention and MLP outputs before residual add
  - logits_scaling:        divides final logits (default 1.0)

All four are absorbed into the weight tensors at load time so the standard
decoder builder can be reused without modification.
"""

from __future__ import annotations

import math

import numpy as np

from .config import ModelConfig
from .checkpoint_mapper import WeightDict, load_standard_weights
from ...parallel_config import normalize_parallel_config
from .dual_profile_decoder_tp_builder import build_dual_profile_tp_decoder_engine
from .standard_decoder_builder import build_standard_decoder_engine


class GranitePlugin:
    name = "granite"
    runtime_strategy = "granite_decoder_kv_cache"
    runtime_capabilities = {"decoder_kv"}

    def matches(self, model_type: str) -> bool:
        return model_type.lower().startswith("granite")

    def load_weights(
        self, model_dir: str, config: ModelConfig,
        *, precision: str = "fp32",
    ) -> WeightDict:
        weights = load_standard_weights(model_dir, config, precision=precision)

        raw = config.raw
        embedding_multiplier = raw.get("embedding_multiplier", 1.0)
        attention_multiplier = raw.get("attention_multiplier", None)
        residual_multiplier = raw.get("residual_multiplier", 1.0)
        logits_scaling = raw.get("logits_scaling", 1.0)

        head_dim = config.head_dim
        standard_attn_scale = 1.0 / math.sqrt(max(head_dim, 1))

        # Fix 1: Granite scales embedding output by embedding_multiplier.
        if embedding_multiplier != 1.0:
            weights["embedding"] = (
                weights["embedding"].astype(np.float32) * embedding_multiplier
            )

        # Fix 2: Granite uses attention_multiplier instead of 1/sqrt(head_dim).
        # Absorb the ratio into Q projection weights so the standard builder's
        # 1/sqrt(head_dim) scaling produces the correct result.
        if attention_multiplier is not None and attention_multiplier != standard_attn_scale:
            q_scale = attention_multiplier / standard_attn_scale
            for layer_idx in range(config.num_hidden_layers):
                key = f"layer.{layer_idx}.w_q"
                weights[key] = weights[key].astype(np.float32) * q_scale

        # Fix 3: Granite multiplies attention and MLP outputs by
        # residual_multiplier before the residual add:
        #   hidden = residual + attn_out * residual_multiplier
        #   hidden = residual + mlp_out * residual_multiplier
        # Absorb into the output projections (w_o and w_down).
        if residual_multiplier != 1.0:
            for layer_idx in range(config.num_hidden_layers):
                o_key = f"layer.{layer_idx}.w_o"
                d_key = f"layer.{layer_idx}.w_down"
                weights[o_key] = weights[o_key].astype(np.float32) * residual_multiplier
                weights[d_key] = weights[d_key].astype(np.float32) * residual_multiplier

        # Fix 4: Granite divides final logits by logits_scaling.
        # Absorb into the output (lm_head) weight matrix.
        if logits_scaling != 1.0:
            weights["w_out"] = (
                weights["w_out"].astype(np.float32) / logits_scaling
            )

        return weights

    def build_engine(
        self, config: ModelConfig, weights: WeightDict,
        max_cache_length: int, *, precision: str = "fp32",
        quant_ctx=None, verbose: bool = False,
        debug_layer_outputs: bool = False,
        parallel_config=None,
    ) -> bytes:
        parallel = normalize_parallel_config(parallel_config)
        if parallel.enabled:
            return build_dual_profile_tp_decoder_engine(
                config, weights, max_cache_length,
                precision=precision,
                quant_ctx=quant_ctx,
                verbose=verbose,
                parallel_config=parallel)

        return build_standard_decoder_engine(
            config, weights, max_cache_length, precision=precision,
            quant_ctx=quant_ctx, verbose=verbose,
            debug_layer_outputs=debug_layer_outputs)


plugin = GranitePlugin()
