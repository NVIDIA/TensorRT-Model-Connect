# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OPT plugin: learned positions, biased LayerNorm, and ReLU MLP."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ...parallel_config import normalize_parallel_config
from .model.model import build_standard_decoder_engine
from .model.parallel import build_dual_profile_tp_decoder_engine
from .weights import (
    WeightDict,
    _has_tensor,
    _load_tensor,
    _open_safetensors,
    _transpose_2d,
)


class OPTPlugin:
    name = "opt"
    runtime_strategy = "opt_decoder_kv_cache"
    runtime_capabilities = {"decoder_kv"}

    def matches(self, model_type: str) -> bool:
        return model_type.lower() == "opt"

    def load_weights(self, model_dir: str, config) -> WeightDict:
        readers = _open_safetensors(Path(model_dir))
        hidden = config.hidden_size
        embed_dim = int(config.raw.get("word_embed_proj_dim", hidden))
        weights = WeightDict()

        token_embedding = _load_tensor(readers, "model.decoder.embed_tokens.weight")
        if token_embedding.shape != (config.vocab_size, embed_dim):
            raise ValueError(
                "OPT token embedding shape mismatch: "
                f"expected {(config.vocab_size, embed_dim)}, got {token_embedding.shape}"
            )
        if embed_dim == hidden:
            weights["embedding"] = token_embedding
        else:
            project_in = _load_tensor(readers, "model.decoder.project_in.weight")
            weights["embedding"] = np.ascontiguousarray(token_embedding @ project_in.T)

        position_embedding = _load_tensor(readers, "model.decoder.embed_positions.weight")
        weights["position_embedding"] = np.ascontiguousarray(position_embedding[2:])

        for layer in range(config.num_hidden_layers):
            prefix = f"layer.{layer}"
            hf = f"model.decoder.layers.{layer}"
            for logical, checkpoint in (
                ("input_norm", "self_attn_layer_norm.weight"),
                ("input_norm_beta", "self_attn_layer_norm.bias"),
                ("post_attn_norm", "final_layer_norm.weight"),
                ("post_attn_norm_beta", "final_layer_norm.bias"),
                ("q_bias", "self_attn.q_proj.bias"),
                ("k_bias", "self_attn.k_proj.bias"),
                ("v_bias", "self_attn.v_proj.bias"),
                ("o_bias", "self_attn.out_proj.bias"),
                ("fc1_bias", "fc1.bias"),
                ("fc2_bias", "fc2.bias"),
            ):
                weights[f"{prefix}.{logical}"] = _load_tensor(readers, f"{hf}.{checkpoint}")
            for logical, checkpoint in (
                ("w_q", "self_attn.q_proj.weight"),
                ("w_k", "self_attn.k_proj.weight"),
                ("w_v", "self_attn.v_proj.weight"),
                ("w_o", "self_attn.out_proj.weight"),
                ("w_fc1", "fc1.weight"),
                ("w_fc2", "fc2.weight"),
            ):
                weights[f"{prefix}.{logical}"] = _transpose_2d(
                    _load_tensor(readers, f"{hf}.{checkpoint}"), checkpoint
                )

        final_norm_key = "model.decoder.final_layer_norm.weight"
        if _has_tensor(readers, final_norm_key):
            weights["final_norm"] = _load_tensor(readers, final_norm_key)
            weights["final_norm_beta"] = _load_tensor(
                readers, "model.decoder.final_layer_norm.bias"
            )
        else:
            weights["final_norm"] = np.ones(hidden, dtype=np.float32)
            weights["final_norm_beta"] = np.zeros(hidden, dtype=np.float32)

        lm_head_key = "lm_head.weight"
        lm_head = (
            _load_tensor(readers, lm_head_key)
            if _has_tensor(readers, lm_head_key)
            else token_embedding
        )
        output = lm_head.T
        if embed_dim != hidden:
            project_out = _load_tensor(readers, "model.decoder.project_out.weight")
            output = project_out.T @ output
        weights["w_out"] = np.ascontiguousarray(output, dtype=np.float32)
        weights["_attention_size"] = hidden
        weights["_kv_attention_size"] = hidden
        weights["_mlp_size"] = int(weights["layer.0.w_fc1"].shape[1])
        return weights

    def build_engine(
        self,
        config,
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
            return build_dual_profile_tp_decoder_engine(
                config,
                weights,
                max_cache_length,
                precision=precision,
                quant_ctx=quant_ctx,
                verbose=verbose,
                parallel_config=parallel,
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


plugin = OPTPlugin()
