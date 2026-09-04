# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V2 family plugin — Multi-head Latent Attention + Mixture of Experts.

DeepSeek-V2 uses Multi-head Latent Attention (MLA) which compresses KV cache
via latent projections, plus Mixture of Experts with shared experts. This
plugin implements the "naive" MLA approach: decompress K/V fully, then cache
the decompressed values using the standard KV cache runtime.

Key architecture details:
  - MLA compresses KV into a latent space (kv_lora_rank) then decompresses
  - Partial RoPE: only qk_rope_head_dim dimensions get RoPE
  - K and V have different per-head sizes (K: nope+rope=192, V: v_head_dim=128)
  - V is zero-padded to match K size for uniform cache_state_size
  - First first_k_dense_replace layers use dense SwiGLU MLP
  - Remaining layers use MoE with shared experts that are always active
  - Standard top-k softmax routing with renormalization for routed experts

V2-Lite specifics:
  - q_lora_rank is null: Q is a direct projection, no LoRA compression
  - num_attention_heads == num_key_value_heads == 16 (after decompression)

Weight key mapping:
  HF: model.layers.{i}.self_attn.q_proj.weight     -> w_q [num_heads*(nope+rope), hidden]
  HF: model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight -> w_kv_a [kv_lora_rank+rope, hidden]
  HF: model.layers.{i}.self_attn.kv_a_layernorm.weight     -> kv_a_norm [kv_lora_rank]
  HF: model.layers.{i}.self_attn.kv_b_proj.weight          -> w_kv_b [num_heads*(nope+v), kv_lora_rank]
  HF: model.layers.{i}.self_attn.o_proj.weight              -> w_o [hidden, num_heads*v_head_dim]
  HF: model.layers.{i}.mlp.gate.weight -> moe_gate [n_routed_experts, hidden]
  HF: model.layers.{i}.mlp.experts.{e}.gate_proj.weight -> expert gate
  HF: model.layers.{i}.mlp.experts.{e}.up_proj.weight   -> expert up
  HF: model.layers.{i}.mlp.experts.{e}.down_proj.weight -> expert down
  HF: model.layers.{i}.mlp.shared_experts.gate_proj.weight -> shared gate
  HF: model.layers.{i}.mlp.shared_experts.up_proj.weight   -> shared up
  HF: model.layers.{i}.mlp.shared_experts.down_proj.weight -> shared down
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import sys
from pathlib import Path

import numpy as np
import tensorrt as trt

from .config import ModelConfig
from .parallel import ParallelConfig
from .checkpoint_mapper import (
    WeightDict,
    _open_safetensors,
    _load_tensor,
    _has_tensor,
    _transpose_2d,
)
from . import graph_ops
from . import moe_routing
from . import graph_blocks
from .parallel import normalize_parallel_config
from .default_decoder import _apply_norm, _mark_debug_output


def _validate_router_score_bias(
    value: np.ndarray,
    checkpoint_key: str,
) -> np.ndarray:
    bias = np.asarray(value, dtype=np.float32)
    if not np.isfinite(bias).all():
        raise ValueError(f"DeepSeek router score bias contains non-finite values: {checkpoint_key}")
    return bias


def _use_fp32_mla_attention(dtype: np.dtype, head_dim: int) -> bool:
    """Use FP32 attention only for TensorRT-supported MLA head dimensions.

    The tiny synthetic DeepSeek-V3 contract model uses a four-element head.
    TensorRT 11.2 cannot build FP32 IAttention for that sub-warp shape, while
    production DeepSeek MLA heads are at least 16 elements wide.
    """
    return dtype == np.float16 and head_dim >= 16


if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


class _DeepSeekV2Model:
    def get_bundle_config_overrides(self, config: ModelConfig) -> dict:
        """Inject head_dim into bundle config.json for C++ runtime.

        The C++ fast_path_config parser computes:
            head_dim = config["head_dim"] or (hidden_size / num_attention_heads)
            attention_size = num_heads * head_dim

        For DeepSeek-V2/MLA, the effective head_dim for K cache is
        qk_nope_head_dim + qk_rope_head_dim (e.g. 128 + 64 = 192), not the
        default hidden_size / num_heads (e.g. 2048 / 16 = 128). We inject
        the correct head_dim so the C++ runtime allocates the right cache.
        """
        raw = config.raw
        qk_nope = raw.get("qk_nope_head_dim", 128)
        qk_rope = raw.get("qk_rope_head_dim", 64)
        return {"head_dim": qk_nope + qk_rope}

    def load_weights(
        self,
        model_dir: str,
        config: ModelConfig,
    ) -> WeightDict:
        model_dir_path = Path(model_dir)
        readers = _open_safetensors(model_dir_path)

        hidden = config.hidden_size
        vocab = config.vocab_size
        num_layers = config.num_hidden_layers
        num_heads = config.num_attention_heads

        # MLA-specific dimensions from config
        raw = config.raw
        qk_nope_head_dim = raw.get("qk_nope_head_dim", 128)
        qk_rope_head_dim = raw.get("qk_rope_head_dim", 64)
        v_head_dim = raw.get("v_head_dim", 128)
        kv_lora_rank = raw.get("kv_lora_rank", 512)
        q_lora_rank = raw.get("q_lora_rank", None)  # None for V2-Lite

        # MoE config
        n_routed_experts = raw.get("n_routed_experts", 64)
        n_shared_experts = raw.get("n_shared_experts", 2)
        num_experts_per_tok = raw.get("num_experts_per_tok", 6)
        first_k_dense_replace = raw.get("first_k_dense_replace", 1)
        moe_layer_freq = raw.get("moe_layer_freq", 1)
        intermediate_size = config.intermediate_size

        # Shared expert intermediate size: proportional to n_shared_experts
        moe_intermediate_size = raw.get("moe_intermediate_size", intermediate_size)
        shared_intermediate = moe_intermediate_size * n_shared_experts

        # K has nope + rope dims per head; V has v_head_dim per head.
        # For uniform cache, we pad V to match K size.
        k_head_dim = qk_nope_head_dim + qk_rope_head_dim  # 192
        attention_size = num_heads * k_head_dim  # cache_state_size for both K and V

        weights = WeightDict()

        # Embedding
        embedding = _load_tensor(readers, "model.embed_tokens.weight")
        assert embedding.shape == (vocab, hidden), (
            f"Embedding shape {embedding.shape} != ({vocab}, {hidden})"
        )
        weights["embedding"] = embedding.astype(np.float32)

        for layer_idx in range(num_layers):
            prefix = f"layer.{layer_idx}"
            hf_prefix = f"model.layers.{layer_idx}"

            # RMSNorm weights
            input_norm = _load_tensor(readers, f"{hf_prefix}.input_layernorm.weight")
            weights[f"{prefix}.input_norm"] = input_norm.astype(np.float32)

            post_norm = _load_tensor(readers, f"{hf_prefix}.post_attention_layernorm.weight")
            weights[f"{prefix}.post_attn_norm"] = post_norm.astype(np.float32)

            # --- MLA attention weights ---

            # Q projection: direct for V2-Lite (q_lora_rank is None)
            # Shape: [num_heads * (qk_nope_head_dim + qk_rope_head_dim), hidden]
            if q_lora_rank is not None and q_lora_rank > 0:
                # V2 full: Q goes through LoRA compression
                q_a_raw = _load_tensor(readers, f"{hf_prefix}.self_attn.q_a_proj.weight")
                weights[f"{prefix}.w_q_a"] = _transpose_2d(q_a_raw, "q_a_proj")
                del q_a_raw

                q_a_norm = _load_tensor(readers, f"{hf_prefix}.self_attn.q_a_layernorm.weight")
                weights[f"{prefix}.q_a_norm"] = q_a_norm.astype(np.float32)

                q_b_raw = _load_tensor(readers, f"{hf_prefix}.self_attn.q_b_proj.weight")
                weights[f"{prefix}.w_q_b"] = _transpose_2d(q_b_raw, "q_b_proj")
                del q_b_raw
            else:
                # V2-Lite: direct Q projection
                q_raw = _load_tensor(readers, f"{hf_prefix}.self_attn.q_proj.weight")
                weights[f"{prefix}.w_q"] = _transpose_2d(q_raw, "q_proj")
                del q_raw

            # KV-A projection with MQA (kv_lora_rank + qk_rope_head_dim, hidden)
            kv_a_raw = _load_tensor(readers, f"{hf_prefix}.self_attn.kv_a_proj_with_mqa.weight")
            weights[f"{prefix}.w_kv_a"] = _transpose_2d(kv_a_raw, "kv_a_proj")
            del kv_a_raw

            # KV-A LayerNorm on the latent (kv_lora_rank dims)
            kv_a_norm = _load_tensor(readers, f"{hf_prefix}.self_attn.kv_a_layernorm.weight")
            weights[f"{prefix}.kv_a_norm"] = kv_a_norm.astype(np.float32)

            # KV-B projection: decompresses latent to per-head K_nope and V
            # Shape: [num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank]
            kv_b_raw = _load_tensor(readers, f"{hf_prefix}.self_attn.kv_b_proj.weight")
            weights[f"{prefix}.w_kv_b"] = _transpose_2d(kv_b_raw, "kv_b_proj")
            del kv_b_raw

            # Output projection: [hidden, num_heads * v_head_dim]
            o_raw = _load_tensor(readers, f"{hf_prefix}.self_attn.o_proj.weight")
            weights[f"{prefix}.w_o"] = _transpose_2d(o_raw, "o_proj")
            del o_raw

            # --- MLP weights (dense or MoE depending on layer) ---
            is_moe_layer = (
                layer_idx >= first_k_dense_replace
                and (layer_idx - first_k_dense_replace) % moe_layer_freq == 0
            )

            if is_moe_layer:
                # Router weight
                router_raw = _load_tensor(readers, f"{hf_prefix}.mlp.gate.weight")
                weights[f"{prefix}.router"] = _transpose_2d(router_raw, "router")
                del router_raw
                correction_bias_key = f"{hf_prefix}.mlp.gate.e_score_correction_bias"
                if _has_tensor(readers, correction_bias_key):
                    weights[f"{prefix}.router_score_bias"] = _validate_router_score_bias(
                        _load_tensor(readers, correction_bias_key),
                        correction_bias_key,
                    )

                # Per-expert weights
                for e in range(n_routed_experts):
                    exp_hf = f"{hf_prefix}.mlp.experts.{e}"
                    gate_raw = _load_tensor(readers, f"{exp_hf}.gate_proj.weight")
                    up_raw = _load_tensor(readers, f"{exp_hf}.up_proj.weight")
                    down_raw = _load_tensor(readers, f"{exp_hf}.down_proj.weight")

                    weights[f"{prefix}.expert.{e}.w_gate"] = _transpose_2d(
                        gate_raw, f"expert_{e}_gate"
                    )
                    weights[f"{prefix}.expert.{e}.w_up"] = _transpose_2d(up_raw, f"expert_{e}_up")
                    weights[f"{prefix}.expert.{e}.w_down"] = _transpose_2d(
                        down_raw, f"expert_{e}_down"
                    )
                    del gate_raw, up_raw, down_raw

                # Shared expert weights (always active)
                shared_hf = f"{hf_prefix}.mlp.shared_experts"
                s_gate_raw = _load_tensor(readers, f"{shared_hf}.gate_proj.weight")
                s_up_raw = _load_tensor(readers, f"{shared_hf}.up_proj.weight")
                s_down_raw = _load_tensor(readers, f"{shared_hf}.down_proj.weight")

                weights[f"{prefix}.shared.w_gate"] = _transpose_2d(s_gate_raw, "shared_gate")
                weights[f"{prefix}.shared.w_up"] = _transpose_2d(s_up_raw, "shared_up")
                weights[f"{prefix}.shared.w_down"] = _transpose_2d(s_down_raw, "shared_down")
                del s_gate_raw, s_up_raw, s_down_raw
            else:
                # Dense MLP
                gate_raw = _load_tensor(readers, f"{hf_prefix}.mlp.gate_proj.weight")
                up_raw = _load_tensor(readers, f"{hf_prefix}.mlp.up_proj.weight")
                down_raw = _load_tensor(readers, f"{hf_prefix}.mlp.down_proj.weight")

                weights[f"{prefix}.w_gate"] = _transpose_2d(gate_raw, "gate_proj")
                weights[f"{prefix}.w_up"] = _transpose_2d(up_raw, "up_proj")
                weights[f"{prefix}.w_down"] = _transpose_2d(down_raw, "down_proj")
                del gate_raw, up_raw, down_raw

        # Final norm
        final_norm_key = "model.norm.weight"
        if _has_tensor(readers, final_norm_key):
            weights["final_norm"] = _load_tensor(readers, final_norm_key).astype(np.float32)
        else:
            weights["final_norm"] = np.ones(hidden, dtype=np.float32)

        # LM head
        lm_head_key = "lm_head.weight"
        if _has_tensor(readers, lm_head_key):
            weights["w_out"] = _transpose_2d(_load_tensor(readers, lm_head_key), "lm_head")
        else:
            weights["w_out"] = _transpose_2d(embedding.copy(), "embedding_tied")

        # Store metadata for engine builder
        weights["_attention_size"] = attention_size  # type: ignore[assignment]
        weights["_qk_nope_head_dim"] = qk_nope_head_dim  # type: ignore[assignment]
        weights["_qk_rope_head_dim"] = qk_rope_head_dim  # type: ignore[assignment]
        weights["_v_head_dim"] = v_head_dim  # type: ignore[assignment]
        weights["_kv_lora_rank"] = kv_lora_rank  # type: ignore[assignment]
        weights["_q_lora_rank"] = q_lora_rank  # type: ignore[assignment]
        weights["_n_routed_experts"] = n_routed_experts  # type: ignore[assignment]
        weights["_n_shared_experts"] = n_shared_experts  # type: ignore[assignment]
        weights["_num_experts_per_tok"] = num_experts_per_tok  # type: ignore[assignment]
        weights["_first_k_dense_replace"] = first_k_dense_replace  # type: ignore[assignment]
        weights["_moe_layer_freq"] = moe_layer_freq  # type: ignore[assignment]
        weights["_moe_intermediate_size"] = moe_intermediate_size  # type: ignore[assignment]
        weights["_shared_intermediate_size"] = shared_intermediate  # type: ignore[assignment]
        weights["_norm_topk_prob"] = raw.get("norm_topk_prob", False)  # type: ignore[assignment]
        weights["_routed_scaling_factor"] = raw.get("routed_scaling_factor", 1.0)  # type: ignore[assignment]
        weights["_scoring_func"] = raw.get("scoring_func", "softmax")  # type: ignore[assignment]
        weights["_topk_method"] = raw.get("topk_method", "greedy")  # type: ignore[assignment]
        weights["_n_group"] = raw.get("n_group", 1)  # type: ignore[assignment]
        weights["_topk_group"] = raw.get("topk_group", 1)  # type: ignore[assignment]

        return weights

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
            parallel.validate()
            from .tp_builder import build_deepseek_v2_tp_engine

            return build_deepseek_v2_tp_engine(
                config,
                weights,
                max_cache_length,
                precision=precision,
                quant_ctx=quant_ctx,
                verbose=verbose,
                debug_layer_outputs=debug_layer_outputs,
                parallel_config=parallel,
            )

        hidden = config.hidden_size
        vocab = config.vocab_size
        num_layers = config.num_hidden_layers
        num_heads = config.num_attention_heads

        # MLA dimensions
        qk_nope_head_dim: int = weights["_qk_nope_head_dim"]
        qk_rope_head_dim: int = weights["_qk_rope_head_dim"]
        v_head_dim: int = weights["_v_head_dim"]
        kv_lora_rank: int = weights["_kv_lora_rank"]
        q_lora_rank = weights["_q_lora_rank"]

        # MoE dimensions
        n_routed_experts: int = weights["_n_routed_experts"]
        n_shared_experts: int = weights["_n_shared_experts"]
        num_experts_per_tok: int = weights["_num_experts_per_tok"]
        first_k_dense_replace: int = weights["_first_k_dense_replace"]
        moe_layer_freq: int = weights["_moe_layer_freq"]
        moe_intermediate: int = weights["_moe_intermediate_size"]
        shared_intermediate: int = weights["_shared_intermediate_size"]
        norm_topk_prob: bool = weights["_norm_topk_prob"]
        routed_scaling_factor: float = weights["_routed_scaling_factor"]
        scoring_func = str(weights["_scoring_func"])
        topk_method = str(weights["_topk_method"])
        n_group = int(weights["_n_group"])
        topk_group = int(weights["_topk_group"])
        moe_routing.validate_router_contract(
            scoring_func=scoring_func,
            topk_method=topk_method,
            n_routed_experts=n_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
            n_group=n_group,
            topk_group=topk_group,
        )
        dense_intermediate = config.intermediate_size

        # K head dim = nope + rope; this is the per-head cache dimension
        k_head_dim = qk_nope_head_dim + qk_rope_head_dim  # 192
        attention_size = num_heads * k_head_dim  # uniform cache size
        attention_window = max_cache_length + 1

        # Attention scale: 1 / sqrt(full_head_dim) where full = nope + rope
        # HF uses: self.scaling = self.qk_head_dim ** (-0.5)
        # YaRN mscale is handled via rope_utils attention_factor which scales
        # cos/sin directly. For V2-Lite, mscale == mscale_all_dim so they
        # cancel out (attention_factor = 1.0). No adjustment needed here.
        attn_scale = 1.0 / np.sqrt(max(k_head_dim, 1))

        logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        trt_config = builder.create_builder_config()

        if precision == "fp16":
            work_np_dtype = np.float16
            work_trt_dtype = trt.float16
        elif precision == "bf16":
            work_np_dtype = np.float16
            work_trt_dtype = trt.bfloat16
        else:
            work_np_dtype = np.float32
            work_trt_dtype = trt.float32

        requested_fp32_layers = frozenset(
            int(layer) for layer in config.raw.get("_fp32_layers", ())
        )
        invalid_fp32_layers = sorted(
            layer for layer in requested_fp32_layers if layer < 0 or layer >= num_layers
        )
        if invalid_fp32_layers:
            raise ValueError(f"fp32_layers contains out-of-range indices: {invalid_fp32_layers}")

        # -----------------------------------------------------------
        # Inputs
        # -----------------------------------------------------------
        token_id = network.add_input("token_id", trt.int32, (1,))
        position_id = network.add_input("position_id", trt.int32, (1,))
        attention_mask = network.add_input("attention_mask", trt.float32, (1, attention_window))

        cache_k_inputs = []
        cache_v_inputs = []
        for i in range(num_layers):
            ck = network.add_input(
                graph_ops.layer_tensor_name("cache_k", i),
                work_trt_dtype,
                (max_cache_length, attention_size),
            )
            cv = network.add_input(
                graph_ops.layer_tensor_name("cache_v", i),
                work_trt_dtype,
                (max_cache_length, attention_size),
            )
            cache_k_inputs.append(ck)
            cache_v_inputs.append(cv)

        if work_trt_dtype != trt.float32:
            attention_mask = network.add_cast(attention_mask, work_trt_dtype).get_output(0)

        # -----------------------------------------------------------
        # Shared constants
        # -----------------------------------------------------------
        embedding_table = graph_ops.add_constant(
            network, (vocab, hidden), weights["embedding"], dtype=work_np_dtype
        )

        # DeepSeek-V2 uses complex (interleaved) RoPE: adjacent dims (d, d+1)
        # share a frequency, matching HF's apply_rotary_emb with torch.polar.
        rope_scaling = config.raw.get("rope_scaling")
        if rope_scaling and rope_scaling.get("type") == "yarn":
            yarn_kwargs = dict(
                scaling_factor=rope_scaling["factor"],
                original_max_position_embeddings=rope_scaling["original_max_position_embeddings"],
                beta_fast=rope_scaling["beta_fast"],
                beta_slow=rope_scaling["beta_slow"],
            )
            cos_half_np = graph_ops.make_yarn_rope_table_half_dim(
                attention_window,
                qk_rope_head_dim,
                config.rope_theta,
                True,
                **yarn_kwargs,
                interleaved=True,
            )
            sin_half_np = graph_ops.make_yarn_rope_table_half_dim(
                attention_window,
                qk_rope_head_dim,
                config.rope_theta,
                False,
                **yarn_kwargs,
                interleaved=True,
            )
        else:
            cos_half_np = graph_ops.make_rope_table_half_dim(
                attention_window, qk_rope_head_dim, config.rope_theta, True, interleaved=True
            )
            sin_half_np = graph_ops.make_rope_table_half_dim(
                attention_window, qk_rope_head_dim, config.rope_theta, False, interleaved=True
            )

        cos_half_tensor = graph_ops.add_constant(
            network, cos_half_np.shape, cos_half_np, dtype=work_np_dtype
        )
        sin_half_tensor = graph_ops.add_constant(
            network, sin_half_np.shape, sin_half_np, dtype=work_np_dtype
        )

        eps_tensor = graph_ops.add_constant(
            network,
            (1, 1),
            np.array([config.rms_norm_eps], dtype=work_np_dtype),
            dtype=work_np_dtype,
        )

        # -----------------------------------------------------------
        # Embedding lookup
        # -----------------------------------------------------------
        gather = network.add_gather(embedding_table, token_id, 0)
        hidden_state = gather.get_output(0)

        if debug_layer_outputs:
            _mark_debug_output(network, hidden_state, "debug_embed")

        # -----------------------------------------------------------
        # Decoder layers
        # -----------------------------------------------------------
        present_k_outputs = []
        present_v_outputs = []

        for layer_idx in range(num_layers):
            prefix = f"layer.{layer_idx}"
            layer_is_fp32 = precision == "fp16" and layer_idx in requested_fp32_layers
            layer_np_dtype = np.float32 if layer_is_fp32 else work_np_dtype
            layer_trt_dtype = trt.float32 if layer_is_fp32 else work_trt_dtype

            def layer_cast(tensor):
                if tensor.dtype == layer_trt_dtype:
                    return tensor
                return network.add_cast(tensor, layer_trt_dtype).get_output(0)

            is_moe_layer = (
                layer_idx >= first_k_dense_replace
                and (layer_idx - first_k_dense_replace) % moe_layer_freq == 0
            )

            result = _add_deepseek_v2_decoder_layer(
                network=network,
                hidden=layer_cast(hidden_state),
                cache_k=layer_cast(cache_k_inputs[layer_idx]),
                cache_v=layer_cast(cache_v_inputs[layer_idx]),
                attention_mask=layer_cast(attention_mask),
                position_id=position_id,
                cos_half_tensor=layer_cast(cos_half_tensor),
                sin_half_tensor=layer_cast(sin_half_tensor),
                attn_scale=attn_scale,
                eps_tensor=layer_cast(eps_tensor),
                weights=weights,
                prefix=prefix,
                hidden_size=hidden,
                num_heads=num_heads,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                v_head_dim=v_head_dim,
                kv_lora_rank=kv_lora_rank,
                q_lora_rank=q_lora_rank,
                attention_size=attention_size,
                max_cache_length=max_cache_length,
                is_moe_layer=is_moe_layer,
                n_routed_experts=n_routed_experts,
                n_shared_experts=n_shared_experts,
                num_experts_per_tok=num_experts_per_tok,
                moe_intermediate=moe_intermediate,
                shared_intermediate=shared_intermediate,
                dense_intermediate=dense_intermediate,
                norm_topk_prob=norm_topk_prob,
                routed_scaling_factor=routed_scaling_factor,
                scoring_func=scoring_func,
                topk_method=topk_method,
                n_group=n_group,
                topk_group=topk_group,
                dtype=layer_np_dtype,
            )

            hidden_state = result["hidden"]
            present_k = result["present_k"]
            present_v = result["present_v"]
            if layer_is_fp32:
                hidden_state = network.add_cast(hidden_state, work_trt_dtype).get_output(0)
                present_k = network.add_cast(present_k, work_trt_dtype).get_output(0)
                present_v = network.add_cast(present_v, work_trt_dtype).get_output(0)
            present_k_outputs.append(present_k)
            present_v_outputs.append(present_v)

            if debug_layer_outputs:
                _mark_debug_output(network, result["post_attn"], f"debug_post_attn_{layer_idx}")
                _mark_debug_output(network, hidden_state, f"debug_hidden_{layer_idx}")

        # -----------------------------------------------------------
        # Final norm
        # -----------------------------------------------------------
        final_norm = weights.get("final_norm")
        if final_norm is not None and len(final_norm) > 0:
            hidden_state = _apply_norm(
                network,
                hidden_state,
                hidden,
                final_norm,
                None,
                eps_tensor,
                "rmsnorm",
                dtype=work_np_dtype,
            )

        # -----------------------------------------------------------
        # LM head (logits)
        # -----------------------------------------------------------
        logits = graph_ops.add_matmul_rhs_constant(
            network, hidden_state, hidden, vocab, weights["w_out"], dtype=work_np_dtype
        )
        b_out = np.zeros(vocab, dtype=work_np_dtype)
        logits = graph_ops.add_bias_sum(network, logits, vocab, b_out, dtype=work_np_dtype)

        if work_trt_dtype != trt.float32:
            logits = network.add_cast(logits, trt.float32).get_output(0)

        logits.name = "logits"
        network.mark_output(logits)

        # -----------------------------------------------------------
        # Present K/V outputs
        # -----------------------------------------------------------
        for i in range(num_layers):
            pk = present_k_outputs[i]
            pv = present_v_outputs[i]
            pk.name = graph_ops.layer_tensor_name("present_k", i)
            pv.name = graph_ops.layer_tensor_name("present_v", i)
            network.mark_output(pk)
            network.mark_output(pv)

        # -----------------------------------------------------------
        # Build engine
        # -----------------------------------------------------------
        if verbose:
            print(
                f"[trtmc build] Building DeepSeek-V2 TRT engine "
                f"({num_layers} layers, hidden={hidden}, "
                f"attn={attention_size}, heads={num_heads}, "
                f"kv_lora_rank={kv_lora_rank}, "
                f"nope={qk_nope_head_dim}, rope={qk_rope_head_dim}, "
                f"v_dim={v_head_dim}, "
                f"experts={n_routed_experts}, shared={n_shared_experts}, "
                f"top_k={num_experts_per_tok}, "
                f"cache={max_cache_length}) ...",
                file=sys.stderr,
            )

        plan = builder.build_serialized_network(network, trt_config)
        if plan is None:
            raise RuntimeError("TensorRT engine build failed")

        return bytes(plan)


# ---------------------------------------------------------------------------
# MLA Attention Block
# ---------------------------------------------------------------------------


def _add_mla_attention_block(
    *,
    network: trt.INetworkDefinition,
    normed: trt.ITensor,
    cache_k: trt.ITensor,
    cache_v: trt.ITensor,
    attention_mask: trt.ITensor,
    position_id: trt.ITensor,
    cos_half_tensor: trt.ITensor,
    sin_half_tensor: trt.ITensor,
    attn_scale: float,
    eps_tensor: trt.ITensor,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    num_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    kv_lora_rank: int,
    q_lora_rank,
    attention_size: int,
    max_cache_length: int,
    dtype: np.dtype = np.float32,
) -> dict[str, trt.ITensor]:
    """Multi-head Latent Attention (MLA) block with naive KV cache.

    Implements the full MLA mechanism:
      1. Q path: direct projection (V2-Lite) or LoRA compression (V2 full)
      2. KV path: compress -> norm -> decompress -> split K_nope / V
      3. Partial RoPE on Q_rope and K_rope
      4. Broadcast K_rope from single-head to all heads
      5. Assemble full K = [K_nope, K_rope], Q = [Q_nope, Q_rope]
      6. Pad V with zeros to match K head dim for uniform cache
      7. Standard scaled dot-product attention with KV cache

    Returns {"attn_out", "present_k", "present_v"}.
    """
    attention_window = max_cache_length + 1
    k_head_dim = qk_nope_head_dim + qk_rope_head_dim  # 192 for V2-Lite
    q_total = num_heads * k_head_dim
    rope_total = num_heads * qk_rope_head_dim

    # ===== Q path =====
    if q_lora_rank is not None and q_lora_rank > 0:
        # V2 full: Q goes through LoRA compression
        # hidden -> q_a_proj -> q_a_layernorm -> q_b_proj
        q_compressed = graph_ops.add_matmul_rhs_constant(
            network, normed, hidden_size, q_lora_rank, weights[f"{prefix}.w_q_a"], dtype=dtype
        )  # [1, q_lora_rank]
        q_compressed = graph_ops.add_rms_norm(
            network,
            q_compressed,
            q_lora_rank,
            weights[f"{prefix}.q_a_norm"],
            eps_tensor,
            dtype=dtype,
        )  # [1, q_lora_rank]
        q = graph_ops.add_matmul_rhs_constant(
            network, q_compressed, q_lora_rank, q_total, weights[f"{prefix}.w_q_b"], dtype=dtype
        )  # [1, q_total]
    else:
        # V2-Lite: direct Q projection
        q = graph_ops.add_matmul_rhs_constant(
            network, normed, hidden_size, q_total, weights[f"{prefix}.w_q"], dtype=dtype
        )  # [1, num_heads * (nope + rope)]

    # Split Q into nope and rope parts per head:
    # q shape: [1, num_heads * (nope + rope)]
    # Reshape to [num_heads, nope + rope] then split
    q_reshaped = network.add_shuffle(q)
    q_reshaped.reshape_dims = (num_heads, k_head_dim)

    # Q_nope: [num_heads, qk_nope_head_dim]
    q_nope_slice = network.add_slice(
        q_reshaped.get_output(0), start=(0, 0), shape=(num_heads, qk_nope_head_dim), stride=(1, 1)
    )
    q_nope = q_nope_slice.get_output(0)

    # Q_rope: [num_heads, qk_rope_head_dim]
    q_rope_slice = network.add_slice(
        q_reshaped.get_output(0),
        start=(0, qk_nope_head_dim),
        shape=(num_heads, qk_rope_head_dim),
        stride=(1, 1),
    )

    # Flatten Q_rope for RoPE application: [1, num_heads * qk_rope_head_dim]
    q_rope_flat = network.add_shuffle(q_rope_slice.get_output(0))
    q_rope_flat.reshape_dims = (1, rope_total)

    # Apply native RoPE to Q_rope
    q_rope_roped = graph_ops.add_apply_rope_native(
        network,
        q_rope_flat.get_output(0),
        num_heads,
        qk_rope_head_dim,
        cos_half_tensor,
        sin_half_tensor,
        position_id,
        qk_rope_head_dim,
        interleaved=True,
    )

    # Reshape back to [num_heads, qk_rope_head_dim]
    q_rope_heads = network.add_shuffle(q_rope_roped)
    q_rope_heads.reshape_dims = (num_heads, qk_rope_head_dim)

    # Assemble full Q: [num_heads, k_head_dim] = [Q_nope, Q_rope]
    q_full_cat = network.add_concatenation([q_nope, q_rope_heads.get_output(0)])
    q_full_cat.axis = 1  # concat on head_dim axis
    # q_full: [num_heads, k_head_dim]

    # ===== KV path =====
    # Step 1: KV-A projection with MQA
    # hidden -> [1, kv_lora_rank + qk_rope_head_dim]
    kv_a_dim = kv_lora_rank + qk_rope_head_dim
    c_kv = graph_ops.add_matmul_rhs_constant(
        network, normed, hidden_size, kv_a_dim, weights[f"{prefix}.w_kv_a"], dtype=dtype
    )

    # Split into latent and k_rope_pass
    # c_kv_latent: [1, kv_lora_rank]
    c_kv_latent_slice = network.add_slice(
        c_kv, start=(0, 0), shape=(1, kv_lora_rank), stride=(1, 1)
    )
    c_kv_latent = c_kv_latent_slice.get_output(0)

    # k_rope_pass: [1, qk_rope_head_dim] -- single-head rope input for K
    k_rope_pass_slice = network.add_slice(
        c_kv, start=(0, kv_lora_rank), shape=(1, qk_rope_head_dim), stride=(1, 1)
    )
    k_rope_pass = k_rope_pass_slice.get_output(0)

    # Step 2: RMSNorm on latent
    c_kv_normed = graph_ops.add_rms_norm(
        network, c_kv_latent, kv_lora_rank, weights[f"{prefix}.kv_a_norm"], eps_tensor, dtype=dtype
    )

    # Step 3: KV-B projection: decompress
    # [1, kv_lora_rank] -> [1, num_heads * (qk_nope_head_dim + v_head_dim)]
    kv_b_out_dim = num_heads * (qk_nope_head_dim + v_head_dim)
    kv_expanded = graph_ops.add_matmul_rhs_constant(
        network, c_kv_normed, kv_lora_rank, kv_b_out_dim, weights[f"{prefix}.w_kv_b"], dtype=dtype
    )

    # Split into K_nope and V per head
    # Reshape to [num_heads, qk_nope_head_dim + v_head_dim]
    kv_per_head = network.add_shuffle(kv_expanded)
    kv_per_head.reshape_dims = (num_heads, qk_nope_head_dim + v_head_dim)

    # K_nope: [num_heads, qk_nope_head_dim]
    k_nope_slice = network.add_slice(
        kv_per_head.get_output(0), start=(0, 0), shape=(num_heads, qk_nope_head_dim), stride=(1, 1)
    )
    k_nope = k_nope_slice.get_output(0)

    # V: [num_heads, v_head_dim]
    v_slice = network.add_slice(
        kv_per_head.get_output(0),
        start=(0, qk_nope_head_dim),
        shape=(num_heads, v_head_dim),
        stride=(1, 1),
    )
    v_heads = v_slice.get_output(0)

    # Step 4: Apply native RoPE to the shared K-rope head, then broadcast.
    k_rope_roped = graph_ops.add_apply_rope_native(
        network,
        k_rope_pass,
        1,
        qk_rope_head_dim,
        cos_half_tensor,
        sin_half_tensor,
        position_id,
        qk_rope_head_dim,
        interleaved=True,
    )
    k_rope_copies = [k_rope_roped for _ in range(num_heads)]
    k_rope_broadcast = network.add_concatenation(k_rope_copies)
    k_rope_broadcast.axis = 0
    k_rope_heads = k_rope_broadcast.get_output(0)

    # Assemble full K: [num_heads, k_head_dim] = [K_nope, K_rope]
    k_full_cat = network.add_concatenation([k_nope, k_rope_heads])
    k_full_cat.axis = 1

    # Step 5: Pad V to match K head dim for uniform cache
    # V is [num_heads, v_head_dim], pad to [num_heads, k_head_dim]
    pad_size = k_head_dim - v_head_dim
    if pad_size > 0:
        zero_pad = graph_ops.add_constant(
            network,
            (num_heads, pad_size),
            np.zeros((num_heads, pad_size), dtype=dtype),
            dtype=dtype,
        )
        v_padded_cat = network.add_concatenation([v_heads, zero_pad])
        v_padded_cat.axis = 1
        v_padded = v_padded_cat.get_output(0)  # [num_heads, k_head_dim]
    else:
        v_padded = v_heads

    # Flatten K and V for cache: [1, attention_size]
    k_flat = network.add_shuffle(k_full_cat.get_output(0))
    k_flat.reshape_dims = (1, attention_size)
    v_flat = network.add_shuffle(v_padded)
    v_flat.reshape_dims = (1, attention_size)

    # Save present K/V (for cache update)
    present_k = k_flat.get_output(0)
    present_v = v_flat.get_output(0)

    # ===== Standard attention with cache =====

    # Concatenate with cache
    all_k = network.add_concatenation([cache_k, k_flat.get_output(0)])
    all_k.axis = 0
    all_v = network.add_concatenation([cache_v, v_flat.get_output(0)])
    all_v.axis = 0

    q_flat = network.add_shuffle(q_full_cat.get_output(0))
    q_flat.reshape_dims = (1, attention_size)
    mask_4d = graph_ops.add_2d_mask_to_4d(network, attention_mask)
    attn_context = graph_ops.add_attention_from_rows(
        network,
        q_flat.get_output(0),
        all_k.get_output(0),
        all_v.get_output(0),
        num_heads=num_heads,
        head_dim=k_head_dim,
        q_seq=1,
        kv_seq=attention_window,
        mask=mask_4d,
        scale=attn_scale,
        fp32_accumulation=_use_fp32_mla_attention(dtype, k_head_dim),
    )

    # Slice out only the v_head_dim portion (remove zero-padding)
    if pad_size > 0:
        context_heads = network.add_shuffle(attn_context)
        context_heads.reshape_dims = (num_heads, k_head_dim)
        context_sliced = network.add_slice(
            context_heads.get_output(0), start=(0, 0), shape=(num_heads, v_head_dim), stride=(1, 1)
        )
        context_for_proj = context_sliced.get_output(0)
        context_flat = network.add_shuffle(context_for_proj)
        context_flat.reshape_dims = (1, num_heads * v_head_dim)
        attn_context = context_flat.get_output(0)

    # Output projection: [1, num_heads * v_head_dim] -> [1, hidden_size]
    v_total = num_heads * v_head_dim
    attn_out = graph_ops.add_matmul_rhs_constant(
        network, attn_context, v_total, hidden_size, weights[f"{prefix}.w_o"], dtype=dtype
    )

    return {
        "attn_out": attn_out,
        "present_k": present_k,
        "present_v": present_v,
    }


# ---------------------------------------------------------------------------
# MoE Block with Shared Experts
# ---------------------------------------------------------------------------


def _add_swiglu_expert(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden_size: int,
    intermediate_size: int,
    w_gate: np.ndarray,
    w_up: np.ndarray,
    w_down: np.ndarray,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Compute a single SwiGLU expert: down(silu(gate(x)) * up(x))."""
    gate = graph_ops.add_matmul_rhs_constant(
        network, inp, hidden_size, intermediate_size, w_gate, dtype=dtype
    )
    up = graph_ops.add_matmul_rhs_constant(
        network, inp, hidden_size, intermediate_size, w_up, dtype=dtype
    )

    sigmoid = network.add_activation(gate, trt.ActivationType.SIGMOID)
    swish = network.add_elementwise(gate, sigmoid.get_output(0), trt.ElementWiseOperation.PROD)
    gated = network.add_elementwise(swish.get_output(0), up, trt.ElementWiseOperation.PROD)

    down = graph_ops.add_matmul_rhs_constant(
        network, gated.get_output(0), intermediate_size, hidden_size, w_down, dtype=dtype
    )
    return down


def _stack_expert_weights(
    weights: WeightDict,
    prefix: str,
    n_routed_experts: int,
    key: str,
    dtype: np.dtype,
) -> np.ndarray:
    """Stack one per-expert projection into a single [experts, rows, cols] array."""
    return np.ascontiguousarray(
        np.stack(
            [
                np.asarray(weights[f"{prefix}.expert.{index}.{key}"])
                for index in range(n_routed_experts)
            ]
        ).astype(dtype)
    )


def _add_native_routed_experts(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    n_routed_experts: int,
    num_experts_per_tok: int,
    top_indices: trt.ITensor,
    scaled_weights: trt.ITensor,
    dtype: np.dtype,
) -> trt.ITensor:
    """Routed-expert output computed by the native TensorRT MoE layer.

    ``set_gated_weights`` takes one stacked tensor per projection covering every
    expert, in the orientation this family already stores them: gate and up as
    ``[experts, hidden, intermediate]`` and down as
    ``[experts, intermediate, hidden]``. No transposition is required.

    The layer applies the routing scores itself, so its output is the weighted
    sum over the selected experts and only those experts are evaluated.
    """
    w_gate = _stack_expert_weights(weights, prefix, n_routed_experts, "w_gate", dtype)
    w_up = _stack_expert_weights(weights, prefix, n_routed_experts, "w_up", dtype)
    w_down = _stack_expert_weights(weights, prefix, n_routed_experts, "w_down", dtype)

    # IMoELayer requires rank-3 hidden states [batch, tokens, hidden]; the
    # decoder carries rank-2 [tokens, hidden].
    def _with_batch_dim(tensor: trt.ITensor, last_dim: int) -> trt.ITensor:
        shuffle = network.add_shuffle(tensor)
        shuffle.reshape_dims = (1, -1, last_dim)
        return shuffle.get_output(0)

    rank = len(tuple(inp.shape))
    if rank == 2:
        moe_hidden = _with_batch_dim(inp, hidden_size)
        moe_indices = _with_batch_dim(top_indices, num_experts_per_tok)
        moe_scores = _with_batch_dim(scaled_weights, num_experts_per_tok)
    elif rank == 3:
        moe_hidden, moe_indices, moe_scores = inp, top_indices, scaled_weights
    else:
        raise ValueError(f"DeepSeek-V2 MoE expects rank-2 or rank-3 hidden states, got rank {rank}")

    moe = network.add_moe(moe_hidden, moe_indices, moe_scores)
    if moe is None:
        raise RuntimeError(
            "TensorRT rejected addMoE for this build; the per-expert path "
            "should have been selected instead"
        )
    moe.set_gated_weights(
        graph_ops.add_constant(network, w_gate.shape, w_gate, dtype=dtype),
        graph_ops.add_constant(network, w_up.shape, w_up, dtype=dtype),
        graph_ops.add_constant(network, w_down.shape, w_down, dtype=dtype),
        trt.MoEActType.SILU,
    )
    routed_out = moe.get_output(0)

    if rank == 2:
        restore = network.add_shuffle(routed_out)
        restore.reshape_dims = (-1, hidden_size)
        routed_out = restore.get_output(0)
    return routed_out


def _add_moe_with_shared_experts(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    n_routed_experts: int,
    moe_intermediate: int,
    num_experts_per_tok: int,
    shared_intermediate: int,
    norm_topk_prob: bool = False,
    routed_scaling_factor: float = 1.0,
    scoring_func: str = "softmax",
    topk_method: str = "greedy",
    n_group: int = 1,
    topk_group: int = 1,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """MoE block with shared experts (DeepSeek-V2 style).

    1. Router logits -> softmax/sigmoid -> top-k selection
    2. Scale weights: renormalize (norm_topk_prob=True) or multiply by
       routed_scaling_factor (norm_topk_prob=False)
    3. Native TensorRT MoE routed-expert output.
    4. Compute shared expert output (always active)
    5. Final = routed_output + shared_output
    """
    top_indices, scaled_weights = moe_routing.add_router(
        network,
        inp,
        weights[f"{prefix}.router"],
        hidden_size=hidden_size,
        n_routed_experts=n_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        scoring_func=scoring_func,
        topk_method=topk_method,
        correction_bias=weights.get(f"{prefix}.router_score_bias"),
        n_group=n_group,
        topk_group=topk_group,
        norm_topk_prob=norm_topk_prob,
        routed_scaling_factor=routed_scaling_factor,
    )

    result = _add_native_routed_experts(
        network,
        inp,
        weights,
        prefix,
        hidden_size,
        n_routed_experts,
        num_experts_per_tok,
        top_indices,
        scaled_weights,
        dtype,
    )
    return _add_shared_expert_residual(
        network, inp, weights, prefix, hidden_size, shared_intermediate, result, dtype
    )


def _add_shared_expert_residual(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    shared_intermediate: int,
    routed_out: trt.ITensor,
    dtype: np.dtype,
) -> trt.ITensor:
    """Add the always-active shared-expert output to the routed-expert output."""
    shared_out = _add_swiglu_expert(
        network,
        inp,
        hidden_size,
        shared_intermediate,
        weights[f"{prefix}.shared.w_gate"],
        weights[f"{prefix}.shared.w_up"],
        weights[f"{prefix}.shared.w_down"],
        dtype=dtype,
    )
    if routed_out.dtype != shared_out.dtype:
        routed_out = network.add_cast(routed_out, shared_out.dtype).get_output(0)
    combined = network.add_elementwise(routed_out, shared_out, trt.ElementWiseOperation.SUM)
    return combined.get_output(0)


# ---------------------------------------------------------------------------
# Decoder Layer
# ---------------------------------------------------------------------------


def _add_deepseek_v2_decoder_layer(
    *,
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    cache_k: trt.ITensor,
    cache_v: trt.ITensor,
    attention_mask: trt.ITensor,
    position_id: trt.ITensor,
    cos_half_tensor: trt.ITensor,
    sin_half_tensor: trt.ITensor,
    attn_scale: float,
    eps_tensor: trt.ITensor,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    num_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    kv_lora_rank: int,
    q_lora_rank,
    attention_size: int,
    max_cache_length: int,
    is_moe_layer: bool,
    n_routed_experts: int,
    n_shared_experts: int,
    num_experts_per_tok: int,
    moe_intermediate: int,
    shared_intermediate: int,
    dense_intermediate: int,
    norm_topk_prob: bool = False,
    routed_scaling_factor: float = 1.0,
    scoring_func: str = "softmax",
    topk_method: str = "greedy",
    n_group: int = 1,
    topk_group: int = 1,
    dtype: np.dtype = np.float32,
) -> dict[str, trt.ITensor]:
    """Add one DeepSeek-V2 decoder layer: MLA attention + (dense MLP or MoE)."""

    # Pre-attention RMSNorm
    norm1 = _apply_norm(
        network,
        hidden,
        hidden_size,
        weights[f"{prefix}.input_norm"],
        None,
        eps_tensor,
        "rmsnorm",
        dtype=dtype,
    )

    # MLA attention block
    attn = _add_mla_attention_block(
        network=network,
        normed=norm1,
        cache_k=cache_k,
        cache_v=cache_v,
        attention_mask=attention_mask,
        position_id=position_id,
        cos_half_tensor=cos_half_tensor,
        sin_half_tensor=sin_half_tensor,
        attn_scale=attn_scale,
        eps_tensor=eps_tensor,
        weights=weights,
        prefix=prefix,
        hidden_size=hidden_size,
        num_heads=num_heads,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        kv_lora_rank=kv_lora_rank,
        q_lora_rank=q_lora_rank,
        attention_size=attention_size,
        max_cache_length=max_cache_length,
        dtype=dtype,
    )
    attn_out = attn["attn_out"]

    # Residual connection after attention
    residual1 = network.add_elementwise(hidden, attn_out, trt.ElementWiseOperation.SUM)

    # Post-attention RMSNorm
    norm2 = _apply_norm(
        network,
        residual1.get_output(0),
        hidden_size,
        weights[f"{prefix}.post_attn_norm"],
        None,
        eps_tensor,
        "rmsnorm",
        dtype=dtype,
    )

    # MLP: either dense or MoE with shared experts
    if is_moe_layer:
        mlp_out = _add_moe_with_shared_experts(
            network,
            norm2,
            weights,
            prefix,
            hidden_size,
            n_routed_experts,
            moe_intermediate,
            num_experts_per_tok,
            shared_intermediate,
            norm_topk_prob=norm_topk_prob,
            routed_scaling_factor=routed_scaling_factor,
            scoring_func=scoring_func,
            topk_method=topk_method,
            n_group=n_group,
            topk_group=topk_group,
            dtype=dtype,
        )
    else:
        mlp_out = graph_blocks.add_swiglu_mlp(
            network,
            norm2,
            weights=weights,
            prefix=prefix,
            hidden_size=hidden_size,
            mlp_size=dense_intermediate,
            dtype=dtype,
        )

    # Residual connection after MLP
    residual2 = network.add_elementwise(
        residual1.get_output(0), mlp_out, trt.ElementWiseOperation.SUM
    )

    return {
        "hidden": residual2.get_output(0),
        "post_attn": residual1.get_output(0),
        "present_k": attn["present_k"],
        "present_v": attn["present_v"],
    }


_BUNDLE_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.model",
)


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _runtime_config(
    model_dir: Path, config: ModelConfig, model: _DeepSeekV2Model, **updates
) -> dict:
    runtime = {
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "bos_token_id": config.bos_token_id,
        "eos_token_id": config.eos_token_id,
        "pad_token_id": config.pad_token_id,
    }
    runtime.update(model.get_bundle_config_overrides(config))
    generation_path = model_dir / "generation_config.json"
    if generation_path.is_file():
        generation = json.loads(generation_path.read_text(encoding="utf-8"))
        if not isinstance(generation, dict):
            raise ValueError("generation_config.json must contain one JSON object")
        if "eos_token_id" in generation:
            runtime["eos_token_id"] = generation["eos_token_id"]
    runtime.update(updates)
    return runtime


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one DeepSeek-V2 bundle through family-owned code."""
    if request.image_height is not None:
        raise NotImplementedError("deepseek_v2 does not support image_height")

    if request.image_width is not None:
        raise NotImplementedError("deepseek_v2 does not support image_width")

    if request.video_num_frames is not None:
        raise NotImplementedError("deepseek_v2 does not support video_num_frames")

    if request.max_batch_size != 1:
        raise NotImplementedError("deepseek_v2 does not support max_batch_size")

    if request.context_parallel_size != 1:
        raise ValueError("this family does not support context parallelism")

    if request.task != "text_generation":
        raise ValueError("deepseek_v2 supports only task=text_generation")
    model_dir = Path(request.model_dir)
    config = ModelConfig.from_dir(model_dir)
    if str(config.model_type).lower() not in {"deepseek_v2", "deepseek_v3"}:
        raise ValueError(f"DeepSeek-V2 does not support model_type={config.model_type!r}")
    precision = str(request.precision).lower()
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("DeepSeek-V2 precision must be fp32, fp16, or bf16")
    model = _DeepSeekV2Model()
    default_length = min(config.max_position_embeddings, 256)
    max_sequence_length = _positive_int(
        request.max_sequence_length or default_length, "max_sequence_length"
    )
    if max_sequence_length > config.max_position_embeddings:
        raise ValueError("DeepSeek-V2 max_sequence_length exceeds checkpoint context capacity")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("DeepSeek-V2 quantization requires a family-owned qualified path")
    if request.fp32_layers:
        raise NotImplementedError("DeepSeek-V2 does not expose mixed-precision layer selection")
    parallel = ParallelConfig(
        tp_size=_positive_int(request.tensor_parallel_size, "tensor_parallel_size")
    )
    parallel.validate()
    config.raw["_model_dir"] = str(model_dir)
    config.raw["_resolved_build_precision"] = precision
    config.raw["_parallel_build_enabled"] = parallel.enabled
    config.raw["_quantized_build_requested"] = False
    weights = model.load_weights(str(model_dir), config)
    writer.set_header(family="deepseek_v2", task=request.task, backend="trt")
    if parallel.enabled:
        for rank in range(parallel.tp_size):
            plan = model.build_engine(
                config,
                weights,
                max_sequence_length,
                precision=precision,
                quant_ctx=None,
                verbose=bool(request.verbose),
                debug_layer_outputs=False,
                parallel_config=parallel.for_rank(rank),
            )
            writer.add_bytes(f"engine.rank{rank}.plan", plan)
        layout = "dual_profile"
    else:
        plan = model.build_engine(
            config,
            weights,
            max_sequence_length,
            precision=precision,
            quant_ctx=None,
            verbose=bool(request.verbose),
            debug_layer_outputs=False,
            parallel_config=parallel,
        )
        writer.add_bytes("engine.plan", plan)
        layout = "single"
    writer.add_json(
        "runtime.json",
        _runtime_config(
            model_dir,
            config,
            model,
            precision=precision,
            max_cache_length=max_sequence_length,
            decoder_engine_layout=layout,
            tensor_parallel_size=parallel.tp_size,
            tensor_parallel_mode="tensor_parallel" if parallel.enabled else "single",
        ),
    )
    for filename in _BUNDLE_FILES:
        path = model_dir / filename
        if path.is_file():
            writer.add_bytes(filename, path.read_bytes())
