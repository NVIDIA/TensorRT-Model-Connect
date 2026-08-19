# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V2 family model — Multi-head Latent Attention + Mixture of Experts.

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

import sys
from pathlib import Path

import numpy as np
from tensorrt_model_connect import trt_compat

from .config import ModelConfig
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
from ...parallel_config import (
    normalize_parallel_config,
    require_tensorrt_11_for_tensor_parallel,
)
from .default_decoder import _apply_norm, _mark_debug_output


trt = trt_compat.get_trt()


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


name = "deepseek_v2"
runtime_strategy = "deepseek_v2_decoder_kv_cache"
runtime_capabilities = {"decoder_kv"}


def matches(config: object) -> bool:
    """Return whether this module owns the parsed model config."""
    model_type = str(getattr(config, "model_type", config))
    return model_type.lower() in ("deepseek_v2", "deepseek_v3")


def get_bundle_config_overrides(config: ModelConfig) -> dict | None:
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


def load_weights(model_dir: str, config: ModelConfig, *, precision: str = "fp32") -> WeightDict:
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

                weights[f"{prefix}.expert.{e}.w_gate"] = _transpose_2d(gate_raw, f"expert_{e}_gate")
                weights[f"{prefix}.expert.{e}.w_up"] = _transpose_2d(up_raw, f"expert_{e}_up")
                weights[f"{prefix}.expert.{e}.w_down"] = _transpose_2d(down_raw, f"expert_{e}_down")
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
        require_tensorrt_11_for_tensor_parallel(
            parallel, feature="DeepSeek-V2 tensor-parallel builds"
        )
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
    # HF uses: scaling = qk_head_dim ** (-0.5)
    # YaRN mscale is handled via rope_utils attention_factor which scales
    # cos/sin directly. For V2-Lite, mscale == mscale_all_dim so they
    # cancel out (attention_factor = 1.0). No adjustment needed here.
    attn_scale = 1.0 / np.sqrt(max(k_head_dim, 1))

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    trt_config = builder.create_builder_config()
    trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    if precision == "fp16":
        work_np_dtype = np.float16
        work_trt_dtype = trt.float16
    elif precision == "bf16":
        work_np_dtype = np.float16
        work_trt_dtype = trt.bfloat16
    else:
        work_np_dtype = np.float32
        work_trt_dtype = trt.float32

    requested_fp32_layers = frozenset(int(layer) for layer in config.raw.get("_fp32_layers", ()))
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
        network, (1, 1), np.array([config.rms_norm_eps], dtype=work_np_dtype), dtype=work_np_dtype
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

    1. Router logits -> softmax -> top-k selection
    2. Scale weights: renormalize (norm_topk_prob=True) or multiply by
       routed_scaling_factor (norm_topk_prob=False)
    3. Compute all routed expert outputs, select top-k, weighted sum
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

    # 5. Compute ALL routed expert outputs and stack
    expert_outputs = []
    for e in range(n_routed_experts):
        exp_out = _add_swiglu_expert(
            network,
            inp,
            hidden_size,
            moe_intermediate,
            weights[f"{prefix}.expert.{e}.w_gate"],
            weights[f"{prefix}.expert.{e}.w_up"],
            weights[f"{prefix}.expert.{e}.w_down"],
            dtype=dtype,
        )
        expert_outputs.append(exp_out)

    stacked = network.add_concatenation(expert_outputs)
    stacked.axis = 0
    stacked_out = stacked.get_output(0)  # [n_routed_experts, hidden_size]

    # 6. Gather selected experts, scale, and sum
    result = None
    for k in range(num_experts_per_tok):
        # Extract index k
        idx_slice = network.add_slice(top_indices, start=(0, k), shape=(1, 1), stride=(1, 1))
        idx_flat = network.add_shuffle(idx_slice.get_output(0))
        idx_flat.reshape_dims = (1,)

        # Extract weight k
        w_slice = network.add_slice(scaled_weights, start=(0, k), shape=(1, 1), stride=(1, 1))
        selected_weight = w_slice.get_output(0)

        # Gather expert output
        expert_out = network.add_gather(stacked_out, idx_flat.get_output(0), 0)
        if selected_weight.dtype != expert_out.get_output(0).dtype:
            selected_weight = network.add_cast(
                selected_weight,
                expert_out.get_output(0).dtype,
            ).get_output(0)

        # Scale
        scaled_expert = network.add_elementwise(
            expert_out.get_output(0), selected_weight, trt.ElementWiseOperation.PROD
        )

        if result is None:
            result = scaled_expert.get_output(0)
        else:
            sum_layer = network.add_elementwise(
                result, scaled_expert.get_output(0), trt.ElementWiseOperation.SUM
            )
            result = sum_layer.get_output(0)

    # 7. Shared expert output (always active)
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

    # 8. Combine: routed_output + shared_output
    combined = network.add_elementwise(result, shared_out, trt.ElementWiseOperation.SUM)

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


def supports_split_decoder_roles(config: ModelConfig) -> bool:
    return False


def _dynamic_kv_profile_rows(
    max_cache_length: int,
    kv_budget: int,
    *,
    bucket_rows: int = 32,
    preferred_rows: list[int] | None = None,
) -> list[int]:
    if max_cache_length < 1:
        return [1]
    start = ((max(kv_budget, 1) + bucket_rows - 1) // bucket_rows) * bucket_rows
    start = max(bucket_rows, min(start, max_cache_length))
    rows: list[int] = []

    def add_row(value: int) -> None:
        rounded = (
            (min(max(value, 1), max_cache_length) + bucket_rows - 1) // bucket_rows
        ) * bucket_rows
        rounded = max(bucket_rows, min(rounded, max_cache_length))
        if rounded not in rows:
            rows.append(rounded)

    for value in preferred_rows or ():
        add_row(value)
    row = start
    while row < max_cache_length:
        add_row(row)
        row = (
            (min(max(row + bucket_rows, row * 2), max_cache_length) + bucket_rows - 1)
            // bucket_rows
        ) * bucket_rows
    add_row(max_cache_length)
    return sorted(rows)


def _sanitize_dynamic_kv_profile_rows(
    rows: list[int] | None,
    max_cache_length: int,
) -> list[int] | None:
    if rows is None:
        return None
    sanitized = sorted({max(1, min(int(value), max_cache_length)) for value in rows})
    if not sanitized:
        raise ValueError("dynamic_kv_profile_rows_override must contain at least one row")
    return sanitized


def build(model_dir: str, output_path: str, **options) -> None:
    """Build a complete deepseek_v2 bundle from checkpoint to serialized artifact."""
    import json
    import time
    from dataclasses import replace
    from datetime import datetime, timezone
    from pathlib import Path

    from tensorrt_model_connect import trt_compat
    from tensorrt_model_connect.build_timing import (
        add_build_timing,
        new_build_timing,
        write_build_timing,
    )
    from tensorrt_model_connect.bundle_writer import (
        BundleInfo,
        BundleSection,
        gpu_name,
        tensorrt_abi,
        tensorrt_version,
        write_bundle,
    )
    from tensorrt_model_connect.parallel_config import (
        normalize_parallel_config,
        rank_engine_section,
        require_tensorrt_11_for_tensor_parallel,
    )

    model_path = Path(model_dir)
    decoder_engine_layout = str(options.get("decoder_engine_layout") or "split")
    if decoder_engine_layout not in {"split", "dual_profile"}:
        raise ValueError(
            "decoder_engine_layout must be 'split' or 'dual_profile', "
            f"got {decoder_engine_layout!r}"
        )
    parallel = normalize_parallel_config(options.get("parallel_config"))
    if parallel.cp_enabled:
        raise NotImplementedError("deepseek_v2 does not support context-parallel builds")

    config = ModelConfig.from_dir(model_path)
    config.raw["_model_dir"] = str(model_path)
    config.raw["_decoder_engine_layout"] = decoder_engine_layout
    config.raw["_fp32_layers"] = sorted(set(options.get("fp32_layers") or ()))
    config.raw["_family_build_options"] = dict(options.get("family_build_options") or {})
    config.raw["_parallel_build_enabled"] = bool(parallel.enabled)
    config.raw["_rtx_build_requested"] = bool(options.get("rtx"))
    config.raw["_runtime_dynamic_kv_requested"] = bool(
        options.get("dynamic_kv_cache") or options.get("triattention_stats_path")
    )
    config.raw["_quantized_build_requested"] = bool(options.get("quantize"))

    precision = str(options.get("precision") or "fp32").lower()
    config.raw["_resolved_build_precision"] = precision
    requested_cache_length = options.get("max_cache_length")
    max_cache_length = int(256) if requested_cache_length is None else int(requested_cache_length)
    if max_cache_length < 1:
        raise ValueError("max_cache_length must be >= 1")

    timing = new_build_timing(options.get("build_timing_path"))
    timing["model_dir"] = str(model_path)
    timing["output_path"] = str(output_path)
    started = time.monotonic()
    write_build_timing(timing)

    weights_started = time.monotonic()
    weights = load_weights(str(model_path), config, precision=precision)
    add_build_timing(timing, "weights_loading_s", time.monotonic() - weights_started)
    write_build_timing(timing)

    quantize = options.get("quantize")
    quant_ctx = None
    quant_plan = None
    if quantize:
        from tensorrt_model_connect.quantization import QuantPlan, build_quant_context
        from . import graph_ops

        quant_plan = QuantPlan.from_build_args(
            precision=precision,
            quantize=str(quantize),
            quant_scales=options.get("quant_scales"),
            quant_calibration_samples=int(options.get("quant_calibration_samples") or 512),
        )
        quant_method = str(
            config.raw.get("quantization_config", {}).get("quant_method", "")
        ).lower()
        if quant_plan.scale_source == "modelopt" and quant_method in {
            "awq",
            "gptq",
            "compressed-tensors",
            "compressed_tensors",
        }:
            quant_plan = replace(quant_plan, scale_source="prequantized")
        quant_ctx = build_quant_context(
            format_name=quant_plan.quant_format,
            model_dir=str(model_path),
            config=config,
            scales_json=options.get("quant_scales"),
            num_calibration_samples=int(options.get("quant_calibration_samples") or 512),
            quant_plan=quant_plan,
            graph_ops=graph_ops,
        )

    dynamic_kv_cache = bool(options.get("dynamic_kv_cache"))
    dynamic_kv_budget = 1
    triattention_config = None
    triattention_section = None
    if options.get("triattention_stats_path"):
        from tensorrt_model_connect.triattention_export import (
            TriAttentionBundleConfig,
            export_triattention_stats_section,
        )

        recent_window = int(options.get("triattention_recent_window", 128))
        divide_length = int(options.get("triattention_divide_length", 128))
        score_aggregation = str(options.get("triattention_score_aggregation", "mean"))
        if recent_window < 0:
            raise ValueError("TriAttention recent_window must be >= 0")
        if divide_length < 1:
            raise ValueError("TriAttention divide_length must be >= 1")
        if score_aggregation not in {"mean", "max"}:
            raise ValueError("TriAttention score_aggregation must be 'mean' or 'max'")
        requested_budget = options.get("triattention_kv_budget")
        dynamic_kv_budget = max_cache_length if requested_budget is None else int(requested_budget)
        if not 1 <= dynamic_kv_budget <= max_cache_length:
            raise ValueError("TriAttention kv_budget must fit max_cache_length")
        triattention_config = TriAttentionBundleConfig(
            kv_budget=dynamic_kv_budget,
            divide_length=divide_length,
            recent_window=recent_window,
            score_aggregation=score_aggregation,
            count_prompt_tokens=bool(options.get("triattention_count_prompt_tokens", True)),
            protect_prefill=bool(options.get("triattention_protect_prefill", True)),
            disable_mlr=bool(options.get("triattention_disable_mlr", False)),
            disable_trig=bool(options.get("triattention_disable_trig", False)),
        )
        triattention_section = export_triattention_stats_section(
            str(options["triattention_stats_path"]),
            config=config,
        )
        dynamic_kv_cache = True

    if dynamic_kv_cache:
        rows = _sanitize_dynamic_kv_profile_rows(
            options.get("dynamic_kv_profile_rows_override"),
            max_cache_length,
        )
        if rows is None:
            preferred_rows = (
                [max(32, dynamic_kv_budget // 2)]
                if triattention_config is not None and dynamic_kv_budget >= 4096
                else None
            )
            rows = _dynamic_kv_profile_rows(
                max_cache_length,
                dynamic_kv_budget,
                preferred_rows=preferred_rows,
            )
        config.raw["dynamic_kv_cache"] = True
        config.raw["_dynamic_kv_opt_length"] = max_cache_length
        config.raw["_dynamic_kv_profile_rows"] = rows

    if parallel.enabled:
        require_tensorrt_11_for_tensor_parallel(
            parallel, feature="deepseek_v2 tensor-parallel builds"
        )

        if quant_ctx is not None:
            raise ValueError("deepseek_v2 tensor-parallel builds do not support quantization")
        if dynamic_kv_cache:
            raise NotImplementedError(
                "Tensor-parallel deepseek_v2 builds do not support dynamic KV cache or TriAttention"
            )

    verbose = bool(options.get("verbose"))

    def build_role(role: str, *, rank_parallel=parallel) -> bytes:
        from tensorrt_model_connect.tvm_ffi.graph_build import engine_role

        previous = config.raw.get("_decoder_engine_role")
        config.raw["_decoder_engine_role"] = role
        try:
            with engine_role(role):
                return build_engine(
                    config,
                    weights,
                    max_cache_length,
                    precision=precision,
                    quant_ctx=quant_ctx,
                    verbose=verbose,
                    parallel_config=rank_parallel,
                )
        finally:
            if previous is None:
                config.raw.pop("_decoder_engine_role", None)
            else:
                config.raw["_decoder_engine_role"] = previous

    from tensorrt_model_connect.tvm_ffi.graph_build import inspection_role

    inspection = inspection_role()
    if inspection is not None:
        build_role(inspection)
        raise RuntimeError("graph inspection did not reach TensorRT serialization")

    compile_started = time.monotonic()
    if parallel.enabled:
        plans = {
            rank: build_role("dual_profile", rank_parallel=parallel.for_rank(rank))
            for rank in range(parallel.tp_size)
        }
        sections = [
            BundleSection(rank_engine_section(rank), plan) for rank, plan in sorted(plans.items())
        ]
        decoder_layout = "dual_profile"
    else:
        split = (
            decoder_engine_layout == "split"
            and not dynamic_kv_cache
            and supports_split_decoder_roles(config)
        )
        if split:
            previous_active = config.raw.get("_active_split_decoder_build")
            config.raw["_active_split_decoder_build"] = True
            try:
                quant_label = str(quantize or "noquant")

                def build_split_role(role: str) -> bytes:
                    scope = (
                        f"split-{config.model_type}-h{config.hidden_size}"
                        f"-l{config.num_hidden_layers}-{precision}-{quant_label}-{role}"
                    )
                    with trt_compat.scoped_timing_cache(scope):
                        return build_role(role)

                prefill_started = time.monotonic()
                prefill_plan = build_split_role("prefill")
                add_build_timing(
                    timing,
                    "trt_compile_prefill_engine_s",
                    time.monotonic() - prefill_started,
                )
                decode_started = time.monotonic()
                plan = build_split_role("decode")
                add_build_timing(
                    timing,
                    "trt_compile_decode_engine_s",
                    time.monotonic() - decode_started,
                )
            finally:
                if previous_active is None:
                    config.raw.pop("_active_split_decoder_build", None)
                else:
                    config.raw["_active_split_decoder_build"] = previous_active
            sections = [
                BundleSection("engine_plan", plan),
                BundleSection("prefill_engine_plan", prefill_plan),
            ]
            decoder_layout = "split"
        else:
            role = "dual_profile" if decoder_engine_layout == "dual_profile" else "decode"
            plan = build_role(role)
            sections = [BundleSection("engine_plan", plan)]
            decoder_layout = "dual_profile" if role == "dual_profile" else "single"
    compile_elapsed = time.monotonic() - compile_started
    add_build_timing(timing, "trt_compile_s", compile_elapsed)
    add_build_timing(timing, "trt_compile_main_engine_s", compile_elapsed)
    write_build_timing(timing)

    if triattention_config is not None and triattention_section is not None:
        sections.append(BundleSection(triattention_config.stats_section, triattention_section))

    from tensorrt_model_connect.tokenizer_conversion import (
        prepare_tokenizer_special_frame,
    )

    tokenizer_frame = prepare_tokenizer_special_frame(
        model_path,
        source_model_id_or_path=options.get("tokenizer_source_model_id_or_path"),
        source_revision=options.get("tokenizer_source_revision"),
    )
    prefix_ids, suffix_ids = tokenizer_frame or ([], [])
    add_special_tokens = bool(prefix_ids or suffix_ids)

    trt_version = tensorrt_version()
    trt_abi = tensorrt_abi(trt_version)
    info = BundleInfo(
        model_id=model_path.name,
        model_type=config.model_type,
        family=name,
        trt_version=trt_version,
        trt_abi=trt_abi,
        gpu_name=gpu_name(),
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        max_cache_length=max_cache_length,
        runtime_strategy=runtime_strategy,
        precision=precision,
        quantization=(quant_plan.quant_format if quant_plan else "none"),
        tokenizer_add_special_tokens=add_special_tokens,
    )

    source_config = model_path / "config.json"
    runtime_config = (
        json.loads(source_config.read_text(encoding="utf-8"))
        if source_config.is_file()
        else {key: value for key, value in config.raw.items() if not str(key).startswith("_")}
    )
    generation_config = model_path / "generation_config.json"
    if generation_config.is_file():
        generation = json.loads(generation_config.read_text(encoding="utf-8"))
        if "eos_token_id" in generation:
            runtime_config["eos_token_id"] = generation["eos_token_id"]
    runtime_config.update(
        {
            "runtime_strategy": runtime_strategy,
            "engine_backend": "trt_rtx" if options.get("rtx") else "trt",
            "trt_version": trt_version,
            "precision": precision,
            "tokenizer_add_special_tokens": int(add_special_tokens),
            "decoder_engine_layout": decoder_layout,
        }
    )
    if trt_abi:
        runtime_config["trt_abi"] = trt_abi
    if options.get("fp32_layers"):
        runtime_config["fp32_layers"] = sorted(set(options["fp32_layers"]))
    if tokenizer_frame is not None:
        runtime_config["tokenizer_special_prefix_ids"] = prefix_ids
        runtime_config["tokenizer_special_suffix_ids"] = suffix_ids
    if quant_plan is not None:
        runtime_config["quantization"] = quant_plan.as_config_dict()
    if dynamic_kv_cache:
        runtime_config["dynamic_kv_cache"] = True
        runtime_config["dynamic_kv_profile_rows"] = config.raw["_dynamic_kv_profile_rows"]
    if triattention_config is not None:
        runtime_config["triattention"] = triattention_config.to_dict()
    runtime_config.update(parallel.to_bundle_config_fields())
    overrides = get_bundle_config_overrides(config)
    if overrides is not None:
        merged = dict(overrides)
        merged.update(runtime_config)
        merged.update(overrides)
        runtime_config = merged
    tokenizer_override = None
    for filename in (
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "vocab.json",
        "merges.txt",
        "vocab.txt",
        "special_tokens_map.json",
        "tokenizer.model",
    ):
        path = model_path / filename
        if filename == "tokenizer.json" and tokenizer_override is not None:
            sections.append(BundleSection(filename, tokenizer_override))
        elif path.is_file():
            sections.append(BundleSection(filename, path.read_bytes()))
    sections.append(
        BundleSection(
            "config.json",
            json.dumps(runtime_config, indent=2).encode("utf-8"),
        )
    )

    from tensorrt_model_connect.tvm_ffi.graph_build import kernel_slots_section

    slot_section = kernel_slots_section()
    if slot_section is not None:
        sections.append(BundleSection("kernel_slots.json", slot_section))
    kernel_manifest = []
    for global_name, library in options.get("kernel_artifacts") or ():
        section_name = f"kernel_{global_name.replace('.', '_')}.so"
        sections.append(BundleSection(section_name, Path(library).read_bytes()))
        kernel_manifest.append(
            {"global_name": global_name, "func_name": "run", "section": section_name}
        )
    if kernel_manifest:
        sections.append(
            BundleSection(
                "kernel_manifest.json",
                json.dumps({"kernels": kernel_manifest}).encode("utf-8"),
            )
        )

    write_started = time.monotonic()
    write_bundle(output_path, info, sections)
    add_build_timing(timing, "bundle_write_s", time.monotonic() - write_started)
    timing["total_s"] = time.monotonic() - started
    write_build_timing(timing)
