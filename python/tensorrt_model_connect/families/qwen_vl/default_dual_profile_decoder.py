# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Split Qwen-VL decoder builder using TensorRT's native KV-cache contract.

Each invocation builds one role: dynamic-Sq batched prefill or fixed-Sq=1
decode. Both roles expose full-capacity cache tensors updated in place through
``IKVCacheUpdateLayer`` and consumed by non-decomposable ``IAttention``.

Tensor contract (matches the C++ runtime KvCache naming):
  Inputs (dynamic shapes — Sq varies by profile)
    token_id        int32   (-1,)
    position_id     int32   (-1,)
    mrope_position_ids int32 (3, -1)
    cache_write_indices int32 (1,)
    key_value_lengths int32 (1,)
    cache_k_i       bf16 (1, Hkv, capacity, 128)     # runtime-owned
    cache_v_i       bf16 (1, Hkv, capacity, 128)
  Outputs
    logits          float32 (1, vocab)               # last-row sliced inside the engine
    present_k_i     bf16 (1, Hkv, capacity, 128)     # aliases cache_k_i
    present_v_i     bf16 (1, Hkv, capacity, 128)     # aliases cache_v_i
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
from tensorrt_model_connect import trt_compat

from . import graph_ops
from . import graph_blocks
from .lora import DynamicLoraConfig
from .utils import (
    const_in_work_dtype as _const_in_work_dtype,
    create_builder_context,
    norm_multi as _norm_multi,
)

trt = trt_compat.get_trt()

if TYPE_CHECKING:
    from .config import ModelConfig
    from .checkpoint_mapper import WeightDict
    from ...quantization.context import QuantContext


_make_matmul_fn = graph_blocks.make_matmul_fn
_NATIVE_PREFILL_CHUNK_TOKENS = 32768
_NATIVE_BUILDER_WORKSPACE_BYTES = 16 << 30


# ---------------------------------------------------------------------------
# MLP helpers.
# ---------------------------------------------------------------------------


def _swiglu_mlp(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    *,
    matmul,
    weights: "WeightDict",
    prefix: str,
    hidden: int,
    mlp_size: int,
) -> trt.ITensor:
    gate = matmul(inp, hidden, mlp_size,
                  weights[f"{prefix}.w_gate"], f"{prefix}.w_gate")
    up = matmul(inp, hidden, mlp_size,
                weights[f"{prefix}.w_up"], f"{prefix}.w_up")
    sigmoid = network.add_activation(gate, trt.ActivationType.SIGMOID)
    swish = network.add_elementwise(
        gate, sigmoid.get_output(0), trt.ElementWiseOperation.PROD)
    gated = network.add_elementwise(
        swish.get_output(0), up, trt.ElementWiseOperation.PROD)
    mlp_out = matmul(gated.get_output(0), mlp_size, hidden,
                     weights[f"{prefix}.w_down"], f"{prefix}.w_down")
    return mlp_out


def _gelu_fc_mlp(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    *,
    matmul,
    weights: "WeightDict",
    prefix: str,
    hidden: int,
    mlp_size: int,
    activation: str,
    work_np_dtype: np.dtype,
) -> trt.ITensor:
    fc1 = matmul(inp, hidden, mlp_size,
                 weights[f"{prefix}.w_fc1"], f"{prefix}.w_fc1")
    fc1_bias = weights.get(f"{prefix}.fc1_bias")
    if fc1_bias is not None:
        fc1 = graph_ops.add_bias_sum(network, fc1, mlp_size, fc1_bias, dtype=work_np_dtype)
    activated = graph_ops.add_activation(network, fc1, activation, dtype=work_np_dtype)
    fc2 = matmul(activated, mlp_size, hidden,
                 weights[f"{prefix}.w_fc2"], f"{prefix}.w_fc2")
    fc2_bias = weights.get(f"{prefix}.fc2_bias")
    if fc2_bias is not None:
        fc2 = graph_ops.add_bias_sum(network, fc2, hidden, fc2_bias, dtype=work_np_dtype)
    return fc2


# ---------------------------------------------------------------------------
# Config guard.
# ---------------------------------------------------------------------------


def _supports_config(config: "ModelConfig", weights: "WeightDict") -> None:
    """Reject configs the dual-profile builder cannot handle."""
    model_type = getattr(config, "model_type", "").lower()
    if "moe" in model_type or "mamba" in model_type or "rwkv" in model_type:
        raise NotImplementedError(
            f"dual_profile_decoder_builder does not support model_type={model_type!r}")
    if "embedding" not in weights:
        raise NotImplementedError("missing embedding weight")
    if "final_norm" not in weights:
        raise NotImplementedError("missing final_norm weight")


# ---------------------------------------------------------------------------
# Main builder.
# ---------------------------------------------------------------------------


def build_dual_profile_decoder_engine(
    config: "ModelConfig",
    weights: "WeightDict",
    max_cache_length: int,
    *,
    precision: str = "bf16",
    opt_prefill_length: int = 64,
    max_prefill_length: int | None = None,
    builder_workspace_bytes: int = _NATIVE_BUILDER_WORKSPACE_BYTES,
    quant_ctx: "QuantContext | None" = None,
    norm_type: str = "rmsnorm",
    mlp_type: str = "swiglu",
    position_type: str = "rope",
    activation: str = "silu",
    partial_rotary_factor: float = 1.0,
    interleaved_rope: bool = False,
    parallel_residual: bool = False,
    scale_attn_weights: bool = True,
    embed_input: bool = False,
    verbose: bool = False,
    dynamic_kv_profile_rows: list[int] | None = None,
    force_decomposed_attention: bool = False,
    profile_mode: str = "dual_profile",
) -> bytes:
    """Build one half of a split native-KV Qwen-VL decoder.

    ``norm_type`` / ``mlp_type`` / ``position_type`` / ``activation`` /
    ``partial_rotary_factor`` / ``interleaved_rope`` / ``parallel_residual`` /
    ``scale_attn_weights`` mirror the same parameters on
    ``build_standard_decoder_engine``.

    ``opt_prefill_length`` and ``max_prefill_length`` bound the prefill
    optimization profile independently from the KV cache capacity.
    ``builder_workspace_bytes`` controls TensorRT tactic workspace.
    """
    _supports_config(config, weights)
    if profile_mode not in ("prefill", "decode"):
        raise ValueError(
            "native Qwen-VL profile_mode must be 'prefill' or 'decode', "
            f"got {profile_mode!r}")
    if precision != "bf16":
        raise ValueError("native Qwen-VL requires BF16")
    if quant_ctx is not None:
        raise ValueError("native Qwen-VL does not support quantization")
    if dynamic_kv_profile_rows:
        raise ValueError("native Qwen-VL uses one fixed physical KV capacity")
    if force_decomposed_attention:
        raise ValueError("native Qwen-VL attention must remain non-decomposable")
    if position_type != "rope":
        raise ValueError("native Qwen-VL requires mRoPE")

    if max_prefill_length is None:
        max_prefill_length = min(
            max_cache_length, _NATIVE_PREFILL_CHUNK_TOKENS)
    max_prefill_length = max(1, min(max_prefill_length, max_cache_length))
    opt_prefill_length = max(1, min(opt_prefill_length, max_prefill_length))

    from .build_routing import (
        native_mrope_settings,
        resolved_head_dim,
    )

    hidden = config.hidden_size
    vocab = config.vocab_size
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = resolved_head_dim(config)
    attention_size = num_heads * head_dim
    mlp_size = config.intermediate_size
    kv_attention_size = graph_blocks.infer_kv_attention_size(
        weights, num_kv_heads=num_kv_heads, head_dim=head_dim)
    rotary_embedding_dim = int(head_dim * partial_rotary_factor)
    lora_config = DynamicLoraConfig.from_model_config(config)
    if lora_config.enabled:
        raise ValueError("native Qwen-VL does not support dynamic LoRA")
    mrope_section, mrope_interleaved = native_mrope_settings(config)
    native_active_rope_inv_freq = graph_ops.make_native_active_rope_inv_freq(
        head_dim, config.rope_theta, partial_rotary_factor)

    builder_context = create_builder_context(
        verbose=verbose,
        workspace_bytes=builder_workspace_bytes,
    )
    builder = builder_context.builder
    network = builder_context.network
    trt_config = builder_context.config

    if precision == "fp16":
        work_np_dtype, work_trt_dtype = np.float16, trt.float16
    elif precision == "bf16":
        work_np_dtype, work_trt_dtype = np.float16, trt.bfloat16
    else:
        work_np_dtype, work_trt_dtype = np.float32, trt.float32

    # ---- Inputs (dynamic Sq) ---------------------------------------------
    token_id = network.add_input("token_id", trt.int32, (-1,))
    _position_id = network.add_input("position_id", trt.int32, (-1,))
    mrope_position_ids = network.add_input(
        "mrope_position_ids", trt.int32, (3, -1))
    cache_write_indices = network.add_input(
        "cache_write_indices", trt.int32, (1,))
    key_value_lengths = network.add_input(
        "key_value_lengths", trt.int32, (1,))
    input_embed_tensor = None
    use_input_embed_tensor = None
    if embed_input:
        input_embed_tensor = network.add_input(
            "input_embed", trt.float32, (-1, hidden))
        use_input_embed_tensor = network.add_input(
            "use_input_embed", trt.float32, (-1, 1))

    cache_shape = (1, num_kv_heads, max_cache_length, head_dim)
    cache_k_inputs: list[trt.ITensor] = []
    cache_v_inputs: list[trt.ITensor] = []
    for i in range(num_layers):
        ck = network.add_input(
            graph_ops.layer_tensor_name("cache_k", i),
            work_trt_dtype, cache_shape)
        cv = network.add_input(
            graph_ops.layer_tensor_name("cache_v", i),
            work_trt_dtype, cache_shape)
        cache_k_inputs.append(ck)
        cache_v_inputs.append(cv)

    # One optimization profile per split-engine role. Physical KV capacity is
    # static; only the active query length varies for prefill.
    def _add_profile(opt_sq: int, max_sq: int, *, fixed: bool = False):
        prof = builder.create_optimization_profile()
        min_sq = opt_sq if fixed else 1
        prof.set_shape("token_id", (min_sq,), (opt_sq,), (max_sq,))
        prof.set_shape("position_id", (min_sq,), (opt_sq,), (max_sq,))
        prof.set_shape(
            "mrope_position_ids",
            (3, min_sq), (3, opt_sq), (3, max_sq))
        if embed_input:
            prof.set_shape(
                "input_embed", (min_sq, hidden), (opt_sq, hidden), (max_sq, hidden))
            prof.set_shape(
                "use_input_embed", (min_sq, 1), (opt_sq, 1), (max_sq, 1))
        trt_config.add_optimization_profile(prof)

    if profile_mode == "prefill":
        _add_profile(opt_prefill_length, max_prefill_length, fixed=False)
    else:
        _add_profile(1, 1, fixed=True)

    # ---- Shared constants ------------------------------------------------
    embedding_table = _const_in_work_dtype(
        network, (vocab, hidden), weights["embedding"],
        work_np_dtype, work_trt_dtype)

    graph_ops.validate_native_rope_dim(rotary_embedding_dim)
    cos_half_table, sin_half_table = graph_ops.add_active_mrope_cache(
        network,
        mrope_position_ids,
        native_active_rope_inv_freq,
        mrope_section,
        work_trt_dtype,
        mrope_interleaved=mrope_interleaved,
    )

    eps_tensor = graph_ops.add_constant(
        network, (1, 1),
        np.array([[config.rms_norm_eps]], dtype=np.float32),
        dtype=np.float32)
    eps_tensor_per_head = graph_ops.add_constant(
        network, (1, 1, 1),
        np.array([[[config.rms_norm_eps]]], dtype=np.float32),
        dtype=np.float32)

    # Attention scale.
    attn_scale = (1.0 / np.sqrt(max(head_dim, 1))) if scale_attn_weights else 1.0

    # Quantization-aware matmul (passes weight_name through to QuantContext).
    matmul = _make_matmul_fn(
        network, work_np_dtype, quant_ctx, lora_config=lora_config)

    # ---- Embedding -------------------------------------------------------
    emb = network.add_gather(embedding_table, token_id, 0)
    hidden_state = emb.get_output(0)  # (Sq, hidden)

    if input_embed_tensor is not None and use_input_embed_tensor is not None:
        token_embed = hidden_state
        if token_embed.dtype != work_trt_dtype:
            token_embed = network.add_cast(token_embed, work_trt_dtype).get_output(0)
        input_embed = input_embed_tensor
        if input_embed.dtype != work_trt_dtype:
            input_embed = network.add_cast(input_embed, work_trt_dtype).get_output(0)
        embed_selector = use_input_embed_tensor
        if embed_selector.dtype != work_trt_dtype:
            embed_selector = network.add_cast(embed_selector, work_trt_dtype).get_output(0)
        one = _const_in_work_dtype(
            network, (1, 1), np.array([[1.0]], dtype=work_np_dtype),
            work_np_dtype, work_trt_dtype)
        inverse_selector = network.add_elementwise(
            one, embed_selector, trt.ElementWiseOperation.SUB).get_output(0)
        token_part = network.add_elementwise(
            inverse_selector, token_embed, trt.ElementWiseOperation.PROD).get_output(0)
        embed_part = network.add_elementwise(
            embed_selector, input_embed, trt.ElementWiseOperation.PROD).get_output(0)
        hidden_state = network.add_elementwise(
            token_part, embed_part, trt.ElementWiseOperation.SUM).get_output(0)

    # Make sure the main hidden stream is in the requested runtime dtype
    # before entering the layer stack (BF16 mode stores fp16 constants).
    if hidden_state.dtype != work_trt_dtype:
        hidden_state = network.add_cast(hidden_state, work_trt_dtype).get_output(0)

    # Optional embedding LayerNorm (Bloom).
    embed_norm = weights.get("embedding_norm")
    if embed_norm is not None:
        embed_norm_beta = weights.get(
            "embedding_norm_beta", np.zeros(hidden, dtype=np.float32))
        hidden_state = _norm_multi(
            network, hidden_state, hidden, embed_norm, embed_norm_beta,
            eps_tensor, "layernorm", work_np_dtype)

    present_k_outs: list[trt.ITensor] = []
    present_v_outs: list[trt.ITensor] = []

    for layer_idx in range(num_layers):
        prefix = f"layer.{layer_idx}"

        # Pre-attention norm.
        normed = _norm_multi(
            network, hidden_state, hidden,
            weights[f"{prefix}.input_norm"],
            weights.get(f"{prefix}.input_norm_beta"),
            eps_tensor, norm_type, work_np_dtype)

        # Q / K / V projections.
        q = matmul(normed, hidden, attention_size,
                   weights[f"{prefix}.w_q"], f"{prefix}.w_q")
        k = matmul(normed, hidden, kv_attention_size,
                   weights[f"{prefix}.w_k"], f"{prefix}.w_k")
        v = matmul(normed, hidden, kv_attention_size,
                   weights[f"{prefix}.w_v"], f"{prefix}.w_v")

        # Optional QKV biases (Qwen2 / GPT-2 / OPT / Bloom / Falcon / etc.).
        q_bias = weights.get(f"{prefix}.q_bias")
        if q_bias is not None:
            q = graph_ops.add_bias_sum(
                network, q, attention_size, q_bias, dtype=work_np_dtype)
        k_bias = weights.get(f"{prefix}.k_bias")
        if k_bias is not None:
            k = graph_ops.add_bias_sum(
                network, k, kv_attention_size, k_bias, dtype=work_np_dtype)
        v_bias = weights.get(f"{prefix}.v_bias")
        if v_bias is not None:
            v = graph_ops.add_bias_sum(
                network, v, kv_attention_size, v_bias, dtype=work_np_dtype)

        # Optional per-head q/k norm (Qwen3).
        q_norm = weights.get(f"{prefix}.q_norm")
        if q_norm is not None:
            q = graph_ops.add_rms_norm_per_head(
                network, q, num_heads, head_dim, q_norm,
                eps_tensor_per_head, dtype=work_np_dtype,
                sequence_length=None)
        k_norm = weights.get(f"{prefix}.k_norm")
        if k_norm is not None:
            k = graph_ops.add_rms_norm_per_head(
                network, k, num_kv_heads, head_dim, k_norm,
                eps_tensor_per_head, dtype=work_np_dtype,
                sequence_length=None)

        # Active mRoPE rows are already assembled in HF frequency order.
        q = graph_ops.add_apply_rope_native_sequence(
            network, q, num_heads, head_dim,
            cos_half_table, sin_half_table, rotary_embedding_dim,
            interleaved=False, sequence_length=None)
        k = graph_ops.add_apply_rope_native_sequence(
            network, k, num_kv_heads, head_dim,
            cos_half_table, sin_half_table, rotary_embedding_dim,
            interleaved=False, sequence_length=None)

        native_attention = graph_ops.add_native_kv_cache_attention_from_rows(
            network,
            q,
            k,
            v,
            cache_k_inputs[layer_idx],
            cache_v_inputs[layer_idx],
            cache_write_indices,
            key_value_lengths,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_seq=None,
            scale=attn_scale,
            tag=f"{prefix}.attn",
            recipe_instance=(
                f"decoder.layers.{layer_idx}.decode_attention"
                if profile_mode == "decode" else None),
        )
        context = native_attention["context"]
        present_k_outs.append(native_attention["present_k"])
        present_v_outs.append(native_attention["present_v"])

        attn_out = matmul(context, attention_size, hidden,
                          weights[f"{prefix}.w_o"], f"{prefix}.w_o")
        o_bias = weights.get(f"{prefix}.o_bias")
        if o_bias is not None:
            attn_out = graph_ops.add_bias_sum(
                network, attn_out, hidden, o_bias, dtype=work_np_dtype)

        # Residual structure: parallel (GPT-NeoX / CodeGen / Falcon-3) vs
        # sequential (everything else).
        if parallel_residual:
            post_attn_norm_w = weights.get(f"{prefix}.post_attn_norm")
            if post_attn_norm_w is not None:
                norm2 = _norm_multi(
                    network, hidden_state, hidden,
                    post_attn_norm_w,
                    weights.get(f"{prefix}.post_attn_norm_beta"),
                    eps_tensor, norm_type, work_np_dtype)
            else:
                norm2 = normed
        else:
            residual1 = network.add_elementwise(
                hidden_state, attn_out, trt.ElementWiseOperation.SUM)
            norm2 = _norm_multi(
                network, residual1.get_output(0), hidden,
                weights[f"{prefix}.post_attn_norm"],
                weights.get(f"{prefix}.post_attn_norm_beta"),
                eps_tensor, norm_type, work_np_dtype)

        # MLP — SwiGLU (Llama-style) or GeluFC (GPT-2-style).
        if mlp_type == "gelu_fc":
            mlp_out = _gelu_fc_mlp(
                network, norm2,
                matmul=matmul, weights=weights, prefix=prefix,
                hidden=hidden, mlp_size=mlp_size,
                activation=activation, work_np_dtype=work_np_dtype)
        else:
            mlp_out = _swiglu_mlp(
                network, norm2,
                matmul=matmul, weights=weights, prefix=prefix,
                hidden=hidden, mlp_size=mlp_size)

        # Final residual.
        if parallel_residual:
            sum_attn = network.add_elementwise(
                hidden_state, attn_out, trt.ElementWiseOperation.SUM)
            residual2 = network.add_elementwise(
                sum_attn.get_output(0), mlp_out, trt.ElementWiseOperation.SUM)
        else:
            residual2 = network.add_elementwise(
                residual1.get_output(0), mlp_out, trt.ElementWiseOperation.SUM)
        hidden_state = residual2.get_output(0)

    # ---- Final norm + LM head -------------------------------------------
    final_norm = weights.get("final_norm")
    if final_norm is not None and len(final_norm) > 0:
        hidden_state = _norm_multi(
            network, hidden_state, hidden, final_norm,
            weights.get("final_norm_beta"),
            eps_tensor, norm_type, work_np_dtype)

    # Only the LAST prompt token's logits matter for the next-token sample,
    # so slice hidden_state from (Sq, hidden) to (1, hidden) before the LM
    # head. This keeps the output contract identical to the single-token
    # engine (logits shape = (1, vocab)) under both profiles and avoids
    # computing (Sq - 1) redundant vocab-sized matmul rows during prefill.
    shape_t = network.add_shape(hidden_state).get_output(0)  # [2] int64
    one_hidden = graph_ops.add_constant(
        network, (2,), np.array([1, hidden], dtype=np.int64), dtype=np.int64)
    start_sub = network.add_elementwise(
        shape_t, one_hidden, trt.ElementWiseOperation.SUB)
    start_t = start_sub.get_output(0)  # [Sq - 1, 0]
    size_t = graph_ops.add_constant(
        network, (2,), np.array([1, hidden], dtype=np.int64), dtype=np.int64)
    slicer = network.add_slice(hidden_state, start=(0, 0), shape=(0, 0), stride=(1, 1))
    slicer.set_input(1, start_t)
    slicer.set_input(2, size_t)
    last_hidden = slicer.get_output(0)

    out_vocab = (weights["w_out"].shape[1]
                 if isinstance(weights["w_out"], np.ndarray) else vocab)
    logits = graph_ops.add_matmul_rhs_constant(
        network, last_hidden, hidden, out_vocab, weights["w_out"],
        dtype=work_np_dtype)
    lm_bias = weights.get("lm_head_bias")
    if lm_bias is not None:
        logits = graph_ops.add_bias_sum(
            network, logits, out_vocab, lm_bias, dtype=work_np_dtype)
    else:
        zero_bias = np.zeros(out_vocab, dtype=work_np_dtype)
        logits = graph_ops.add_bias_sum(
            network, logits, out_vocab, zero_bias, dtype=work_np_dtype)

    if work_trt_dtype != trt.float32:
        logits = network.add_cast(logits, trt.float32).get_output(0)
    logits.name = "logits"
    network.mark_output(logits)

    for i in range(num_layers):
        pk = present_k_outs[i]
        pv = present_v_outs[i]
        pk.name = graph_ops.layer_tensor_name("present_k", i)
        pv.name = graph_ops.layer_tensor_name("present_v", i)
        network.mark_output(pk)
        network.mark_output(pv)

    if verbose:
        mode_label = {
            "prefill": "prefill-profile",
            "decode": "decode-profile",
            "dual_profile": "dual-profile",
        }[profile_mode]
        print(f"[trtmc build] Building {mode_label} engine "
              f"(layers={num_layers}, hidden={hidden}, attn={attention_size}, "
              f"kv={kv_attention_size}, "
              f"mlp={mlp_size}, cache={max_cache_length}, "
              f"opt_prefill={opt_prefill_length}, max_prefill={max_prefill_length}, "
              f"norm={norm_type}, mlp_type={mlp_type}, pos={position_type}, "
              f"precision={precision}) ...",
              file=sys.stderr)

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("dual-profile decoder engine build failed")
    return bytes(plan)
