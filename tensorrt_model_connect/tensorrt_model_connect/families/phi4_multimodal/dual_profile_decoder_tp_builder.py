"""Tensor-parallel dual-profile decoder engine builder for Phi-4-multimodal.

Produces one rank-local TensorRT engine that handles both prefill
(multi-token) and decode (single-token) phases by switching between two
optimization profiles at runtime:
  * Profile 0 (prefill): Sq ranges over [1, opt_prefill_length, max_prefill_length].
  * Profile 1 (decode):  Sq fixed to 1.

Scope: Phi-4-multimodal text decoder only. Hardcodes the Phi-4 recipe —
RMSNorm, SwiGLU MLP, **partial RoPE (`partial_rotary_factor=0.75`)**,
GQA, sequential residual, no biases (LoRA adapters are ignored at load
time), no q/k norms, tied or untied LM head. Variants Phi-4 doesn't use
(LayerNorm, gelu_fc, full RoPE override, parallel residual, ALiBi /
learned position) are intentionally absent.

Tensor-parallel shape contract: rank-local widths only (Q/K/V column-
sharded, w_o / w_down row-sharded). Each row-parallel join inserts a
TRT 11.0+ ``IDistCollectiveLayer`` ALL_REDUCE SUM. Vision encoder
stays single-device — only the text decoder is rank-local.

I/O contract matches families/llama/dual_profile_decoder_tp_builder.py.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
from tensorrt_model_connect import trt_compat

from ... import graph_ops
from ... import graph_blocks
from ...parallel_config import (
    add_all_reduce_sum,
    normalize_parallel_config,
    shard_standard_decoder_weights,
)

trt = trt_compat.get_trt()

if TYPE_CHECKING:
    from ...config import ModelConfig
    from ...checkpoint_mapper import WeightDict


def _const_in_work_dtype(
    network: trt.INetworkDefinition,
    shape: tuple,
    values: np.ndarray,
    work_np_dtype: np.dtype,
    work_trt_dtype: trt.DataType,
) -> trt.ITensor:
    const = graph_ops.add_constant(network, shape, values, dtype=work_np_dtype)
    if const.dtype != work_trt_dtype:
        const = network.add_cast(const, work_trt_dtype).get_output(0)
    return const


def _rmsnorm(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden: int,
    gamma: np.ndarray,
    eps_tensor: trt.ITensor,
    dtype: np.dtype,
) -> trt.ITensor:
    return graph_ops.add_rms_norm(network, inp, hidden, gamma, eps_tensor, dtype=dtype)


def _swiglu_mlp(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    *,
    weights: "WeightDict",
    prefix: str,
    hidden: int,
    mlp_size: int,
    work_np_dtype: np.dtype,
) -> trt.ITensor:
    gate = graph_ops.add_matmul_rhs_constant(
        network, inp, hidden, mlp_size,
        weights[f"{prefix}.w_gate"], dtype=work_np_dtype)
    up = graph_ops.add_matmul_rhs_constant(
        network, inp, hidden, mlp_size,
        weights[f"{prefix}.w_up"], dtype=work_np_dtype)
    sigmoid = network.add_activation(gate, trt.ActivationType.SIGMOID)
    swish = network.add_elementwise(
        gate, sigmoid.get_output(0), trt.ElementWiseOperation.PROD)
    gated = network.add_elementwise(
        swish.get_output(0), up, trt.ElementWiseOperation.PROD)
    return graph_ops.add_matmul_rhs_constant(
        network, gated.get_output(0), mlp_size, hidden,
        weights[f"{prefix}.w_down"], dtype=work_np_dtype)


def _supports_config(config: "ModelConfig", weights: "WeightDict") -> None:
    if "embedding" not in weights:
        raise NotImplementedError("missing embedding weight")
    if "final_norm" not in weights:
        raise NotImplementedError("missing final_norm weight")


def build_dual_profile_tp_decoder_engine(
    config: "ModelConfig",
    weights: "WeightDict",
    max_cache_length: int,
    *,
    precision: str = "fp16",
    opt_prefill_length: int = 64,
    max_prefill_length: int | None = None,
    quant_ctx=None,
    verbose: bool = False,
    parallel_config=None,
) -> bytes:
    """Build a rank-local TP engine for the Phi-4-multimodal text decoder.

    The caller invokes this builder once per rank; ``engine_builder.py``
    packages the rank-local plans into one bundle under section names
    ``engine_plan_tp_rank<rank>``. Supports tp_size 2 and 4 (Phi-4-multimodal
    has 24 attention heads — not divisible by 8). Quantization is rejected
    (TP+quant is a follow-up).
    """
    _supports_config(config, weights)
    parallel = normalize_parallel_config(parallel_config)
    if not parallel.enabled:
        raise ValueError(
            "dual_profile_decoder_tp_builder requires parallel.mode=tensor_parallel "
            "and tp_size > 1")
    if quant_ctx is not None:
        raise ValueError(
            "Tensor-parallel Phi-4-multimodal builds do not support quantization yet")

    weights = shard_standard_decoder_weights(config, weights, parallel)

    if max_prefill_length is None:
        max_prefill_length = max_cache_length
    max_prefill_length = max(1, min(max_prefill_length, max_cache_length))
    opt_prefill_length = max(1, min(opt_prefill_length, max_prefill_length))

    attention_size = weights.get("_attention_size", config.attention_size)
    mlp_size = weights.get("_mlp_size", config.intermediate_size)
    hidden = config.hidden_size
    vocab = config.vocab_size
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads // parallel.tp_size
    num_kv_heads = config.num_key_value_heads // parallel.tp_size
    head_dim = attention_size // num_heads
    kv_attention_size = graph_blocks.infer_kv_attention_size(
        weights, num_kv_heads=num_kv_heads, head_dim=head_dim)
    partial_rotary_factor = float(config.raw.get("partial_rotary_factor", 0.75))
    rotary_embedding_dim = int(head_dim * partial_rotary_factor)

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    trt_config = builder.create_builder_config()
    trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    if precision == "fp16":
        work_np_dtype, work_trt_dtype = np.float16, trt.float16
    elif precision == "bf16":
        work_np_dtype, work_trt_dtype = np.float16, trt.bfloat16
    else:
        work_np_dtype, work_trt_dtype = np.float32, trt.float32

    # ---- Inputs --------------------------------------------------------
    token_id = network.add_input("token_id", trt.int32, (-1,))
    position_id = network.add_input("position_id", trt.int32, (-1,))
    attention_mask = network.add_input("attention_mask", trt.float32, (-1, -1))

    cache_shape = (max_cache_length, kv_attention_size)
    cache_k_inputs: list[trt.ITensor] = []
    cache_v_inputs: list[trt.ITensor] = []
    for i in range(num_layers):
        cache_k_inputs.append(network.add_input(
            graph_ops.layer_tensor_name("cache_k", i), work_trt_dtype, cache_shape))
        cache_v_inputs.append(network.add_input(
            graph_ops.layer_tensor_name("cache_v", i), work_trt_dtype, cache_shape))

    if work_trt_dtype != trt.float32:
        attention_mask_work = network.add_cast(
            attention_mask, work_trt_dtype).get_output(0)
    else:
        attention_mask_work = attention_mask

    def _add_profile(opt_sq: int, max_sq: int, *, fixed: bool):
        prof = builder.create_optimization_profile()
        min_sq = opt_sq if fixed else 1
        prof.set_shape("token_id", (min_sq,), (opt_sq,), (max_sq,))
        prof.set_shape("position_id", (min_sq,), (opt_sq,), (max_sq,))
        prof.set_shape(
            "attention_mask",
            (min_sq, max_cache_length + min_sq),
            (opt_sq, max_cache_length + opt_sq),
            (max_sq, max_cache_length + max_sq))
        trt_config.add_optimization_profile(prof)

    _add_profile(opt_prefill_length, max_prefill_length, fixed=False)  # prefill
    _add_profile(1, 1, fixed=True)                                     # decode

    # ---- Embedding + RoPE tables --------------------------------------
    embedding_table = _const_in_work_dtype(
        network, (vocab, hidden), weights["embedding"],
        work_np_dtype, work_trt_dtype)

    kmax = max_cache_length + max_prefill_length
    graph_ops.validate_native_rope_dim(rotary_embedding_dim)
    cos_half_np = graph_ops.make_rope_table_half_dim(
        kmax, head_dim, config.rope_theta, True,
        partial_rotary_factor, interleaved=False)
    sin_half_np = graph_ops.make_rope_table_half_dim(
        kmax, head_dim, config.rope_theta, False,
        partial_rotary_factor, interleaved=False)
    cos_half_table = _const_in_work_dtype(
        network, cos_half_np.shape, cos_half_np, work_np_dtype, work_trt_dtype)
    sin_half_table = _const_in_work_dtype(
        network, sin_half_np.shape, sin_half_np, work_np_dtype, work_trt_dtype)

    eps_tensor = graph_ops.add_constant(
        network, (1, 1),
        np.array([[config.rms_norm_eps]], dtype=np.float32), dtype=np.float32)

    attn_scale = 1.0 / np.sqrt(max(head_dim, 1))

    # ---- Forward pass --------------------------------------------------
    emb = network.add_gather(embedding_table, token_id, 0)
    hidden_state = emb.get_output(0)
    if hidden_state.dtype != work_trt_dtype:
        hidden_state = network.add_cast(hidden_state, work_trt_dtype).get_output(0)

    mask_4d = graph_ops.add_2d_mask_to_4d(network, attention_mask_work)

    present_k_outs: list[trt.ITensor] = []
    present_v_outs: list[trt.ITensor] = []

    for layer_idx in range(num_layers):
        prefix = f"layer.{layer_idx}"

        # Pre-attention RMSNorm.
        normed = _rmsnorm(
            network, hidden_state, hidden,
            weights[f"{prefix}.input_norm"], eps_tensor, work_np_dtype)

        # Column-sharded Q / K / V projections (no biases).
        q = graph_ops.add_matmul_rhs_constant(
            network, normed, hidden, attention_size,
            weights[f"{prefix}.w_q"], dtype=work_np_dtype)
        k = graph_ops.add_matmul_rhs_constant(
            network, normed, hidden, kv_attention_size,
            weights[f"{prefix}.w_k"], dtype=work_np_dtype)
        v = graph_ops.add_matmul_rhs_constant(
            network, normed, hidden, kv_attention_size,
            weights[f"{prefix}.w_v"], dtype=work_np_dtype)

        # Partial RoPE on Q and K (rotary_embedding_dim = head_dim * 0.75).
        q = graph_ops.add_apply_rope_native(
            network, q, num_heads, head_dim,
            cos_half_table, sin_half_table, position_id,
            rotary_embedding_dim, False, sequence_length=None)
        k = graph_ops.add_apply_rope_native(
            network, k, num_kv_heads, head_dim,
            cos_half_table, sin_half_table, position_id,
            rotary_embedding_dim, False, sequence_length=None)

        present_k_outs.append(k)
        present_v_outs.append(v)

        all_k = network.add_concatenation([cache_k_inputs[layer_idx], k])
        all_k.axis = 0
        all_v = network.add_concatenation([cache_v_inputs[layer_idx], v])
        all_v.axis = 0

        context = graph_ops.add_attention_from_rows(
            network, q, all_k.get_output(0), all_v.get_output(0),
            num_heads=num_heads, head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            q_seq=None, kv_seq=None, causal=False, mask=mask_4d,
            scale=attn_scale, tag=f"{prefix}.attn")

        # Row-sharded W_O + ALL_REDUCE join.
        attn_out = graph_ops.add_matmul_rhs_constant(
            network, context, attention_size, hidden,
            weights[f"{prefix}.w_o"], dtype=work_np_dtype)
        attn_out = add_all_reduce_sum(network, attn_out, parallel.tp_size)

        # Sequential residual + post-attention RMSNorm.
        residual1 = network.add_elementwise(
            hidden_state, attn_out, trt.ElementWiseOperation.SUM)
        norm2 = _rmsnorm(
            network, residual1.get_output(0), hidden,
            weights[f"{prefix}.post_attn_norm"], eps_tensor, work_np_dtype)

        # SwiGLU MLP (column-sharded gate/up, row-sharded down).
        mlp_out = _swiglu_mlp(
            network, norm2, weights=weights, prefix=prefix,
            hidden=hidden, mlp_size=mlp_size, work_np_dtype=work_np_dtype)
        mlp_out = add_all_reduce_sum(network, mlp_out, parallel.tp_size)

        residual2 = network.add_elementwise(
            residual1.get_output(0), mlp_out, trt.ElementWiseOperation.SUM)
        hidden_state = residual2.get_output(0)

    # ---- Final norm + LM head -----------------------------------------
    hidden_state = _rmsnorm(
        network, hidden_state, hidden, weights["final_norm"],
        eps_tensor, work_np_dtype)

    # Slice the last token's hidden state before the LM head.
    shape_t = network.add_shape(hidden_state).get_output(0)
    one_hidden = graph_ops.add_constant(
        network, (2,), np.array([1, hidden], dtype=np.int64), dtype=np.int64)
    start_sub = network.add_elementwise(
        shape_t, one_hidden, trt.ElementWiseOperation.SUB)
    start_t = start_sub.get_output(0)
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
        print(f"[trtmc-build] Building Phi-4-multimodal TP decoder engine "
              f"(layers={num_layers}, hidden={hidden}, attn={attention_size}, "
              f"kv={kv_attention_size}, mlp={mlp_size}, cache={max_cache_length}, "
              f"opt_prefill={opt_prefill_length}, max_prefill={max_prefill_length}, "
              f"partial_rotary={partial_rotary_factor}, "
              f"precision={precision}, tp={parallel.tp_size}, rank={parallel.rank}) ...",
              file=sys.stderr)

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("Phi-4-multimodal tensor-parallel engine build failed")
    return bytes(plan)
