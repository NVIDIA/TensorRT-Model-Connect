# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RWKV family model -- Linear Attention / Recurrent model.

RWKV replaces transformer attention with a linear attention WKV mechanism
that operates recurrently at inference time. Each layer has two blocks:
  1. Time-Mixing (attention replacement):
     time-shift blending + R/K/V projections + WKV recurrence + sigmoid
     gating + output projection
  2. Channel-Mixing (FFN replacement):
     time-shift blending + key projection + squared ReLU + receptance
     gating + value projection

No attention mask, no position IDs, no KV cache -- pure recurrent with
5 state tensors per layer:
  - attn_state:  [hidden_size]  previous token hidden for time-mixing shift
  - ff_state:    [hidden_size]  previous token hidden for channel-mixing shift
  - num_state:   [hidden_size]  WKV numerator accumulator
  - den_state:   [hidden_size]  WKV denominator accumulator
  - max_state:   [hidden_size]  WKV max value tracker (numerical stability)

Weight key mapping (HF -> canonical):
  rwkv.embeddings.weight              -> embedding [vocab, hidden]
  rwkv.blocks.{i}.ln1.weight          -> layer.{i}.attn_norm [hidden]
  rwkv.blocks.{i}.ln1.bias            -> layer.{i}.attn_norm_beta [hidden]
  rwkv.blocks.{i}.ln2.weight          -> layer.{i}.ffn_norm [hidden]
  rwkv.blocks.{i}.ln2.bias            -> layer.{i}.ffn_norm_beta [hidden]
  rwkv.blocks.{i}.attention.time_decay -> layer.{i}.time_decay [hidden]
  rwkv.blocks.{i}.attention.time_first -> layer.{i}.time_first [hidden]
  rwkv.blocks.{i}.attention.time_mix_key -> layer.{i}.time_mix_key [hidden]
  rwkv.blocks.{i}.attention.time_mix_value -> layer.{i}.time_mix_value [hidden]
  rwkv.blocks.{i}.attention.time_mix_receptance -> layer.{i}.time_mix_receptance [hidden]
  rwkv.blocks.{i}.attention.key.weight -> layer.{i}.w_attn_k [hidden, hidden]
  rwkv.blocks.{i}.attention.value.weight -> layer.{i}.w_attn_v [hidden, hidden]
  rwkv.blocks.{i}.attention.receptance.weight -> layer.{i}.w_attn_r [hidden, hidden]
  rwkv.blocks.{i}.attention.output.weight -> layer.{i}.w_attn_o [hidden, hidden]
  rwkv.blocks.{i}.feed_forward.time_mix_key -> layer.{i}.time_mix_ffn_key [hidden]
  rwkv.blocks.{i}.feed_forward.time_mix_receptance -> layer.{i}.time_mix_ffn_receptance [hidden]
  rwkv.blocks.{i}.feed_forward.key.weight -> layer.{i}.w_ffn_k [intermediate, hidden]
  rwkv.blocks.{i}.feed_forward.value.weight -> layer.{i}.w_ffn_v [hidden, intermediate]
  rwkv.blocks.{i}.feed_forward.receptance.weight -> layer.{i}.w_ffn_r [hidden, hidden]
  rwkv.ln_out.weight                   -> final_norm [hidden]
  rwkv.ln_out.bias                     -> final_norm_beta [hidden]
  head.weight (or tied)                -> w_lm_head [hidden, vocab]
"""

from __future__ import annotations

import json
import re
import tempfile
import time

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
from ...parallel_config import (
    normalize_parallel_config,
    require_tensorrt_11_for_tensor_parallel,
)


trt = trt_compat.get_trt()

name = "rwkv"
runtime_strategy = "rwkv_recurrent"


def matches(config) -> bool:
    """Return whether this module owns the resolved model config."""
    model_type = getattr(config, "model_type", config)
    model_type = str(model_type)
    return model_type.lower() == "rwkv"


def load_weights(model_dir: str, config: ModelConfig) -> WeightDict:
    """Load RWKV weights from safetensors."""
    model_dir_path = Path(model_dir)
    readers = _open_safetensors(model_dir_path)

    hidden = config.hidden_size
    vocab = config.vocab_size
    num_layers = config.num_hidden_layers
    intermediate = config.raw.get("intermediate_size", hidden * 4)

    weights = WeightDict()

    # Detect HF key prefix: "rwkv." (transformers) or "backbone." (older)
    if _has_tensor(readers, "rwkv.embeddings.weight"):
        embed_key = "rwkv.embeddings.weight"
        block_prefix = "rwkv.blocks"
        attn_key = "attention"
        ffn_key = "feed_forward"
        final_norm_key = "rwkv.ln_out.weight"
        final_norm_bias_key = "rwkv.ln_out.bias"
    elif _has_tensor(readers, "backbone.embeddings.weight"):
        embed_key = "backbone.embeddings.weight"
        block_prefix = "backbone.blocks"
        attn_key = "att"
        ffn_key = "ffn"
        final_norm_key = "backbone.norm_f.weight"
        final_norm_bias_key = "backbone.norm_f.bias"
    else:
        raise ValueError(
            "Cannot detect RWKV weight key prefix. Expected "
            "'rwkv.embeddings.weight' or 'backbone.embeddings.weight'"
        )

    # Embedding
    embedding = _load_tensor(readers, embed_key)
    assert embedding.shape == (vocab, hidden), (
        f"Embedding shape {embedding.shape} != ({vocab}, {hidden})"
    )
    weights["embedding"] = embedding.astype(np.float32)

    # Pre-layer norm (block 0 only in some RWKV-4 variants)
    pre_ln_key = f"{block_prefix}.0.pre_ln.weight"
    if _has_tensor(readers, pre_ln_key):
        weights["pre_ln_weight"] = _load_tensor(readers, pre_ln_key).astype(np.float32)
        weights["pre_ln_bias"] = _load_tensor(readers, f"{block_prefix}.0.pre_ln.bias").astype(
            np.float32
        )

    for layer_idx in range(num_layers):
        prefix = f"layer.{layer_idx}"
        hf_block = f"{block_prefix}.{layer_idx}"

        # LayerNorm 1 (before time-mixing)
        weights[f"{prefix}.attn_norm"] = _load_tensor(readers, f"{hf_block}.ln1.weight").astype(
            np.float32
        )
        weights[f"{prefix}.attn_norm_beta"] = _load_tensor(readers, f"{hf_block}.ln1.bias").astype(
            np.float32
        )

        # LayerNorm 2 (before channel-mixing)
        weights[f"{prefix}.ffn_norm"] = _load_tensor(readers, f"{hf_block}.ln2.weight").astype(
            np.float32
        )
        weights[f"{prefix}.ffn_norm_beta"] = _load_tensor(readers, f"{hf_block}.ln2.bias").astype(
            np.float32
        )

        # Time-mixing parameters
        # HF stores time_decay in log-space; the model applies -exp(time_decay)
        raw_time_decay = _load_tensor(readers, f"{hf_block}.{attn_key}.time_decay").astype(
            np.float32
        )
        weights[f"{prefix}.time_decay"] = -np.exp(raw_time_decay)
        weights[f"{prefix}.time_first"] = _load_tensor(
            readers, f"{hf_block}.{attn_key}.time_first"
        ).astype(np.float32)
        weights[f"{prefix}.time_mix_key"] = _load_tensor(
            readers, f"{hf_block}.{attn_key}.time_mix_key"
        ).astype(np.float32)
        weights[f"{prefix}.time_mix_value"] = _load_tensor(
            readers, f"{hf_block}.{attn_key}.time_mix_value"
        ).astype(np.float32)
        weights[f"{prefix}.time_mix_receptance"] = _load_tensor(
            readers, f"{hf_block}.{attn_key}.time_mix_receptance"
        ).astype(np.float32)

        # Attention projections (transpose for TRT matmul)
        weights[f"{prefix}.w_attn_k"] = _transpose_2d(
            _load_tensor(readers, f"{hf_block}.{attn_key}.key.weight"), "attn_k"
        )
        weights[f"{prefix}.w_attn_v"] = _transpose_2d(
            _load_tensor(readers, f"{hf_block}.{attn_key}.value.weight"), "attn_v"
        )
        weights[f"{prefix}.w_attn_r"] = _transpose_2d(
            _load_tensor(readers, f"{hf_block}.{attn_key}.receptance.weight"), "attn_r"
        )
        weights[f"{prefix}.w_attn_o"] = _transpose_2d(
            _load_tensor(readers, f"{hf_block}.{attn_key}.output.weight"), "attn_o"
        )

        # Channel-mixing (FFN) parameters
        weights[f"{prefix}.time_mix_ffn_key"] = _load_tensor(
            readers, f"{hf_block}.{ffn_key}.time_mix_key"
        ).astype(np.float32)
        weights[f"{prefix}.time_mix_ffn_receptance"] = _load_tensor(
            readers, f"{hf_block}.{ffn_key}.time_mix_receptance"
        ).astype(np.float32)

        # FFN projections (transpose for TRT matmul)
        weights[f"{prefix}.w_ffn_k"] = _transpose_2d(
            _load_tensor(readers, f"{hf_block}.{ffn_key}.key.weight"), "ffn_k"
        )
        weights[f"{prefix}.w_ffn_v"] = _transpose_2d(
            _load_tensor(readers, f"{hf_block}.{ffn_key}.value.weight"), "ffn_v"
        )
        weights[f"{prefix}.w_ffn_r"] = _transpose_2d(
            _load_tensor(readers, f"{hf_block}.{ffn_key}.receptance.weight"), "ffn_r"
        )

    # Final LayerNorm
    if _has_tensor(readers, final_norm_key):
        weights["final_norm"] = _load_tensor(readers, final_norm_key).astype(np.float32)
    else:
        weights["final_norm"] = np.ones(hidden, dtype=np.float32)

    if _has_tensor(readers, final_norm_bias_key):
        weights["final_norm_beta"] = _load_tensor(readers, final_norm_bias_key).astype(np.float32)
    else:
        weights["final_norm_beta"] = np.zeros(hidden, dtype=np.float32)

    # LM head (may be tied to embeddings)
    lm_head_key = "head.weight"
    if _has_tensor(readers, lm_head_key):
        weights["w_lm_head"] = _transpose_2d(_load_tensor(readers, lm_head_key), "lm_head")
    else:
        # Tied embeddings: [vocab, hidden] -> [hidden, vocab]
        weights["w_lm_head"] = _transpose_2d(embedding.copy(), "embedding_tied")

    # Store RWKV-specific dimensions for engine builder
    weights["_intermediate_size"] = intermediate  # type: ignore[assignment]

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
    """Build TRT engine for RWKV linear attention.

    max_cache_length is accepted for API compatibility but is not used
    by RWKV (recurrent state is constant size regardless of sequence length).

    Engine inputs:
      token_id: int32 [1]
      attn_state_0..N: float32 [1, hidden_size]
      ff_state_0..N: float32 [1, hidden_size]
      num_state_0..N: float32 [1, hidden_size]
      den_state_0..N: float32 [1, hidden_size]
      max_state_0..N: float32 [1, hidden_size]

    Engine outputs:
      logits: float32 [1, vocab]
      present_attn_0..N: float32 [1, hidden_size]
      present_ff_0..N: float32 [1, hidden_size]
      present_num_0..N: float32 [1, hidden_size]
      present_den_0..N: float32 [1, hidden_size]
      present_max_0..N: float32 [1, hidden_size]
    """
    parallel = normalize_parallel_config(parallel_config)
    if parallel.enabled:
        require_tensorrt_11_for_tensor_parallel(parallel, feature="RWKV tensor-parallel builds")
        from .tp_builder import build_rwkv_tp_engine

        return build_rwkv_tp_engine(
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
    intermediate: int = weights["_intermediate_size"]
    if precision == "fp16":
        work_np_dtype, work_trt_dtype = np.float16, trt.float16
    elif precision == "fp32":
        work_np_dtype, work_trt_dtype = np.float32, trt.float32
    else:
        raise ValueError(f"RWKV supports precision='fp32' or 'fp16', got {precision!r}")

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    trt_config = builder.create_builder_config()
    trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    # -----------------------------------------------------------
    # Inputs
    # -----------------------------------------------------------
    token_id = network.add_input("token_id", trt.int32, (1,))

    attn_state_inputs = []
    ff_state_inputs = []
    num_state_inputs = []
    den_state_inputs = []
    max_state_inputs = []
    for i in range(num_layers):
        attn_s = network.add_input(
            graph_ops.layer_tensor_name("attn_state", i), trt.float32, (1, hidden)
        )
        ff_s = network.add_input(
            graph_ops.layer_tensor_name("ff_state", i), trt.float32, (1, hidden)
        )
        num_s = network.add_input(
            graph_ops.layer_tensor_name("num_state", i), trt.float32, (1, hidden)
        )
        den_s = network.add_input(
            graph_ops.layer_tensor_name("den_state", i), trt.float32, (1, hidden)
        )
        max_s = network.add_input(
            graph_ops.layer_tensor_name("max_state", i), trt.float32, (1, hidden)
        )
        attn_state_inputs.append(attn_s)
        ff_state_inputs.append(ff_s)
        num_state_inputs.append(num_s)
        den_state_inputs.append(den_s)
        max_state_inputs.append(max_s)

    # -----------------------------------------------------------
    # Shared constants
    # -----------------------------------------------------------
    embedding_table = graph_ops.add_constant(
        network, (vocab, hidden), weights["embedding"], dtype=work_np_dtype
    )

    eps_tensor = graph_ops.add_constant(
        network, (1, 1), np.array([config.rms_norm_eps], dtype=work_np_dtype), dtype=work_np_dtype
    )

    one_const = graph_ops.add_constant(
        network, (1, 1), np.array([1.0], dtype=work_np_dtype), dtype=work_np_dtype
    )

    # -----------------------------------------------------------
    # Embedding lookup
    # -----------------------------------------------------------
    gather = network.add_gather(embedding_table, token_id, 0)
    hidden_state = gather.get_output(0)  # [1, hidden]
    if hidden_state.dtype != work_trt_dtype:
        hidden_state = network.add_cast(hidden_state, work_trt_dtype).get_output(0)

    # Optional pre-layer norm (some RWKV-4 variants)
    if "pre_ln_weight" in weights:
        hidden_state = graph_ops.add_layer_norm(
            network,
            hidden_state,
            hidden,
            weights["pre_ln_weight"],
            weights["pre_ln_bias"],
            eps_tensor,
            dtype=work_np_dtype,
        )

    if debug_layer_outputs:
        _mark_debug_output(network, hidden_state, "debug_embed")

    # -----------------------------------------------------------
    # RWKV layers
    # -----------------------------------------------------------
    present_attn_outputs = []
    present_ff_outputs = []
    present_num_outputs = []
    present_den_outputs = []
    present_max_outputs = []

    for layer_idx in range(num_layers):
        prefix = f"layer.{layer_idx}"
        attn_state = attn_state_inputs[layer_idx]
        ff_state = ff_state_inputs[layer_idx]
        if attn_state.dtype != work_trt_dtype:
            attn_state = network.add_cast(attn_state, work_trt_dtype).get_output(0)
        if ff_state.dtype != work_trt_dtype:
            ff_state = network.add_cast(ff_state, work_trt_dtype).get_output(0)

        result = _add_rwkv_layer(
            network=network,
            hidden=hidden_state,
            attn_state_in=attn_state,
            ff_state_in=ff_state,
            num_state_in=num_state_inputs[layer_idx],
            den_state_in=den_state_inputs[layer_idx],
            max_state_in=max_state_inputs[layer_idx],
            eps_tensor=eps_tensor,
            one_const=one_const,
            weights=weights,
            prefix=prefix,
            hidden_size=hidden,
            intermediate_size=intermediate,
            dtype=work_np_dtype,
        )

        hidden_state = result["hidden"]
        present_attn_outputs.append(result["present_attn"])
        present_ff_outputs.append(result["present_ff"])
        present_num_outputs.append(result["present_num"])
        present_den_outputs.append(result["present_den"])
        present_max_outputs.append(result["present_max"])

        if debug_layer_outputs:
            _mark_debug_output(network, hidden_state, f"debug_hidden_{layer_idx}")

    # -----------------------------------------------------------
    # Final LayerNorm
    # -----------------------------------------------------------
    hidden_state = graph_ops.add_layer_norm(
        network,
        hidden_state,
        hidden,
        weights["final_norm"],
        weights["final_norm_beta"],
        eps_tensor,
        dtype=work_np_dtype,
    )

    # -----------------------------------------------------------
    # LM head (logits)
    # -----------------------------------------------------------
    logits = graph_ops.add_matmul_rhs_constant(
        network, hidden_state, hidden, vocab, weights["w_lm_head"], dtype=work_np_dtype
    )
    if logits.dtype != trt.float32:
        logits = network.add_cast(logits, trt.float32).get_output(0)

    logits.name = "logits"
    network.mark_output(logits)

    # -----------------------------------------------------------
    # Present state outputs
    # -----------------------------------------------------------
    for i in range(num_layers):
        pa = present_attn_outputs[i]
        pf = present_ff_outputs[i]
        pn = present_num_outputs[i]
        pd = present_den_outputs[i]
        pm = present_max_outputs[i]

        if pa.dtype != trt.float32:
            pa = network.add_cast(pa, trt.float32).get_output(0)
        if pf.dtype != trt.float32:
            pf = network.add_cast(pf, trt.float32).get_output(0)
        if pn.dtype != trt.float32:
            pn = network.add_cast(pn, trt.float32).get_output(0)
        if pd.dtype != trt.float32:
            pd = network.add_cast(pd, trt.float32).get_output(0)
        if pm.dtype != trt.float32:
            pm = network.add_cast(pm, trt.float32).get_output(0)

        pa.name = graph_ops.layer_tensor_name("present_attn", i)
        pf.name = graph_ops.layer_tensor_name("present_ff", i)
        pn.name = graph_ops.layer_tensor_name("present_num", i)
        pd.name = graph_ops.layer_tensor_name("present_den", i)
        pm.name = graph_ops.layer_tensor_name("present_max", i)

        network.mark_output(pa)
        network.mark_output(pf)
        network.mark_output(pn)
        network.mark_output(pd)
        network.mark_output(pm)

    # -----------------------------------------------------------
    # Build engine
    # -----------------------------------------------------------
    if verbose:
        print(
            f"[trtmc build] Building RWKV TRT engine ({num_layers} layers, "
            f"hidden={hidden}, intermediate={intermediate}, "
            f"precision={precision}) ...",
            file=sys.stderr,
        )

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("TensorRT engine build failed")

    return bytes(plan)


def _mark_debug_output(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    name: str,
) -> None:
    """Mark a tensor as a network output for debug inspection."""
    identity = network.add_identity(tensor)
    cast = network.add_cast(identity.get_output(0), trt.float32)
    out = cast.get_output(0)
    out.name = name
    network.mark_output(out)


def _add_rwkv_layer(
    *,
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    attn_state_in: trt.ITensor,
    ff_state_in: trt.ITensor,
    num_state_in: trt.ITensor,
    den_state_in: trt.ITensor,
    max_state_in: trt.ITensor,
    eps_tensor: trt.ITensor,
    one_const: trt.ITensor,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    intermediate_size: int,
    dtype: np.dtype = np.float32,
) -> dict[str, trt.ITensor]:
    """Add one RWKV layer (time-mixing + channel-mixing).

    Returns: {hidden, present_attn, present_ff, present_num, present_den, present_max}
    """
    # ================================================================
    # TIME-MIXING BLOCK
    # ================================================================

    # ===== 1. LayerNorm (with bias) =====
    normed_attn = graph_ops.add_layer_norm(
        network,
        hidden,
        hidden_size,
        weights[f"{prefix}.attn_norm"],
        weights[f"{prefix}.attn_norm_beta"],
        eps_tensor,
        dtype=dtype,
    )
    # normed_attn: [1, hidden_size]

    # The normed output before time-shift becomes present_attn for next step
    present_attn = normed_attn

    # ===== 2. Time-shift blending for R, K, V =====
    # For each of R, K, V: blended = mix * normed + (1 - mix) * attn_state_in

    def _time_shift_blend(normed, prev_state, mix_weights_key):
        """Element-wise lerp: mix * normed + (1 - mix) * prev_state."""
        mix = graph_ops.add_constant(
            network, (1, hidden_size), weights[mix_weights_key], dtype=dtype
        )
        one_minus_mix = network.add_elementwise(one_const, mix, trt.ElementWiseOperation.SUB)
        cur_part = network.add_elementwise(normed, mix, trt.ElementWiseOperation.PROD)
        prev_part = network.add_elementwise(
            prev_state, one_minus_mix.get_output(0), trt.ElementWiseOperation.PROD
        )
        blended = network.add_elementwise(
            cur_part.get_output(0), prev_part.get_output(0), trt.ElementWiseOperation.SUM
        )
        return blended.get_output(0)

    xk = _time_shift_blend(normed_attn, attn_state_in, f"{prefix}.time_mix_key")
    xv = _time_shift_blend(normed_attn, attn_state_in, f"{prefix}.time_mix_value")
    xr = _time_shift_blend(normed_attn, attn_state_in, f"{prefix}.time_mix_receptance")

    # ===== 3. Projections =====
    # R = sigmoid(xr @ w_attn_r)
    r_proj = graph_ops.add_matmul_rhs_constant(
        network, xr, hidden_size, hidden_size, weights[f"{prefix}.w_attn_r"], dtype=dtype
    )
    r_gate = network.add_activation(r_proj, trt.ActivationType.SIGMOID)

    # K = xk @ w_attn_k
    k_proj = graph_ops.add_matmul_rhs_constant(
        network, xk, hidden_size, hidden_size, weights[f"{prefix}.w_attn_k"], dtype=dtype
    )
    # K: [1, hidden_size]

    # V = xv @ w_attn_v
    v_proj = graph_ops.add_matmul_rhs_constant(
        network, xv, hidden_size, hidden_size, weights[f"{prefix}.w_attn_v"], dtype=dtype
    )
    # V: [1, hidden_size]

    # ===== 4. WKV recurrence (numerically stable) =====
    # The WKV max/exp accumulator remains FP32. It is recurrent across tokens,
    # and reducing it to FP16 compounds error even when projections are FP16.
    k_recur = k_proj
    v_recur = v_proj
    if k_recur.dtype != trt.float32:
        k_recur = network.add_cast(k_recur, trt.float32).get_output(0)
    if v_recur.dtype != trt.float32:
        v_recur = network.add_cast(v_recur, trt.float32).get_output(0)
    time_decay = graph_ops.add_constant(network, (1, hidden_size), weights[f"{prefix}.time_decay"])
    time_first = graph_ops.add_constant(network, (1, hidden_size), weights[f"{prefix}.time_first"])

    # decay_plus_max = max_state_in + time_decay
    decay_plus_max = network.add_elementwise(max_state_in, time_decay, trt.ElementWiseOperation.SUM)

    # tf_plus_k = time_first + K
    tf_plus_k = network.add_elementwise(time_first, k_recur, trt.ElementWiseOperation.SUM)

    # ---- WKV output (uses time_first as bonus, NO decay) ----
    # HF: max_for_output = max(max_state, key + time_first)
    # time_decay is ONLY used in the state update, not the output.
    q_out = network.add_elementwise(
        tf_plus_k.get_output(0), max_state_in, trt.ElementWiseOperation.MAX
    )

    # e2 = exp(key + time_first - q)
    tf_k_minus_q = network.add_elementwise(
        tf_plus_k.get_output(0), q_out.get_output(0), trt.ElementWiseOperation.SUB
    )
    exp_tf_k = network.add_unary(tf_k_minus_q.get_output(0), trt.UnaryOperation.EXP)

    # e1 = exp(max_state - q)  (NO time_decay here)
    ms_minus_q = network.add_elementwise(
        max_state_in, q_out.get_output(0), trt.ElementWiseOperation.SUB
    )
    exp_dpm = network.add_unary(ms_minus_q.get_output(0), trt.UnaryOperation.EXP)

    # wkv_num = exp(tf+k-q) * V + exp(dpm-q) * num_state
    term1_num = network.add_elementwise(
        exp_tf_k.get_output(0), v_recur, trt.ElementWiseOperation.PROD
    )
    term2_num = network.add_elementwise(
        exp_dpm.get_output(0), num_state_in, trt.ElementWiseOperation.PROD
    )
    wkv_num = network.add_elementwise(
        term1_num.get_output(0), term2_num.get_output(0), trt.ElementWiseOperation.SUM
    )

    # wkv_den = exp(tf+k-q) + exp(dpm-q) * den_state
    term2_den = network.add_elementwise(
        exp_dpm.get_output(0), den_state_in, trt.ElementWiseOperation.PROD
    )
    wkv_den = network.add_elementwise(
        exp_tf_k.get_output(0), term2_den.get_output(0), trt.ElementWiseOperation.SUM
    )

    # wkv = wkv_num / wkv_den
    wkv = network.add_elementwise(
        wkv_num.get_output(0), wkv_den.get_output(0), trt.ElementWiseOperation.DIV
    )

    # ---- State update (does NOT use time_first) ----
    # q2 = max(K, decay_plus_max)
    q2 = network.add_elementwise(
        k_recur, decay_plus_max.get_output(0), trt.ElementWiseOperation.MAX
    )

    # exp(K - q2)
    k_minus_q2 = network.add_elementwise(k_recur, q2.get_output(0), trt.ElementWiseOperation.SUB)
    exp_k_q2 = network.add_unary(k_minus_q2.get_output(0), trt.UnaryOperation.EXP)

    # exp(decay_plus_max - q2)
    dpm_minus_q2 = network.add_elementwise(
        decay_plus_max.get_output(0), q2.get_output(0), trt.ElementWiseOperation.SUB
    )
    exp_dpm_q2 = network.add_unary(dpm_minus_q2.get_output(0), trt.UnaryOperation.EXP)

    # present_num = exp(K-q2)*V + exp(dpm-q2)*num_state
    st_term1 = network.add_elementwise(
        exp_k_q2.get_output(0), v_recur, trt.ElementWiseOperation.PROD
    )
    st_term2 = network.add_elementwise(
        exp_dpm_q2.get_output(0), num_state_in, trt.ElementWiseOperation.PROD
    )
    present_num = network.add_elementwise(
        st_term1.get_output(0), st_term2.get_output(0), trt.ElementWiseOperation.SUM
    )

    # present_den = exp(K-q2) + exp(dpm-q2)*den_state
    st_den_term2 = network.add_elementwise(
        exp_dpm_q2.get_output(0), den_state_in, trt.ElementWiseOperation.PROD
    )
    present_den = network.add_elementwise(
        exp_k_q2.get_output(0), st_den_term2.get_output(0), trt.ElementWiseOperation.SUM
    )

    # present_max = q2
    present_max = q2.get_output(0)

    # ===== 5. Gated output + residual =====
    wkv_for_gate = wkv.get_output(0)
    if wkv_for_gate.dtype != r_gate.get_output(0).dtype:
        wkv_for_gate = network.add_cast(wkv_for_gate, r_gate.get_output(0).dtype).get_output(0)
    gated = network.add_elementwise(
        r_gate.get_output(0), wkv_for_gate, trt.ElementWiseOperation.PROD
    )

    # out_proj: [1, hidden_size] @ w_attn_o -> [1, hidden_size]
    attn_out = graph_ops.add_matmul_rhs_constant(
        network,
        gated.get_output(0),
        hidden_size,
        hidden_size,
        weights[f"{prefix}.w_attn_o"],
        dtype=dtype,
    )

    # Residual add
    residual_attn = network.add_elementwise(hidden, attn_out, trt.ElementWiseOperation.SUM)
    hidden_after_attn = residual_attn.get_output(0)

    # ================================================================
    # CHANNEL-MIXING (FFN) BLOCK
    # ================================================================

    # ===== 1. LayerNorm =====
    normed_ffn = graph_ops.add_layer_norm(
        network,
        hidden_after_attn,
        hidden_size,
        weights[f"{prefix}.ffn_norm"],
        weights[f"{prefix}.ffn_norm_beta"],
        eps_tensor,
        dtype=dtype,
    )

    # The normed output before time-shift becomes present_ff for next step
    present_ff = normed_ffn

    # ===== 2. Time-shift for key and receptance =====
    xk_ffn = _time_shift_blend(normed_ffn, ff_state_in, f"{prefix}.time_mix_ffn_key")
    xr_ffn = _time_shift_blend(normed_ffn, ff_state_in, f"{prefix}.time_mix_ffn_receptance")

    # ===== 3. Key projection + squared ReLU =====
    k_ffn = graph_ops.add_matmul_rhs_constant(
        network, xk_ffn, hidden_size, intermediate_size, weights[f"{prefix}.w_ffn_k"], dtype=dtype
    )
    k_activated = graph_ops.add_activation(network, k_ffn, "squared_relu", dtype=dtype)

    # ===== 4. Receptance gate =====
    r_ffn = graph_ops.add_matmul_rhs_constant(
        network, xr_ffn, hidden_size, hidden_size, weights[f"{prefix}.w_ffn_r"], dtype=dtype
    )
    r_ffn_gate = network.add_activation(r_ffn, trt.ActivationType.SIGMOID)

    # ===== 5. Value projection + gating + residual =====
    kv_ffn = graph_ops.add_matmul_rhs_constant(
        network,
        k_activated,
        intermediate_size,
        hidden_size,
        weights[f"{prefix}.w_ffn_v"],
        dtype=dtype,
    )
    gated_ffn = network.add_elementwise(
        r_ffn_gate.get_output(0), kv_ffn, trt.ElementWiseOperation.PROD
    )

    # Residual add
    residual_ffn = network.add_elementwise(
        hidden_after_attn, gated_ffn.get_output(0), trt.ElementWiseOperation.SUM
    )

    return {
        "hidden": residual_ffn.get_output(0),
        "present_attn": present_attn,
        "present_ff": present_ff,
        "present_num": present_num.get_output(0),
        "present_den": present_den.get_output(0),
        "present_max": present_max,
    }


requires_tokenizer = True


def _detect_tokenizer_frame(
    source: str, *, revision: str | None = None
) -> tuple[list[int], list[int]] | None:
    try:
        from transformers import AutoTokenizer

        kwargs = {"trust_remote_code": True}
        if revision:
            kwargs["revision"] = revision
        if not Path(source).is_dir():
            kwargs["local_files_only"] = True
        tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
        default_ids = list(tokenizer.encode("hello"))
        plain_ids = list(tokenizer.encode("hello", add_special_tokens=False))
    except Exception:
        return None
    if default_ids == plain_ids:
        return [], []
    if not plain_ids:
        return default_ids, []
    for start in range(len(default_ids) - len(plain_ids) + 1):
        if default_ids[start : start + len(plain_ids)] == plain_ids:
            return default_ids[:start], default_ids[start + len(plain_ids) :]
    return None


def _ensure_tokenizer_json(model_dir: Path) -> None:
    tokenizer_path = model_dir / "tokenizer.json"
    if tokenizer_path.is_file():
        return
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
        with tempfile.TemporaryDirectory(prefix="trtmc-tokenizer-") as temporary:
            generated = Path(temporary) / "tokenizer.json"
            backend = getattr(tokenizer, "backend_tokenizer", None)
            if backend is None:
                backend = getattr(tokenizer, "_tokenizer", None)
            if backend is not None and hasattr(backend, "save"):
                backend.save(str(generated))
            if not generated.is_file():
                tokenizer.save_pretrained(temporary)
            if not generated.is_file():
                raise RuntimeError("tokenizer conversion did not create tokenizer.json")
            with tempfile.NamedTemporaryFile(
                dir=model_dir, prefix=".trtmc-tokenizer-", suffix=".json", delete=False
            ) as output:
                temporary_path = Path(output.name)
                output.write(generated.read_bytes())
            temporary_path.replace(tokenizer_path)
    except Exception as exc:
        print(
            "[trtmc build] Warning: could not generate tokenizer.json "
            f"(C++ runtime may fail to create tokenizer): {exc}",
            file=sys.stderr,
        )


def _apply_generation_config_eos(model_dir: Path, config: dict) -> None:
    path = model_dir / "generation_config.json"
    if not path.is_file():
        return
    generation_config = json.loads(path.read_text(encoding="utf-8"))
    if "eos_token_id" in generation_config:
        config["eos_token_id"] = generation_config["eos_token_id"]


def _build_local_engine(
    config, weights, max_cache_length, precision, quant_ctx, verbose, parallel, options
):
    from tensorrt_model_connect.tvm_ffi.graph_build import engine_role, inspection_role

    role = (
        "dual_profile"
        if str(options.get("decoder_engine_layout") or "split") == "dual_profile"
        else "decode"
    )

    def build_role(selected_role: str) -> bytes:
        with engine_role(selected_role):
            return build_engine(
                config,
                weights,
                max_cache_length,
                precision=precision,
                quant_ctx=quant_ctx,
                verbose=verbose,
                parallel_config=parallel,
            )

    target_role = inspection_role()
    if target_role is not None:
        build_role(target_role)
        raise RuntimeError("graph inspection did not reach TensorRT serialization")
    return build_role(role), ("dual_profile" if role == "dual_profile" else "single")


def build(model_dir: str, output_path: str, **options) -> None:
    """Build the complete rwkv bundle inside its owning family module."""
    from dataclasses import replace
    from datetime import datetime, timezone

    from tensorrt_model_connect import trt_compat as build_trt_compat
    from tensorrt_model_connect.build_timing import (
        add_build_timing,
        new_build_timing,
        write_build_timing,
    )
    from tensorrt_model_connect.bundle_writer import BundleInfo, BundleSection, write_bundle
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
        raise NotImplementedError("rwkv does not support context-parallel builds")
    if options.get("dynamic_kv_cache") or options.get("triattention_stats_path"):
        raise ValueError("rwkv does not use a decoder KV-cache runtime")

    config = ModelConfig.from_dir(model_path)
    config.raw["_model_dir"] = str(model_path)
    config.raw["_decoder_engine_layout"] = decoder_engine_layout
    config.raw["_fp32_layers"] = sorted(set(options.get("fp32_layers") or ()))
    config.raw["_family_build_options"] = dict(options.get("family_build_options") or {})
    config.raw["_parallel_build_enabled"] = bool(parallel.enabled)
    config.raw["_rtx_build_requested"] = bool(options.get("rtx"))
    config.raw["_runtime_dynamic_kv_requested"] = False
    config.raw["_quantized_build_requested"] = bool(options.get("quantize"))
    config.raw["_disable_dual_profile_decoder"] = True
    precision = str(options.get("precision") or "fp32").lower()
    config.raw["_resolved_build_precision"] = precision
    requested_cache_length = options.get("max_cache_length")
    max_cache_length = int(256 if requested_cache_length is None else requested_cache_length)
    if max_cache_length < 1:
        raise ValueError("max_cache_length must be >= 1")

    timing = new_build_timing(options.get("build_timing_path"))
    timing["model_dir"] = str(model_path)
    timing["output_path"] = str(output_path)
    started = time.monotonic()
    write_build_timing(timing)

    weights_started = time.monotonic()
    weights = load_weights(str(model_path), config)
    add_build_timing(timing, "weights_loading_s", time.monotonic() - weights_started)
    write_build_timing(timing)

    quantize = options.get("quantize")
    quant_ctx = None
    quant_plan = None
    if quantize:
        from tensorrt_model_connect.quantization import QuantPlan, build_quant_context
        from . import graph_ops as family_graph_ops

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
            graph_ops=family_graph_ops,
        )

    if parallel.enabled:
        require_tensorrt_11_for_tensor_parallel(parallel, feature="rwkv tensor-parallel builds")
        if quant_ctx is not None:
            raise ValueError("rwkv tensor-parallel builds do not support quantization")

    verbose = bool(options.get("verbose"))
    compile_started = time.monotonic()
    if parallel.enabled:
        plans = {
            rank: build_engine(
                config,
                weights,
                max_cache_length,
                precision=precision,
                quant_ctx=quant_ctx,
                verbose=verbose,
                parallel_config=parallel.for_rank(rank),
            )
            for rank in range(parallel.tp_size)
        }
        sections = [
            BundleSection(rank_engine_section(rank), plan) for rank, plan in sorted(plans.items())
        ]
        decoder_layout = "dual_profile"
    else:
        plan, decoder_layout = _build_local_engine(
            config, weights, max_cache_length, precision, quant_ctx, verbose, parallel, options
        )
        sections = [BundleSection("engine_plan", plan)]
    compile_elapsed = time.monotonic() - compile_started
    add_build_timing(timing, "trt_compile_s", compile_elapsed)
    add_build_timing(timing, "trt_compile_main_engine_s", compile_elapsed)
    write_build_timing(timing)

    tokenizer_source = str(options.get("tokenizer_source_model_id_or_path") or model_path)
    tokenizer_frame = _detect_tokenizer_frame(
        tokenizer_source,
        revision=(
            str(options["tokenizer_source_revision"])
            if options.get("tokenizer_source_revision")
            else None
        ),
    )
    _ensure_tokenizer_json(model_path)
    if tokenizer_frame is None:
        tokenizer_frame = _detect_tokenizer_frame(str(model_path))
    prefix_ids, suffix_ids = tokenizer_frame or ([], [])
    add_special_tokens = bool(prefix_ids or suffix_ids)

    trt_version = build_trt_compat.tensorrt_version() or "unknown"
    version_match = re.search(r"(\d+)\.(\d+)", trt_version)
    trt_abi = f"{version_match.group(1)}.{version_match.group(2)}" if version_match else ""
    try:
        from tensorrt_model_connect.runtime_provider.target import _probe_current_target_with_device

        gpu_name = str(_probe_current_target_with_device()[0]["gpu_name"])
    except Exception:
        gpu_name = ""
    info = BundleInfo(
        model_id=model_path.name,
        model_type=config.model_type,
        family=name,
        trt_version=trt_version,
        trt_abi=trt_abi,
        gpu_name=gpu_name,
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
        else dict(config.raw)
    )
    _apply_generation_config_eos(model_path, runtime_config)
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
    if tokenizer_frame is not None:
        runtime_config["tokenizer_special_prefix_ids"] = prefix_ids
        runtime_config["tokenizer_special_suffix_ids"] = suffix_ids
    if options.get("fp32_layers"):
        runtime_config["fp32_layers"] = sorted(set(options["fp32_layers"]))
    if quant_plan is not None:
        runtime_config["quantization"] = quant_plan.as_config_dict()
    runtime_config.update(parallel.to_bundle_config_fields())

    from tensorrt_model_connect.tvm_ffi.graph_build import kernel_slots_section

    slot_section = kernel_slots_section()
    if slot_section is not None:
        sections.append(BundleSection("kernel_slots.json", slot_section))

    embedded_config = False
    for filename in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.model",
        "preprocessor_config.json",
        "processor_config.json",
    ):
        path = model_path / filename
        if filename == "config.json":
            sections.append(
                BundleSection(filename, json.dumps(runtime_config, indent=2).encode("utf-8"))
            )
            embedded_config = True
        elif path.is_file():
            sections.append(BundleSection(filename, path.read_bytes()))
    if not embedded_config:
        sections.append(
            BundleSection("config.json", json.dumps(runtime_config, indent=2).encode("utf-8"))
        )

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
