# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NemotronH family model -- Hybrid Mamba-2 + MLP + Attention decoder.

NemotronH (NVIDIA) uses a heterogeneous layer stack with three layer types
defined by hybrid_override_pattern (e.g. "M-M-M-MM-M-M-M*-..."):
  M = Mamba-2 SSM layer
  - = MLP layer (up_proj -> relu2 -> down_proj)
  * = Attention layer (GQA, no RoPE, no bias)

Key differences from Mamba-1 (existing mamba.py):
  Mamba-2 uses State Space Duality (SSD):
    - in_proj -> split into [gate, hidden_B_C, dt]
    - conv1d over hidden_B_C (d_inner + 2*n_groups*d_state channels)
    - After conv+SiLU, split hidden_B_C -> [hidden, B, C]
    - Multi-head SSM (nheads * headdim = d_inner)
    - A is a scalar per head (not per d_inner like Mamba-1)
    - dt from in_proj directly (no separate x_proj/dt_proj)
    - Gated RMSNorm on SSM output: norm(y) * silu(gate)
    - SSM state: [nheads, headdim, d_state] (headdim-aware)

NemotronH Nano 9B: 56 layers (27 mamba2 + 25 mlp + 4 attention)
  - MLP layers: up_proj -> relu2 -> down_proj (NO gate_proj)
  - Attention layers: q/k/v/o_proj (GQA, no RoPE, no bias)

Weight key mapping (HF -> engine):
  backbone.embeddings.weight                           -> embedding
  backbone.layers.{i}.norm.weight                      -> layer.{i}.norm
  backbone.layers.{i}.mixer.in_proj.weight             -> Mamba-2 in_proj
  backbone.layers.{i}.mixer.conv1d.weight/bias         -> Mamba-2 conv state
  backbone.layers.{i}.mixer.dt_bias                    -> Mamba-2 timestep bias
  backbone.layers.{i}.mixer.A_log                      -> Mamba-2 SSM A
  backbone.layers.{i}.mixer.D                          -> Mamba-2 skip connection
  backbone.layers.{i}.mixer.norm.weight                -> Mamba-2 gated RMSNorm
  backbone.layers.{i}.mixer.out_proj.weight            -> Mamba-2 output proj
  backbone.layers.{i}.mixer.up_proj.weight             -> MLP up
  backbone.layers.{i}.mixer.down_proj.weight           -> MLP down
  backbone.layers.{i}.mixer.q/k/v/o_proj.weight        -> Attention QKV + out
  backbone.norm_f.weight                               -> final_norm
  lm_head.weight                                       -> w_lm_head
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
from . import graph_blocks
from ...parallel_config import (
    normalize_parallel_config,
    require_tensorrt_11_for_tensor_parallel,
)


trt = trt_compat.get_trt()


def _parse_layer_types(pattern: str) -> list[str]:
    """Parse hybrid_override_pattern: M=mamba2, -=mlp, *=attention."""
    mapping = {"M": "mamba2", "-": "mlp", "*": "attention"}
    return [mapping[ch] for ch in pattern if ch in mapping]


name = "nemotron_h"
runtime_strategy = "nemotron_h_hybrid_mamba_attention"


def matches(config) -> bool:
    """Return whether this module owns the resolved model config."""
    model_type = getattr(config, "model_type", config)
    model_type = str(model_type)
    return model_type.lower() in {"nemotron_h", "nemotron_hybrid"}


def load_weights(model_dir: str, config: ModelConfig) -> WeightDict:
    model_dir_path = Path(model_dir)
    readers = _open_safetensors(model_dir_path)

    hidden = config.hidden_size
    vocab = config.vocab_size
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    head_dim = config.head_dim
    raw = config.raw

    # Parse layer types from hybrid_override_pattern
    pattern = raw.get("hybrid_override_pattern", "M" * num_layers)
    layer_types = _parse_layer_types(pattern)
    assert len(layer_types) == num_layers, (
        f"Pattern length {len(layer_types)} != num_hidden_layers {num_layers}"
    )

    # Mamba-2 dimensions
    mamba_num_heads = raw.get("mamba_num_heads", 64)
    mamba_head_dim = raw.get("mamba_head_dim", 64)
    d_inner = mamba_num_heads * mamba_head_dim
    n_groups = raw.get("n_groups", 8)
    d_state = raw.get("ssm_state_size", raw.get("mamba_state_dim", 128))
    d_conv = raw.get("conv_kernel", 4)
    conv_dim = d_inner + 2 * n_groups * d_state

    # MLP dimensions
    mlp_intermediate = config.intermediate_size

    # Attention dimensions
    q_dim = num_heads * head_dim
    weights = WeightDict()

    # Embedding
    embedding = _load_tensor(readers, "backbone.embeddings.weight")
    assert embedding.shape == (vocab, hidden), (
        f"Embedding shape {embedding.shape} != ({vocab}, {hidden})"
    )
    weights["embedding"] = embedding.astype(np.float32)

    mamba_count = 0
    attn_count = 0

    for layer_idx in range(num_layers):
        lt = layer_types[layer_idx]
        prefix = f"layer.{layer_idx}"
        hf_prefix = f"backbone.layers.{layer_idx}"

        # RMSNorm (all layer types)
        norm = _load_tensor(readers, f"{hf_prefix}.norm.weight")
        weights[f"{prefix}.input_norm"] = norm.astype(np.float32)

        if lt == "mamba2":
            # in_proj: [proj_size, hidden] where proj_size = d_inner + conv_dim + mamba_num_heads
            in_proj_raw = _load_tensor(readers, f"{hf_prefix}.mixer.in_proj.weight")
            weights[f"{prefix}.mamba_in_proj"] = _transpose_2d(in_proj_raw, "mamba_in_proj")

            # conv1d: [conv_dim, 1, d_conv] -> [conv_dim, d_conv]
            conv1d_w = _load_tensor(readers, f"{hf_prefix}.mixer.conv1d.weight")
            weights[f"{prefix}.conv1d_weight"] = conv1d_w.reshape(conv_dim, d_conv).astype(
                np.float32
            )

            conv1d_b = _load_tensor(readers, f"{hf_prefix}.mixer.conv1d.bias")
            weights[f"{prefix}.conv1d_bias"] = conv1d_b.astype(np.float32)

            # out_proj: [hidden, d_inner]
            out_proj_raw = _load_tensor(readers, f"{hf_prefix}.mixer.out_proj.weight")
            weights[f"{prefix}.mamba_out_proj"] = _transpose_2d(out_proj_raw, "mamba_out_proj")

            # A_log: [mamba_num_heads]
            A_log = _load_tensor(readers, f"{hf_prefix}.mixer.A_log")
            A = -np.exp(A_log.astype(np.float32))
            weights[f"{prefix}.A"] = A

            # D: [mamba_num_heads]
            D = _load_tensor(readers, f"{hf_prefix}.mixer.D")
            weights[f"{prefix}.D"] = D.astype(np.float32)

            # dt_bias: [mamba_num_heads]
            dt_bias = _load_tensor(readers, f"{hf_prefix}.mixer.dt_bias")
            weights[f"{prefix}.dt_bias"] = dt_bias.astype(np.float32)

            # Gated RMSNorm: [d_inner]
            norm_key = f"{hf_prefix}.mixer.norm.weight"
            if _has_tensor(readers, norm_key):
                weights[f"{prefix}.mamba_norm"] = _load_tensor(readers, norm_key).astype(np.float32)
            else:
                weights[f"{prefix}.mamba_norm"] = np.ones(d_inner, dtype=np.float32)

            mamba_count += 1

        elif lt == "mlp":
            # MLP: up_proj -> relu2 -> down_proj (NO gate_proj)
            up_raw = _load_tensor(readers, f"{hf_prefix}.mixer.up_proj.weight")
            down_raw = _load_tensor(readers, f"{hf_prefix}.mixer.down_proj.weight")
            weights[f"{prefix}.w_up"] = _transpose_2d(up_raw, "up_proj")
            weights[f"{prefix}.w_down"] = _transpose_2d(down_raw, "down_proj")

        elif lt == "attention":
            # Attention: q/k/v/o projections (no bias, no RoPE)
            q_raw = _load_tensor(readers, f"{hf_prefix}.mixer.q_proj.weight")
            k_raw = _load_tensor(readers, f"{hf_prefix}.mixer.k_proj.weight")
            v_raw = _load_tensor(readers, f"{hf_prefix}.mixer.v_proj.weight")
            o_raw = _load_tensor(readers, f"{hf_prefix}.mixer.o_proj.weight")

            q_t = _transpose_2d(q_raw, "q_proj")
            k_t = _transpose_2d(k_raw, "k_proj")
            v_t = _transpose_2d(v_raw, "v_proj")
            o_t = _transpose_2d(o_raw, "o_proj")

            # Compact GQA/MQA K/V

            weights[f"{prefix}.w_q"] = q_t
            weights[f"{prefix}.w_k"] = k_t
            weights[f"{prefix}.w_v"] = v_t
            weights[f"{prefix}.w_o"] = o_t

            attn_count += 1

    # Final norm
    final_norm_key = "backbone.norm_f.weight"
    if _has_tensor(readers, final_norm_key):
        weights["final_norm"] = _load_tensor(readers, final_norm_key).astype(np.float32)
    else:
        weights["final_norm"] = np.ones(hidden, dtype=np.float32)

    # LM head
    lm_head_key = "lm_head.weight"
    if _has_tensor(readers, lm_head_key):
        weights["w_lm_head"] = _transpose_2d(_load_tensor(readers, lm_head_key), "lm_head")
    else:
        weights["w_lm_head"] = _transpose_2d(embedding.copy(), "embedding_tied")

    # Metadata for engine builder
    weights["_layer_types"] = layer_types
    weights["_d_inner"] = d_inner
    weights["_d_state"] = d_state
    weights["_d_conv"] = d_conv
    weights["_conv_dim"] = conv_dim
    weights["_mamba_num_heads"] = mamba_num_heads
    weights["_mamba_head_dim"] = mamba_head_dim
    weights["_n_groups"] = n_groups
    weights["_num_mamba_layers"] = mamba_count
    weights["_num_attention_layers"] = attn_count
    weights["_attention_size"] = q_dim
    weights["_mlp_size"] = mlp_intermediate

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
    """Build hybrid TRT engine with heterogeneous layer stack."""
    parallel = normalize_parallel_config(parallel_config)
    if parallel.enabled:
        require_tensorrt_11_for_tensor_parallel(
            parallel, feature="Nemotron-H tensor-parallel builds"
        )
        from .tp_builder import build_nemotron_h_tp_engine

        return build_nemotron_h_tp_engine(
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
    requested_fp32_layers = frozenset(int(layer) for layer in config.raw.get("_fp32_layers", ()))
    invalid_fp32_layers = sorted(
        layer for layer in requested_fp32_layers if layer < 0 or layer > num_layers
    )
    if invalid_fp32_layers:
        raise ValueError(f"fp32_layers contains out-of-range indices: {invalid_fp32_layers}")

    layer_types: list[str] = weights["_layer_types"]
    d_inner: int = weights["_d_inner"]
    d_state: int = weights["_d_state"]
    d_conv: int = weights["_d_conv"]
    conv_dim: int = weights["_conv_dim"]
    mamba_num_heads: int = weights["_mamba_num_heads"]
    mamba_head_dim: int = weights["_mamba_head_dim"]
    n_groups: int = weights["_n_groups"]
    num_mamba: int = weights["_num_mamba_layers"]
    num_attn: int = weights["_num_attention_layers"]
    attention_size: int = weights["_attention_size"]
    mlp_size: int = weights["_mlp_size"]
    if precision == "fp16":
        work_np_dtype, work_trt_dtype = np.float16, trt.float16
    elif precision == "fp32":
        work_np_dtype, work_trt_dtype = np.float32, trt.float32
    else:
        raise ValueError(f"Unsupported Nemotron-H precision {precision!r}; expected fp32 or fp16")
    use_fp32_io = precision == "fp16" and num_layers in requested_fp32_layers
    io_np_dtype = np.float32 if use_fp32_io else work_np_dtype
    io_trt_dtype = trt.float32 if use_fp32_io else work_trt_dtype

    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = attention_size // num_heads
    kv_attention_size = graph_blocks.infer_kv_attention_size(
        weights, num_kv_heads=num_kv_heads, head_dim=head_dim
    )
    attention_window = max_cache_length + 1

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    trt_config = builder.create_builder_config()
    trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    # --- Inputs ---
    token_id = network.add_input("token_id", trt.int32, (1,))
    position_id = network.add_input("position_id", trt.int32, (1,))
    attention_mask = network.add_input("attention_mask", trt.float32, (1, attention_window))

    conv_state_inputs = []
    ssm_state_inputs = []
    for mi in range(num_mamba):
        cs = network.add_input(
            graph_ops.layer_tensor_name("conv_state", mi), trt.float32, (conv_dim, d_conv)
        )
        ss = network.add_input(
            graph_ops.layer_tensor_name("ssm_state", mi),
            trt.float32,
            (mamba_num_heads, mamba_head_dim, d_state),
        )
        conv_state_inputs.append(cs)
        ssm_state_inputs.append(ss)

    cache_k_inputs = []
    cache_v_inputs = []
    for ai in range(num_attn):
        ck = network.add_input(
            graph_ops.layer_tensor_name("cache_k", ai),
            work_trt_dtype,
            (max_cache_length, kv_attention_size),
        )
        cv = network.add_input(
            graph_ops.layer_tensor_name("cache_v", ai),
            work_trt_dtype,
            (max_cache_length, kv_attention_size),
        )
        cache_k_inputs.append(ck)
        cache_v_inputs.append(cv)

    # --- Shared constants ---
    embedding_table = graph_ops.add_constant(
        network, (vocab, hidden), weights["embedding"], dtype=io_np_dtype
    )
    eps_tensor = graph_ops.add_constant(
        network,
        (1, 1),
        np.array([config.rms_norm_eps], dtype=work_np_dtype),
        dtype=work_np_dtype,
    )
    io_eps_tensor = (
        graph_ops.add_constant(
            network, (1, 1), np.array([config.rms_norm_eps], dtype=np.float32), dtype=np.float32
        )
        if use_fp32_io
        else eps_tensor
    )

    # --- Embedding ---
    gather = network.add_gather(embedding_table, token_id, 0)
    hidden_state = gather.get_output(0)

    if debug_layer_outputs:
        _mark_debug_output(network, hidden_state, "debug_embed")

    # --- Layer stack ---
    present_conv_outputs = []
    present_ssm_outputs = []
    present_k_outputs = []
    present_v_outputs = []
    mamba_counter = 0
    attn_counter = 0

    for layer_idx in range(num_layers):
        prefix = f"layer.{layer_idx}"
        lt = layer_types[layer_idx]
        use_fp32_layer = precision == "fp16" and layer_idx in requested_fp32_layers
        layer_np_dtype = np.float32 if use_fp32_layer else work_np_dtype
        layer_trt_dtype = trt.float32 if use_fp32_layer else work_trt_dtype
        layer_hidden = hidden_state
        layer_eps = eps_tensor
        if layer_hidden.dtype != layer_trt_dtype:
            layer_hidden = network.add_cast(layer_hidden, layer_trt_dtype).get_output(0)
        if layer_eps.dtype != layer_trt_dtype:
            layer_eps = network.add_cast(layer_eps, layer_trt_dtype).get_output(0)

        if lt == "mamba2":
            conv_state = conv_state_inputs[mamba_counter]
            ssm_state = ssm_state_inputs[mamba_counter]
            if conv_state.dtype != layer_trt_dtype:
                conv_state = network.add_cast(conv_state, layer_trt_dtype).get_output(0)
            result = _add_mamba2_layer(
                network=network,
                hidden=layer_hidden,
                conv_state_in=conv_state,
                ssm_state_in=ssm_state,
                eps_tensor=layer_eps,
                weights=weights,
                prefix=prefix,
                hidden_size=hidden,
                d_inner=d_inner,
                d_state=d_state,
                d_conv=d_conv,
                conv_dim=conv_dim,
                mamba_num_heads=mamba_num_heads,
                mamba_head_dim=mamba_head_dim,
                n_groups=n_groups,
                dtype=layer_np_dtype,
            )
            hidden_state = result["hidden"]
            present_conv_outputs.append(result["present_conv"])
            present_ssm_outputs.append(result["present_ssm"])
            mamba_counter += 1

        elif lt == "mlp":
            result = _add_mlp_layer(
                network=network,
                hidden=layer_hidden,
                eps_tensor=layer_eps,
                weights=weights,
                prefix=prefix,
                hidden_size=hidden,
                mlp_size=mlp_size,
                dtype=layer_np_dtype,
            )
            hidden_state = result["hidden"]

        elif lt == "attention":
            cache_k = cache_k_inputs[attn_counter]
            cache_v = cache_v_inputs[attn_counter]
            layer_mask = attention_mask
            if cache_k.dtype != layer_trt_dtype:
                cache_k = network.add_cast(cache_k, layer_trt_dtype).get_output(0)
            if cache_v.dtype != layer_trt_dtype:
                cache_v = network.add_cast(cache_v, layer_trt_dtype).get_output(0)
            if layer_mask.dtype != layer_trt_dtype:
                layer_mask = network.add_cast(layer_mask, layer_trt_dtype).get_output(0)
            result = graph_blocks.add_attention_block(
                network,
                layer_hidden,
                cache_k,
                cache_v,
                layer_mask,
                position_id,
                weights=weights,
                prefix=prefix,
                hidden_size=hidden,
                attention_size=attention_size,
                kv_attention_size=kv_attention_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                max_cache_length=max_cache_length,
                eps_tensor=layer_eps,
                dtype=layer_np_dtype,
            )
            # add_attention_block does NOT apply residual
            residual = network.add_elementwise(
                layer_hidden, result["attn_out"], trt.ElementWiseOperation.SUM
            )
            hidden_state = residual.get_output(0)
            present_k_outputs.append(result["present_k"])
            present_v_outputs.append(result["present_v"])
            attn_counter += 1

        if debug_layer_outputs:
            _mark_debug_output(network, hidden_state, f"debug_hidden_{layer_idx}")

    # --- Final norm ---
    if hidden_state.dtype != io_trt_dtype:
        hidden_state = network.add_cast(hidden_state, io_trt_dtype).get_output(0)
    final_norm = weights.get("final_norm")
    if final_norm is not None and len(final_norm) > 0:
        hidden_state = graph_ops.add_rms_norm(
            network, hidden_state, hidden, final_norm, io_eps_tensor, dtype=io_np_dtype
        )

    # --- LM head ---
    logits = graph_ops.add_matmul_rhs_constant(
        network, hidden_state, hidden, vocab, weights["w_lm_head"], dtype=io_np_dtype
    )
    b_out = np.zeros(vocab, dtype=io_np_dtype)
    logits = graph_ops.add_bias_sum(network, logits, vocab, b_out, dtype=io_np_dtype)
    if logits.dtype != trt.float32:
        logits = network.add_cast(logits, trt.float32).get_output(0)
    logits.name = "logits"
    network.mark_output(logits)

    # --- Present state outputs ---
    for mi in range(num_mamba):
        pc = present_conv_outputs[mi]
        ps = present_ssm_outputs[mi]
        if pc.dtype != trt.float32:
            pc = network.add_cast(pc, trt.float32).get_output(0)
        if ps.dtype != trt.float32:
            ps = network.add_cast(ps, trt.float32).get_output(0)
        pc.name = graph_ops.layer_tensor_name("present_conv", mi)
        ps.name = graph_ops.layer_tensor_name("present_ssm", mi)
        network.mark_output(pc)
        network.mark_output(ps)

    for ai in range(num_attn):
        pk = present_k_outputs[ai]
        pv = present_v_outputs[ai]
        if pk.dtype != work_trt_dtype:
            pk = network.add_cast(pk, work_trt_dtype).get_output(0)
        if pv.dtype != work_trt_dtype:
            pv = network.add_cast(pv, work_trt_dtype).get_output(0)
        pk.name = graph_ops.layer_tensor_name("present_k", ai)
        pv.name = graph_ops.layer_tensor_name("present_v", ai)
        network.mark_output(pk)
        network.mark_output(pv)

    # --- Build ---
    if verbose:
        print(
            f"[trtmc build] Building NemotronH hybrid TRT engine "
            f"({num_layers} layers: {num_mamba} mamba2 + "
            f"{sum(1 for t in layer_types if t == 'mlp')} mlp + "
            f"{num_attn} attention, "
            f"hidden={hidden}, d_inner={d_inner}, "
            f"d_state={d_state}, nheads={mamba_num_heads}, "
            f"cache={max_cache_length}) ...",
            file=sys.stderr,
        )

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("TensorRT engine build failed")

    return bytes(plan)


def get_bundle_config_overrides(config: ModelConfig) -> dict:
    """Inject hybrid-specific config fields into the bundle."""
    raw = config.raw
    pattern = raw.get("hybrid_override_pattern", "")
    layer_types = _parse_layer_types(pattern)

    mamba_num_heads = raw.get("mamba_num_heads", 64)
    mamba_head_dim = raw.get("mamba_head_dim", 64)
    d_inner = mamba_num_heads * mamba_head_dim
    n_groups = raw.get("n_groups", 8)
    d_state = raw.get("ssm_state_size", raw.get("mamba_state_dim", 128))
    d_conv = raw.get("conv_kernel", 4)

    num_mamba = sum(1 for lt in layer_types if lt == "mamba2")
    num_attn = sum(1 for lt in layer_types if lt == "attention")

    conv_dim = d_inner + 2 * n_groups * d_state

    return {
        "layer_types": layer_types,
        "num_mamba_layers": num_mamba,
        "num_attention_layers": num_attn,
        "d_inner": d_inner,
        "mamba_d_state": d_state,
        "mamba_d_conv": d_conv,
        "mamba_nheads": mamba_num_heads,
        "mamba_head_dim": mamba_head_dim,
        "conv_dim": conv_dim,
        "n_groups": n_groups,
    }


def _mark_debug_output(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    name: str,
) -> None:
    identity = network.add_identity(tensor)
    cast = network.add_cast(identity.get_output(0), trt.float32)
    out = cast.get_output(0)
    out.name = name
    network.mark_output(out)


def _add_mamba2_layer(
    *,
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    conv_state_in: trt.ITensor,
    ssm_state_in: trt.ITensor,
    eps_tensor: trt.ITensor,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    d_inner: int,
    d_state: int,
    d_conv: int,
    conv_dim: int,
    mamba_num_heads: int,
    mamba_head_dim: int,
    n_groups: int,
    dtype: np.dtype = np.float32,
) -> dict[str, trt.ITensor]:
    """Add one Mamba-2 SSD layer (single-step decode).

    Mamba-2 in_proj splits: [gate(d_inner), hidden_B_C(conv_dim), dt(nheads)]
    Conv1d operates on hidden_B_C (d_inner + 2*n_groups*d_state channels).
    After conv+SiLU, split: hidden[d_inner], B[n_groups*d_state], C[n_groups*d_state].
    SSM state shape: [nheads, headdim, d_state] for full headdim-aware state.

    Returns: {hidden, present_conv, present_ssm}
    """
    groups_state_size = n_groups * d_state

    # ===== 1. RMSNorm =====
    normed = graph_ops.add_rms_norm(
        network, hidden, hidden_size, weights[f"{prefix}.input_norm"], eps_tensor, dtype=dtype
    )

    # ===== 2. Input projection =====
    proj_dim = d_inner + conv_dim + mamba_num_heads
    projected = graph_ops.add_matmul_rhs_constant(
        network, normed, hidden_size, proj_dim, weights[f"{prefix}.mamba_in_proj"], dtype=dtype
    )  # [1, proj_dim]

    # Split: gate [d_inner], hidden_B_C [conv_dim], dt [nheads]
    offset = 0
    gate_slice = network.add_slice(projected, start=(0, offset), shape=(1, d_inner), stride=(1, 1))
    gate = gate_slice.get_output(0)
    offset += d_inner

    hbc_slice = network.add_slice(projected, start=(0, offset), shape=(1, conv_dim), stride=(1, 1))
    hidden_B_C = hbc_slice.get_output(0)
    offset += conv_dim

    dt_slice = network.add_slice(
        projected, start=(0, offset), shape=(1, mamba_num_heads), stride=(1, 1)
    )
    dt_raw = dt_slice.get_output(0)

    # ===== 3. Conv1d step on hidden_B_C =====
    # conv_state_in: [conv_dim, d_conv]
    # hidden_B_C: [1, conv_dim] -> [conv_dim, 1]
    hbc_col = network.add_shuffle(hidden_B_C)
    hbc_col.reshape_dims = (conv_dim, 1)

    if d_conv > 1:
        slice_layer = network.add_slice(
            conv_state_in, start=(0, 1), shape=(conv_dim, d_conv - 1), stride=(1, 1)
        )
        new_conv_state = network.add_concatenation(
            [slice_layer.get_output(0), hbc_col.get_output(0)]
        )
        new_conv_state.axis = 1
        present_conv = new_conv_state.get_output(0)
    else:
        present_conv = hbc_col.get_output(0)

    conv_w = graph_ops.add_constant(
        network, (conv_dim, d_conv), weights[f"{prefix}.conv1d_weight"], dtype=dtype
    )
    conv_prod = network.add_elementwise(present_conv, conv_w, trt.ElementWiseOperation.PROD)
    conv_sum = network.add_reduce(
        conv_prod.get_output(0), trt.ReduceOperation.SUM, 1 << 1, keep_dims=True
    )
    conv_flat = network.add_shuffle(conv_sum.get_output(0))
    conv_flat.reshape_dims = (1, conv_dim)
    conv_out = graph_ops.add_bias_sum(
        network, conv_flat.get_output(0), conv_dim, weights[f"{prefix}.conv1d_bias"], dtype=dtype
    )
    hbc_activated = graph_ops.add_activation(network, conv_out, "silu", dtype=dtype)

    # ===== 4. Split hidden, B, C from activated output =====
    hidden_x_slice = network.add_slice(
        hbc_activated, start=(0, 0), shape=(1, d_inner), stride=(1, 1)
    )
    hidden_x = hidden_x_slice.get_output(0)

    B_raw_slice = network.add_slice(
        hbc_activated, start=(0, d_inner), shape=(1, groups_state_size), stride=(1, 1)
    )
    B_raw = B_raw_slice.get_output(0)

    C_raw_slice = network.add_slice(
        hbc_activated,
        start=(0, d_inner + groups_state_size),
        shape=(1, groups_state_size),
        stride=(1, 1),
    )
    C_raw = C_raw_slice.get_output(0)

    # ===== 5. dt: add bias + softplus =====
    dt_bias_const = graph_ops.add_constant(
        network, (1, mamba_num_heads), weights[f"{prefix}.dt_bias"], dtype=dtype
    )
    dt_biased = network.add_elementwise(dt_raw, dt_bias_const, trt.ElementWiseOperation.SUM)
    # The checkpoint contains dt_bias values as large as 33.5. A naive FP16
    # exp overflows above ~11, while the original Mamba kernel evaluates this
    # softplus stably. Keep this scalar recurrence boundary in FP32.
    dt_for_state = dt_biased.get_output(0)
    if dt_for_state.dtype != trt.float32:
        dt_for_state = network.add_cast(dt_for_state, trt.float32).get_output(0)
    dt_exp = network.add_unary(dt_for_state, trt.UnaryOperation.EXP)
    one = graph_ops.add_constant(
        network, (1, 1), np.array([1.0], dtype=np.float32), dtype=np.float32
    )
    dt_exp_p1 = network.add_elementwise(dt_exp.get_output(0), one, trt.ElementWiseOperation.SUM)
    dt_softplus = network.add_unary(dt_exp_p1.get_output(0), trt.UnaryOperation.LOG)
    dt = dt_softplus.get_output(0)  # [1, mamba_num_heads]

    # ===== 6. Multi-head SSM step =====
    # A: [nheads] -> [nheads, 1, 1] for broadcast
    A_const = graph_ops.add_constant(
        network,
        (mamba_num_heads, 1, 1),
        weights[f"{prefix}.A"].reshape(mamba_num_heads, 1, 1),
        dtype=np.float32,
    )

    # dt: [1, nheads] -> [nheads, 1, 1]
    dt_col = network.add_shuffle(dt)
    dt_col.reshape_dims = (mamba_num_heads, 1, 1)

    # dA = exp(dt * A): broadcast to [nheads, headdim, d_state]
    dtA = network.add_elementwise(dt_col.get_output(0), A_const, trt.ElementWiseOperation.PROD)
    dA = network.add_unary(dtA.get_output(0), trt.UnaryOperation.EXP)

    # B: [1, n_groups*d_state] -> [n_groups, d_state] -> expand to [nheads, d_state]
    B_grouped = network.add_shuffle(B_raw)
    B_grouped.reshape_dims = (n_groups, d_state)
    heads_per_group = mamba_num_heads // n_groups

    if heads_per_group > 1:
        B_3d = network.add_shuffle(B_grouped.get_output(0))
        B_3d.reshape_dims = (n_groups, 1, d_state)
        tile_ones = graph_ops.add_constant(
            network,
            (1, heads_per_group, 1),
            np.ones((1, heads_per_group, 1), dtype=dtype),
            dtype=dtype,
        )
        B_tiled = network.add_elementwise(
            B_3d.get_output(0), tile_ones, trt.ElementWiseOperation.PROD
        )
        B_heads_s = network.add_shuffle(B_tiled.get_output(0))
        B_heads_s.reshape_dims = (mamba_num_heads, d_state)
        B_heads = B_heads_s.get_output(0)
    else:
        B_heads = B_grouped.get_output(0)

    # C: same group expansion
    C_grouped = network.add_shuffle(C_raw)
    C_grouped.reshape_dims = (n_groups, d_state)

    if heads_per_group > 1:
        C_3d = network.add_shuffle(C_grouped.get_output(0))
        C_3d.reshape_dims = (n_groups, 1, d_state)
        C_tiled = network.add_elementwise(
            C_3d.get_output(0), tile_ones, trt.ElementWiseOperation.PROD
        )
        C_heads_s = network.add_shuffle(C_tiled.get_output(0))
        C_heads_s.reshape_dims = (mamba_num_heads, d_state)
        C_heads = C_heads_s.get_output(0)
    else:
        C_heads = C_grouped.get_output(0)

    # x: [1, d_inner] -> [nheads, headdim]
    x_heads = network.add_shuffle(hidden_x)
    x_heads.reshape_dims = (mamba_num_heads, mamba_head_dim)

    # dBx[h,d,s] = dt[h] * B[h,s] * x[h,d]
    # dt_B: [nheads, 1, 1] * [nheads, 1, d_state] -> [nheads, 1, d_state]
    B_3d_expand = network.add_shuffle(B_heads)
    B_3d_expand.reshape_dims = (mamba_num_heads, 1, d_state)
    B_for_state = B_3d_expand.get_output(0)
    if B_for_state.dtype != trt.float32:
        B_for_state = network.add_cast(B_for_state, trt.float32).get_output(0)
    dt_B = network.add_elementwise(dt_col.get_output(0), B_for_state, trt.ElementWiseOperation.PROD)

    # x: [nheads, headdim] -> [nheads, headdim, 1]
    x_3d = network.add_shuffle(x_heads.get_output(0))
    x_3d.reshape_dims = (mamba_num_heads, mamba_head_dim, 1)
    x_for_state = x_3d.get_output(0)
    if x_for_state.dtype != trt.float32:
        x_for_state = network.add_cast(x_for_state, trt.float32).get_output(0)

    # dBx: [nheads, headdim, 1] * [nheads, 1, d_state] -> [nheads, headdim, d_state]
    dBx = network.add_elementwise(x_for_state, dt_B.get_output(0), trt.ElementWiseOperation.PROD)

    # SSM update: new_ssm = dA * ssm_state + dBx
    # ssm_state_in: [nheads, headdim, d_state]
    decay = network.add_elementwise(dA.get_output(0), ssm_state_in, trt.ElementWiseOperation.PROD)
    new_ssm = network.add_elementwise(
        decay.get_output(0), dBx.get_output(0), trt.ElementWiseOperation.SUM
    )
    present_ssm = new_ssm.get_output(0)  # [nheads, headdim, d_state]

    # y[h,d] = sum_s(ssm_state[h,d,s] * C[h,s])
    # C: [nheads, d_state] -> [nheads, d_state, 1]
    C_col = network.add_shuffle(C_heads)
    C_col.reshape_dims = (mamba_num_heads, d_state, 1)
    C_for_state = C_col.get_output(0)
    if C_for_state.dtype != trt.float32:
        C_for_state = network.add_cast(C_for_state, trt.float32).get_output(0)
    # batch matmul: [nheads, headdim, d_state] @ [nheads, d_state, 1] -> [nheads, headdim, 1]
    y_matmul = network.add_matrix_multiply(
        present_ssm, trt.MatrixOperation.NONE, C_for_state, trt.MatrixOperation.NONE
    )
    y_squeeze = network.add_shuffle(y_matmul.get_output(0))
    y_squeeze.reshape_dims = (mamba_num_heads, mamba_head_dim)

    # D skip: D[h] * x[h,d]
    D_const = graph_ops.add_constant(
        network,
        (mamba_num_heads, 1),
        weights[f"{prefix}.D"].reshape(mamba_num_heads, 1),
        dtype=np.float32,
    )
    x_for_skip = x_heads.get_output(0)
    if x_for_skip.dtype != trt.float32:
        x_for_skip = network.add_cast(x_for_skip, trt.float32).get_output(0)
    Dx = network.add_elementwise(D_const, x_for_skip, trt.ElementWiseOperation.PROD)

    y_plus_D = network.add_elementwise(
        y_squeeze.get_output(0), Dx.get_output(0), trt.ElementWiseOperation.SUM
    )
    # [nheads, headdim] -> [1, d_inner]
    y_flat = network.add_shuffle(y_plus_D.get_output(0))
    y_flat.reshape_dims = (1, d_inner)
    y_for_gate = y_flat.get_output(0)
    if y_for_gate.dtype != gate.dtype:
        y_for_gate = network.add_cast(y_for_gate, gate.dtype).get_output(0)

    # ===== 7. Gated Group RMSNorm (norm_before_gate=False) =====
    # HF: output = weight * group_rms_norm(y * silu(gate))
    # Gate is applied BEFORE normalization. RMSNorm is per-group,
    # with group_size = d_inner // n_groups.
    mamba_norm_w = weights[f"{prefix}.mamba_norm"]
    eps_small = graph_ops.add_constant(
        network, (1, 1), np.array([1e-5], dtype=np.float32), dtype=np.float32
    )

    # Step 1: Apply silu(gate) to y BEFORE norm
    gate_activated = graph_ops.add_activation(network, gate, "silu", dtype=dtype)
    y_gated = network.add_elementwise(y_for_gate, gate_activated, trt.ElementWiseOperation.PROD)

    # Step 2: Group RMSNorm — reshape to [n_groups, group_size], norm per group
    group_size = d_inner // n_groups
    y_grouped = network.add_shuffle(y_gated.get_output(0))
    y_grouped.reshape_dims = (n_groups, group_size)
    norm_input = y_grouped.get_output(0)
    norm_output_dtype = norm_input.dtype
    if dtype != np.float32:
        norm_input = network.add_cast(norm_input, trt.float32).get_output(0)

    sq = network.add_elementwise(norm_input, norm_input, trt.ElementWiseOperation.PROD)
    mean = network.add_reduce(sq.get_output(0), trt.ReduceOperation.AVG, 1 << 1, keep_dims=True)
    denom_in = network.add_elementwise(mean.get_output(0), eps_small, trt.ElementWiseOperation.SUM)
    sqrt_l = network.add_unary(denom_in.get_output(0), trt.UnaryOperation.SQRT)
    recip = network.add_unary(sqrt_l.get_output(0), trt.UnaryOperation.RECIP)
    normalized = network.add_elementwise(
        norm_input, recip.get_output(0), trt.ElementWiseOperation.PROD
    )

    # Reshape back to [1, d_inner] and apply weight
    y_flat_normed = network.add_shuffle(normalized.get_output(0))
    y_flat_normed.reshape_dims = (1, d_inner)
    gamma_t = graph_ops.add_constant(network, (1, d_inner), mamba_norm_w, dtype=np.float32)
    gated = network.add_elementwise(
        y_flat_normed.get_output(0), gamma_t, trt.ElementWiseOperation.PROD
    )
    gated_tensor = gated.get_output(0)
    if gated_tensor.dtype != norm_output_dtype:
        gated_tensor = network.add_cast(gated_tensor, norm_output_dtype).get_output(0)

    # ===== 8. Output projection + residual =====
    out = graph_ops.add_matmul_rhs_constant(
        network,
        gated_tensor,
        d_inner,
        hidden_size,
        weights[f"{prefix}.mamba_out_proj"],
        dtype=dtype,
    )

    residual = network.add_elementwise(hidden, out, trt.ElementWiseOperation.SUM)

    return {
        "hidden": residual.get_output(0),
        "present_conv": present_conv,
        "present_ssm": present_ssm,
    }


def _add_mlp_layer(
    *,
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    eps_tensor: trt.ITensor,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    mlp_size: int,
    dtype: np.dtype = np.float32,
) -> dict[str, trt.ITensor]:
    """Add MLP layer: RMSNorm -> up -> relu2 -> down -> residual."""
    normed = graph_ops.add_rms_norm(
        network, hidden, hidden_size, weights[f"{prefix}.input_norm"], eps_tensor, dtype=dtype
    )

    up = graph_ops.add_matmul_rhs_constant(
        network, normed, hidden_size, mlp_size, weights[f"{prefix}.w_up"], dtype=dtype
    )
    activated = graph_ops.add_activation(network, up, "relu2", dtype=dtype)
    down = graph_ops.add_matmul_rhs_constant(
        network, activated, mlp_size, hidden_size, weights[f"{prefix}.w_down"], dtype=dtype
    )

    residual = network.add_elementwise(hidden, down, trt.ElementWiseOperation.SUM)

    return {"hidden": residual.get_output(0)}


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
    """Build the complete nemotron_h bundle inside its owning family module."""
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
        raise NotImplementedError("nemotron_h does not support context-parallel builds")
    if options.get("dynamic_kv_cache") or options.get("triattention_stats_path"):
        raise ValueError("nemotron_h does not use a decoder KV-cache runtime")

    config = ModelConfig.from_dir(model_path)
    config.raw["_model_dir"] = str(model_path)
    config.raw["_decoder_engine_layout"] = decoder_engine_layout
    config.raw["_fp32_layers"] = sorted(set(options.get("fp32_layers") or ()))
    config.raw["_family_build_options"] = dict(options.get("family_build_options") or {})
    config.raw["_parallel_build_enabled"] = bool(parallel.enabled)
    config.raw["_rtx_build_requested"] = bool(options.get("rtx"))
    config.raw["_runtime_dynamic_kv_requested"] = False
    config.raw["_quantized_build_requested"] = bool(options.get("quantize"))
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
        require_tensorrt_11_for_tensor_parallel(
            parallel, feature="nemotron_h tensor-parallel builds"
        )
        if quant_ctx is not None:
            raise ValueError("nemotron_h tensor-parallel builds do not support quantization")

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
    overrides = get_bundle_config_overrides(config)
    if overrides is not None:
        merged = dict(overrides)
        merged.update(runtime_config)
        merged.update(overrides)
        runtime_config = merged

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
