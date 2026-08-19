# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mamba family model -- Selective State Space Model (SSM).

Mamba replaces attention entirely with a recurrent SSM block. Key differences:
  - NO attention mask, NO position_id, NO KV cache
  - Instead: conv_state + ssm_state per layer (constant memory per step)
  - Input projection splits into x (SSM path) and z (gate)
  - Causal conv1d with cached state for single-step inference
  - Selective scan: input-dependent discretization of continuous SSM

Architecture per layer (single step):
  1. RMSNorm(hidden) -> in_proj -> split into x, z
  2. Conv1d step: shift conv_state left, append x, depthwise convolve
  3. SiLU activation on convolved x
  4. x_proj -> split into dt, B, C (input-dependent SSM params)
  5. dt_proj: dt_rank -> intermediate (with bias + softplus)
  6. Selective scan: ssm_state = exp(dt*A) * ssm_state + dt*B*x; y = C*ssm_state + D*x
  7. Output: y * silu(z) -> out_proj -> residual add

Weight key mapping:
  HF: backbone.layers.{i}.mixer.in_proj.weight    -> [2*d_inner, d_model]
  HF: backbone.layers.{i}.mixer.conv1d.weight      -> [d_inner, 1, conv_kernel]
  HF: backbone.layers.{i}.mixer.conv1d.bias         -> [d_inner]
  HF: backbone.layers.{i}.mixer.x_proj.weight       -> [dt_rank+2*state_size, d_inner]
  HF: backbone.layers.{i}.mixer.dt_proj.weight      -> [d_inner, dt_rank]
  HF: backbone.layers.{i}.mixer.dt_proj.bias        -> [d_inner]
  HF: backbone.layers.{i}.mixer.A_log               -> [d_inner, state_size]
  HF: backbone.layers.{i}.mixer.D                   -> [d_inner]
  HF: backbone.layers.{i}.mixer.out_proj.weight     -> [d_model, d_inner]
  HF: backbone.layers.{i}.norm.weight               -> [d_model]
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
from ...parallel_config import normalize_parallel_config, require_tensorrt_11_for_tensor_parallel


trt = trt_compat.get_trt()

name = "mamba"
runtime_strategy = "mamba_ssm_recurrent"


def matches(config) -> bool:
    """Return whether this module owns the resolved model config."""
    model_type = getattr(config, "model_type", config)
    model_type = str(model_type)
    return model_type.lower() == "mamba"


def load_weights(model_dir: str, config: ModelConfig) -> WeightDict:
    """Load Mamba weights from safetensors."""
    model_dir_path = Path(model_dir)
    readers = _open_safetensors(model_dir_path)

    hidden = config.hidden_size
    vocab = config.vocab_size
    num_layers = config.num_hidden_layers
    d_inner = config.raw.get("intermediate_size", config.raw.get("d_inner", hidden * 2))
    state_size = config.raw.get("state_size", 16)
    conv_kernel = config.raw.get("conv_kernel", 4)
    dt_rank = config.raw.get("time_step_rank", 48)

    weights = WeightDict()

    # Embedding
    embedding = _load_tensor(readers, "backbone.embeddings.weight")
    assert embedding.shape == (vocab, hidden), (
        f"Embedding shape {embedding.shape} != ({vocab}, {hidden})"
    )
    weights["embedding"] = embedding.astype(np.float32)

    for layer_idx in range(num_layers):
        prefix = f"layer.{layer_idx}"
        hf_prefix = f"backbone.layers.{layer_idx}"

        # RMSNorm
        norm = _load_tensor(readers, f"{hf_prefix}.norm.weight")
        weights[f"{prefix}.norm"] = norm.astype(np.float32)

        # in_proj: [2*d_inner, hidden] -> split into x_proj [d_inner, hidden] and z_proj [d_inner, hidden]
        in_proj_raw = _load_tensor(readers, f"{hf_prefix}.mixer.in_proj.weight")
        assert in_proj_raw.shape == (2 * d_inner, hidden), (
            f"in_proj shape {in_proj_raw.shape} != ({2 * d_inner}, {hidden})"
        )
        # Transpose to [hidden, d_inner] for matmul
        weights[f"{prefix}.w_in_x"] = _transpose_2d(
            in_proj_raw[:d_inner, :], "in_proj_x"
        )  # [hidden, d_inner]
        weights[f"{prefix}.w_in_z"] = _transpose_2d(
            in_proj_raw[d_inner:, :], "in_proj_z"
        )  # [hidden, d_inner]
        del in_proj_raw

        # conv1d: [d_inner, 1, conv_kernel] -> flatten to [d_inner, conv_kernel]
        conv1d_weight = _load_tensor(readers, f"{hf_prefix}.mixer.conv1d.weight")
        # Shape: [d_inner, 1, conv_kernel] -> squeeze to [d_inner, conv_kernel]
        conv1d_weight = conv1d_weight.reshape(d_inner, conv_kernel).astype(np.float32)
        weights[f"{prefix}.conv1d_weight"] = conv1d_weight

        conv1d_bias = _load_tensor(readers, f"{hf_prefix}.mixer.conv1d.bias")
        weights[f"{prefix}.conv1d_bias"] = conv1d_bias.astype(np.float32)

        # x_proj: [dt_rank + 2*state_size, d_inner] -- projects x to dt, B, C
        x_proj_raw = _load_tensor(readers, f"{hf_prefix}.mixer.x_proj.weight")
        expected_xproj_rows = dt_rank + 2 * state_size
        assert x_proj_raw.shape == (expected_xproj_rows, d_inner), (
            f"x_proj shape {x_proj_raw.shape} != ({expected_xproj_rows}, {d_inner})"
        )
        # Split into dt_proj_in [dt_rank, d_inner], B_proj [state_size, d_inner], C_proj [state_size, d_inner]
        # Transpose each to [d_inner, out_dim] for matmul
        weights[f"{prefix}.w_dt_in"] = _transpose_2d(
            x_proj_raw[:dt_rank, :], "dt_proj_in"
        )  # [d_inner, dt_rank]
        weights[f"{prefix}.w_B"] = _transpose_2d(
            x_proj_raw[dt_rank : dt_rank + state_size, :], "B_proj"
        )  # [d_inner, state_size]
        weights[f"{prefix}.w_C"] = _transpose_2d(
            x_proj_raw[dt_rank + state_size :, :], "C_proj"
        )  # [d_inner, state_size]
        del x_proj_raw

        # dt_proj: [d_inner, dt_rank] -- projects dt_rank -> d_inner
        dt_proj_raw = _load_tensor(readers, f"{hf_prefix}.mixer.dt_proj.weight")
        assert dt_proj_raw.shape == (d_inner, dt_rank), (
            f"dt_proj shape {dt_proj_raw.shape} != ({d_inner}, {dt_rank})"
        )
        weights[f"{prefix}.w_dt_out"] = _transpose_2d(dt_proj_raw, "dt_proj")  # [dt_rank, d_inner]
        del dt_proj_raw

        dt_proj_bias = _load_tensor(readers, f"{hf_prefix}.mixer.dt_proj.bias")
        weights[f"{prefix}.dt_proj_bias"] = dt_proj_bias.astype(np.float32)

        # A_log: [d_inner, state_size] -- compute A = -exp(A_log)
        A_log = _load_tensor(readers, f"{hf_prefix}.mixer.A_log")
        assert A_log.shape == (d_inner, state_size)
        A = -np.exp(A_log.astype(np.float32))
        weights[f"{prefix}.A"] = A  # [d_inner, state_size]

        # D: [d_inner] -- skip connection scalar
        D = _load_tensor(readers, f"{hf_prefix}.mixer.D")
        weights[f"{prefix}.D"] = D.astype(np.float32)

        # out_proj: [hidden, d_inner] -> transpose to [d_inner, hidden]
        out_proj_raw = _load_tensor(readers, f"{hf_prefix}.mixer.out_proj.weight")
        weights[f"{prefix}.w_out"] = _transpose_2d(out_proj_raw, "out_proj")  # [d_inner, hidden]
        del out_proj_raw

    # Final norm
    final_norm_key = "backbone.norm_f.weight"
    if _has_tensor(readers, final_norm_key):
        weights["final_norm"] = _load_tensor(readers, final_norm_key).astype(np.float32)
    else:
        weights["final_norm"] = np.ones(hidden, dtype=np.float32)

    # LM head (tied to embeddings for mamba-130m-hf)
    lm_head_key = "lm_head.weight"
    if _has_tensor(readers, lm_head_key):
        weights["w_lm_head"] = _transpose_2d(_load_tensor(readers, lm_head_key), "lm_head")
    else:
        # Tied embeddings: [vocab, hidden] -> [hidden, vocab]
        weights["w_lm_head"] = _transpose_2d(embedding.copy(), "embedding_tied")

    # Store Mamba-specific dimensions for engine builder
    weights["_d_inner"] = d_inner  # type: ignore[assignment]
    weights["_state_size"] = state_size  # type: ignore[assignment]
    weights["_conv_kernel"] = conv_kernel  # type: ignore[assignment]
    weights["_dt_rank"] = dt_rank  # type: ignore[assignment]

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
    """Build TRT engine for Mamba SSM.

    max_cache_length is accepted for API compatibility but is not used
    by Mamba (SSM state is constant size regardless of sequence length).

    Engine inputs:
      token_id: int32 [1]
      conv_state_0..N: float32 [1, d_inner * conv_kernel]
      ssm_state_0..N: float32 [1, d_inner * state_size]

    Engine outputs:
      logits: float32 [1, vocab]
      present_conv_0..N: float32 [1, d_inner * conv_kernel]
      present_ssm_0..N: float32 [1, d_inner * state_size]
    """
    parallel = normalize_parallel_config(parallel_config)
    if parallel.enabled:
        require_tensorrt_11_for_tensor_parallel(parallel, feature="Mamba tensor-parallel builds")
        from .tp_builder import build_mamba_tp_engine

        return build_mamba_tp_engine(
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
    d_inner: int = weights["_d_inner"]
    state_size: int = weights["_state_size"]
    conv_kernel: int = weights["_conv_kernel"]
    dt_rank: int = weights["_dt_rank"]
    if precision == "fp16":
        work_np_dtype, work_trt_dtype = np.float16, trt.float16
    elif precision == "fp32":
        work_np_dtype, work_trt_dtype = np.float32, trt.float32
    else:
        raise ValueError(f"Mamba supports precision='fp32' or 'fp16', got {precision!r}")

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    trt_config = builder.create_builder_config()
    trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    # -----------------------------------------------------------
    # Inputs
    # -----------------------------------------------------------
    token_id = network.add_input("token_id", trt.int32, (1,))

    conv_state_inputs = []
    ssm_state_inputs = []
    for i in range(num_layers):
        cs = network.add_input(
            graph_ops.layer_tensor_name("conv_state", i), trt.float32, (d_inner, conv_kernel)
        )
        ss = network.add_input(
            graph_ops.layer_tensor_name("ssm_state", i), trt.float32, (d_inner, state_size)
        )
        conv_state_inputs.append(cs)
        ssm_state_inputs.append(ss)

    # -----------------------------------------------------------
    # Shared constants
    # -----------------------------------------------------------
    embedding_table = graph_ops.add_constant(
        network, (vocab, hidden), weights["embedding"], dtype=work_np_dtype
    )

    eps_tensor = graph_ops.add_constant(
        network, (1, 1), np.array([config.rms_norm_eps], dtype=work_np_dtype), dtype=work_np_dtype
    )

    # -----------------------------------------------------------
    # Embedding lookup
    # -----------------------------------------------------------
    gather = network.add_gather(embedding_table, token_id, 0)
    hidden_state = gather.get_output(0)  # [1, hidden]
    if hidden_state.dtype != work_trt_dtype:
        hidden_state = network.add_cast(hidden_state, work_trt_dtype).get_output(0)

    if debug_layer_outputs:
        _mark_debug_output(network, hidden_state, "debug_embed")

    # -----------------------------------------------------------
    # Mamba layers
    # -----------------------------------------------------------
    present_conv_outputs = []
    present_ssm_outputs = []

    for layer_idx in range(num_layers):
        prefix = f"layer.{layer_idx}"
        conv_state = conv_state_inputs[layer_idx]
        ssm_state = ssm_state_inputs[layer_idx]
        if conv_state.dtype != work_trt_dtype:
            conv_state = network.add_cast(conv_state, work_trt_dtype).get_output(0)

        result = _add_mamba_layer(
            network=network,
            hidden=hidden_state,
            conv_state_in=conv_state,
            ssm_state_in=ssm_state,
            eps_tensor=eps_tensor,
            weights=weights,
            prefix=prefix,
            hidden_size=hidden,
            d_inner=d_inner,
            state_size=state_size,
            conv_kernel=conv_kernel,
            dt_rank=dt_rank,
            dtype=work_np_dtype,
        )

        hidden_state = result["hidden"]
        present_conv_outputs.append(result["present_conv"])
        present_ssm_outputs.append(result["present_ssm"])

        if debug_layer_outputs:
            _mark_debug_output(network, hidden_state, f"debug_hidden_{layer_idx}")

    # -----------------------------------------------------------
    # Final norm
    # -----------------------------------------------------------
    final_norm = weights.get("final_norm")
    if final_norm is not None and len(final_norm) > 0:
        hidden_state = graph_ops.add_rms_norm(
            network, hidden_state, hidden, final_norm, eps_tensor, dtype=work_np_dtype
        )

    # -----------------------------------------------------------
    # LM head (logits)
    # -----------------------------------------------------------
    logits = graph_ops.add_matmul_rhs_constant(
        network, hidden_state, hidden, vocab, weights["w_lm_head"], dtype=work_np_dtype
    )
    # Zero bias
    b_out = np.zeros(vocab, dtype=work_np_dtype)
    logits = graph_ops.add_bias_sum(network, logits, vocab, b_out, dtype=work_np_dtype)
    if logits.dtype != trt.float32:
        logits = network.add_cast(logits, trt.float32).get_output(0)

    logits.name = "logits"
    network.mark_output(logits)

    # -----------------------------------------------------------
    # Present conv/ssm state outputs
    # -----------------------------------------------------------
    for i in range(num_layers):
        pc = present_conv_outputs[i]
        ps = present_ssm_outputs[i]
        if pc.dtype != trt.float32:
            pc = network.add_cast(pc, trt.float32).get_output(0)
        if ps.dtype != trt.float32:
            ps = network.add_cast(ps, trt.float32).get_output(0)
        pc.name = graph_ops.layer_tensor_name("present_conv", i)
        ps.name = graph_ops.layer_tensor_name("present_ssm", i)
        network.mark_output(pc)
        network.mark_output(ps)

    # -----------------------------------------------------------
    # Build engine
    # -----------------------------------------------------------
    if verbose:
        print(
            f"[trtmc build] Building Mamba TRT engine ({num_layers} layers, "
            f"hidden={hidden}, d_inner={d_inner}, state_size={state_size}, "
            f"conv_kernel={conv_kernel}, dt_rank={dt_rank}, "
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


def _add_mamba_layer(
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
    state_size: int,
    conv_kernel: int,
    dt_rank: int,
    dtype: np.dtype = np.float32,
) -> dict[str, trt.ITensor]:
    """Add one Mamba SSM layer.

    Returns: {hidden, present_conv, present_ssm}
    """
    # ===== 1. RMSNorm =====
    normed = graph_ops.add_rms_norm(
        network, hidden, hidden_size, weights[f"{prefix}.norm"], eps_tensor, dtype=dtype
    )

    # ===== 2. Input projection: normed -> x, z =====
    # x: [1, d_inner]  (SSM path)
    x = graph_ops.add_matmul_rhs_constant(
        network, normed, hidden_size, d_inner, weights[f"{prefix}.w_in_x"], dtype=dtype
    )
    # z: [1, d_inner]  (gate path)
    z = graph_ops.add_matmul_rhs_constant(
        network, normed, hidden_size, d_inner, weights[f"{prefix}.w_in_z"], dtype=dtype
    )

    # ===== 3. Conv1d step =====
    # conv_state_in: [d_inner, conv_kernel]
    # Shift left: drop column 0, append x as new column
    # New conv_state = [conv_state_in[:, 1:], x^T]

    # x reshaped to [d_inner, 1] for concatenation
    x_col = network.add_shuffle(x)
    x_col.reshape_dims = (d_inner, 1)

    if conv_kernel > 1:
        # Slice conv_state_in[:, 1:] using a slice layer
        # conv_state_in shape: [d_inner, conv_kernel]
        # We want columns [1:conv_kernel] -> shape [d_inner, conv_kernel-1]
        slice_layer = network.add_slice(
            conv_state_in, start=(0, 1), shape=(d_inner, conv_kernel - 1), stride=(1, 1)
        )

        # Concatenate: [d_inner, conv_kernel-1] + [d_inner, 1] = [d_inner, conv_kernel]
        new_conv_state = network.add_concatenation([slice_layer.get_output(0), x_col.get_output(0)])
        new_conv_state.axis = 1
        present_conv = new_conv_state.get_output(0)  # [d_inner, conv_kernel]
    else:
        # conv_kernel == 1: conv_state is just x
        present_conv = x_col.get_output(0)

    # Depthwise convolution: element-wise multiply conv_state by weights, then sum over kernel dim
    # conv1d_weight: [d_inner, conv_kernel]
    # present_conv: [d_inner, conv_kernel]
    # Result: sum(present_conv * conv1d_weight, dim=1) + bias -> [d_inner]

    conv_w = graph_ops.add_constant(
        network, (d_inner, conv_kernel), weights[f"{prefix}.conv1d_weight"], dtype=dtype
    )
    conv_prod = network.add_elementwise(present_conv, conv_w, trt.ElementWiseOperation.PROD)
    # Reduce sum over axis 1 (kernel dim): [d_inner, conv_kernel] -> [d_inner, 1]
    conv_sum = network.add_reduce(
        conv_prod.get_output(0), trt.ReduceOperation.SUM, 1 << 1, keep_dims=True
    )
    # Reshape to [1, d_inner]
    conv_flat = network.add_shuffle(conv_sum.get_output(0))
    conv_flat.reshape_dims = (1, d_inner)

    # Add conv bias
    conv_out = graph_ops.add_bias_sum(
        network, conv_flat.get_output(0), d_inner, weights[f"{prefix}.conv1d_bias"], dtype=dtype
    )

    # SiLU activation on conv output
    conv_activated = graph_ops.add_activation(network, conv_out, "silu", dtype=dtype)
    # conv_activated: [1, d_inner]

    # ===== 4. SSM parameters from x =====
    # x_proj splits into dt, B, C
    # dt_in: [1, dt_rank]
    dt_in = graph_ops.add_matmul_rhs_constant(
        network, conv_activated, d_inner, dt_rank, weights[f"{prefix}.w_dt_in"], dtype=dtype
    )
    # B: [1, state_size]
    B = graph_ops.add_matmul_rhs_constant(
        network, conv_activated, d_inner, state_size, weights[f"{prefix}.w_B"], dtype=dtype
    )
    # C: [1, state_size]
    C = graph_ops.add_matmul_rhs_constant(
        network, conv_activated, d_inner, state_size, weights[f"{prefix}.w_C"], dtype=dtype
    )

    # ===== 5. dt_proj: dt_rank -> d_inner with bias + softplus =====
    dt = graph_ops.add_matmul_rhs_constant(
        network, dt_in, dt_rank, d_inner, weights[f"{prefix}.w_dt_out"], dtype=dtype
    )  # [1, d_inner]
    dt = graph_ops.add_bias_sum(
        network, dt, d_inner, weights[f"{prefix}.dt_proj_bias"], dtype=dtype
    )

    # Softplus and the recurrent scan stay FP32. Real Mamba checkpoints contain
    # A = -exp(A_log) values well outside FP16 range, so casting A would create
    # infinities before the first decode step.
    dt_fp32 = dt
    if dt_fp32.dtype != trt.float32:
        dt_fp32 = network.add_cast(dt_fp32, trt.float32).get_output(0)

    # Softplus: log(1 + exp(x))
    # For numerical stability: softplus(x) = x when x > 20, else log(1+exp(x))
    # In TRT: exp -> add 1 -> log
    dt_exp = network.add_unary(dt_fp32, trt.UnaryOperation.EXP)
    one = graph_ops.add_constant(network, (1, 1), np.array([1.0], dtype=np.float32))
    dt_exp_p1 = network.add_elementwise(dt_exp.get_output(0), one, trt.ElementWiseOperation.SUM)
    dt_softplus = network.add_unary(dt_exp_p1.get_output(0), trt.UnaryOperation.LOG)
    dt_final = dt_softplus.get_output(0)  # [1, d_inner]

    # ===== 6. Selective scan =====
    # A: [d_inner, state_size] (negative, precomputed as -exp(A_log))
    # dt: [1, d_inner]
    # B: [1, state_size]
    # C: [1, state_size]
    # ssm_state_in: [d_inner, state_size]
    # conv_activated (x after conv): [1, d_inner]

    # Discretize A: A_bar = exp(dt * A)
    # dt: [1, d_inner] -> reshape to [d_inner, 1] for broadcasting with A [d_inner, state_size]
    dt_col = network.add_shuffle(dt_final)
    dt_col.reshape_dims = (d_inner, 1)

    A_const = graph_ops.add_constant(network, (d_inner, state_size), weights[f"{prefix}.A"])

    # dt * A: [d_inner, 1] * [d_inner, state_size] -> [d_inner, state_size] (broadcast)
    dtA = network.add_elementwise(dt_col.get_output(0), A_const, trt.ElementWiseOperation.PROD)
    # exp(dt * A)
    A_bar = network.add_unary(dtA.get_output(0), trt.UnaryOperation.EXP)
    # A_bar: [d_inner, state_size]

    # Discretize B: dt_B = dt * B
    # dt: [d_inner, 1], B: [1, state_size] -> broadcast to [d_inner, state_size]
    B_fp32 = B
    if B_fp32.dtype != trt.float32:
        B_fp32 = network.add_cast(B_fp32, trt.float32).get_output(0)
    B_reshape = network.add_shuffle(B_fp32)
    B_reshape.reshape_dims = (1, state_size)
    dt_B = network.add_elementwise(
        dt_col.get_output(0), B_reshape.get_output(0), trt.ElementWiseOperation.PROD
    )
    # dt_B: [d_inner, state_size]

    # x for scan: conv_activated [1, d_inner] -> [d_inner, 1]
    scan_x = conv_activated
    if scan_x.dtype != trt.float32:
        scan_x = network.add_cast(scan_x, trt.float32).get_output(0)
    x_col2 = network.add_shuffle(scan_x)
    x_col2.reshape_dims = (d_inner, 1)

    # dt_B * x: [d_inner, state_size] * [d_inner, 1] -> [d_inner, state_size]
    dtBx = network.add_elementwise(
        dt_B.get_output(0), x_col2.get_output(0), trt.ElementWiseOperation.PROD
    )

    # SSM update: new_ssm = A_bar * old_ssm + dt_B * x
    decay = network.add_elementwise(
        A_bar.get_output(0), ssm_state_in, trt.ElementWiseOperation.PROD
    )
    new_ssm = network.add_elementwise(
        decay.get_output(0), dtBx.get_output(0), trt.ElementWiseOperation.SUM
    )
    present_ssm = new_ssm.get_output(0)  # [d_inner, state_size]

    # Output: y = C * new_ssm  (sum over state_size dim)
    # C: [1, state_size] -> [1, state_size], new_ssm: [d_inner, state_size]
    # We want: for each d in d_inner: y[d] = sum_s(C[s] * new_ssm[d, s])
    # = matmul new_ssm @ C^T
    # new_ssm [d_inner, state_size] @ C^T [state_size, 1] -> [d_inner, 1]
    C_fp32 = C
    if C_fp32.dtype != trt.float32:
        C_fp32 = network.add_cast(C_fp32, trt.float32).get_output(0)
    C_reshape2 = network.add_shuffle(C_fp32)
    C_reshape2.reshape_dims = (state_size, 1)
    y_matmul = network.add_matrix_multiply(
        present_ssm, trt.MatrixOperation.NONE, C_reshape2.get_output(0), trt.MatrixOperation.NONE
    )
    # y_matmul: [d_inner, 1] -> reshape to [1, d_inner]
    y_flat = network.add_shuffle(y_matmul.get_output(0))
    y_flat.reshape_dims = (1, d_inner)

    # D * x (skip connection): D [d_inner] * conv_activated [1, d_inner]
    D_const = graph_ops.add_constant(network, (1, d_inner), weights[f"{prefix}.D"])
    Dx = network.add_elementwise(D_const, scan_x, trt.ElementWiseOperation.PROD)

    # y = y + D*x
    y = network.add_elementwise(
        y_flat.get_output(0), Dx.get_output(0), trt.ElementWiseOperation.SUM
    )
    # y: [1, d_inner]

    # ===== 7. Gate and output =====
    # y * silu(z)
    y_for_gate = y.get_output(0)
    if y_for_gate.dtype != hidden.dtype:
        y_for_gate = network.add_cast(y_for_gate, hidden.dtype).get_output(0)
    z_activated = graph_ops.add_activation(network, z, "silu", dtype=dtype)
    gated = network.add_elementwise(y_for_gate, z_activated, trt.ElementWiseOperation.PROD)

    # out_proj: [1, d_inner] @ [d_inner, hidden] -> [1, hidden]
    out = graph_ops.add_matmul_rhs_constant(
        network, gated.get_output(0), d_inner, hidden_size, weights[f"{prefix}.w_out"], dtype=dtype
    )

    # Residual connection
    residual = network.add_elementwise(hidden, out, trt.ElementWiseOperation.SUM)

    return {
        "hidden": residual.get_output(0),
        "present_conv": present_conv,
        "present_ssm": present_ssm,
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
    """Build the complete mamba bundle inside its owning family module."""
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
        raise NotImplementedError("mamba does not support context-parallel builds")
    if options.get("dynamic_kv_cache") or options.get("triattention_stats_path"):
        raise ValueError("mamba does not use a decoder KV-cache runtime")

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
        require_tensorrt_11_for_tensor_parallel(parallel, feature="mamba tensor-parallel builds")
        if quant_ctx is not None:
            raise ValueError("mamba tensor-parallel builds do not support quantization")

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
