# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chronos-Bolt native TensorRT family model."""

from __future__ import annotations

import sys
import json
import re
import tempfile
import time

from pathlib import Path
from typing import Any

import numpy as np

from tensorrt_model_connect import trt_compat

from . import graph_ops
from .checkpoint_mapper import (
    WeightDict,
    _load_tensor,
    _open_safetensors,
    _target_np_dtype,
    _transpose_2d,
)
from .config import ModelConfig
from ...parallel_config import normalize_parallel_config, require_tensorrt_11_for_tensor_parallel
from .time_series_trt import (
    add_gelu,
    add_linear,
    add_named_output,
    add_patchify,
    add_scalar,
    build_serialized_network,
    cache_replicated_tp_plan,
    create_network,
    maybe_return_replicated_tp_plan,
)


trt = trt_compat.get_trt()


def _chronos_raw_config(config: Any) -> dict[str, Any]:
    raw = getattr(config, "raw", {}) or {}
    chronos_cfg = raw.get("chronos_config")
    if isinstance(chronos_cfg, dict):
        return chronos_cfg
    return raw


def _first_positive_int(raw: dict[str, Any], keys: tuple[str, ...], fallback: int) -> int:
    for key in keys:
        value = raw.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return fallback


def _load_all_tensors(
    model_dir: str | Path,
    *,
    precision: str,
    fp32_layers: tuple[int, ...],
    num_encoder_layers: int,
    num_decoder_layers: int,
) -> WeightDict:
    readers = _open_safetensors(Path(model_dir))
    target_dtype = _target_np_dtype(precision)
    weights = WeightDict()
    fp32_layers_set = frozenset(fp32_layers)
    # Encoder/decoder blocks are followed by input, output, shared, bias, Q/K selectors.
    input_selector = num_encoder_layers + num_decoder_layers
    output_selector = input_selector + 1
    shared_selector = input_selector + 2
    bias_selector = input_selector + 3
    decoder_self_qk_selector = input_selector + 4
    tensor_map = getattr(readers, "tensor_map", {})
    for name in sorted(tensor_map):
        arr = _load_tensor(readers, name)
        selected_fp32 = (
            any(
                layer in fp32_layers_set and name.startswith(f"encoder.block.{layer}.")
                for layer in range(num_encoder_layers)
            )
            or any(
                num_encoder_layers + layer in fp32_layers_set
                and name.startswith(f"decoder.block.{layer}.")
                for layer in range(num_decoder_layers)
            )
            or (input_selector in fp32_layers_set and name.startswith("input_patch_embedding."))
            or (output_selector in fp32_layers_set and name.startswith("output_patch_embedding."))
            or (shared_selector in fp32_layers_set and name == "shared.weight")
        )
        decoder_self_qk = (
            ".layer.0.SelfAttention.q.weight" in name or ".layer.0.SelfAttention.k.weight" in name
        ) and name.startswith("decoder.block.")
        if decoder_self_qk and decoder_self_qk_selector not in fp32_layers_set:
            selected_fp32 = False
        tensor_precision = (
            "fp32"
            if selected_fp32 and (bias_selector in fp32_layers_set or not name.endswith(".bias"))
            else precision
        )
        if (
            arr.ndim == 2
            and "relative_attention_bias" not in name
            and (
                ".SelfAttention." in name
                or ".EncDecAttention." in name
                or ".DenseReluDense." in name
            )
        ):
            weights[name] = _transpose_2d(arr, name, precision=tensor_precision)
        else:
            dtype = (
                np.float32
                if (
                    (
                        selected_fp32
                        and (bias_selector in fp32_layers_set or not name.endswith(".bias"))
                    )
                    or name.endswith("layer_norm.weight")
                    or name.endswith("final_layer_norm.weight")
                    or "relative_attention_bias" in name
                )
                else target_dtype
            )
            weights[name] = np.ascontiguousarray(arr, dtype=dtype)
    return weights


def _is_finite(network: trt.INetworkDefinition, x: trt.ITensor) -> trt.ITensor:
    eq = network.add_elementwise(x, x, trt.ElementWiseOperation.EQUAL).get_output(0)
    return eq


def _add_residual_block(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weights: WeightDict,
    *,
    prefix: str,
    precision: str,
    activation: str = "relu",
) -> trt.ITensor:
    hidden = add_linear(
        network,
        inp,
        weights[f"{prefix}.hidden_layer.weight"],
        weights.get(f"{prefix}.hidden_layer.bias"),
        precision=precision,
        fp32_accumulation=(precision == "fp16"),
    )
    if activation == "gelu":
        hidden = add_gelu(network, hidden)
    else:
        hidden = network.add_activation(hidden, trt.ActivationType.RELU).get_output(0)
    out = add_linear(
        network,
        hidden,
        weights[f"{prefix}.output_layer.weight"],
        weights.get(f"{prefix}.output_layer.bias"),
        precision=precision,
        fp32_accumulation=(precision == "fp16"),
    )
    residual = add_linear(
        network,
        inp,
        weights[f"{prefix}.residual_layer.weight"],
        weights.get(f"{prefix}.residual_layer.bias"),
        precision=precision,
        fp32_accumulation=(precision == "fp16"),
    )
    return network.add_elementwise(out, residual, trt.ElementWiseOperation.SUM).get_output(0)


def _make_encoder_mask(
    network: trt.INetworkDefinition,
    attention_mask: trt.ITensor,
    *,
    seq_len: int,
    num_heads: int,
    rel_bias: np.ndarray | None,
    num_buckets: int,
    max_distance: int,
) -> trt.ITensor:
    one = add_scalar(network, (1, seq_len), 1.0)
    invalid = network.add_elementwise(one, attention_mask, trt.ElementWiseOperation.SUB).get_output(
        0
    )
    invalid = network.add_elementwise(
        invalid,
        add_scalar(network, (1, seq_len), -1.0e9),
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    mask = network.add_shuffle(invalid)
    mask.reshape_dims = (1, 1, 1, seq_len)
    mask_t = mask.get_output(0)
    if rel_bias is not None:
        buckets = graph_ops.make_t5_relative_position_bias(
            num_heads=num_heads,
            max_seq_len=seq_len,
            num_buckets=num_buckets,
            max_distance=max_distance,
        )
        bias = rel_bias[buckets.flatten()].reshape(seq_len, seq_len, num_heads).transpose(2, 0, 1)
        bias_t = graph_ops.add_constant(
            network,
            (1, num_heads, seq_len, seq_len),
            bias.reshape(1, num_heads, seq_len, seq_len).astype(np.float32),
            dtype=np.float32,
        )
        mask_t = network.add_elementwise(mask_t, bias_t, trt.ElementWiseOperation.SUM).get_output(0)
        return mask_t
    mask_heads = network.add_concatenation([mask_t] * num_heads)
    mask_heads.axis = 1
    return mask_heads.get_output(0)


def _add_t5_attention_rows(
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    weights: WeightDict,
    *,
    prefix: str,
    hidden_size: int,
    num_heads: int,
    head_dim: int,
    q_seq: int,
    kv: trt.ITensor | None = None,
    kv_seq: int | None = None,
    mask: trt.ITensor | None = None,
) -> trt.ITensor:
    kv_in = hidden if kv is None else kv
    kv_seq = q_seq if kv_seq is None else kv_seq
    q = graph_ops.add_matmul_rhs_constant(
        network,
        hidden,
        hidden_size,
        num_heads * head_dim,
        weights[f"{prefix}.q.weight"],
        fp32_accumulation=(hidden.dtype == trt.float16),
    )
    k = graph_ops.add_matmul_rhs_constant(
        network,
        kv_in,
        hidden_size,
        num_heads * head_dim,
        weights[f"{prefix}.k.weight"],
        fp32_accumulation=(kv_in.dtype == trt.float16),
    )
    v = graph_ops.add_matmul_rhs_constant(
        network,
        kv_in,
        hidden_size,
        num_heads * head_dim,
        weights[f"{prefix}.v.weight"],
        fp32_accumulation=(kv_in.dtype == trt.float16),
    )
    if mask is not None and mask.dtype != q.dtype:
        mask = network.add_cast(mask, q.dtype).get_output(0)
    ctx = graph_ops.add_attention_from_rows(
        network,
        q,
        k,
        v,
        num_heads=num_heads,
        head_dim=head_dim,
        q_seq=q_seq,
        kv_seq=kv_seq,
        mask=mask,
        scale=1.0,
    )
    return graph_ops.add_matmul_rhs_constant(
        network,
        ctx,
        num_heads * head_dim,
        hidden_size,
        weights[f"{prefix}.o.weight"],
        fp32_accumulation=(ctx.dtype == trt.float16),
    )


def _add_t5_ffn(
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    weights: WeightDict,
    *,
    prefix: str,
    hidden_size: int,
    d_ff: int,
    eps_t: trt.ITensor,
) -> trt.ITensor:
    norm = graph_ops.add_rms_norm(
        network,
        hidden,
        hidden_size,
        weights[f"{prefix}.layer_norm.weight"],
        eps_t,
        dtype=(np.float16 if hidden.dtype == trt.float16 else np.float32),
    )
    ff = graph_ops.add_matmul_rhs_constant(
        network,
        norm,
        hidden_size,
        d_ff,
        weights[f"{prefix}.DenseReluDense.wi.weight"],
        fp32_accumulation=(norm.dtype == trt.float16),
    )
    ff = network.add_activation(ff, trt.ActivationType.RELU).get_output(0)
    ff = graph_ops.add_matmul_rhs_constant(
        network,
        ff,
        d_ff,
        hidden_size,
        weights[f"{prefix}.DenseReluDense.wo.weight"],
        fp32_accumulation=(ff.dtype == trt.float16),
    )
    return network.add_elementwise(hidden, ff, trt.ElementWiseOperation.SUM).get_output(0)


def _add_encoder(
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    attention_mask: trt.ITensor,
    weights: WeightDict,
    *,
    raw: dict[str, Any],
    seq_len: int,
    eps_t: trt.ITensor,
    precision: str,
    fp32_layers: frozenset[int],
) -> trt.ITensor:
    hidden_size = int(raw.get("d_model", 256))
    num_heads = int(raw.get("num_heads", 4))
    head_dim = int(raw.get("d_kv", hidden_size // num_heads))
    d_ff = int(raw.get("d_ff", 1024))
    num_layers = int(raw.get("num_layers", 4))
    rel_bias = weights.get("encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight")
    enc_mask = _make_encoder_mask(
        network,
        attention_mask,
        seq_len=seq_len,
        num_heads=num_heads,
        rel_bias=rel_bias,
        num_buckets=int(raw.get("relative_attention_num_buckets", 32)),
        max_distance=int(raw.get("relative_attention_max_distance", 128)),
    )
    for layer_idx in range(num_layers):
        layer_precision = "fp32" if precision == "fp16" and layer_idx in fp32_layers else precision
        layer_dtype = trt.float16 if layer_precision == "fp16" else trt.float32
        if hidden.dtype != layer_dtype:
            hidden = network.add_cast(hidden, layer_dtype).get_output(0)
        pfx = f"encoder.block.{layer_idx}"
        norm = graph_ops.add_rms_norm(
            network,
            hidden,
            hidden_size,
            weights[f"{pfx}.layer.0.layer_norm.weight"],
            eps_t,
            dtype=(np.float16 if hidden.dtype == trt.float16 else np.float32),
        )
        attn = _add_t5_attention_rows(
            network,
            norm,
            weights,
            prefix=f"{pfx}.layer.0.SelfAttention",
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            q_seq=seq_len,
            mask=enc_mask,
        )
        hidden = network.add_elementwise(hidden, attn, trt.ElementWiseOperation.SUM).get_output(0)
        hidden = _add_t5_ffn(
            network,
            hidden,
            weights,
            prefix=f"{pfx}.layer.1",
            hidden_size=hidden_size,
            d_ff=d_ff,
            eps_t=eps_t,
        )
    return graph_ops.add_rms_norm(
        network,
        hidden,
        hidden_size,
        weights["encoder.final_layer_norm.weight"],
        eps_t,
        dtype=(np.float16 if hidden.dtype == trt.float16 else np.float32),
    )


def _add_decoder(
    network: trt.INetworkDefinition,
    encoder_hidden: trt.ITensor,
    encoder_mask: trt.ITensor,
    weights: WeightDict,
    *,
    raw: dict[str, Any],
    seq_len: int,
    eps_t: trt.ITensor,
    precision: str,
    fp32_layers: frozenset[int],
    encoder_layer_count: int,
) -> trt.ITensor:
    hidden_size = int(raw.get("d_model", 256))
    num_heads = int(raw.get("num_heads", 4))
    head_dim = int(raw.get("d_kv", hidden_size // num_heads))
    d_ff = int(raw.get("d_ff", 1024))
    num_layers = int(raw.get("num_decoder_layers", raw.get("num_layers", 4)))
    shared = graph_ops.add_constant(
        network, tuple(weights["shared.weight"].shape), weights["shared.weight"], dtype=np.float32
    )
    token_id = graph_ops.add_constant(network, (1,), np.array([0], dtype=np.int32), dtype=np.int32)
    hidden = network.add_gather(shared, token_id, 0).get_output(0)

    one_mask = graph_ops.add_constant(
        network,
        (1, num_heads, 1, 1),
        np.zeros((1, num_heads, 1, 1), dtype=np.float32),
        dtype=np.float32,
    )
    cross_mask = _make_encoder_mask(
        network,
        encoder_mask,
        seq_len=seq_len,
        num_heads=num_heads,
        rel_bias=None,
        num_buckets=int(raw.get("relative_attention_num_buckets", 32)),
        max_distance=int(raw.get("relative_attention_max_distance", 128)),
    )
    cross_mask_slice = network.add_slice(
        cross_mask, start=(0, 0, 0, 0), shape=(1, num_heads, 1, seq_len), stride=(1, 1, 1, 1)
    ).get_output(0)

    for layer_idx in range(num_layers):
        selector = encoder_layer_count + layer_idx
        layer_precision = "fp32" if precision == "fp16" and selector in fp32_layers else precision
        layer_dtype = trt.float16 if layer_precision == "fp16" else trt.float32
        if hidden.dtype != layer_dtype:
            hidden = network.add_cast(hidden, layer_dtype).get_output(0)
        pfx = f"decoder.block.{layer_idx}"
        norm = graph_ops.add_rms_norm(
            network,
            hidden,
            hidden_size,
            weights[f"{pfx}.layer.0.layer_norm.weight"],
            eps_t,
            dtype=(np.float16 if hidden.dtype == trt.float16 else np.float32),
        )
        attn = _add_t5_attention_rows(
            network,
            norm,
            weights,
            prefix=f"{pfx}.layer.0.SelfAttention",
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            q_seq=1,
            mask=one_mask,
        )
        hidden = network.add_elementwise(hidden, attn, trt.ElementWiseOperation.SUM).get_output(0)
        norm = graph_ops.add_rms_norm(
            network,
            hidden,
            hidden_size,
            weights[f"{pfx}.layer.1.layer_norm.weight"],
            eps_t,
            dtype=(np.float16 if hidden.dtype == trt.float16 else np.float32),
        )
        cross = _add_t5_attention_rows(
            network,
            norm,
            weights,
            prefix=f"{pfx}.layer.1.EncDecAttention",
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            q_seq=1,
            kv=encoder_hidden,
            kv_seq=seq_len,
            mask=cross_mask_slice,
        )
        hidden = network.add_elementwise(hidden, cross, trt.ElementWiseOperation.SUM).get_output(0)
        hidden = _add_t5_ffn(
            network,
            hidden,
            weights,
            prefix=f"{pfx}.layer.2",
            hidden_size=hidden_size,
            d_ff=d_ff,
            eps_t=eps_t,
        )
    return graph_ops.add_rms_norm(
        network,
        hidden,
        hidden_size,
        weights["decoder.final_layer_norm.weight"],
        eps_t,
        dtype=(np.float16 if hidden.dtype == trt.float16 else np.float32),
    )


def _build_chronos_network(
    config: ModelConfig,
    weights: WeightDict,
    *,
    precision: str,
    verbose: bool,
) -> bytes:
    raw = config.raw
    chronos = _chronos_raw_config(config)
    context_length = _first_positive_int(
        chronos, ("context_length", "input_length", "max_context_length"), 2048
    )
    patch_size = int(chronos.get("input_patch_size", 16))
    patch_stride = int(chronos.get("input_patch_stride", patch_size))
    if patch_size != patch_stride:
        raise NotImplementedError(
            "Chronos-Bolt native TRT builder requires non-overlapping input patches"
        )
    num_patches = context_length // patch_size
    seq_len = num_patches + (1 if bool(chronos.get("use_reg_token", False)) else 0)
    hidden_size = int(raw.get("d_model", 256))
    num_encoder_layers = int(raw.get("num_layers", 4))
    num_decoder_layers = int(raw.get("num_decoder_layers", num_encoder_layers))
    input_selector = num_encoder_layers + num_decoder_layers
    output_selector = input_selector + 1
    decoder_self_qk_selector = input_selector + 4
    fp32_layers = frozenset(int(layer) for layer in raw.get("_fp32_layers", ()))
    invalid_fp32_layers = sorted(
        layer for layer in fp32_layers if layer < 0 or layer > decoder_self_qk_selector
    )
    if invalid_fp32_layers:
        raise ValueError(
            f"fp32_layers contains out-of-range Chronos-Bolt selectors: {invalid_fp32_layers}"
        )
    prediction_length = int(chronos.get("prediction_length", 64))
    num_quantiles = len(chronos.get("quantiles", []))

    builder, network = create_network(verbose=verbose)
    context = network.add_input("context", trt.float32, (1, context_length))
    finite = _is_finite(network, context)
    mask = network.add_cast(finite, trt.float32).get_output(0)
    context_zero = network.add_select(
        finite,
        context,
        add_scalar(network, (1, context_length), 0.0),
    ).get_output(0)

    denom = network.add_reduce(mask, trt.ReduceOperation.SUM, 1 << 1, keep_dims=True).get_output(0)
    denom = network.add_elementwise(
        denom, add_scalar(network, (1, 1), 1.0), trt.ElementWiseOperation.MAX
    ).get_output(0)
    loc = network.add_reduce(
        context_zero, trt.ReduceOperation.SUM, 1 << 1, keep_dims=True
    ).get_output(0)
    loc = network.add_elementwise(loc, denom, trt.ElementWiseOperation.DIV).get_output(0)
    centered = network.add_elementwise(context_zero, loc, trt.ElementWiseOperation.SUB).get_output(
        0
    )
    centered = network.add_elementwise(centered, mask, trt.ElementWiseOperation.PROD).get_output(0)
    var = network.add_reduce(
        network.add_elementwise(centered, centered, trt.ElementWiseOperation.PROD).get_output(0),
        trt.ReduceOperation.SUM,
        1 << 1,
        keep_dims=True,
    ).get_output(0)
    var = network.add_elementwise(var, denom, trt.ElementWiseOperation.DIV).get_output(0)
    scale = network.add_unary(var, trt.UnaryOperation.SQRT).get_output(0)
    scale = network.add_elementwise(
        scale, add_scalar(network, (1, 1), 1.0e-5), trt.ElementWiseOperation.MAX
    ).get_output(0)
    normalized = network.add_elementwise(centered, scale, trt.ElementWiseOperation.DIV).get_output(
        0
    )

    norm3 = network.add_shuffle(normalized)
    norm3.reshape_dims = (1, context_length, 1)
    patches = add_patchify(
        network,
        norm3.get_output(0),
        context_length=context_length,
        channels=1,
        patch_length=patch_size,
        patch_stride=patch_stride,
        num_patches=num_patches,
    )
    patches = network.add_shuffle(patches).get_output(0)
    mask3 = network.add_shuffle(mask)
    mask3.reshape_dims = (1, context_length, 1)
    patch_mask = add_patchify(
        network,
        mask3.get_output(0),
        context_length=context_length,
        channels=1,
        patch_length=patch_size,
        patch_stride=patch_stride,
        num_patches=num_patches,
    )
    patch_mask = network.add_shuffle(patch_mask).get_output(0)
    patch_mask_sum = network.add_reduce(
        patch_mask, trt.ReduceOperation.SUM, 1 << 3, keep_dims=False
    ).get_output(0)
    patch_mask_flat = network.add_shuffle(patch_mask_sum)
    patch_mask_flat.reshape_dims = (1, num_patches)
    attention_mask = network.add_elementwise(
        patch_mask_flat.get_output(0),
        add_scalar(network, (1, num_patches), 0.0),
        trt.ElementWiseOperation.GREATER,
    ).get_output(0)
    attention_mask = network.add_cast(attention_mask, trt.float32).get_output(0)

    cat = network.add_concatenation([patches, patch_mask])
    cat.axis = 3
    emb = _add_residual_block(
        network,
        cat.get_output(0),
        weights,
        prefix="input_patch_embedding",
        precision=("fp32" if precision == "fp16" and input_selector in fp32_layers else precision),
        activation=str(raw.get("dense_act_fn", "relu")).lower(),
    )
    emb2 = network.add_shuffle(emb)
    emb2.reshape_dims = (num_patches, hidden_size)
    emb = emb2.get_output(0)

    if bool(chronos.get("use_reg_token", False)):
        reg_dtype = np.float16 if emb.dtype == trt.float16 else np.float32
        reg = weights["shared.weight"][1:2, :].reshape(1, hidden_size).astype(reg_dtype)
        reg_t = graph_ops.add_constant(network, (1, hidden_size), reg, dtype=reg_dtype)
        cat_emb = network.add_concatenation([emb, reg_t])
        cat_emb.axis = 0
        emb = cat_emb.get_output(0)
        one = graph_ops.add_constant(
            network, (1, 1), np.ones((1, 1), dtype=np.float32), dtype=np.float32
        )
        cat_mask = network.add_concatenation([attention_mask, one])
        cat_mask.axis = 1
        attention_mask = cat_mask.get_output(0)

    eps_t = graph_ops.add_constant(
        network,
        (1, 1),
        np.array([float(raw.get("layer_norm_epsilon", 1.0e-6))], dtype=np.float32),
        dtype=np.float32,
    )
    encoder_hidden = _add_encoder(
        network,
        emb,
        attention_mask,
        weights,
        raw=raw,
        seq_len=seq_len,
        eps_t=eps_t,
        precision=precision,
        fp32_layers=fp32_layers,
    )
    decoder_hidden = _add_decoder(
        network,
        encoder_hidden,
        attention_mask,
        weights,
        raw=raw,
        seq_len=seq_len,
        eps_t=eps_t,
        precision=precision,
        fp32_layers=fp32_layers,
        encoder_layer_count=num_encoder_layers,
    )
    preds = _add_residual_block(
        network,
        decoder_hidden,
        weights,
        prefix="output_patch_embedding",
        precision=("fp32" if precision == "fp16" and output_selector in fp32_layers else precision),
        activation=str(raw.get("dense_act_fn", "relu")).lower(),
    )
    pred_shuf = network.add_shuffle(preds)
    pred_shuf.reshape_dims = (1, num_quantiles, prediction_length)
    pred_t = pred_shuf.get_output(0)
    if pred_t.dtype != trt.float32:
        pred_t = network.add_cast(pred_t, trt.float32).get_output(0)
    scale3 = network.add_shuffle(scale)
    scale3.reshape_dims = (1, 1, 1)
    loc3 = network.add_shuffle(loc)
    loc3.reshape_dims = (1, 1, 1)
    pred_t = network.add_elementwise(
        pred_t, scale3.get_output(0), trt.ElementWiseOperation.PROD
    ).get_output(0)
    pred_t = network.add_elementwise(
        pred_t, loc3.get_output(0), trt.ElementWiseOperation.SUM
    ).get_output(0)
    add_named_output(network, pred_t, "quantile_preds")

    return build_serialized_network(
        builder, network, precision=precision, verbose=verbose, tag="chronos_bolt"
    )


name = "chronos_bolt"
runtime_strategy = "chronos_bolt_trt"


def matches(config) -> bool:
    """Return whether this module owns the resolved model config."""
    if isinstance(config, str):
        model_type = config
        mt = (model_type or "").lower()
        return (
            mt in {"chronos_bolt", "chronos-bolt", "chronosbolt"}
            or mt.startswith("chronos_bolt")
            or mt.startswith("chronos-bolt")
            or mt.startswith("chronosbolt")
        )
    raw = getattr(config, "raw", {}) or {}
    if not isinstance(raw, dict):
        return False
    chronos_cfg = raw.get("chronos_config")
    if not isinstance(chronos_cfg, dict):
        return False
    architectures = raw.get("architectures") or []
    if isinstance(architectures, str):
        architectures = [architectures]
    if any("ChronosBoltModelForForecasting" in str(arch) for arch in architectures):
        return True
    return str(getattr(config, "model_type", "")).lower() == "t5"


def load_weights(model_dir: str, config: ModelConfig, *, precision: str = "fp32") -> dict:
    raw = config.raw
    num_encoder_layers = int(raw.get("num_layers", 4))
    return _load_all_tensors(
        model_dir,
        precision=precision,
        fp32_layers=tuple(raw.get("_fp32_layers", ())),
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=int(raw.get("num_decoder_layers", num_encoder_layers)),
    )


def build_engine(
    config: ModelConfig,
    weights: dict,
    max_cache_length: int,
    *,
    precision: str = "fp32",
    verbose: bool = False,
    parallel_config=None,
) -> bytes:
    del max_cache_length
    parallel = normalize_parallel_config(parallel_config)
    if parallel.enabled:
        require_tensorrt_11_for_tensor_parallel(
            parallel, feature="Chronos-Bolt replicated tensor-parallel bundles"
        )
        cached = maybe_return_replicated_tp_plan(weights, parallel)
        if cached is not None:
            return cached

    plan = _build_chronos_network(config, weights, precision=precision, verbose=verbose)
    cache_replicated_tp_plan(weights, parallel, plan)
    return plan


def get_bundle_config_overrides(config: ModelConfig) -> dict:
    raw = _chronos_raw_config(config)
    out = {
        "context_length": _first_positive_int(
            raw,
            ("context_length", "input_length", "max_context_length"),
            fallback=int(config.raw.get("max_cache_length", 2048) or 2048),
        )
    }
    prediction_length = _first_positive_int(
        raw,
        ("prediction_length", "forecast_length", "horizon_length"),
        fallback=0,
    )
    if prediction_length > 0:
        out["prediction_length"] = prediction_length
    quantiles = raw.get("quantiles")
    if isinstance(quantiles, (list, tuple)) and quantiles:
        out["quantiles"] = [float(value) for value in quantiles]
        out["num_quantiles"] = len(quantiles)
    return out


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


def _build_local_engine(config, weights, max_cache_length, precision, verbose, parallel, options):
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
                verbose=verbose,
                parallel_config=parallel,
            )

    target_role = inspection_role()
    if target_role is not None:
        build_role(target_role)
        raise RuntimeError("graph inspection did not reach TensorRT serialization")
    return build_role(role), ("dual_profile" if role == "dual_profile" else "single")


def build(model_dir: str, output_path: str, **options) -> None:
    """Build the complete chronos_bolt bundle inside its owning family module."""
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
        raise NotImplementedError("chronos_bolt does not support context-parallel builds")
    if options.get("dynamic_kv_cache") or options.get("triattention_stats_path"):
        raise ValueError("chronos_bolt does not use a decoder KV-cache runtime")

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
    weights = load_weights(str(model_path), config, precision=precision)
    add_build_timing(timing, "weights_loading_s", time.monotonic() - weights_started)
    write_build_timing(timing)

    quantize = options.get("quantize")
    quant_ctx = None
    quant_plan = None
    if quantize:
        raise ValueError("chronos_bolt does not support quantized builds")

    if parallel.enabled:
        require_tensorrt_11_for_tensor_parallel(
            parallel, feature="chronos_bolt tensor-parallel builds"
        )
        if quant_ctx is not None:
            raise ValueError("chronos_bolt tensor-parallel builds do not support quantization")

    verbose = bool(options.get("verbose"))
    compile_started = time.monotonic()
    if parallel.enabled:
        plans = {
            rank: build_engine(
                config,
                weights,
                max_cache_length,
                precision=precision,
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
            config, weights, max_cache_length, precision, verbose, parallel, options
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
