# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chronos-Bolt native TensorRT family plugin."""

from __future__ import annotations

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
                layer in fp32_layers_set
                and name.startswith(f"encoder.block.{layer}.")
                for layer in range(num_encoder_layers)
            )
            or any(
                num_encoder_layers + layer in fp32_layers_set
                and name.startswith(f"decoder.block.{layer}.")
                for layer in range(num_decoder_layers)
            )
            or (
                input_selector in fp32_layers_set
                and name.startswith("input_patch_embedding.")
            )
            or (
                output_selector in fp32_layers_set
                and name.startswith("output_patch_embedding.")
            )
            or (shared_selector in fp32_layers_set and name == "shared.weight")
        )
        decoder_self_qk = (
            ".layer.0.SelfAttention.q.weight" in name
            or ".layer.0.SelfAttention.k.weight" in name
        ) and name.startswith("decoder.block.")
        if (
            decoder_self_qk
            and decoder_self_qk_selector not in fp32_layers_set
        ):
            selected_fp32 = False
        tensor_precision = (
            "fp32"
            if selected_fp32 and (
                bias_selector in fp32_layers_set or not name.endswith(".bias"))
            else precision
        )
        if arr.ndim == 2 and "relative_attention_bias" not in name and (
            ".SelfAttention." in name
            or ".EncDecAttention." in name
            or ".DenseReluDense." in name
        ):
            weights[name] = _transpose_2d(
                arr, name, precision=tensor_precision)
        else:
            dtype = np.float32 if (
                (
                    selected_fp32
                    and (
                        bias_selector in fp32_layers_set
                        or not name.endswith(".bias")
                    )
                )
                or
                name.endswith("layer_norm.weight")
                or name.endswith("final_layer_norm.weight")
                or "relative_attention_bias" in name
            ) else target_dtype
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
    )
    residual = add_linear(
        network,
        inp,
        weights[f"{prefix}.residual_layer.weight"],
        weights.get(f"{prefix}.residual_layer.bias"),
        precision=precision,
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
    invalid = network.add_elementwise(
        one, attention_mask, trt.ElementWiseOperation.SUB
    ).get_output(0)
    invalid = network.add_elementwise(
        invalid, add_scalar(network, (1, seq_len), -1.0e9),
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
            network, (1, num_heads, seq_len, seq_len),
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
        network, hidden, hidden_size, num_heads * head_dim, weights[f"{prefix}.q.weight"])
    k = graph_ops.add_matmul_rhs_constant(
        network, kv_in, hidden_size, num_heads * head_dim, weights[f"{prefix}.k.weight"])
    v = graph_ops.add_matmul_rhs_constant(
        network, kv_in, hidden_size, num_heads * head_dim, weights[f"{prefix}.v.weight"])
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
        network, ctx, num_heads * head_dim, hidden_size, weights[f"{prefix}.o.weight"])


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
        network, norm, hidden_size, d_ff, weights[f"{prefix}.DenseReluDense.wi.weight"])
    ff = network.add_activation(ff, trt.ActivationType.RELU).get_output(0)
    ff = graph_ops.add_matmul_rhs_constant(
        network, ff, d_ff, hidden_size, weights[f"{prefix}.DenseReluDense.wo.weight"])
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
        layer_precision = (
            "fp32"
            if precision == "fp16" and layer_idx in fp32_layers
            else precision
        )
        layer_dtype = (
            trt.float16 if layer_precision == "fp16" else trt.float32)
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
        network, tuple(weights["shared.weight"].shape), weights["shared.weight"], dtype=np.float32)
    token_id = graph_ops.add_constant(
        network, (1,), np.array([0], dtype=np.int32), dtype=np.int32)
    hidden = network.add_gather(shared, token_id, 0).get_output(0)

    one_mask = graph_ops.add_constant(
        network, (1, num_heads, 1, 1), np.zeros((1, num_heads, 1, 1), dtype=np.float32),
        dtype=np.float32)
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
        cross_mask, start=(0, 0, 0, 0), shape=(1, num_heads, 1, seq_len),
        stride=(1, 1, 1, 1)).get_output(0)

    for layer_idx in range(num_layers):
        selector = encoder_layer_count + layer_idx
        layer_precision = (
            "fp32"
            if precision == "fp16" and selector in fp32_layers
            else precision
        )
        layer_dtype = (
            trt.float16 if layer_precision == "fp16" else trt.float32)
        if hidden.dtype != layer_dtype:
            hidden = network.add_cast(hidden, layer_dtype).get_output(0)
        pfx = f"decoder.block.{layer_idx}"
        norm = graph_ops.add_rms_norm(
            network, hidden, hidden_size,
            weights[f"{pfx}.layer.0.layer_norm.weight"], eps_t,
            dtype=(np.float16 if hidden.dtype == trt.float16 else np.float32))
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
            network, hidden, hidden_size,
            weights[f"{pfx}.layer.1.layer_norm.weight"], eps_t,
            dtype=(np.float16 if hidden.dtype == trt.float16 else np.float32))
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
        chronos, ("context_length", "input_length", "max_context_length"), 2048)
    patch_size = int(chronos.get("input_patch_size", 16))
    patch_stride = int(chronos.get("input_patch_stride", patch_size))
    if patch_size != patch_stride:
        raise NotImplementedError("Chronos-Bolt native TRT builder requires non-overlapping input patches")
    num_patches = context_length // patch_size
    seq_len = num_patches + (1 if bool(chronos.get("use_reg_token", False)) else 0)
    hidden_size = int(raw.get("d_model", 256))
    num_encoder_layers = int(raw.get("num_layers", 4))
    num_decoder_layers = int(
        raw.get("num_decoder_layers", num_encoder_layers))
    input_selector = num_encoder_layers + num_decoder_layers
    output_selector = input_selector + 1
    decoder_self_qk_selector = input_selector + 4
    fp32_layers = frozenset(
        int(layer) for layer in raw.get("_fp32_layers", ()))
    invalid_fp32_layers = sorted(
        layer for layer in fp32_layers
        if layer < 0 or layer > decoder_self_qk_selector)
    if invalid_fp32_layers:
        raise ValueError(
            "fp32_layers contains out-of-range Chronos-Bolt selectors: "
            f"{invalid_fp32_layers}")
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
        denom, add_scalar(network, (1, 1), 1.0), trt.ElementWiseOperation.MAX).get_output(0)
    loc = network.add_reduce(
        context_zero, trt.ReduceOperation.SUM, 1 << 1, keep_dims=True).get_output(0)
    loc = network.add_elementwise(loc, denom, trt.ElementWiseOperation.DIV).get_output(0)
    centered = network.add_elementwise(context_zero, loc, trt.ElementWiseOperation.SUB).get_output(0)
    centered = network.add_elementwise(centered, mask, trt.ElementWiseOperation.PROD).get_output(0)
    var = network.add_reduce(
        network.add_elementwise(centered, centered, trt.ElementWiseOperation.PROD).get_output(0),
        trt.ReduceOperation.SUM, 1 << 1, keep_dims=True,
    ).get_output(0)
    var = network.add_elementwise(var, denom, trt.ElementWiseOperation.DIV).get_output(0)
    scale = network.add_unary(var, trt.UnaryOperation.SQRT).get_output(0)
    scale = network.add_elementwise(
        scale, add_scalar(network, (1, 1), 1.0e-5),
        trt.ElementWiseOperation.MAX).get_output(0)
    normalized = network.add_elementwise(
        centered, scale, trt.ElementWiseOperation.DIV).get_output(0)

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
        patch_mask, trt.ReduceOperation.SUM, 1 << 3, keep_dims=False).get_output(0)
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
        network, cat.get_output(0), weights,
        prefix="input_patch_embedding",
        precision=(
            "fp32"
            if precision == "fp16" and input_selector in fp32_layers
            else precision
        ),
        activation=str(raw.get("dense_act_fn", "relu")).lower())
    emb2 = network.add_shuffle(emb)
    emb2.reshape_dims = (num_patches, hidden_size)
    emb = emb2.get_output(0)

    if bool(chronos.get("use_reg_token", False)):
        reg_dtype = np.float16 if emb.dtype == trt.float16 else np.float32
        reg = weights["shared.weight"][1:2, :].reshape(
            1, hidden_size).astype(reg_dtype)
        reg_t = graph_ops.add_constant(
            network, (1, hidden_size), reg, dtype=reg_dtype)
        cat_emb = network.add_concatenation([emb, reg_t])
        cat_emb.axis = 0
        emb = cat_emb.get_output(0)
        one = graph_ops.add_constant(network, (1, 1), np.ones((1, 1), dtype=np.float32), dtype=np.float32)
        cat_mask = network.add_concatenation([attention_mask, one])
        cat_mask.axis = 1
        attention_mask = cat_mask.get_output(0)

    eps_t = graph_ops.add_constant(
        network, (1, 1), np.array([float(raw.get("layer_norm_epsilon", 1.0e-6))], dtype=np.float32),
        dtype=np.float32)
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
        network, decoder_hidden, weights,
        prefix="output_patch_embedding",
        precision=(
            "fp32"
            if precision == "fp16" and output_selector in fp32_layers
            else precision
        ),
        activation=str(raw.get("dense_act_fn", "relu")).lower())
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
        pred_t, scale3.get_output(0), trt.ElementWiseOperation.PROD).get_output(0)
    pred_t = network.add_elementwise(
        pred_t, loc3.get_output(0), trt.ElementWiseOperation.SUM).get_output(0)
    add_named_output(network, pred_t, "quantile_preds")

    return build_serialized_network(
        builder, network, precision=precision, verbose=verbose, tag="chronos_bolt")


class ChronosBoltPlugin:
    name = "chronos_bolt"
    runtime_strategy = "chronos_bolt_trt"

    def matches(self, model_type: str) -> bool:
        mt = (model_type or "").lower()
        return (
            mt in {"chronos_bolt", "chronos-bolt", "chronosbolt"}
            or mt.startswith("chronos_bolt")
            or mt.startswith("chronos-bolt")
            or mt.startswith("chronosbolt")
        )

    def matches_config(self, config: ModelConfig | str) -> bool:
        if isinstance(config, str):
            return self.matches(config)
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

    def load_weights(
        self,
        model_dir: str,
        config: ModelConfig,
        *,
        precision: str = "fp32",
    ) -> dict:
        raw = config.raw
        num_encoder_layers = int(raw.get("num_layers", 4))
        return _load_all_tensors(
            model_dir,
            precision=precision,
            fp32_layers=tuple(raw.get("_fp32_layers", ())),
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=int(
                raw.get("num_decoder_layers", num_encoder_layers)),
        )

    def build_engine(
        self,
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
                parallel, feature="Chronos-Bolt replicated tensor-parallel bundles")
            cached = maybe_return_replicated_tp_plan(weights, parallel)
            if cached is not None:
                return cached

        plan = _build_chronos_network(
            config, weights, precision=precision, verbose=verbose)
        cache_replicated_tp_plan(weights, parallel, plan)
        return plan

    def get_bundle_config_overrides(self, config: ModelConfig) -> dict:
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


plugin = ChronosBoltPlugin()
