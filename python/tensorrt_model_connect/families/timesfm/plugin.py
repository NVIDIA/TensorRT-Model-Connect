# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TimesFM native TensorRT family plugin."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tensorrt_model_connect import trt_compat

from . import graph_ops
from .checkpoint_mapper import (
    WeightDict,
    _load_tensor,
    _open_safetensors,
    _target_np_dtype,
)
from .config import ModelConfig
from ...parallel_config import normalize_parallel_config, require_tensorrt_11_for_tensor_parallel
from .time_series_trt import (
    add_linear,
    add_named_output,
    add_scalar,
    build_serialized_network,
    cache_replicated_tp_plan,
    create_network,
    maybe_return_replicated_tp_plan,
)


trt = trt_compat.get_trt()


def _context_length(config, fallback: int) -> int:
    value = getattr(config, "context_length", 0) or fallback
    if value <= 0:
        value = fallback
    return int(value)


def _load_all_tensors(
    model_dir: str | Path,
    *,
    precision: str,
    fp32_layers: tuple[int, ...],
    num_layers: int,
) -> WeightDict:
    readers = _open_safetensors(Path(model_dir))
    target_dtype = _target_np_dtype(precision)
    weights = WeightDict()
    fp32_layers_set = frozenset(fp32_layers)
    # Decoder blocks are followed by input, frequency, horizon, and bias selectors.
    input_selector = num_layers
    frequency_selector = num_layers + 1
    horizon_selector = num_layers + 2
    bias_selector = num_layers + 3
    tensor_map = getattr(readers, "tensor_map", {})
    for name in sorted(tensor_map):
        arr = _load_tensor(readers, name)
        selected_fp32 = (
            any(
                layer in fp32_layers_set
                and name.startswith(f"decoder.layers.{layer}.")
                for layer in range(num_layers)
            )
            or (
                input_selector in fp32_layers_set
                and name.startswith("decoder.input_ff_layer.")
            )
            or (
                frequency_selector in fp32_layers_set
                and name.startswith("decoder.freq_emb.")
            )
            or (
                horizon_selector in fp32_layers_set
                and name.startswith("horizon_ff_layer.")
            )
        )
        dtype = np.float32 if (
            (
                selected_fp32
                and (
                    bias_selector in fp32_layers_set
                    or not name.endswith(".bias")
                )
                and not name.endswith(".self_attn.k_proj.bias")
            )
            or name.endswith("layer_norm.weight")
            or name.endswith("layer_norm.bias")
            or name.endswith("layernorm.weight")
            or name.endswith("input_layernorm.weight")
            or name.endswith("scaling")
        ) else target_dtype
        weights[name] = np.ascontiguousarray(arr, dtype=dtype)
    return weights


def _require_supported(raw: dict) -> None:
    if bool(raw.get("use_positional_embedding", False)):
        raise NotImplementedError("TimesFM native TRT builder does not support positional embedding profiles")
    if int(raw.get("hidden_size", 0)) != int(raw.get("num_attention_heads", 1)) * int(raw.get("head_dim", 0)):
        raise NotImplementedError("TimesFM native TRT builder requires hidden_size == heads * head_dim")


def _patchify_2d(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    *,
    num_patches: int,
    patch_length: int,
) -> trt.ITensor:
    shuf = network.add_shuffle(tensor)
    shuf.reshape_dims = (1, num_patches, patch_length)
    return shuf.get_output(0)


def _add_residual_block(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weights: WeightDict,
    *,
    prefix: str,
    precision: str,
) -> trt.ITensor:
    hidden = add_linear(
        network,
        inp,
        weights[f"{prefix}.input_layer.weight"],
        weights.get(f"{prefix}.input_layer.bias"),
        precision=precision,
    )
    hidden = graph_ops.add_silu(network, hidden)
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


def _softplus_np(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def _heads_from_rows(
    network: trt.INetworkDefinition,
    x: trt.ITensor,
    *,
    num_heads: int,
    head_dim: int,
    seq_len: int,
) -> trt.ITensor:
    shuf = network.add_shuffle(x)
    shuf.reshape_dims = (1, seq_len, num_heads, head_dim)
    shuf.second_transpose = (0, 2, 1, 3)
    return shuf.get_output(0)


def _rows_from_heads(
    network: trt.INetworkDefinition,
    x: trt.ITensor,
    *,
    hidden_size: int,
    seq_len: int,
) -> trt.ITensor:
    shuf = network.add_shuffle(x)
    shuf.first_transpose = (0, 2, 1, 3)
    shuf.reshape_dims = (1, seq_len, hidden_size)
    return shuf.get_output(0)


def _add_padding_mask(
    network: trt.INetworkDefinition,
    patched_pads: trt.ITensor,
    *,
    num_patches: int,
    num_heads: int,
) -> tuple[trt.ITensor, trt.ITensor]:
    patched_padding = network.add_reduce(
        patched_pads, trt.ReduceOperation.MIN, 1 << 2, keep_dims=False
    ).get_output(0)
    pad_shuf = network.add_shuffle(patched_padding)
    pad_shuf.reshape_dims = (1, 1, 1, num_patches)
    neg = add_scalar(network, (1, 1, 1, 1), -1.0e9)
    pad_mask = network.add_elementwise(
        pad_shuf.get_output(0), neg, trt.ElementWiseOperation.PROD
    ).get_output(0)
    causal = np.triu(np.ones((num_patches, num_patches), dtype=np.float32), k=1) * -1.0e9
    causal_t = graph_ops.add_constant(
        network, (1, 1, num_patches, num_patches),
        causal.reshape(1, 1, num_patches, num_patches), dtype=np.float32)
    mask = network.add_elementwise(
        pad_mask, causal_t, trt.ElementWiseOperation.SUM
    ).get_output(0)
    mask_heads = network.add_concatenation([mask] * num_heads)
    mask_heads.axis = 1
    return patched_padding, mask_heads.get_output(0)


def _add_decoder_layer(
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    patched_padding: trt.ITensor,
    attn_mask: trt.ITensor,
    weights: WeightDict,
    *,
    layer_idx: int,
    raw: dict,
    precision: str,
) -> trt.ITensor:
    hidden_size = int(raw.get("hidden_size", 1))
    num_heads = int(raw.get("num_attention_heads", 1))
    head_dim = int(raw.get("head_dim", hidden_size // num_heads))
    num_patches = int(raw.get("context_length", 1)) // int(raw.get("patch_length", 1))
    prefix = f"decoder.layers.{layer_idx}"

    residual = hidden
    norm = graph_ops.add_rms_norm_last_dim(
        network,
        hidden,
        hidden_size,
        weights[f"{prefix}.input_layernorm.weight"].astype(np.float32),
        add_scalar(network, (1, 1, 1), float(raw.get("rms_norm_eps", 1.0e-6))),
        dtype=(np.float16 if hidden.dtype == trt.float16 else np.float32),
    )
    q = add_linear(network, norm, weights[f"{prefix}.self_attn.q_proj.weight"],
                   weights.get(f"{prefix}.self_attn.q_proj.bias"), precision=precision)
    k = add_linear(network, norm, weights[f"{prefix}.self_attn.k_proj.weight"],
                   weights.get(f"{prefix}.self_attn.k_proj.bias"), precision=precision)
    v = add_linear(network, norm, weights[f"{prefix}.self_attn.v_proj.weight"],
                   weights.get(f"{prefix}.self_attn.v_proj.bias"), precision=precision)
    qh = _heads_from_rows(
        network, q, num_heads=num_heads, head_dim=head_dim, seq_len=num_patches)
    kh = _heads_from_rows(
        network, k, num_heads=num_heads, head_dim=head_dim, seq_len=num_patches)
    vh = _heads_from_rows(
        network, v, num_heads=num_heads, head_dim=head_dim, seq_len=num_patches)
    scale = _softplus_np(weights[f"{prefix}.self_attn.scaling"].astype(np.float32))
    scale = scale * (1.442695041 / np.sqrt(float(head_dim)))
    scale_t = graph_ops.add_constant(
        network, (1, 1, 1, head_dim),
        scale.reshape(1, 1, 1, head_dim),
        dtype=(np.float16 if qh.dtype == trt.float16 else np.float32))
    qh = network.add_elementwise(qh, scale_t, trt.ElementWiseOperation.PROD).get_output(0)
    if attn_mask.dtype != qh.dtype:
        attn_mask = network.add_cast(attn_mask, qh.dtype).get_output(0)
    ctx = graph_ops.add_attention_core(
        network, qh, kh, vh, causal=False, mask=attn_mask, scale=1.0)
    ctx_rows = _rows_from_heads(
        network, ctx, hidden_size=hidden_size, seq_len=num_patches)
    attn_out = add_linear(
        network,
        ctx_rows,
        weights[f"{prefix}.self_attn.o_proj.weight"],
        weights.get(f"{prefix}.self_attn.o_proj.bias"),
        precision=precision,
    )
    hidden = network.add_elementwise(residual, attn_out, trt.ElementWiseOperation.SUM).get_output(0)

    mlp_norm = graph_ops.add_layer_norm_native(
        network,
        hidden,
        hidden_size,
        weights[f"{prefix}.mlp.layer_norm.weight"].astype(np.float32),
        weights[f"{prefix}.mlp.layer_norm.bias"].astype(np.float32),
        1.0e-6,
    )
    mlp = add_linear(
        network,
        mlp_norm,
        weights[f"{prefix}.mlp.gate_proj.weight"],
        weights.get(f"{prefix}.mlp.gate_proj.bias"),
        precision=precision,
    )
    mlp = network.add_activation(mlp, trt.ActivationType.RELU).get_output(0)
    mlp = add_linear(
        network,
        mlp,
        weights[f"{prefix}.mlp.down_proj.weight"],
        weights.get(f"{prefix}.mlp.down_proj.bias"),
        precision=precision,
    )
    keep = network.add_elementwise(
        add_scalar(network, (1, num_patches), 1.0),
        patched_padding,
        trt.ElementWiseOperation.SUB,
    ).get_output(0)
    keep_s = network.add_shuffle(keep)
    keep_s.reshape_dims = (1, num_patches, 1)
    keep_t = keep_s.get_output(0)
    if keep_t.dtype != mlp.dtype:
        keep_t = network.add_cast(keep_t, mlp.dtype).get_output(0)
    mlp = network.add_elementwise(
        mlp, keep_t, trt.ElementWiseOperation.PROD
    ).get_output(0)
    return network.add_elementwise(hidden, mlp, trt.ElementWiseOperation.SUM).get_output(0)


def _build_timesfm_network(
    config: ModelConfig,
    weights: WeightDict,
    *,
    precision: str,
    verbose: bool,
) -> bytes:
    raw = config.raw
    _require_supported(raw)
    context_length = int(raw.get("context_length", 2048))
    patch_length = int(raw.get("patch_length", 32))
    num_patches = context_length // patch_length
    hidden_size = int(raw.get("hidden_size", 1280))
    num_heads = int(raw.get("num_attention_heads", 16))
    horizon = int(raw.get("horizon_length", 128))
    quantile_count = len(raw.get("quantiles", []))
    num_layers = int(raw.get("num_hidden_layers", 1))
    input_selector = num_layers
    frequency_selector = num_layers + 1
    horizon_selector = num_layers + 2
    bias_selector = num_layers + 3
    fp32_layers = frozenset(
        int(layer) for layer in raw.get("_fp32_layers", ()))
    invalid_fp32_layers = sorted(
        layer for layer in fp32_layers
        if layer < 0 or layer > bias_selector)
    if invalid_fp32_layers:
        raise ValueError(
            "fp32_layers contains out-of-range TimesFM selectors: "
            f"{invalid_fp32_layers}")

    builder, network = create_network(verbose=verbose)
    past_values = network.add_input("past_values", trt.float32, (1, context_length))
    padding = network.add_input("past_values_padding", trt.int32, (1, context_length))
    freq = network.add_input("freq", trt.int32, (1,))

    patched_inputs = _patchify_2d(
        network, past_values, num_patches=num_patches, patch_length=patch_length)
    patched_pads_i = _patchify_2d(
        network, padding, num_patches=num_patches, patch_length=patch_length)
    patched_pads = network.add_cast(patched_pads_i, trt.float32).get_output(0)
    valid = network.add_elementwise(
        add_scalar(network, (1, num_patches, patch_length), 1.0),
        patched_pads,
        trt.ElementWiseOperation.SUB,
    ).get_output(0)
    patched_inputs = network.add_elementwise(
        patched_inputs, valid, trt.ElementWiseOperation.PROD
    ).get_output(0)

    last_values = network.add_slice(
        patched_inputs, start=(0, num_patches - 1, 0),
        shape=(1, 1, patch_length), stride=(1, 1, 1)).get_output(0)
    last_valid = network.add_slice(
        valid, start=(0, num_patches - 1, 0),
        shape=(1, 1, patch_length), stride=(1, 1, 1)).get_output(0)
    denom = network.add_reduce(
        last_valid, trt.ReduceOperation.SUM, 1 << 2, keep_dims=True
    ).get_output(0)
    denom = network.add_elementwise(
        denom, add_scalar(network, (1, 1, 1), 1.0), trt.ElementWiseOperation.MAX
    ).get_output(0)
    masked_sum = network.add_reduce(
        network.add_elementwise(last_values, last_valid, trt.ElementWiseOperation.PROD).get_output(0),
        trt.ReduceOperation.SUM, 1 << 2, keep_dims=True,
    ).get_output(0)
    mu = network.add_elementwise(masked_sum, denom, trt.ElementWiseOperation.DIV).get_output(0)
    centered_last = network.add_elementwise(
        last_values, mu, trt.ElementWiseOperation.SUB).get_output(0)
    centered_last = network.add_elementwise(
        centered_last, last_valid, trt.ElementWiseOperation.PROD).get_output(0)
    var = network.add_reduce(
        network.add_elementwise(centered_last, centered_last, trt.ElementWiseOperation.PROD).get_output(0),
        trt.ReduceOperation.SUM, 1 << 2, keep_dims=True,
    ).get_output(0)
    var = network.add_elementwise(var, denom, trt.ElementWiseOperation.DIV).get_output(0)
    sigma = network.add_unary(var, trt.UnaryOperation.SQRT).get_output(0)
    sigma = network.add_elementwise(
        sigma,
        add_scalar(network, (1, 1, 1), float(raw.get("tolerance", 1.0e-6))),
        trt.ElementWiseOperation.MAX,
    ).get_output(0)

    normalized = network.add_elementwise(
        network.add_elementwise(patched_inputs, mu, trt.ElementWiseOperation.SUB).get_output(0),
        sigma,
        trt.ElementWiseOperation.DIV,
    ).get_output(0)
    normalized = network.add_elementwise(normalized, valid, trt.ElementWiseOperation.PROD).get_output(0)
    concat = network.add_concatenation([normalized, patched_pads])
    concat.axis = 2
    hidden = _add_residual_block(
        network, concat.get_output(0), weights,
        prefix="decoder.input_ff_layer",
        precision=(
            "fp32"
            if precision == "fp16" and input_selector in fp32_layers
            else precision
        ))

    frequency_precision = (
        "fp32"
        if precision == "fp16" and frequency_selector in fp32_layers
        else precision
    )
    frequency_dtype = (
        np.float16 if frequency_precision == "fp16" else np.float32)
    freq_w = graph_ops.add_constant(
        network,
        tuple(weights["decoder.freq_emb.weight"].shape),
        weights["decoder.freq_emb.weight"],
        dtype=frequency_dtype,
    )
    freq_emb = network.add_gather(freq_w, freq, 0).get_output(0)
    freq_shuf = network.add_shuffle(freq_emb)
    freq_shuf.reshape_dims = (1, 1, hidden_size)
    freq_t = freq_shuf.get_output(0)
    if frequency_precision == "fp32" and hidden.dtype != trt.float32:
        hidden = network.add_cast(hidden, trt.float32).get_output(0)
    elif freq_t.dtype != hidden.dtype:
        freq_t = network.add_cast(freq_t, hidden.dtype).get_output(0)
    hidden = network.add_elementwise(
        hidden, freq_t, trt.ElementWiseOperation.SUM).get_output(0)

    patched_padding, attn_mask = _add_padding_mask(
        network, patched_pads, num_patches=num_patches, num_heads=num_heads)
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
        hidden = _add_decoder_layer(
            network,
            hidden,
            patched_padding,
            attn_mask,
            weights,
            layer_idx=layer_idx,
            raw=raw,
            precision=layer_precision,
        )

    last_hidden = network.add_slice(
        hidden, start=(0, num_patches - 1, 0),
        shape=(1, 1, hidden_size), stride=(1, 1, 1)).get_output(0)
    forecast = _add_residual_block(
        network, last_hidden, weights,
        prefix="horizon_ff_layer",
        precision=(
            "fp32"
            if precision == "fp16" and horizon_selector in fp32_layers
            else precision
        ))
    full = network.add_shuffle(forecast)
    full.reshape_dims = (1, horizon, quantile_count + 1)
    full_t = full.get_output(0)
    if full_t.dtype != trt.float32:
        full_t = network.add_cast(full_t, trt.float32).get_output(0)
    full_t = network.add_elementwise(full_t, sigma, trt.ElementWiseOperation.PROD).get_output(0)
    full_t = network.add_elementwise(full_t, mu, trt.ElementWiseOperation.SUM).get_output(0)
    mean = network.add_slice(
        full_t, start=(0, 0, 0), shape=(1, horizon, 1), stride=(1, 1, 1)).get_output(0)
    mean_shuf = network.add_shuffle(mean)
    mean_shuf.reshape_dims = (1, horizon)
    add_named_output(network, mean_shuf.get_output(0), "mean_predictions")
    add_named_output(network, full_t, "full_predictions")

    return build_serialized_network(
        builder, network, precision=precision, verbose=verbose, tag="timesfm")


class TimesFmPlugin:
    name = "timesfm"
    runtime_strategy = "timesfm_trt"

    def matches(self, model_type: str) -> bool:
        mt = (model_type or "").lower()
        return "timesfm" in mt or "times_fm" in mt

    def load_weights(
        self,
        model_dir: str,
        config: ModelConfig,
        *,
        precision: str = "fp32",
    ) -> dict:
        num_layers = int(config.raw.get("num_hidden_layers", 1))
        return _load_all_tensors(
            model_dir,
            precision=precision,
            fp32_layers=tuple(config.raw.get("_fp32_layers", ())),
            num_layers=num_layers,
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
                parallel, feature="TimesFM replicated tensor-parallel bundles")
            cached = maybe_return_replicated_tp_plan(weights, parallel)
            if cached is not None:
                return cached

        plan = _build_timesfm_network(
            config, weights, precision=precision, verbose=verbose)
        cache_replicated_tp_plan(weights, parallel, plan)
        return plan

    def get_bundle_config_overrides(self, config: ModelConfig) -> dict:
        horizon = int(config.raw.get("horizon_length", config.raw.get("prediction_length", 0)) or 0)
        out = {"timesfm_default_freq": int(config.raw.get("freq", 0) or 0)}
        if horizon > 0:
            out["timesfm_prediction_length"] = horizon
        return out


plugin = TimesFmPlugin()
