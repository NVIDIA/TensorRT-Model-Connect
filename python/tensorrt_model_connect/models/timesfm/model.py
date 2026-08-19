# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TimesFM native TensorRT family model."""

from __future__ import annotations

import sys
import json
import re
import tempfile
import time

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
                layer in fp32_layers_set and name.startswith(f"decoder.layers.{layer}.")
                for layer in range(num_layers)
            )
            or (input_selector in fp32_layers_set and name.startswith("decoder.input_ff_layer."))
            or (frequency_selector in fp32_layers_set and name.startswith("decoder.freq_emb."))
            or (horizon_selector in fp32_layers_set and name.startswith("horizon_ff_layer."))
        )
        dtype = (
            np.float32
            if (
                (
                    selected_fp32
                    and (bias_selector in fp32_layers_set or not name.endswith(".bias"))
                    and not name.endswith(".self_attn.k_proj.bias")
                )
                or name.endswith("layer_norm.weight")
                or name.endswith("layer_norm.bias")
                or name.endswith("layernorm.weight")
                or name.endswith("input_layernorm.weight")
                or name.endswith("scaling")
            )
            else target_dtype
        )
        weights[name] = np.ascontiguousarray(arr, dtype=dtype)
    return weights


def _require_supported(raw: dict) -> None:
    if bool(raw.get("use_positional_embedding", False)):
        raise NotImplementedError(
            "TimesFM native TRT builder does not support positional embedding profiles"
        )
    if int(raw.get("hidden_size", 0)) != int(raw.get("num_attention_heads", 1)) * int(
        raw.get("head_dim", 0)
    ):
        raise NotImplementedError(
            "TimesFM native TRT builder requires hidden_size == heads * head_dim"
        )


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


def _select_normalization_patch(
    network: trt.INetworkDefinition,
    patched_inputs: trt.ITensor,
    valid: trt.ITensor,
    *,
    num_patches: int,
    patch_length: int,
) -> tuple[trt.ITensor, trt.ITensor]:
    """Match TimesFM's first-sufficient-patch normalization contract."""
    last_patch = num_patches - 1
    selected_inputs = network.add_slice(
        patched_inputs,
        start=(0, last_patch, 0),
        shape=(1, 1, patch_length),
        stride=(1, 1, 1),
    ).get_output(0)
    selected_valid = network.add_slice(
        valid,
        start=(0, last_patch, 0),
        shape=(1, 1, patch_length),
        stride=(1, 1, 1),
    ).get_output(0)

    # HF selects the first patch with at least three non-padding values and
    # falls back to the final patch when no patch qualifies. Traverse in
    # reverse so every earlier eligible patch replaces the current selection.
    minimum_valid_exclusive = add_scalar(network, (1, 1, 1), 2.0)
    for patch_index in range(last_patch - 1, -1, -1):
        patch_inputs = network.add_slice(
            patched_inputs,
            start=(0, patch_index, 0),
            shape=(1, 1, patch_length),
            stride=(1, 1, 1),
        ).get_output(0)
        patch_valid = network.add_slice(
            valid,
            start=(0, patch_index, 0),
            shape=(1, 1, patch_length),
            stride=(1, 1, 1),
        ).get_output(0)
        valid_count = network.add_reduce(
            patch_valid,
            trt.ReduceOperation.SUM,
            1 << 2,
            keep_dims=True,
        ).get_output(0)
        eligible = network.add_elementwise(
            valid_count,
            minimum_valid_exclusive,
            trt.ElementWiseOperation.GREATER,
        ).get_output(0)
        selected_inputs = network.add_select(eligible, patch_inputs, selected_inputs).get_output(0)
        selected_valid = network.add_select(eligible, patch_valid, selected_valid).get_output(0)

    return selected_inputs, selected_valid


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
        network,
        (1, 1, num_patches, num_patches),
        causal.reshape(1, 1, num_patches, num_patches),
        dtype=np.float32,
    )
    mask = network.add_elementwise(pad_mask, causal_t, trt.ElementWiseOperation.SUM).get_output(0)
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
    q = add_linear(
        network,
        norm,
        weights[f"{prefix}.self_attn.q_proj.weight"],
        weights.get(f"{prefix}.self_attn.q_proj.bias"),
        precision=precision,
    )
    k = add_linear(
        network,
        norm,
        weights[f"{prefix}.self_attn.k_proj.weight"],
        weights.get(f"{prefix}.self_attn.k_proj.bias"),
        precision=precision,
    )
    v = add_linear(
        network,
        norm,
        weights[f"{prefix}.self_attn.v_proj.weight"],
        weights.get(f"{prefix}.self_attn.v_proj.bias"),
        precision=precision,
    )
    qh = _heads_from_rows(network, q, num_heads=num_heads, head_dim=head_dim, seq_len=num_patches)
    kh = _heads_from_rows(network, k, num_heads=num_heads, head_dim=head_dim, seq_len=num_patches)
    vh = _heads_from_rows(network, v, num_heads=num_heads, head_dim=head_dim, seq_len=num_patches)
    scale = _softplus_np(weights[f"{prefix}.self_attn.scaling"].astype(np.float32))
    scale = scale * (1.442695041 / np.sqrt(float(head_dim)))
    scale_t = graph_ops.add_constant(
        network,
        (1, 1, 1, head_dim),
        scale.reshape(1, 1, 1, head_dim),
        dtype=(np.float16 if qh.dtype == trt.float16 else np.float32),
    )
    qh = network.add_elementwise(qh, scale_t, trt.ElementWiseOperation.PROD).get_output(0)
    if attn_mask.dtype != qh.dtype:
        attn_mask = network.add_cast(attn_mask, qh.dtype).get_output(0)
    ctx = graph_ops.add_attention_core(network, qh, kh, vh, causal=False, mask=attn_mask, scale=1.0)
    ctx_rows = _rows_from_heads(network, ctx, hidden_size=hidden_size, seq_len=num_patches)
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
    mlp = network.add_elementwise(mlp, keep_t, trt.ElementWiseOperation.PROD).get_output(0)
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
    fp32_layers = frozenset(int(layer) for layer in raw.get("_fp32_layers", ()))
    invalid_fp32_layers = sorted(
        layer for layer in fp32_layers if layer < 0 or layer > bias_selector
    )
    if invalid_fp32_layers:
        raise ValueError(
            f"fp32_layers contains out-of-range TimesFM selectors: {invalid_fp32_layers}"
        )

    builder, network = create_network(verbose=verbose)
    past_values = network.add_input("past_values", trt.float32, (1, context_length))
    padding = network.add_input("past_values_padding", trt.int32, (1, context_length))
    freq = network.add_input("freq", trt.int32, (1,))

    patched_inputs = _patchify_2d(
        network, past_values, num_patches=num_patches, patch_length=patch_length
    )
    patched_pads_i = _patchify_2d(
        network, padding, num_patches=num_patches, patch_length=patch_length
    )
    patched_pads = network.add_cast(patched_pads_i, trt.float32).get_output(0)
    valid = network.add_elementwise(
        add_scalar(network, (1, num_patches, patch_length), 1.0),
        patched_pads,
        trt.ElementWiseOperation.SUB,
    ).get_output(0)
    patched_inputs = network.add_elementwise(
        patched_inputs, valid, trt.ElementWiseOperation.PROD
    ).get_output(0)

    normalization_values, normalization_valid = _select_normalization_patch(
        network,
        patched_inputs,
        valid,
        num_patches=num_patches,
        patch_length=patch_length,
    )
    denom = network.add_reduce(
        normalization_valid, trt.ReduceOperation.SUM, 1 << 2, keep_dims=True
    ).get_output(0)
    denom = network.add_elementwise(
        denom, add_scalar(network, (1, 1, 1), 1.0), trt.ElementWiseOperation.MAX
    ).get_output(0)
    masked_sum = network.add_reduce(
        network.add_elementwise(
            normalization_values,
            normalization_valid,
            trt.ElementWiseOperation.PROD,
        ).get_output(0),
        trt.ReduceOperation.SUM,
        1 << 2,
        keep_dims=True,
    ).get_output(0)
    mu = network.add_elementwise(masked_sum, denom, trt.ElementWiseOperation.DIV).get_output(0)
    normalization_centered = network.add_elementwise(
        normalization_values, mu, trt.ElementWiseOperation.SUB
    ).get_output(0)
    normalization_centered = network.add_elementwise(
        normalization_centered,
        normalization_valid,
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    var = network.add_reduce(
        network.add_elementwise(
            normalization_centered,
            normalization_centered,
            trt.ElementWiseOperation.PROD,
        ).get_output(0),
        trt.ReduceOperation.SUM,
        1 << 2,
        keep_dims=True,
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
    normalized = network.add_elementwise(
        normalized, valid, trt.ElementWiseOperation.PROD
    ).get_output(0)
    concat = network.add_concatenation([normalized, patched_pads])
    concat.axis = 2
    hidden = _add_residual_block(
        network,
        concat.get_output(0),
        weights,
        prefix="decoder.input_ff_layer",
        precision=("fp32" if precision == "fp16" and input_selector in fp32_layers else precision),
    )

    frequency_precision = (
        "fp32" if precision == "fp16" and frequency_selector in fp32_layers else precision
    )
    frequency_dtype = np.float16 if frequency_precision == "fp16" else np.float32
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
    hidden = network.add_elementwise(hidden, freq_t, trt.ElementWiseOperation.SUM).get_output(0)

    patched_padding, attn_mask = _add_padding_mask(
        network, patched_pads, num_patches=num_patches, num_heads=num_heads
    )
    for layer_idx in range(num_layers):
        layer_precision = "fp32" if precision == "fp16" and layer_idx in fp32_layers else precision
        layer_dtype = trt.float16 if layer_precision == "fp16" else trt.float32
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
        hidden, start=(0, num_patches - 1, 0), shape=(1, 1, hidden_size), stride=(1, 1, 1)
    ).get_output(0)
    forecast = _add_residual_block(
        network,
        last_hidden,
        weights,
        prefix="horizon_ff_layer",
        precision=(
            "fp32" if precision == "fp16" and horizon_selector in fp32_layers else precision
        ),
    )
    full = network.add_shuffle(forecast)
    full.reshape_dims = (1, horizon, quantile_count + 1)
    full_t = full.get_output(0)
    if full_t.dtype != trt.float32:
        full_t = network.add_cast(full_t, trt.float32).get_output(0)
    full_t = network.add_elementwise(full_t, sigma, trt.ElementWiseOperation.PROD).get_output(0)
    full_t = network.add_elementwise(full_t, mu, trt.ElementWiseOperation.SUM).get_output(0)
    mean = network.add_slice(
        full_t, start=(0, 0, 0), shape=(1, horizon, 1), stride=(1, 1, 1)
    ).get_output(0)
    mean_shuf = network.add_shuffle(mean)
    mean_shuf.reshape_dims = (1, horizon)
    add_named_output(network, mean_shuf.get_output(0), "mean_predictions")
    add_named_output(network, full_t, "full_predictions")

    return build_serialized_network(
        builder, network, precision=precision, verbose=verbose, tag="timesfm"
    )


name = "timesfm"
runtime_strategy = "timesfm_trt"


def matches(config) -> bool:
    """Return whether this module owns the resolved model config."""
    model_type = getattr(config, "model_type", config)
    model_type = str(model_type)
    mt = (model_type or "").lower()
    return "timesfm" in mt or "times_fm" in mt


def load_weights(model_dir: str, config: ModelConfig, *, precision: str = "fp32") -> dict:
    num_layers = int(config.raw.get("num_hidden_layers", 1))
    return _load_all_tensors(
        model_dir,
        precision=precision,
        fp32_layers=tuple(config.raw.get("_fp32_layers", ())),
        num_layers=num_layers,
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
            parallel, feature="TimesFM replicated tensor-parallel bundles"
        )
        cached = maybe_return_replicated_tp_plan(weights, parallel)
        if cached is not None:
            return cached

    plan = _build_timesfm_network(config, weights, precision=precision, verbose=verbose)
    cache_replicated_tp_plan(weights, parallel, plan)
    return plan


def get_bundle_config_overrides(config: ModelConfig) -> dict:
    horizon = int(config.raw.get("horizon_length", config.raw.get("prediction_length", 0)) or 0)
    out = {"timesfm_default_freq": int(config.raw.get("freq", 0) or 0)}
    if horizon > 0:
        out["timesfm_prediction_length"] = horizon
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
    """Build the complete timesfm bundle inside its owning family module."""
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
        raise NotImplementedError("timesfm does not support context-parallel builds")
    if options.get("dynamic_kv_cache") or options.get("triattention_stats_path"):
        raise ValueError("timesfm does not use a decoder KV-cache runtime")

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
        raise ValueError("timesfm does not support quantized builds")

    if parallel.enabled:
        require_tensorrt_11_for_tensor_parallel(parallel, feature="timesfm tensor-parallel builds")
        if quant_ctx is not None:
            raise ValueError("timesfm tensor-parallel builds do not support quantization")

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
