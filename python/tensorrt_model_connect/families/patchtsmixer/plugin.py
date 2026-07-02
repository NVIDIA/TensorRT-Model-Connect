# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PatchTSMixer native TensorRT family plugin."""

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
)
from .config import ModelConfig
from ...parallel_config import normalize_parallel_config, require_tensorrt_11_for_tensor_parallel
from .time_series_trt import (
    add_gelu,
    add_linear,
    add_named_output,
    add_patchify,
    add_std_scale,
    build_serialized_network,
    cache_replicated_tp_plan,
    create_network,
    maybe_return_replicated_tp_plan,
)


trt = trt_compat.get_trt()


def _raw_config_value(config, key: str, default=None):
    raw = getattr(config, "raw", None)
    if isinstance(raw, dict) and key in raw:
        return raw[key]
    return getattr(config, key, default)


def _normalize_task_kind(task: str) -> str:
    task = task.lower().strip()
    if "regress" in task:
        return "regression"
    if "class" in task:
        return "classification"
    if "pretrain" in task:
        return "pretraining"
    if "forecast" in task or "predict" in task or "prediction" in task:
        return "prediction"
    return task


def infer_patchtsmixer_task_kind(config) -> str:
    task = _raw_config_value(config, "task_type", "")
    if isinstance(task, str) and task.strip():
        return _normalize_task_kind(task)

    architectures = _raw_config_value(config, "architectures", [])
    if isinstance(architectures, str):
        architectures = [architectures]
    for arch in architectures or []:
        arch_l = str(arch).lower()
        if "pretrain" in arch_l:
            return "pretraining"
        if "regress" in arch_l:
            return "regression"
        if "class" in arch_l:
            return "classification"
        if "predict" in arch_l:
            return "prediction"

    if _raw_config_value(config, "prediction_length", None) is not None:
        return "prediction"
    if _raw_config_value(config, "num_targets", None) is not None:
        return "regression"
    return "prediction"


def _load_all_tensors(model_dir: str | Path, *, precision: str) -> WeightDict:
    readers = _open_safetensors(Path(model_dir))
    target_dtype = _target_np_dtype(precision)
    weights = WeightDict()
    tensor_map = getattr(readers, "tensor_map", {})
    for name in sorted(tensor_map):
        arr = _load_tensor(readers, name)
        dtype = np.float32 if ".norm." in name else target_dtype
        weights[name] = np.ascontiguousarray(arr, dtype=dtype)
    return weights


def _require_supported(raw: dict[str, Any], task_kind: str) -> None:
    if task_kind != "prediction":
        raise NotImplementedError("PatchTSMixer native TRT builder currently supports prediction profiles")
    if bool(raw.get("self_attn", False)):
        raise NotImplementedError("PatchTSMixer native TRT builder does not support self_attn profiles")
    if str(raw.get("mode", "common_channel")).lower() != "common_channel":
        raise NotImplementedError("PatchTSMixer native TRT builder currently supports common_channel mode")
    if "layer" not in str(raw.get("norm_mlp", "LayerNorm")).lower():
        raise NotImplementedError("PatchTSMixer native TRT builder currently supports LayerNorm mixer blocks")
    if not bool(raw.get("gated_attn", False)):
        raise NotImplementedError("PatchTSMixer native TRT builder currently expects gated_attn=True")
    if str(raw.get("loss", "mse")).lower() != "mse":
        raise NotImplementedError("PatchTSMixer native TRT builder currently supports MSE heads")
    if raw.get("prediction_channel_indices") not in (None, [], ()):
        raise NotImplementedError("PatchTSMixer native TRT builder does not support channel-filtered heads")


def _add_layer_norm(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weights: WeightDict,
    *,
    prefix: str,
    hidden_size: int,
    eps: float,
) -> trt.ITensor:
    return graph_ops.add_layer_norm_native(
        network,
        inp,
        hidden_size,
        weights[f"{prefix}.weight"].astype(np.float32),
        weights[f"{prefix}.bias"].astype(np.float32),
        eps,
    )


def _transpose_last_two(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    *,
    shape: tuple[int, int, int, int],
) -> trt.ITensor:
    shuf = network.add_shuffle(inp)
    shuf.first_transpose = (0, 1, 3, 2)
    shuf.reshape_dims = shape
    return shuf.get_output(0)


def _softmax_last(network: trt.INetworkDefinition, inp: trt.ITensor) -> trt.ITensor:
    softmax = network.add_softmax(inp)
    softmax.axes = 1 << (len(tuple(inp.shape)) - 1)
    return softmax.get_output(0)


def _add_gated_block(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weights: WeightDict,
    *,
    prefix: str,
    precision: str,
) -> trt.ITensor:
    logits = add_linear(
        network,
        inp,
        weights[f"{prefix}.attn_layer.weight"],
        weights.get(f"{prefix}.attn_layer.bias"),
        precision=precision,
    )
    probs = _softmax_last(network, logits)
    return network.add_elementwise(inp, probs, trt.ElementWiseOperation.PROD).get_output(0)


def _add_mlp(
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
        weights[f"{prefix}.fc1.weight"],
        weights.get(f"{prefix}.fc1.bias"),
        precision=precision,
    )
    hidden = add_gelu(network, hidden)
    return add_linear(
        network,
        hidden,
        weights[f"{prefix}.fc2.weight"],
        weights.get(f"{prefix}.fc2.bias"),
        precision=precision,
    )


def _add_mixer_layer(
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    weights: WeightDict,
    *,
    layer_idx: int,
    raw: dict[str, Any],
    precision: str,
) -> trt.ITensor:
    channels = int(raw.get("num_input_channels", 1))
    num_patches = int(raw.get("num_patches", 1))
    hidden_size = int(raw.get("d_model", 1))
    eps = float(raw.get("norm_eps", 1.0e-5))
    prefix = f"model.encoder.mlp_mixer_encoder.mixers.{layer_idx}"

    residual = hidden
    x = _add_layer_norm(
        network,
        hidden,
        weights,
        prefix=f"{prefix}.patch_mixer.norm.norm",
        hidden_size=hidden_size,
        eps=eps,
    )
    x = _transpose_last_two(
        network, x, shape=(1, channels, hidden_size, num_patches))
    x = _add_mlp(
        network, x, weights,
        prefix=f"{prefix}.patch_mixer.mlp", precision=precision)
    x = _add_gated_block(
        network, x, weights,
        prefix=f"{prefix}.patch_mixer.gating_block", precision=precision)
    x = _transpose_last_two(
        network, x, shape=(1, channels, num_patches, hidden_size))
    hidden = network.add_elementwise(residual, x, trt.ElementWiseOperation.SUM).get_output(0)

    residual = hidden
    x = _add_layer_norm(
        network,
        hidden,
        weights,
        prefix=f"{prefix}.feature_mixer.norm.norm",
        hidden_size=hidden_size,
        eps=eps,
    )
    x = _add_mlp(
        network, x, weights,
        prefix=f"{prefix}.feature_mixer.mlp", precision=precision)
    x = _add_gated_block(
        network, x, weights,
        prefix=f"{prefix}.feature_mixer.gating_block", precision=precision)
    return network.add_elementwise(residual, x, trt.ElementWiseOperation.SUM).get_output(0)


def _build_patchtsmixer_network(
    config: ModelConfig,
    weights: WeightDict,
    *,
    precision: str,
    verbose: bool,
) -> bytes:
    raw = config.raw
    task_kind = weights["_task_kind"]
    _require_supported(raw, task_kind)

    context_length = int(raw.get("context_length", 1))
    channels = int(raw.get("num_input_channels", 1))
    patch_length = int(raw.get("patch_length", 1))
    patch_stride = int(raw.get("patch_stride", patch_length))
    num_patches = int(raw.get("num_patches", 1))
    hidden_size = int(raw.get("d_model", 1))
    num_layers = int(raw.get("num_layers", 1))

    builder, network = create_network(verbose=verbose)
    past_values = network.add_input(
        "past_values", trt.float32, (1, context_length, channels))
    observed = network.add_input(
        "observed_mask", trt.float32, (1, context_length, channels))

    scaled, loc, scale = add_std_scale(
        network,
        past_values,
        observed,
        channels=channels,
        minimum_scale=float(raw.get("minimum_scale", 1.0e-5)),
    )
    patches = add_patchify(
        network,
        scaled,
        context_length=context_length,
        channels=channels,
        patch_length=patch_length,
        patch_stride=patch_stride,
        num_patches=num_patches,
    )
    hidden = add_linear(
        network,
        patches,
        weights["model.encoder.patcher.weight"],
        weights.get("model.encoder.patcher.bias"),
        precision=precision,
    )

    for layer_idx in range(num_layers):
        hidden = _add_mixer_layer(
            network,
            hidden,
            weights,
            layer_idx=layer_idx,
            raw=raw,
            precision=precision,
        )

    flat = network.add_shuffle(hidden)
    flat.reshape_dims = (1, channels, num_patches * hidden_size)
    forecast = add_linear(
        network,
        flat.get_output(0),
        weights["head.base_forecast_block.weight"],
        weights.get("head.base_forecast_block.bias"),
        precision=precision,
    )
    out = network.add_shuffle(forecast)
    out.first_transpose = (0, 2, 1)
    out.reshape_dims = (1, int(raw.get("prediction_length", 1)), channels)
    y = out.get_output(0)
    y = network.add_elementwise(y, scale, trt.ElementWiseOperation.PROD).get_output(0)
    y = network.add_elementwise(y, loc, trt.ElementWiseOperation.SUM).get_output(0)
    add_named_output(network, y, "prediction_outputs")

    return build_serialized_network(
        builder, network, precision=precision, verbose=verbose, tag="patchtsmixer")


class PatchTSMixerPlugin:
    name = "patchtsmixer"
    runtime_strategy = "patchtsmixer_trt"

    def matches(self, model_type: str) -> bool:
        mt = (model_type or "").lower()
        return "patchtsmixer" in mt or "patch_tsmixer" in mt

    def load_weights(
        self,
        model_dir: str,
        config: ModelConfig,
        *,
        precision: str = "fp32",
    ) -> dict:
        weights = _load_all_tensors(model_dir, precision=precision)
        weights["_task_kind"] = infer_patchtsmixer_task_kind(config)
        return weights

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
                parallel, feature="PatchTSMixer replicated tensor-parallel bundles")
            cached = maybe_return_replicated_tp_plan(weights, parallel)
            if cached is not None:
                return cached

        plan = _build_patchtsmixer_network(
            config, weights, precision=precision, verbose=verbose)
        cache_replicated_tp_plan(weights, parallel, plan)
        return plan


plugin = PatchTSMixerPlugin()
