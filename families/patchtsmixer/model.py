# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Complete PatchTSMixer build implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import tensorrt as trt

from . import graph_ops
from .checkpoint_mapper import (
    load_tensor,
    open_safetensors,
    target_numpy_dtype,
)
from .time_series_trt import (
    add_gelu,
    add_linear,
    add_named_output,
    add_patchify,
    add_std_scale,
    build_serialized_network,
    create_network,
)

if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


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


def infer_patchtsmixer_task_kind(config: dict[str, Any]) -> str:
    task = config.get("task_type", "")
    if isinstance(task, str) and task.strip():
        return _normalize_task_kind(task)

    architectures = config.get("architectures", [])
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

    if config.get("prediction_length") is not None:
        return "prediction"
    if config.get("num_targets") is not None:
        return "regression"
    raise ValueError("PatchTSMixer config must declare task_type or prediction_length")


def _load_all_tensors(
    model_dir: str | Path,
    *,
    precision: str,
    fp32_layers: tuple[int, ...] = (),
    num_layers: int,
) -> dict[str, Any]:
    readers = open_safetensors(Path(model_dir))
    target_dtype = target_numpy_dtype(precision)
    # Selectors: mixer blocks, patcher/head, four operations per block, biases.
    fp32_prefixes = tuple(
        f"model.encoder.mlp_mixer_encoder.mixers.{layer}."
        for layer in fp32_layers
        if layer < num_layers
    )
    fp32_patcher = num_layers in fp32_layers
    fp32_head = num_layers + 1 in fp32_layers
    fp32_biases = num_layers + 2 + num_layers * 4 in fp32_layers
    fp32_operation_prefixes: list[str] = []
    operation_names = (
        "patch_mixer.mlp",
        "patch_mixer.gating_block",
        "feature_mixer.mlp",
        "feature_mixer.gating_block",
    )
    for layer in range(num_layers):
        for operation_offset, operation_name in enumerate(operation_names):
            selector = num_layers + 2 + layer * 4 + operation_offset
            if selector in fp32_layers:
                fp32_operation_prefixes.append(
                    f"model.encoder.mlp_mixer_encoder.mixers.{layer}.{operation_name}."
                )
    weights: dict[str, Any] = {}
    for name in sorted(readers):
        arr = load_tensor(readers, name)
        selected_linear_weight = (
            name.startswith(fp32_prefixes)
            or name.startswith(tuple(fp32_operation_prefixes))
            or (fp32_patcher and name.startswith("model.encoder.patcher."))
            or (fp32_head and name.startswith("head."))
        ) and (fp32_biases or not name.endswith(".bias"))
        dtype = np.float32 if (".norm." in name or selected_linear_weight) else target_dtype
        weights[name] = np.ascontiguousarray(arr, dtype=dtype)
    return weights


def _require_supported(raw: dict[str, Any], task_kind: str) -> None:
    if task_kind != "prediction":
        raise NotImplementedError(
            "PatchTSMixer native TRT builder currently supports prediction profiles"
        )
    if bool(raw.get("self_attn", False)):
        raise NotImplementedError(
            "PatchTSMixer native TRT builder does not support self_attn profiles"
        )
    if str(raw.get("mode", "common_channel")).lower() != "common_channel":
        raise NotImplementedError(
            "PatchTSMixer native TRT builder currently supports common_channel mode"
        )
    if "layer" not in str(raw.get("norm_mlp", "LayerNorm")).lower():
        raise NotImplementedError(
            "PatchTSMixer native TRT builder currently supports LayerNorm mixer blocks"
        )
    if not bool(raw.get("gated_attn", False)):
        raise NotImplementedError(
            "PatchTSMixer native TRT builder currently expects gated_attn=True"
        )
    if str(raw.get("loss", "mse")).lower() != "mse":
        raise NotImplementedError("PatchTSMixer native TRT builder currently supports MSE heads")
    if raw.get("prediction_channel_indices") not in (None, [], ()):
        raise NotImplementedError(
            "PatchTSMixer native TRT builder does not support channel-filtered heads"
        )


def _add_layer_norm(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weights: dict[str, Any],
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
    weights: dict[str, Any],
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
    if inp.dtype != logits.dtype:
        inp = network.add_cast(inp, logits.dtype).get_output(0)
    probs = _softmax_last(network, logits)
    return network.add_elementwise(inp, probs, trt.ElementWiseOperation.PROD).get_output(0)


def _add_mlp(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weights: dict[str, Any],
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
    weights: dict[str, Any],
    *,
    layer_idx: int,
    raw: dict[str, Any],
    precision: str,
    fp32_layers: frozenset[int],
    num_layers: int,
) -> trt.ITensor:
    channels = int(raw.get("num_input_channels", 1))
    num_patches = int(raw.get("num_patches", 1))
    hidden_size = int(raw.get("d_model", 1))
    eps = float(raw.get("norm_eps", 1.0e-5))
    prefix = f"model.encoder.mlp_mixer_encoder.mixers.{layer_idx}"
    operation_base = num_layers + 2 + layer_idx * 4

    def operation_precision(offset: int) -> str:
        if precision == "fp16" and operation_base + offset in fp32_layers:
            return "fp32"
        return precision

    residual = hidden
    x = _add_layer_norm(
        network,
        hidden,
        weights,
        prefix=f"{prefix}.patch_mixer.norm.norm",
        hidden_size=hidden_size,
        eps=eps,
    )
    x = _transpose_last_two(network, x, shape=(1, channels, hidden_size, num_patches))
    x = _add_mlp(
        network, x, weights, prefix=f"{prefix}.patch_mixer.mlp", precision=operation_precision(0)
    )
    x = _add_gated_block(
        network,
        x,
        weights,
        prefix=f"{prefix}.patch_mixer.gating_block",
        precision=operation_precision(1),
    )
    x = _transpose_last_two(network, x, shape=(1, channels, num_patches, hidden_size))
    if x.dtype != residual.dtype:
        x = network.add_cast(x, residual.dtype).get_output(0)
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
        network, x, weights, prefix=f"{prefix}.feature_mixer.mlp", precision=operation_precision(2)
    )
    x = _add_gated_block(
        network,
        x,
        weights,
        prefix=f"{prefix}.feature_mixer.gating_block",
        precision=operation_precision(3),
    )
    if x.dtype != residual.dtype:
        x = network.add_cast(x, residual.dtype).get_output(0)
    return network.add_elementwise(residual, x, trt.ElementWiseOperation.SUM).get_output(0)


def _build_patchtsmixer_network(
    config: dict[str, Any],
    weights: dict[str, Any],
    *,
    precision: str,
    verbose: bool,
) -> bytes:
    raw = config
    task_kind = weights["_task_kind"]
    _require_supported(raw, task_kind)

    context_length = int(raw.get("context_length", 1))
    channels = int(raw.get("num_input_channels", 1))
    patch_length = int(raw.get("patch_length", 1))
    patch_stride = int(raw.get("patch_stride", patch_length))
    num_patches = int(raw.get("num_patches", 1))
    hidden_size = int(raw.get("d_model", 1))
    num_layers = int(raw.get("num_layers", 1))
    fp32_layers = frozenset(int(layer) for layer in raw.get("_fp32_layers", ()))
    invalid_fp32_layers = sorted(
        layer for layer in fp32_layers if layer < 0 or layer > num_layers + 2 + num_layers * 4
    )
    if invalid_fp32_layers:
        raise ValueError(f"fp32_layers contains out-of-range indices: {invalid_fp32_layers}")

    builder, network = create_network(verbose=verbose)
    past_values = network.add_input("past_values", trt.float32, (1, context_length, channels))
    observed = network.add_input("observed_mask", trt.float32, (1, context_length, channels))

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
        precision=("fp32" if precision == "fp16" and num_layers in fp32_layers else precision),
    )

    for layer_idx in range(num_layers):
        boundary_dtype = hidden.dtype
        layer_is_fp32 = precision == "fp16" and layer_idx in fp32_layers
        layer_precision = "fp32" if layer_is_fp32 else precision
        if layer_is_fp32 and hidden.dtype != trt.float32:
            hidden = network.add_cast(hidden, trt.float32).get_output(0)
        hidden = _add_mixer_layer(
            network,
            hidden,
            weights,
            layer_idx=layer_idx,
            raw=raw,
            precision=layer_precision,
            fp32_layers=fp32_layers,
            num_layers=num_layers,
        )
        next_stage_is_fp32 = layer_idx + 1 in fp32_layers or (
            layer_idx == num_layers - 1 and num_layers + 1 in fp32_layers
        )
        if layer_is_fp32 and not next_stage_is_fp32 and hidden.dtype != boundary_dtype:
            hidden = network.add_cast(hidden, boundary_dtype).get_output(0)

    flat = network.add_shuffle(hidden)
    flat.reshape_dims = (1, channels, num_patches * hidden_size)
    forecast = add_linear(
        network,
        flat.get_output(0),
        weights["head.base_forecast_block.weight"],
        weights.get("head.base_forecast_block.bias"),
        precision=("fp32" if precision == "fp16" and num_layers + 1 in fp32_layers else precision),
    )
    out = network.add_shuffle(forecast)
    out.first_transpose = (0, 2, 1)
    out.reshape_dims = (1, int(raw.get("prediction_length", 1)), channels)
    y = out.get_output(0)
    if y.dtype != trt.float32:
        y = network.add_cast(y, trt.float32).get_output(0)
    y = network.add_elementwise(y, scale, trt.ElementWiseOperation.PROD).get_output(0)
    y = network.add_elementwise(y, loc, trt.ElementWiseOperation.SUM).get_output(0)
    add_named_output(network, y, "prediction_outputs")

    return build_serialized_network(
        builder, network, precision=precision, verbose=verbose, tag="patchtsmixer"
    )


def _read_config(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "config.json"
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"PatchTSMixer checkpoint is missing config.json: {model_dir}"
        ) from exc
    if not isinstance(config, dict):
        raise ValueError("PatchTSMixer config.json must contain one JSON object")
    if config.get("model_type") != "patchtsmixer":
        raise ValueError(
            "PatchTSMixer requires config.json model_type='patchtsmixer', got "
            f"{config.get('model_type')!r}"
        )
    required = (
        "context_length",
        "num_input_channels",
        "patch_length",
        "patch_stride",
        "num_patches",
        "d_model",
        "num_layers",
        "prediction_length",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"PatchTSMixer config is missing required fields: {', '.join(missing)}")
    return config


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one strict PatchTSMixer bundle without shared model orchestration."""
    if request.dynamic_kv_cache:
        raise NotImplementedError("patchtsmixer does not support dynamic_kv_cache")

    if request.image_height is not None:
        raise NotImplementedError("patchtsmixer does not support image_height")

    if request.image_width is not None:
        raise NotImplementedError("patchtsmixer does not support image_width")

    if request.video_num_frames is not None:
        raise NotImplementedError("patchtsmixer does not support video_num_frames")

    if request.max_batch_size != 1:
        raise NotImplementedError("patchtsmixer does not support max_batch_size")

    if request.context_parallel_size != 1:
        raise ValueError("this family does not support context parallelism")

    if request.task != "time_series_forecast":
        raise ValueError("PatchTSMixer supports only time_series_forecast")
    model_dir = Path(request.model_dir)
    precision = request.precision.lower()
    if precision not in {"fp16", "fp32"}:
        raise ValueError(f"PatchTSMixer supports only fp16 or fp32, got {precision!r}")
    if request.max_sequence_length is not None:
        raise ValueError("PatchTSMixer does not accept max_sequence_length")
    if request.tensor_parallel_size not in {1, 2, 4, 8}:
        raise ValueError("PatchTSMixer tensor_parallel_size must be 1, 2, 4, or 8")
    if request.quantization is not None:
        raise ValueError("PatchTSMixer does not support quantization")

    config = _read_config(model_dir)
    task_kind = infer_patchtsmixer_task_kind(config)
    _require_supported(config, task_kind)
    num_layers = int(config["num_layers"])
    weights = _load_all_tensors(
        model_dir,
        precision=precision,
        fp32_layers=tuple(request.fp32_layers),
        num_layers=num_layers,
    )
    weights["_task_kind"] = task_kind
    config["_fp32_layers"] = tuple(request.fp32_layers)
    plan = _build_patchtsmixer_network(
        config,
        weights,
        precision=precision,
        verbose=request.verbose,
    )

    writer.set_header(
        family="patchtsmixer",
        task=request.task,
        backend=request.backend,
    )
    if request.tensor_parallel_size == 1:
        writer.add_bytes("engine.plan", plan)
    else:
        for rank in range(request.tensor_parallel_size):
            writer.add_bytes(f"engine.rank{rank}.plan", plan)
    writer.add_json(
        "runtime.json",
        {
            "context_length": int(config["context_length"]),
            "num_input_channels": int(config["num_input_channels"]),
            "prediction_length": int(config["prediction_length"]),
            "tensor_parallel_size": request.tensor_parallel_size,
        },
    )
