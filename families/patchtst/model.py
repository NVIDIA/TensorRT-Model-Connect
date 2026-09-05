# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Complete PatchTST checkpoint-to-bundle build path."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import tensorrt as trt

from . import graph_ops
from .checkpoint_mapper import (
    WeightDict,
    _load_tensor,
    _open_safetensors,
    _target_np_dtype,
)
from .config import ModelConfig
from .time_series_trt import (
    add_batch_norm_last_dim,
    add_gelu,
    add_linear,
    add_named_output,
    add_patchify,
    add_scalar,
    add_squareplus,
    add_std_scale,
    build_serialized_network,
    create_network,
)

if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


def _config_value(config: Any, key: str, fallback: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, fallback)
    return getattr(config, key, fallback)


def _normalize_task_type(config: Any) -> str:
    explicit = str(
        _config_value(config, "patchtst_task", _config_value(config, "task_type", ""))
    ).lower()
    if explicit:
        if "class" in explicit:
            return "classification"
        if "regress" in explicit:
            return "regression"
        if "forecast" in explicit or "predict" in explicit:
            return "forecast"

    problem_type = str(_config_value(config, "problem_type", "")).lower()
    if "class" in problem_type:
        return "classification"
    if "regress" in problem_type:
        return "regression"

    architectures = _config_value(config, "architectures", [])
    if isinstance(architectures, str):
        architectures = [architectures]
    for arch in architectures or []:
        arch_l = str(arch).lower()
        if "class" in arch_l:
            return "classification"
        if "regress" in arch_l:
            return "regression"
        if "forecast" in arch_l or "predict" in arch_l:
            return "forecast"
    return "forecast"


def _load_all_tensors(
    model_dir: str | Path,
    *,
    precision: str,
    fp32_layers: tuple[int, ...] = (),
    depth: int,
) -> WeightDict:
    readers = _open_safetensors(Path(model_dir))
    target_dtype = _target_np_dtype(precision)
    # Selectors: whole blocks, embedding/position/head, grouped ops, linears, biases.
    fp32_prefixes = tuple(
        f"model.encoder.layers.{layer}." for layer in fp32_layers if layer < depth
    )
    fp32_embedding = depth in fp32_layers
    fp32_position = depth + 1 in fp32_layers
    fp32_head = depth + 2 in fp32_layers
    fp32_biases = depth + 3 + depth * 8 in fp32_layers
    fp32_operation_prefixes: list[str] = []
    for layer in range(depth):
        operation_base = depth + 3 + layer * 2
        if operation_base in fp32_layers:
            fp32_operation_prefixes.append(f"model.encoder.layers.{layer}.self_attn.")
        if operation_base + 1 in fp32_layers:
            fp32_operation_prefixes.append(f"model.encoder.layers.{layer}.ff.")
    fine_operation_names = (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.out_proj",
        "ff.0",
        "ff.3",
    )
    fine_operation_start = depth + 3 + depth * 2
    for layer in range(depth):
        for operation_offset, operation_name in enumerate(fine_operation_names):
            selector = fine_operation_start + layer * 6 + operation_offset
            if selector in fp32_layers:
                fp32_operation_prefixes.append(f"model.encoder.layers.{layer}.{operation_name}.")
    weights = WeightDict()
    tensor_map = getattr(readers, "tensor_map", {})
    for name in sorted(tensor_map):
        arr = _load_tensor(readers, name)
        selected_linear_weight = (
            (
                name.startswith(fp32_prefixes)
                or name.startswith(tuple(fp32_operation_prefixes))
                or (fp32_embedding and name.startswith("model.encoder.embedder.input_embedding."))
                or (fp32_head and name.startswith("head."))
            )
            and (fp32_biases or not name.endswith(".bias"))
            and not name.endswith(".self_attn.k_proj.bias")
        )
        dtype = (
            np.float32
            if (
                name.endswith(("running_mean", "running_var"))
                or ".norm" in name
                or "layernorm" in name
                or selected_linear_weight
                or (fp32_position and name.startswith("model.encoder.positional_encoder."))
            )
            else target_dtype
        )
        weights[name] = np.ascontiguousarray(arr, dtype=dtype)
    return weights


def _num_patches(raw: dict[str, Any]) -> int:
    context_length = int(raw.get("context_length", 1))
    patch_length = int(raw.get("patch_length", 1))
    patch_stride = int(raw.get("patch_stride", patch_length))
    return (max(context_length, patch_length) - patch_length) // patch_stride + 1


def _require_supported(raw: dict[str, Any], task_type: str) -> None:
    if task_type not in {"forecast", "regression"}:
        raise NotImplementedError(
            "PatchTST native TRT builder currently supports forecast/regression profiles"
        )
    if not bool(raw.get("share_embedding", True)):
        raise NotImplementedError("PatchTST native TRT builder requires share_embedding=True")
    if not bool(raw.get("share_projection", True)):
        raise NotImplementedError("PatchTST native TRT builder requires share_projection=True")
    if bool(raw.get("channel_attention", False)):
        raise NotImplementedError("PatchTST native TRT builder does not support channel_attention")
    if not bool(raw.get("pre_norm", True)):
        raise NotImplementedError("PatchTST native TRT builder requires pre_norm=True")
    if str(raw.get("activation_function", "gelu")).lower() != "gelu":
        raise NotImplementedError("PatchTST native TRT builder currently supports GELU FFN only")
    if (
        str(raw.get("scaling", "std")).lower() not in {"std", "true"}
        and raw.get("scaling") is not True
    ):
        raise NotImplementedError("PatchTST native TRT builder currently supports std scaling")
    if str(raw.get("norm_type", "batchnorm")).lower() not in {"batchnorm", "layernorm"}:
        raise NotImplementedError("PatchTST native TRT builder supports batchnorm/layernorm only")
    if task_type == "forecast" and raw.get("loss") != "mse":
        raise NotImplementedError(
            "PatchTST forecast native TRT builder currently supports MSE heads"
        )


def _linear_key(prefix: str, name: str) -> tuple[str, str | None]:
    return f"{prefix}.{name}.weight", f"{prefix}.{name}.bias"


def _apply_distribution_domain_map(
    network: trt.INetworkDefinition,
    tensors: list[trt.ITensor],
    *,
    distribution_output: str,
) -> list[trt.ITensor]:
    distribution = distribution_output.lower()
    if distribution == "normal":
        if len(tensors) != 2:
            raise ValueError("PatchTST normal regression head expects loc and scale tensors")
        return [tensors[0], add_squareplus(network, tensors[1])]
    if distribution == "student_t":
        if len(tensors) != 3:
            raise ValueError(
                "PatchTST student_t regression head expects df, loc, and scale tensors"
            )
        two = add_scalar(network, tuple(tensors[0].shape), 2.0, dtype=np.float32)
        df = network.add_elementwise(
            add_squareplus(network, tensors[0]), two, trt.ElementWiseOperation.SUM
        ).get_output(0)
        return [df, tensors[1], add_squareplus(network, tensors[2])]
    if distribution == "negative_binomial":
        if len(tensors) != 2:
            raise ValueError(
                "PatchTST negative_binomial regression head expects total_count and logits tensors"
            )
        return [add_squareplus(network, tensors[0]), tensors[1]]
    raise NotImplementedError(
        f"PatchTST native TRT regression builder does not support {distribution_output!r} distribution heads"
    )


def _add_norm(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weights: WeightDict,
    *,
    layer_idx: int,
    norm_name: str,
    hidden_size: int,
    raw: dict[str, Any],
) -> trt.ITensor:
    prefix = f"model.encoder.layers.{layer_idx}.{norm_name}"
    eps = float(raw.get("norm_eps", 1.0e-5))
    if str(raw.get("norm_type", "batchnorm")).lower() == "batchnorm":
        return add_batch_norm_last_dim(
            network,
            inp,
            width=hidden_size,
            gamma=weights[f"{prefix}.batchnorm.weight"],
            beta=weights[f"{prefix}.batchnorm.bias"],
            running_mean=weights[f"{prefix}.batchnorm.running_mean"],
            running_var=weights[f"{prefix}.batchnorm.running_var"],
            eps=eps,
        )
    return graph_ops.add_layer_norm_native(
        network,
        inp,
        hidden_size,
        weights[f"{prefix}.weight"].astype(np.float32),
        weights[f"{prefix}.bias"].astype(np.float32),
        eps,
    )


def _add_encoder_layer(
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    weights: WeightDict,
    *,
    layer_idx: int,
    raw: dict[str, Any],
    linear_precisions: tuple[str, str, str, str, str, str],
) -> trt.ITensor:
    channels = int(raw.get("num_input_channels", 1))
    hidden_size = int(raw.get("d_model", 1))
    num_heads = int(raw.get("num_attention_heads", 1))
    head_dim = hidden_size // num_heads
    seq_len = _num_patches(raw) + (1 if bool(raw.get("use_cls_token", False)) else 0)
    prefix = f"model.encoder.layers.{layer_idx}"

    channel_rows: list[trt.ITensor] = []
    for channel in range(channels):
        row_slice = network.add_slice(
            hidden,
            start=(0, channel, 0, 0),
            shape=(1, 1, seq_len, hidden_size),
            stride=(1, 1, 1, 1),
        ).get_output(0)
        row = network.add_shuffle(row_slice)
        row.reshape_dims = (seq_len, hidden_size)
        row_t = row.get_output(0)

        normed = _add_norm(
            network,
            row_t,
            weights,
            layer_idx=layer_idx,
            norm_name="norm_sublayer1",
            hidden_size=hidden_size,
            raw=raw,
        )
        qw, qb = _linear_key(prefix, "self_attn.q_proj")
        kw, kb = _linear_key(prefix, "self_attn.k_proj")
        vw, vb = _linear_key(prefix, "self_attn.v_proj")
        ow, ob = _linear_key(prefix, "self_attn.out_proj")
        q = add_linear(
            network, normed, weights[qw], weights.get(qb), precision=linear_precisions[0]
        )
        k = add_linear(
            network, normed, weights[kw], weights.get(kb), precision=linear_precisions[1]
        )
        v = add_linear(
            network, normed, weights[vw], weights.get(vb), precision=linear_precisions[2]
        )
        attention_dtype = trt.float32 if "fp32" in linear_precisions[:3] else trt.float16
        q = q if q.dtype == attention_dtype else network.add_cast(q, attention_dtype).get_output(0)
        k = k if k.dtype == attention_dtype else network.add_cast(k, attention_dtype).get_output(0)
        v = v if v.dtype == attention_dtype else network.add_cast(v, attention_dtype).get_output(0)
        ctx = graph_ops.add_attention_from_rows(
            network,
            q,
            k,
            v,
            num_heads=num_heads,
            head_dim=head_dim,
            q_seq=seq_len,
            kv_seq=seq_len,
            causal=False,
            tag=f"patchtst.l{layer_idx}.c{channel}",
        )
        attn = add_linear(
            network, ctx, weights[ow], weights.get(ob), precision=linear_precisions[3]
        )
        if attn.dtype != row_t.dtype:
            attn = network.add_cast(attn, row_t.dtype).get_output(0)
        row_t = network.add_elementwise(row_t, attn, trt.ElementWiseOperation.SUM).get_output(0)

        normed = _add_norm(
            network,
            row_t,
            weights,
            layer_idx=layer_idx,
            norm_name="norm_sublayer3",
            hidden_size=hidden_size,
            raw=raw,
        )
        fw0, fb0 = _linear_key(prefix, "ff.0")
        fw1, fb1 = _linear_key(prefix, "ff.3")
        ff = add_linear(
            network, normed, weights[fw0], weights.get(fb0), precision=linear_precisions[4]
        )
        ff = add_gelu(network, ff)
        ff = add_linear(network, ff, weights[fw1], weights.get(fb1), precision=linear_precisions[5])
        if ff.dtype != row_t.dtype:
            ff = network.add_cast(ff, row_t.dtype).get_output(0)
        row_t = network.add_elementwise(row_t, ff, trt.ElementWiseOperation.SUM).get_output(0)

        out = network.add_shuffle(row_t)
        out.reshape_dims = (1, 1, seq_len, hidden_size)
        channel_rows.append(out.get_output(0))

    cat = network.add_concatenation(channel_rows)
    cat.axis = 1
    return cat.get_output(0)


def _build_patchtst_network(
    config: ModelConfig,
    weights: WeightDict,
    *,
    precision: str,
    verbose: bool,
) -> bytes:
    raw = config.raw
    task_type = weights["_task_type"]
    _require_supported(raw, task_type)

    context_length = int(raw.get("context_length", 1))
    channels = int(raw.get("num_input_channels", 1))
    patch_length = int(raw.get("patch_length", 1))
    patch_stride = int(raw.get("patch_stride", patch_length))
    num_patches = _num_patches(raw)
    hidden_size = int(raw.get("d_model", 1))
    depth = int(raw.get("num_hidden_layers", 1))
    fp32_layers = frozenset(int(layer) for layer in raw.get("_fp32_layers", ()))
    invalid_fp32_layers = sorted(
        layer for layer in fp32_layers if layer < 0 or layer > depth + 3 + depth * 8
    )
    if invalid_fp32_layers:
        raise ValueError(f"fp32_layers contains out-of-range indices: {invalid_fp32_layers}")
    use_cls_token = bool(raw.get("use_cls_token", False))
    seq_len = num_patches + (1 if use_cls_token else 0)

    builder, network = create_network(verbose=verbose)
    past_values = network.add_input("past_values", trt.float32, (1, context_length, channels))
    observed = network.add_input("past_observed_mask", trt.float32, (1, context_length, channels))

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

    emb_w = weights["model.encoder.embedder.input_embedding.weight"]
    emb_b = weights.get("model.encoder.embedder.input_embedding.bias")
    embedding_precision = "fp32" if precision == "fp16" and depth in fp32_layers else precision
    hidden = add_linear(network, patches, emb_w, emb_b, precision=embedding_precision)

    position_dtype = np.float16 if hidden.dtype == trt.float16 else np.float32
    pos = weights["model.encoder.positional_encoder.position_enc"].astype(position_dtype)
    if use_cls_token:
        patch_pos = graph_ops.add_constant(
            network,
            (1, 1, num_patches, hidden_size),
            pos[1:, :].reshape(1, 1, num_patches, hidden_size),
            dtype=position_dtype,
        )
        hidden = network.add_elementwise(
            hidden, patch_pos, trt.ElementWiseOperation.SUM
        ).get_output(0)
        cls = weights["model.encoder.positional_encoder.cls_token"].astype(position_dtype)
        cls_pos = cls.reshape(1, 1, 1, hidden_size) + pos[:1, :].reshape(1, 1, 1, hidden_size)
        cls_pos = np.tile(cls_pos, (1, channels, 1, 1))
        cls_t = graph_ops.add_constant(
            network, (1, channels, 1, hidden_size), cls_pos, dtype=position_dtype
        )
        cat = network.add_concatenation([cls_t, hidden])
        cat.axis = 2
        hidden = cat.get_output(0)
    else:
        pos_t = graph_ops.add_constant(
            network,
            (1, 1, num_patches, hidden_size),
            pos.reshape(1, 1, num_patches, hidden_size),
            dtype=position_dtype,
        )
        hidden = network.add_elementwise(hidden, pos_t, trt.ElementWiseOperation.SUM).get_output(0)

    for layer_idx in range(depth):
        boundary_dtype = hidden.dtype
        layer_is_fp32 = precision == "fp16" and layer_idx in fp32_layers
        layer_precision = "fp32" if layer_is_fp32 else precision
        operation_base = depth + 3 + layer_idx * 2
        attention_precision = (
            "fp32" if precision == "fp16" and operation_base in fp32_layers else layer_precision
        )
        ff_precision = (
            "fp32" if precision == "fp16" and operation_base + 1 in fp32_layers else layer_precision
        )
        fine_operation_start = depth + 3 + depth * 2
        fine_operation_base = fine_operation_start + layer_idx * 6
        linear_precisions = tuple(
            "fp32"
            if precision == "fp16" and fine_operation_base + offset in fp32_layers
            else (attention_precision if offset < 4 else ff_precision)
            for offset in range(6)
        )
        if layer_is_fp32 and hidden.dtype != trt.float32:
            hidden = network.add_cast(hidden, trt.float32).get_output(0)
        hidden = _add_encoder_layer(
            network,
            hidden,
            weights,
            layer_idx=layer_idx,
            raw=raw,
            linear_precisions=linear_precisions,
        )
        if layer_is_fp32 and hidden.dtype != boundary_dtype:
            hidden = network.add_cast(hidden, boundary_dtype).get_output(0)

    head_precision = "fp32" if precision == "fp16" and depth + 2 in fp32_layers else precision

    if task_type == "forecast":
        channel_outputs: list[trt.ITensor] = []
        for channel in range(channels):
            if use_cls_token:
                pooled = network.add_slice(
                    hidden,
                    start=(0, channel, 0, 0),
                    shape=(1, 1, 1, hidden_size),
                    stride=(1, 1, 1, 1),
                ).get_output(0)
                shuf = network.add_shuffle(pooled)
                shuf.reshape_dims = (1, hidden_size)
                pooled_t = shuf.get_output(0)
            else:
                pooled = network.add_slice(
                    hidden,
                    start=(0, channel, 0, 0),
                    shape=(1, 1, seq_len, hidden_size),
                    stride=(1, 1, 1, 1),
                ).get_output(0)
                shuf = network.add_shuffle(pooled)
                shuf.reshape_dims = (1, seq_len * hidden_size)
                pooled_t = shuf.get_output(0)
            pred = add_linear(
                network,
                pooled_t,
                weights["head.projection.weight"],
                weights.get("head.projection.bias"),
                precision=head_precision,
            )
            pred3 = network.add_shuffle(pred)
            pred3.reshape_dims = (1, int(raw.get("prediction_length", 1)), 1)
            channel_outputs.append(pred3.get_output(0))
        cat = network.add_concatenation(channel_outputs)
        cat.axis = 2
        y = cat.get_output(0)
        if y.dtype != trt.float32:
            y = network.add_cast(y, trt.float32).get_output(0)
        y = network.add_elementwise(y, scale, trt.ElementWiseOperation.PROD).get_output(0)
        y = network.add_elementwise(y, loc, trt.ElementWiseOperation.SUM).get_output(0)
        add_named_output(network, y, "prediction_outputs")
    else:
        if str(raw.get("pooling_type", "mean")).lower() != "mean":
            raise NotImplementedError(
                "PatchTST regression native TRT builder supports mean pooling only"
            )
        pooled = network.add_reduce(
            hidden, trt.ReduceOperation.AVG, 1 << 2, keep_dims=False
        ).get_output(0)
        flat = network.add_shuffle(pooled)
        flat.reshape_dims = (1, channels * hidden_size)
        flat_t = flat.get_output(0)
        outputs: list[trt.ITensor] = []
        idx = 0
        while f"head.projection.proj.{idx}.weight" in weights:
            pred = add_linear(
                network,
                flat_t,
                weights[f"head.projection.proj.{idx}.weight"],
                weights.get(f"head.projection.proj.{idx}.bias"),
                precision=head_precision,
            )
            outputs.append(pred)
            idx += 1
        if not outputs:
            pred = add_linear(
                network,
                flat_t,
                weights["head.projection.weight"],
                weights.get("head.projection.bias"),
                precision=head_precision,
            )
            pred3 = network.add_shuffle(pred)
            pred3.reshape_dims = (1, int(raw.get("num_targets", 1)))
            add_named_output(network, pred3.get_output(0), "regression_outputs")
        else:
            outputs = _apply_distribution_domain_map(
                network,
                outputs,
                distribution_output=str(raw.get("distribution_output", "")),
            )
            reshaped_outputs: list[trt.ITensor] = []
            for pred in outputs:
                pred3 = network.add_shuffle(pred)
                pred3.reshape_dims = (1, int(raw.get("num_targets", 1)), 1)
                reshaped_outputs.append(pred3.get_output(0))
            cat = network.add_concatenation(reshaped_outputs)
            cat.axis = 2
            add_named_output(network, cat.get_output(0), "regression_outputs")

    return build_serialized_network(
        builder, network, precision=precision, verbose=verbose, tag="patchtst"
    )


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one PatchTST bundle without shared model orchestration."""
    if request.dynamic_kv_cache:
        raise NotImplementedError("patchtst does not support dynamic_kv_cache")

    if request.image_height is not None:
        raise NotImplementedError("patchtst does not support image_height")

    if request.image_width is not None:
        raise NotImplementedError("patchtst does not support image_width")

    if request.video_num_frames is not None:
        raise NotImplementedError("patchtst does not support video_num_frames")

    if request.max_batch_size != 1:
        raise NotImplementedError("patchtst does not support max_batch_size")

    if request.context_parallel_size != 1:
        raise ValueError("this family does not support context parallelism")

    if request.task != "time_series_forecast":
        raise ValueError("PatchTST supports only time_series_forecast")
    model_dir = Path(request.model_dir)
    config = ModelConfig.from_dir(model_dir)
    if config.model_type != "patchtst":
        raise ValueError(f"PatchTST requires model_type='patchtst', got {config.model_type!r}")
    precision = request.precision.lower()
    if precision not in {"fp16", "fp32"}:
        raise ValueError("PatchTST supports only fp16 and fp32 builds")
    if request.max_sequence_length is not None:
        raise ValueError("PatchTST derives context_length from its checkpoint")
    if request.quantization is not None:
        raise ValueError("PatchTST does not support quantization")
    if request.tensor_parallel_size not in {1, 2, 4, 8}:
        raise ValueError("PatchTST tensor_parallel_size must be 1, 2, 4, or 8")

    config.raw["_fp32_layers"] = sorted(set(request.fp32_layers))
    depth = int(config.raw.get("num_hidden_layers", 1))
    weights = _load_all_tensors(
        model_dir,
        precision=precision,
        fp32_layers=tuple(config.raw["_fp32_layers"]),
        depth=depth,
    )
    task_type = _normalize_task_type(config.raw)
    weights["_task_type"] = task_type
    plan = _build_patchtst_network(
        config,
        weights,
        precision=precision,
        verbose=request.verbose,
    )

    writer.set_header(family="patchtst", task=request.task, backend=request.backend)
    if request.tensor_parallel_size == 1:
        writer.add_bytes("engine.plan", plan)
    else:
        for rank in range(request.tensor_parallel_size):
            writer.add_bytes(f"engine.rank{rank}.plan", plan)
    writer.add_json(
        "runtime.json",
        {
            "context_length": int(config.raw["context_length"]),
            "num_input_channels": int(config.raw["num_input_channels"]),
            "prediction_length": int(
                config.raw.get("prediction_length", config.raw.get("num_targets", 1))
            ),
            "task": task_type,
            "tensor_parallel_size": request.tensor_parallel_size,
        },
    )
