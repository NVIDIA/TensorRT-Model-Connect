# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PatchTST native TensorRT family model."""

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
)
from .config import ModelConfig
from ...parallel_config import normalize_parallel_config, require_tensorrt_11_for_tensor_parallel
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
    cache_replicated_tp_plan,
    create_network,
    maybe_return_replicated_tp_plan,
)


trt = trt_compat.get_trt()


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


name = "patchtst"
runtime_strategy = "patchtst_trt"


def matches(config) -> bool:
    """Return whether this module owns the resolved model config."""
    model_type = getattr(config, "model_type", config)
    model_type = str(model_type)
    mt = (model_type or "").lower()
    return mt == "patchtst" or mt.startswith("patchtst")


def load_weights(model_dir: str, config: ModelConfig, *, precision: str = "fp32") -> dict:
    weights = _load_all_tensors(
        model_dir,
        precision=precision,
        fp32_layers=tuple(config.raw.get("_fp32_layers", ())),
        depth=int(config.raw.get("num_hidden_layers", 1)),
    )
    weights["_task_type"] = _normalize_task_type(config.raw)
    return weights


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
            parallel, feature="PatchTST replicated tensor-parallel bundles"
        )
        cached = maybe_return_replicated_tp_plan(weights, parallel)
        if cached is not None:
            return cached

    plan = _build_patchtst_network(config, weights, precision=precision, verbose=verbose)
    cache_replicated_tp_plan(weights, parallel, plan)
    return plan


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
    """Build the complete patchtst bundle inside its owning family module."""
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
        raise NotImplementedError("patchtst does not support context-parallel builds")
    if options.get("dynamic_kv_cache") or options.get("triattention_stats_path"):
        raise ValueError("patchtst does not use a decoder KV-cache runtime")

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
        raise ValueError("patchtst does not support quantized builds")

    if parallel.enabled:
        require_tensorrt_11_for_tensor_parallel(parallel, feature="patchtst tensor-parallel builds")
        if quant_ctx is not None:
            raise ValueError("patchtst tensor-parallel builds do not support quantization")

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
