# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PatchTSMixer native TensorRT family model."""

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


def _load_all_tensors(
    model_dir: str | Path,
    *,
    precision: str,
    fp32_layers: tuple[int, ...] = (),
    num_layers: int,
) -> WeightDict:
    readers = _open_safetensors(Path(model_dir))
    target_dtype = _target_np_dtype(precision)
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
    weights = WeightDict()
    tensor_map = getattr(readers, "tensor_map", {})
    for name in sorted(tensor_map):
        arr = _load_tensor(readers, name)
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
    if inp.dtype != logits.dtype:
        inp = network.add_cast(inp, logits.dtype).get_output(0)
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
        if layer_is_fp32 and hidden.dtype != boundary_dtype:
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


name = "patchtsmixer"
runtime_strategy = "patchtsmixer_trt"


def matches(config) -> bool:
    """Return whether this module owns the resolved model config."""
    model_type = getattr(config, "model_type", config)
    model_type = str(model_type)
    mt = (model_type or "").lower()
    return "patchtsmixer" in mt or "patch_tsmixer" in mt


def load_weights(model_dir: str, config: ModelConfig, *, precision: str = "fp32") -> dict:
    weights = _load_all_tensors(
        model_dir,
        precision=precision,
        fp32_layers=tuple(config.raw.get("_fp32_layers", ())),
        num_layers=int(config.raw.get("num_layers", 1)),
    )
    weights["_task_kind"] = infer_patchtsmixer_task_kind(config)
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
            parallel, feature="PatchTSMixer replicated tensor-parallel bundles"
        )
        cached = maybe_return_replicated_tp_plan(weights, parallel)
        if cached is not None:
            return cached

    plan = _build_patchtsmixer_network(config, weights, precision=precision, verbose=verbose)
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
    """Build the complete patchtsmixer bundle inside its owning family module."""
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
        raise NotImplementedError("patchtsmixer does not support context-parallel builds")
    if options.get("dynamic_kv_cache") or options.get("triattention_stats_path"):
        raise ValueError("patchtsmixer does not use a decoder KV-cache runtime")

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
        raise ValueError("patchtsmixer does not support quantized builds")

    if parallel.enabled:
        require_tensorrt_11_for_tensor_parallel(
            parallel, feature="patchtsmixer tensor-parallel builds"
        )
        if quant_ctx is not None:
            raise ValueError("patchtsmixer tensor-parallel builds do not support quantization")

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
