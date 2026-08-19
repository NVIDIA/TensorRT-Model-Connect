# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SAM2 family registration and model-owned TensorRT Python build hooks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from tensorrt_model_connect import trt_compat

from .checkpoint_mapper import Checkpoint, load_checkpoint, load_public_core_checkpoint
from .model_config import (
    CHECKPOINT_RELATIVE_PATH,
    PUBLIC_CHECKPOINT_RELATIVE_PATH,
    resolve_package_root,
    resolve_public_file,
    resolve_public_package_root,
)


_PRECISION = "mixed_bf16_fp32"
_PUBLIC_CORE_VARIANT = "public_sam2_1_small_with_synthetic_bbox_v1"
_DEFAULT_WORKSPACE_BYTES = 8 << 30
_GENERIC_EMBED_CANDIDATES = (
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
)
_EXTRA_PLAN_SECTIONS = (
    "sam2_prompt_engine_plan",
    "sam2_recurrent_h1_engine_plan",
    "sam2_recurrent_h2_engine_plan",
    "sam2_recurrent_h3_engine_plan",
    "sam2_recurrent_h4_engine_plan",
)


def _checkpoint(weights: dict[str, Any]) -> Checkpoint:
    value = weights.get("_sam2_checkpoint")
    if not isinstance(value, Checkpoint):
        raise ValueError("SAM2 weights do not contain an authenticated checkpoint")
    return value


def _validate_supported_build(model_dir: str, package_root: Path, config: Any) -> None:
    raw = config.raw
    unsupported_modes = [
        label
        for key, label in (
            ("_rtx_build_requested", "TensorRT-RTX"),
            ("_parallel_build_enabled", "parallel build"),
            ("_runtime_dynamic_kv_requested", "dynamic KV/TriAttention"),
            ("_quantized_build_requested", "quantization"),
            ("_fp32_layers", "FP32 layer overrides"),
        )
        if raw.get(key)
    ]
    all_options = raw.get("_family_build_options", {})
    options = all_options.get("sam2", {}) if isinstance(all_options, dict) else {}
    if options:
        unsupported_modes.append("family build options")
    if unsupported_modes:
        raise ValueError(f"SAM2 does not support {', '.join(unsupported_modes)}")

    checked_directories = {Path(model_dir).resolve(), package_root.resolve()}
    unexpected = sorted(
        str(directory / name)
        for directory in checked_directories
        for name in _GENERIC_EMBED_CANDIDATES
        if (directory / name).exists()
    )
    if unexpected:
        raise ValueError(
            "SAM2 package contains files that the generic bundle writer would embed as "
            f"unsupported sections: {', '.join(unexpected)}"
        )


def _serialize(
    checkpoint: Checkpoint,
    populate: Callable[[Any, Any, Checkpoint], Any],
    *,
    expected_inputs: int,
    expected_outputs: int,
    section: str,
    verbose: bool,
) -> bytes:
    try:
        trt = trt_compat.get_trt()
    except ImportError as exc:
        raise RuntimeError("SAM2 builds require the TensorRT 11 Python package") from exc
    if int(str(trt.__version__).split(".", 1)[0]) != 11:
        raise RuntimeError(f"SAM2 requires TensorRT 11, found {trt.__version__}")

    severity = trt.Logger.VERBOSE if verbose else trt.Logger.WARNING
    logger = trt.Logger(severity)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    if network is None:
        raise RuntimeError("TensorRT failed to create the SAM2 network")
    # TensorRT holds host weight pointers until serialization completes.
    # Retain the graph owner (and its generated NumPy buffers) for that lifetime.
    graph_owner = populate(trt, network, checkpoint)
    if (
        network.num_inputs != expected_inputs
        or network.num_outputs != expected_outputs
        or network.num_layers <= 0
    ):
        raise RuntimeError(f"SAM2 graph did not satisfy its exact contract: {section}")

    build_config = builder.create_builder_config()
    if build_config is None:
        raise RuntimeError("TensorRT failed to create the SAM2 builder configuration")
    build_config.builder_optimization_level = 3
    build_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, _DEFAULT_WORKSPACE_BYTES)
    build_config.clear_flag(trt.BuilderFlag.TF32)
    build_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

    plan = builder.build_serialized_network(network, build_config)
    if plan is None:
        raise RuntimeError(f"TensorRT produced no SAM2 plan: {section}")
    result = bytes(plan)
    del graph_owner
    return result


name = "sam2"
runtime_strategy = "sam2_bbox_video_tracking"
default_build_precision = _PRECISION
requires_tokenizer = False

def matches(config: object) -> bool:
    model_type = str(getattr(config, "model_type", config))
    normalized = model_type.lower().replace("-", "_").replace(".", "_")
    return normalized in {
        "sam2",
        "sam2_bbox_video_tracking",
        "sam2_video_tracking",
    }

def _require_precision(precision: str) -> None:
    if precision != _PRECISION:
        raise ValueError(f"SAM2 supports only {_PRECISION!r} precision, got {precision!r}")

def load_weights(
    model_dir: str, _config: Any, *, precision: str = _PRECISION
) -> dict[str, Any]:
    _require_precision(precision)
    root = resolve_package_root(model_dir)
    public_root = resolve_public_package_root(model_dir)
    if root is None and public_root is None:
        raise ValueError(f"unsupported SAM2 package: {model_dir}")
    root = root or public_root
    assert root is not None
    _validate_supported_build(model_dir, root, _config)
    if public_root is not None:
        _config.raw["_sam2_checkpoint_variant"] = _PUBLIC_CORE_VARIANT
        return {
            "_sam2_checkpoint": load_public_core_checkpoint(
                resolve_public_file(public_root, PUBLIC_CHECKPOINT_RELATIVE_PATH)
            )
        }
    return {"_sam2_checkpoint": load_checkpoint(root / CHECKPOINT_RELATIVE_PATH)}

def get_bundle_config_overrides(config: Any) -> dict[str, Any] | None:
    variant = config.raw.get("_sam2_checkpoint_variant")
    return {"sam2_checkpoint_variant": variant} if variant else None

def build_engine(
    _config: Any,
    weights: dict[str, Any],
    _max_cache_length: int,
    *,
    precision: str = _PRECISION,
    verbose: bool = False,
) -> bytes:
    _require_precision(precision)
    from .image_builder import populate_image_network

    return _serialize(
        _checkpoint(weights),
        populate_image_network,
        expected_inputs=1,
        expected_outputs=9,
        section="engine_plan",
        verbose=verbose,
    )

def build_extra_engines(
    _config: Any,
    weights: dict[str, Any],
    _max_cache_length: int,
    *,
    precision: str = _PRECISION,
    verbose: bool = False,
) -> dict[str, bytes]:
    _require_precision(precision)
    from .tracker_builder import populate_tracker_network

    checkpoint = _checkpoint(weights)
    plans: dict[str, bytes] = {}
    for history_frames, section in enumerate(_EXTRA_PLAN_SECTIONS):

        def populate(
            trt: Any, network: Any, source: Checkpoint, h: int = history_frames
        ) -> Any:
            return populate_tracker_network(trt, network, source, h)

        plans[section] = _serialize(
            checkpoint,
            populate,
            expected_inputs=4 if history_frames == 0 else 5,
            expected_outputs=3,
            section=section,
            verbose=verbose,
        )
    return plans


def build(model_dir: str, output_path: str, **options: object) -> None:
    """Build the complete six-plan SAM2 bundle."""
    import json
    from datetime import datetime, timezone

    from ...bundle_writer import (
        BundleInfo,
        BundleSection,
        gpu_name,
        tensorrt_abi,
        tensorrt_version,
        write_bundle,
    )
    from .config import ModelConfig

    precision = str(options.get("precision") or _PRECISION)
    _require_precision(precision)
    if int(options.get("max_batch_size") or 1) != 1:
        raise ValueError("SAM2 supports only max_batch_size=1")

    model_path = Path(model_dir)
    config = ModelConfig.from_dir(model_path)
    config.raw["_rtx_build_requested"] = bool(options.get("rtx"))
    config.raw["_parallel_build_enabled"] = bool(options.get("parallel_config"))
    config.raw["_runtime_dynamic_kv_requested"] = bool(
        options.get("dynamic_kv_cache") or options.get("triattention_stats_path")
    )
    config.raw["_quantized_build_requested"] = bool(options.get("quantize"))
    config.raw["_fp32_layers"] = list(options.get("fp32_layers") or ())
    config.raw["_family_build_options"] = dict(
        options.get("family_build_options") or {}
    )

    weights = load_weights(str(model_path), config, precision=precision)
    verbose = bool(options.get("verbose"))
    sections = [
        BundleSection(
            "engine_plan",
            build_engine(config, weights, 1, precision=precision, verbose=verbose),
        )
    ]
    sections.extend(
        BundleSection(section, plan)
        for section, plan in build_extra_engines(
            config, weights, 1, precision=precision, verbose=verbose
        ).items()
    )

    version = tensorrt_version()
    abi = tensorrt_abi(version)
    runtime_config = {
        key: value for key, value in config.raw.items() if not key.startswith("_")
    }
    runtime_config.update(
        {
            "runtime_strategy": runtime_strategy,
            "engine_backend": "trt",
            "trt_version": version,
            "precision": precision,
            "tokenizer_add_special_tokens": 0,
        }
    )
    overrides = get_bundle_config_overrides(config)
    if overrides:
        runtime_config.update(overrides)
    if abi:
        runtime_config["trt_abi"] = abi
    sections.append(
        BundleSection(
            "config.json", json.dumps(runtime_config, indent=2).encode("utf-8")
        )
    )

    write_bundle(
        output_path,
        BundleInfo(
            model_id=model_path.name,
            model_type=config.model_type,
            family=name,
            trt_version=version,
            trt_abi=abi,
            gpu_name=gpu_name(),
            created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_cache_length=1,
            runtime_strategy=runtime_strategy,
            precision=precision,
        ),
        sections,
    )
