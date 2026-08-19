# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fast Foundation Stereo family plugin."""

from __future__ import annotations

import json
from pathlib import Path

from .config import ModelConfig


_CHECKPOINT = Path("weights/23-36-37/model_best_bp2_serialize.pth")
_POST_SECTION = "fast_foundation_stereo_post_engine_plan"
_NATIVE_PLUGIN_SECTION = "fast_foundation_stereo_native_plugin_so"


class WeightDict(dict):
    """Model-owned weight payload."""


def config_from_dir(model_dir: str | Path) -> dict | None:
    """Return the fixed runtime config for one complete model checkout."""

    model_path = Path(model_dir)
    required = (
        model_path / "core/foundation_stereo.py",
        model_path / "core/submodule.py",
        model_path / _CHECKPOINT,
    )
    if not all(path.is_file() for path in required):
        return None

    benchmark: dict = {}
    benchmark_path = model_path / "reference/benchmark_result.json"
    if benchmark_path.is_file():
        try:
            loaded = json.loads(benchmark_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                benchmark = loaded
        except (OSError, json.JSONDecodeError):
            pass

    return {
        "model_type": "fast_foundation_stereo",
        "architectures": ["FastFoundationStereo"],
        "runtime_strategy": "fast_foundation_stereo_disparity",
        "vocab_size": 0,
        "hidden_size": 0,
        "num_hidden_layers": 0,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "max_position_embeddings": 1,
        "stereo_input_height": 700,
        "stereo_input_width": 700,
        "stereo_engine_height": 704,
        "stereo_engine_width": 704,
        "stereo_max_disparity": int(benchmark.get("max_disp", 192)),
        "stereo_valid_iters": int(benchmark.get("valid_iters", 8)),
        "stereo_cv_groups": 8,
        "stereo_normalize_gwc": True,
        "stereo_post_engine_section": _POST_SECTION,
        "stereo_accuracy_metric": "cosine_epe_bad2",
        "stereo_min_cosine": 0.999,
        "stereo_max_mean_abs_error": 0.5,
        "stereo_max_bad_2px_fraction": 0.02,
        "requires_tokenizer": False,
    }


name = "fast_foundation_stereo"
runtime_strategy = "fast_foundation_stereo_disparity"
requires_tokenizer = False

def matches(config: object) -> bool:
    model_type = str(getattr(config, "model_type", config))
    return (model_type or "").lower().replace("-", "_") in {
        "fast_foundation_stereo",
        "foundation_stereo_lite",
    }

def load_weights(
    model_dir: str,
    config: ModelConfig,
    *,
    precision: str = "fp32",
) -> WeightDict:
    del precision
    model_path = Path(model_dir).resolve()
    checkpoint = model_path / _CHECKPOINT
    source = model_path / "core/foundation_stereo.py"
    if not checkpoint.is_file() or not source.is_file():
        raise FileNotFoundError(
            "Fast Foundation Stereo requires core/foundation_stereo.py and "
            f"{_CHECKPOINT.as_posix()} under {model_path}"
        )
    config.raw["_fast_foundation_stereo_model_dir"] = str(model_path)
    return WeightDict({"_fast_foundation_stereo_model_dir": str(model_path)})

def _model_dir(config: ModelConfig, weights: WeightDict) -> str:
    model_dir = weights.get("_fast_foundation_stereo_model_dir") or config.raw.get(
        "_fast_foundation_stereo_model_dir"
    )
    if not model_dir:
        raise RuntimeError("Fast Foundation Stereo model directory was not loaded")
    return str(model_dir)

def build_engine(
    config: ModelConfig,
    weights: WeightDict,
    max_cache_length: int,
    *,
    precision: str = "fp16",
    quant_ctx=None,
    verbose: bool = False,
) -> bytes:
    del max_cache_length, quant_ctx
    from .builder import build_feature_engine

    return build_feature_engine(
        _model_dir(config, weights),
        precision=precision,
        max_disparity=int(config.raw.get("stereo_max_disparity", 192)),
        valid_iters=int(config.raw.get("stereo_valid_iters", 8)),
        verbose=verbose,
    )

def build_extra_engines(
    config: ModelConfig,
    weights: WeightDict,
    max_cache_length: int,
    *,
    precision: str = "fp16",
    quant_ctx=None,
    verbose: bool = False,
) -> dict[str, bytes]:
    del max_cache_length, quant_ctx
    from .builder import build_post_engine
    from .native_plugin_builder import ensure_native_plugin

    native_plugin = ensure_native_plugin(verbose=verbose)
    return {
        _POST_SECTION: build_post_engine(
            _model_dir(config, weights),
            precision=precision,
            max_disparity=int(config.raw.get("stereo_max_disparity", 192)),
            valid_iters=int(config.raw.get("stereo_valid_iters", 8)),
            verbose=verbose,
        ),
        _NATIVE_PLUGIN_SECTION: native_plugin.read_bytes(),
    }


def build(model_dir: str, output_path: str, **options: object) -> None:
    """Build the complete Fast Foundation Stereo bundle."""
    from datetime import datetime, timezone

    from ...bundle_writer import (
        BundleInfo,
        BundleSection,
        gpu_name,
        tensorrt_abi,
        tensorrt_version,
        write_bundle,
    )

    precision = str(options.get("precision") or "fp16").lower()
    if precision != "fp16":
        raise ValueError("Fast Foundation Stereo supports only 'fp16' precision")
    unsupported = {
        "quantization": options.get("quantize"),
        "parallel build": options.get("parallel_config"),
        "dynamic KV": options.get("dynamic_kv_cache"),
        "TriAttention": options.get("triattention_stats_path"),
        "TensorRT-RTX": options.get("rtx"),
        "FP32 layer overrides": options.get("fp32_layers"),
    }
    requested = [label for label, value in unsupported.items() if value]
    if requested:
        raise ValueError(
            "Fast Foundation Stereo does not support " + ", ".join(requested)
        )
    if int(options.get("max_batch_size") or 1) != 1:
        raise ValueError("Fast Foundation Stereo supports only max_batch_size=1")

    model_path = Path(model_dir)
    config = ModelConfig.from_dir(model_path)
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
    abi = tensorrt_abi(version)
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
