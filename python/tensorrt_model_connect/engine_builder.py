# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve a model family and delegate the complete build to its ``model.py``."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType

from . import trt_compat
from .config import ModelConfig
from .engine_build_budget import enforce_single_full_bundle_build
from .families import (
    find_model,
    load_model_by_id,
    resolve_config_from_model_dir,
    resolve_diffusion_family_id,
    resolve_family_model_dir,
    resolve_nemo_archive_model_dir,
)
from .hf_snapshot import hf_snapshot_allow_patterns
from .parallel_config import ParallelConfig


def _setup_trt_import(rtx: bool) -> None:
    """Select the TensorRT backend before importing a family model module."""
    if not rtx:
        return
    trt_compat.configure_backend(rtx=True)
    print("[trtmc build] Using TensorRT-RTX backend", file=sys.stderr)


def _raise_friendly_download_error(model_id: str, exc: Exception) -> None:
    """Re-raise Hugging Face download errors with actionable messages."""
    exc_type = type(exc).__name__
    if "RepositoryNotFound" in exc_type:
        raise RuntimeError(
            f"Model '{model_id}' not found on HuggingFace Hub. "
            "Check the repo ID for typos (format: 'org/model-name'). "
            "If it's a private repo, run: huggingface-cli login"
        ) from exc
    if "GatedRepo" in exc_type:
        raise RuntimeError(
            f"Model '{model_id}' is gated. Accept the license at "
            f"https://huggingface.co/{model_id} then run: huggingface-cli login"
        ) from exc
    if "LocalEntryNotFound" in exc_type or "EntryNotFound" in exc_type:
        raise RuntimeError(
            f"Model '{model_id}' exists but required files are missing. "
            "The model may use a non-standard layout."
        ) from exc
    if "HTTPError" in exc_type or "ConnectionError" in exc_type:
        raise RuntimeError(
            f"Network error downloading '{model_id}': {exc}. "
            "Check your internet connection and try again."
        ) from exc
    if "OSError" in exc_type and "disk" in str(exc).lower():
        raise RuntimeError(
            f"Disk error downloading '{model_id}': {exc}. "
            "Check available disk space."
        ) from exc
    raise RuntimeError(
        f"Failed to download '{model_id}' from HuggingFace Hub: {exc}"
    ) from exc


def _is_hf_model_dir(path: Path) -> bool:
    return (path / "config.json").is_file() or (path / "model_index.json").is_file()


def _is_family_model_dir(path: Path) -> bool:
    return path.is_dir() and resolve_config_from_model_dir(path) is not None


def _resolve_nemo_archive(nemo_path: Path) -> str:
    resolved = resolve_nemo_archive_model_dir(nemo_path)
    if resolved is None:
        raise RuntimeError(
            f"No family-owned NeMo archive adapter recognized {nemo_path}"
        )
    return resolved


def _resolve_model(model_id_or_path: str, *, revision: str | None = None) -> str:
    """Return a local family-readable directory for a path or Hub model ID."""
    local = Path(model_id_or_path)
    if local.is_dir():
        staged = resolve_family_model_dir(local)
        if staged is not None:
            return staged
    if local.is_dir() and (_is_hf_model_dir(local) or _is_family_model_dir(local)):
        return str(local)
    if local.is_file() and local.suffix == ".nemo":
        return _resolve_nemo_archive(local)
    if local.is_dir():
        nemo_files = sorted(local.glob("*.nemo"))
        if nemo_files:
            return _resolve_nemo_archive(nemo_files[0])

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required for auto-downloading models. "
            "Install it with: pip install huggingface_hub"
        ) from exc

    print(f"[trtmc build] Downloading {model_id_or_path} ...", file=sys.stderr)
    try:
        local_dir = snapshot_download(
            repo_id=model_id_or_path,
            revision=revision,
            allow_patterns=hf_snapshot_allow_patterns(),
        )
    except Exception as exc:
        _raise_friendly_download_error(model_id_or_path, exc)

    downloaded = Path(local_dir)
    staged = resolve_family_model_dir(downloaded)
    if staged is not None:
        print(f"[trtmc build] Downloaded to {local_dir}", file=sys.stderr)
        return staged
    if _is_hf_model_dir(downloaded):
        print(f"[trtmc build] Downloaded to {local_dir}", file=sys.stderr)
        return str(downloaded)
    nemo_files = sorted(downloaded.glob("*.nemo"))
    if nemo_files:
        return _resolve_nemo_archive(nemo_files[0])
    print(f"[trtmc build] Downloaded to {local_dir}", file=sys.stderr)
    return str(downloaded)


def _resolve_family_model_from_model_dir(
    model_dir: str | Path,
) -> tuple[str, ModuleType]:
    """Resolve and load the single candidate whose model module claims ownership."""
    path = Path(model_dir)
    for filename in ("model_index.json", "config.json"):
        config_path = path / filename
        if not config_path.is_file():
            continue
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Model config must be a JSON object: {config_path}")
        pipeline_class = raw.get("_class_name")
        if isinstance(pipeline_class, str) and pipeline_class:
            family_id = resolve_diffusion_family_id(pipeline_class)
            if family_id:
                model = load_model_by_id(family_id)
                if model is None:
                    raise ValueError(f"No model module for family {family_id!r}")
                return family_id, model
        if filename == "model_index.json":
            break

    config = ModelConfig.from_dir(path)
    model = find_model(config)
    if model is None:
        raise ValueError(f"No model family owns model_type={config.model_type!r}")
    family_id = model.__package__.rsplit(".", 1)[-1]
    return family_id, model


def _dispatch_model_build(
    model_dir: str | Path,
    output_path: str | Path,
    options: dict[str, object],
) -> None:
    """Perform the only central build dispatch."""
    _setup_trt_import(bool(options.get("rtx")))
    family_id, model = _resolve_family_model_from_model_dir(model_dir)
    print(f"[trtmc build] Family: {family_id}", file=sys.stderr)
    model.build(str(model_dir), str(output_path), **options)


@enforce_single_full_bundle_build
def build(
    model_id_or_path: str,
    output_path: str,
    max_cache_length: int | None = None,
    *,
    model_revision: str | None = None,
    decoder_engine_layout: str = "split",
    dynamic_kv_cache: bool = False,
    dynamic_kv_profile_rows_override: list[int] | None = None,
    precision: str | None = None,
    fp32_layers: list[int] | None = None,
    quantize: str | None = None,
    quant_scales: str | None = None,
    quant_calibration_samples: int = 512,
    verbose: bool = False,
    kernel_artifacts: list[tuple[str, str]] | None = None,
    fp8_scales: dict | str | None = None,
    save_fp8_scales: str | None = None,
    rtx: bool = False,
    triattention_stats_path: str | None = None,
    triattention_kv_budget: int | None = None,
    triattention_divide_length: int = 128,
    triattention_recent_window: int = 128,
    triattention_score_aggregation: str = "mean",
    triattention_count_prompt_tokens: bool = True,
    triattention_protect_prefill: bool = True,
    triattention_disable_mlr: bool = False,
    triattention_disable_trig: bool = False,
    family_build_options: dict | None = None,
    parallel_config: ParallelConfig | None = None,
    diffusion_overrides: dict | None = None,
    build_timing_path: str | None = None,
    max_batch_size: int = 1,
) -> None:
    """Resolve a local or Hub model and delegate one complete family build."""
    options = dict(locals())
    options.pop("model_id_or_path")
    options.pop("output_path")
    options.pop("model_revision")
    options["tokenizer_source_model_id_or_path"] = model_id_or_path
    options["tokenizer_source_revision"] = model_revision

    revision = {"revision": model_revision} if model_revision else {}
    model_dir = _resolve_model(model_id_or_path, **revision)
    _dispatch_model_build(model_dir, output_path, options)
