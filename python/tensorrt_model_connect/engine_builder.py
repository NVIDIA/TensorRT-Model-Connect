# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Orchestrator: load model → build engine → write bundle."""

from __future__ import annotations

import importlib
import inspect
import json
import os
import re
import sys
import tempfile
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from .build_timing import (
    add_build_timing as _add_build_timing,
    new_build_timing as _new_build_timing,
    write_build_timing as _write_build_timing,
)
from .config import ModelConfig
from .engine_build_budget import enforce_single_full_bundle_build
from .families import (
    available_plugin_ids,
    family_has_capability,
    family_prefers_native_default_build,
    find_plugin,
    find_diffusion_plugin,
    resolve_config_from_model_dir,
    _resolve_diffusion_family_id,
    resolve_family_id,
    resolve_family_model_dir,
    resolve_nemo_archive_model_dir,
)
from .hf_snapshot import GENERIC_HF_ALLOW_PATTERNS, hf_snapshot_allow_patterns
from .bundle_writer import BundleInfo, BundleSection, write_bundle
from . import trt_compat
from .triattention_export import (
    TriAttentionBundleConfig,
    export_triattention_stats_section,
)
from .parallel_config import (
    ParallelConfig,
    normalize_parallel_config,
    rank_engine_section,
    require_tensorrt_11_for_distributed,
    require_tensorrt_11_for_tensor_parallel,
)

try:
    from cuda.bindings import runtime as cudart
except ImportError:
    try:
        from cuda import cudart  # type: ignore[no-redef]
    except ImportError:  # pragma: no cover - depends on build environment
        cudart = None  # type: ignore[assignment]


class _OmittedMaxCacheLength(int):
    """Preserve the public 256 default while detecting an omitted argument."""


_OMITTED_MAX_CACHE_LENGTH = _OmittedMaxCacheLength(256)


def _setup_trt_import(rtx: bool) -> None:
    """Select the TensorRT Python backend before any TRT API is touched."""
    if not rtx:
        return
    trt_compat.configure_backend(rtx=True)
    print("[trtmc build] Using TensorRT-RTX backend", file=sys.stderr)


def _build_timing_phase(timing: dict, key: str) -> float:
    phases = timing.setdefault("phases", {})
    try:
        return float(phases.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _compile_time_excluding_component_weight_load(
    components_elapsed: float,
    weights_before_components: float,
    build_timing: dict,
) -> float:
    weights_after_components = _build_timing_phase(build_timing, "weights_loading_s")
    component_weight_elapsed = max(
        0.0, weights_after_components - weights_before_components)
    return max(0.0, components_elapsed - component_weight_elapsed)


def _untracked_compile_time(
    measured_compile_elapsed: float,
    compile_before_components: float,
    build_timing: dict,
) -> float:
    compile_after_components = _build_timing_phase(build_timing, "trt_compile_s")
    tracked_compile_elapsed = max(
        0.0, compile_after_components - compile_before_components)
    return max(0.0, measured_compile_elapsed - tracked_compile_elapsed)


# Backward-compatible alias for existing callers and builder contract tests.
_HF_ALLOW_PATTERNS = [*GENERIC_HF_ALLOW_PATTERNS]


def _compute_dynamic_kv_profile_rows(
    max_cache_length: int,
    kv_budget: int,
    *,
    bucket_rows: int = 32,
    preferred_rows: list[int] | None = None,
) -> list[int]:
    """Return ascending profile upper bounds for dynamic-KV engines.

    The runtime only changes KV shapes at coarse row buckets, so a small set of
    range profiles is enough. Each returned value is the maximum KV rows for
    one optimization profile.
    """
    if max_cache_length < 1:
        return [1]

    start = ((max(kv_budget, 1) + bucket_rows - 1) // bucket_rows) * bucket_rows
    start = max(bucket_rows, min(start, max_cache_length))

    rows: list[int] = []

    def add_row(value: int) -> None:
        rounded = ((min(max(value, 1), max_cache_length) + bucket_rows - 1) // bucket_rows) * bucket_rows
        rounded = max(bucket_rows, min(rounded, max_cache_length))
        if rounded not in rows:
            rows.append(rounded)

    if preferred_rows:
        for value in preferred_rows:
            add_row(value)

    row = start
    while row < max_cache_length:
        add_row(row)
        next_row = max(row + bucket_rows, row * 2)
        row = ((min(next_row, max_cache_length) + bucket_rows - 1) // bucket_rows) * bucket_rows
    add_row(max_cache_length)
    rows.sort()
    return rows


def _sanitize_dynamic_kv_profile_rows(
    rows: list[int] | None,
    max_cache_length: int,
) -> list[int] | None:
    if rows is None:
        return None
    sanitized: list[int] = []
    for value in rows:
        clamped = max(1, min(int(value), max_cache_length))
        if clamped not in sanitized:
            sanitized.append(clamped)
    sanitized.sort()
    if not sanitized:
        raise ValueError("dynamic_kv_profile_rows_override must contain at least one row")
    return sanitized


def add_dynamic_batch_profile(
    builder,
    config,
    network,
    *,
    input_names: list[str],
    max_batch: int,
    opt_batch: int,
    static_shape: dict[str, tuple[int, ...]],
) -> None:
    """Attach one TensorRT profile with a dynamic leading batch dimension.

    Used by every diffusion engine builder so the leading dim of each named
    input is dynamic in the range ``[1, max_batch]`` with kOPT=``opt_batch``.
    ``static_shape[name]`` is the per-input shape *without* the batch dim
    (e.g. ``(num_patches, hidden_dim)``).
    """
    del network  # The profile API is name-based; arg kept for call-site clarity.
    if max_batch < 1:
        raise ValueError(f"max_batch must be >= 1 (got {max_batch})")
    if not (1 <= opt_batch <= max_batch):
        raise ValueError(
            "opt_batch must satisfy 1 <= opt_batch <= max_batch "
            f"(got opt_batch={opt_batch}, max_batch={max_batch})"
        )
    missing = [name for name in input_names if name not in static_shape]
    if missing:
        raise KeyError(
            f"static_shape missing entries for: {', '.join(missing)}"
        )
    profile = builder.create_optimization_profile()
    for name in input_names:
        tail = tuple(static_shape[name])
        profile.set_shape(
            name,
            min=(1, *tail),
            opt=(opt_batch, *tail),
            max=(max_batch, *tail),
        )
    config.add_optimization_profile(profile)


def _raise_friendly_download_error(model_id: str, exc: Exception) -> None:
    """Re-raise HF download errors with clear, actionable messages."""
    exc_type = type(exc).__name__

    if "RepositoryNotFound" in exc_type:
        raise RuntimeError(
            f"Model '{model_id}' not found on HuggingFace Hub. "
            f"Check the repo ID for typos (format: 'org/model-name'). "
            f"If it's a private repo, run: huggingface-cli login"
        ) from exc

    if "GatedRepo" in exc_type:
        raise RuntimeError(
            f"Model '{model_id}' is gated. Accept the license at "
            f"https://huggingface.co/{model_id} then run: huggingface-cli login"
        ) from exc

    if "LocalEntryNotFound" in exc_type or "EntryNotFound" in exc_type:
        raise RuntimeError(
            f"Model '{model_id}' exists but required files are missing. "
            f"The model may use a non-standard layout."
        ) from exc

    if "HTTPError" in exc_type or "ConnectionError" in exc_type:
        raise RuntimeError(
            f"Network error downloading '{model_id}': {exc}. "
            f"Check your internet connection and try again."
        ) from exc

    if "OSError" in exc_type and "disk" in str(exc).lower():
        raise RuntimeError(
            f"Disk error downloading '{model_id}': {exc}. "
            f"Check available disk space."
        ) from exc

    # Fallback: re-raise with context
    raise RuntimeError(
        f"Failed to download '{model_id}' from HuggingFace Hub: {exc}"
    ) from exc


def _call_supports_kwarg(func, name: str) -> bool:
    """Return True when a callable explicitly accepts a kwarg or **kwargs."""
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return False
    if name in sig.parameters:
        return True
    return any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in sig.parameters.values()
    )


def _normalize_bundle_sections(
    plugin,
    raw_sections,
    *,
    hook_name: str = "diffusion_bundle_sections",
) -> list[BundleSection]:
    sections: list[BundleSection] = []
    for index, item in enumerate(raw_sections):
        if isinstance(item, BundleSection):
            section = item
        elif isinstance(item, dict):
            section = BundleSection(item.get("name"), item.get("data"))
        else:
            try:
                name, data = item
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"Plugin {plugin.name}.{hook_name}() item "
                    f"{index} must be a BundleSection, dict, or (name, data) pair"
                ) from exc
            section = BundleSection(name, data)
        if not isinstance(section.name, str) or not section.name:
            raise TypeError(
                f"Plugin {plugin.name}.{hook_name}() item "
                f"{index} has invalid section name {section.name!r}"
            )
        if not isinstance(section.data, (bytes, bytearray)):
            raise TypeError(
                f"Plugin {plugin.name}.{hook_name}() item "
                f"{index} data must be bytes"
            )
        sections.append(BundleSection(section.name, bytes(section.data)))
    return sections


def _diffusion_bundle_sections_from_plugin(
    plugin,
    components: dict,
    parallel: ParallelConfig,
) -> list[BundleSection]:
    bundle_sections = getattr(plugin, "diffusion_bundle_sections", None)
    if not callable(bundle_sections):
        raise ValueError(
            f"Plugin {plugin.name} must implement diffusion_bundle_sections() "
            "so bundle section policy stays model-owned"
        )
    kwargs = {}
    if _call_supports_kwarg(bundle_sections, "parallel_config"):
        kwargs["parallel_config"] = parallel
    raw_sections = bundle_sections(components, **kwargs)
    if raw_sections is None:
        raise ValueError(
            f"Plugin {plugin.name}.diffusion_bundle_sections() returned None"
        )
    return _normalize_bundle_sections(plugin, raw_sections)


def _diffusion_tokenizer_add_special_tokens_from_plugin(
    plugin,
    model_dir_path: Path,
) -> bool:
    detector = getattr(plugin, "diffusion_tokenizer_add_special_tokens", None)
    if not callable(detector):
        raise ValueError(
            f"Plugin {plugin.name} must implement "
            "diffusion_tokenizer_add_special_tokens() so tokenizer priority "
            "policy stays model-owned"
        )
    kwargs = {}
    if _call_supports_kwarg(detector, "detect_tokenizer_add_special_tokens"):
        kwargs["detect_tokenizer_add_special_tokens"] = _detect_tokenizer_add_special_tokens
    return bool(detector(model_dir_path, **kwargs))


def _diffusion_tokenizer_special_frame_from_plugin(
    plugin,
    model_dir_path: Path,
    *,
    detect_tokenizer_special_frame=None,
) -> tuple[list[int], list[int]] | None:
    detector = getattr(plugin, "diffusion_tokenizer_special_frame", None)
    if not callable(detector):
        return None
    if detect_tokenizer_special_frame is None:
        detect_tokenizer_special_frame = _detect_tokenizer_special_frame
    kwargs = {}
    if _call_supports_kwarg(detector, "detect_tokenizer_special_frame"):
        kwargs["detect_tokenizer_special_frame"] = detect_tokenizer_special_frame
    frame = detector(model_dir_path, **kwargs)
    if frame is None:
        return None
    if not isinstance(frame, tuple) or len(frame) != 2:
        raise TypeError(
            f"Plugin {plugin.name}.diffusion_tokenizer_special_frame() must "
            "return (prefix_ids, suffix_ids) or None"
        )
    prefix, suffix = frame
    return [int(token_id) for token_id in prefix], [int(token_id) for token_id in suffix]


def _diffusion_tokenizer_bundle_sections_from_plugin(
    plugin,
    model_dir_path: Path,
) -> list[BundleSection]:
    tokenizer_sections = getattr(plugin, "diffusion_tokenizer_bundle_sections", None)
    if not callable(tokenizer_sections):
        raise ValueError(
            f"Plugin {plugin.name} must implement "
            "diffusion_tokenizer_bundle_sections() so tokenizer bundle section "
            "policy stays model-owned"
        )
    kwargs = {}
    if _call_supports_kwarg(tokenizer_sections, "ensure_tokenizer_json"):
        kwargs["ensure_tokenizer_json"] = (
            lambda tokenizer_dir: _ensure_tokenizer_json(tokenizer_dir, plugin=plugin)
        )
    raw_sections = tokenizer_sections(model_dir_path, **kwargs)
    if raw_sections is None:
        raise ValueError(
            f"Plugin {plugin.name}.diffusion_tokenizer_bundle_sections() returned None"
        )
    return _normalize_bundle_sections(
        plugin,
        raw_sections,
        hook_name="diffusion_tokenizer_bundle_sections",
    )


def _tokenizer_json_bundle_override_from_plugin(
    plugin,
    model_dir_path: Path,
) -> bytes | None:
    provider = getattr(plugin, "tokenizer_json_bundle_override", None)
    if not callable(provider):
        return None
    payload = provider(model_dir_path)
    if payload is None:
        return None
    if not isinstance(payload, bytes):
        raise TypeError(
            f"Plugin {plugin.name}.tokenizer_json_bundle_override() must "
            "return bytes or None"
        )
    return payload


def _quant_format_name(quant_ctx) -> str | None:
    quant_format = getattr(getattr(quant_ctx, "profile", None), "format", None)
    return getattr(quant_format, "name", None)


def _plugin_supports_parallel_quantization(plugin, quant_ctx) -> bool:
    supports = getattr(plugin, "supports_parallel_quantization", None)
    if not callable(supports):
        return False
    return bool(supports(_quant_format_name(quant_ctx)))


def _plugin_graph_ops_module(plugin):
    """Load the graph helper module owned by the selected family plugin."""
    provided = getattr(plugin, "graph_ops", None)
    if provided is not None:
        return provided

    module_name = getattr(plugin.__class__, "__module__", "")
    if module_name.endswith(".plugin"):
        package_name = module_name.rsplit(".", 1)[0]
    else:
        package_name = module_name
    if not package_name:
        raise RuntimeError(
            f"Cannot resolve family graph_ops module for {plugin!r}")
    return importlib.import_module(f"{package_name}.graph_ops")


def _plugin_supports_split_decoder_roles(plugin, config: ModelConfig) -> bool:
    """Return True when a family explicitly opts into split decoder roles."""
    supports = getattr(plugin, "supports_split_decoder_roles", None)
    if callable(supports):
        return bool(supports(config))
    if isinstance(supports, bool):
        return supports
    return family_has_capability(config, "split_decoder_roles")


def _apply_family_builder_capabilities(config: ModelConfig) -> None:
    """Thread family-owned builder capabilities through generic config flags."""
    if family_has_capability(config, "disable_dual_profile_decoder"):
        config.raw["_disable_dual_profile_decoder"] = True


def _plugin_runtime_capabilities(plugin) -> set[str]:
    raw = getattr(plugin, "runtime_capabilities", ())
    if isinstance(raw, str):
        return {raw}
    try:
        return {str(item) for item in raw}
    except TypeError:
        return set()


def _is_decoder_kv_runtime(plugin, runtime_strategy: str) -> bool:
    del runtime_strategy
    if "decoder_kv" in _plugin_runtime_capabilities(plugin):
        return True
    return False


def _can_build_split_decoder_engines(
    plugin,
    config: ModelConfig,
    runtime_strategy: str,
    *,
    dynamic_kv_cache: bool,
    triattention_enabled: bool,
) -> bool:
    """Return True when split prefill/decode engines are supported.

    The split layout relies on ``standard_decoder_builder`` honoring the
    internal ``_decoder_engine_role`` passthrough. Custom MoE, recurrent,
    TriAttention, and dynamic-KV runtimes keep their existing single-engine
    behavior until they opt into the same contract. Embed-input families must
    additionally opt in because their prefill engine has a different input
    contract from an ordinary text decoder.
    """
    if not _is_decoder_kv_runtime(plugin, runtime_strategy):
        return False
    if dynamic_kv_cache or triattention_enabled:
        return False
    if (bool(getattr(plugin, "embed_input", False)) and
            not bool(getattr(plugin, "supports_split_embed_input", False))):
        return False
    return _plugin_supports_split_decoder_roles(plugin, config)


def _load_plugin_weights(
    plugin,
    model_dir: str,
    config: ModelConfig,
    *,
    precision: str,
):
    """Call plugin.load_weights(), forwarding precision when supported."""
    kwargs = {}
    if _call_supports_kwarg(plugin.load_weights, "precision"):
        kwargs["precision"] = precision
    return plugin.load_weights(model_dir, config, **kwargs)


def _is_hf_model_dir(path: Path) -> bool:
    """Return True if path contains a standard HF model entrypoint config."""
    return (path / "config.json").exists() or (path / "model_index.json").exists()


def _resolve_diffusion_entrypoint(model_dir: Path) -> tuple[dict, object | None] | None:
    """Resolve a conventional Diffusers index or a family-claimed root config."""
    index_path = model_dir / "model_index.json"
    if index_path.is_file():
        config = json.loads(index_path.read_text())
        if not isinstance(config, dict):
            raise ValueError(f"Diffusion model index must be a JSON object: {index_path}")
        pipeline_class = str(config.get("_class_name", "") or "")
        return config, find_diffusion_plugin(pipeline_class) or find_plugin(
            pipeline_class.lower()
        )

    config_path = model_dir / "config.json"
    if not config_path.is_file():
        return None
    config = json.loads(config_path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"Model config must be a JSON object: {config_path}")
    pipeline_class = config.get("_class_name")
    if not isinstance(pipeline_class, str) or not pipeline_class:
        return None

    plugin = find_diffusion_plugin(pipeline_class) or find_plugin(
        pipeline_class.lower()
    )
    return (config, plugin) if plugin is not None else None


def _is_family_model_dir(path: Path) -> bool:
    """Return True when a family-owned config adapter can parse the directory."""
    return path.is_dir() and resolve_config_from_model_dir(path) is not None


def _resolve_model(model_id_or_path: str, *, revision: str | None = None) -> str:
    """Resolve a HuggingFace repo ID or local path to a local directory.

    If model_id_or_path is an existing directory with config.json, returns it
    directly. Otherwise, downloads via huggingface_hub.snapshot_download().
    Handles .nemo archives through family-owned adapters.
    """
    local = Path(model_id_or_path)
    if local.is_dir():
        staged = resolve_family_model_dir(local)
        if staged is not None:
            return staged
    if local.is_dir() and (_is_hf_model_dir(local) or _is_family_model_dir(local)):
        return str(local)

    # Handle NeMo archive snapshots.
    if local.is_file() and local.suffix == ".nemo":
        return _resolve_nemo_archive(local)

    # Handle HF directories that contain .nemo files
    if local.is_dir():
        nemo_files = list(local.glob("*.nemo"))
        if nemo_files:
            return _resolve_nemo_archive(nemo_files[0])

    # Treat as HuggingFace repo ID — download to HF cache.
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for auto-downloading models. "
            "Install it with: pip install huggingface_hub"
        )

    print(f"[trtmc build] Downloading {model_id_or_path} ...", file=sys.stderr)
    try:
        local_dir = snapshot_download(
            repo_id=model_id_or_path,
            revision=revision,
            allow_patterns=hf_snapshot_allow_patterns(),
            local_files_only=os.environ.get("HF_HUB_OFFLINE", "").lower()
            in {"1", "on", "true", "yes"},
        )
    except Exception as exc:
        _raise_friendly_download_error(model_id_or_path, exc)

    # Prefer HF config when both HF files and .nemo are present.
    dl_path = Path(local_dir)
    staged = resolve_family_model_dir(dl_path)
    if staged is not None:
        print(f"[trtmc build] Downloaded to {local_dir}", file=sys.stderr)
        return staged
    if _is_hf_model_dir(dl_path):
        print(f"[trtmc build] Downloaded to {local_dir}", file=sys.stderr)
        return local_dir

    # Fallback for NeMo-only snapshots.
    nemo_files = sorted(dl_path.glob("*.nemo"))
    if nemo_files:
        return _resolve_nemo_archive(nemo_files[0])

    print(f"[trtmc build] Downloaded to {local_dir}", file=sys.stderr)
    return local_dir


def _resolve_nemo_archive(nemo_path: Path) -> str:
    """Resolve a .nemo archive through family-owned archive adapters."""
    resolved = resolve_nemo_archive_model_dir(nemo_path)
    if resolved is None:
        raise RuntimeError(
            f"No family-owned NeMo archive adapter recognized {nemo_path}"
        )
    return resolved


def _get_trt_version() -> str:
    return trt_compat.tensorrt_version() or "unknown"


def _trt_abi_from_version(version: str) -> str:
    match = re.search(r"(\d+)\.(\d+)", version or "")
    if not match:
        return ""
    return f"{match.group(1)}.{match.group(2)}"


def _get_gpu_name() -> str:
    if cudart is None:
        return ""
    try:
        success = (
            cudart.cudaError_t.cudaSuccess
            if hasattr(cudart, "cudaError_t")
            else 0
        )
        status, device = cudart.cudaGetDevice()
        if status != success:
            return ""
        status, properties = cudart.cudaGetDeviceProperties(device)
        if status != success:
            return ""
        name = properties.name
        if isinstance(name, bytes):
            return name.decode("utf-8", errors="replace").rstrip("\x00")
        return str(name).rstrip("\x00")
    except Exception:
        pass
    return ""


def _apply_generation_config_eos(model_dir: Path, config: dict) -> None:
    """Apply Hugging Face generation-config EOS precedence to runtime config."""
    generation_config_path = model_dir / "generation_config.json"
    if not generation_config_path.exists():
        return
    generation_config = json.loads(generation_config_path.read_text(encoding="utf-8"))
    if "eos_token_id" in generation_config:
        config["eos_token_id"] = generation_config["eos_token_id"]


def _detect_tokenizer_add_special_tokens(model_dir: Path) -> bool:
    """Detect whether the HF tokenizer adds special tokens (BOS/EOS) by default.

    The C++ runtime calls the native tokenizer with a single add-special flag,
    so this mirrors the default HF ``tokenizer.encode(text)`` behavior.
    tokenizer_config.json is only a fallback because some tokenizers expose
    stale add_bos/add_eos fields while still adding a post-processor token by
    default.
    """
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        ids_default = tok.encode("hello")
        ids_without = tok.encode("hello", add_special_tokens=False)
        return ids_default != ids_without
    except Exception:
        pass

    # Fallback for lightweight/unit-test environments without a loadable tokenizer.
    tok_config_path = model_dir / "tokenizer_config.json"
    if tok_config_path.exists():
        try:
            tok_cfg = json.load(open(tok_config_path))
            if bool(tok_cfg.get("add_bos_token", False)):
                return True
            if bool(tok_cfg.get("add_eos_token", False)):
                return True
        except Exception:
            pass

    return False


def _detect_tokenizer_special_frame(
    model_dir: Path | str,
    *,
    revision: str | None = None,
    local_files_only: bool = False,
) -> tuple[list[int], list[int]] | None:
    """Return exact HF add-special prefix/suffix IDs when they are representable.

    Some tokenizers add BOS by default but not EOS. A single
    add-special boolean is not enough for the native C++ tokenizer to mirror
    that behavior, so bundle the exact frame when HF exposes it as a simple
    prefix/suffix around the no-special tokenization.
    """
    try:
        from transformers import AutoTokenizer

        tokenizer_kwargs = {"trust_remote_code": True}
        if revision:
            tokenizer_kwargs["revision"] = revision
        if local_files_only:
            tokenizer_kwargs["local_files_only"] = True
        tok = AutoTokenizer.from_pretrained(str(model_dir), **tokenizer_kwargs)
        ids_default = list(tok.encode("hello"))
        ids_without = list(tok.encode("hello", add_special_tokens=False))
    except Exception:
        return None

    if ids_default == ids_without:
        return [], []
    if not ids_without:
        return ids_default, []

    needle_len = len(ids_without)
    for start in range(0, len(ids_default) - needle_len + 1):
        if ids_default[start:start + needle_len] == ids_without:
            return ids_default[:start], ids_default[start + needle_len:]
    return None


def _wordpiece_tokenizer_needs_rebuild(model_dir: Path) -> bool:
    """Return whether a cached WordPiece tokenizer is smaller than the model.

    Some Hugging Face repositories, including ConvBERT, publish ``vocab.txt``
    without ``tokenizer.json``.  A tokenizer generated before ``vocab.txt`` is
    available can contain only the special tokens and then remain in the shared
    snapshot cache.  Rebuild only when all inputs prove that this happened.
    """
    tokenizer_path = model_dir / "tokenizer.json"
    vocab_path = model_dir / "vocab.txt"
    config_path = model_dir / "config.json"
    if not (tokenizer_path.exists() and vocab_path.exists() and config_path.exists()):
        return False

    try:
        tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        model = tokenizer.get("model", {})
        vocab = model.get("vocab", {})
        expected_vocab_size = int(config.get("vocab_size", 0))
        if model.get("type") != "WordPiece":
            return False
        if not isinstance(vocab, dict) or not vocab or expected_vocab_size <= 0:
            return False
        tokenizer_id_space = max(int(token_id) for token_id in vocab.values()) + 1
        source_id_space = len(vocab_path.read_text(encoding="utf-8").splitlines())
        return (
            tokenizer_id_space < expected_vocab_size
            and source_id_space >= expected_vocab_size
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _ensure_tokenizer_json(model_dir: Path, *, plugin=None) -> None:
    """Generate a native tokenizer.json from the Hugging Face fast tokenizer.

    Slow tokenizers commonly save only their source vocabulary files.  The
    C++ runtime needs the serialized Rust tokenizer backend, which is exposed
    by fast tokenizers as ``backend_tokenizer``.
    """
    tokenizer_path = model_dir / "tokenizer.json"
    rebuild_wordpiece = _wordpiece_tokenizer_needs_rebuild(model_dir)
    family_ensure = getattr(plugin, "ensure_tokenizer_json", None)
    if tokenizer_path.exists() and not rebuild_wordpiece:
        if callable(family_ensure):
            kwargs = {}
            if _call_supports_kwarg(family_ensure, "previous_error"):
                kwargs["previous_error"] = None
            if not bool(family_ensure(model_dir, **kwargs)):
                raise RuntimeError(
                    "family tokenizer validation rejected existing tokenizer.json"
                )
        return
    family_first = (
        getattr(plugin, "tokenizer_json_conversion_policy", "")
        == "family_first"
    )
    if family_first and callable(family_ensure):
        kwargs = {}
        if _call_supports_kwarg(family_ensure, "previous_error"):
            kwargs["previous_error"] = None
        if bool(family_ensure(model_dir, **kwargs)):
            return
    if rebuild_wordpiece:
        print(
            "[trtmc build] Rebuilding undersized WordPiece tokenizer.json "
            "from vocab.txt",
            file=sys.stderr,
        )

    tokenizer_conversion_error: str | None = None

    # --- Attempt 1: standard HF conversion in an isolated directory ---
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
        with tempfile.TemporaryDirectory(prefix="trtmc-tokenizer-") as temporary_dir:
            generated_path = Path(temporary_dir) / "tokenizer.json"
            backend_tokenizer = getattr(tok, "backend_tokenizer", None)
            if backend_tokenizer is None:
                backend_tokenizer = getattr(tok, "_tokenizer", None)
            if backend_tokenizer is not None and hasattr(backend_tokenizer, "save"):
                backend_tokenizer.save(str(generated_path))
            if not generated_path.exists():
                tok.save_pretrained(temporary_dir)
            if not generated_path.exists():
                raise RuntimeError(
                    "tokenizer conversion did not create tokenizer.json"
                )
            with tempfile.NamedTemporaryFile(
                dir=model_dir,
                prefix=".trtmc-tokenizer-",
                suffix=".json",
                delete=False,
            ) as output:
                temporary_path = Path(output.name)
                output.write(generated_path.read_bytes())
            temporary_path.replace(tokenizer_path)
        print("[trtmc build] Generated tokenizer.json from source tokenizer",
              file=sys.stderr)
        return
    except Exception as e:
        tokenizer_conversion_error = f"fast tokenizer conversion failed: {e}"

    if callable(family_ensure):
        kwargs = {}
        if _call_supports_kwarg(family_ensure, "previous_error"):
            kwargs["previous_error"] = tokenizer_conversion_error
        if bool(family_ensure(model_dir, **kwargs)):
            return

    detail = tokenizer_conversion_error or "no tokenizer conversion was attempted"
    print(
        "[trtmc build] Warning: could not generate tokenizer.json "
        f"(C++ runtime may fail to create tokenizer): {detail}",
        file=sys.stderr,
    )


def _prepare_tokenizer_special_frame(
    model_dir: Path,
    *,
    plugin=None,
    source_model_id_or_path: str | None = None,
    source_revision: str | None = None,
) -> tuple[list[int], list[int]] | None:
    """Generate tokenizer.json without changing the source tokenizer contract."""

    source = source_model_id_or_path or str(model_dir)
    source_is_remote = not Path(source).is_dir()
    source_frame = _detect_tokenizer_special_frame(
        source,
        revision=source_revision,
        local_files_only=source_is_remote,
    )
    _ensure_tokenizer_json(model_dir, plugin=plugin)
    if source_frame is not None:
        return source_frame
    return _detect_tokenizer_special_frame(model_dir)


@enforce_single_full_bundle_build
def build_bundle(
    model_dir: str,
    output_path: str,
    max_cache_length: int | None = None,
    *,
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
    tokenizer_source_model_id_or_path: str | None = None,
    tokenizer_source_revision: str | None = None,
) -> None:
    """Full pipeline: load HF model → build TRT engine → write .bundle artifact.

    Args:
        model_dir: Path to HF model directory with config.json + safetensors.
        output_path: Where to write the .bundle artifact.
        max_cache_length: KV cache length for the engine. ``None`` selects the
            model-owned default; decoder families may use the model's official
            context capacity.
        decoder_engine_layout: ``"split"`` builds separate prefill/decode
            engines for supported decoder LLMs. ``"dual_profile"`` keeps the
            low-VRAM single-engine/multi-profile layout.
        verbose: Print detailed logs.
    """
    owner_options = dict(locals())
    owner_options.pop("model_dir")
    owner_options.pop("output_path")

    if decoder_engine_layout not in ("split", "dual_profile"):
        raise ValueError(
            "decoder_engine_layout must be 'split' or 'dual_profile', "
            f"got {decoder_engine_layout!r}")
    _setup_trt_import(rtx)
    parallel = normalize_parallel_config(parallel_config)
    try:
        print(
            f"[trtmc build] Builder TensorRT resolved: {trt_compat.resolved_summary()}",
            file=sys.stderr,
        )
    except ImportError as exc:
        raise ImportError(
            "TensorRT Python bindings are required for raw TRT builds. "
            "Install a matching tensorrt package in the active Python environment."
        ) from exc
    model_dir_path = Path(model_dir)
    t0 = time.monotonic()
    build_timing = _new_build_timing(build_timing_path)
    build_timing["model_dir"] = str(model_dir_path)
    build_timing["output_path"] = str(output_path)
    _write_build_timing(build_timing)

    diffusion_entrypoint = _resolve_diffusion_entrypoint(model_dir_path)
    if diffusion_entrypoint is not None:
        if max_cache_length is None:
            max_cache_length = 256
        _, diffusion_plugin = diffusion_entrypoint
        precision = str(
            precision
            or getattr(diffusion_plugin, "default_build_precision", "fp32")
        ).lower()
        fp8_scales = getattr(build_bundle, '_fp8_scales', None)
        save_fp8_scales = getattr(build_bundle, '_save_fp8_scales', None)
        _build_diffusion_bundle(
            model_dir_path, output_path, max_cache_length,
            precision=precision, fp32_layers=fp32_layers,
            verbose=verbose, t0=t0,
            fp8_scales=fp8_scales, save_fp8_scales=save_fp8_scales,
            rtx=rtx,
            family_build_options=family_build_options,
            diffusion_overrides=diffusion_overrides,
            build_timing=build_timing,
            parallel_config=parallel,
            max_batch_size=max_batch_size)
        return

    if parallel.cp_enabled:
        raise NotImplementedError(
            "Context parallelism is currently supported for diffusion family plugins only")

    # 1. Parse config
    config = ModelConfig.from_dir(model_dir_path)
    config.raw["_model_dir"] = str(model_dir_path)
    config.raw["_decoder_engine_layout"] = decoder_engine_layout
    config.raw["_fp32_layers"] = sorted(set(fp32_layers or ()))
    config.raw["_family_build_options"] = dict(family_build_options or {})
    config.raw["_parallel_build_enabled"] = bool(parallel.enabled)
    config.raw["_rtx_build_requested"] = bool(rtx)
    config.raw["_runtime_dynamic_kv_requested"] = bool(
        dynamic_kv_cache or triattention_stats_path
    )
    # Family routing chooses defaults before the quantization context exists.
    # Preserve the user's build mode on the parsed config so a native-only
    # family does not select its full-context defaults for an unsupported
    # quantized build and only fall back after loading the checkpoint.
    config.raw["_quantized_build_requested"] = bool(quantize)
    _apply_family_builder_capabilities(config)
    print(f"[trtmc build] Model: {config.model_type} "
          f"(layers={config.num_hidden_layers}, hidden={config.hidden_size}, "
          f"vocab={config.vocab_size})", file=sys.stderr)

    # 2. Find family plugin
    plugin = find_plugin(config)
    if plugin is None:
        supported = ", ".join(available_plugin_ids())
        raise ValueError(
            f"No family plugin for model_type={config.model_type!r}. "
            f"Supported: {supported}")

    if family_has_capability(config, "model_owned_build"):
        missing = object()
        declared_build = inspect.getattr_static(plugin, "build", missing)
        if declared_build is missing:
            raise TypeError(
                f"Family {plugin.name} declares model_owned_build without "
                "a concrete build binding"
            )
        if not callable(declared_build):
            raise TypeError(
                f"Family {plugin.name} model-owned build binding is not callable"
            )
        owner_build = getattr(plugin, "build")
        if not callable(owner_build):
            raise TypeError(
                f"Family {plugin.name} model-owned build binding is not callable"
            )
        owner_build(str(model_dir_path), output_path, **owner_options)
        return

    print(f"[trtmc build] Family: {plugin.name}", file=sys.stderr)
    default_precision = getattr(plugin, "default_build_precision", "fp32")
    if callable(default_precision):
        default_precision = default_precision(config)
    precision = str(precision or default_precision).lower()
    config.raw["_resolved_build_precision"] = precision
    if max_cache_length is None:
        default_capacity = getattr(plugin, "default_max_cache_length", None)
        max_cache_length = int(
            default_capacity(config) if callable(default_capacity) else 256
        )
        print(
            "[trtmc build] KV cache capacity: "
            f"{max_cache_length} tokens (model default)",
            file=sys.stderr,
        )
    if max_cache_length < 1:
        raise ValueError("max_cache_length must be >= 1")

    validate_build_request = getattr(plugin, "validate_build_request", None)
    if callable(validate_build_request):
        validate_build_request(config)

    # 3. Load weights
    t1 = time.monotonic()
    print("[trtmc build] Loading weights ...", file=sys.stderr)
    try:
        weights = _load_plugin_weights(
            plugin, str(model_dir_path), config, precision=precision)
    finally:
        weights_elapsed = time.monotonic() - t1
        _add_build_timing(build_timing, "weights_loading_s", weights_elapsed)
        _write_build_timing(build_timing)
    print(f"[trtmc build] Weights loaded [{weights_elapsed:.1f}s]", file=sys.stderr)

    # 3b. Build quantization context (if requested)
    quant_ctx = None
    quant_plan = None
    if quantize:
        quant_t0 = time.monotonic()
        from .quantization import QuantPlan, build_quant_context
        try:
            quant_plan = QuantPlan.from_build_args(
                precision=precision,
                quantize=quantize,
                quant_scales=quant_scales,
                quant_calibration_samples=quant_calibration_samples,
            )
            quant_method = str(config.raw.get(
                "quantization_config", {}).get("quant_method", "")).lower()
            if (quant_plan.scale_source == "modelopt"
                    and quant_method in {
                        "awq", "gptq", "compressed-tensors", "compressed_tensors"
                    }):
                quant_plan = replace(quant_plan, scale_source="prequantized")
            exclude_patterns = (plugin.quant_exclude_patterns(quant_plan.quant_format)
                                if hasattr(plugin, 'quant_exclude_patterns') else None)
            quant_ctx = build_quant_context(
                format_name=quant_plan.quant_format,
                model_dir=str(model_dir_path),
                config=config,
                exclude_patterns=exclude_patterns,
                scales_json=quant_scales,
                num_calibration_samples=quant_calibration_samples,
                plugin=plugin,
                quant_plan=quant_plan,
                graph_ops=_plugin_graph_ops_module(plugin),
            )
        finally:
            _add_build_timing(
                build_timing, "quantization_context_s",
                time.monotonic() - quant_t0)
            _write_build_timing(build_timing)
        print(f"[trtmc build] Quantization: {quant_plan.quant_format}",
              file=sys.stderr)

    # 4. Build TRT engine
    triattention_cfg = None
    triattention_section = None
    runtime_strategy = getattr(plugin, "runtime_strategy", "") or ""
    enable_dynamic_kv_cache = bool(dynamic_kv_cache)
    dynamic_kv_profile_rows = _sanitize_dynamic_kv_profile_rows(
        dynamic_kv_profile_rows_override,
        max_cache_length,
    )
    if triattention_stats_path:
        if not _is_decoder_kv_runtime(plugin, runtime_strategy):
            raise ValueError(
                "TriAttention is only supported for decoder KV-cache runtimes. "
                f"Found runtime_strategy={runtime_strategy!r}."
            )
        if triattention_recent_window < 0:
            raise ValueError(
                "TriAttention recent_window must be >= 0. "
                f"Got recent_window={triattention_recent_window}."
            )
        if triattention_divide_length < 1:
            raise ValueError(
                "TriAttention divide_length must be >= 1. "
                f"Got divide_length={triattention_divide_length}."
            )
        if triattention_score_aggregation not in ("mean", "max"):
            raise ValueError(
                "TriAttention score_aggregation must be 'mean' or 'max'. "
                f"Got {triattention_score_aggregation!r}."
            )
        kv_budget = int(
            triattention_kv_budget
            if triattention_kv_budget is not None
            else max_cache_length
        )
        if kv_budget < 1 or kv_budget > max_cache_length:
            raise ValueError(
                "TriAttention kv_budget must be in [1, max_cache_length]. "
                f"Got kv_budget={kv_budget}, max_cache_length={max_cache_length}."
            )
        triattention_cfg = TriAttentionBundleConfig(
            kv_budget=kv_budget,
            divide_length=triattention_divide_length,
            recent_window=triattention_recent_window,
            score_aggregation=triattention_score_aggregation,
            count_prompt_tokens=triattention_count_prompt_tokens,
            protect_prefill=triattention_protect_prefill,
            disable_mlr=triattention_disable_mlr,
            disable_trig=triattention_disable_trig,
        )
        triattention_section = export_triattention_stats_section(
            triattention_stats_path,
            config=config,
        )
        print(
            "[trtmc build] TriAttention: embedded calibration stats "
            f"from {triattention_stats_path} (kv_budget={kv_budget}, "
            f"divide_length={triattention_divide_length}, "
            f"recent_window={triattention_recent_window})",
            file=sys.stderr,
        )
        if dynamic_kv_profile_rows is None:
            preferred_rows: list[int] | None = None
            if kv_budget >= 4096:
                preferred_rows = [max(32, kv_budget // 2)]
            dynamic_kv_profile_rows = _compute_dynamic_kv_profile_rows(
                max_cache_length,
                kv_budget,
                preferred_rows=preferred_rows,
            )
        enable_dynamic_kv_cache = True

    if enable_dynamic_kv_cache:
        if not _is_decoder_kv_runtime(plugin, runtime_strategy):
            raise ValueError(
                "dynamic_kv_cache is only supported for decoder KV-cache runtimes. "
                f"Found runtime_strategy={runtime_strategy!r}."
            )
        if dynamic_kv_profile_rows is None:
            dynamic_kv_profile_rows = _compute_dynamic_kv_profile_rows(max_cache_length, 1)
        config.raw["dynamic_kv_cache"] = True
        config.raw["_dynamic_kv_opt_length"] = max_cache_length
        config.raw["_dynamic_kv_profile_rows"] = dynamic_kv_profile_rows

    if parallel.enabled:
        require_tensorrt_11_for_tensor_parallel(parallel)
        if quant_ctx is not None and not _plugin_supports_parallel_quantization(
            plugin, quant_ctx
        ):
            raise ValueError("Tensor-parallel decoder builds do not support quantization yet")
        if enable_dynamic_kv_cache:
            raise NotImplementedError(
                "Tensor-parallel decoder builds do not support dynamic_kv_cache "
                "or TriAttention yet")
        if not _call_supports_kwarg(plugin.build_engine, "parallel_config"):
            raise ValueError(
                f"Plugin {plugin.name} does not support tensor-parallel builds")
        print(
            f"[trtmc-build] Building tensor-parallel TRT engines "
            f"(tp={parallel.tp_size}, cache={max_cache_length}) ...",
            file=sys.stderr,
        )

    # Pass precision/quant_ctx only if the plugin accepts them (not all do).
    extra_kwargs = {}
    if _call_supports_kwarg(plugin.build_engine, 'precision'):
        extra_kwargs['precision'] = precision
    if _call_supports_kwarg(plugin.build_engine, 'quant_ctx'):
        extra_kwargs['quant_ctx'] = quant_ctx
    if _call_supports_kwarg(plugin.build_engine, 'parallel_config'):
        extra_kwargs['parallel_config'] = parallel

    def _split_timing_cache_scope(role: str) -> str:
        quant_label = quantize or "noquant"
        return (
            f"split-{config.model_type}-h{config.hidden_size}"
            f"-l{config.num_hidden_layers}-{precision}-{quant_label}-{role}"
        )

    def _build_plugin_engine_with_role(role: str) -> bytes:
        from .tvm_ffi.graph_build import engine_role

        previous_role = config.raw.get("_decoder_engine_role")
        config.raw["_decoder_engine_role"] = role
        try:
            with engine_role(role):
                return plugin.build_engine(
                    config, weights, max_cache_length, verbose=verbose,
                    **extra_kwargs)
        finally:
            if previous_role is None:
                config.raw.pop("_decoder_engine_role", None)
            else:
                config.raw["_decoder_engine_role"] = previous_role

    def _build_split_plugin_engine_with_role(role: str) -> bytes:
        previous_active = config.raw.get("_active_split_decoder_build")
        config.raw["_active_split_decoder_build"] = True
        try:
            with trt_compat.scoped_timing_cache(_split_timing_cache_scope(role)):
                return _build_plugin_engine_with_role(role)
        finally:
            if previous_active is None:
                config.raw.pop("_active_split_decoder_build", None)
            else:
                config.raw["_active_split_decoder_build"] = previous_active

    from .tvm_ffi.graph_build import inspection_role

    target_inspection_role = inspection_role()
    if target_inspection_role is not None:
        _build_plugin_engine_with_role(target_inspection_role)
        raise RuntimeError("graph inspection did not reach TensorRT serialization")

    split_supported = (
        not parallel.enabled and
        decoder_engine_layout == "split" and
        _can_build_split_decoder_engines(
            plugin,
            config,
            runtime_strategy,
            dynamic_kv_cache=enable_dynamic_kv_cache,
            triattention_enabled=triattention_cfg is not None,
        )
    )

    engine_plan: bytes
    prefill_engine_plan: bytes | None = None
    tp_engine_plans: dict[int, bytes] = {}
    actual_decoder_engine_layout = "single"
    engine_t0 = time.monotonic()
    try:
        if parallel.enabled:
            for rank in range(parallel.tp_size):
                rank_kwargs = dict(extra_kwargs)
                rank_kwargs["parallel_config"] = parallel.for_rank(rank)
                print(f"[trtmc-build]   rank {rank}/{parallel.tp_size} ...",
                      file=sys.stderr)
                tp_engine_plans[rank] = plugin.build_engine(
                    config, weights, max_cache_length, verbose=verbose,
                    **rank_kwargs)
            engine_plan = tp_engine_plans[0]
            actual_decoder_engine_layout = "dual_profile"
        elif split_supported:
            print(
                f"[trtmc build] Building split decoder engines "
                f"(cache={max_cache_length}) ...",
                file=sys.stderr,
            )
            prefill_t0 = time.monotonic()
            prefill_engine_plan = _build_split_plugin_engine_with_role("prefill")
            prefill_elapsed = time.monotonic() - prefill_t0
            _add_build_timing(
                build_timing, "trt_compile_prefill_engine_s", prefill_elapsed)
            print(
                f"[trtmc build] Prefill engine built [{prefill_elapsed:.1f}s] "
                f"({len(prefill_engine_plan) / (1024 * 1024):.1f} MB)",
                file=sys.stderr,
            )

            decode_t0 = time.monotonic()
            engine_plan = _build_split_plugin_engine_with_role("decode")
            decode_elapsed = time.monotonic() - decode_t0
            _add_build_timing(
                build_timing, "trt_compile_decode_engine_s", decode_elapsed)
            print(
                f"[trtmc build] Decode engine built [{decode_elapsed:.1f}s] "
                f"({len(engine_plan) / (1024 * 1024):.1f} MB)",
                file=sys.stderr,
            )
            actual_decoder_engine_layout = "split"
        else:
            if decoder_engine_layout == "split" and _is_decoder_kv_runtime(plugin, runtime_strategy):
                print(
                    "[trtmc build] Split decoder layout is not supported for "
                    f"family={plugin.name}; using existing single-engine path",
                    file=sys.stderr,
                )
            print(f"[trtmc build] Building TRT engine (cache={max_cache_length}) ...",
                  file=sys.stderr)
            role = "dual_profile" if decoder_engine_layout == "dual_profile" else "decode"
            engine_plan = _build_plugin_engine_with_role(role)
            if decoder_engine_layout == "dual_profile":
                actual_decoder_engine_layout = "dual_profile"
    finally:
        engine_elapsed = time.monotonic() - engine_t0
        _add_build_timing(build_timing, "trt_compile_s", engine_elapsed)
        _add_build_timing(build_timing, "trt_compile_main_engine_s", engine_elapsed)
        _write_build_timing(build_timing)
    if actual_decoder_engine_layout == "split":
        total_mb = (len(engine_plan) + len(prefill_engine_plan or b"")) / (1024 * 1024)
        print(f"[trtmc build] Split engines built [{engine_elapsed:.1f}s] "
              f"({total_mb:.1f} MB total)", file=sys.stderr)
    elif parallel.enabled:
        total_mb = sum(len(plan) for plan in tp_engine_plans.values()) / (1024 * 1024)
        print(f"[trtmc-build] Tensor-parallel engines built [{engine_elapsed:.1f}s] "
              f"({total_mb:.1f} MB total)", file=sys.stderr)
    else:
        print(f"[trtmc build] Engine built [{engine_elapsed:.1f}s] "
              f"({len(engine_plan) / (1024 * 1024):.1f} MB)", file=sys.stderr)

    # 4b. Build vision engine (optional, VL models only)
    vision_plan = None
    build_vision = getattr(plugin, 'build_vision_engine', None)
    if build_vision is not None:
        print("[trtmc build] Building vision encoder engine ...",
              file=sys.stderr)
        vision_t0 = time.monotonic()
        try:
            build_vision_kwargs = {"verbose": verbose}
            if _call_supports_kwarg(build_vision, "precision"):
                build_vision_kwargs["precision"] = precision
            vision_plan = build_vision(
                str(model_dir_path), config, weights, **build_vision_kwargs)
        finally:
            vision_elapsed = time.monotonic() - vision_t0
            _add_build_timing(build_timing, "trt_compile_s", vision_elapsed)
            _add_build_timing(
                build_timing, "trt_compile_vision_engine_s", vision_elapsed)
            _write_build_timing(build_timing)
        if vision_plan is not None:
            print(f"[trtmc build] Vision engine built [{vision_elapsed:.1f}s] "
                  f"({len(vision_plan) / (1024 * 1024):.1f} MB)",
                  file=sys.stderr)

    # 4c. Build extra engines (optional for multi-engine model families)
    extra_engines = {}
    build_extra = getattr(plugin, 'build_extra_engines', None)
    if build_extra is not None:
        print("[trtmc build] Building extra engines ...", file=sys.stderr)
        extra_t0 = time.monotonic()
        compile_before_extra = _build_timing_phase(build_timing, "trt_compile_s")
        try:
            build_extra_kwargs = {"verbose": verbose}
            if _call_supports_kwarg(build_extra, "precision"):
                build_extra_kwargs["precision"] = precision
            if _call_supports_kwarg(build_extra, "quant_ctx"):
                build_extra_kwargs["quant_ctx"] = quant_ctx
            if _call_supports_kwarg(build_extra, "build_timing"):
                build_extra_kwargs["build_timing"] = build_timing
            if _call_supports_kwarg(build_extra, "parallel_config"):
                build_extra_kwargs["parallel_config"] = parallel
            extra_engines = build_extra(
                config, weights, max_cache_length, **build_extra_kwargs) or {}
        finally:
            extra_elapsed = time.monotonic() - extra_t0
            untracked_extra_elapsed = _untracked_compile_time(
                extra_elapsed, compile_before_extra, build_timing)
            _add_build_timing(build_timing, "trt_compile_s", untracked_extra_elapsed)
            _add_build_timing(
                build_timing, "trt_compile_extra_engines_s", extra_elapsed)
            _write_build_timing(build_timing)
        print(f"[trtmc build] Extra engines built [{extra_elapsed:.1f}s]",
              file=sys.stderr)
        for ename, eplan in extra_engines.items():
            print(f"[trtmc build]   {ename}: {len(eplan) / (1024 * 1024):.1f} MB",
                  file=sys.stderr)

    # 5. Ensure tokenizer.json before detecting its special-token frame.  A
    # repaired tokenizer may use different special-token IDs than a stale one.
    requires_tokenizer = bool(getattr(plugin, "requires_tokenizer", True))
    if requires_tokenizer:
        tokenizer_json_t0 = time.monotonic()
        tokenizer_special_frame = _prepare_tokenizer_special_frame(
            model_dir_path,
            plugin=plugin,
            source_model_id_or_path=tokenizer_source_model_id_or_path,
            source_revision=tokenizer_source_revision,
        )
        _add_build_timing(
            build_timing, "tokenizer_json_ensure_s",
            time.monotonic() - tokenizer_json_t0)
        _write_build_timing(build_timing)
    else:
        tokenizer_special_frame = _detect_tokenizer_special_frame(
            model_dir_path)

    # 5b. Detect tokenizer special-tokens behavior from HF config
    tokenizer_t0 = time.monotonic()
    if tokenizer_special_frame is None:
        tokenizer_special_prefix_ids: list[int] = []
        tokenizer_special_suffix_ids: list[int] = []
        tokenizer_add_special_tokens = _detect_tokenizer_add_special_tokens(
            model_dir_path)
    else:
        tokenizer_special_prefix_ids, tokenizer_special_suffix_ids = (
            tokenizer_special_frame)
        tokenizer_add_special_tokens = bool(
            tokenizer_special_prefix_ids or tokenizer_special_suffix_ids)
    _add_build_timing(
        build_timing, "tokenizer_special_tokens_detection_s",
        time.monotonic() - tokenizer_t0)
    _write_build_timing(build_timing)

    # 6. Write bundle
    trt_version = _get_trt_version()
    trt_abi = _trt_abi_from_version(trt_version)
    info = BundleInfo(
        model_id=model_dir_path.name,
        model_type=config.model_type,
        family=plugin.name,
        trt_version=trt_version,
        trt_abi=trt_abi,
        gpu_name=_get_gpu_name(),
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        max_cache_length=max_cache_length,
        runtime_strategy=getattr(plugin, "runtime_strategy", ""),
        precision=precision,
        quantization=(quant_plan.quant_format if quant_plan else "none"),
        tokenizer_add_special_tokens=tokenizer_add_special_tokens,
    )

    if parallel.enabled:
        sections = [
            BundleSection(rank_engine_section(rank), plan)
            for rank, plan in sorted(tp_engine_plans.items())
        ]
    else:
        # For split decoder bundles, keep ``engine_plan`` as the decode-only
        # engine for compatibility with existing tools and add the prefill engine
        # under a role-specific section.
        sections = [BundleSection("engine_plan", engine_plan)]
        if prefill_engine_plan is not None:
            sections.append(BundleSection("prefill_engine_plan", prefill_engine_plan))

    # Add vision engine section if present
    if vision_plan is not None:
        sections.append(BundleSection("vision_engine_plan", vision_plan))

    from .tvm_ffi.graph_build import kernel_slots_section

    slot_section = kernel_slots_section()
    if slot_section is not None:
        sections.append(BundleSection("kernel_slots.json", slot_section))

    # Add extra engine sections owned by the active family plugin.
    for ename, eplan in extra_engines.items():
        sections.append(BundleSection(ename, eplan))

    if triattention_section is not None and triattention_cfg is not None:
        sections.append(
            BundleSection(triattention_cfg.stats_section, triattention_section)
        )

    def make_runtime_config_json(source: bytes | None) -> bytes:
        cfg_dict = json.loads(source) if source is not None else dict(config.raw)
        _apply_generation_config_eos(model_dir_path, cfg_dict)
        runtime_strategy = getattr(plugin, "runtime_strategy", None)
        if runtime_strategy:
            cfg_dict["runtime_strategy"] = runtime_strategy
        elif triattention_cfg is not None:
            raise ValueError(
                "TriAttention requires the family plugin to declare a "
                "model-owned decoder runtime_strategy."
            )
        cfg_dict["engine_backend"] = "trt_rtx" if rtx else "trt"
        cfg_dict["trt_version"] = trt_version
        if trt_abi:
            cfg_dict["trt_abi"] = trt_abi
        cfg_dict["precision"] = precision
        if fp32_layers:
            cfg_dict["fp32_layers"] = sorted(set(fp32_layers))
        cfg_dict["tokenizer_add_special_tokens"] = int(
            tokenizer_add_special_tokens)
        if tokenizer_special_frame is not None:
            cfg_dict["tokenizer_special_prefix_ids"] = (
                tokenizer_special_prefix_ids)
            cfg_dict["tokenizer_special_suffix_ids"] = (
                tokenizer_special_suffix_ids)
        cfg_dict["decoder_engine_layout"] = actual_decoder_engine_layout
        if quant_plan is not None:
            cfg_dict["quantization"] = quant_plan.as_config_dict()
        elif quantize:
            cfg_dict["quantization"] = {"format": quantize}
        if triattention_cfg is not None:
            cfg_dict["triattention"] = triattention_cfg.to_dict()
        cfg_dict.update(parallel.to_bundle_config_fields())
        if enable_dynamic_kv_cache:
            cfg_dict["dynamic_kv_cache"] = True
            cfg_dict["dynamic_kv_profile_rows"] = config.raw.get(
                "_dynamic_kv_profile_rows", [max_cache_length]
            )
        embed_input = getattr(plugin, "embed_input", False)
        if embed_input:
            cfg_dict["embed_input"] = True
        if vision_plan is not None:
            cfg_dict["has_vision_engine"] = True
        # Inject VL config from plugin (image_token_id, prompt template, etc.)
        get_vl_config = getattr(plugin, 'get_vl_config', None)
        if get_vl_config is not None:
            vl_cfg = get_vl_config(config)
            if vl_cfg is not None:
                cfg_dict.update(vl_cfg)
        # Inject segmentation config from plugin
        get_seg_config = getattr(plugin, 'get_segmentation_config', None)
        if get_seg_config is not None:
            seg_cfg = get_seg_config(config)
            if seg_cfg is not None:
                cfg_dict.update(seg_cfg)
        # Inject detection config from plugin
        get_det_config = getattr(plugin, 'get_detection_config', None)
        if get_det_config is not None:
            det_cfg = get_det_config(config)
            if det_cfg is not None:
                cfg_dict.update(det_cfg)
        # Inject audio config from plugin
        get_audio_config = getattr(plugin, 'get_audio_config', None)
        if get_audio_config is not None:
            audio_cfg = get_audio_config(config)
            if audio_cfg is not None:
                cfg_dict.update(audio_cfg)
        # Inject optional LoRA/adaptor config from plugin.
        get_lora_config = getattr(plugin, 'get_lora_config', None)
        if get_lora_config is not None:
            lora_cfg = get_lora_config(config)
            if lora_cfg is not None:
                cfg_dict.update(lora_cfg)
        # Inject generic config overrides from plugin.
        # Build the final dict so overrides appear FIRST in the
        # serialized JSON.  The C++ fast_path_config parser uses
        # flat text search (text.find) which picks up the first
        # occurrence of a key.  For models with nested configs, a nested
        # copy of "hidden_size" etc. would otherwise shadow the
        # top-level value.
        get_overrides = getattr(plugin, 'get_bundle_config_overrides', None)
        if get_overrides is not None:
            overrides = get_overrides(config)
            if overrides is not None:
                # Put overrides first, then original dict.  Dict
                # union preserves insertion order; overrides keys
                # appear before any nested dicts.
                merged = dict(overrides)
                merged.update(cfg_dict)
                # Ensure overrides win for top-level keys.
                merged.update(overrides)
                cfg_dict = merged
        return json.dumps(cfg_dict, indent=2).encode("utf-8")

    # Embed tokenizer + config files. If the source model uses a family-owned
    # non-HF config adapter, synthesize config.json for the C++ runtime from
    # the parsed ModelConfig.
    embedded_config_json = False
    tokenizer_json_override = _tokenizer_json_bundle_override_from_plugin(
        plugin,
        model_dir_path,
    )
    for filename in ("config.json", "tokenizer.json", "tokenizer_config.json",
                     "chat_template.jinja", "vocab.json", "merges.txt",
                     "special_tokens_map.json", "tokenizer.model",
                     "preprocessor_config.json", "processor_config.json"):
        file_path = model_dir_path / filename
        if file_path.exists() or (
            filename == "tokenizer.json" and tokenizer_json_override is not None
        ):
            if filename == "tokenizer.json" and tokenizer_json_override is not None:
                data = tokenizer_json_override
            else:
                data = file_path.read_bytes()
            # Inject runtime_strategy and VL fields into config.json.
            if filename == "config.json":
                data = make_runtime_config_json(data)
                embedded_config_json = True
            sections.append(BundleSection(filename, data))
    if not embedded_config_json:
        sections.append(BundleSection("config.json", make_runtime_config_json(None)))

    # Package explicitly supplied FFI kernel .so files into the bundle.
    if kernel_artifacts:
        manifest_entries = []
        for global_name, so_path in kernel_artifacts:
            section_name = f"kernel_{global_name.replace('.', '_')}.so"
            so_data = Path(so_path).read_bytes()
            sections.append(BundleSection(section_name, so_data))
            manifest_entries.append({
                "global_name": global_name,
                "func_name": "run",
                "section": section_name,
            })
        manifest_json = json.dumps({"kernels": manifest_entries}).encode("utf-8")
        sections.append(BundleSection("kernel_manifest.json", manifest_json))

    write_t0 = time.monotonic()
    write_bundle(output_path, info, sections)
    _add_build_timing(build_timing, "bundle_write_s", time.monotonic() - write_t0)
    t4 = time.monotonic()
    build_timing["total_s"] = t4 - t0
    _write_build_timing(build_timing)
    print(f"[trtmc build] Bundle saved: {output_path} [{t4 - t0:.1f}s total]",
          file=sys.stderr)


def _build_diffusion_bundle(
    model_dir_path: Path,
    output_path: str,
    max_cache_length: int,
    *,
    precision: str = "fp32",
    fp32_layers: list[int] | None = None,
    verbose: bool = False,
    t0: float = 0.0,
    fp8_scales: dict | None = None,
    save_fp8_scales: str | None = None,
    rtx: bool = False,
    family_build_options: dict | None = None,
    diffusion_overrides: dict | None = None,
    build_timing: dict | None = None,
    parallel_config: ParallelConfig | None = None,
    max_batch_size: int = 1,
) -> None:
    """Build a diffusion model bundle from a diffusers-format directory."""
    if build_timing is None:
        build_timing = _new_build_timing()
    parallel = normalize_parallel_config(parallel_config)
    entrypoint = _resolve_diffusion_entrypoint(model_dir_path)
    if entrypoint is None:
        raise ValueError(
            f"No supported diffusion entrypoint found in {model_dir_path}"
        )
    pipeline_config, plugin = entrypoint
    pipeline_class = str(pipeline_config.get("_class_name", "") or "")

    print(f"[trtmc build] Diffusion pipeline: {pipeline_class}",
          file=sys.stderr)

    if plugin is None:
        supported = ", ".join(available_plugin_ids())
        raise ValueError(
            f"No family plugin for diffusion pipeline {pipeline_class!r}. "
            f"Supported: {supported}")

    model_type = getattr(plugin, 'name', pipeline_class.lower())
    if parallel.distributed:
        require_tensorrt_11_for_distributed(
            parallel, feature="Diffusion distributed builds")
    config = ModelConfig(model_type=model_type, raw=dict(pipeline_config))
    config.raw["max_cache_length"] = max_cache_length
    config.raw["_fp32_layers"] = sorted(set(fp32_layers or ()))
    config.raw["_family_build_options"] = dict(family_build_options or {})
    if diffusion_overrides:
        config.raw.update(diffusion_overrides)
    config.raw["_source_model_ref"] = getattr(
        build_bundle, "_model_id_or_path_orig", str(model_dir_path)
    )

    print(f"[trtmc build] Family: {plugin.name}", file=sys.stderr)

    # Load weights (lightweight — just paths for diffusion)
    t1 = time.monotonic()
    try:
        weights = _load_plugin_weights(
            plugin, str(model_dir_path), config, precision=precision)
    finally:
        weights_elapsed = time.monotonic() - t1
        _add_build_timing(build_timing, "weights_loading_s", weights_elapsed)
        _write_build_timing(build_timing)
    print(f"[trtmc build] Weights loaded [{weights_elapsed:.1f}s]", file=sys.stderr)

    # Propagate transformer config to ModelConfig so get_diffusion_config can access it
    if "_transformer_config" in weights:
        config.raw["_transformer_config"] = weights["_transformer_config"]

    # Prefer a family-provided scale asset before running live calibration.
    if fp8_scales == "auto":
        precomputed_fn = getattr(plugin, "fp8_precomputed_scales", None)
        if callable(precomputed_fn):
            print(
                f"[trtmc build] Resolving packaged FP8 scales for {plugin.name} ...",
                file=sys.stderr,
            )
            precomputed_scales = precomputed_fn(str(model_dir_path), config)
            if precomputed_scales is not None:
                if not isinstance(precomputed_scales, dict) or not precomputed_scales:
                    raise ValueError(
                        f"Plugin {plugin.name}.fp8_precomputed_scales() must "
                        "return a non-empty dictionary or None"
                    )
                fp8_scales = precomputed_scales
                print(
                    f"[trtmc build] Loaded {len(fp8_scales)} precomputed FP8 layer scales",
                    file=sys.stderr,
                )

    # Auto-calibrate FP8 when the family has no matching packaged asset.
    if fp8_scales == "auto":
        calibrate_fn = getattr(plugin, 'fp8_calibrate', None)
        if calibrate_fn is None:
            raise ValueError(
                f"Plugin {plugin.name} does not support FP8 auto-calibration. "
                f"Use --fp8-scales with a pre-computed scales JSON instead.")
        print(f"[trtmc build] Running FP8 auto-calibration for {plugin.name} ...",
              file=sys.stderr)
        calibrate_t0 = time.monotonic()
        try:
            fp8_scales = calibrate_fn(str(model_dir_path), config)
        finally:
            _add_build_timing(
                build_timing, "fp8_calibration_s",
                time.monotonic() - calibrate_t0)
            _write_build_timing(build_timing)
        print(f"[trtmc build] Calibrated {len(fp8_scales)} layers",
              file=sys.stderr)

    # Save FP8 scales to JSON if requested
    if save_fp8_scales and isinstance(fp8_scales, dict):
        save_scales_t0 = time.monotonic()
        with open(save_fp8_scales, "w") as _sf:
            json.dump(fp8_scales, _sf, indent=2)
        _add_build_timing(
            build_timing, "fp8_scales_write_s",
            time.monotonic() - save_scales_t0)
        _write_build_timing(build_timing)
        print(f"[trtmc build] Saved FP8 scales to {save_fp8_scales} "
              f"({len(fp8_scales)} layers)", file=sys.stderr)

    # Build all component engines
    build_components = getattr(plugin, 'build_components', None)
    if build_components is None:
        raise ValueError(
            f"Plugin {plugin.name} does not support build_components()")

    components_t0 = time.monotonic()
    weights_before_components = _build_timing_phase(
        build_timing, "weights_loading_s")
    compile_before_components = _build_timing_phase(build_timing, "trt_compile_s")
    try:
        build_components_kwargs = {
            "verbose": verbose,
            "fp8_scales": fp8_scales,
        }
        if _call_supports_kwarg(build_components, "precision"):
            build_components_kwargs["precision"] = precision
        if _call_supports_kwarg(build_components, "build_timing"):
            build_components_kwargs["build_timing"] = build_timing
        if _call_supports_kwarg(build_components, "parallel_config"):
            build_components_kwargs["parallel_config"] = parallel
        elif parallel.distributed:
            raise NotImplementedError(
                f"Plugin {plugin.name} does not accept parallel_config for "
                f"diffusion {parallel.mode}")
        # Only forward max_batch_size to plugins that opted in. Plugins that
        # don't accept it (older or non-batchified ones) silently stay on B=1.
        if _call_supports_kwarg(build_components, "max_batch_size"):
            build_components_kwargs["max_batch_size"] = max_batch_size
        components = build_components(
            str(model_dir_path), config, weights, **build_components_kwargs)
    finally:
        components_elapsed = time.monotonic() - components_t0
        compile_elapsed = _compile_time_excluding_component_weight_load(
            components_elapsed, weights_before_components, build_timing)
        untracked_compile_elapsed = _untracked_compile_time(
            compile_elapsed, compile_before_components, build_timing)
        _add_build_timing(
            build_timing, "trt_compile_s", untracked_compile_elapsed)
        _add_build_timing(
            build_timing, "trt_compile_diffusion_components_s",
            compile_elapsed)
        _write_build_timing(build_timing)
    if components is None:
        raise ValueError(
            f"Plugin {plugin.name}.build_components() returned None")

    print(f"[trtmc build] All engines built [{components_elapsed:.1f}s]",
          file=sys.stderr)

    # Assemble model-owned bundle sections.
    sections = _diffusion_bundle_sections_from_plugin(plugin, components, parallel)
    for section in sections:
        print(
            f"  {section.name}: {len(section.data) / (1024 * 1024):.1f} MB",
            file=sys.stderr,
        )

    trt_version = _get_trt_version()
    trt_abi = _trt_abi_from_version(trt_version)
    tokenizer_t0 = time.monotonic()
    tokenizer_special_frame = _diffusion_tokenizer_special_frame_from_plugin(
        plugin, model_dir_path)
    if tokenizer_special_frame is None:
        tokenizer_special_prefix_ids: list[int] = []
        tokenizer_special_suffix_ids: list[int] = []
        tokenizer_add_special_tokens = _diffusion_tokenizer_add_special_tokens_from_plugin(
            plugin, model_dir_path)
    else:
        tokenizer_special_prefix_ids, tokenizer_special_suffix_ids = tokenizer_special_frame
        tokenizer_add_special_tokens = bool(
            tokenizer_special_prefix_ids or tokenizer_special_suffix_ids)
    _add_build_timing(
        build_timing, "tokenizer_special_tokens_detection_s",
        time.monotonic() - tokenizer_t0)
    _write_build_timing(build_timing)

    # Build config.json. Plugins that need a variant-specific schema can
    # return a pre-rendered JSON blob via components["config_json"]; other
    # diffusion plugins fall through to the generic construction below.
    if "config_json" in components:
        cfg_data = components["config_json"]
        if not isinstance(cfg_data, (bytes, bytearray)):
            raise TypeError(
                f"Plugin {plugin.name} returned components['config_json'] "
                f"as {type(cfg_data).__name__}; expected bytes."
            )
    else:
        _effective_precision = "bf16" if fp8_scales else precision
        cfg_dict = {
            "model_type": model_type,
            "runtime_strategy": getattr(plugin, "runtime_strategy", "diffusion"),
            "precision": _effective_precision,
            "engine_backend": "trt_rtx" if rtx else "trt",
            "trt_version": trt_version,
            "tokenizer_add_special_tokens": int(tokenizer_add_special_tokens),
        }
        if tokenizer_special_frame is not None:
            cfg_dict["tokenizer_special_prefix_ids"] = tokenizer_special_prefix_ids
            cfg_dict["tokenizer_special_suffix_ids"] = tokenizer_special_suffix_ids
        if trt_abi:
            cfg_dict["trt_abi"] = trt_abi
        if fp8_scales:
            cfg_dict["quantization"] = {"format": "fp8"}

        # Inject diffusion config from plugin.
        get_bundle_config = getattr(plugin, "diffusion_bundle_config", None)
        if not callable(get_bundle_config):
            raise ValueError(
                f"Plugin {plugin.name} must implement diffusion_bundle_config() "
                "so component-derived config stays model-owned"
            )
        diff_cfg = get_bundle_config(config, components=components)
        if diff_cfg is not None:
            cfg_dict.update(diff_cfg)
        cfg_dict.update(parallel.to_bundle_config_fields())

        cfg_data = json.dumps(cfg_dict, indent=2).encode("utf-8")

    sections.append(BundleSection("config.json", cfg_data))

    tokenizer_json_t0 = time.monotonic()
    sections.extend(_diffusion_tokenizer_bundle_sections_from_plugin(
        plugin, model_dir_path))
    _add_build_timing(
        build_timing, "tokenizer_json_ensure_s",
        time.monotonic() - tokenizer_json_t0)
    _write_build_timing(build_timing)

    # Resolve per-component batch envelope. Plugins that batchified record
    # the envelope on components["max_batch_size_envelope"]. When absent
    # (older plugins, or N=1), the BundleInfo field stays None and the
    # JSON header simply omits the block — back-compat with PR 1 readers.
    mbs_envelope = components.get("max_batch_size_envelope")
    if mbs_envelope is None and max_batch_size > 1:
        # Plugin didn't expose an envelope but caller asked for >1 — fall
        # back to a sane default derived from CLI policy (see Decision C).
        mbs_envelope = {
            "dit": int(max_batch_size),
            "text_encoder": min(int(max_batch_size) * 2, 8),
            "vae": 1,
        }

    # Write bundle
    info = BundleInfo(
        model_id=model_dir_path.name,
        model_type=model_type,
        family=plugin.name,
        trt_version=trt_version,
        trt_abi=trt_abi,
        gpu_name=_get_gpu_name(),
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        runtime_strategy=getattr(plugin, "runtime_strategy", "diffusion"),
        precision=precision,
        quantization="fp8" if fp8_scales else "none",
        max_cache_length=max_cache_length,
        tokenizer_add_special_tokens=tokenizer_add_special_tokens,
        max_batch_size=mbs_envelope,
    )

    write_t0 = time.monotonic()
    write_bundle(output_path, info, sections)
    _add_build_timing(build_timing, "bundle_write_s", time.monotonic() - write_t0)
    t4 = time.monotonic()
    build_timing["total_s"] = t4 - t0
    _write_build_timing(build_timing)
    print(f"[trtmc build] Bundle saved: {output_path} [{t4 - t0:.1f}s total]",
          file=sys.stderr)


def _build_native_impl(
    model_id_or_path: str,
    output_path: str,
    max_cache_length: int | None = _OMITTED_MAX_CACHE_LENGTH,
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
    """Build a .bundle artifact from a HuggingFace model ID or local path.

    Like HF transformers, accepts either:
    - A HuggingFace repo ID such as ``"org/model-name"`` (auto-downloads)
    - A local directory such as ``"models/hf/org__model-name"``

    Args:
        model_id_or_path: HF repo ID or local directory with config.json + safetensors.
        output_path: Where to write the .bundle artifact.
        model_revision: Optional Hugging Face revision to resolve for remote model IDs.
        max_cache_length: Explicit KV cache length for the engine. ``None``
            lets the selected family resolve its model-owned default.
        decoder_engine_layout: ``"split"`` or ``"dual_profile"``.
        verbose: Print detailed TRT builder logs.
        fp8_scales: Per-layer FP8 scales dict, or ``"auto"`` for auto-calibration.
        save_fp8_scales: Path to save calibrated FP8 scales JSON.
    """
    if max_cache_length is _OMITTED_MAX_CACHE_LENGTH:
        max_cache_length = None
    revision_kwargs = {"revision": model_revision} if model_revision else {}
    model_dir = _resolve_model(model_id_or_path, **revision_kwargs)
    build_bundle._model_id_or_path_orig = model_id_or_path
    build_bundle._fp8_scales = fp8_scales
    build_bundle._save_fp8_scales = save_fp8_scales
    build_bundle(model_dir, output_path, max_cache_length,
                 decoder_engine_layout=decoder_engine_layout,
                 dynamic_kv_cache=dynamic_kv_cache,
                 dynamic_kv_profile_rows_override=dynamic_kv_profile_rows_override,
                 precision=precision,
                 fp32_layers=fp32_layers,
                 quantize=quantize,
                 quant_scales=quant_scales,
                 quant_calibration_samples=quant_calibration_samples,
                 verbose=verbose,
                 rtx=rtx,
                 triattention_stats_path=triattention_stats_path,
                 triattention_kv_budget=triattention_kv_budget,
                 triattention_divide_length=triattention_divide_length,
                 triattention_recent_window=triattention_recent_window,
                 triattention_score_aggregation=triattention_score_aggregation,
                 triattention_count_prompt_tokens=triattention_count_prompt_tokens,
                 triattention_protect_prefill=triattention_protect_prefill,
                 triattention_disable_mlr=triattention_disable_mlr,
                 triattention_disable_trig=triattention_disable_trig,
                 family_build_options=family_build_options,
                 parallel_config=parallel_config,
                 diffusion_overrides=diffusion_overrides,
                 build_timing_path=build_timing_path,
                 max_batch_size=max_batch_size,
                 tokenizer_source_model_id_or_path=model_id_or_path,
                 tokenizer_source_revision=model_revision)


def _optimized_request_value(value):
    """Normalize one existing build option for a capsule-owned adapter.

    The shared router deliberately does not interpret option names or values.
    It only converts the established Python API's value types into the JSON
    data accepted by the isolated build-adapter protocol.
    """

    if value is None or isinstance(value, (bool, str, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, ParallelConfig):
        return value.to_config_dict()
    if isinstance(value, dict):
        return {
            str(key): _optimized_request_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_optimized_request_value(item) for item in value]
    raise TypeError(
        "Build option cannot be represented for an optimized-runtime adapter: "
        f"{type(value).__name__}"
    )


def _try_build_optimized_runtime(
    model_id_or_path: str,
    output_path: str | Path,
    public_options: dict,
    *,
    model_revision: str | None = None,
):
    """Try a model-family-owned integration for the current platform.

    The model is resolved before dispatch so discovery is bounded to its one
    owning family. Every interpretation of ``public_options`` remains inside
    the selected model adapter.
    """

    from .runtime_provider.orchestrator import try_build_optimized_runtime

    resolved_model_ref = model_id_or_path
    model_config: ModelConfig | None = None
    try:
        revision_kwargs = {"revision": model_revision} if model_revision else {}
        resolved_model_ref = _resolve_model(model_id_or_path, **revision_kwargs)
        model_dir = Path(resolved_model_ref)
        if (model_dir / "model_index.json").exists():
            model_index = json.loads((model_dir / "model_index.json").read_text())
            selected_family = str(
                _resolve_diffusion_family_id(
                    str(model_index.get("_class_name", "") or "")
                )
                or ""
            )
        else:
            model_config = ModelConfig.from_dir(model_dir)
            selected_family = str(resolve_family_id(model_config) or "")
    except Exception:
        # Optimized dispatch is optional. Preserve the native path's exact
        # model-resolution behavior and diagnostics when family discovery
        # cannot resolve the request in this environment.
        return None
    if not selected_family:
        return None
    if (
        model_config is not None
        and family_prefers_native_default_build(model_config)
    ):
        return None

    return try_build_optimized_runtime(
        resolved_model_ref,
        output_path,
        family_name=selected_family,
        parameters={
            "public_options": {
                key: _optimized_request_value(value)
                for key, value in sorted(public_options.items())
            }
        },
    )


def build(
    model_id_or_path: str,
    output_path: str,
    max_cache_length: int | None = _OMITTED_MAX_CACHE_LENGTH,
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
    """Build through a matching model capsule, otherwise use the native path.

    Optimized-runtime selection uses the active platform and forwards the
    normalized effective public options as opaque data; the model-owned adapter
    owns all translation. Native families may select their own precision when
    the caller leaves ``precision`` unset.
    """

    build_arguments = dict(locals())
    if build_arguments["max_cache_length"] is _OMITTED_MAX_CACHE_LENGTH:
        build_arguments["max_cache_length"] = None
    public_options = {
        name: value
        for name, value in build_arguments.items()
        if name not in {"model_id_or_path", "model_revision", "output_path"}
    }
    # Preserve the established optimized-runtime default while allowing the
    # native builder to resolve a model-owned precision.
    if public_options["precision"] is None:
        public_options["precision"] = "fp32"
    if public_options["max_cache_length"] is None:
        public_options["max_cache_length"] = 256
    revision_kwargs = (
        {"model_revision": model_revision} if model_revision else {}
    )
    optimized = _try_build_optimized_runtime(
        model_id_or_path,
        output_path,
        public_options,
        **revision_kwargs,
    )
    if optimized is not None:
        return
    _build_native_impl(**build_arguments)
