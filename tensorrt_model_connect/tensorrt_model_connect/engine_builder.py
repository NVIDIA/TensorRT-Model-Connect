"""Orchestrator: load model → build engine → write bundle."""

from __future__ import annotations

import inspect
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .build_timing import (
    add_build_timing as _add_build_timing,
    new_build_timing as _new_build_timing,
    write_build_timing as _write_build_timing,
)
from .config import ModelConfig
from .families import find_plugin, find_diffusion_plugin, _ALL_PLUGINS
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
    require_tensorrt_11_for_tensor_parallel,
)


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


# Standard HF file patterns to download (matches what the builder needs).
_HF_ALLOW_PATTERNS = [
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "model.safetensors",
    "model-*.safetensors",
    "model.safetensors-*.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "*.yml",
    "*.yaml",
    "checkpoint_*",
    "checkpoint_*/**",
    "model.npz",
    "elf_params.npz",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "*.model",
    "*.spm",
    "*.py",
    # Diffusers format
    "model_index.json",
    "scheduler/**",
    "text_encoder/**",
    "text_encoder_2/**",
    "transformer/**",
    "vae/**",
    "tokenizer/**",
    "tokenizer_2/**",
    "*/config.json",
    "*/model.safetensors",
    "*/model-*.safetensors",
    "*/model.safetensors.index.json",
    "*/diffusion_pytorch_model.safetensors",
    "*/diffusion_pytorch_model-*.safetensors",
    "*/diffusion_pytorch_model.safetensors.index.json",
    "scheduler/*",
    "tokenizer/*",
]


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


def _plugin_uses_standard_decoder_builder(plugin) -> bool:
    """Best-effort check for family plugins routed through the standard decoder."""
    try:
        source = inspect.getsource(plugin.build_engine)
    except (OSError, TypeError):
        return False
    return "build_standard_decoder_engine" in source


def _plugin_supports_split_decoder_roles(plugin) -> bool:
    """Return True when the family's standard builder honors split roles."""
    if not _plugin_uses_standard_decoder_builder(plugin):
        return False
    build_globals = getattr(plugin.build_engine, "__globals__", {})
    standard_builder = build_globals.get("build_standard_decoder_engine")
    if standard_builder is None:
        return False
    try:
        source = inspect.getsource(standard_builder)
    except (OSError, TypeError):
        return False
    return (
        "_decoder_engine_role" in source
        and "profile_mode" in source
    )


def _can_build_split_decoder_engines(
    plugin,
    runtime_strategy: str,
    *,
    dynamic_kv_cache: bool,
    triattention_enabled: bool,
) -> bool:
    """Return True when split prefill/decode engines are supported.

    The split layout relies on ``standard_decoder_builder`` honoring the
    internal ``_decoder_engine_role`` passthrough. Custom MoE, recurrent,
    VL/embed-input, TriAttention, and dynamic-KV runtimes keep their existing
    single-engine behavior until they opt into the same contract.
    """
    if runtime_strategy not in ("decoder_kv_cache", "decoder_moe"):
        return False
    if dynamic_kv_cache or triattention_enabled:
        return False
    if bool(getattr(plugin, "embed_input", False)):
        return False
    return _plugin_supports_split_decoder_roles(plugin)


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


def _is_elf_model_dir(path: Path) -> bool:
    """Return True for the official ELF YAML + checkpoint directory layout."""
    if not path.is_dir():
        return False
    has_checkpoint = any(path.glob("checkpoint_*")) or any(
        (path / name).exists() for name in ("model.npz", "elf_params.npz")
    )
    if not has_checkpoint:
        return False
    for yaml_path in [*path.glob("*.yaml"), *path.glob("*.yml")]:
        try:
            import yaml  # type: ignore[import-untyped]

            data = yaml.safe_load(yaml_path.read_text()) or {}
        except Exception:
            continue
        if isinstance(data, dict) and str(data.get("model", "")).upper().replace("_", "-") in {
            "ELF-B",
            "ELF-M",
            "ELF-L",
        }:
            return True
    return False


def _resolve_model(model_id_or_path: str) -> str:
    """Resolve a HuggingFace repo ID or local path to a local directory.

    If model_id_or_path is an existing directory with config.json, returns it
    directly. Otherwise, downloads via huggingface_hub.snapshot_download().
    Handles .nemo archives by extracting config and creating a synthetic dir.
    """
    local = Path(model_id_or_path)
    if local.is_dir() and (_is_hf_model_dir(local) or _is_elf_model_dir(local)):
        return str(local)

    # Handle .nemo archives (NeMo models like MagpieTTS)
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
            allow_patterns=_HF_ALLOW_PATTERNS + ["*.nemo"],
        )
    except Exception as exc:
        _raise_friendly_download_error(model_id_or_path, exc)

    # Prefer HF config when both HF files and .nemo are present.
    dl_path = Path(local_dir)
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
    """Extract a .nemo archive and create a synthetic HF-compatible directory.

    NeMo .nemo files are tar archives containing model_config.yaml and
    model_weights.ckpt. We extract the YAML config, generate a synthetic
    config.json with model_type for plugin dispatch, and store the .nemo
    path for the plugin's load_weights() to use.
    """
    import json
    import tempfile

    print(f"[trtmc build] Resolving NeMo archive: {nemo_path}", file=sys.stderr)

    # Extract model_config.yaml from the tar
    import tarfile
    cfg = {}
    with tarfile.open(str(nemo_path), "r") as tar:
        for member in tar.getmembers():
            if member.name.endswith("model_config.yaml"):
                import yaml
                f = tar.extractfile(member)
                if f is not None:
                    cfg = yaml.safe_load(f)
                break

    # Determine model_type from NeMo config
    target = cfg.get("target", "") or cfg.get("_target_", "")
    model_type = "unknown"
    if "MagpieTTS" in target or "magpietts" in target.lower():
        model_type = "magpie_tts"
    elif ("EncDecRNNT" in target or "Transducer" in target
          or "rnnt" in target.lower()):
        model_type = "nemotron_speech_streaming"
    elif "EncDecMultiTaskModel" in target or "canary" in target.lower():
        model_type = "canary"
    elif cfg.get("model_type", ""):
        model_type = cfg["model_type"]

    # Create a temp dir that looks like an HF model dir
    tmp_dir = tempfile.mkdtemp(prefix="trtmc_nemo_")
    tmp_path = Path(tmp_dir)

    # Write synthetic config.json for ModelConfig.from_dir()
    enc_cfg = cfg.get("encoder", {})
    dec_cfg = cfg.get("decoder", cfg.get("transf_decoder", {}))
    hidden = enc_cfg.get("d_model", 768)
    # Decoder fields vary by NeMo model type
    dec_layers = dec_cfg.get("n_layers",
                             dec_cfg.get("num_layers", 12))
    dec_heads = dec_cfg.get("sa_n_heads",
                            dec_cfg.get("num_attention_heads", 12))
    dec_ffn = dec_cfg.get("d_ffn",
                          dec_cfg.get("inner_size", 3072))
    synthetic_config = {
        "model_type": model_type,
        "hidden_size": hidden,
        "num_hidden_layers": dec_layers,
        "num_attention_heads": dec_heads,
        "intermediate_size": dec_ffn,
        "vocab_size": 2380,  # Will be overridden from weights
        "rms_norm_eps": 1e-5,
        "_nemo_archive_path": str(nemo_path),
    }
    with open(tmp_path / "config.json", "w") as f:
        json.dump(synthetic_config, f, indent=2)

    # Symlink the .nemo file into the temp dir for easy access
    nemo_link = tmp_path / nemo_path.name
    if not nemo_link.exists():
        import os
        os.symlink(str(nemo_path.resolve()), str(nemo_link))

    print(f"[trtmc build] NeMo resolved: model_type={model_type}, "
          f"tmp_dir={tmp_dir}", file=sys.stderr)
    return tmp_dir


def _get_trt_version() -> str:
    return trt_compat.tensorrt_version() or "unknown"


def _trt_abi_from_version(version: str) -> str:
    match = re.search(r"(\d+)\.(\d+)", version or "")
    if not match:
        return ""
    return f"{match.group(1)}.{match.group(2)}"


def _get_gpu_name() -> str:
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.strip().split("\n")[0]
    except Exception:
        pass
    return ""


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


def _detect_diffusion_tokenizer_add_special_tokens(model_dir: Path) -> bool:
    """Detect add-special behavior from the tokenizer embedded in diffusion bundles."""
    for tok_subdir in ("tokenizer_2", "tokenizer"):
        tok_dir = model_dir / tok_subdir
        if tok_dir.is_dir():
            return _detect_tokenizer_add_special_tokens(tok_dir)
    return _detect_tokenizer_add_special_tokens(model_dir)


def _ensure_tokenizer_json(model_dir: Path) -> None:
    """If the model directory lacks tokenizer.json, generate it from the
    slow tokenizer using HF transformers. This ensures the C++ runtime can
    always load the tokenizer natively (BPE / WordPiece / Unigram).

    Fallback chain:
      1. AutoTokenizer(use_fast=False).save_pretrained() — works for most models
      2. SentencePiece .spm → tokenizers.Unigram conversion — for Marian / NLLB
    """
    if (model_dir / "tokenizer.json").exists():
        return

    # --- Attempt 1: standard HF slow → fast conversion ---
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(model_dir), use_fast=False)
        tok.save_pretrained(str(model_dir))
        if (model_dir / "tokenizer.json").exists():
            print("[trtmc build] Generated tokenizer.json from slow tokenizer",
                  file=sys.stderr)
            return
    except Exception:
        pass

    # --- Attempt 2: build from SentencePiece .spm + vocab.json ---
    # Marian/NLLB models have source.spm (encoder-side SentencePiece) and
    # vocab.json (combined source+target vocabulary with IDs).  We build a
    # Unigram tokenizer.json using the full combined vocab (so IDs match the
    # TRT engine) with scores from the SPM model for source tokens and a
    # default low score for target-only tokens.
    spm_candidates = list(model_dir.glob("*.spm"))
    source_spm = model_dir / "source.spm"
    spm_path = source_spm if source_spm.exists() else (spm_candidates[0] if spm_candidates else None)
    vocab_json_path = model_dir / "vocab.json"
    if spm_path is not None:
        try:
            import sentencepiece as spm_lib
            from tokenizers import Tokenizer, normalizers, pre_tokenizers, decoders
            from tokenizers.models import Unigram

            sp = spm_lib.SentencePieceProcessor()
            sp.Load(str(spm_path))
            # Build score lookup from SPM model
            spm_scores = {}
            for i in range(sp.GetPieceSize()):
                spm_scores[sp.IdToPiece(i)] = sp.GetScore(i)
            min_score = min(spm_scores.values()) if spm_scores else 0.0
            default_score = min_score - 10.0  # worse than any real token

            # Build combined vocab with correct IDs from vocab.json
            if vocab_json_path.exists():
                with open(vocab_json_path) as f:
                    combined_vocab = json.load(f)
                # combined_vocab is {token_str: id_int}, build id-ordered list
                max_id = max(combined_vocab.values())
                vocab = [("", default_score)] * (max_id + 1)
                for token, tid in combined_vocab.items():
                    score = spm_scores.get(token, default_score)
                    vocab[tid] = (token, score)
            else:
                # Fallback: use SPM vocab only
                vocab = [(sp.IdToPiece(i), sp.GetScore(i)) for i in range(sp.GetPieceSize())]

            unk_id = combined_vocab.get("<unk>", 0) if vocab_json_path.exists() else 0

            tokenizer = Tokenizer(Unigram(vocab, unk_id))
            tokenizer.normalizer = normalizers.Sequence([
                normalizers.Prepend(prepend="\u2581"),
                normalizers.Replace(" ", "\u2581"),
            ])
            tokenizer.pre_tokenizer = pre_tokenizers.Sequence([])
            tokenizer.decoder = decoders.Metaspace()

            out_path = str(model_dir / "tokenizer.json")
            tokenizer.save(out_path)
            print(f"[trtmc build] Generated tokenizer.json from {spm_path.name} "
                  f"({len(vocab)} tokens)", file=sys.stderr)
            return
        except Exception as e:
            print(f"[trtmc build] Warning: SentencePiece conversion failed: {e}",
                  file=sys.stderr)

    print("[trtmc build] Warning: could not generate tokenizer.json "
          "(C++ runtime may fail to create tokenizer)", file=sys.stderr)


def build_bundle(
    model_dir: str,
    output_path: str,
    max_cache_length: int = 256,
    *,
    decoder_engine_layout: str = "split",
    dynamic_kv_cache: bool = False,
    dynamic_kv_profile_rows_override: list[int] | None = None,
    precision: str = "fp32",
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
    # audio_magpie.* build-time fields. max_source_positions replaces
    # the TRTMC_MAGPIE_MAX_SOURCE_POS env var; passed to families via
    # config.raw, same passthrough pattern.
    audio_magpie_max_source_positions: int = 0,
    parallel_config: ParallelConfig | None = None,
    diffusion_overrides: dict | None = None,
    build_timing_path: str | None = None,
) -> None:
    """Full pipeline: load HF model → build TRT engine → write .trtfb bundle.

    Args:
        model_dir: Path to HF model directory with config.json + safetensors.
        output_path: Where to write the .trtfb bundle.
        max_cache_length: KV cache length for the engine.
        decoder_engine_layout: ``"split"`` builds separate prefill/decode
            engines for supported decoder LLMs. ``"dual_profile"`` keeps the
            low-VRAM single-engine/multi-profile layout.
        verbose: Print detailed logs.
    """
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

    # Detect diffusers format (model_index.json present)
    is_diffusers = (model_dir_path / "model_index.json").exists()

    if is_diffusers:
        fp8_scales = getattr(build_bundle, '_fp8_scales', None)
        save_fp8_scales = getattr(build_bundle, '_save_fp8_scales', None)
        _build_diffusion_bundle(
            model_dir_path, output_path, max_cache_length,
            precision=precision, verbose=verbose, t0=t0,
            fp8_scales=fp8_scales, save_fp8_scales=save_fp8_scales,
            rtx=rtx,
            diffusion_overrides=diffusion_overrides,
            build_timing=build_timing,
            parallel_config=parallel)
        return

    # 1. Parse config
    config = ModelConfig.from_dir(model_dir_path)
    config.raw["_decoder_engine_layout"] = decoder_engine_layout
    config.raw["_audio_magpie_max_source_positions"] = audio_magpie_max_source_positions
    print(f"[trtmc build] Model: {config.model_type} "
          f"(layers={config.num_hidden_layers}, hidden={config.hidden_size}, "
          f"vocab={config.vocab_size})", file=sys.stderr)

    # 2. Find family plugin
    plugin = find_plugin(config.model_type)
    if plugin is None:
        supported = ", ".join(p.name for p in _ALL_PLUGINS)
        raise ValueError(
            f"No family plugin for model_type={config.model_type!r}. "
            f"Supported: {supported}")

    print(f"[trtmc build] Family: {plugin.name}", file=sys.stderr)

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
    runtime_strategy = getattr(plugin, "runtime_strategy", "") or "decoder_kv_cache"
    enable_dynamic_kv_cache = bool(dynamic_kv_cache)
    dynamic_kv_profile_rows = _sanitize_dynamic_kv_profile_rows(
        dynamic_kv_profile_rows_override,
        max_cache_length,
    )
    if triattention_stats_path:
        if runtime_strategy not in ("decoder_kv_cache", "decoder_moe"):
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
        if runtime_strategy not in ("decoder_kv_cache", "decoder_moe"):
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
        if quant_ctx is not None:
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
        previous_role = config.raw.get("_decoder_engine_role")
        config.raw["_decoder_engine_role"] = role
        try:
            return plugin.build_engine(
                config, weights, max_cache_length, verbose=verbose,
                **extra_kwargs)
        finally:
            if previous_role is None:
                config.raw.pop("_decoder_engine_role", None)
            else:
                config.raw["_decoder_engine_role"] = previous_role

    def _build_split_plugin_engine_with_role(role: str) -> bytes:
        with trt_compat.scoped_timing_cache(_split_timing_cache_scope(role)):
            return _build_plugin_engine_with_role(role)

    split_supported = (
        not parallel.enabled and
        decoder_engine_layout == "split" and
        _can_build_split_decoder_engines(
            plugin,
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
            if decoder_engine_layout == "split" and runtime_strategy in (
                "decoder_kv_cache", "decoder_moe"
            ):
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
            vision_plan = build_vision(
                str(model_dir_path), config, weights, verbose=verbose)
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

    # 4c. Build extra engines (optional, multi-engine models like Bark)
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
            if _call_supports_kwarg(build_extra, "build_timing"):
                build_extra_kwargs["build_timing"] = build_timing
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

    # 5. Detect tokenizer special-tokens behavior from HF config
    tokenizer_t0 = time.monotonic()
    tokenizer_add_special_tokens = _detect_tokenizer_add_special_tokens(
        model_dir_path)
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

    # Add extra engine sections (coarse, fine, codec for Bark, etc.)
    for ename, eplan in extra_engines.items():
        sections.append(BundleSection(ename, eplan))

    if triattention_section is not None and triattention_cfg is not None:
        sections.append(
            BundleSection(triattention_cfg.stats_section, triattention_section)
        )

    # If model lacks tokenizer.json (fast format), generate it from the
    # slow tokenizer so the C++ runtime can always load via AutoTokenizer.
    # Skip for non-text models (segmentation, audio) that don't use tokenizers.
    runtime_strategy = getattr(plugin, "runtime_strategy", "")
    if runtime_strategy not in (
        "segmentation",
        "neural_operator",
        "object_detection",
        "prompted_segmentation",
        "image_classification",
    ):
        tokenizer_json_t0 = time.monotonic()
        _ensure_tokenizer_json(model_dir_path)
        _add_build_timing(
            build_timing, "tokenizer_json_ensure_s",
            time.monotonic() - tokenizer_json_t0)
        _write_build_timing(build_timing)

    # Inject encoder_only config overrides
    if runtime_strategy == "encoder_only":
        # Use max_cache_length as max_seq_length for encoder
        pass

    def make_runtime_config_json(source: bytes | None) -> bytes:
        cfg_dict = json.loads(source) if source is not None else dict(config.raw)
        runtime_strategy = getattr(plugin, "runtime_strategy", None)
        if runtime_strategy:
            cfg_dict["runtime_strategy"] = runtime_strategy
        elif triattention_cfg is not None:
            cfg_dict["runtime_strategy"] = "decoder_kv_cache"
        cfg_dict["engine_backend"] = "trt_rtx" if rtx else "trt"
        cfg_dict["trt_version"] = trt_version
        if trt_abi:
            cfg_dict["trt_abi"] = trt_abi
        cfg_dict["precision"] = precision
        cfg_dict["tokenizer_add_special_tokens"] = int(
            tokenizer_add_special_tokens)
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
        # Inject generic config overrides from plugin.
        # Build the final dict so overrides appear FIRST in the
        # serialized JSON.  The C++ fast_path_config parser uses
        # flat text search (text.find) which picks up the first
        # occurrence of a key.  For models with nested configs
        # (e.g. Qwen3-Omni thinker_config.text_config) the nested
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

    # Embed tokenizer + config files. If the source model is a GitHub ELF
    # directory with only train_*.yml, synthesize config.json for the C++
    # runtime from the parsed ModelConfig.
    embedded_config_json = False
    for filename in ("config.json", "tokenizer.json", "tokenizer_config.json",
                     "vocab.json", "merges.txt", "special_tokens_map.json",
                     "tokenizer.model", "preprocessor_config.json"):
        file_path = model_dir_path / filename
        if file_path.exists():
            data = file_path.read_bytes()
            # Inject runtime_strategy and VL fields into config.json.
            if filename == "config.json":
                data = make_runtime_config_json(data)
                embedded_config_json = True
            sections.append(BundleSection(filename, data))
    if not embedded_config_json:
        sections.append(BundleSection("config.json", make_runtime_config_json(None)))

    # Package FFI kernel .so files into the bundle
    if kernel_artifacts:
        import json as _json
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
        manifest_json = _json.dumps({"kernels": manifest_entries}).encode("utf-8")
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
    verbose: bool = False,
    t0: float = 0.0,
    fp8_scales: dict | None = None,
    save_fp8_scales: str | None = None,
    rtx: bool = False,
    diffusion_overrides: dict | None = None,
    build_timing: dict | None = None,
    parallel_config: ParallelConfig | None = None,
) -> None:
    """Build a diffusion model bundle from a diffusers-format directory."""
    if build_timing is None:
        build_timing = _new_build_timing()
    parallel = normalize_parallel_config(parallel_config)
    if parallel.enabled:
        raise NotImplementedError(
            "Tensor-parallel diffusion builds are not implemented yet; "
            "decoder TP is the active first target.")
    # Parse model_index.json to determine pipeline type
    model_index = json.loads(
        (model_dir_path / "model_index.json").read_text())
    pipeline_class = model_index.get("_class_name", "")

    print(f"[trtmc build] Diffusion pipeline: {pipeline_class}",
          file=sys.stderr)

    # Auto-discover plugin from pipeline_classes attribute
    plugin = find_diffusion_plugin(pipeline_class)
    if plugin is None:
        # Fallback: try model_type-based lookup with lowercased pipeline class
        plugin = find_plugin(pipeline_class.lower())
    if plugin is None:
        supported = ", ".join(p.name for p in _ALL_PLUGINS)
        raise ValueError(
            f"No family plugin for diffusion pipeline {pipeline_class!r}. "
            f"Supported: {supported}")

    model_type = getattr(plugin, 'name', pipeline_class.lower())
    config = ModelConfig(model_type=model_type, raw=model_index)
    config.raw["max_cache_length"] = max_cache_length
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

    # Auto-calibrate FP8 if requested
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

    # Assemble bundle sections
    sections = []

    # Text encoder plans
    text_encoders = components.get("text_encoders", [])
    for i, (enc_name, enc_plan) in enumerate(text_encoders):
        sections.append(BundleSection(f"text_encoder_{i}_plan", enc_plan))
        print(
            f"  text_encoder_{i} ({enc_name}): "
            f"{len(enc_plan) / (1024 * 1024):.1f} MB",
            file=sys.stderr,
        )

    # Denoiser plan
    denoiser_plan = components["denoiser"]
    sections.append(BundleSection("denoiser_plan", denoiser_plan))
    print(f"  denoiser: {len(denoiser_plan) / (1024 * 1024):.1f} MB",
          file=sys.stderr)

    # VAE decoder plan
    vae_plan = components["vae_decoder"]
    sections.append(BundleSection("vae_decoder_plan", vae_plan))
    print(f"  vae_decoder: {len(vae_plan) / (1024 * 1024):.1f} MB",
          file=sys.stderr)

    # Preprocessor weights (patch embedding, timestep MLP, text projection)
    if "preprocessor_weights" in components:
        pp_data = components["preprocessor_weights"]
        sections.append(BundleSection("preprocessor_weights", pp_data))
        print(f"  preprocessor_weights: {len(pp_data) / (1024):.1f} KB",
              file=sys.stderr)

    trt_version = _get_trt_version()
    trt_abi = _trt_abi_from_version(trt_version)
    tokenizer_t0 = time.monotonic()
    tokenizer_add_special_tokens = _detect_diffusion_tokenizer_add_special_tokens(
        model_dir_path)
    _add_build_timing(
        build_timing, "tokenizer_special_tokens_detection_s",
        time.monotonic() - tokenizer_t0)
    _write_build_timing(build_timing)

    # Build config.json. Plugins that need a variant-specific schema can
    # return a pre-rendered JSON blob via components["config_json"] (e.g.
    # Qwen-Image, which has its own bundle schema built by
    # qwen_image_bundle_config.build_bundle_config()). Existing plugins
    # (Z-Image / FLUX / Wan / PixArt) don't return config_json and fall
    # through to the inline construction below.
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
            "num_text_encoders": len(components["text_encoders"]),
            "tokenizer_add_special_tokens": int(tokenizer_add_special_tokens),
        }
        if trt_abi:
            cfg_dict["trt_abi"] = trt_abi
        if fp8_scales:
            cfg_dict["quantization"] = {"format": "fp8"}

        # Inject diffusion config from plugin
        get_diff_config = getattr(plugin, 'get_diffusion_config', None)
        if get_diff_config is not None:
            diff_cfg = get_diff_config(config)
            if diff_cfg is not None:
                cfg_dict.update(diff_cfg)

        cfg_data = json.dumps(cfg_dict, indent=2).encode("utf-8")

    sections.append(BundleSection("config.json", cfg_data))

    # Ensure tokenizer.json exists for diffusion tokenizer directories.
    # SentencePiece-only tokenizers (T5, PixArt) may lack tokenizer.json
    # which the native C++ tokenizer needs.
    tokenizer_json_t0 = time.monotonic()
    for tok_subdir in ("tokenizer_2", "tokenizer"):
        tok_dir = model_dir_path / tok_subdir
        if tok_dir.is_dir() and not (tok_dir / "tokenizer.json").exists():
            _ensure_tokenizer_json(tok_dir)
    _add_build_timing(
        build_timing, "tokenizer_json_ensure_s",
        time.monotonic() - tokenizer_json_t0)
    _write_build_timing(build_timing)

    # Embed tokenizer files from tokenizer subdirectories.
    # Multi-encoder models (FLUX, SD3) have tokenizer/ (CLIP) and
    # tokenizer_2/ (T5).  Prefer tokenizer_2/ if it has tokenizer.json
    # (fast tokenizer format) since T5 provides the main text conditioning.
    # Fall back to tokenizer/ for single-tokenizer models (Wan, Z-Image).
    _tok_filenames = ("tokenizer.json", "tokenizer_config.json",
                      "special_tokens_map.json", "vocab.json",
                      "merges.txt", "spiece.model", "tokenizer.model")
    _tok_embedded = set()

    for tok_subdir in ("tokenizer_2", "tokenizer"):
        tokenizer_dir = model_dir_path / tok_subdir
        if not tokenizer_dir.is_dir():
            continue
        for filename in _tok_filenames:
            if filename in _tok_embedded:
                continue  # already embedded from higher-priority dir
            file_path = tokenizer_dir / filename
            if file_path.exists():
                sections.append(BundleSection(filename, file_path.read_bytes()))
                _tok_embedded.add(filename)

    # For dual-tokenizer models (FLUX): also embed CLIP tokenizer files
    # under prefixed names so the C++ runtime can create a separate CLIP
    # tokenizer.  CLIP lives in tokenizer/ (BPE with vocab.json + merges.txt).
    _clip_file_map = {
        "tokenizer.json": "clip_tokenizer.json",
        "vocab.json": "clip_vocab.json",
        "merges.txt": "clip_merges.txt",
        "tokenizer_config.json": "clip_tokenizer_config.json",
        "special_tokens_map.json": "clip_special_tokens_map.json",
    }
    clip_tokenizer_dir = model_dir_path / "tokenizer"
    if clip_tokenizer_dir.is_dir() and (model_dir_path / "tokenizer_2").is_dir():
        for src_name, dst_name in _clip_file_map.items():
            file_path = clip_tokenizer_dir / src_name
            if file_path.exists():
                sections.append(BundleSection(dst_name, file_path.read_bytes()))

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
        max_cache_length=max_cache_length,
        tokenizer_add_special_tokens=tokenizer_add_special_tokens,
    )

    write_t0 = time.monotonic()
    write_bundle(output_path, info, sections)
    _add_build_timing(build_timing, "bundle_write_s", time.monotonic() - write_t0)
    t4 = time.monotonic()
    build_timing["total_s"] = t4 - t0
    _write_build_timing(build_timing)
    print(f"[trtmc build] Bundle saved: {output_path} [{t4 - t0:.1f}s total]",
          file=sys.stderr)


def build(
    model_id_or_path: str,
    output_path: str,
    max_cache_length: int = 256,
    *,
    decoder_engine_layout: str = "split",
    dynamic_kv_cache: bool = False,
    dynamic_kv_profile_rows_override: list[int] | None = None,
    precision: str = "fp32",
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
    audio_magpie_max_source_positions: int = 0,
    parallel_config: ParallelConfig | None = None,
    diffusion_overrides: dict | None = None,
    build_timing_path: str | None = None,
) -> None:
    """Build a .trtfb bundle from a HuggingFace model ID or local path.

    Like HF transformers, accepts either:
    - A HuggingFace repo ID: ``"Qwen/Qwen3-0.6B"`` (auto-downloads)
    - A local directory: ``"models/hf/Qwen__Qwen3-0.6B"``

    Args:
        model_id_or_path: HF repo ID or local directory with config.json + safetensors.
        output_path: Where to write the .trtfb bundle.
        max_cache_length: KV cache length for the engine.
        decoder_engine_layout: ``"split"`` or ``"dual_profile"``.
        verbose: Print detailed TRT builder logs.
        fp8_scales: Per-layer FP8 scales dict, or ``"auto"`` for auto-calibration.
        save_fp8_scales: Path to save calibrated FP8 scales JSON.
    """
    model_dir = _resolve_model(model_id_or_path)
    build_bundle._model_id_or_path_orig = model_id_or_path
    build_bundle._fp8_scales = fp8_scales
    build_bundle._save_fp8_scales = save_fp8_scales
    build_bundle(model_dir, output_path, max_cache_length,
                 decoder_engine_layout=decoder_engine_layout,
                 dynamic_kv_cache=dynamic_kv_cache,
                 dynamic_kv_profile_rows_override=dynamic_kv_profile_rows_override,
                 precision=precision,
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
                 audio_magpie_max_source_positions=audio_magpie_max_source_positions,
                 parallel_config=parallel_config,
                 diffusion_overrides=diffusion_overrides,
                 build_timing_path=build_timing_path)
