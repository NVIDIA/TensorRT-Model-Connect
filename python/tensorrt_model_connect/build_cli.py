"""Builder CLI implementation for ``trtmc build``."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import struct
import sys
from pathlib import Path


def _get_version() -> str:
    """Get package version, trying importlib.metadata first, then __init__."""
    try:
        from importlib.metadata import version
        return version("tensorrt-model-connect")
    except Exception:
        pass
    try:
        from . import __version__
        return __version__
    except ImportError:
        return "0.1.0"


__version__ = _get_version()


def _cmd_build(args: argparse.Namespace) -> int:
    if not args.model:
        print("Error: model (HF repo ID or local directory) required",
              file=sys.stderr)
        return 1
    if not args.output:
        print("Error: -o / --output required", file=sys.stderr)
        return 1

    build_model_ref = args.model

    # Backend dispatch: default to auto-selection of the native TRT backend.
    method_name = getattr(args, 'method', 'auto')
    if method_name == 'auto':
        try:
            method_name, build_model_ref = _auto_select_build_backend(args.model)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1

    if not getattr(args, "_skip_profile_resolution", False):
        try:
            build_model_ref, build_family = _resolve_build_model_metadata(
                build_model_ref,
                method_name,
            )
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            if getattr(args, "verbose", False):
                import traceback

                traceback.print_exc()
            return 1

        reexec_rc = _maybe_reexec_build_in_profile(
            args,
            build_model_ref,
            build_family,
        )
        if reexec_rc is not None:
            return reexec_rc

    from .parallel_config import ParallelConfig

    tp_size = int(getattr(args, "tensor_parallel_size", 1) or 1)
    parallel_config = (
        ParallelConfig(mode="tensor_parallel", tp_size=tp_size)
        if tp_size > 1
        else None
    )

    # RTX selection MUST happen before any TensorRT API is touched.
    if getattr(args, 'rtx', False):
        from . import trt_compat
        trt_compat.configure_backend(rtx=True)
        print("[trtmc build] Using TensorRT-RTX backend", file=sys.stderr)

    # Raw TRT path imports builder modules that bind trt_compat.get_trt().
    from .engine_builder import build
    from .quantization import canonicalize_quant_format

    # FP8 quantization: --fp8-scales (pre-computed) or --fp8 (auto-calibrate)
    fp8_scales = None
    fp8_auto = getattr(args, 'fp8', False)
    if getattr(args, 'fp8_scales', None):
        import json as _json
        with open(args.fp8_scales) as _f:
            fp8_scales = _json.load(_f)
        print(f"[trtmc build] Loaded FP8 scales from {args.fp8_scales} "
              f"({len(fp8_scales)} layers)", file=sys.stderr)
    elif fp8_auto:
        # Sentinel: engine_builder will call plugin.fp8_calibrate()
        fp8_scales = "auto"
        print("[trtmc build] FP8 auto-calibration enabled", file=sys.stderr)

    save_fp8_scales = getattr(args, 'save_fp8_scales', None)
    quantize = canonicalize_quant_format(getattr(args, "quantize", None))

    # Resolve the registry-backed build-time config up front (before build),
    # so build-time namespaces can feed kwargs directly. Importing
    # runtime_config triggers registration of any schema modules declared
    # under tensorrt_model_connect.runtime_config.schemas.
    cli_cfg = getattr(args, "config", None)
    cli_sets = getattr(args, "set_flags", None) or []
    resolved_bundle = None
    if cli_cfg or cli_sets:
        from .runtime_config import resolve_cli_config
        from .runtime_config.schemas import load_all as _load_schemas
        _load_schemas()
        try:
            resolved_bundle = resolve_cli_config(
                config_path=cli_cfg, set_tokens=cli_sets)
        except (ValueError, FileNotFoundError, KeyError) as exc:
            print(f"Error resolving config: {exc}", file=sys.stderr)
            return 1
        try:
            audio_magpie_max_source_positions = int(resolved_bundle.get(
                "audio_magpie", "max_source_positions"))
        except KeyError:
            audio_magpie_max_source_positions = 0
    else:
        audio_magpie_max_source_positions = 0

    try:
        build(
            model_id_or_path=build_model_ref,
            output_path=args.output,
            max_cache_length=args.max_cache_length,
            decoder_engine_layout=getattr(args, "decoder_engine_layout", "split"),
            dynamic_kv_cache=getattr(args, "dynamic_kv_cache", False),
            dynamic_kv_profile_rows_override=getattr(args, "dynamic_kv_profile_rows", None),
            precision=args.precision,
            quantize=quantize,
            quant_scales=args.quant_scales,
            quant_calibration_samples=args.quant_calibration_samples,
            verbose=args.verbose,
            fp8_scales=fp8_scales,
            save_fp8_scales=save_fp8_scales,
            rtx=getattr(args, 'rtx', False),
            triattention_stats_path=getattr(args, "triattention_stats", None),
            triattention_kv_budget=getattr(args, "triattention_kv_budget", None),
            triattention_divide_length=getattr(args, "triattention_divide_length", 128),
            triattention_recent_window=getattr(args, "triattention_recent_window", 0),
            triattention_score_aggregation=getattr(
                args, "triattention_score_aggregation", "mean"),
            triattention_count_prompt_tokens=getattr(
                args, "triattention_count_prompt_tokens", True),
            triattention_protect_prefill=getattr(args, "triattention_protect_prefill", True),
            triattention_disable_mlr=getattr(args, "triattention_disable_mlr", False),
            triattention_disable_trig=getattr(args, "triattention_disable_trig", False),
            audio_magpie_max_source_positions=audio_magpie_max_source_positions,
            parallel_config=parallel_config,
            build_timing_path=getattr(args, "build_timing_json", None),
            max_batch_size=int(getattr(args, "max_batch_size", 1) or 1),
            diffusion_overrides={
                key: value
                for key, value in {
                    "image_height": getattr(args, "image_height", None),
                    "image_width": getattr(args, "image_width", None),
                    "video_height": getattr(args, "video_height", None),
                    "video_width": getattr(args, "video_width", None),
                    "video_num_frames": getattr(args, "video_num_frames", None),
                    "num_inference_steps": getattr(args, "num_inference_steps", None),
                }.items()
                if value is not None
            },
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Emit effective_config.json alongside the bundle when the caller
    # supplied --config/--set. The ConfigBundle was already resolved above
    # (so build() could consume namespaced kwargs); here we just serialize.
    if resolved_bundle is not None:
        from .runtime_config import write_effective_config_next_to
        path = write_effective_config_next_to(resolved_bundle, args.output)
        print(f"[trtmc build] Wrote effective config: {path}", file=sys.stderr)
    return 0


def _resolve_build_model_metadata(model_ref: str, method_name: str) -> tuple[str, str]:
    """Return (resolved_model_ref, family_name) for the selected build backend."""
    del method_name
    from .config import ModelConfig
    from .engine_builder import _resolve_model, find_diffusion_plugin, find_plugin

    resolved_model_ref = _resolve_model(model_ref)
    model_dir = Path(resolved_model_ref)

    if (model_dir / "model_index.json").exists():
        model_index = json.loads((model_dir / "model_index.json").read_text())
        plugin = find_diffusion_plugin(str(model_index.get("_class_name", "") or ""))
        return resolved_model_ref, getattr(plugin, "name", "")

    config = ModelConfig.from_dir(model_dir)
    plugin = find_plugin(config.model_type)
    return resolved_model_ref, getattr(plugin, "name", "")


def _resolve_build_profile_name(family_name: str) -> str:
    from .python_profiles import normalize_execution_profiles

    return normalize_execution_profiles(None, family=family_name).get("build", "base")


def _maybe_reexec_build_in_profile(
    args: argparse.Namespace,
    build_model_ref: str,
    build_family: str,
) -> int | None:
    """Re-exec build under a declared Python profile when the family requires it."""
    from .python_profiles import (
        DEFAULT_PROFILE,
        resolve_profile_python,
    )

    required_profile = _resolve_build_profile_name(build_family)
    if required_profile == DEFAULT_PROFILE:
        return None

    active_profile = str(getattr(args, "active_python_profile", "") or "").strip()
    if active_profile == required_profile:
        return None

    target_python = resolve_profile_python(required_profile, sys.executable)
    current_python = str(Path(sys.executable).absolute())
    if current_python == target_python:
        return None

    env = os.environ.copy()
    cmd = [
        target_python,
        "-m",
        "tensorrt_model_connect.__main__",
        *sys.argv[1:],
        "--active-python-profile",
        required_profile,
    ]
    print(
        f"[trtmc build] Switching build to Python profile {required_profile!r}: "
        f"{target_python}",
        file=sys.stderr,
    )
    return subprocess.run(cmd, env=env).returncode


def _auto_select_build_backend(model_ref: str) -> tuple[str, str]:
    """Return (method_name, resolved_model_ref) for the best available backend.

    The selection rule is:
      1. Use the raw TensorRT Network API backend when a native family plugin
         exists for the model.
    """
    from .config import ModelConfig
    from .engine_builder import _resolve_model, find_plugin, find_diffusion_plugin

    resolved_model_ref = _resolve_model(model_ref)
    model_dir = Path(resolved_model_ref)

    if (model_dir / "model_index.json").exists():
        model_index = json.loads((model_dir / "model_index.json").read_text())
        pipeline_class = str(model_index.get("_class_name", "") or "")
        raw_supported = (
            find_diffusion_plugin(pipeline_class) is not None
            or find_plugin(pipeline_class.lower()) is not None
        )
    else:
        config = ModelConfig.from_dir(model_dir)
        raw_plugin = find_plugin(config.model_type)
        raw_supported = raw_plugin is not None

    if raw_supported:
        print("[trtmc build] Auto-selected backend: trt", file=sys.stderr)
        return "trt", resolved_model_ref

    raise RuntimeError(
        "No native TRT family plugin matched this model. "
        "Choose a model with native TRT support."
    )



def _parse_profile_rows(value: str) -> list[int]:
    rows: list[int] = []
    for part in value.split(","):
        text = part.strip()
        if not text:
            continue
        try:
            rows.append(int(text))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid dynamic KV profile row {text!r}; expected comma-separated integers"
            ) from exc
    if not rows:
        raise argparse.ArgumentTypeError(
            "Expected at least one integer in --dynamic-kv-profile-rows"
        )
    return rows


def _read_bundle_header(bundle_path: str) -> dict:
    """Read and return the JSON header from a .trtfb bundle."""
    with open(bundle_path, "rb") as f:
        magic = f.read(8)
        if magic != b"TRTFB\x00\x01\x00":
            raise ValueError(f"Not a valid .trtfb bundle: {bundle_path}")
        header_len = struct.unpack("<Q", f.read(8))[0]
        header_json = f.read(header_len).decode("utf-8")
    return json.loads(header_json)


def list_engine_sections(bundle_path: str) -> list[dict]:
    """List all TRT engine plan sections in a bundle.

    Returns list of dicts: [{name, size_bytes, size_mb, role}]
    where role is 'primary', 'vision', 'text_encoder', 'denoiser', 'vae', etc.
    """
    header = _read_bundle_header(bundle_path)
    sections = header.get("sections", {})

    engines = []
    for name, meta in sections.items():
        is_tp_rank_plan = name.startswith("engine_plan_tp_rank")
        if not name.endswith("_plan") and name != "engine_plan" and not is_tp_rank_plan:
            continue
        size_bytes = meta.get("size", 0)

        # Infer role from section name
        if name == "engine_plan":
            role = "decode" if "prefill_engine_plan" in sections else "primary"
        elif name == "prefill_engine_plan":
            role = "prefill"
        elif is_tp_rank_plan:
            role = name.replace("engine_plan_", "")
        elif "vision" in name:
            role = "vision"
        elif "text_encoder" in name:
            role = "text_encoder"
        elif "denoiser" in name:
            role = "denoiser"
        elif "vae" in name:
            role = "vae"
        elif "lt_" in name or "local_transformer" in name:
            role = "local_transformer"
        else:
            role = name.replace("_plan", "")

        engines.append({
            "name": name,
            "size_bytes": size_bytes,
            "size_mb": round(size_bytes / (1024 * 1024), 1),
            "role": role,
        })

    return engines


def _cmd_inspect(args: argparse.Namespace) -> int:
    bundle_path = args.bundle_path
    if not bundle_path:
        print("Error: bundle path required", file=sys.stderr)
        return 1

    try:
        header = _read_bundle_header(bundle_path)

        if getattr(args, 'list_engines', False):
            # Engine-only listing mode
            engines = list_engine_sections(bundle_path)
            if not engines:
                print("No engine sections found.", file=sys.stderr)
                return 1
            print(f"{'Section':<30} {'Size':>10} {'Role':<16}")
            print(f"{'-'*30} {'-'*10} {'-'*16}")
            for e in engines:
                print(f"{e['name']:<30} {e['size_mb']:>8.1f} MB {e['role']:<16}")
            return 0

        fields = [
            ("Model ID", "model_id"),
            ("Model type", "model_type"),
            ("Family", "family"),
            ("TRT version", "trt_version"),
            ("TRT ABI", "trt_abi"),
            ("GPU", "gpu_name"),
            ("Created", "created_at"),
            ("Vocab size", "vocab_size"),
            ("Hidden size", "hidden_size"),
            ("Layers", "num_layers"),
            ("Attention heads", "num_attention_heads"),
            ("KV heads", "num_key_value_heads"),
            ("Max cache length", "max_cache_length"),
            ("Precision", "precision"),
            ("Quantization", "quantization"),
            ("Engine backend", "engine_backend"),
        ]
        for label, key in fields:
            print(f"{label + ':':<20} {header.get(key, '')}")

        sections = header.get("sections", {})
        if sections:
            print(f"{'Sections:':<20}")
            for name, meta in sections.items():
                size_mb = meta.get("size", 0) / (1024 * 1024)
                print(f"  {name}: {size_mb:.1f} MB")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def _cmd_version(_args: argparse.Namespace) -> int:
    print(f"trtmc build {__version__}")
    from . import trt_compat
    trt_version = trt_compat.module_version("tensorrt")
    if trt_version:
        print(f"TensorRT:  {trt_version}")
    else:
        print("TensorRT:  not installed")
    trt_rtx_version = trt_compat.module_version("tensorrt_rtx")
    if trt_rtx_version:
        print(f"TensorRT-RTX: {trt_rtx_version}")
    else:
        print("TensorRT-RTX: not installed")
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    """Compatibility wrapper for tests and callers that import command handlers."""
    return _cmd_build(args)


def cmd_inspect(args: argparse.Namespace) -> int:
    """Compatibility wrapper accepting both historic ``bundle`` and ``bundle_path`` args."""
    if not hasattr(args, "bundle_path") and hasattr(args, "bundle"):
        args.bundle_path = args.bundle
    return _cmd_inspect(args)


def cmd_version(args: argparse.Namespace) -> int:
    """Compatibility wrapper for tests and callers that import command handlers."""
    return _cmd_version(args)


def main() -> None:
    # RTX selection MUST happen before ANY tensorrt_model_connect module touches TRT.
    # We do an early argv scan before argparse touches anything.
    if "--rtx" in sys.argv:
        try:
            from . import trt_compat
            trt_compat.configure_backend(rtx=True)
            print("[trtmc build] Using TensorRT-RTX backend", file=sys.stderr)
        except ImportError:
            print("Error: --rtx requires tensorrt_rtx. Install: pip install tensorrt-rtx",
                  file=sys.stderr)
            sys.exit(1)

    parser = argparse.ArgumentParser(
        prog="trtmc",
        description="Build .trtfb bundles from HuggingFace models",
    )
    subparsers = parser.add_subparsers(dest="command")

    # trtmc build <model> -o <out.trtfb>
    build_p = subparsers.add_parser("build", help="Build a .trtfb bundle")
    build_p.add_argument("model",
                         help="HF repo ID (e.g. Qwen/Qwen3-0.6B) or local directory")
    build_p.add_argument("-o", "--output", required=True,
                         help="Output .trtfb file path")
    build_p.add_argument("--trust-remote-code", action="store_true",
                         help="Allow Hugging Face model code that requires trust_remote_code")
    build_p.add_argument("--max-cache-length", type=int, default=256,
                         help="KV cache length (default: 256)")
    build_p.add_argument(
        "--decoder-engine-layout",
        choices=["split", "dual_profile"],
        default="split",
        help="Decoder engine layout for supported LLMs: split builds separate "
             "prefill/decode engines (default); dual_profile keeps one "
             "low-VRAM engine with multiple optimization profiles",
    )
    build_p.add_argument("--dynamic-kv-cache", action="store_true",
                         help="Build decoder bundles with runtime-resizable KV cache support")
    # TP is a narrow build-only path, not a generic runtime-config namespace.
    build_p.add_argument(
        "--tensor-parallel-size",
        "--tp-size",
        dest="tensor_parallel_size",
        type=int,
        choices=[1, 2, 4, 8],
        default=1,
        help="Build a tensor-parallel decoder bundle with this TP size")
    build_p.add_argument(
        "--dynamic-kv-profile-rows",
        type=_parse_profile_rows,
        default=None,
        help="Comma-separated dynamic-KV optimization profile upper bounds "
             "(overrides the builder's default profile schedule)",
    )
    build_p.add_argument("--image-height", type=int, default=None,
                         help="Diffusion image height override")
    build_p.add_argument("--image-width", type=int, default=None,
                         help="Diffusion image width override")
    build_p.add_argument("--video-height", type=int, default=None,
                         help="Diffusion video height override")
    build_p.add_argument("--video-width", type=int, default=None,
                         help="Diffusion video width override")
    build_p.add_argument("--video-num-frames", type=int, default=None,
                         help="Diffusion video frame count override")
    build_p.add_argument("--num-inference-steps", type=int, default=None,
                         help="Diffusion denoising step count override")
    build_p.add_argument(
        "--max-batch-size", type=int, default=1,
        help="Build diffusion bundle whose engines support batch sizes up to N "
             "(default: 1). Applied per component using the family policy: "
             "DiT honors N, text encoder caps at min(2N, 8), VAE always builds "
             "B=1 (the runtime slices)."
    )
    build_p.add_argument("--precision", choices=["fp32", "fp16", "bf16"],
                         default="fp32",
                         help="Engine precision (default: fp32)")
    build_p.add_argument("--quantize",
                         choices=["fp8", "int8", "int8_sq", "int4", "int4_awq", "nvfp4", "w4a8"],
                         default=None,
                         help="Quantization format (default: none)")
    build_p.add_argument("--quant-scales",
                         default=None,
                         help="Path to pre-computed quantization scales JSON (skips calibration)")
    build_p.add_argument("--quant-calibration-samples",
                         type=int, default=512,
                         help="Number of calibration samples for PTQ (default: 512)")
    build_p.add_argument("--method", type=str, default="auto",
                         choices=["auto", "trt"],
                         help="Engine definition method: auto (default, native TRT) or trt")
    build_p.add_argument("--verbose", action="store_true",
                         help="Verbose TRT builder output")
    build_p.add_argument("--fp8", action="store_true",
                         help="Enable FP8 quantization (auto-calibrate via ModelOpt)")
    build_p.add_argument("--fp8-scales", default=None,
                         help="Path to pre-computed FP8 scales JSON (skips calibration)")
    build_p.add_argument("--save-fp8-scales", default=None,
                         help="Save calibrated FP8 scales to JSON (reuse with --fp8-scales)")
    build_p.add_argument("--rtx", action="store_true",
                         help="Build engine for TRT-RTX (portable, JIT-compiled at runtime)")
    build_p.add_argument("--triattention-stats", default=None,
                         help="Path to upstream TriAttention calibration stats (.pt) to embed")
    build_p.add_argument("--triattention-kv-budget", type=int, default=None,
                         help="Runtime KV budget for experimental TriAttention compaction")
    build_p.add_argument("--triattention-divide-length", type=int, default=128,
                         help="Trigger TriAttention compaction when cache reaches "
                              "kv_budget + divide_length (default: 128)")
    build_p.add_argument("--triattention-recent-window", type=int, default=128,
                         help="Always keep this many recent tokens when TriAttention is enabled")
    build_p.add_argument("--triattention-score-aggregation",
                         choices=["mean", "max"], default="mean",
                         help="How to aggregate TriAttention offset scores")
    build_p.add_argument("--triattention-count-prompt-tokens",
                         dest="triattention_count_prompt_tokens",
                         action="store_true", default=True,
                         help="Count prompt tokens against the TriAttention KV budget")
    build_p.add_argument("--triattention-no-count-prompt-tokens",
                         dest="triattention_count_prompt_tokens",
                         action="store_false",
                         help="Exclude prompt tokens from the TriAttention KV budget")
    build_p.add_argument("--triattention-protect-prefill",
                         dest="triattention_protect_prefill",
                         action="store_true", default=True,
                         help="Prefer retaining prompt tokens during TriAttention compaction "
                              "(default: enabled)")
    build_p.add_argument("--triattention-no-protect-prefill",
                         dest="triattention_protect_prefill",
                         action="store_false",
                         help="Allow prompt tokens to be pruned during TriAttention compaction")
    build_p.add_argument("--triattention-disable-mlr", action="store_true",
                         help="Disable TriAttention's magnitude-based additive term")
    build_p.add_argument("--triattention-disable-trig", action="store_true",
                         help="Disable TriAttention's trig scoring term")
    build_p.add_argument(
        "--build-timing-json", default=None,
        help="Write structured build timing JSON to this path")
    build_p.add_argument(
        "--active-python-profile", default="", help=argparse.SUPPRESS)

    # Generic two-flag config surface. New features register a namespaced
    # schema and are consumed through these flags without growing the CLI.
    # Adding a new feature MUST NOT add a new flag here.
    build_p.add_argument(
        "--config", default=None, metavar="FILE",
        help="Config profile file (.json/.yaml). Contributes to the session "
             "layer; combine with --set for ad-hoc overrides.")
    build_p.add_argument(
        "--set", action="append", dest="set_flags", default=None,
        metavar="NS.FIELD=VALUE",
        help="Set one config field for this session (repeatable). Uses the "
             "schema's declared type; unknown namespaces/fields fail fast.")

    # python -m tensorrt_model_connect inspect <bundle.trtfb>
    inspect_p = subparsers.add_parser("inspect",
                                      help="Inspect a .trtfb bundle")
    inspect_p.add_argument("bundle_path", help=".trtfb file to inspect")
    inspect_p.add_argument("--list-engines", action="store_true",
                           help="List only TRT engine plan sections with roles")

    # python -m tensorrt_model_connect version
    subparsers.add_parser("version", help="Show version info")

    # Keep direct module compatibility: `python -m tensorrt_model_connect
    # <model-dir> -o out.trtfb` still means build. The public native CLI uses
    # explicit `trtmc build`.
    command_names = {"build", "inspect", "version"}
    cli_argv = sys.argv[1:]
    if cli_argv and cli_argv[0] not in command_names and cli_argv[0] not in ("--help", "-h"):
        cli_argv = ["build"] + cli_argv
    args = parser.parse_args(cli_argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "build": _cmd_build,
        "inspect": _cmd_inspect,
        "version": _cmd_version,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    sys.exit(handler(args))
