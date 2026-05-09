"""E2E orchestrator — coordinates the full lifecycle for one model case.

The orchestrator depends ONLY on contracts (Dependency Inversion Principle).
It does not import any concrete runner, reference, or comparator. All
concrete implementations are resolved at runtime via the registry.

Lifecycle per case:
    1. Initialize artifact sink and write case snapshot.
    2. Run preflight checks.
    3. Resolve or build bundle.
    4. For each stage:
       a. Execute TRT strategy runner.
       b. Execute reference backend runner.
       c. Compare outputs.
       d. Persist stage artifacts.
    5. Execute determinism reruns for designated stages.
    6. Aggregate stage outcomes to final status.
    7. Write final result.json.
    8. Return E2EResult for pytest assertion.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import (
    CompareResult,
    E2ECase,
    E2EResult,
    E2EStatus,
    FailureType,
    PreflightRequirement,
    RunContext,
    StageOutput,
    StageStatus,
    ThresholdProfile,
)
from . import _case_artifact_dir, save_full_stderr
from .python_profiles import profile_env_var
from .registry import get_comparator, get_contract_plugin, get_reference, get_runner

logger = logging.getLogger(__name__)

_TRTMC_TIMING_RE = re.compile(
    r"^\[trtmc\.timing\]\s+"
    r"prefill_ms=(?P<prefill_ms>[-+0-9.eE]+)\s+"
    r"decode_ms=(?P<decode_ms>[-+0-9.eE]+)\s+"
    r"total_ms=(?P<total_ms>[-+0-9.eE]+)\s*$",
    re.MULTILINE,
)
_TRTMC_LOAD_TIMING_RE = re.compile(
    r"^\[trtmc\.load_timing\]\s+.*?label=\"(?P<label>[^\"]+)\".*?"
    r"load_deserialize_ms=(?P<ms>[-+0-9.eE]+)",
    re.MULTILINE,
)
_TRTMC_ENGINE_TIMING_RE = re.compile(
    r"^\[trtmc\.engine_timing\]\s+.*?label=\"(?P<label>[^\"]+)\".*?"
    r"execute_ms=(?P<ms>[-+0-9.eE]+)",
    re.MULTILINE,
)

_MIGRATED_RUNTIME_STRATEGIES = frozenset({
    "decoder_kv_cache",
    "decoder_moe",
    "ssm_recurrent",
    "rwkv_recurrent",
    "hybrid_mamba_attention",
    "encoder_only",
    "embedding",
    "reranking",
    "vision_language",
    "segmentation",
    "prompted_segmentation",
    "object_detection",
    "neural_operator",
    "text_to_audio",
    "speech_to_text",
    "speech_to_text_rnnt",
    "speech_to_speech",
    "omni_multimodal",
    "diffusion",
    "patchtst_torchtrt",
    "patchtsmixer_torchtrt",
    "timesfm_torchtrt",
    "chronos_bolt_torchtrt",
})
_NEW_RUNTIME_MARKER = "backend=trt_new_runtime"
_LEGACY_RUNTIME_MARKER = "Runtime path: compatibility factory mode"


# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------


def _check_binary_exists(ctx: RunContext, req: PreflightRequirement) -> tuple[bool, str]:
    """Check that the trtmc binary exists."""
    path = req.args.get("path", ctx.binary_path)
    if path and Path(path).is_file():
        return True, f"Binary found: {path}"
    return False, f"Binary not found: {path}"


def _check_gpu_memory(ctx: RunContext, req: PreflightRequirement) -> tuple[bool, str]:
    """Check GPU has enough memory."""
    min_gb = req.args.get("min_gb", 0)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            total_mb = int(result.stdout.strip().splitlines()[0])
            total_gb = total_mb / 1024
            if total_gb >= min_gb:
                return True, f"GPU memory: {total_gb:.1f} GB >= {min_gb} GB"
            return False, f"GPU memory: {total_gb:.1f} GB < {min_gb} GB required"
    except Exception as e:
        return False, f"GPU memory check failed: {e}"
    return False, "GPU memory check failed: unknown error"


def _check_hf_auth(ctx: RunContext, req: PreflightRequirement) -> tuple[bool, str]:
    """Check that HF auth token is present."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return True, "HF auth token found"
    hf_token_path = Path.home() / ".huggingface" / "token"
    if hf_token_path.is_file():
        return True, "HF auth token found in ~/.huggingface/token"
    return False, "HF auth token not found (set HF_TOKEN or login with huggingface-cli)"


def _check_asset_exists(ctx: RunContext, req: PreflightRequirement) -> tuple[bool, str]:
    """Check that a required asset file exists."""
    asset_path = req.args.get("path", "")
    if not asset_path:
        return False, "Asset path not specified"
    # Resolve relative paths against project root, then e2e data dir
    p = Path(asset_path)
    if not p.is_absolute():
        project_root = Path(__file__).resolve().parent.parent.parent
        candidate = project_root / asset_path
        if candidate.is_file():
            p = candidate
        else:
            # Fallback: try e2e data dir with just the filename
            e2e_data = project_root / "tests" / "e2e" / "data"
            p = e2e_data / Path(asset_path).name
    if p.is_file():
        return True, f"Asset found: {p}"
    return False, f"Asset not found: {p}"


def _check_python_module(ctx: RunContext, req: PreflightRequirement) -> tuple[bool, str]:
    """Check that a Python module is importable."""
    module = req.args.get("module", "")
    phase = str(req.args.get("phase", "reference") or "reference")
    timeout_s = int(req.args.get("timeout_s", 30))
    if not module:
        return False, "Module name not specified"
    try:
        python = {
            "build": ctx.build_python_path(),
            "runtime": ctx.runtime_python_path(),
            "reference": ctx.reference_python_path(),
        }.get(phase, ctx.reference_python_path())
        python = python or sys.executable
        result = subprocess.run(
            [
                python,
                "-c",
                (
                    "import importlib.util, sys; "
                    f"sys.exit(0 if importlib.util.find_spec({module!r}) else 1)"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if result.returncode == 0:
            return True, f"Module {module} available in {phase} profile"
        return False, f"Module {module} not available in {phase} profile"
    except Exception as exc:
        return False, f"Module check failed in {phase} profile: {exc}"


_PREFLIGHT_CHECKERS = {
    "binary_exists": _check_binary_exists,
    "gpu_memory_min_gb": _check_gpu_memory,
    "hf_auth_token_present": _check_hf_auth,
    "asset_exists": _check_asset_exists,
    "python_module_available": _check_python_module,
}


def run_preflight(
    case: E2ECase,
    ctx: RunContext,
) -> tuple[bool, list[dict[str, Any]]]:
    """Run all preflight requirements for a case.

    Returns (all_passed, details) where details is a list of
    per-requirement dicts with keys: kind, passed, message, gating.
    """
    details: list[dict[str, Any]] = []
    all_gating_passed = True

    for req in case.preflight:
        checker = _PREFLIGHT_CHECKERS.get(req.kind)
        if checker is None:
            passed = False
            message = f"Unknown preflight kind: {req.kind}"
        else:
            try:
                passed, message = checker(ctx, req)
            except Exception as e:
                passed = False
                message = f"Preflight check {req.kind} raised: {e}"

        details.append({
            "kind": req.kind,
            "passed": passed,
            "message": message,
            "gating": req.gating,
        })

        if not passed and req.gating:
            all_gating_passed = False

    return all_gating_passed, details


# ---------------------------------------------------------------------------
# Bundle resolution
# ---------------------------------------------------------------------------


def _resolve_bundle(
    case: E2ECase,
    ctx: RunContext,
) -> tuple[str | None, float | None, str, dict[str, Any]]:
    """Resolve or build the bundle.

    Returns (path, build_time_s, error_msg, build_info) where build_info
    contains the subprocess command, stdout, stderr, and returncode when a
    build was executed.  build_info is empty when the bundle already exists.
    """
    engine_dir = Path(ctx.engine_dir)
    bundle_path = engine_dir / case.bundle

    if bundle_path.is_file() and not ctx.rebuild:
        return str(bundle_path), None, "", {}

    # Build the bundle
    hf_id = case.hf_id
    if hf_id and not os.path.isabs(hf_id):
        project_root = Path(__file__).resolve().parents[2]
        candidate = project_root / hf_id
        if candidate.exists():
            hf_id = str(candidate)
    max_cache = case.inputs.get("max_cache_length", 256)

    build_args = case.metadata.get("build_args", {})
    build_method = _manifest_build_method(build_args)

    build_python = ctx.build_python_path() or sys.executable
    cmd = [
        build_python, "-m", "tensorrt_model_connect.__main__", "build",
        hf_id, "-o", str(bundle_path),
        "--max-cache-length", str(max_cache),
    ]
    diffusion_build_args = {
        "image_height": "--image-height",
        "image_width": "--image-width",
        "video_height": "--video-height",
        "video_width": "--video-width",
        "video_num_frames": "--video-num-frames",
        "num_inference_steps": "--num-inference-steps",
    }
    for input_key, cli_arg in diffusion_build_args.items():
        value = case.inputs.get(input_key)
        if value is not None:
            cmd.extend([cli_arg, str(value)])
    if build_method:
        cmd.extend(["--method", build_method])
    precision = case.metadata.get("precision", "fp32")
    if precision != "fp32":
        cmd.extend(["--precision", precision])
    quantization = case.metadata.get("quantization", {})
    if isinstance(quantization, dict):
        quant_format = quantization.get("format")
        if quant_format and quant_format != "none":
            cmd.extend(["--quantize", str(quant_format)])
            scale_artifact = quantization.get("scale_artifact")
            if scale_artifact:
                cmd.extend(["--quant-scales", str(scale_artifact)])
            calibration_samples = quantization.get("calibration_samples")
            if calibration_samples is not None:
                cmd.extend([
                    "--quant-calibration-samples",
                    str(calibration_samples),
                ])
    if case.metadata.get("trust_remote_code"):
        cmd.append("--trust-remote-code")
    fp8_scales = case.metadata.get("fp8_scales")
    if fp8_scales:
        # Resolve relative to tests/e2e/data/
        scales_path = Path(__file__).parent.parent / "e2e" / "data" / fp8_scales
        if scales_path.is_file():
            cmd.extend(["--fp8-scales", str(scales_path)])

    logger.info("Building bundle: %s", " ".join(cmd))
    t0 = time.monotonic()
    env = os.environ.copy()
    if ctx.build_profile and ctx.build_profile != "base":
        cmd.extend(["--active-python-profile", ctx.build_profile])
    build_timing_path: Path | None = None
    if ctx.artifacts_dir:
        build_timing_path = Path(ctx.artifacts_dir) / case.name / "build_timing.json"
    else:
        build_timing_path = Path(ctx.engine_dir) / f"{case.name}.build_timing.json"
    build_timing_path.parent.mkdir(parents=True, exist_ok=True)
    cmd.extend(["--build-timing-json", str(build_timing_path)])
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=3600, env=env)
        elapsed = time.monotonic() - t0
    except subprocess.TimeoutExpired:
        build_timing = _load_build_timing(build_timing_path)
        build_info = {
            "command": cmd,
            "returncode": -1,
            "stdout": "",
            "stderr": "timeout",
        }
        if build_timing:
            build_info["timing"] = build_timing
            build_info["timing_path"] = str(build_timing_path)
        return None, None, f"Bundle build timed out for {hf_id}", {
            **build_info,
        }
    except Exception as e:
        build_timing = _load_build_timing(build_timing_path)
        build_info = {
            "command": cmd,
            "returncode": -1,
            "stdout": "",
            "stderr": str(e),
        }
        if build_timing:
            build_info["timing"] = build_timing
            build_info["timing_path"] = str(build_timing_path)
        return None, None, f"Bundle build failed for {hf_id}: {e}", {
            **build_info,
        }

    build_info: dict[str, Any] = {
        "command": cmd,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    build_timing = _load_build_timing(build_timing_path)
    if build_timing:
        build_info["timing"] = build_timing
        build_info["timing_path"] = str(build_timing_path)

    if result.returncode != 0:
        truncated, log_path = save_full_stderr(
            result.stderr, ctx.artifacts_dir or "", "bundle_build", case.name)
        msg = f"Bundle build failed for {hf_id} (rc={result.returncode}):\n{truncated}"
        if log_path:
            msg += f" (full stderr: {log_path})"
        return None, elapsed, msg, build_info

    return str(bundle_path), elapsed, "", build_info


# ---------------------------------------------------------------------------
# Threshold resolution
# ---------------------------------------------------------------------------


def _resolve_threshold(
    case: E2ECase,
) -> ThresholdProfile:
    """Build a ThresholdProfile from defaults + overrides.

    Resolution chain: defaults -> profile -> per-model -> manifest inline.
    """
    # Start with conservative defaults
    metrics: dict[str, float] = {
        "logit_atol": 1e-3,
        "layer_atol": 0.05,
        "token_agreement_rate": 0.8,
        "logit_cosine_p5": 0.99,
        "logit_rel_l2_p95": 0.05,
        "stable_top1_match_rate": 0.9,
        "unstable_topk_hit_rate": 0.8,
        "normalized_text_edit_distance": 0.2,
    }

    # Try to load strategy-specific defaults
    try:
        from .thresholds import load_defaults
        strategy_defaults = load_defaults(case.task_strategy)
        if strategy_defaults:
            metrics.update(strategy_defaults)
    except ImportError:
        pass

    # Apply per-model overrides from manifest
    metrics.update(case.threshold_overrides)

    return ThresholdProfile(
        task_strategy=case.task_strategy,
        profile_name=case.comparison_profile,
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _log_stage_subprocess(
    sink: Any,
    stage_name: str,
    output: StageOutput,
    prefix: str,
) -> None:
    """Extract subprocess info from StageOutput.metadata and log to the sink.

    Runners store subprocess details in metadata under various conventions:
    - Text generation: metadata = {"cpp": {...}, "debug_runner": {...}}
    - Vision language: metadata has "command", "returncode", "stdout", "stderr"
    - Other runners: flat metadata with "command", "returncode", etc.

    This function handles all conventions and writes log files for each
    subprocess found in the metadata.
    """
    meta = output.metadata
    if not meta:
        return

    # Check for nested sub-metadata dicts (e.g., text_generation has "cpp" and "debug_runner")
    nested_found = False
    for key, value in meta.items():
        if isinstance(value, dict) and "returncode" in value:
            nested_found = True
            label = f"{stage_name}_{prefix}_{key}"
            cmd = value.get("command", [])
            rc = value.get("returncode", -1)
            stdout = value.get("stdout", "")
            stderr = value.get("stderr", "")
            sink.log_command(
                command=cmd if isinstance(cmd, list) else [str(cmd)],
                rc=rc,
                stdout=str(stdout),
                stderr=str(stderr),
                label=label,
            )

    # If no nested dicts, check for flat metadata with command/returncode
    if not nested_found and "returncode" in meta:
        label = f"{stage_name}_{prefix}"
        cmd = meta.get("command", [])
        rc = meta.get("returncode", -1)
        stdout = meta.get("stdout", "")
        stderr = meta.get("stderr", "")
        sink.log_command(
            command=cmd if isinstance(cmd, list) else [str(cmd)],
            rc=rc,
            stdout=str(stdout),
            stderr=str(stderr),
            label=label,
        )


def _auto_register_artifacts(sink: Any, output: StageOutput, prefix: str) -> None:
    """Scan StageOutput.data for known artifact keys and register on sink."""
    _ARTIFACT_KEYS = {
        "wav_path": "wav",
        "frames_dir": "frames",
        "logits_path": "logits",
        "features_path": "features",
        "output_path": "output",
        "segmentation_map_path": "segmentation_map",
        "segmented_image_path": "segmented_image",
    }
    for data_key, artifact_key in _ARTIFACT_KEYS.items():
        value = output.data.get(data_key)
        if value and isinstance(value, str):
            # Store relative to artifacts dir if possible
            base = str(sink.base_dir)
            rel = value
            if value.startswith(base):
                rel = value[len(base):].lstrip("/")
            sink.register_artifact(f"{prefix}_{artifact_key}", rel)


def _collect_runtime_guard_payloads(value: Any) -> list[tuple[list[str], list[str], int | None]]:
    """Collect (command argv, stderr payloads, returncode) tuples from nested output structures."""
    payloads: list[tuple[list[str], list[str], int | None]] = []

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            command = node.get("command")
            command_argv: list[str] = []
            if isinstance(command, list):
                command_argv = [str(part) for part in command]
            elif isinstance(command, str) and command:
                try:
                    command_argv = shlex.split(command)
                except ValueError:
                    command_argv = [command]

            stderr_texts: list[str] = []

            for key in ("stderr", "stderr_truncated"):
                text = node.get(key)
                if isinstance(text, str) and text:
                    stderr_texts.append(text)

            stderr_log = node.get("stderr_log")
            if isinstance(stderr_log, str) and stderr_log:
                try:
                    stderr_texts.append(Path(stderr_log).read_text(encoding="utf-8"))
                except OSError:
                    pass

            returncode = node.get("returncode")
            if not isinstance(returncode, int):
                returncode = None

            if command_argv:
                payloads.append((command_argv, stderr_texts, returncode))

            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    return payloads


def _validate_trt_runtime_path(
    case: E2ECase,
    ctx: RunContext,
    output: StageOutput,
) -> str | None:
    """Ensure TRT E2E subprocesses for migrated strategies use the new runtime path."""
    if case.runtime_strategy not in _MIGRATED_RUNTIME_STRATEGIES:
        return None

    payloads = _collect_runtime_guard_payloads({
        "metadata": output.metadata,
        "data": output.data,
    })
    if not payloads:
        return None

    binary_name = Path(ctx.binary_path).name if ctx.binary_path else ""
    relevant_stderr: list[str] = []
    for command_argv, stderr_texts, returncode in payloads:
        executable = command_argv[0] if command_argv else ""
        executable_name = Path(executable).name if executable else ""
        if ctx.binary_path and executable == ctx.binary_path:
            combined_payload_stderr = "\n".join(stderr_texts)
            if returncode not in (None, 0) and _NEW_RUNTIME_MARKER not in combined_payload_stderr \
                    and _LEGACY_RUNTIME_MARKER not in combined_payload_stderr:
                continue
            relevant_stderr.extend(stderr_texts)
            continue
        if binary_name and executable_name == binary_name:
            combined_payload_stderr = "\n".join(stderr_texts)
            if returncode not in (None, 0) and _NEW_RUNTIME_MARKER not in combined_payload_stderr \
                    and _LEGACY_RUNTIME_MARKER not in combined_payload_stderr:
                continue
            relevant_stderr.extend(stderr_texts)
    if not relevant_stderr:
        return None

    combined_stderr = "\n".join(relevant_stderr)
    if _LEGACY_RUNTIME_MARKER in combined_stderr:
        return (
            f"E2E runtime guard failed for strategy {case.runtime_strategy}: "
            "TRT subprocess used legacy compatibility factory mode"
        )
    if _NEW_RUNTIME_MARKER not in combined_stderr:
        return (
            f"E2E runtime guard failed for strategy {case.runtime_strategy}: "
            "TRT subprocess did not confirm the new runtime path"
        )
    return None


def _build_repro_commands(
    case: E2ECase,
    ctx: RunContext,
    bundle_path: str | None,
    build_info: dict[str, Any],
) -> dict[str, str]:
    """Construct shell commands that reproduce each phase of the E2E test.

    Returns a dict with keys like "build_bundle", "trt_inference", "rerun_test",
    each mapped to a copy-pasteable shell command string.
    """
    repro: dict[str, str] = {}

    # Build command
    max_cache = case.inputs.get("max_cache_length", 256)
    bundle_target = bundle_path or str(Path(ctx.engine_dir) / case.bundle)
    build_parts = [
        ctx.build_python_path() or "python", "-m", "tensorrt_model_connect.__main__", "build", case.hf_id,
        "-o", bundle_target,
        "--max-cache-length", str(max_cache),
    ]
    build_method = _manifest_build_method(case.metadata.get("build_args", {}))
    if build_method:
        build_parts.extend(["--method", build_method])
    if case.metadata.get("trust_remote_code"):
        build_parts.append("--trust-remote-code")
    repro["build_bundle"] = " ".join(build_parts)

    # TRT inference command (task-specific C++ binary entrypoint)
    if bundle_path and ctx.binary_path:
        image = (case.inputs.get("image") or case.inputs.get("test_image")
                 or case.inputs.get("image_path"))
        task_strategy = case.task_strategy or ""
        if task_strategy == "neural_operator":
            infer_parts = [ctx.binary_path, "solve", bundle_path]
            branch_input = case.inputs.get("branch_input")
            trunk_input = case.inputs.get("trunk_input")
            if branch_input is not None or trunk_input is not None:
                if branch_input is not None:
                    infer_parts.extend(["--branch-input", _csv_arg(branch_input)])
                if trunk_input is not None:
                    infer_parts.extend(["--trunk-input", _csv_arg(trunk_input)])
            else:
                field_input = case.inputs.get("field_input")
                if field_input is not None:
                    infer_parts.extend(["--field-input", _csv_arg(field_input)])
        elif task_strategy == "diffusion_media_generation":
            infer_parts = [
                ctx.binary_path, "generate-video", bundle_path,
                "--prompt", _shell_quote(case.inputs.get("prompt", case.inputs.get("test_prompt", ""))),
                "--output", "/tmp/trtmc_frames",
                "--num-steps", str(case.inputs.get("num_inference_steps", 30)),
            ]
            guidance_scale = case.inputs.get("guidance_scale")
            if guidance_scale is not None:
                infer_parts.extend(["--guidance-scale", str(guidance_scale)])
            if "seed" in case.inputs:
                infer_parts.extend(["--seed", str(case.inputs["seed"])])
            if case.family == "ltx_video" and ctx.artifacts_dir:
                latent_path = Path(_case_artifact_dir(ctx.artifacts_dir, case.name)) / "initial_latents.raw"
                infer_parts.extend(["--initial-latents-raw", str(latent_path)])
        elif task_strategy == "prompted_segmentation":
            infer_parts = [
                ctx.binary_path, "segment-sam", bundle_path,
                "--image", str(image or ""),
                "--output", "/tmp/trtmc_masks",
                "--point-x", str(case.inputs.get("point_x", 0.5)),
                "--point-y", str(case.inputs.get("point_y", 0.5)),
            ]
            if not case.inputs.get("is_foreground", True):
                infer_parts.append("--background")
        elif task_strategy == "segmentation":
            infer_parts = [
                ctx.binary_path, "segment", bundle_path,
                "--image", str(image or ""),
                "--output", "/tmp/trtmc_segmentation.png",
            ]
        else:
            infer_parts = [
                ctx.binary_path, "run", bundle_path,
                "--prompt", _shell_quote(case.inputs.get("prompt", "")),
                "--max-new-tokens", str(case.inputs.get("max_new_tokens", 20)),
            ]
            if image:
                infer_parts.extend(["--image", str(image)])
        runtime_cli_python = ctx.runtime_cli_hf_python()
        if runtime_cli_python and task_strategy != "neural_operator":
            infer_parts.extend(["--hf-python", runtime_cli_python])
        repro["trt_inference"] = " ".join(infer_parts)

    # Rerun this exact test case
    rerun_parts = [
        "pytest", f"tests/test_e2e.py::test_e2e[{case.name}]", "-v",
        "--engine-dir", ctx.engine_dir,
        "--trtmc-binary", ctx.binary_path,
        "--hf-python", ctx.hf_python or "python",
    ]
    repro["rerun_test"] = " ".join(rerun_parts)

    profile_exports: list[str] = []
    for profile_name, python in (
        (ctx.build_profile, ctx.build_python_path()),
        (ctx.runtime_profile, ctx.runtime_python_path()),
        (ctx.reference_profile, ctx.reference_python_path()),
    ):
        if profile_name == "base" or not python:
            continue
        env_var = profile_env_var(profile_name)
        export_cmd = f"export {env_var}={python}"
        if export_cmd not in profile_exports:
            profile_exports.append(export_cmd)
    if profile_exports:
        repro["profile_env"] = "\n".join(profile_exports)

    # Rerun with forced rebuild
    rebuild_parts = list(rerun_parts) + ["--rebuild-engines"]
    repro["rerun_test_rebuild"] = " ".join(rebuild_parts)

    return repro


def _shell_quote(s: str) -> str:
    """Simple shell quoting for inclusion in repro commands."""
    if not s:
        return '""'
    # If string contains special chars, wrap in single quotes
    if any(c in s for c in " \t\n\"'\\$!&|;(){}[]<>?*~`#"):
        # Escape single quotes inside
        escaped = s.replace("'", "'\\''")
        return f"'{escaped}'"
    return s


def _csv_arg(value: Any) -> str:
    """Serialize numeric E2E inputs into the CLI CSV form used by solve()."""
    if isinstance(value, (list, tuple)):
        return ",".join(str(v) for v in value)
    return str(value)


def _manifest_build_method(build_args: dict[str, Any]) -> str | None:
    """Translate manifest build args to a CLI --method value.

    Returning None means "use the CLI default", which is now auto-selection.
    """
    backend = str(build_args.get("backend", build_args.get("method", "")) or "").lower()
    if backend in {"torchtrt", "torch_trt"} or build_args.get("torch_trt", False):
        return "torchtrt"
    if backend == "trt":
        return "trt"
    if backend == "auto":
        return None
    return None


def _load_build_timing(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _structured_build_detail_timing(build_info: dict[str, Any]) -> dict[str, float]:
    """Extract detailed build timings from the structured build timing JSON."""
    raw_timing = build_info.get("timing", {})
    if not isinstance(raw_timing, dict):
        return {}
    phases = raw_timing.get("phases", {})
    if not isinstance(phases, dict):
        return {}

    details: dict[str, float] = {}
    for key, value in phases.items():
        if value is None:
            continue
        try:
            details[str(key)] = float(value)
        except (TypeError, ValueError):
            continue

    return details


def _sum_timing(
    timing: dict[str, float],
    prefixes: tuple[str, ...],
    *,
    exclude: tuple[str, ...] = (),
    exclude_prefixes: tuple[str, ...] = (),
) -> float:
    return sum(
        float(value)
        for key, value in timing.items()
        if (
            value is not None
            and key not in exclude
            and not any(key.startswith(prefix) for prefix in exclude_prefixes)
            and any(key.startswith(prefix) for prefix in prefixes)
        )
    )


def _timing_label_key(label: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", label.strip().lower()).strip("_")
    return cleaned or "engine"


def _read_log_ref(ref: Any) -> str:
    if not isinstance(ref, str) or not ref:
        return ""
    raw_path = Path(ref)
    candidates = [raw_path]
    if raw_path.name:
        candidates.append(Path.cwd() / raw_path.name)
    for path in candidates:
        if path.is_file():
            try:
                return path.read_text(errors="replace")
            except OSError:
                return ""
    return ""


def _stage_text_blobs(output: StageOutput) -> list[str]:
    blobs: list[str] = []
    seen_blobs: set[str] = set()

    def append_blob(text: str) -> None:
        if not text:
            return
        key = text if len(text) < 10000 else f"{len(text)}:{text[:2000]}:{text[-2000:]}"
        if key in seen_blobs:
            return
        seen_blobs.add(key)
        blobs.append(text)

    def visit(value: Any) -> None:
        if isinstance(value, str):
            append_blob(value)
        elif isinstance(value, dict):
            log_bases: set[str] = set()
            for key, child in value.items():
                if key.endswith("_log") or key == "stderr_log":
                    log_text = _read_log_ref(child)
                    if log_text:
                        append_blob(log_text)
                    log_bases.add("stderr" if key == "stderr_log" else key[:-4])
            for key, child in value.items():
                if key.endswith("_log") or key == "stderr_log":
                    continue
                if key in log_bases or (key == "stderr_truncated" and "stderr" in log_bases):
                    continue
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    visit(output.metadata or {})
    visit(output.data or {})
    if output.text:
        blobs.append(str(output.text))
    return blobs


def _extract_labeled_timing(
    pattern: re.Pattern[str],
    blobs: list[str],
) -> dict[str, float]:
    timings: dict[str, float] = {}
    for text in blobs:
        for match in pattern.finditer(text or ""):
            try:
                seconds = float(match.group("ms")) / 1000.0
            except (TypeError, ValueError):
                continue
            key = _timing_label_key(match.group("label"))
            timings[key] = timings.get(key, 0.0) + seconds
    return timings


def _extract_cli_generation_timing(blobs: list[str]) -> float | None:
    for text in blobs:
        match = _TRTMC_TIMING_RE.search(text or "")
        if match is None:
            continue
        try:
            return float(match.group("total_ms")) / 1000.0
        except (TypeError, ValueError):
            continue
    return None


def _collect_trt_stage_timing(output: StageOutput, stage_name: str) -> dict[str, float]:
    timings: dict[str, float] = {}
    blobs = _stage_text_blobs(output)

    engine_components = _extract_labeled_timing(_TRTMC_ENGINE_TIMING_RE, blobs)
    if engine_components:
        total = sum(engine_components.values())
        timings[f"trt_engine_{stage_name}_s"] = total
        for label, value in engine_components.items():
            timings[f"trt_component_engine_{stage_name}_{label}_s"] = value
    else:
        engine_time_s = _extract_cli_generation_timing(blobs)
        if engine_time_s is None:
            engine_time_s = _extract_trt_engine_time_s(output)
        if engine_time_s is not None:
            timings[f"trt_engine_{stage_name}_s"] = engine_time_s

    load_components = _extract_labeled_timing(_TRTMC_LOAD_TIMING_RE, blobs)
    if load_components:
        total = sum(load_components.values())
        timings[f"trt_load_deserialize_{stage_name}_s"] = total
        for label, value in load_components.items():
            timings[f"trt_component_load_deserialize_{stage_name}_{label}_s"] = value
    else:
        load_deserialize_s = _extract_trt_load_deserialize_time_s(output)
        if load_deserialize_s is not None:
            timings[f"trt_load_deserialize_{stage_name}_s"] = load_deserialize_s

    return timings


def _extract_trt_engine_time_s(output: StageOutput) -> float | None:
    metadata = output.metadata or {}
    candidates: list[Any] = [
        metadata.get("trt_engine_s"),
        metadata.get("engine_s"),
    ]
    cpp_meta = metadata.get("cpp")
    if isinstance(cpp_meta, dict):
        candidates.extend([
            cpp_meta.get("trt_engine_s"),
            cpp_meta.get("engine_s"),
        ])

    for value in candidates:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _extract_trt_load_deserialize_time_s(output: StageOutput) -> float | None:
    metadata = output.metadata or {}
    candidates: list[Any] = [
        metadata.get("trt_load_deserialize_s"),
        metadata.get("load_deserialize_s"),
    ]
    cpp_meta = metadata.get("cpp")
    if isinstance(cpp_meta, dict):
        candidates.extend([
            cpp_meta.get("trt_load_deserialize_s"),
            cpp_meta.get("load_deserialize_s"),
        ])

    for value in candidates:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _build_detailed_timing(
    timing: dict[str, float],
    build_info: dict[str, Any] | None,
) -> dict[str, float]:
    """Build normalized timing categories for the HTML report."""
    details = _structured_build_detail_timing(build_info or {})

    inference = _sum_timing(timing, ("trt_engine_",))
    if inference:
        details["inference_s"] = inference

    load_deserialize = _sum_timing(timing, ("trt_load_deserialize_",))
    if load_deserialize:
        details["trt_load_deserialization_s"] = load_deserialize

    for key, value in timing.items():
        if not key.startswith(("trt_component_engine_", "trt_component_load_deserialize_")):
            continue
        if value is None:
            continue
        try:
            details[key] = float(value)
        except (TypeError, ValueError):
            continue

    trt_validation = _sum_timing(
        timing,
        ("trt_",),
        exclude=("trt_compile_s", "trt_build_s"),
        exclude_prefixes=("trt_engine_", "trt_load_deserialize_", "trt_component_"),
    )
    if trt_validation:
        details["trt_validation_s"] = trt_validation

    reference = _sum_timing(timing, ("ref_",))
    if reference:
        details["reference_s"] = reference

    comparison = _sum_timing(timing, ("compare_", "contract_"))
    if comparison:
        details["comparison_s"] = comparison

    preflight = timing.get("preflight_s")
    if preflight is not None:
        details["preflight_s"] = float(preflight)

    return details


class E2EOrchestrator:
    """Coordinates the full E2E lifecycle for one model case.

    Depends only on contracts — all concrete implementations are
    resolved at runtime via the registry.
    """

    def run(self, case: E2ECase, ctx: RunContext) -> E2EResult:
        """Execute the full E2E lifecycle for a single model case.

        Returns an E2EResult with all stage outcomes, timing, and artifacts.
        """
        from .artifact_sink import FileArtifactSink

        timestamp = datetime.now(timezone.utc).isoformat()
        timing: dict[str, float] = {}

        # Initialize artifact sink
        artifacts_dir = ctx.artifacts_dir or "/tmp/e2e_artifacts"
        sink = FileArtifactSink(artifacts_dir, case)

        # Collect environment fingerprint
        env_fp = sink.ensure_env_fingerprint(ctx)

        # Partial skip: TRT inference still runs; only HF reference/comparison
        # is skipped. Driven by manifest `skip_comparison` (see manifest_loader).
        # `skip_reason` is kept as a legacy fallback — when the manifest-level
        # `skip` is set, test_e2e.py short-circuits with pytest.skip before
        # this orchestrator even runs, so this path is normally unreachable.
        skip_reason = (
            case.metadata.get("skip_comparison_reason")
            or case.metadata.get("skip_reason")
        )

        # 1. Preflight
        t0 = time.monotonic()
        preflight_ok, preflight_details = run_preflight(case, ctx)
        timing["preflight_s"] = time.monotonic() - t0

        if not preflight_ok:
            result = E2EResult(
                case_name=case.name,
                status=E2EStatus.SKIP.value,
                failure_type=FailureType.PRECHECK_FAIL.value,
                oracle_level=case.oracle_level,
                stages={},
                determinism={"preflight": preflight_details},
                timing=timing,
                detailed_timing=_build_detailed_timing(timing, {}),
                env_fingerprint=env_fp,
                timestamp=timestamp,
            )
            sink.finalize(result)
            return result

        # 2. Resolve or build bundle
        bundle_path, build_time, build_err, build_info = _resolve_bundle(case, ctx)
        if build_time is not None:
            timing["bundle_build_s"] = build_time

        # Log build subprocess output
        if build_info:
            sink.log_command(
                command=build_info.get("command", []),
                rc=build_info.get("returncode", -1),
                stdout=build_info.get("stdout", ""),
                stderr=build_info.get("stderr", ""),
                label="build",
            )
            timing_path = build_info.get("timing_path")
            if isinstance(timing_path, str) and timing_path:
                try:
                    sink.register_artifact(
                        "build_timing_json",
                        str(Path(timing_path).relative_to(sink.base_dir)),
                    )
                except ValueError:
                    sink.register_artifact("build_timing_json", timing_path)

        if bundle_path is None:
            repro = _build_repro_commands(case, ctx, None, build_info)
            result = E2EResult(
                case_name=case.name,
                status=E2EStatus.FAIL.value,
                failure_type=FailureType.BUILD_FAIL.value,
                oracle_level=case.oracle_level,
                stages={},
                determinism={"build_error": build_err},
                timing=timing,
                detailed_timing=_build_detailed_timing(timing, build_info),
                env_fingerprint=env_fp,
                timestamp=timestamp,
                repro_commands=repro,
            )
            sink.finalize(result)
            return result

        # Update context with resolved bundle path
        ctx_with_bundle = RunContext(
            case=case,
            artifacts_dir=artifacts_dir,
            binary_path=ctx.binary_path,
            hf_python=ctx.hf_python,
            build_python=ctx.build_python,
            runtime_python=ctx.runtime_python,
            reference_python=ctx.reference_python,
            build_profile=ctx.build_profile,
            runtime_profile=ctx.runtime_profile,
            reference_profile=ctx.reference_profile,
            ld_library_path=ctx.ld_library_path,
            engine_dir=ctx.engine_dir,
            rebuild=ctx.rebuild,
            verbose=ctx.verbose,
        )

        # Resolve runner, reference, comparator, and contract plugin
        runner = get_runner(case.task_strategy)
        reference = get_reference(case.reference_backend)
        comparator = get_comparator(case.task_strategy)
        contract_plugin = get_contract_plugin(case.reference_family) if case.reference_family else None
        threshold = _resolve_threshold(case)

        # Apply contract plugin configuration to the case metadata
        if contract_plugin is not None:
            try:
                ref_config = contract_plugin.configure_reference(case)
                if ref_config:
                    case.metadata["contract_config"] = ref_config
            except Exception as e:
                logger.warning("Contract plugin configure_reference failed: %s", e)

            # Expose runtime paths so plugins can invoke auxiliary binaries
            # and reference Python tools for contract checks.
            case.metadata["_ctx"] = {
                "engine_dir": ctx.engine_dir,
                "binary_path": ctx.binary_path,
                "hf_python": ctx.runtime_cli_hf_python(),
                "runtime_python": ctx.runtime_python_path(),
                "reference_python": ctx.reference_python_path(),
                "artifacts_dir": artifacts_dir,
            }

        # 3. Execute stages
        stage_results: dict[str, CompareResult] = {}
        all_stages_pass = True
        required_validation_skipped = False
        baseline_trt_outputs: dict[str, StageOutput] = {}

        for stage in case.stages:
            stage_name = stage.name

            # Lane filtering: skip stages not in this case's CI lane
            if case.ci_lane and stage.ci_lanes:
                if case.ci_lane not in stage.ci_lanes:
                    logger.info(
                        "Skipping stage %s for lane %s (stage lanes: %s)",
                        stage_name, case.ci_lane, stage.ci_lanes,
                    )
                    stage_results[stage_name] = CompareResult(
                        stage_name=stage_name,
                        status=StageStatus.SKIPPED.value,
                        message=f"Skipped: stage not in CI lane {case.ci_lane}",
                    )
                    continue

            # TRT run
            trt_output: StageOutput | None = None
            if runner is not None:
                t0 = time.monotonic()
                try:
                    trt_output = runner.run_stage(case, stage, ctx_with_bundle)
                    timing[f"trt_{stage_name}_s"] = time.monotonic() - t0
                    timing.update(_collect_trt_stage_timing(trt_output, stage_name))
                    sink.write_stage_output(stage_name, trt_output, prefix="trt")
                    _log_stage_subprocess(sink, stage_name, trt_output, "trt")
                    _auto_register_artifacts(sink, trt_output, "trt")
                    runtime_path_error = _validate_trt_runtime_path(
                        case, ctx_with_bundle, trt_output)
                    if runtime_path_error is not None:
                        stage_results[stage_name] = CompareResult(
                            stage_name=stage_name,
                            status=StageStatus.ERROR.value,
                            message=f"TRT run failed: {runtime_path_error}",
                        )
                        if stage.required:
                            all_stages_pass = False
                        continue
                except Exception as e:
                    timing[f"trt_{stage_name}_s"] = time.monotonic() - t0
                    tb = traceback.format_exc()
                    logger.error("TRT run failed for stage %s: %s\n%s", stage_name, e, tb)
                    stage_results[stage_name] = CompareResult(
                        stage_name=stage_name,
                        status=StageStatus.ERROR.value,
                        message=f"TRT run failed: {e}\n{tb}",
                    )
                    if stage.required:
                        all_stages_pass = False
                    continue
            else:
                # No runner registered — produce no-op output
                trt_output = StageOutput(
                    stage_name=stage_name,
                    data={"status": "no_runner_registered"},
                    metadata={"runner": "none"},
                )
                sink.write_stage_output(stage_name, trt_output, prefix="trt")
                logger.warning(
                    "No runner registered for strategy %s, stage %s",
                    case.task_strategy, stage_name,
                )

            # Store baseline TRT output for determinism reruns
            if trt_output is not None and runner is not None:
                baseline_trt_outputs[stage_name] = trt_output

            # Reference run (skipped when skip_reason is set)
            ref_output: StageOutput | None = None
            if skip_reason:
                logger.info(
                    "Skipping reference for %s stage %s: %s",
                    case.name, stage_name, skip_reason,
                )
            elif reference is not None:
                t0 = time.monotonic()
                try:
                    ref_output = reference.run_stage(case, stage, ctx_with_bundle)
                    timing[f"ref_{stage_name}_s"] = time.monotonic() - t0
                    sink.write_stage_output(stage_name, ref_output, prefix="ref")
                    _log_stage_subprocess(sink, stage_name, ref_output, "ref")
                    _auto_register_artifacts(sink, ref_output, "ref")
                except Exception as e:
                    timing[f"ref_{stage_name}_s"] = time.monotonic() - t0
                    tb = traceback.format_exc()
                    logger.error("Reference run failed for stage %s: %s\n%s", stage_name, e, tb)
                    stage_results[stage_name] = CompareResult(
                        stage_name=stage_name,
                        status=StageStatus.ERROR.value,
                        message=f"Reference run failed: {e}\n{tb}",
                    )
                    if stage.required:
                        all_stages_pass = False
                    continue
            else:
                logger.warning(
                    "No reference backend registered for %s, stage %s — skipping comparison",
                    case.reference_backend, stage_name,
                )

            # Comparison: contract plugin (acceptance) then numeric comparator (nightly)
            if trt_output is not None and ref_output is not None:
                compare_result: CompareResult | None = None

                # Contract verification via plugin (if available)
                if contract_plugin is not None:
                    t0 = time.monotonic()
                    try:
                        compare_result = contract_plugin.verify(
                            trt_output, ref_output, case, threshold)
                        timing[f"contract_{stage_name}_s"] = time.monotonic() - t0
                    except Exception as e:
                        timing[f"contract_{stage_name}_s"] = time.monotonic() - t0
                        tb = traceback.format_exc()
                        compare_result = CompareResult(
                            stage_name=stage_name,
                            status=StageStatus.ERROR.value,
                            message=f"Contract verification failed: {e}\n{tb}",
                        )

                # Numeric parity via existing comparator (fallback or nightly supplement)
                if compare_result is None and comparator is not None:
                    t0 = time.monotonic()
                    try:
                        compare_result = comparator.compare(
                            trt_output, ref_output, threshold, stage)
                        timing[f"compare_{stage_name}_s"] = time.monotonic() - t0
                    except Exception as e:
                        timing[f"compare_{stage_name}_s"] = time.monotonic() - t0
                        tb = traceback.format_exc()
                        compare_result = CompareResult(
                            stage_name=stage_name,
                            status=StageStatus.ERROR.value,
                            message=f"Comparison failed: {e}\n{tb}",
                        )

                if compare_result is not None:
                    stage_results[stage_name] = compare_result
                    sink.write_compare(stage_name, compare_result)
                    if compare_result.status == StageStatus.SKIPPED.value and stage.required:
                        required_validation_skipped = True
                        all_stages_pass = False
                    elif not compare_result.passed and stage.required:
                        all_stages_pass = False
                else:
                    stage_results[stage_name] = CompareResult(
                        stage_name=stage_name,
                        status=StageStatus.SKIPPED.value,
                        message="TRT and reference ran (no contract plugin or comparator)",
                    )
                    if stage.required:
                        required_validation_skipped = True
                        all_stages_pass = False
            elif trt_output is not None and ref_output is None:
                # No reference — record as skipped
                stage_results[stage_name] = CompareResult(
                    stage_name=stage_name,
                    status=StageStatus.SKIPPED.value,
                    message="TRT run succeeded (no reference available)",
                )
                if stage.required:
                    required_validation_skipped = True
                    all_stages_pass = False

        # 4. Determinism reruns (if configured)
        determinism_results: dict[str, Any] = {}
        reruns = case.determinism.get("reruns", 0)
        if reruns > 0 and runner is not None:
            determinism_results["reruns_requested"] = reruns
            determinism_results["per_stage"] = {}
            determinism_ok = True

            for stage in case.stages:
                if not stage.required:
                    continue
                stage_name = stage.name
                baseline = baseline_trt_outputs.get(stage_name)
                if baseline is None:
                    continue

                rerun_outputs: list[dict[str, Any]] = []
                for i in range(reruns):
                    t0 = time.monotonic()
                    try:
                        rerun_out = runner.run_stage(case, stage, ctx_with_bundle)
                        elapsed = time.monotonic() - t0
                        timing[f"determinism_{stage_name}_rerun_{i}_s"] = elapsed

                        # Compare text output
                        text_match = (
                            rerun_out.text == baseline.text
                        ) if baseline.text is not None else True

                        # Compare data keys
                        data_match = True
                        for key in baseline.data:
                            if key not in rerun_out.data:
                                data_match = False
                                break
                            try:
                                if baseline.data[key] != rerun_out.data[key]:
                                    data_match = False
                                    break
                            except (TypeError, ValueError):
                                pass  # non-comparable types, skip

                        matched = text_match and data_match
                        rerun_outputs.append({
                            "rerun_index": i,
                            "text_match": text_match,
                            "data_match": data_match,
                            "matched": matched,
                            "timing_s": elapsed,
                        })
                        if not matched:
                            determinism_ok = False

                        sink.log_command(
                            command=[f"determinism_rerun_{stage_name}_{i}"],
                            rc=0 if matched else 1,
                            stdout=rerun_out.text or "",
                            stderr="",
                            label=f"determinism_{stage_name}_rerun_{i}",
                        )
                    except Exception as e:
                        elapsed = time.monotonic() - t0
                        timing[f"determinism_{stage_name}_rerun_{i}_s"] = elapsed
                        tb = traceback.format_exc()
                        logger.error(
                            "Determinism rerun %d for stage %s failed: %s\n%s",
                            i, stage_name, e, tb,
                        )
                        rerun_outputs.append({
                            "rerun_index": i,
                            "matched": False,
                            "error": str(e),
                            "timing_s": elapsed,
                        })
                        determinism_ok = False

                        sink.log_command(
                            command=[f"determinism_rerun_{stage_name}_{i}"],
                            rc=1,
                            stdout="",
                            stderr=f"{e}\n{tb}",
                            label=f"determinism_{stage_name}_rerun_{i}",
                        )

                determinism_results["per_stage"][stage_name] = rerun_outputs

            determinism_results["status"] = (
                "deterministic" if determinism_ok else "non_deterministic"
            )
            if not determinism_ok:
                all_stages_pass = False

        # 5. Build reproducibility commands
        repro = _build_repro_commands(case, ctx, bundle_path, build_info)

        # 6. Aggregate result
        if all_stages_pass:
            status = E2EStatus.PASS.value
            failure_type = None
        elif required_validation_skipped and not any(
            cr.status in (StageStatus.FAILED.value, StageStatus.ERROR.value)
            for cr in stage_results.values()
        ):
            status = E2EStatus.SKIP.value
            failure_type = None
        else:
            status = E2EStatus.FAIL.value
            # Determine failure type from stage results
            failure_type = FailureType.COMPARE_FAIL.value
            for cr in stage_results.values():
                if not cr.passed:
                    if "TRT run failed" in cr.message:
                        failure_type = FailureType.TRT_RUN_FAIL.value
                        break
                    elif "Reference run failed" in cr.message:
                        failure_type = FailureType.REFERENCE_RUN_FAIL.value
                        break
            # Override with determinism failure if that was the cause
            if determinism_results.get("status") == "non_deterministic":
                failure_type = FailureType.DETERMINISM_FAIL.value

        result = E2EResult(
            case_name=case.name,
            status=status,
            failure_type=failure_type,
            oracle_level=case.oracle_level,
            stages=stage_results,
            determinism=determinism_results,
            timing=timing,
            detailed_timing=_build_detailed_timing(timing, build_info),
            env_fingerprint=env_fp,
            timestamp=timestamp,
            repro_commands=repro,
        )

        # 7. Finalize artifacts
        try:
            sink.finalize(result)
        except Exception as e:
            logger.error("Failed to finalize artifacts: %s", e)
            result.failure_type = FailureType.ARTIFACT_WRITE_FAIL.value

        return result
