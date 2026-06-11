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

import inspect
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import (
    CompareResult,
    E2ECase,
    E2EResult,
    E2EStatus,
    FailureType,
    PluginRuntimeContext,
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
    "image_classification",
    "object_detection",
    "neural_operator",
    "text_to_audio",
    "speech_to_text",
    "speech_to_text_rnnt",
    "speech_to_speech",
    "omni_multimodal",
    "diffusion",
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
                    "import importlib; "
                    f"importlib.import_module({module!r})"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if result.returncode == 0:
            return True, f"Module {module} importable in {phase} profile"
        detail = (result.stderr or result.stdout).strip().splitlines()
        suffix = f": {detail[-1]}" if detail else ""
        return False, f"Module {module} not importable in {phase} profile{suffix}"
    except Exception as exc:
        return False, f"Module check failed in {phase} profile: {exc}"


def _check_command_available(ctx: RunContext, req: PreflightRequirement) -> tuple[bool, str]:
    """Check that a required command is available on PATH."""
    command = str(req.args.get("command", "") or "")
    if not command:
        return False, "Command name not specified"
    path = shutil.which(command)
    if path:
        return True, f"Command found: {path}"
    return False, f"Command not found on PATH: {command}"


def _visible_gpu_count() -> int | None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible and visible.lower() not in {"all", "void", "none"}:
        devices = [part.strip() for part in visible.split(",") if part.strip()]
        return len(devices)
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return len([line for line in result.stdout.splitlines() if line.strip()])


def _check_gpu_count_min(ctx: RunContext, req: PreflightRequirement) -> tuple[bool, str]:
    """Check that enough GPUs are visible for a multi-device E2E case."""
    required = int(req.args.get("count", req.args.get("min_count", 0)) or 0)
    if required <= 0:
        return False, "GPU count requirement not specified"
    count = _visible_gpu_count()
    if count is None:
        return False, "GPU count check failed"
    if count >= required:
        return True, f"Visible GPUs: {count} >= {required}"
    return False, f"Visible GPUs: {count} < {required} required"


_PREFLIGHT_CHECKERS = {
    "binary_exists": _check_binary_exists,
    "gpu_memory_min_gb": _check_gpu_memory,
    "hf_auth_token_present": _check_hf_auth,
    "asset_exists": _check_asset_exists,
    "python_module_available": _check_python_module,
    "command_available": _check_command_available,
    "gpu_count_min": _check_gpu_count_min,
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
    _append_manifest_build_args(cmd, build_args)
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
    edit_condition_image = build_args.get("edit_condition_image")
    if (
        not edit_condition_image
        and case.family == "qwen_image"
        and str(case.metadata.get("task_mode", "")).lower() == "edit"
    ):
        edit_condition_image = case.inputs.get("image")
    if edit_condition_image:
        edit_condition_path = Path(str(edit_condition_image))
        if not edit_condition_path.is_absolute():
            edit_condition_path = Path(__file__).resolve().parents[2] / edit_condition_path
        env["TRTMC_QWEN_IMAGE_EDIT_CONDITION_IMAGE"] = str(edit_condition_path)
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
        command_names = {Path(part).name for part in command_argv if part}
        command_contains_binary = bool(
            ctx.binary_path and (
                ctx.binary_path in command_argv
                or Path(ctx.binary_path).name in command_names
            )
        )
        if ctx.binary_path and (executable == ctx.binary_path or command_contains_binary):
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
    _append_manifest_build_args(build_parts, case.metadata.get("build_args", {}))
    if case.metadata.get("trust_remote_code"):
        build_parts.append("--trust-remote-code")
    repro["build_bundle"] = " ".join(build_parts)

    # TRT inference command (task-specific C++ binary entrypoint)
    if bundle_path and ctx.binary_path:
        image = (case.inputs.get("image") or case.inputs.get("test_image")
                 or case.inputs.get("image_path"))
        task_strategy = case.task_strategy or ""
        if task_strategy == "diffusion_text_generation":
            infer_parts = [
                ctx.binary_path, "run", bundle_path,
                "--prompt", _shell_quote(
                    case.inputs.get("prompt")
                    or case.inputs.get("source_text")
                    or case.inputs.get("condition_text")
                    or ""
                ),
                "--output", "/tmp/trtmc_elf_samples.jsonl",
            ]
            if "max_new_tokens" in case.inputs:
                infer_parts.extend(["--max-new-tokens", str(case.inputs["max_new_tokens"])])
            if int(case.inputs.get("num_samples", 1)) > 1:
                infer_parts.extend(["--num-samples", str(case.inputs["num_samples"])])
            num_steps = case.inputs.get("num_sampling_steps", case.inputs.get("num_steps"))
            if num_steps is not None:
                infer_parts.extend(["--num-steps", str(num_steps)])
            self_cond = case.inputs.get("self_cond_cfg_scale", case.inputs.get("guidance_scale"))
            if self_cond is not None:
                infer_parts.extend(["--guidance-scale", str(self_cond)])
            if "cfg_scale" in case.inputs:
                infer_parts.extend(["--cfg-scale", str(case.inputs["cfg_scale"])])
            if "sde_gamma" in case.inputs:
                infer_parts.extend(["--sde-gamma", str(case.inputs["sde_gamma"])])
            if "seed" in case.inputs:
                infer_parts.extend(["--seed", str(case.inputs["seed"])])
            condition_latents = (
                case.inputs.get("condition_latents_raw")
                or case.inputs.get("condition_latents_path")
            )
            condition_mask = (
                case.inputs.get("condition_mask_raw")
                or case.inputs.get("condition_mask_path")
            )
            if condition_latents:
                infer_parts.extend(["--condition-latents-raw", str(condition_latents)])
            if condition_mask:
                infer_parts.extend(["--condition-mask-raw", str(condition_mask)])
        elif task_strategy == "neural_operator":
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
            is_qwen_image = (
                case.runtime_strategy == "diffusion_qwen_image"
                or case.family == "qwen_image"
            )
            if is_qwen_image:
                infer_parts = [
                    ctx.binary_path, "run", bundle_path,
                    "--prompt", _shell_quote(case.inputs.get("prompt", case.inputs.get("test_prompt", ""))),
                    "--output", "/tmp/trtmc_qwen_image/output.png",
                    "--num-inference-steps", str(case.inputs.get("num_inference_steps", 20)),
                ]
                negative_prompt = case.inputs.get("negative_prompt")
                if negative_prompt is not None:
                    infer_parts.extend(["--negative-prompt", _shell_quote(str(negative_prompt))])
                cfg_scale = case.inputs.get("cfg_scale")
                if cfg_scale is None:
                    cfg_scale = case.inputs.get("guidance_scale")
                if cfg_scale is not None:
                    infer_parts.extend(["--cfg-scale", str(cfg_scale)])
                height = case.inputs.get("height") or case.inputs.get("image_height")
                if height is not None:
                    infer_parts.extend(["--height", str(height)])
                width = case.inputs.get("width") or case.inputs.get("image_width")
                if width is not None:
                    infer_parts.extend(["--width", str(width)])
                if "seed" in case.inputs:
                    infer_parts.extend(["--seed", str(case.inputs["seed"])])
                # Shared-initial-latents path: the runner pre-computes the same
                # raw bytes the HF reference will consume. Mirrors the LTX
                # plumbing immediately below.
                if ctx.artifacts_dir:
                    qi_latent_path = Path(
                        _case_artifact_dir(ctx.artifacts_dir, case.name)
                    ) / "initial_latents.raw"
                    infer_parts.extend(["--initial-latents-raw", str(qi_latent_path)])
            else:
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
            is_sam3 = (
                case.family == "sam3"
                or case.reference_family == "prompted_segmentation_sam3"
            )
            infer_parts = [
                ctx.binary_path, "segment-sam", bundle_path,
                "--image", str(image or ""),
                "--output", "/tmp/trtmc_masks",
            ]
            if is_sam3:
                prompt = (
                    case.inputs.get("prompt")
                    or case.inputs.get("text_prompt")
                    or case.metadata.get("text_prompt")
                    or ""
                )
                infer_parts.extend(["--prompt", _shell_quote(str(prompt))])
            else:
                infer_parts.extend([
                    "--point-x", str(case.inputs.get("point_x", 0.5)),
                    "--point-y", str(case.inputs.get("point_y", 0.5)),
                ])
            if not is_sam3 and not case.inputs.get("is_foreground", True):
                infer_parts.append("--background")
        elif task_strategy == "segmentation":
            infer_parts = [
                ctx.binary_path, "segment", bundle_path,
                "--image", str(image or ""),
                "--output", "/tmp/trtmc_segmentation.png",
            ]
        elif task_strategy == "image_classification":
            infer_parts = [
                ctx.binary_path, "classify", bundle_path,
                "--image", str(image or ""),
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
        infer_parts = _wrap_distributed_repro_command(infer_parts, case)
        repro["trt_inference"] = " ".join(infer_parts)

    # Rerun this exact test case
    rerun_parts = [
        "pytest", f"tests/test_e2e.py::test_e2e[{case.name}]", "-v",
        "--engine-dir", ctx.engine_dir,
        "--trtmc-binary", ctx.binary_path,
        "--hf-python", ctx.hf_python or "python",
    ]
    if _distributed_runtime_config(case):
        rerun_parts.append("--multi-device-only")
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
    if backend == "trt":
        return "trt"
    if backend == "auto":
        return None
    return None


def _manifest_tensor_parallel_size(build_args: dict[str, Any]) -> int | None:
    if not isinstance(build_args, dict):
        return None
    parallel = build_args.get("parallel", {})
    if isinstance(parallel, dict):
        value = parallel.get("tp_size", parallel.get("tensor_parallel_size"))
        mode = str(parallel.get("mode", "") or "").lower()
    else:
        value = build_args.get("tp_size", build_args.get("tensor_parallel_size"))
        mode = str(build_args.get("parallel_mode", "") or "").lower()
    if value is None:
        return None
    try:
        tp_size = int(value)
    except (TypeError, ValueError):
        return None
    if tp_size > 1 or mode == "tensor_parallel":
        return tp_size
    return None


def _append_manifest_build_args(cmd: list[str], build_args: dict[str, Any]) -> None:
    tp_size = _manifest_tensor_parallel_size(build_args)
    if tp_size is not None and tp_size > 1:
        cmd.extend(["--tp-size", str(tp_size)])


def _distributed_runtime_config(case: E2ECase) -> dict[str, Any]:
    config = case.metadata.get("distributed_runtime", {})
    return config if isinstance(config, dict) and config.get("enabled") else {}


def _wrap_distributed_repro_command(cmd: list[str], case: E2ECase) -> list[str]:
    config = _distributed_runtime_config(case)
    if not config:
        return cmd
    launcher = str(config.get("launcher", "mpirun") or "mpirun")
    world_size = int(config.get("world_size", config.get("tp_size", 2)) or 2)
    launcher_args = config.get("launcher_args")
    if isinstance(launcher_args, list):
        return [launcher] + [str(arg) for arg in launcher_args] + cmd
    return [launcher, "--tag-output", "-np", str(world_size)] + cmd


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


def _build_plugin_runtime_context(ctx: RunContext) -> PluginRuntimeContext:
    """Create the typed runtime context exposed to contract plugins."""
    return PluginRuntimeContext(
        engine_dir=ctx.engine_dir,
        binary_path=ctx.binary_path,
        hf_python=ctx.runtime_cli_hf_python(),
        runtime_python=ctx.runtime_python_path(),
        reference_python=ctx.reference_python_path(),
        artifacts_dir=ctx.artifacts_dir,
    )


def _plugin_accepts_runtime_context(contract_plugin: Any) -> bool:
    """Return True when a contract plugin opts into PluginRuntimeContext."""
    try:
        parameters = inspect.signature(contract_plugin.verify).parameters
    except (TypeError, ValueError):
        return False

    return (
        "runtime_context" in parameters
        or any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in parameters.values()
        )
    )


def _verify_contract_plugin(
    contract_plugin: Any,
    trt_output: StageOutput,
    ref_output: StageOutput,
    case: E2ECase,
    threshold: ThresholdProfile,
    runtime_context: PluginRuntimeContext,
) -> CompareResult:
    """Call a contract plugin while preserving legacy verify signatures."""
    if _plugin_accepts_runtime_context(contract_plugin):
        return contract_plugin.verify(
            trt_output,
            ref_output,
            case,
            threshold,
            runtime_context=runtime_context,
        )
    return contract_plugin.verify(trt_output, ref_output, case, threshold)


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


@dataclass
class _RunState:
    case: E2ECase
    original_ctx: RunContext
    ctx: RunContext
    sink: Any
    timestamp: str
    artifacts_dir: str
    env_fingerprint: dict[str, Any]
    timing: dict[str, float] = field(default_factory=dict)
    skip_reason: str | None = None
    bundle_path: str | None = None
    build_info: dict[str, Any] = field(default_factory=dict)
    runner: Any = None
    reference: Any = None
    comparator: Any = None
    contract_plugin: Any = None
    threshold: ThresholdProfile | None = None
    plugin_runtime_context: PluginRuntimeContext | None = None
    stage_results: dict[str, CompareResult] = field(default_factory=dict)
    baseline_trt_outputs: dict[str, StageOutput] = field(default_factory=dict)
    determinism_results: dict[str, Any] = field(default_factory=dict)
    all_stages_pass: bool = True
    required_validation_skipped: bool = False


class E2EOrchestrator:
    """Coordinates the full E2E lifecycle for one model case.

    Depends only on contracts — all concrete implementations are
    resolved at runtime via the registry.
    """

    def run(self, case: E2ECase, ctx: RunContext) -> E2EResult:
        """Execute the full E2E lifecycle for a single model case.

        Returns an E2EResult with all stage outcomes, timing, and artifacts.
        """
        state = self._create_run_state(case, ctx)

        terminal_result = self._preflight_phase(state)
        if terminal_result is not None:
            return terminal_result

        terminal_result = self._bundle_resolution_phase(state)
        if terminal_result is not None:
            return terminal_result

        self._runtime_setup_phase(state)
        self._stage_execution_phase(state)
        self._determinism_phase(state)
        return self._finalization_phase(state)

    def _create_run_state(self, case: E2ECase, ctx: RunContext) -> _RunState:
        from .artifact_sink import FileArtifactSink

        timestamp = datetime.now(timezone.utc).isoformat()
        artifacts_dir = ctx.artifacts_dir or "/tmp/e2e_artifacts"
        sink = FileArtifactSink(artifacts_dir, case)
        env_fp = sink.ensure_env_fingerprint(ctx)
        skip_reason = (
            case.metadata.get("skip_comparison_reason")
            or case.metadata.get("skip_reason")
        )

        return _RunState(
            case=case,
            original_ctx=ctx,
            ctx=ctx,
            sink=sink,
            timestamp=timestamp,
            artifacts_dir=artifacts_dir,
            env_fingerprint=env_fp,
            skip_reason=skip_reason,
        )

    def _preflight_phase(self, state: _RunState) -> E2EResult | None:
        t0 = time.monotonic()
        preflight_ok, preflight_details = run_preflight(state.case, state.ctx)
        state.timing["preflight_s"] = time.monotonic() - t0

        if preflight_ok:
            return None

        result = E2EResult(
            case_name=state.case.name,
            status=E2EStatus.SKIP.value,
            failure_type=FailureType.PRECHECK_FAIL.value,
            oracle_level=state.case.oracle_level,
            stages={},
            determinism={"preflight": preflight_details},
            timing=state.timing,
            detailed_timing=_build_detailed_timing(state.timing, {}),
            env_fingerprint=state.env_fingerprint,
            timestamp=state.timestamp,
        )
        state.sink.finalize(result)
        return result

    def _bundle_resolution_phase(self, state: _RunState) -> E2EResult | None:
        bundle_path, build_time, build_err, build_info = _resolve_bundle(
            state.case, state.ctx)
        state.bundle_path = bundle_path
        state.build_info = build_info

        if build_time is not None:
            state.timing["bundle_build_s"] = build_time

        self._record_build_outputs(state)
        if bundle_path is not None:
            return None

        repro = _build_repro_commands(state.case, state.ctx, None, build_info)
        result = E2EResult(
            case_name=state.case.name,
            status=E2EStatus.FAIL.value,
            failure_type=FailureType.BUILD_FAIL.value,
            oracle_level=state.case.oracle_level,
            stages={},
            determinism={"build_error": build_err},
            timing=state.timing,
            detailed_timing=_build_detailed_timing(state.timing, build_info),
            env_fingerprint=state.env_fingerprint,
            timestamp=state.timestamp,
            repro_commands=repro,
        )
        state.sink.finalize(result)
        return result

    def _record_build_outputs(self, state: _RunState) -> None:
        build_info = state.build_info
        if not build_info:
            return

        state.sink.log_command(
            command=build_info.get("command", []),
            rc=build_info.get("returncode", -1),
            stdout=build_info.get("stdout", ""),
            stderr=build_info.get("stderr", ""),
            label="build",
        )
        timing_path = build_info.get("timing_path")
        if isinstance(timing_path, str) and timing_path:
            try:
                state.sink.register_artifact(
                    "build_timing_json",
                    str(Path(timing_path).relative_to(state.sink.base_dir)),
                )
            except ValueError:
                state.sink.register_artifact("build_timing_json", timing_path)

    def _runtime_setup_phase(self, state: _RunState) -> None:
        ctx = state.original_ctx
        state.ctx = RunContext(
            case=state.case,
            artifacts_dir=state.artifacts_dir,
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
        state.plugin_runtime_context = _build_plugin_runtime_context(state.ctx)

        state.runner = get_runner(state.case.task_strategy)
        state.reference = get_reference(state.case.reference_backend)
        state.comparator = get_comparator(state.case.task_strategy)
        state.contract_plugin = (
            get_contract_plugin(state.case.reference_family)
            if state.case.reference_family else None
        )
        state.threshold = _resolve_threshold(state.case)
        self._configure_contract_plugin(state)

    def _configure_contract_plugin(self, state: _RunState) -> None:
        if state.contract_plugin is None:
            return

        try:
            ref_config = state.contract_plugin.configure_reference(state.case)
            if ref_config:
                state.case.metadata["contract_config"] = ref_config
        except Exception as e:
            logger.warning("Contract plugin configure_reference failed: %s", e)

    def _stage_execution_phase(self, state: _RunState) -> None:
        for stage in state.case.stages:
            if self._skip_stage_for_lane(state, stage):
                continue

            trt_output = self._run_trt_stage(state, stage)
            if trt_output is None:
                continue

            if state.runner is not None:
                state.baseline_trt_outputs[stage.name] = trt_output

            ref_output = self._run_reference_stage(state, stage)
            if stage.name in state.stage_results:
                continue

            self._comparison_phase(state, stage, trt_output, ref_output)

    def _skip_stage_for_lane(self, state: _RunState, stage: Any) -> bool:
        if not state.case.ci_lane or not stage.ci_lanes:
            return False
        if state.case.ci_lane in stage.ci_lanes:
            return False

        logger.info(
            "Skipping stage %s for lane %s (stage lanes: %s)",
            stage.name, state.case.ci_lane, stage.ci_lanes,
        )
        state.stage_results[stage.name] = CompareResult(
            stage_name=stage.name,
            status=StageStatus.SKIPPED.value,
            message=f"Skipped: stage not in CI lane {state.case.ci_lane}",
        )
        return True

    def _run_trt_stage(
        self,
        state: _RunState,
        stage: Any,
    ) -> StageOutput | None:
        if state.runner is None:
            trt_output = StageOutput(
                stage_name=stage.name,
                data={"status": "no_runner_registered"},
                metadata={"runner": "none"},
            )
            state.sink.write_stage_output(stage.name, trt_output, prefix="trt")
            logger.warning(
                "No runner registered for strategy %s, stage %s",
                state.case.task_strategy, stage.name,
            )
            return trt_output

        t0 = time.monotonic()
        try:
            trt_output = state.runner.run_stage(state.case, stage, state.ctx)
            state.timing[f"trt_{stage.name}_s"] = time.monotonic() - t0
            state.timing.update(_collect_trt_stage_timing(trt_output, stage.name))
            state.sink.write_stage_output(stage.name, trt_output, prefix="trt")
            _log_stage_subprocess(state.sink, stage.name, trt_output, "trt")
            _auto_register_artifacts(state.sink, trt_output, "trt")
            runtime_path_error = _validate_trt_runtime_path(
                state.case, state.ctx, trt_output)
            if runtime_path_error is not None:
                state.stage_results[stage.name] = CompareResult(
                    stage_name=stage.name,
                    status=StageStatus.ERROR.value,
                    message=f"TRT run failed: {runtime_path_error}",
                )
                self._mark_required_stage_failed(state, stage)
                return None
            return trt_output
        except Exception as e:
            state.timing[f"trt_{stage.name}_s"] = time.monotonic() - t0
            tb = traceback.format_exc()
            logger.error("TRT run failed for stage %s: %s\n%s", stage.name, e, tb)
            state.stage_results[stage.name] = CompareResult(
                stage_name=stage.name,
                status=StageStatus.ERROR.value,
                message=f"TRT run failed: {e}\n{tb}",
            )
            self._mark_required_stage_failed(state, stage)
            return None

    def _run_reference_stage(
        self,
        state: _RunState,
        stage: Any,
    ) -> StageOutput | None:
        if state.skip_reason:
            logger.info(
                "Skipping reference for %s stage %s: %s",
                state.case.name, stage.name, state.skip_reason,
            )
            return None

        if state.reference is None:
            logger.warning(
                "No reference backend registered for %s, stage %s — skipping comparison",
                state.case.reference_backend, stage.name,
            )
            return None

        t0 = time.monotonic()
        try:
            ref_output = state.reference.run_stage(state.case, stage, state.ctx)
            state.timing[f"ref_{stage.name}_s"] = time.monotonic() - t0
            state.sink.write_stage_output(stage.name, ref_output, prefix="ref")
            _log_stage_subprocess(state.sink, stage.name, ref_output, "ref")
            _auto_register_artifacts(state.sink, ref_output, "ref")
            return ref_output
        except Exception as e:
            state.timing[f"ref_{stage.name}_s"] = time.monotonic() - t0
            tb = traceback.format_exc()
            logger.error(
                "Reference run failed for stage %s: %s\n%s", stage.name, e, tb)
            state.stage_results[stage.name] = CompareResult(
                stage_name=stage.name,
                status=StageStatus.ERROR.value,
                message=f"Reference run failed: {e}\n{tb}",
            )
            self._mark_required_stage_failed(state, stage)
            return None

    def _comparison_phase(
        self,
        state: _RunState,
        stage: Any,
        trt_output: StageOutput,
        ref_output: StageOutput | None,
    ) -> None:
        if ref_output is None:
            result = CompareResult(
                stage_name=stage.name,
                status=StageStatus.SKIPPED.value,
                message="TRT run succeeded (no reference available)",
            )
            state.stage_results[stage.name] = result
            if stage.required:
                state.required_validation_skipped = True
                state.all_stages_pass = False
            return

        compare_result = self._contract_comparison(
            state, stage, trt_output, ref_output)
        if compare_result is None:
            compare_result = self._numeric_comparison(
                state, stage, trt_output, ref_output)

        if compare_result is None:
            compare_result = CompareResult(
                stage_name=stage.name,
                status=StageStatus.SKIPPED.value,
                message="TRT and reference ran (no contract plugin or comparator)",
            )

        state.stage_results[stage.name] = compare_result
        state.sink.write_compare(stage.name, compare_result)
        if compare_result.status == StageStatus.SKIPPED.value and stage.required:
            state.required_validation_skipped = True
            state.all_stages_pass = False
        elif not compare_result.passed and stage.required:
            state.all_stages_pass = False

    def _contract_comparison(
        self,
        state: _RunState,
        stage: Any,
        trt_output: StageOutput,
        ref_output: StageOutput,
    ) -> CompareResult | None:
        if state.contract_plugin is None:
            return None

        t0 = time.monotonic()
        try:
            if state.plugin_runtime_context is None:
                state.plugin_runtime_context = _build_plugin_runtime_context(state.ctx)
            result = _verify_contract_plugin(
                state.contract_plugin,
                trt_output,
                ref_output,
                state.case,
                state.threshold,
                state.plugin_runtime_context,
            )
            state.timing[f"contract_{stage.name}_s"] = time.monotonic() - t0
            return result
        except Exception as e:
            state.timing[f"contract_{stage.name}_s"] = time.monotonic() - t0
            tb = traceback.format_exc()
            return CompareResult(
                stage_name=stage.name,
                status=StageStatus.ERROR.value,
                message=f"Contract verification failed: {e}\n{tb}",
            )

    def _numeric_comparison(
        self,
        state: _RunState,
        stage: Any,
        trt_output: StageOutput,
        ref_output: StageOutput,
    ) -> CompareResult | None:
        if state.comparator is None:
            return None

        t0 = time.monotonic()
        try:
            result = state.comparator.compare(
                trt_output, ref_output, state.threshold, stage)
            state.timing[f"compare_{stage.name}_s"] = time.monotonic() - t0
            return result
        except Exception as e:
            state.timing[f"compare_{stage.name}_s"] = time.monotonic() - t0
            tb = traceback.format_exc()
            return CompareResult(
                stage_name=stage.name,
                status=StageStatus.ERROR.value,
                message=f"Comparison failed: {e}\n{tb}",
            )

    def _determinism_phase(self, state: _RunState) -> None:
        reruns = state.case.determinism.get("reruns", 0)
        if reruns <= 0 or state.runner is None:
            return

        state.determinism_results["reruns_requested"] = reruns
        state.determinism_results["per_stage"] = {}
        determinism_ok = True

        for stage in state.case.stages:
            if not stage.required:
                continue
            baseline = state.baseline_trt_outputs.get(stage.name)
            if baseline is None:
                continue

            rerun_outputs: list[dict[str, Any]] = []
            for i in range(reruns):
                matched = self._run_determinism_rerun(
                    state, stage, baseline, i, rerun_outputs)
                if not matched:
                    determinism_ok = False

            state.determinism_results["per_stage"][stage.name] = rerun_outputs

        state.determinism_results["status"] = (
            "deterministic" if determinism_ok else "non_deterministic"
        )
        if not determinism_ok:
            state.all_stages_pass = False

    def _run_determinism_rerun(
        self,
        state: _RunState,
        stage: Any,
        baseline: StageOutput,
        rerun_index: int,
        rerun_outputs: list[dict[str, Any]],
    ) -> bool:
        t0 = time.monotonic()
        try:
            rerun_out = state.runner.run_stage(state.case, stage, state.ctx)
            elapsed = time.monotonic() - t0
            state.timing[f"determinism_{stage.name}_rerun_{rerun_index}_s"] = elapsed

            text_match = (
                rerun_out.text == baseline.text
            ) if baseline.text is not None else True
            data_match = self._stage_data_matches(baseline, rerun_out)
            matched = text_match and data_match
            rerun_outputs.append({
                "rerun_index": rerun_index,
                "text_match": text_match,
                "data_match": data_match,
                "matched": matched,
                "timing_s": elapsed,
            })
            state.sink.log_command(
                command=[f"determinism_rerun_{stage.name}_{rerun_index}"],
                rc=0 if matched else 1,
                stdout=rerun_out.text or "",
                stderr="",
                label=f"determinism_{stage.name}_rerun_{rerun_index}",
            )
            return matched
        except Exception as e:
            elapsed = time.monotonic() - t0
            state.timing[f"determinism_{stage.name}_rerun_{rerun_index}_s"] = elapsed
            tb = traceback.format_exc()
            logger.error(
                "Determinism rerun %d for stage %s failed: %s\n%s",
                rerun_index, stage.name, e, tb,
            )
            rerun_outputs.append({
                "rerun_index": rerun_index,
                "matched": False,
                "error": str(e),
                "timing_s": elapsed,
            })
            state.sink.log_command(
                command=[f"determinism_rerun_{stage.name}_{rerun_index}"],
                rc=1,
                stdout="",
                stderr=f"{e}\n{tb}",
                label=f"determinism_{stage.name}_rerun_{rerun_index}",
            )
            return False

    def _stage_data_matches(
        self,
        baseline: StageOutput,
        rerun_out: StageOutput,
    ) -> bool:
        for key in baseline.data:
            if key not in rerun_out.data:
                return False
            try:
                if baseline.data[key] != rerun_out.data[key]:
                    return False
            except (TypeError, ValueError):
                pass
        return True

    def _finalization_phase(self, state: _RunState) -> E2EResult:
        repro = _build_repro_commands(
            state.case, state.original_ctx, state.bundle_path, state.build_info)
        status, failure_type = self._aggregate_status(state)

        result = E2EResult(
            case_name=state.case.name,
            status=status,
            failure_type=failure_type,
            oracle_level=state.case.oracle_level,
            stages=state.stage_results,
            determinism=state.determinism_results,
            timing=state.timing,
            detailed_timing=_build_detailed_timing(
                state.timing, state.build_info),
            env_fingerprint=state.env_fingerprint,
            timestamp=state.timestamp,
            repro_commands=repro,
        )

        try:
            state.sink.finalize(result)
        except Exception as e:
            logger.error("Failed to finalize artifacts: %s", e)
            result.failure_type = FailureType.ARTIFACT_WRITE_FAIL.value

        return result

    def _aggregate_status(self, state: _RunState) -> tuple[str, str | None]:
        if state.all_stages_pass:
            return E2EStatus.PASS.value, None

        if state.required_validation_skipped and not any(
            cr.status in (StageStatus.FAILED.value, StageStatus.ERROR.value)
            for cr in state.stage_results.values()
        ):
            return E2EStatus.SKIP.value, None

        failure_type = FailureType.COMPARE_FAIL.value
        for cr in state.stage_results.values():
            if not cr.passed:
                if "TRT run failed" in cr.message:
                    failure_type = FailureType.TRT_RUN_FAIL.value
                    break
                if "Reference run failed" in cr.message:
                    failure_type = FailureType.REFERENCE_RUN_FAIL.value
                    break

        if state.determinism_results.get("status") == "non_deterministic":
            failure_type = FailureType.DETERMINISM_FAIL.value

        return E2EStatus.FAIL.value, failure_type

    def _mark_required_stage_failed(self, state: _RunState, stage: Any) -> None:
        if stage.required:
            state.all_stages_pass = False
