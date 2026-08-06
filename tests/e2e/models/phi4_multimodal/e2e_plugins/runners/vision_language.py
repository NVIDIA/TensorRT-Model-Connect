# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vision-language strategy runner — TRT inference for VL models.

Handles Qwen2.5-VL, Qwen3-VL (DeepStack), and generic VL models.
Uses subprocess isolation for all GPU-intensive operations.

Stages:
  - "vision_encode": Run vision encoder via debug runner subprocess.
  - "text_decode": Run VLTrtRunner in Python subprocess for per-step logits
    and vision features (used for detailed numerical comparison).
  - "full_generation": Run full VL pipeline via C++ binary with --image flag.

Auto-discovered by the registry via the module-level ``plugin`` attribute.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from .. import save_full_stderr, _case_artifact_dir
from ..contracts import E2ECase, RunContext, StageOutput, StageSpec
from ._runtime_common import (
    _detect_trt_runtime_error,
    _distributed_runtime_config,
    _ensure_distributed_runtime_env,
    _extract_rank_zero_stdout,
    _extract_trtmc_load_timing,
    _extract_trtmc_timing,
    _maybe_start_gpu_memory_sampler,
    _strip_mpi_stream_tags,
    _wrap_distributed_command,
)

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parents[6]
TOOLS_DIR = PROJECT_DIR / "tools"


class VisionLanguageRunner:
    """TRT inference runner for vision-language generation models."""

    @property
    def strategy_name(self) -> str:
        return "vision_language_generation"

    def run_stage(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        if stage.name == "vision_encode":
            return self._run_vision_encode(case, ctx)
        elif stage.name == "text_decode":
            return self._run_text_decode(case, ctx)
        elif stage.name == "full_generation":
            return self._run_full_generation(case, ctx)
        else:
            return StageOutput(
                stage_name=stage.name,
                metadata={"error": f"Unknown stage: {stage.name}"},
            )

    def _resolve_bundle_path(self, case: E2ECase, ctx: RunContext) -> str:
        bundle = case.bundle or f"{case.name}.bundle"
        if os.path.isabs(bundle):
            return bundle
        return os.path.join(ctx.engine_dir, bundle)

    def _resolve_image_path(self, case: E2ECase, ctx: RunContext) -> str | None:
        image = (case.inputs.get("image") or case.inputs.get("test_image")
                 or case.inputs.get("image_path"))
        if not image:
            return None
        p = Path(image)
        if p.is_absolute():
            return str(p)
        # Try relative to engine_dir first, then project root
        for base in [ctx.engine_dir, str(PROJECT_DIR)]:
            candidate = os.path.join(base, image)
            if os.path.isfile(candidate):
                return candidate
        return str(p)

    def _run_vision_encode(
        self, case: E2ECase, ctx: RunContext
    ) -> StageOutput:
        """Run vision encoder via diff_vl.py subprocess (vision-only mode)."""
        bundle_path = self._resolve_bundle_path(case, ctx)
        image_path = self._resolve_image_path(case, ctx)

        if not image_path or not os.path.isfile(image_path):
            return StageOutput(
                stage_name="vision_encode",
                metadata={"error": f"Image not found: {image_path}",
                          "skipped": True},
            )

        diff_vl = TOOLS_DIR / "diff_vl.py"
        cmd = [
            sys.executable, str(diff_vl),
            "--bundle", str(bundle_path),
            "--image", str(image_path),
            "--vision-only",
        ]

        # Add HF model for numerical comparison if available
        if case.hf_id:
            cmd.extend(["--model", case.hf_id])
            atol = case.inputs.get("vision_atol", 0.1)
            cmd.extend(["--atol", str(atol)])

        env = dict(os.environ)
        if ctx.ld_library_path:
            env["LD_LIBRARY_PATH"] = ctx.ld_library_path

        t0 = time.monotonic()
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600, env=env)
        except subprocess.TimeoutExpired:
            return StageOutput(
                stage_name="vision_encode",
                timing_s=time.monotonic() - t0,
                metadata={"error": "Vision encode subprocess timed out",
                          "command": cmd},
            )
        elapsed = time.monotonic() - t0

        passed = result.returncode == 0
        # Parse metrics from stderr output
        metrics = _parse_diff_vl_metrics(result.stderr)

        stderr_truncated, stderr_log = save_full_stderr(
            result.stderr or "", ctx.artifacts_dir or "",
            "vision_encode", case.name)
        meta: dict = {
            "command": cmd,
            "returncode": result.returncode,
            "stdout": result.stdout or "",
            "stderr": result.stderr or "",
        }
        if stderr_log:
            meta["stderr_log"] = stderr_log

        return StageOutput(
            stage_name="vision_encode",
            data={
                "passed": passed,
                "metrics": metrics,
            },
            timing_s=elapsed,
            metadata=meta,
        )

    def _run_text_decode(
        self, case: E2ECase, ctx: RunContext
    ) -> StageOutput:
        """Run VLTrtRunner subprocess to get per-step logits and vision features.

        Uses a subprocess that loads the bundle, encodes the image, tokenizes
        the prompt, runs autoregressive decode steps, and outputs per-step
        logits as a .npy file for numerical comparison.
        """
        bundle_path = self._resolve_bundle_path(case, ctx)
        image_path = self._resolve_image_path(case, ctx)
        prompt = case.inputs.get("prompt", "Describe this image.")
        max_new_tokens = case.inputs.get("max_new_tokens", 30)

        if not image_path or not os.path.isfile(image_path):
            return StageOutput(
                stage_name="text_decode",
                metadata={"error": f"Image not found: {image_path}",
                          "skipped": True},
            )

        artifacts = ctx.artifacts_dir or "/tmp/claude"
        model_dir = _case_artifact_dir(artifacts, case.name) if ctx.artifacts_dir else artifacts
        logits_path = os.path.join(model_dir, "vl_logits.npy")
        features_path = os.path.join(model_dir, "vl_features.npy")
        text_path = os.path.join(model_dir, "vl_text.txt")
        os.makedirs(model_dir, exist_ok=True)

        # Build inline script that runs VLTrtRunner in an isolated process
        script = _VL_TEXT_DECODE_SCRIPT.format(
            bundle_path=bundle_path,
            image_path=image_path,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            logits_path=logits_path,
            features_path=features_path,
            text_path=text_path,
        )

        env = dict(os.environ)
        if ctx.ld_library_path:
            env["LD_LIBRARY_PATH"] = ctx.ld_library_path

        t0 = time.monotonic()
        try:
            result = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True, text=True, timeout=600, env=env)
        except subprocess.TimeoutExpired:
            return StageOutput(
                stage_name="text_decode",
                timing_s=time.monotonic() - t0,
                metadata={"error": "VL text decode subprocess timed out"},
            )
        elapsed = time.monotonic() - t0

        data: dict = {"prompt": prompt, "image_path": str(image_path)}
        generated_text = ""

        # Load outputs if subprocess succeeded
        if result.returncode == 0:
            if os.path.isfile(logits_path):
                data["logits_path"] = logits_path
            if os.path.isfile(features_path):
                data["features_path"] = features_path
            if os.path.isfile(text_path):
                generated_text = Path(text_path).read_text().strip()
                data["generated_text"] = generated_text

        # Parse any metrics from stdout
        metrics = {}
        if result.stdout:
            for line in result.stdout.splitlines():
                if "=" in line:
                    try:
                        key, val = line.split("=", 1)
                        metrics[key.strip()] = float(val.strip())
                    except (ValueError, IndexError):
                        pass
            data["parsed_metrics"] = metrics

        stderr_truncated, stderr_log = save_full_stderr(
            result.stderr or "", ctx.artifacts_dir or "",
            "text_decode", case.name)
        meta_td: dict = {
            "returncode": result.returncode,
            "stdout": result.stdout or "",
            "stderr": result.stderr or "",
        }
        if stderr_log:
            meta_td["stderr_log"] = stderr_log

        return StageOutput(
            stage_name="text_decode",
            text=generated_text,
            logits=logits_path if os.path.isfile(logits_path) else None,
            data=data,
            timing_s=elapsed,
            metadata=meta_td,
        )

    def _run_full_generation(
        self, case: E2ECase, ctx: RunContext
    ) -> StageOutput:
        """Run full VL generation via C++ binary with --image flag."""
        bundle_path = self._resolve_bundle_path(case, ctx)
        image_path = self._resolve_image_path(case, ctx)
        prompt = case.inputs.get("prompt", "Describe this image.")
        max_new_tokens = case.inputs.get("max_new_tokens", 30)

        if not image_path or not os.path.isfile(image_path):
            return StageOutput(
                stage_name="full_generation",
                metadata={"error": f"Image not found: {image_path}",
                          "skipped": True},
            )

        if not ctx.binary_path or not os.path.isfile(ctx.binary_path):
            return StageOutput(
                stage_name="full_generation",
                metadata={"error": f"Binary not found: {ctx.binary_path}",
                          "skipped": True},
            )

        cmd = [
            str(ctx.binary_path), "run", str(bundle_path),
            "--prompt", prompt,
            "--image", str(image_path),
            "--max-new-tokens", str(max_new_tokens),
        ]
        runtime_cli_python = ctx.runtime_cli_hf_python()
        if runtime_cli_python:
            cmd.extend(["--hf-python", str(runtime_cli_python)])

        contract_config = case.metadata.get("contract_config", {})
        if contract_config.get("use_chat_template"):
            cmd.append("--chat-template")
        if contract_config.get("enable_thinking") is False:
            cmd.append("--no-thinking")

        env = dict(os.environ)
        if ctx.ld_library_path:
            env["LD_LIBRARY_PATH"] = ctx.ld_library_path
        distributed_runtime = _distributed_runtime_config(case)
        if distributed_runtime:
            _ensure_distributed_runtime_env(case, ctx, env)
            extra_env = distributed_runtime.get("env", {})
            if isinstance(extra_env, dict):
                env.update({str(k): str(v) for k, v in extra_env.items()})
            cmd = _wrap_distributed_command(cmd, case, env)

        t0 = time.monotonic()
        memory_sampler = _maybe_start_gpu_memory_sampler(
            distributed_runtime, ctx, case, env)
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600, env=env)
        except subprocess.TimeoutExpired:
            meta: dict = {
                "error": "C++ VL generation timed out",
                "command": cmd,
            }
            if memory_sampler is not None:
                meta["gpu_memory"] = memory_sampler.stop()
            return StageOutput(
                stage_name="full_generation",
                timing_s=time.monotonic() - t0,
                metadata=meta,
            )
        elapsed = time.monotonic() - t0
        memory_meta = memory_sampler.stop() if memory_sampler is not None else None

        parse_stderr = (
            _strip_mpi_stream_tags(result.stderr)
            if distributed_runtime else result.stderr
        )
        raw_text = (
            _extract_rank_zero_stdout(result.stdout)
            if distributed_runtime else result.stdout.strip()
        )

        # Strip chat template / prompt prefix from C++ binary output.
        # The C++ binary may output the full conversation including the
        # user prompt and assistant prefix. HF reference only returns
        # the generated portion, so we strip accordingly.
        generated_text = raw_text
        # Try stripping after "assistant\n" marker (common chat template)
        for marker in ["assistant\n", "assistant:", "ASSISTANT:"]:
            if marker in generated_text:
                generated_text = generated_text.split(marker, 1)[-1].strip()
                break
        # If the prompt itself appears as prefix, strip it
        if prompt and generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()

        # Persist generated text for human inspection
        if ctx.artifacts_dir and generated_text:
            art_dir = Path(_case_artifact_dir(ctx.artifacts_dir, case.name))
            txt_path = art_dir / "vl_output.txt"
            txt_path.write_text(generated_text, encoding="utf-8")

        stderr_truncated, stderr_log = save_full_stderr(
            result.stderr or "", ctx.artifacts_dir or "",
            "vl_full_generation", case.name)
        meta_fg: dict = {
            "command": cmd,
            "returncode": result.returncode,
            "stdout": result.stdout or "",
            "stderr": result.stderr or "",
        }
        if distributed_runtime:
            meta_fg["distributed_runtime"] = distributed_runtime
            meta_fg["rank_zero_stdout"] = raw_text
            meta_fg["stderr_without_mpi_tags"] = parse_stderr
        if memory_meta is not None:
            meta_fg["gpu_memory"] = memory_meta
        meta_fg.update(_extract_trtmc_timing(parse_stderr))
        meta_fg.update(_extract_trtmc_load_timing(parse_stderr))
        runtime_error = _detect_trt_runtime_error(parse_stderr)
        if runtime_error:
            meta_fg["runtime_error_detected"] = runtime_error
            if result.returncode == 0:
                meta_fg["effective_returncode"] = -1
                meta_fg["error"] = "TensorRT runtime error detected in stderr"
        if stderr_log:
            meta_fg["stderr_log"] = stderr_log

        return StageOutput(
            stage_name="full_generation",
            text=generated_text,
            data={
                "generated_text": generated_text,
                "prompt": prompt,
                "image_path": str(image_path),
            },
            timing_s=elapsed,
            metadata=meta_fg,
        )


# Inline subprocess script for VLTrtRunner text decode.
# Uses format() with named placeholders for paths/params.
_VL_TEXT_DECODE_SCRIPT = """\
import sys, os, numpy as np

bundle_path = "{bundle_path}"
image_path = "{image_path}"
prompt = "{prompt}"
max_new_tokens = {max_new_tokens}
logits_path = "{logits_path}"
features_path = "{features_path}"
text_path = "{text_path}"

from tests.e2e.models.phi4_multimodal.e2e_plugins.runners.vl_debug_runner import VLTrtRunner

runner = VLTrtRunner(bundle_path)
if runner.vision_runner is None:
    print("no_vision_engine=1")
    sys.exit(0)

# Encode image and save features
features = runner.encode_image(image_path)
np.save(features_path, features)
print(f"features_shape={features.shape}")
print(f"features_mean={{features.mean():.6f}}")
print(f"features_std={{features.std():.6f}}")

# Format prompt and tokenize
formatted = runner.format_prompt(prompt)

try:
    from transformers import AutoTokenizer
    import tempfile
    from pathlib import Path
    from tests.e2e.models.phi4_multimodal.e2e_plugins.runners.vl_debug_runner import load_section_from_bundle
    tok_data = load_section_from_bundle(bundle_path, "tokenizer.json")
    if tok_data:
        with tempfile.TemporaryDirectory() as tmpdir:
            for name in ["tokenizer.json", "tokenizer_config.json",
                         "special_tokens_map.json"]:
                data = load_section_from_bundle(bundle_path, name)
                if data:
                    (Path(tmpdir) / name).write_bytes(data)
            tokenizer = AutoTokenizer.from_pretrained(tmpdir)
    else:
        model_source = runner.config.get("model_source", "")
        tokenizer = AutoTokenizer.from_pretrained(model_source)
except Exception as e:
    print(f"tokenizer_error={{e}}")
    sys.exit(1)

input_ids = tokenizer.encode(formatted, add_special_tokens=False)
print(f"input_tokens={{len(input_ids)}}")

# Generate and collect per-step logits
output_ids = runner.generate_vl(input_ids, features, max_new_tokens)
new_ids = output_ids[len(input_ids):]
output_text = tokenizer.decode(new_ids, skip_special_tokens=True)

with open(text_path, "w") as f:
    f.write(output_text)

# Save per-step logits if the runner collected them
if hasattr(runner, "step_logits") and runner.step_logits:
    logits_array = np.stack(runner.step_logits, axis=0)
    np.save(logits_path, logits_array)
    print(f"logits_steps={{logits_array.shape[0]}}")
    print(f"logits_vocab={{logits_array.shape[1]}}")

print(f"generated_tokens={{len(new_ids)}}")
print(f"output_length={{len(output_text)}}")
"""


def _parse_diff_vl_metrics(stderr_text: str) -> dict:
    """Parse structured metrics from diff_vl.py stderr output."""
    metrics = {}
    if not stderr_text:
        return metrics

    for line in stderr_text.splitlines():
        if "cosine_sim=" in line:
            try:
                val = line.split("cosine_sim=")[1].split(",")[0].split()[0]
                metrics["cosine_sim"] = float(val)
            except (IndexError, ValueError):
                pass
        if "max_diff=" in line:
            try:
                val = line.split("max_diff=")[1].split(",")[0].split()[0]
                metrics["max_diff"] = float(val)
            except (IndexError, ValueError):
                pass
        if "mean_diff=" in line:
            try:
                val = line.split("mean_diff=")[1].split(",")[0].split()[0]
                metrics["mean_diff"] = float(val)
            except (IndexError, ValueError):
                pass
        if "PASS" in line:
            metrics["vision_pass"] = True
        if "FAIL" in line:
            metrics["vision_fail"] = True

    return metrics


plugin = VisionLanguageRunner()
