# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Diffusion media generation strategy runner.

Executes TRT diffusion inference through the generic media-generation CLI.
Supports per-stage execution (t5_encode, dit_step, vae_decode, end_to_end)
and full pipeline via the C++ binary.

All GPU work runs in subprocesses for memory isolation.
"""

from __future__ import annotations

import json
import hashlib
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

from .. import save_full_stderr, _case_artifact_dir
from ..contracts import E2ECase, RunContext, StageOutput, StageSpec
from ._runtime_common import (
    _distributed_runtime_config,
    _ensure_distributed_runtime_env,
    _extract_rank_zero_stdout,
    _maybe_start_gpu_memory_sampler,
    _strip_mpi_stream_tags,
    _wrap_distributed_command,
)

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parents[6]
TOOLS_DIR = PROJECT_DIR / "tools"


def _find_trt_lib_dir() -> str:
    """Find TRT library directory from the Python tensorrt_libs package."""
    try:
        import importlib.util
        spec = importlib.util.find_spec("tensorrt_libs")
        if spec and spec.submodule_search_locations:
            return spec.submodule_search_locations[0]
    except ImportError:
        pass
    return ""


def _build_ld_library_path(ctx: RunContext) -> str:
    """Build LD_LIBRARY_PATH from context or auto-detect."""
    if ctx.ld_library_path:
        return ctx.ld_library_path
    trt_lib = _find_trt_lib_dir()
    parts = []
    if trt_lib:
        parts.append(trt_lib)
    parts.append("/usr/local/cuda/lib64")
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    if existing:
        parts.append(existing)
    return ":".join(parts)


def _resolve_bundle_path(case: E2ECase, ctx: RunContext) -> str:
    """Resolve the full path to the .trtfb bundle."""
    bundle_name = case.bundle or case.inputs.get("bundle", "")
    if not bundle_name:
        bundle_name = f"{case.name}.trtfb"
    if os.path.isabs(bundle_name):
        return bundle_name
    return os.path.join(ctx.engine_dir, bundle_name)


def _resolve_cached_model_ref(hf_id: str) -> str:
    """Prefer a local snapshot and patch known-bad tokenizer configs in /tmp."""
    if not hf_id:
        return hf_id
    local_path = Path(hf_id)
    if local_path.exists():
        return hf_id
    try:
        from huggingface_hub import snapshot_download

        snapshot = Path(snapshot_download(hf_id, local_files_only=True))
    except Exception:
        return hf_id

    tok_cfg = snapshot / "tokenizer" / "tokenizer_config.json"
    if not tok_cfg.exists():
        return str(snapshot)

    try:
        cfg = json.loads(tok_cfg.read_text())
    except Exception:
        return str(snapshot)

    if not isinstance(cfg.get("extra_special_tokens"), list):
        return str(snapshot)

    patched_root = Path(tempfile.gettempdir()) / "trtmc_hf_patched" / hashlib.sha256(
        str(snapshot).encode("utf-8")).hexdigest()
    patched_cfg = patched_root / "tokenizer" / "tokenizer_config.json"
    if not patched_cfg.exists():
        shutil.copytree(snapshot, patched_root, dirs_exist_ok=True)
        patched = json.loads(patched_cfg.read_text())
        patched.pop("extra_special_tokens", None)
        patched_cfg.write_text(json.dumps(patched, indent=2))
    return str(patched_root)


class DiffusionMediaRunner:
    """TRT strategy runner for diffusion media generation pipelines."""

    @property
    def strategy_name(self) -> str:
        return "diffusion_media_generation"

    def run_stage(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Execute one diffusion stage and return its output."""
        dispatch = {
            "t5_encode": self._run_t5_encode,
            "dit_step": self._run_dit_step,
            "vae_decode": self._run_vae_decode,
            "end_to_end": self._run_end_to_end,
            "end_to_end_video": self._run_end_to_end,
            "debug_pipeline": self._run_debug_pipeline,
            "generate": self._run_end_to_end,
            "frame_quality": self._run_frame_quality,
        }
        handler = dispatch.get(stage.name)
        if handler is None:
            return StageOutput(
                stage_name=stage.name,
                data={"error": f"Unknown diffusion stage: {stage.name}"},
                metadata={"status": "unsupported_stage"},
            )
        return handler(case, stage, ctx)

    def _run_debug_pipeline(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Run debug_diffusion_pipeline.py for full 9-step TRT-vs-HF comparison."""
        bundle_path = _resolve_bundle_path(case, ctx)
        script = TOOLS_DIR / "debug_diffusion_pipeline.py"
        model_id = case.hf_id

        cmd = [
            sys.executable, str(script),
            "--bundle", bundle_path,
            "--model-id", model_id,
            "--num-steps", str(case.inputs.get("num_inference_steps", 10)),
        ]

        t0 = time.monotonic()
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=3600)
        elapsed = time.monotonic() - t0

        stderr_truncated, stderr_log = save_full_stderr(
            result.stderr or "", ctx.artifacts_dir or "",
            "debug_pipeline", case.name)
        dbg_data: dict = {
            "passed": result.returncode == 0,
            "returncode": result.returncode,
            "output": result.stdout,
            "stderr": stderr_truncated,
        }
        if stderr_log:
            dbg_data["stderr_log"] = stderr_log

        return StageOutput(
            stage_name=stage.name,
            data=dbg_data,
            text=result.stdout,
            timing_s=elapsed,
            metadata={"command": cmd},
        )

    def _run_t5_encode(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Run T5 text encoding stage via debug_diffusion_pipeline subprocess."""
        bundle_path = _resolve_bundle_path(case, ctx)
        model_ref = _resolve_cached_model_ref(case.hf_id)
        max_length = int(case.inputs.get("text_max_length", 512))

        # Run as a subprocess that loads the TRT T5 engine and encodes text
        prompt_text = case.inputs.get("prompt", "A cat sitting on a beach")
        script_code = textwrap.dedent(f"""\
            import sys
            sys.path.insert(0, {str(TOOLS_DIR)!r})
            import numpy as np
            from tensorrt_model_connect.families.flux.diffusion_runner import DiffusionRunner

            runner = DiffusionRunner({bundle_path!r})
            prompt_text = {prompt_text!r}
            model_ref = {model_ref!r}
            max_length = {max_length}

            from transformers import AutoTokenizer
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_ref, subfolder="tokenizer", use_fast=False)
            except (ValueError, OSError):
                tokenizer = AutoTokenizer.from_pretrained(model_ref, use_fast=False)
            tokens = tokenizer(prompt_text, return_tensors="np", padding="max_length",
                               max_length=max_length, truncation=True)
            input_ids = tokens["input_ids"].astype(np.int32)
            text_output = runner.encode_text(input_ids)
            import os as _os
            _arts = {str(Path(_case_artifact_dir(ctx.artifacts_dir, case.name)) if ctx.artifacts_dir else Path('/tmp/claude'))!r}
            _os.makedirs(_arts, exist_ok=True)
            _npy_path = _os.path.join(_arts, "trt_t5_output.npy")
            np.save(_npy_path, text_output)
            print("output_path=" + _npy_path)
            print("shape=" + str(list(text_output.shape)))
            print("mean=" + format(float(text_output.mean()), ".6f"))
            print("std=" + format(float(text_output.std()), ".6f"))
        """)
        python = ctx.runtime_python_path() or sys.executable
        t0 = time.monotonic()
        result = subprocess.run(
            [python, "-c", script_code],
            capture_output=True, text=True, timeout=600,
            env={**os.environ, "LD_LIBRARY_PATH": _build_ld_library_path(ctx)},
        )
        elapsed = time.monotonic() - t0

        stderr_truncated, stderr_log = save_full_stderr(
            result.stderr or "", ctx.artifacts_dir or "",
            "t5_encode", case.name)
        data: dict = {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": stderr_truncated,
        }
        if stderr_log:
            data["stderr_log"] = stderr_log
        # Parse output_path from stdout (printed by the subprocess)
        for line in (result.stdout or "").splitlines():
            if line.startswith("output_path="):
                npy_path = line.split("=", 1)[1].strip()
                if os.path.exists(npy_path):
                    data["output_path"] = npy_path
                break

        return StageOutput(
            stage_name=stage.name,
            data=data,
            text=result.stdout,
            timing_s=elapsed,
            metadata={"command": "t5_encode_subprocess"},
        )

    def _run_dit_step(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Run a single DiT forward pass via debug_diffusion_pipeline."""
        # Delegate to debug_pipeline which includes dit_step comparison
        return self._run_debug_pipeline(case, stage, ctx)

    def _run_vae_decode(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Run VAE decoder stage."""
        # VAE decoding is part of the full pipeline; run end_to_end
        return self._run_end_to_end(case, stage, ctx)

    def _run_end_to_end(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Run full generation via the C++ binary.

        The shared runner only covers the generic ``trtmc generate-video``
        path. Family-specific inputs such as fixed latents or image-only
        commands are owned by model plugins.
        """
        bundle_path = _resolve_bundle_path(case, ctx)
        binary = ctx.binary_path
        prompt = case.inputs.get("prompt", "A cat sitting on a beach")
        num_steps = case.inputs.get("num_inference_steps", 30)
        ld_path = _build_ld_library_path(ctx)

        with tempfile.TemporaryDirectory(prefix="trtmc_frames_") as frame_dir:
            cmd = [
                binary, "generate-video", bundle_path,
                "--prompt", prompt,
                "--num-steps", str(num_steps),
            ]
            guidance_scale = case.inputs.get("guidance_scale")
            if guidance_scale is not None:
                cmd.extend(["--guidance-scale", str(guidance_scale)])
            if "seed" in case.inputs:
                cmd.extend(["--seed", str(case.inputs["seed"])])
            output_target = frame_dir
            runtime_cli_python = ctx.runtime_cli_hf_python()
            if runtime_cli_python:
                cmd.extend(["--hf-python", runtime_cli_python])

            env = {**os.environ, "LD_LIBRARY_PATH": ld_path}
            distributed_runtime = _distributed_runtime_config(case)
            output_frame_dir = frame_dir
            if distributed_runtime:
                _ensure_distributed_runtime_env(case, ctx, env)
                extra_env = distributed_runtime.get("env", {})
                if isinstance(extra_env, dict):
                    env.update({str(k): str(v) for k, v in extra_env.items()})
                wrapper = (
                    'rank="${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-${RANK:-0}}}"; '
                    'out="$1/rank_${rank}"; mkdir -p "$out"; shift; '
                    'exec "$@" --output "$out"'
                )
                cmd = [
                    "bash", "-lc", wrapper, "trtmc_rank_output", frame_dir,
                ] + cmd
                output_frame_dir = os.path.join(frame_dir, "rank_0")
                cmd = _wrap_distributed_command(cmd, case, env)
            else:
                cmd.extend(["--output", output_target])

            t0 = time.monotonic()
            memory_sampler = _maybe_start_gpu_memory_sampler(
                distributed_runtime, ctx, case, env)
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, encoding="utf-8",
                    errors="replace", timeout=3600, env=env)
            finally:
                memory_meta = (
                    memory_sampler.stop() if memory_sampler is not None else None)
            elapsed = time.monotonic() - t0
            stdout_text = (
                _extract_rank_zero_stdout(result.stdout)
                if distributed_runtime else result.stdout
            )
            stderr_text = (
                _strip_mpi_stream_tags(result.stderr)
                if distributed_runtime else result.stderr
            )

            # Count frames
            frame_files = sorted(Path(output_frame_dir).glob("frame_*.png"))
            num_frames = len(frame_files)

            # Compute frame statistics if frames exist
            frame_stats = {}
            if num_frames > 0:
                frame_stats = self._compute_frame_stats(output_frame_dir)

            # Copy frame paths for artifact persistence
            frame_paths = [str(f) for f in frame_files]

            # Persist frames to artifacts_dir before tempdir cleanup
            artifact_frames_dir = None
            if ctx.artifacts_dir and num_frames > 0:
                artifact_frames_dir = os.path.join(
                    _case_artifact_dir(ctx.artifacts_dir, case.name), "frames")
                os.makedirs(artifact_frames_dir, exist_ok=True)
                for fp in frame_files:
                    shutil.copy2(str(fp), artifact_frames_dir)
                frame_paths = [
                    os.path.join(artifact_frames_dir, fp.name)
                    for fp in frame_files
                ]

            stderr_truncated, stderr_log = save_full_stderr(
                stderr_text or "", ctx.artifacts_dir or "",
                "end_to_end", case.name)
            e2e_data: dict = {
                "returncode": result.returncode,
                "num_frames": num_frames,
                "frame_stats": frame_stats,
                "frames_dir": artifact_frames_dir or frame_dir,
                "frame_paths": frame_paths,
                "stdout": stdout_text,
                "stderr": stderr_truncated,
                # Passed through to the comparator for CLIP semantic metrics.
                "prompt": case.inputs.get("prompt") or case.inputs.get("test_prompt"),
            }
            if stderr_log:
                e2e_data["stderr_log"] = stderr_log
            if distributed_runtime:
                e2e_data["raw_stdout"] = result.stdout
                e2e_data["distributed_runtime"] = distributed_runtime
            if memory_meta is not None:
                e2e_data["gpu_memory"] = memory_meta

            metadata = {"command": cmd}
            if distributed_runtime:
                metadata["distributed_runtime"] = distributed_runtime
                metadata["rank_zero_stdout"] = stdout_text
            return StageOutput(
                stage_name=stage.name,
                data=e2e_data,
                text=stdout_text,
                timing_s=elapsed,
                metadata=metadata,
            )

    def _run_frame_quality(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Generate frames and compute pixel statistics for quality checks."""
        return self._run_end_to_end(case, stage, ctx)

    @staticmethod
    def _compute_frame_stats(frame_dir: str) -> dict:
        """Load PNG frames and return aggregate pixel statistics."""
        try:
            import numpy as np
            from PIL import Image
        except ImportError:
            return {"error": "PIL or numpy not available"}

        frames = sorted(Path(frame_dir).glob("frame_*.png"))
        if not frames:
            return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

        all_pixels = []
        for fp in frames:
            img = Image.open(fp).convert("RGB")
            arr = np.array(img, dtype=np.float32) / 255.0
            all_pixels.append(arr.flatten())

        combined = np.concatenate(all_pixels)
        return {
            "count": len(frames),
            "mean": float(np.mean(combined)),
            "std": float(np.std(combined)),
            "min": float(np.min(combined)),
            "max": float(np.max(combined)),
        }


plugin = DiffusionMediaRunner()
