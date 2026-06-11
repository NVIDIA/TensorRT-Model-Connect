"""Diffusion media generation strategy runner.

Executes TRT diffusion inference for Wan-style text-to-video pipelines.
Supports per-stage execution (t5_encode, dit_step, vae_decode, end_to_end)
and full pipeline via the C++ binary.

Crossover stages allow mixing TRT and HF components to isolate regressions:
- crossover_ref_t5_trt_dit: HF T5 text encoding + TRT DiT denoising
- crossover_trt_t5_ref_dit: TRT T5 text encoding + HF DiT denoising

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
from .text_generation import (
    _distributed_runtime_config,
    _ensure_distributed_runtime_env,
    _extract_rank_zero_stdout,
    _maybe_start_gpu_memory_sampler,
    _strip_mpi_stream_tags,
    _wrap_distributed_command,
)

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parents[3]
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


def _ltx_initial_latents_path(case: E2ECase, ctx: RunContext) -> str:
    if ctx.artifacts_dir:
        base_dir = _case_artifact_dir(ctx.artifacts_dir, case.name)
    else:
        base_dir = os.path.join(tempfile.gettempdir(), "trtmc_ltx_latents", case.name)
    return os.path.join(base_dir, "initial_latents.raw")


def _qwen_image_initial_latents_path(case: E2ECase, ctx: RunContext) -> str:
    if ctx.artifacts_dir:
        base_dir = _case_artifact_dir(ctx.artifacts_dir, case.name)
    else:
        base_dir = os.path.join(
            tempfile.gettempdir(), "trtmc_qwen_image_latents", case.name)
    return os.path.join(base_dir, "initial_latents.raw")


def _ensure_qwen_image_initial_latents(
        case: E2ECase, ctx: RunContext, bundle_path: str) -> str | None:
    """Pre-compute shared initial latents for the Qwen-Image E2E case.

    Writes a raw fp32 buffer of shape ``[1, 16, h_lat, w_lat]`` (UNPACKED,
    C-major) that both the TRT C++ pipeline (via ``--initial-latents-raw``)
    and the HF diffusers reference (via ``latents=`` after ``_pack_latents``)
    consume verbatim. This eliminates the RNG mismatch between
    ``std::mt19937`` (TRT side) and ``torch.Generator`` (HF side) that would
    otherwise drive the PSNR floor below the gate.

    Trace: IT-E2E-QIMG-01, UD-FAM-QWEN-IMAGE-01.
    """
    is_qwen_image = (
        case.runtime_strategy == "diffusion_qwen_image"
        or case.family == "qwen_image"
    )
    if not is_qwen_image:
        return None

    output_path = _qwen_image_initial_latents_path(case, ctx)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        return output_path

    height = int(
        case.inputs.get("height")
        or case.inputs.get("image_height")
        or 1024)
    width = int(
        case.inputs.get("width")
        or case.inputs.get("image_width")
        or 1024)
    seed = int(case.inputs.get("seed", case.determinism.get("seed", 42)))
    # Qwen-Image VAE: 8x spatial compression, 16 latent channels.
    vae_scale = 8
    latent_channels = 16
    h_lat = height // vae_scale
    w_lat = width // vae_scale

    # NumPy RNG suffices here — both sides read the SAME bytes from disk, so
    # the exact RNG used to seed them is immaterial; what matters is byte
    # identity between the TRT and HF subprocesses.
    import numpy as np
    rng = np.random.default_rng(seed)
    latents = rng.standard_normal(
        (1, latent_channels, h_lat, w_lat), dtype=np.float32)
    latents.tofile(output_path)
    return output_path


def _ensure_ltx_initial_latents(case: E2ECase, ctx: RunContext, bundle_path: str) -> str | None:
    if case.family != "ltx_video":
        return None

    output_path = _ltx_initial_latents_path(case, ctx)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    seed = int(case.inputs.get("seed", case.determinism.get("seed", 42)))
    script = textwrap.dedent(f"""\
        import sys
        from pathlib import Path
        sys.path.insert(0, {str(TOOLS_DIR)!r})
        import torch
        from diffusion_helpers import load_bundle_config

        cfg = load_bundle_config({bundle_path!r})
        video_num_frames = int({int(case.inputs.get("video_num_frames", 0))} or cfg.get("video_num_frames", 9))
        video_height = int({int(case.inputs.get("video_height", 0))} or cfg.get("video_height", 256))
        video_width = int({int(case.inputs.get("video_width", 0))} or cfg.get("video_width", 256))
        z_dim = int(cfg.get("z_dim", 128))
        scale_t = max(int(cfg.get("scale_factor_temporal", 8)), 1)
        scale_s = max(int(cfg.get("scale_factor_spatial", 32)), 1)
        pt, ph, pw = [int(v) for v in cfg.get("patch_size", [1, 1, 1])]
        t_lat = (video_num_frames - 1) // scale_t + 1
        h_lat = video_height // scale_s
        w_lat = video_width // scale_s
        if t_lat % pt or h_lat % ph or w_lat % pw:
            raise RuntimeError(f"latent shape {{t_lat}}x{{h_lat}}x{{w_lat}} is not divisible by patch {{pt}}x{{ph}}x{{pw}}")

        generator = torch.Generator("cuda").manual_seed({seed})
        latents = torch.randn(
            (1, z_dim, t_lat, h_lat, w_lat),
            generator=generator,
            device="cuda",
            dtype=torch.float32,
        )
        latents = latents.reshape(
            1, -1, t_lat // pt, pt, h_lat // ph, ph, w_lat // pw, pw
        ).permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3).contiguous()

        out = Path({output_path!r})
        out.parent.mkdir(parents=True, exist_ok=True)
        latents.cpu().numpy().astype("<f4", copy=False).tofile(out)
        print(str(out))
    """)

    python = ctx.runtime_python_path() or sys.executable
    result = subprocess.run(
        [python, "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "LD_LIBRARY_PATH": _build_ld_library_path(ctx)},
    )
    if result.returncode != 0:
        raise RuntimeError(
            "failed to create shared LTX initial latents: "
            + (result.stderr or result.stdout or "").strip()
        )
    return output_path


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
            "crossover_ref_t5_trt_dit": self._run_crossover_ref_t5_trt_dit,
            "crossover_trt_t5_ref_dit": self._run_crossover_trt_t5_ref_dit,
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
        max_length = 120 if case.family == "pixart" else 512

        # Run as a subprocess that loads the TRT T5 engine and encodes text
        prompt_text = case.inputs.get("prompt", "A cat sitting on a beach")
        script_code = textwrap.dedent(f"""\
            import sys
            sys.path.insert(0, {str(TOOLS_DIR)!r})
            import numpy as np
            from tensorrt_model_connect.diffusion_runner import DiffusionRunner

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

        Most diffusion families use ``trtmc generate-video`` which writes
        ``frame_*.png`` into the output dir. Qwen-Image is image-only and
        uses ``trtmc run`` (which dispatches to ``generate_image()``); we
        target a ``frame_0000.png`` output filename so the existing
        comparator path (frame globbing) still works for single-frame T2I.
        """
        bundle_path = _resolve_bundle_path(case, ctx)
        binary = ctx.binary_path
        prompt = case.inputs.get("prompt", "A cat sitting on a beach")
        num_steps = case.inputs.get("num_inference_steps", 30)
        ld_path = _build_ld_library_path(ctx)

        # Qwen-Image (and any image-only diffusion that publishes the
        # ``diffusion_qwen_image`` runtime strategy) drives the CLI through
        # ``trtmc run`` with the diffusion-text-to-image flag set.
        is_qwen_image = (
            case.runtime_strategy == "diffusion_qwen_image"
            or case.family == "qwen_image"
        )

        try:
            initial_latents_raw = _ensure_ltx_initial_latents(case, ctx, bundle_path)
        except Exception as exc:
            return StageOutput(
                stage_name=stage.name,
                data={"returncode": 1, "stderr": str(exc), "num_frames": 0},
                text="",
                timing_s=0.0,
                metadata={"command": "create_ltx_initial_latents"},
            )

        try:
            qwen_image_initial_latents_raw = _ensure_qwen_image_initial_latents(
                case, ctx, bundle_path)
        except Exception as exc:
            return StageOutput(
                stage_name=stage.name,
                data={"returncode": 1, "stderr": str(exc), "num_frames": 0},
                text="",
                timing_s=0.0,
                metadata={"command": "create_qwen_image_initial_latents"},
            )

        with tempfile.TemporaryDirectory(prefix="trtmc_frames_") as frame_dir:
            if is_qwen_image:
                # ``trtmc run`` writes a single PNG; target frame_0000.png
                # so the comparator's frame_*.png glob still picks it up.
                output_png = os.path.join(frame_dir, "frame_0000.png")
                cmd = [
                    binary, "run", bundle_path,
                    "--prompt", prompt,
                    "--num-inference-steps", str(num_steps),
                ]
                negative_prompt = case.inputs.get("negative_prompt")
                if negative_prompt is not None:
                    cmd.extend(["--negative-prompt", str(negative_prompt)])
                cfg_scale = case.inputs.get("cfg_scale")
                if cfg_scale is None:
                    cfg_scale = case.inputs.get("guidance_scale")
                if cfg_scale is not None:
                    cmd.extend(["--cfg-scale", str(cfg_scale)])
                height = case.inputs.get("height") or case.inputs.get("image_height")
                if height is not None:
                    cmd.extend(["--height", str(height)])
                width = case.inputs.get("width") or case.inputs.get("image_width")
                if width is not None:
                    cmd.extend(["--width", str(width)])
                if "seed" in case.inputs:
                    cmd.extend(["--seed", str(case.inputs["seed"])])
                image_path = case.inputs.get("image") or case.inputs.get("image_path")
                if image_path:
                    cmd.extend(["--image", str(image_path)])
                if qwen_image_initial_latents_raw:
                    cmd.extend([
                        "--initial-latents-raw", qwen_image_initial_latents_raw])
                output_target = output_png
            else:
                cmd = [
                    binary, "generate-video", bundle_path,
                    "--prompt", prompt,
                    "--num-steps", str(num_steps),
                ]
                if initial_latents_raw:
                    cmd.extend(["--initial-latents-raw", initial_latents_raw])
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
                    cmd, capture_output=True, text=True, timeout=3600, env=env)
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

    def _run_crossover_ref_t5_trt_dit(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Crossover: HF T5 text encoding -> TRT DiT denoising.

        Isolates DiT engine quality by feeding HF-produced text embeddings
        into the TRT denoiser. If this passes but end_to_end fails, the
        regression is in the T5 engine.
        """
        bundle_path = _resolve_bundle_path(case, ctx)
        model_id = case.hf_id
        prompt = case.inputs.get("prompt", "A cat sitting on a beach")
        python = ctx.runtime_python_path() or sys.executable

        script_code = f"""
import sys, json, time
sys.path.insert(0, {str(TOOLS_DIR)!r})
import numpy as np
import torch
import transformers

transformers.logging.set_verbosity_error()

# Step 1: HF T5 encoding
from diffusers import WanPipeline
try:
    pipe = WanPipeline.from_pretrained(
        {model_id!r}, torch_dtype=torch.float32, low_cpu_mem_usage=False)
except ValueError as e:
    if "keep_in_fp32_modules" not in str(e):
        raise
    pipe = WanPipeline.from_pretrained(
        {model_id!r}, torch_dtype=torch.float32, low_cpu_mem_usage=True)
if hasattr(pipe, "text_encoder"):
    te = pipe.text_encoder
    if hasattr(te, "tie_weights"):
        te.tie_weights()
    shared = getattr(te, "shared", None)
    embed = getattr(getattr(te, "encoder", None), "embed_tokens", None)
    if shared is not None and embed is not None and shared.weight.shape == embed.weight.shape:
        if shared.weight.data_ptr() != embed.weight.data_ptr():
            te.encoder.embed_tokens = te.shared
tokens = pipe.tokenizer(
    {prompt!r}, return_tensors="pt", padding="max_length",
    max_length=512, truncation=True)
with torch.no_grad():
    hf_t5_out = pipe.text_encoder(tokens.input_ids)[0].numpy()

# Step 2: TRT DiT with HF text embeddings
from tensorrt_model_connect.diffusion_runner import DiffusionRunner
from diffusion_helpers import load_bundle_config, project_text as project_text_np
from diffusion_helpers import compute_timestep_embedding as compute_timestep_embedding_np
from diffusion_helpers import load_pp_weights

runner = DiffusionRunner({bundle_path!r})
pp = load_pp_weights({bundle_path!r})
cfg = load_bundle_config({bundle_path!r})

text_proj = project_text_np(hf_t5_out, pp)
z_dim = cfg.get("z_dim", 16)
pt, ph, pw = cfg.get("patch_size", [1, 2, 2])
vh, vw, vf = cfg.get("video_height", 480), cfg.get("video_width", 832), cfg.get("video_num_frames", 17)
sft, sfs = cfg.get("scale_factor_temporal", 4), cfg.get("scale_factor_spatial", 8)
t_lat = (vf - 1) // sft + 1
h_lat, w_lat = vh // sfs, vw // sfs
nt, nh, nw = t_lat // pt, h_lat // ph, w_lat // pw

rng = np.random.default_rng(42)
latents = rng.standard_normal((1, z_dim, t_lat, h_lat, w_lat)).astype(np.float32)
rope_cos, rope_sin = runner._compute_3d_rope(nt, nh, nw, 128)
temb_6d, time_embed = compute_timestep_embedding_np(999.0, pp)
patches = runner._patchify(latents, [pt, ph, pw])
hidden = patches @ pp["patch_embedding.weight"] + pp["patch_embedding.bias"]

dit_out = runner._run_engine("denoiser", {{
    "hidden_states": hidden,
    "timestep_embedding": temb_6d.reshape(1, -1),
    "time_embed": time_embed.reshape(1, -1),
    "encoder_hidden_states": text_proj,
    "rotary_cos": rope_cos, "rotary_sin": rope_sin,
}})["output"]

result = {{
    "dit_output_shape": list(dit_out.shape),
    "dit_output_mean": float(dit_out.mean()),
    "dit_output_std": float(dit_out.std()),
    "dit_output_range": [float(dit_out.min()), float(dit_out.max())],
}}
print(json.dumps(result))
np.save("/tmp/crossover_ref_t5_trt_dit.npy", dit_out)
"""
        t0 = time.monotonic()
        result = subprocess.run(
            [python, "-c", script_code],
            capture_output=True, text=True, timeout=3600,
            env={**os.environ, "LD_LIBRARY_PATH": _build_ld_library_path(ctx)},
        )
        elapsed = time.monotonic() - t0

        stderr_truncated, stderr_log = save_full_stderr(
            result.stderr or "", ctx.artifacts_dir or "",
            "crossover_ref_t5_trt_dit", case.name)
        data: dict = {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": stderr_truncated,
            "crossover_type": "ref_t5_trt_dit",
        }
        if stderr_log:
            data["stderr_log"] = stderr_log
        try:
            import json as json_mod
            parsed = json_mod.loads(result.stdout.strip())
            data.update(parsed)
        except Exception:
            pass

        npy_path = "/tmp/crossover_ref_t5_trt_dit.npy"
        if os.path.exists(npy_path):
            data["output_path"] = npy_path

        return StageOutput(
            stage_name=stage.name,
            data=data,
            timing_s=elapsed,
            metadata={"command": "crossover_ref_t5_trt_dit"},
        )

    def _run_crossover_trt_t5_ref_dit(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Crossover: TRT T5 text encoding -> HF DiT denoising.

        Isolates T5 engine quality by feeding TRT-produced text embeddings
        into the HF denoiser. If this passes but end_to_end fails, the
        regression is in the DiT engine.
        """
        bundle_path = _resolve_bundle_path(case, ctx)
        model_id = case.hf_id
        prompt = case.inputs.get("prompt", "A cat sitting on a beach")
        python = ctx.runtime_python_path() or sys.executable

        script_code = f"""
import sys, json
sys.path.insert(0, {str(TOOLS_DIR)!r})
import numpy as np
import torch
import transformers

transformers.logging.set_verbosity_error()

# Step 1: TRT T5 encoding
from tensorrt_model_connect.diffusion_runner import DiffusionRunner
from diffusion_helpers import load_bundle_config

runner = DiffusionRunner({bundle_path!r})
cfg = load_bundle_config({bundle_path!r})

from transformers import AutoTokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained({model_id!r})
except (ValueError, OSError):
    tokenizer = AutoTokenizer.from_pretrained({model_id!r}, subfolder="tokenizer")
tokens = tokenizer(
    {prompt!r}, return_tensors="np", padding="max_length",
    max_length=512, truncation=True)
input_ids = tokens["input_ids"].astype(np.int32)
trt_t5_out = runner.encode_text(input_ids)

# Step 2: HF DiT with TRT text embeddings
from diffusers import WanPipeline
try:
    pipe = WanPipeline.from_pretrained(
        {model_id!r}, torch_dtype=torch.float32, low_cpu_mem_usage=False)
except ValueError as e:
    if "keep_in_fp32_modules" not in str(e):
        raise
    pipe = WanPipeline.from_pretrained(
        {model_id!r}, torch_dtype=torch.float32, low_cpu_mem_usage=True)
if hasattr(pipe, "text_encoder"):
    te = pipe.text_encoder
    if hasattr(te, "tie_weights"):
        te.tie_weights()
    shared = getattr(te, "shared", None)
    embed = getattr(getattr(te, "encoder", None), "embed_tokens", None)
    if shared is not None and embed is not None and shared.weight.shape == embed.weight.shape:
        if shared.weight.data_ptr() != embed.weight.data_ptr():
            te.encoder.embed_tokens = te.shared

z_dim = cfg.get("z_dim", 16)
vh, vw, vf = cfg.get("video_height", 480), cfg.get("video_width", 832), cfg.get("video_num_frames", 17)
sft, sfs = cfg.get("scale_factor_temporal", 4), cfg.get("scale_factor_spatial", 8)
t_lat = (vf - 1) // sft + 1
h_lat, w_lat = vh // sfs, vw // sfs

torch.manual_seed(42)
test_latent = torch.randn(1, z_dim, t_lat, h_lat, w_lat)
text_torch = torch.from_numpy(trt_t5_out.copy())
timestep = torch.tensor([999.0])

with torch.no_grad():
    hf_out = pipe.transformer(
        hidden_states=test_latent,
        timestep=timestep,
        encoder_hidden_states=text_torch,
    ).sample.numpy()

result = {{
    "dit_output_shape": list(hf_out.shape),
    "dit_output_mean": float(hf_out.mean()),
    "dit_output_std": float(hf_out.std()),
    "dit_output_range": [float(hf_out.min()), float(hf_out.max())],
}}
print(json.dumps(result))
np.save("/tmp/crossover_trt_t5_ref_dit.npy", hf_out)
"""
        t0 = time.monotonic()
        result = subprocess.run(
            [python, "-c", script_code],
            capture_output=True, text=True, timeout=3600,
            env={**os.environ, "LD_LIBRARY_PATH": _build_ld_library_path(ctx)},
        )
        elapsed = time.monotonic() - t0

        stderr_truncated, stderr_log = save_full_stderr(
            result.stderr or "", ctx.artifacts_dir or "",
            "crossover_trt_t5_ref_dit", case.name)
        data: dict = {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": stderr_truncated,
            "crossover_type": "trt_t5_ref_dit",
        }
        if stderr_log:
            data["stderr_log"] = stderr_log
        try:
            import json as json_mod
            parsed = json_mod.loads(result.stdout.strip())
            data.update(parsed)
        except Exception:
            pass

        npy_path = "/tmp/crossover_trt_t5_ref_dit.npy"
        if os.path.exists(npy_path):
            data["output_path"] = npy_path

        return StageOutput(
            stage_name=stage.name,
            data=data,
            timing_s=elapsed,
            metadata={"command": "crossover_trt_t5_ref_dit"},
        )

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
