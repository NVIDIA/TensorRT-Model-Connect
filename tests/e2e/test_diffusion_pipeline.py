# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E: Diffusion pipeline — build bundle, validate components, generate video, check quality.

Discovers manifests with test_type=="diffusion" and runs a multi-stage validation:

1. Build bundle from HF model via trtmc build (subprocess)
2. Run debug_diffusion_pipeline.py for 9-step TRT-vs-HF component comparison
3. Run C++ binary: generate-video (30 steps, PNG frames)
4. Check frame pixel statistics (catches washed-out / all-black / low-contrast)
5. Save results.json + sample frames alongside the bundle

Usage:
    pytest tests/e2e/test_diffusion_pipeline.py -v \
      --engine-dir /mnt/storage/tensorrt-model-connect/engines \
      --trtmc-binary ./build/trtmc --hf-python .venv/bin/python \
      --rebuild-engines
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from tests.e2e_harness.manifest_loader import iter_manifest_paths

PROJECT_DIR = Path(__file__).resolve().parents[2]
TOOLS_DIR = PROJECT_DIR / "tools"
MODELS_DIR = Path(__file__).resolve().parent / "models"


# ---------------------------------------------------------------------------
# Discover diffusion models from manifests
# ---------------------------------------------------------------------------

def _load_diffusion_models():
    """Load all model manifests with test_type=='diffusion'."""
    models = []
    for model_file in iter_manifest_paths(MODELS_DIR):
        with open(model_file) as f:
            entry = json.load(f)
        if entry.get("test_type") == "diffusion":
            models.append(entry)
    return models


def _diffusion_model_ids():
    return [m["name"] for m in _load_diffusion_models()]


def _diffusion_model_by_name(name):
    for m in _load_diffusion_models():
        if m["name"] == name:
            return m
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_diffusion_bundle(trtmc_binary, hf_id, bundle_path, build_args, precision="fp32"):
    """Build a diffusion .trtfb bundle as a subprocess."""
    cmd = [
        str(trtmc_binary), "build",
        hf_id, "-o", str(bundle_path),
    ]
    max_cache = build_args.get("max_cache_length", 256)
    cmd.extend(["--max-cache-length", str(max_cache)])
    if precision != "fp32":
        cmd.extend(["--precision", str(precision)])

    t0 = time.monotonic()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    elapsed = time.monotonic() - t0

    if result.returncode != 0:
        pytest.fail(
            f"Diffusion bundle build failed for {hf_id} (rc={result.returncode}):\n"
            f"{result.stderr[-2000:]}")

    return elapsed


def _run_debug_pipeline(bundle_path, model_id, num_steps):
    """Run debug_diffusion_pipeline.py as subprocess. Returns (passed, output, time_s)."""
    script = TOOLS_DIR / "debug_diffusion_pipeline.py"
    cmd = [
        sys.executable, str(script),
        "--bundle", str(bundle_path),
        "--model-id", model_id,
        "--num-steps", str(num_steps),
    ]

    t0 = time.monotonic()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    elapsed = time.monotonic() - t0

    return {
        "passed": result.returncode == 0,
        "returncode": result.returncode,
        "output": result.stdout,
        "stderr": result.stderr,
        "time_s": elapsed,
    }


def _run_cpp_generate_video(binary, bundle_path, prompt, output_dir,
                             num_steps, hf_python, ld_library_path):
    """Run C++ generate-video command. Returns (num_frames, time_s, stderr)."""
    cmd = [
        str(binary), "generate-video", str(bundle_path),
        "--prompt", prompt,
        "--output", str(output_dir),
        "--num-steps", str(num_steps),
    ]
    if hf_python:
        cmd.extend(["--hf-python", str(hf_python)])

    env = {"LD_LIBRARY_PATH": ld_library_path}

    t0 = time.monotonic()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, env=env)
    elapsed = time.monotonic() - t0

    # Parse frame count from stdout: "Generated N frames in DIR"
    num_frames = -1
    for line in result.stdout.splitlines():
        if line.startswith("Generated "):
            try:
                num_frames = int(line.split()[1])
            except (IndexError, ValueError):
                pass

    return {
        "num_frames": num_frames,
        "returncode": result.returncode,
        "time_s": elapsed,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _compute_frame_stats(frame_dir):
    """Load PNG frames from a directory, return aggregate pixel statistics.

    Returns dict with keys: count, mean, std, min, max.
    Pixel values are normalized to [0, 1].
    """
    from PIL import Image

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


def _get_gpu_name():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            return r.stdout.strip().split("\n")[0]
    except Exception:
        pass
    return "unknown"


def _save_results(bundle_path, results_dict):
    """Save results JSON next to the bundle file."""
    bundle_p = Path(bundle_path)
    results_path = bundle_p.with_suffix(".results.json")

    class _NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.bool_):
                return bool(obj)
            return super().default(obj)

    with open(results_path, "w") as f:
        json.dump(results_dict, f, indent=2, cls=_NumpyEncoder)

    return str(results_path)


def _copy_sample_frames(frame_dir, engine_dir, model_name, expected_frames):
    """Copy first/middle/last frames alongside the bundle for visual inspection."""
    frames = sorted(Path(frame_dir).glob("frame_*.png"))
    if not frames:
        return {}

    samples = {}
    indices = {
        "first": 0,
        "middle": len(frames) // 2,
        "last": len(frames) - 1,
    }

    for label, idx in indices.items():
        if idx < len(frames):
            src = frames[idx]
            dst = Path(engine_dir) / f"{model_name}.frame_{label}.png"
            shutil.copy2(str(src), str(dst))
            samples[label] = src.name

    return samples


# ---------------------------------------------------------------------------
# Fixture: diffusion model entry
# ---------------------------------------------------------------------------

@pytest.fixture(params=_diffusion_model_ids() or ["__no_diffusion_models__"])
def diffusion_entry(request, engine_dir, trtmc_binary):
    """Parametrized fixture yielding one diffusion model entry at a time."""
    name = request.param
    if name == "__no_diffusion_models__":
        pytest.skip("No diffusion models in manifests")

    entry = _diffusion_model_by_name(name)
    if entry is None:
        pytest.skip(f"Diffusion model not found: {name}")

    bundle_path = engine_dir / entry["bundle"]
    rebuild = request.config.getoption("--rebuild-engines")

    if bundle_path.is_file() and not rebuild:
        entry["bundle_path"] = str(bundle_path)
        entry["was_cached"] = True
        entry["build_time_s"] = None
        return entry

    # Build the bundle
    hf_id = entry["hf_id"]
    build_args = entry.get("build_args", {})
    precision = entry.get("precision", "fp32")
    build_time = _build_diffusion_bundle(
        trtmc_binary, hf_id, bundle_path, build_args, precision)

    entry["bundle_path"] = str(bundle_path)
    entry["was_cached"] = False
    entry["build_time_s"] = build_time
    return entry


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.e2e
def test_diffusion_build(diffusion_entry, engine_dir):
    """Build diffusion bundle from HF and verify it exists."""
    bundle_path = Path(diffusion_entry["bundle_path"])
    assert bundle_path.is_file(), f"Bundle not found: {bundle_path}"
    assert bundle_path.stat().st_size > 0, f"Bundle is empty: {bundle_path}"

    print(f"\n[diffusion_build] {diffusion_entry['name']}: "
          f"bundle={bundle_path.stat().st_size / (1024**2):.0f} MB"
          f"{', cached' if diffusion_entry['was_cached'] else ''}")


@pytest.mark.e2e
def test_diffusion_debug_pipeline(diffusion_entry):
    """Run debug_diffusion_pipeline.py — validates all 9 TRT-vs-HF component checks."""
    bundle_path = diffusion_entry["bundle_path"]
    hf_id = diffusion_entry["hf_id"]
    num_steps = diffusion_entry.get("num_inference_steps", 30)

    result = _run_debug_pipeline(bundle_path, hf_id, num_steps)

    assert result["passed"], (
        f"debug_diffusion_pipeline FAILED for {diffusion_entry['name']} "
        f"(rc={result['returncode']}):\n"
        f"{result['output'][-2000:]}\n"
        f"STDERR:\n{result['stderr'][-1000:]}")

    print(f"\n[diffusion_debug] {diffusion_entry['name']}: "
          f"9/9 PASS ({result['time_s']:.0f}s)")


@pytest.mark.e2e
def test_diffusion_cpp_generate(diffusion_entry, trtmc_binary, hf_python,
                                 ld_library_path, engine_dir):
    """Run C++ generate-video and verify correct frame count."""
    bundle_path = diffusion_entry["bundle_path"]
    prompt = diffusion_entry.get("test_prompt", "A cat sitting on a beach")
    num_steps = diffusion_entry.get("num_inference_steps", 30)
    expected_frames = diffusion_entry.get("video_num_frames", 17)

    with tempfile.TemporaryDirectory(prefix="trtmc_frames_") as frame_dir:
        result = _run_cpp_generate_video(
            trtmc_binary, bundle_path, prompt, frame_dir,
            num_steps, hf_python, ld_library_path)

        assert result["returncode"] == 0, (
            f"C++ generate-video failed (rc={result['returncode']}):\n"
            f"{result['stderr'][-2000:]}")
        assert result["num_frames"] == expected_frames, (
            f"Expected {expected_frames} frames, got {result['num_frames']}")

        # Verify PNG files actually exist
        frames = sorted(Path(frame_dir).glob("frame_*.png"))
        assert len(frames) == expected_frames, (
            f"Expected {expected_frames} PNG files, found {len(frames)}")

        print(f"\n[diffusion_cpp] {diffusion_entry['name']}: "
              f"{result['num_frames']} frames generated ({result['time_s']:.0f}s)")


@pytest.mark.e2e
def test_diffusion_frame_quality(diffusion_entry, trtmc_binary, hf_python,
                                  ld_library_path, engine_dir):
    """Generate frames and check pixel statistics for visual quality.

    Catches washed-out, all-black, all-white, or low-contrast output.
    """
    bundle_path = diffusion_entry["bundle_path"]
    prompt = diffusion_entry.get("test_prompt", "A cat sitting on a beach")
    num_steps = diffusion_entry.get("num_inference_steps", 30)
    expected_frames = diffusion_entry.get("video_num_frames", 17)
    min_mean = diffusion_entry.get("min_pixel_mean", 0.15)
    max_mean = diffusion_entry.get("max_pixel_mean", 0.85)
    min_std = diffusion_entry.get("min_pixel_std", 0.05)

    with tempfile.TemporaryDirectory(prefix="trtmc_quality_") as frame_dir:
        result = _run_cpp_generate_video(
            trtmc_binary, bundle_path, prompt, frame_dir,
            num_steps, hf_python, ld_library_path)

        if result["returncode"] != 0:
            pytest.fail(
                f"C++ generate-video failed (rc={result['returncode']}):\n"
                f"{result['stderr'][-2000:]}")

        stats = _compute_frame_stats(frame_dir)

        assert stats["count"] == expected_frames, (
            f"Expected {expected_frames} frames, got {stats['count']}")
        assert stats["mean"] >= min_mean, (
            f"Frame mean too low ({stats['mean']:.3f} < {min_mean}) — likely all-black")
        assert stats["mean"] <= max_mean, (
            f"Frame mean too high ({stats['mean']:.3f} > {max_mean}) — likely all-white")
        assert stats["std"] >= min_std, (
            f"Frame std too low ({stats['std']:.3f} < {min_std}) — likely washed-out or flat")

        # Copy sample frames for manual inspection
        samples = _copy_sample_frames(
            frame_dir, engine_dir, diffusion_entry["name"], expected_frames)

        # Save results.json
        results_dict = {
            "model_id": diffusion_entry["hf_id"],
            "test_type": "diffusion",
            "num_inference_steps": num_steps,
            "num_frames": stats["count"],
            "pixel_stats": {
                "mean": stats["mean"],
                "std": stats["std"],
                "min": stats["min"],
                "max": stats["max"],
            },
            "debug_pipeline_passed": True,
            "frame_quality_passed": True,
            "sample_frames": samples,
            "gpu_name": _get_gpu_name(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        results_path = _save_results(bundle_path, results_dict)

        print(f"\n[diffusion_quality] {diffusion_entry['name']}: PASS")
        print(f"  mean={stats['mean']:.3f}, std={stats['std']:.3f}")
        print(f"  Results saved: {results_path}")
