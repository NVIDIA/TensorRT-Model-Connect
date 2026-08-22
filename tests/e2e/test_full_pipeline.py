# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E: Full pipeline — build bundle, run inference (C++ & Python), compare to HF reference.

Exercises the complete flow: tensorrt_model_connect.build() -> C++ runtime -> Python TrtRunner -> HF
transformers, then saves a results JSON alongside the bundle for traceability.

All GPU-heavy work (TRT inference, HF model loading) runs in subprocesses to avoid OOM
when multiple engines share a single GPU.

Usage:
    # Full pipeline with fresh build (GPU required):
    pytest tests/e2e/test_full_pipeline.py -v \
      --engine-dir /mnt/storage/tensorrt-model-connect/engines --rebuild-engines

    # Use cached bundles (default):
    pytest tests/e2e/test_full_pipeline.py -v \
      --engine-dir /mnt/storage/tensorrt-model-connect/engines

    # With explicit binary and python paths:
    pytest tests/e2e/test_full_pipeline.py -v \
      --engine-dir /mnt/storage/tensorrt-model-connect/engines \
      --trtmc-binary ./build/trtmc \
      --hf-python .venv/bin/python
"""

from __future__ import annotations

import json
import os
import re
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

PROJECT_DIR = Path(__file__).resolve().parents[2]
TOOLS_DIR = PROJECT_DIR / "tools"

_SPEECH_RUNTIME_STRATEGIES = {
    "speech_to_text",
    "speech_to_text_rnnt",
    "speech_to_speech",
}

# Ensure tools/ is importable for compare_logits, run_hf, etc.
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_cpp_inference(binary, bundle_path, prompt, max_new_tokens,
                       ld_library_path, image_path=None):
    """Run C++ trtmc binary inference (subprocess — no in-process GPU usage).

    Returns:
        dict with keys: text, time_s, returncode, stderr
    """
    cmd = [
        str(binary), "run", str(bundle_path),
        "--prompt", prompt,
        "--max-new-tokens", str(max_new_tokens),
    ]
    if image_path:
        cmd.extend(["--image", str(image_path)])

    env = {"LD_LIBRARY_PATH": ld_library_path}

    t0 = time.monotonic()
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=300, env=env)
    elapsed = time.monotonic() - t0

    return {
        "text": result.stdout.strip(),
        "time_s": elapsed,
        "returncode": result.returncode,
        "stderr": result.stderr,
    }


def _run_diff_logits_subprocess(hf_id, atol, max_cache_length, max_new_tokens,
                                trust_remote_code=False):
    """Run diff_logits.py as a subprocess (HF model + TRT engine in separate process).

    Returns:
        dict with keys: passed, max_diff, returncode, output
    """
    diff_logits = TOOLS_DIR / "diff_logits.py"
    cmd = [
        sys.executable, str(diff_logits),
        "--model", hf_id,
        "--atol", str(atol),
        "--max-cache-length", str(max_cache_length),
        "--max-new-tokens", str(max_new_tokens),
        "--battery",
    ]
    if trust_remote_code:
        cmd.append("--trust-remote-code")

    t0 = time.monotonic()
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=1800)
    elapsed = time.monotonic() - t0

    # Parse max_abs_logit_diff from output
    max_diff = None
    for line in result.stdout.splitlines():
        if "max_abs_logit_diff" in line:
            try:
                max_diff = float(line.split(":")[-1].strip())
            except ValueError:
                pass

    return {
        "passed": result.returncode == 0,
        "max_diff": max_diff,
        "returncode": result.returncode,
        "output": result.stdout,
        "stderr": result.stderr,
        "time_s": elapsed,
    }


def _run_diff_vl_subprocess(bundle_path, image_path, hf_id, binary,
                            ld_library_path, atol=0.1, max_new_tokens=20):
    """Run diff_vl.py as a subprocess (all GPU work isolated).

    Returns:
        dict with keys: passed, returncode, output, time_s
    """
    diff_vl = TOOLS_DIR / "diff_vl.py"
    cmd = [
        sys.executable, str(diff_vl),
        "--bundle", str(bundle_path),
        "--image", str(image_path),
        "--model", hf_id,
        "--atol", str(atol),
        "--max-new-tokens", str(max_new_tokens),
        "--binary", str(binary),
    ]

    env = {"LD_LIBRARY_PATH": ld_library_path}

    t0 = time.monotonic()
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=1800, env=env)
    elapsed = time.monotonic() - t0

    return {
        "passed": result.returncode == 0,
        "returncode": result.returncode,
        "output": result.stdout,
        "stderr": result.stderr,
        "time_s": elapsed,
    }


def _run_perf_compare_subprocess(hf_id, bundle_path, prompt, max_new_tokens,
                                  trust_remote_code=False,
                                  warmup=2, iterations=3):
    """Run perf_compare.py as a subprocess (TRT + HF both in separate process).

    Returns:
        dict with keys: passed, perf_data (parsed JSON or None), time_s, output
    """
    perf_compare = TOOLS_DIR / "perf_compare.py"
    json_path = Path("/tmp/claude") / f"{Path(bundle_path).stem}_perf.json"

    cmd = [
        sys.executable, str(perf_compare),
        "--model", hf_id,
        "--bundle", str(bundle_path),
        "--prompt", prompt,
        "--max-new-tokens", str(max_new_tokens),
        "--warmup", str(warmup),
        "--iterations", str(iterations),
        "--json", str(json_path),
    ]
    if trust_remote_code:
        cmd.append("--trust-remote-code")

    t0 = time.monotonic()
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=1800)
    elapsed = time.monotonic() - t0

    perf_data = None
    if json_path.is_file():
        try:
            with open(json_path) as f:
                perf_data = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    return {
        "passed": result.returncode == 0,
        "perf_data": perf_data,
        "time_s": elapsed,
        "output": result.stdout,
        "stderr": result.stderr,
    }


def _get_gpu_name():
    """Get GPU name via nvidia-smi, or return 'unknown'."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            return r.stdout.strip().split("\n")[0]
    except Exception:
        pass
    return "unknown"


def _get_trt_version():
    """Get TensorRT version string, or 'unknown'."""
    try:
        import tensorrt as trt
        return trt.__version__
    except Exception:
        return "unknown"


def _save_results(bundle_path, results_dict):
    """Save results JSON next to the bundle file.

    Writes <bundle_stem>.results.json in the same directory as the bundle.
    Handles numpy types for JSON serialization.
    """
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.e2e
def test_full_pipeline(built_bundle, trtmc_binary, ld_library_path):
    """Full pipeline: build -> C++ inference -> diff_logits (HF vs Python runner)."""
    entry = built_bundle["entry"]
    bundle_path = built_bundle["path"]

    runtime_strategy = str(entry.get("runtime_strategy") or "")

    # Skip VL models (handled by test_full_pipeline_vlm)
    if runtime_strategy.endswith("_vision_language"):
        pytest.skip("VL model — use test_full_pipeline_vlm")

    # Skip diffusion models (handled by test_diffusion_pipeline)
    if entry.get("test_type") == "diffusion":
        pytest.skip("Diffusion model — use test_diffusion_pipeline")

    # Skip segmentation models (handled by test_segmentation_pipeline)
    if entry.get("test_type") == "segmentation" or entry.get("task_strategy") == "segmentation":
        pytest.skip("Segmentation model — use test_segmentation_pipeline")

    # Skip audio models (handled by test_audio_pipeline)
    if (
        entry.get("test_type") == "audio"
        or entry.get("test_type") == "transcription"
        or entry.get("runtime_strategy") == "text_to_audio"
        or entry.get("runtime_strategy") in _SPEECH_RUNTIME_STRATEGIES
    ):
        pytest.skip("Audio model — use test_audio_pipeline")

    # Skip gated models (require HF auth for diff_logits)
    if entry.get("gated"):
        pytest.skip("Gated model — requires HF authentication")

    prompt = entry.get("prompt", "The capital of France is")
    max_new_tokens = entry.get("max_new_tokens", 30)
    atol = entry.get("logit_atol", 1e-3)
    max_cache_length = entry.get("max_cache_length", 256)
    trust_remote_code = entry.get("trust_remote_code", False)
    hf_id = entry["hf_id"]

    # Step 1: C++ binary inference (subprocess)
    cpp_result = _run_cpp_inference(
        trtmc_binary, bundle_path, prompt, max_new_tokens,
        ld_library_path)
    assert cpp_result["returncode"] == 0, (
        f"C++ inference failed (rc={cpp_result['returncode']}):\n"
        f"{cpp_result['stderr']}")
    assert len(cpp_result["text"]) > 0, "C++ inference produced no output"

    # Step 2: diff_logits — HF vs Python TrtRunner (subprocess, memory-isolated)
    diff_result = _run_diff_logits_subprocess(
        hf_id, atol, max_cache_length, max_new_tokens,
        trust_remote_code=trust_remote_code)

    # Step 3: perf_compare — TRT vs HF performance (subprocess, informational)
    # Skip for SSM/Mamba models (perf_compare rejects them)
    assert runtime_strategy, "model entry must declare runtime_strategy"
    perf_result = None
    if runtime_strategy != "mamba_ssm_recurrent":
        perf_result = _run_perf_compare_subprocess(
            hf_id, bundle_path, prompt, max_new_tokens,
            trust_remote_code=trust_remote_code)
        if not perf_result["passed"]:
            print(f"\n[full_pipeline] WARNING: perf_compare failed for "
                  f"{entry['name']} (non-fatal):\n"
                  f"{perf_result['output'][-1000:]}")

    # Step 4: Build and save results
    bundle_size = os.path.getsize(bundle_path)
    results = {
        "model": {
            "name": entry["name"],
            "hf_id": hf_id,
            "family": entry.get("family", "unknown"),
            "runtime_strategy": runtime_strategy,
        },
        "build": {
            "was_cached": built_bundle["was_cached"],
            "build_time_s": built_bundle["build_time_s"],
            "engine_size_bytes": bundle_size,
            "max_cache_length": max_cache_length,
            "trt_version": _get_trt_version(),
            "gpu_name": _get_gpu_name(),
        },
        "inference": {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "generated_text_cpp": cpp_result["text"],
        },
        "comparison": {
            "logit_atol": atol,
            "logit_parity_pass": diff_result["passed"],
            "max_logit_diff": diff_result["max_diff"],
        },
        "timing": {
            "build_time_s": built_bundle["build_time_s"],
            "cpp_inference_time_s": cpp_result["time_s"],
            "diff_logits_time_s": diff_result["time_s"],
        },
        "status": "PASS" if diff_result["passed"] else "FAIL",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Merge perf data if available
    if perf_result and perf_result["perf_data"]:
        results["performance"] = perf_result["perf_data"]
        results["timing"]["perf_compare_time_s"] = perf_result["time_s"]

    results_path = _save_results(bundle_path, results)

    # Assertions (correctness only — perf is informational)
    assert diff_result["passed"], (
        f"diff_logits FAILED for {entry['name']}:\n"
        f"{diff_result['output'][-2000:]}")

    perf_summary = ""
    if perf_result and perf_result["perf_data"]:
        sp = perf_result["perf_data"].get("speedup", {})
        total_sp = sp.get("total")
        if total_sp is not None:
            perf_summary = f", speedup={total_sp:.2f}x"
    print(f"\n[full_pipeline] {entry['name']}: PASS "
          f"(max_diff={diff_result['max_diff']}, "
          f"cpp_text={len(cpp_result['text'])} chars{perf_summary})")
    print(f"  Results saved: {results_path}")


@pytest.mark.e2e
def test_full_pipeline_vlm(built_bundle, trtmc_binary, ld_library_path,
                           engine_dir):
    """Full VL pipeline: build -> C++ inference -> diff_vl (HF vs Python runner)."""
    entry = built_bundle["entry"]
    bundle_path = built_bundle["path"]

    if not str(entry.get("runtime_strategy") or "").endswith("_vision_language"):
        pytest.skip(f"{entry['name']} is not a VL model")

    test_image = entry.get("test_image")
    if not test_image:
        pytest.skip(f"No test_image configured for {entry['name']}")

    # Resolve image path (relative to engine_dir)
    image_path = Path(test_image)
    if not image_path.is_absolute():
        image_path = engine_dir / test_image
    if not image_path.is_file():
        pytest.skip(f"Test image not found: {image_path}")

    prompt = entry.get("prompt", "Describe this image.")
    max_new_tokens = entry.get("max_new_tokens", 30)
    atol = entry.get("logit_atol", 1e-3)
    hf_id = entry["hf_id"]

    # Step 1: C++ binary with image (subprocess)
    cpp_result = _run_cpp_inference(
        trtmc_binary, bundle_path, prompt, max_new_tokens,
        ld_library_path, image_path=str(image_path))
    assert cpp_result["returncode"] == 0, (
        f"C++ VL inference failed (rc={cpp_result['returncode']}):\n"
        f"{cpp_result['stderr']}")
    assert len(cpp_result["text"]) > 0, "C++ VL inference produced no output"

    # Step 2: diff_vl — HF vs Python VLTrtRunner (subprocess, memory-isolated)
    diff_result = _run_diff_vl_subprocess(
        bundle_path, str(image_path), hf_id, trtmc_binary,
        ld_library_path, atol=atol, max_new_tokens=max_new_tokens)

    # Step 3: Build and save results
    bundle_size = os.path.getsize(bundle_path)
    results = {
        "model": {
            "name": entry["name"],
            "hf_id": hf_id,
            "family": entry.get("family", "unknown"),
            "runtime_strategy": "qwen_vl_vision_language",
        },
        "build": {
            "was_cached": built_bundle["was_cached"],
            "build_time_s": built_bundle["build_time_s"],
            "engine_size_bytes": bundle_size,
            "max_cache_length": entry.get("max_cache_length", 256),
            "trt_version": _get_trt_version(),
            "gpu_name": _get_gpu_name(),
        },
        "inference": {
            "prompt": prompt,
            "test_image": str(image_path),
            "max_new_tokens": max_new_tokens,
            "generated_text_cpp": cpp_result["text"],
        },
        "comparison": {
            "diff_vl_pass": diff_result["passed"],
        },
        "timing": {
            "build_time_s": built_bundle["build_time_s"],
            "cpp_inference_time_s": cpp_result["time_s"],
            "diff_vl_time_s": diff_result["time_s"],
        },
        "status": "PASS" if diff_result["passed"] else "FAIL",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    results_path = _save_results(bundle_path, results)

    # Assertions
    assert diff_result["passed"], (
        f"diff_vl FAILED for {entry['name']}:\n"
        f"{diff_result['output'][-2000:]}")

    print(f"\n[full_pipeline_vlm] {entry['name']}: PASS")
    print(f"  C++ output: {cpp_result['text'][:200]}")
    print(f"  Results saved: {results_path}")


def _run_segmentation_hf_subprocess(hf_id, image_path, output_path):
    """Run HF SegFormer inference in a subprocess, return per-pixel class map.

    Writes a .npy file with shape [H, W] of int32 class indices.
    Returns dict with keys: returncode, output, stderr, time_s
    """
    script = (
        "import sys, numpy as np, torch\n"
        "from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor\n"
        "from PIL import Image\n"
        "hf_id, img_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]\n"
        "proc = SegformerImageProcessor.from_pretrained(hf_id)\n"
        "model = SegformerForSemanticSegmentation.from_pretrained(hf_id).eval()\n"
        "img = Image.open(img_path).convert('RGB')\n"
        "inputs = proc(images=img, return_tensors='pt')\n"
        "with torch.no_grad():\n"
        "    logits = model(**inputs).logits\n"
        "pred = logits.argmax(dim=1)[0].numpy().astype(np.int32)\n"
        "np.save(out_path, pred)\n"
        "print(f'shape={pred.shape} classes={len(np.unique(pred))}')\n"
    )

    t0 = time.monotonic()
    result = subprocess.run(
        [sys.executable, "-c", script, hf_id, str(image_path), str(output_path)],
        capture_output=True, text=True, timeout=300)
    elapsed = time.monotonic() - t0

    return {
        "returncode": result.returncode,
        "output": result.stdout,
        "stderr": result.stderr,
        "time_s": elapsed,
    }


@pytest.mark.e2e
def test_segmentation_pipeline(built_bundle, trtmc_binary,
                               ld_library_path, engine_dir):
    """Full segmentation pipeline: build -> C++ segment -> compare with HF."""
    entry = built_bundle["entry"]
    bundle_path = built_bundle["path"]

    if entry.get("task_strategy") != "segmentation":
        pytest.skip(f"{entry['name']} is not a segmentation model")

    test_image = entry.get("test_image")
    if not test_image:
        pytest.skip(f"No test_image configured for {entry['name']}")

    image_path = Path(test_image)
    if not image_path.is_absolute():
        # Resolve relative to tests/e2e/ directory
        e2e_dir = Path(__file__).resolve().parent
        image_path = e2e_dir / test_image
    if not image_path.is_file():
        pytest.skip(f"Test image not found: {image_path}")

    output_path = Path("/tmp/claude") / f"{entry['name']}_seg_output.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: C++ segmentation
    cmd = [
        str(trtmc_binary), "segment", str(bundle_path),
        "--image", str(image_path),
        "--output", str(output_path),
    ]

    env = {"LD_LIBRARY_PATH": ld_library_path}
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)

    assert result.returncode == 0, (
        f"Segmentation failed (rc={result.returncode}):\n{result.stderr}")
    assert output_path.is_file(), "Segmentation output not written"

    # Step 2: HF reference (subprocess to isolate GPU memory)
    hf_npy_path = Path("/tmp/claude") / f"{entry['name']}_hf_seg.npy"
    hf_result = _run_segmentation_hf_subprocess(
        entry["hf_id"], str(image_path), str(hf_npy_path))
    assert hf_result["returncode"] == 0, (
        f"HF segmentation failed:\n{hf_result['stderr']}")

    # Step 3: Compare C++ output vs HF
    from PIL import Image as PILImage
    trt_class_img = np.array(PILImage.open(output_path).convert("L"))
    hf_class_map = np.load(str(hf_npy_path))  # [H, W]

    # Upsample TRT (128x128) to HF resolution via nearest-neighbor
    trt_pil = PILImage.fromarray(trt_class_img)
    trt_upsampled = np.array(
        trt_pil.resize((hf_class_map.shape[1], hf_class_map.shape[0]),
                       PILImage.NEAREST)).astype(np.int32)

    pixel_agreement = float((trt_upsampled == hf_class_map).mean())
    min_agreement = entry.get("min_pixel_agreement", 0.95)

    # Save results
    results = {
        "model": {
            "name": entry["name"],
            "hf_id": entry["hf_id"],
            "family": entry.get("family", "unknown"),
            "runtime_strategy": entry.get("runtime_strategy"),
        },
        "build": {
            "was_cached": built_bundle["was_cached"],
            "build_time_s": built_bundle["build_time_s"],
            "engine_size_bytes": os.path.getsize(bundle_path),
            "trt_version": _get_trt_version(),
            "gpu_name": _get_gpu_name(),
        },
        "comparison": {
            "pixel_agreement": pixel_agreement,
            "min_pixel_agreement": min_agreement,
            "trt_output_shape": list(trt_class_img.shape),
            "hf_output_shape": list(hf_class_map.shape),
            "trt_unique_classes": int(len(np.unique(trt_class_img))),
            "hf_unique_classes": int(len(np.unique(hf_class_map))),
        },
        "status": "PASS" if pixel_agreement >= min_agreement else "FAIL",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    results_path = _save_results(bundle_path, results)

    assert pixel_agreement >= min_agreement, (
        f"Pixel agreement {pixel_agreement:.4f} < {min_agreement} "
        f"for {entry['name']}")

    print(f"\n[segmentation_pipeline] {entry['name']}: PASS")
    print(f"  Pixel agreement: {pixel_agreement:.4f} "
          f"(min={min_agreement})")
    print(f"  TRT classes: {int(len(np.unique(trt_class_img)))}, "
          f"HF classes: {int(len(np.unique(hf_class_map)))}")
    print(f"  Results saved: {results_path}")


def _run_segmentation_engine_parity_subprocess(hf_id, image_path):
    """Build debug engine, preprocess with HF, run both TRT and HF, compare.

    Returns dict with keys: returncode, pixel_agreement, cosine, max_diff,
    output, stderr, time_s.
    """
    script = (
        "import sys, numpy as np, torch\n"
        "from transformers import (SegformerForSemanticSegmentation,\n"
        "                          SegformerImageProcessor)\n"
        "from PIL import Image\n"
        "hf_id, img_path = sys.argv[1], sys.argv[2]\n"
        "proc = SegformerImageProcessor.from_pretrained(hf_id)\n"
        "img = Image.open(img_path).convert('RGB')\n"
        "inputs = proc(images=img, return_tensors='pt')\n"
        "pv = inputs['pixel_values']\n"
        "# HF\n"
        "model = SegformerForSemanticSegmentation.from_pretrained(\n"
        "    hf_id, torch_dtype=torch.float32).eval()\n"
        "with torch.no_grad():\n"
        "    hf_logits = model(pv).logits[0].numpy()\n"
        "del model\n"
        "# TRT\n"
        "from tensorrt_model_connect.engine_builder import _resolve_model\n"
        "from tensorrt_model_connect.config import ModelConfig\n"
        "from tensorrt_model_connect.families import find_plugin\n"
        "model_dir = _resolve_model(hf_id)\n"
        "config = ModelConfig.from_dir(model_dir)\n"
        "plugin = find_plugin(config.model_type)\n"
        "weights = plugin.load_weights(model_dir, config)\n"
        "engine_plan = plugin.build_engine(config, weights, 0)\n"
        "import tensorrt as trt\n"
        "try:\n"
        "    from cuda.bindings import runtime as cudart\n"
        "except ImportError:\n"
        "    from cuda import cudart\n"
        "def chk(s):\n"
        "    ok = getattr(cudart,'cudaError_t',type(None))\n"
        "    if hasattr(ok,'cudaSuccess') and s!=ok.cudaSuccess: raise RuntimeError(s)\n"
        "    elif not hasattr(ok,'cudaSuccess') and s!=0: raise RuntimeError(s)\n"
        "rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))\n"
        "engine = rt.deserialize_cuda_engine(engine_plan)\n"
        "ctx = engine.create_execution_context()\n"
        "pv_np = pv.numpy().astype(np.float32)\n"
        "out_shape = tuple(engine.get_tensor_shape('logits'))\n"
        "trt_out = np.zeros(out_shape, dtype=np.float32)\n"
        "e,stream = cudart.cudaStreamCreate(); chk(e)\n"
        "H2D = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice\n"
        "D2H = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost\n"
        "e,d_in = cudart.cudaMalloc(pv_np.nbytes); chk(e)\n"
        "e,d_out = cudart.cudaMalloc(trt_out.nbytes); chk(e)\n"
        "cudart.cudaMemcpyAsync(d_in,pv_np.ctypes.data,pv_np.nbytes,H2D,stream)\n"
        "ctx.set_tensor_address('pixel_values',d_in)\n"
        "ctx.set_tensor_address('logits',d_out)\n"
        "ctx.execute_async_v3(stream)\n"
        "cudart.cudaMemcpyAsync(trt_out.ctypes.data,d_out,trt_out.nbytes,D2H,stream)\n"
        "cudart.cudaStreamSynchronize(stream)\n"
        "trt_logits = trt_out[0]\n"
        "cudart.cudaFree(d_in); cudart.cudaFree(d_out)\n"
        "cudart.cudaStreamDestroy(stream)\n"
        "# Compare\n"
        "diff = np.abs(hf_logits - trt_logits)\n"
        "hf_pred = np.argmax(hf_logits, axis=0)\n"
        "trt_pred = np.argmax(trt_logits, axis=0)\n"
        "agree = float((hf_pred == trt_pred).mean())\n"
        "cos = float(np.dot(hf_logits.flatten(),trt_logits.flatten()) / \n"
        "            (np.linalg.norm(hf_logits.flatten())*np.linalg.norm(trt_logits.flatten())+1e-12))\n"
        "print(f'pixel_agreement={agree}')\n"
        "print(f'cosine={cos}')\n"
        "print(f'max_diff={float(diff.max())}')\n"
    )

    t0 = time.monotonic()
    result = subprocess.run(
        [sys.executable, "-c", script, hf_id, str(image_path)],
        capture_output=True, text=True, timeout=600)
    elapsed = time.monotonic() - t0

    # Parse metrics from stdout
    metrics = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, val = line.split("=", 1)
            try:
                metrics[key.strip()] = float(val.strip())
            except ValueError:
                pass

    return {
        "returncode": result.returncode,
        "pixel_agreement": metrics.get("pixel_agreement"),
        "cosine": metrics.get("cosine"),
        "max_diff": metrics.get("max_diff"),
        "output": result.stdout,
        "stderr": result.stderr,
        "time_s": elapsed,
    }


@pytest.mark.e2e
def test_segmentation_engine_parity(built_bundle, engine_dir):
    """Engine parity: same HF-preprocessed input -> TRT vs HF logits match."""
    entry = built_bundle["entry"]

    if entry.get("task_strategy") != "segmentation":
        pytest.skip(f"{entry['name']} is not a segmentation model")

    test_image = entry.get("test_image")
    if not test_image:
        pytest.skip(f"No test_image configured for {entry['name']}")

    image_path = Path(test_image)
    if not image_path.is_absolute():
        e2e_dir = Path(__file__).resolve().parent
        image_path = e2e_dir / test_image
    if not image_path.is_file():
        pytest.skip(f"Test image not found: {image_path}")

    result = _run_segmentation_engine_parity_subprocess(
        entry["hf_id"], str(image_path))

    assert result["returncode"] == 0, (
        f"Engine parity subprocess failed:\n{result['stderr']}")

    pixel_agreement = result["pixel_agreement"]
    cosine = result["cosine"]
    max_diff = result["max_diff"]

    assert pixel_agreement is not None, "Failed to parse pixel_agreement"
    assert pixel_agreement >= 0.99, (
        f"Engine parity pixel agreement {pixel_agreement:.4f} < 0.99 "
        f"for {entry['name']}")
    assert cosine is not None and cosine >= 0.9999, (
        f"Engine parity cosine {cosine} < 0.9999 for {entry['name']}")

    print(f"\n[segmentation_engine_parity] {entry['name']}: PASS")
    print(f"  Pixel agreement: {pixel_agreement:.4f}")
    print(f"  Cosine similarity: {cosine:.6f}")
    print(f"  Max logit diff: {max_diff:.4f}")


def _run_diff_audio_stage4_subprocess(hf_id, prompt, json_path,
                                      max_semantic_tokens=100):
    """Run diff_audio.py --stage 4 as a subprocess (GPU-isolated).

    Returns:
        dict with keys: passed, returncode, output, stderr, time_s,
                        stage_data (parsed JSON or None)
    """
    diff_audio = TOOLS_DIR / "diff_audio.py"
    cmd = [
        sys.executable, str(diff_audio),
        "--model", hf_id,
        "--prompt", prompt,
        "--stage", "4",
        "--max-semantic-tokens", str(max_semantic_tokens),
        "--json", str(json_path),
    ]

    t0 = time.monotonic()
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=1800)
    elapsed = time.monotonic() - t0

    # Parse structured results from JSON
    stage_data = None
    if Path(json_path).is_file():
        try:
            with open(json_path) as f:
                stage_data = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    return {
        "passed": result.returncode == 0,
        "returncode": result.returncode,
        "output": result.stdout,
        "stderr": result.stderr,
        "time_s": elapsed,
        "stage_data": stage_data,
    }


def _run_cpp_audio_generation(binary, bundle_path, prompt, output_wav,
                               ld_library_path):
    """Run C++ generate-audio and return (success, file_size)."""
    cmd = [
        str(binary), "generate-audio", str(bundle_path),
        "--prompt", prompt,
        "--output", str(output_wav),
    ]

    env = {"LD_LIBRARY_PATH": ld_library_path}
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=600, env=env)

    if result.returncode != 0:
        return False, 0, result.stderr
    if not Path(output_wav).is_file():
        return False, 0, "No output WAV file"

    file_size = os.path.getsize(output_wav)
    return True, file_size, result.stderr


def _load_bundle_sections(bundle_path: str) -> tuple[dict, list[tuple[str, bytes]]]:
    """Load a .bundle artifact header and all raw sections."""
    with open(bundle_path, "rb") as f:
        magic = f.read(8)
        if magic != b"BUNDLE\x01\x00":
            raise ValueError(f"Not a valid .bundle artifact: {bundle_path}")

        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
        sections_meta = header.get("sections", {})
        data_start = 16 + header_len

        sections: list[tuple[str, bytes]] = []
        for name, meta in sections_meta.items():
            offset = int(meta["offset"])
            size = int(meta["size"])
            f.seek(data_start + offset)
            sections.append((name, f.read(size)))
    return header, sections


def _write_bundle_sections(
    out_path: str | Path, header: dict, sections: list[tuple[str, bytes]],
) -> None:
    """Write a .bundle artifact from header metadata + raw section bytes."""
    section_meta: dict[str, dict[str, int]] = {}
    offset = 0
    for name, data in sections:
        section_meta[name] = {"offset": offset, "size": len(data)}
        offset += len(data)

    out_header = dict(header)
    out_header["sections"] = section_meta
    header_json = json.dumps(out_header, indent=2).encode("utf-8")

    with open(out_path, "wb") as f:
        f.write(b"BUNDLE\x01\x00")
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)
        for _, data in sections:
            f.write(data)


def _make_speech_greedy_bundle(
    src_bundle: str, dst_bundle: str | Path,
) -> None:
    """Create a temp bundle copy with deterministic speech decode settings."""
    header, sections = _load_bundle_sections(src_bundle)
    new_sections: list[tuple[str, bytes]] = []
    patched = False

    for name, data in sections:
        if name != "config.json":
            new_sections.append((name, data))
            continue

        cfg_text = data.decode("utf-8", errors="ignore").rstrip("\x00").strip()
        cfg = json.loads(cfg_text)
        cfg["speech_depth_temperature"] = 0.0
        cfg["speech_depth_top_k"] = 0
        cfg["speech_system_prompt"] = ""
        cfg["speech_text_prompt_ids"] = []
        new_sections.append((name, json.dumps(cfg, indent=2).encode("utf-8")))
        patched = True

    if not patched:
        raise RuntimeError("Bundle missing config.json section; cannot patch speech decode mode")

    _write_bundle_sections(dst_bundle, header, new_sections)


def _run_cpp_speech_generation(binary, bundle_path, input_wav, output_wav,
                               max_frames, ld_library_path):
    """Run C++ speak command and return subprocess result metadata."""
    cmd = [
        str(binary), "speak", str(bundle_path),
        "--audio-in", str(input_wav),
        "--audio-out", str(output_wav),
        "--max-new-tokens", str(max_frames),
    ]

    env = {"LD_LIBRARY_PATH": ld_library_path}
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=1200, env=env)

    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _parse_speech_output_tokens(stderr_text: str) -> np.ndarray:
    """Parse `[speech] Output frame N: ...` tokens from C++ stderr."""
    per_frame: dict[int, list[int]] = {}
    for line in stderr_text.splitlines():
        m = re.match(r"^\[speech\]\s+Output frame\s+(\d+):\s*(.*)$", line.strip())
        if not m:
            continue
        frame_idx = int(m.group(1))
        token_text = m.group(2).strip()
        if not token_text:
            continue
        tokens = [int(x) for x in token_text.split()]
        per_frame[frame_idx] = tokens

    if not per_frame:
        return np.zeros((0, 0), dtype=np.int32)

    max_frame = max(per_frame.keys())
    num_cb = len(per_frame[min(per_frame.keys())])
    out = np.zeros((max_frame + 1, num_cb), dtype=np.int32)
    for fidx, toks in per_frame.items():
        if len(toks) == num_cb:
            out[fidx, :] = np.asarray(toks, dtype=np.int32)
    return out


def _read_wav_f32(path: str | Path) -> np.ndarray:
    """Read WAV PCM16/float32 and return mono float32 samples in [-1, 1]."""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 44 or data[:4] != b"RIFF":
        return np.zeros((0,), dtype=np.float32)

    pos = 12
    while pos + 8 <= len(data):
        chunk_id = data[pos:pos + 4]
        chunk_size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        if pos + 8 + chunk_size > len(data):
            break
        if chunk_id == b"data":
            payload = data[pos + 8:pos + 8 + chunk_size]
            fmt_pos = data.find(b"fmt ")
            if fmt_pos >= 0 and fmt_pos + 24 <= len(data):
                fmt_tag = struct.unpack("<H", data[fmt_pos + 8:fmt_pos + 10])[0]
                bits = struct.unpack("<H", data[fmt_pos + 22:fmt_pos + 24])[0]
                if fmt_tag == 1 and bits == 16:
                    return (
                        np.frombuffer(payload, dtype=np.int16).astype(np.float32) / 32768.0
                    )
                if fmt_tag == 3 and bits == 32:
                    return np.frombuffer(payload, dtype=np.float32)
            break
        pos += 8 + chunk_size + (chunk_size & 1)
    return np.zeros((0,), dtype=np.float32)


def _speech_token_match_metrics(
    reference: np.ndarray, actual: np.ndarray,
) -> tuple[float, float, int]:
    """Return token_match, frame_exact, and compared frame count."""
    if reference.ndim != 2 or actual.ndim != 2:
        return 0.0, 0.0, 0
    n_frames = min(reference.shape[0], actual.shape[0])
    n_cb = min(reference.shape[1], actual.shape[1])
    if n_frames <= 0 or n_cb <= 0:
        return 0.0, 0.0, 0

    ref = reference[:n_frames, :n_cb]
    got = actual[:n_frames, :n_cb]
    eq = (ref == got)
    token_match = float(eq.mean())
    frame_exact = float(eq.all(axis=1).mean())
    return token_match, frame_exact, int(n_frames)


@pytest.mark.e2e
def test_bark_stage_parity(built_bundle, trtmc_binary,
                           ld_library_path):
    """Bark stage-by-stage greedy parity: TRT engine vs HF, per stage.

    1. Runs diff_audio.py --stage 4 (builds TRT engines from HF, greedy
       parity for all 4 Bark stages) as a subprocess.
    2. Runs C++ generate-audio to produce a .wav file alongside the bundle.
    3. Saves a results.json next to the bundle with per-stage metrics.
    """
    entry = built_bundle["entry"]
    bundle_path = built_bundle["path"]

    if entry.get("runtime_strategy") != "text_to_audio":
        pytest.skip(f"{entry['name']} is not an audio model")

    hf_id = entry["hf_id"]
    prompt = entry.get("prompt", "Hello, this is a test.")

    # Step 1: Stage 4 parity (subprocess — GPU-isolated)
    json_path = Path("/tmp/claude") / f"{entry['name']}_stage4.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    parity_result = _run_diff_audio_stage4_subprocess(
        hf_id, prompt, json_path, max_semantic_tokens=100)

    # Step 2: C++ generate-audio — produce a .wav next to the bundle
    bundle_p = Path(bundle_path)
    wav_path = bundle_p.with_suffix(".wav")
    cpp_ok, wav_size, cpp_stderr = _run_cpp_audio_generation(
        trtmc_binary, bundle_path, prompt, str(wav_path),
        ld_library_path)

    # Step 3: Build and save results.json next to the bundle
    stage_data = parity_result.get("stage_data") or {}
    results = {
        "model": {
            "name": entry["name"],
            "hf_id": hf_id,
            "family": entry.get("family", "unknown"),
            "runtime_strategy": "text_to_audio",
        },
        "build": {
            "was_cached": built_bundle["was_cached"],
            "build_time_s": built_bundle["build_time_s"],
            "engine_size_bytes": os.path.getsize(bundle_path),
            "trt_version": _get_trt_version(),
            "gpu_name": _get_gpu_name(),
        },
        "stage_parity": stage_data.get("stages", {}),
        "stage_parity_pass": parity_result["passed"],
        "inference": {
            "prompt": prompt,
            "cpp_generate_audio_ok": cpp_ok,
            "wav_file": str(wav_path) if cpp_ok else None,
            "wav_size_bytes": wav_size,
        },
        "timing": {
            "build_time_s": built_bundle["build_time_s"],
            "stage4_parity_time_s": parity_result["time_s"],
        },
        "status": "PASS" if parity_result["passed"] else "FAIL",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    results_path = _save_results(bundle_path, results)

    # Assertions
    assert parity_result["passed"], (
        f"Bark stage parity FAILED for {entry['name']}:\n"
        f"{parity_result['stderr'][-2000:]}")

    # WAV generation is informational (not gating)
    wav_status = f", wav={wav_size}B" if cpp_ok else ", wav=FAILED"
    stages = stage_data.get("stages", {})
    sem_info = stages.get("semantic", {})
    coarse_info = stages.get("coarse", {})
    codec_info = stages.get("codec", {})
    print(f"\n[bark_stage_parity] {entry['name']}: PASS "
          f"(sem={sem_info.get('matched', '?')}/{sem_info.get('hf_tokens', '?')}, "
          f"coarse={coarse_info.get('matched', '?')}/{coarse_info.get('hf_tokens', '?')}, "
          f"codec_cos={codec_info.get('cosine', '?')}"
          f"{wav_status})")
    print(f"  Results saved: {results_path}")
    if cpp_ok:
        print(f"  WAV saved: {wav_path}")


@pytest.mark.e2e
def test_audio_pipeline(built_bundle, trtmc_binary,
                        ld_library_path):
    """Full audio pipeline: build -> C++ generate-audio -> verify output."""
    entry = built_bundle["entry"]
    bundle_path = built_bundle["path"]

    if entry.get("runtime_strategy") != "text_to_audio":
        pytest.skip(f"{entry['name']} is not an audio model")

    prompt = entry.get("prompt", "Hello, this is a test.")
    output_path = Path("/tmp") / f"{entry['name']}_audio_output.wav"

    # Run audio generation via C++ binary
    cmd = [
        str(trtmc_binary), "generate-audio", str(bundle_path),
        "--prompt", prompt,
        "--output", str(output_path),
    ]

    env = {"LD_LIBRARY_PATH": ld_library_path}
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)

    assert result.returncode == 0, (
        f"Audio generation failed (rc={result.returncode}):\n{result.stderr}")
    assert output_path.is_file(), "Audio output not written"

    # Verify WAV file has content
    file_size = os.path.getsize(output_path)
    assert file_size > 44, "WAV file is too small (header only?)"

    print(f"\n[audio_pipeline] {entry['name']}: PASS")
    print(f"  Output: {output_path} ({file_size} bytes)")


@pytest.mark.e2e
def test_speech_to_speech_pipeline(model_entry, trtmc_binary,
                                   ld_library_path):
    """Speech pipeline parity: run `trtmc speak` and compare tokens to reference."""
    entry = model_entry
    bundle_path = entry["bundle_path"]

    if entry.get("runtime_strategy") != "speech_to_speech":
        pytest.skip(f"{entry['name']} is not a speech-to-speech model")

    e2e_dir = Path(__file__).resolve().parent
    input_audio = Path(entry.get("test_input_audio", "data/Recording.wav"))
    if not input_audio.is_absolute():
        input_audio = e2e_dir / input_audio
    if not input_audio.is_file():
        pytest.skip(f"Input audio not found: {input_audio}")

    ref_tokens_path = entry.get("speech_reference_tokens")
    if not ref_tokens_path:
        pytest.skip("No speech_reference_tokens configured")
    ref_tokens_file = Path(ref_tokens_path)
    if not ref_tokens_file.is_absolute():
        ref_tokens_file = e2e_dir / ref_tokens_file
    if not ref_tokens_file.is_file():
        pytest.skip(f"Speech reference tokens not found: {ref_tokens_file}")

    max_frames = int(entry.get("speech_test_max_frames", 50))
    min_token_match = float(entry.get("speech_min_token_match", 0.80))
    min_frame_exact = float(entry.get("speech_min_frame_exact", 0.70))
    min_rms = float(entry.get("speech_min_rms", 1e-3))

    temp_dir = Path("/tmp/claude") / f"{entry['name']}_speech"
    temp_dir.mkdir(parents=True, exist_ok=True)
    greedy_bundle = temp_dir / f"{Path(bundle_path).stem}.greedy.bundle"
    output_wav = temp_dir / f"{entry['name']}_speech_output.wav"

    _make_speech_greedy_bundle(bundle_path, greedy_bundle)

    run = _run_cpp_speech_generation(
        trtmc_binary, greedy_bundle, input_audio, output_wav,
        max_frames, ld_library_path)
    assert run["returncode"] == 0, (
        f"Speech generation failed (rc={run['returncode']}):\n{run['stderr']}")
    assert output_wav.is_file(), "Speech output WAV not written"

    out_size = os.path.getsize(output_wav)
    assert out_size > 44, "Speech output WAV is too small (header only?)"

    audio = _read_wav_f32(output_wav)
    assert audio.size > 0, "Speech output audio is empty"
    rms = float(np.sqrt(np.mean(audio * audio)))
    assert rms >= min_rms, f"Speech output RMS too low ({rms:.6f} < {min_rms})"

    got_tokens = _parse_speech_output_tokens(run["stderr"])
    assert got_tokens.size > 0, (
        "Failed to parse output tokens from stderr; missing `[speech] Output frame ...` logs")

    ref_tokens = np.load(str(ref_tokens_file)).astype(np.int32)
    token_match, frame_exact, compared_frames = _speech_token_match_metrics(
        ref_tokens, got_tokens)
    assert compared_frames >= 10, (
        f"Insufficient frames for parity check: {compared_frames}")
    assert token_match >= min_token_match, (
        f"Speech token match {token_match:.4f} < {min_token_match:.4f} "
        f"(frames={compared_frames})")
    assert frame_exact >= min_frame_exact, (
        f"Speech frame-exact {frame_exact:.4f} < {min_frame_exact:.4f} "
        f"(frames={compared_frames})")

    results = {
        "model": {
            "name": entry["name"],
            "hf_id": entry["hf_id"],
            "family": entry.get("family", "unknown"),
            "runtime_strategy": entry.get("runtime_strategy", "personaplex_speech_to_speech"),
        },
        "build": {
            "engine_size_bytes": os.path.getsize(bundle_path),
            "trt_version": _get_trt_version(),
            "gpu_name": _get_gpu_name(),
        },
        "inference": {
            "input_audio": str(input_audio),
            "output_audio": str(output_wav),
            "output_wav_size_bytes": out_size,
            "output_rms": rms,
            "max_frames": max_frames,
        },
        "comparison": {
            "reference_tokens": str(ref_tokens_file),
            "compared_frames": compared_frames,
            "token_match": token_match,
            "frame_exact_match": frame_exact,
            "min_token_match": min_token_match,
            "min_frame_exact": min_frame_exact,
        },
        "status": "PASS",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    results_path = _save_results(bundle_path, results)

    print(f"\n[speech_pipeline] {entry['name']}: PASS "
          f"(token_match={token_match:.4f}, frame_exact={frame_exact:.4f}, "
          f"rms={rms:.4f})")
    print(f"  Output: {output_wav} ({out_size} bytes)")
    print(f"  Results saved: {results_path}")
