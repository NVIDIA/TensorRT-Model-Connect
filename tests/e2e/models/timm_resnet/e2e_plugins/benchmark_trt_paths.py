#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare raw TensorRT API and ONNX->trtexec engines for timm ResNet.

The raw API path uses TensorRT-Model-Connect's timm_resnet family plugin.  The
ONNX path exports the same timm PyTorch model, builds an engine with trtexec,
and benchmarks both plans with identical input tensors via TensorRT Python.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_MODEL_ID = "timm/vit_base_patch16_224.augreg_in21k_ft_in1k"


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "CMakeLists.txt").is_file() and (parent / "python").is_dir():
            return parent
    raise RuntimeError("Cannot locate TensorRT-Model-Connect repository root")


def _find_trtexec(explicit: str | None) -> str:
    candidates = []
    if explicit:
        candidates.append(explicit)
    which = shutil.which("trtexec")
    if which:
        candidates.append(which)
    candidates.extend([
        "/usr/src/tensorrt/bin/trtexec",
        "/opt/tensorrt/bin/trtexec",
        "/usr/local/tensorrt/bin/trtexec",
    ])
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    raise FileNotFoundError(
        "trtexec not found; pass --trtexec or add TensorRT's bin directory to PATH"
    )


def _create_timm_model(model_id: str):
    import timm

    try:
        return timm.create_model(f"hf-hub:{model_id}", pretrained=True)
    except Exception:
        return timm.create_model(f"hf_hub:{model_id}", pretrained=True)


def _build_api_engine(model_id: str, plan_path: Path, *, verbose: bool) -> None:
    from tensorrt_model_connect.config import ModelConfig
    from tensorrt_model_connect.engine_builder import _resolve_model
    from tensorrt_model_connect.families import find_plugin

    model_dir = Path(_resolve_model(model_id))
    config = ModelConfig.from_dir(model_dir)
    plugin = find_plugin(config.model_type)
    if plugin is None or plugin.name != "timm_resnet":
        raise RuntimeError(
            f"Expected timm_resnet plugin for {config.model_type!r}, got {plugin}"
        )

    weights = plugin.load_weights(str(model_dir), config, precision="fp32")
    plan = plugin.build_engine(
        config,
        weights,
        max_cache_length=1,
        precision="fp32",
        verbose=verbose,
    )
    plan_path.write_bytes(plan)


def _export_onnx(model_id: str, onnx_path: Path) -> None:
    import torch

    model = _create_timm_model(model_id)
    model.eval()
    dummy = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["pixel_values"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )


def _build_trtexec_engine(
    trtexec: str,
    onnx_path: Path,
    plan_path: Path,
    log_path: Path,
) -> None:
    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={plan_path}",
        "--skipInference",
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=1800)
    log_path.write_text(
        "COMMAND: " + " ".join(cmd) + "\n\nSTDOUT:\n" + result.stdout +
        "\n\nSTDERR:\n" + result.stderr,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"trtexec failed with rc={result.returncode}; see {log_path}")


_TRTEXEC_TIMING_RE = re.compile(
    r"GPU Compute Time: min = (?P<min>[0-9.]+) ms, max = (?P<max>[0-9.]+) ms, "
    r"mean = (?P<mean>[0-9.]+) ms, median = (?P<median>[0-9.]+) ms"
)


def _benchmark_plan_with_trtexec(
    trtexec: str,
    plan_path: Path,
    log_path: Path,
    *,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    cmd = [
        trtexec,
        f"--loadEngine={plan_path}",
        f"--warmUp={max(200, warmup)}",
        "--duration=0",
        f"--iterations={iterations}",
        "--noDataTransfers",
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=1800)
    log_path.write_text(
        "COMMAND: " + " ".join(cmd) + "\n\nSTDOUT:\n" + result.stdout +
        "\n\nSTDERR:\n" + result.stderr,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"trtexec benchmark failed with rc={result.returncode}; see {log_path}")
    matches = list(_TRTEXEC_TIMING_RE.finditer(result.stdout + "\n" + result.stderr))
    if not matches:
        raise RuntimeError(f"Could not parse trtexec GPU timing from {log_path}")
    timing = matches[-1].groupdict()
    return {
        "mean_ms": float(timing["mean"]),
        "median_ms": float(timing["median"]),
        "stdev_ms": None,
        "min_ms": float(timing["min"]),
        "max_ms": float(timing["max"]),
        "iterations": iterations,
        "warmup": max(200, warmup),
        "log": str(log_path),
    }


def _input_from_image(image_path: Path) -> np.ndarray:
    from PIL import Image

    target = 224
    crop_pct = 0.9
    resize_short = int(target / crop_pct + 0.5)
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    if height <= width:
        resized_h = resize_short
        resized_w = max(1, int(width * resize_short / height + 0.5))
    else:
        resized_w = resize_short
        resized_h = max(1, int(height * resize_short / width + 0.5))
    image = image.resize((resized_w, resized_h), Image.Resampling.NEAREST)
    left = max(0, (resized_w - target) // 2)
    top = max(0, (resized_h - target) // 2)
    image = image.crop((left, top, left + target, top + target))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return np.transpose(arr, (2, 0, 1))[None, ...].copy()


def _make_input(seed: int, image: str | None) -> np.ndarray:
    if image:
        return _input_from_image(Path(image))
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, size=(1, 3, 224, 224)).astype(np.float32)


def _torch_dtype_for_trt(trt_dtype):
    import tensorrt as trt
    import torch

    mapping = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int32: torch.int32,
        trt.bool: torch.bool,
    }
    if hasattr(trt, "bfloat16"):
        mapping[trt.bfloat16] = torch.bfloat16
    if hasattr(trt, "int64"):
        mapping[trt.int64] = torch.int64
    return mapping[trt_dtype]


def _load_engine(plan_path: Path):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan_path.read_bytes())
    if engine is None:
        raise RuntimeError(f"Failed to deserialize {plan_path}")
    return engine


def _prepare_context(plan_path: Path, input_np: np.ndarray):
    import tensorrt as trt
    import torch

    engine = _load_engine(plan_path)
    context = engine.create_execution_context()
    tensors: dict[str, Any] = {}
    outputs: dict[str, Any] = {}

    for idx in range(engine.num_io_tensors):
        name = engine.get_tensor_name(idx)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            if hasattr(context, "set_input_shape"):
                context.set_input_shape(name, tuple(input_np.shape))
            tensor = torch.as_tensor(input_np, device="cuda").contiguous()
        else:
            shape = tuple(int(dim) for dim in context.get_tensor_shape(name))
            dtype = _torch_dtype_for_trt(engine.get_tensor_dtype(name))
            tensor = torch.empty(shape, dtype=dtype, device="cuda")
            outputs[name] = tensor
        tensors[name] = tensor
        context.set_tensor_address(name, int(tensor.data_ptr()))

    return engine, context, tensors, outputs


def _run_once(context, outputs: dict[str, Any], stream) -> np.ndarray:
    import torch

    with torch.cuda.stream(stream):
        context.execute_async_v3(stream_handle=stream.cuda_stream)
    torch.cuda.synchronize()
    if "logits" in outputs:
        out = outputs["logits"]
    else:
        out = next(iter(outputs.values()))
    return out.detach().float().cpu().numpy().reshape(-1)


def _benchmark_plan(
    plan_path: Path,
    input_np: np.ndarray,
    *,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    import torch

    _engine, context, _tensors, outputs = _prepare_context(plan_path, input_np)
    stream = torch.cuda.Stream()
    for _ in range(warmup):
        with torch.cuda.stream(stream):
            context.execute_async_v3(stream_handle=stream.cuda_stream)
    torch.cuda.synchronize()

    times_ms: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            start.record(stream)
            context.execute_async_v3(stream_handle=stream.cuda_stream)
            end.record(stream)
        torch.cuda.synchronize()
        times_ms.append(float(start.elapsed_time(end)))

    logits = _run_once(context, outputs, stream)
    return {
        "mean_ms": float(statistics.fmean(times_ms)),
        "median_ms": float(statistics.median(times_ms)),
        "stdev_ms": float(statistics.pstdev(times_ms)),
        "min_ms": float(min(times_ms)),
        "max_ms": float(max(times_ms)),
        "iterations": iterations,
        "warmup": warmup,
        "logits": logits,
    }


def _summarize(
    api: dict[str, Any],
    onnx: dict[str, Any],
    *,
    max_ratio: float,
    max_abs_diff: float,
) -> dict[str, Any]:
    api_logits = api.pop("logits")
    onnx_logits = onnx.pop("logits")
    abs_diff = np.abs(api_logits - onnx_logits)
    top_api = int(np.argmax(api_logits))
    top_onnx = int(np.argmax(onnx_logits))
    api_over_onnx = api["mean_ms"] / onnx["mean_ms"]
    symmetric_ratio = (
        max(api["mean_ms"], onnx["mean_ms"]) / min(api["mean_ms"], onnx["mean_ms"])
    )
    return {
        "api_engine": api,
        "onnx_trtexec_engine": onnx,
        "api_over_onnx_ratio": float(api_over_onnx),
        "perf_ratio_max_over_min": float(symmetric_ratio),
        "perf_within_threshold": bool(api_over_onnx <= max_ratio),
        "max_allowed_perf_ratio": max_ratio,
        "output": {
            "max_abs_diff": float(abs_diff.max()),
            "mean_abs_diff": float(abs_diff.mean()),
            "top1_api": top_api,
            "top1_onnx_trtexec": top_onnx,
            "top1_match": top_api == top_onnx,
            "max_allowed_abs_diff": max_abs_diff,
            "within_threshold": bool(abs_diff.max() <= max_abs_diff),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--out-dir",
        default=str(_repo_root() / "artifacts" / "timm_resnet_trt_path_comparison"),
    )
    parser.add_argument("--trtexec", default=None)
    parser.add_argument("--image", default=None)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-ratio", type=float, default=1.05)
    parser.add_argument("--max-abs-diff", type=float, default=0.05)
    parser.add_argument(
        "--perf-runner",
        choices=("trtexec", "python"),
        default="trtexec",
        help=(
            "Use direct trtexec GPU Compute Time for the performance gate "
            "or Python CUDA events. Python is still used for correctness."
        ),
    )
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    api_plan = out_dir / "api.plan"
    onnx_path = out_dir / "model.onnx"
    onnx_plan = out_dir / "onnx_trtexec.plan"
    trtexec_log = out_dir / "trtexec.log"
    result_path = out_dir / "result.json"

    trtexec = _find_trtexec(args.trtexec)

    t0 = time.monotonic()
    if args.rebuild or not api_plan.is_file():
        _build_api_engine(args.model_id, api_plan, verbose=args.verbose)
    if args.rebuild or not onnx_path.is_file():
        _export_onnx(args.model_id, onnx_path)
    if args.rebuild or not onnx_plan.is_file():
        _build_trtexec_engine(trtexec, onnx_path, onnx_plan, trtexec_log)

    input_np = _make_input(args.seed, args.image)
    api_python = _benchmark_plan(
        api_plan, input_np, warmup=args.warmup, iterations=args.iterations)
    onnx_python = _benchmark_plan(
        onnx_plan, input_np, warmup=args.warmup, iterations=args.iterations)
    if args.perf_runner == "trtexec":
        api_perf = _benchmark_plan_with_trtexec(
            trtexec,
            api_plan,
            out_dir / "trtexec_api_bench.log",
            warmup=args.warmup,
            iterations=args.iterations,
        )
        onnx_perf = _benchmark_plan_with_trtexec(
            trtexec,
            onnx_plan,
            out_dir / "trtexec_onnx_bench.log",
            warmup=args.warmup,
            iterations=args.iterations,
        )
        api = {**api_perf, "logits": api_python["logits"]}
        onnx = {**onnx_perf, "logits": onnx_python["logits"]}
    else:
        api = api_python.copy()
        onnx = onnx_python.copy()
    summary = _summarize(
        api, onnx, max_ratio=args.max_ratio, max_abs_diff=args.max_abs_diff)
    summary["perf_runner"] = args.perf_runner
    summary["python_api_engine"] = {
        key: value for key, value in api_python.items() if key != "logits"}
    summary["python_onnx_trtexec_engine"] = {
        key: value for key, value in onnx_python.items() if key != "logits"}
    summary.update({
        "model_id": args.model_id,
        "api_plan": str(api_plan),
        "onnx_path": str(onnx_path),
        "onnx_trtexec_plan": str(onnx_plan),
        "trtexec": trtexec,
        "trtexec_log": str(trtexec_log),
        "input": {"image": args.image, "seed": args.seed},
        "total_elapsed_s": time.monotonic() - t0,
    })
    result_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(
        "api_mean_ms={api:.6f} onnx_trtexec_mean_ms={onnx:.6f} "
        "api_over_onnx={ratio:.6f} top1_match={top1} max_abs_diff={diff:.6g} "
        "result={result}".format(
            api=summary["api_engine"]["mean_ms"],
            onnx=summary["onnx_trtexec_engine"]["mean_ms"],
            ratio=summary["api_over_onnx_ratio"],
            top1=summary["output"]["top1_match"],
            diff=summary["output"]["max_abs_diff"],
            result=result_path,
        )
    )
    if not summary["perf_within_threshold"]:
        return 2
    if not summary["output"]["top1_match"] or not summary["output"]["within_threshold"]:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
