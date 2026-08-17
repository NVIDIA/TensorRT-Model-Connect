# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark Torch and split TensorRT Fast Foundation Stereo backends."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

try:
    from .trt_runner import (
        SplitTensorRTRunner,
        load_named_disparity_reference,
        load_native_plugin_libraries,
    )
except ImportError:  # Direct execution: python tests/.../benchmark.py
    from trt_runner import (
        SplitTensorRTRunner,
        load_named_disparity_reference,
        load_native_plugin_libraries,
    )


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def list_images(directory: Path) -> dict[str, Path]:
    suffixes = {".png", ".jpg", ".jpeg", ".bmp"}
    return {
        path.name: path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in suffixes
    }


def select_pairs(
    left_dir: Path, right_dir: Path, num_pairs: int, start_index: int
) -> list[tuple[str, Path, Path]]:
    left = list_images(left_dir)
    right = list_images(right_dir)
    names = sorted(set(left) & set(right))
    selected = names[start_index : start_index + num_pairs]
    if len(selected) != num_pairs:
        raise RuntimeError(
            f"need {num_pairs} paired images from index {start_index}, "
            f"found {len(selected)}; left={len(left)}, right={len(right)}, "
            f"paired={len(names)}"
        )
    return [(name, left[name], right[name]) for name in selected]


def load_and_prepare(left_path, right_path, scale, device, input_padder):
    left = imageio.imread(left_path)
    right = imageio.imread(right_path)
    if left.ndim == 2:
        left = np.repeat(left[..., None], 3, axis=2)
    if right.ndim == 2:
        right = np.repeat(right[..., None], 3, axis=2)
    left = np.ascontiguousarray(left[..., :3])
    right = np.ascontiguousarray(right[..., :3])
    if left.shape[:2] != right.shape[:2]:
        raise ValueError(
            f"image shape mismatch: {left_path}={left.shape}, {right_path}={right.shape}"
        )

    left = cv2.resize(left, fx=scale, fy=scale, dsize=None)
    right = cv2.resize(right, dsize=(left.shape[1], left.shape[0]))
    left_tensor = torch.as_tensor(left).to(device).float()[None].permute(0, 3, 1, 2)
    right_tensor = torch.as_tensor(right).to(device).float()[None].permute(0, 3, 1, 2)
    padder = input_padder(left_tensor.shape, divis_by=32, force_square=False)
    left_tensor, right_tensor = padder.pad(left_tensor, right_tensor)
    return left_tensor, right_tensor, padder, left.shape[:2]


def summarize(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array))
    standard_deviation = float(np.std(array))
    return {
        "count": len(values),
        "mean_ms": mean,
        "median_ms": float(np.median(array)),
        "p90_ms": percentile(values, 90),
        "p95_ms": percentile(values, 95),
        "p99_ms": percentile(values, 99),
        "min_ms": float(np.min(array)),
        "max_ms": float(np.max(array)),
        "stddev_ms": standard_deviation,
        "coefficient_of_variation": standard_deviation / mean if mean else 0.0,
    }


def artifact(path: Path) -> dict[str, str | int]:
    resolved = path.resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return {
        "path": str(resolved),
        "sha256": digest.hexdigest(),
        "bytes": resolved.stat().st_size,
    }


def cosine_similarity(actual: np.ndarray, expected: np.ndarray) -> float:
    actual64 = actual.astype(np.float64, copy=False).reshape(-1)
    expected64 = expected.astype(np.float64, copy=False).reshape(-1)
    denominator = np.linalg.norm(actual64) * np.linalg.norm(expected64)
    if denominator == 0:
        return 1.0 if np.array_equal(actual64, expected64) else 0.0
    return float(np.dot(actual64, expected64) / denominator)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_root", type=Path)
    parser.add_argument("--backend", choices=("torch", "trt", "trt-single"), required=True)
    parser.add_argument("--feature-engine", type=Path)
    parser.add_argument("--post-engine", type=Path)
    parser.add_argument("--engine", type=Path)
    parser.add_argument(
        "--plugin-library",
        action="append",
        default=[],
        type=Path,
        help="family plugin DSO to load before TensorRT engine deserialization",
    )
    parser.add_argument("--cuda-graphs", action="store_true")
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--max-mean-abs-error", type=float, default=0.5)
    parser.add_argument("--max-bad-2px-fraction", type=float, default=0.02)
    parser.add_argument("--num-pairs", type=int, default=5)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--valid-iters", type=int, default=8)
    parser.add_argument("--max-disp", type=int, default=192)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if args.num_pairs <= 0 or args.warmup_iters < 0 or args.iters <= 0:
        raise ValueError("num-pairs and iters must be positive; warmup-iters cannot be negative")
    if not 0.0 <= args.min_cosine <= 1.0:
        raise ValueError("minimum cosine must be in [0, 1]")
    if not np.isfinite(args.max_mean_abs_error) or args.max_mean_abs_error < 0:
        raise ValueError("maximum mean absolute error must be finite and nonnegative")
    if not 0.0 <= args.max_bad_2px_fraction <= 1.0:
        raise ValueError("maximum bad-2px fraction must be in [0, 1]")
    if args.backend == "trt" and (not args.feature_engine or not args.post_engine):
        parser.error("--feature-engine and --post-engine are required for --backend trt")
    if args.backend == "trt-single" and not args.engine:
        parser.error("--engine is required for --backend trt-single")

    model_root = args.model_root.resolve()
    checkpoint = model_root / "weights/23-36-37/model_best_bp2_serialize.pth"
    os.chdir(model_root)
    sys.path.insert(0, str(model_root))
    from core.utils.utils import InputPadder
    from Utils import AMP_DTYPE

    input_root = (args.input_root or model_root / "rectified_stereo_rgb").resolve()
    pairs = select_pairs(
        input_root / "cam_1", input_root / "cam_0", args.num_pairs, args.start_index
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model_init_start = time.perf_counter()
    loaded_plugin_libraries: list[str] = []
    if args.backend == "torch":
        checkpoint_model = torch.load(checkpoint, map_location="cpu", weights_only=False)
        checkpoint_model.args.valid_iters = args.valid_iters
        checkpoint_model.args.max_disp = args.max_disp
        model = checkpoint_model.cuda().eval()

        def run_forward(left_tensor, right_tensor):
            with torch.amp.autocast("cuda", enabled=True, dtype=AMP_DTYPE):
                return model.forward(
                    left_tensor,
                    right_tensor,
                    iters=args.valid_iters,
                    test_mode=True,
                    optimize_build_volume="pytorch1",
                )

    elif args.backend == "trt":
        loaded_plugin_libraries = load_native_plugin_libraries(args.plugin_library)
        model = SplitTensorRTRunner(
            args.feature_engine.resolve(),
            args.post_engine.resolve(),
        )

        def run_forward(left_tensor, right_tensor):
            return model(left_tensor, right_tensor)

    else:
        import tensorrt as trt

        loaded_plugin_libraries = load_native_plugin_libraries(args.plugin_library)

        class SingleEngineRunner:
            def __init__(self, engine_path: Path):
                logger = trt.Logger(trt.Logger.WARNING)
                runtime = trt.Runtime(logger)
                self.engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
                if self.engine is None:
                    raise RuntimeError(f"failed to deserialize {engine_path}")
                self.context = self.engine.create_execution_context()

            @staticmethod
            def torch_dtype(dtype):
                mapping = {
                    trt.DataType.FLOAT: torch.float32,
                    trt.DataType.HALF: torch.float16,
                    trt.DataType.BF16: torch.bfloat16,
                    trt.DataType.INT32: torch.int32,
                    trt.DataType.INT8: torch.int8,
                    trt.DataType.BOOL: torch.bool,
                }
                return mapping[dtype]

            def __call__(self, left_tensor, right_tensor):
                inputs = {
                    "left_image": left_tensor,
                    "right_image": right_tensor,
                }
                for name, value in tuple(inputs.items()):
                    expected = self.torch_dtype(self.engine.get_tensor_dtype(name))
                    inputs[name] = value.to(expected).contiguous()
                    self.context.set_input_shape(name, tuple(inputs[name].shape))

                outputs = {}
                for index in range(self.engine.num_io_tensors):
                    name = self.engine.get_tensor_name(index)
                    if self.engine.get_tensor_mode(name) != trt.TensorIOMode.OUTPUT:
                        continue
                    outputs[name] = torch.empty(
                        tuple(self.context.get_tensor_shape(name)),
                        device="cuda",
                        dtype=self.torch_dtype(self.engine.get_tensor_dtype(name)),
                    )
                for name, value in {**inputs, **outputs}.items():
                    self.context.set_tensor_address(name, int(value.data_ptr()))
                if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
                    raise RuntimeError("TensorRT single-engine enqueue failed")
                return outputs["disparity"]

        model = SingleEngineRunner(args.engine.resolve())
        mean = torch.tensor([123.675, 116.28, 103.53], device="cuda").view(1, 3, 1, 1)
        std = torch.tensor([58.395, 57.12, 57.375], device="cuda").view(1, 3, 1, 1)

        def run_forward(left_tensor, right_tensor):
            return model((left_tensor - mean) / std, (right_tensor - mean) / std)

    if args.backend.startswith("trt") and args.cuda_graphs:
        eager_forward = run_forward
        graph = None
        static_left = torch.empty((1, 3, 704, 704), device="cuda")
        static_right = torch.empty_like(static_left)
        static_output = None

        def run_forward(left_tensor, right_tensor):
            nonlocal graph, static_output
            static_left.copy_(left_tensor)
            static_right.copy_(right_tensor)
            if graph is None:
                # TensorRT requires one successful enqueue after shapes and
                # addresses are established before CUDA graph capture. The
                # extra launches are warmup-only and outside timed samples.
                for _ in range(3):
                    static_output = eager_forward(static_left, static_right)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    static_output = eager_forward(static_left, static_right)
            graph.replay()
            return static_output

    torch.cuda.synchronize()
    model_init_ms = (time.perf_counter() - model_init_start) * 1000.0
    inference_stream = torch.cuda.Stream()

    def run_one_iteration(save_output=False):
        preprocess_start = time.perf_counter()
        prepared = [
            load_and_prepare(left, right, args.scale, "cuda", InputPadder)
            for _, left, right in pairs
        ]
        torch.cuda.synchronize()
        preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0

        infer_start = time.perf_counter()
        disparities = []
        with torch.inference_mode():
            for _, (left_tensor, right_tensor, padder, image_shape) in zip(pairs, prepared):
                inference_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(inference_stream):
                    disparity = run_forward(left_tensor, right_tensor)
                torch.cuda.current_stream().wait_stream(inference_stream)
                disparity = padder.unpad(disparity.float()).cpu().numpy().reshape(image_shape)
                disparities.append(np.clip(disparity, 0, None).astype(np.float32))
        torch.cuda.synchronize()
        infer_ms = (time.perf_counter() - infer_start) * 1000.0
        output = np.stack(disparities) if save_output else None
        return preprocess_ms, infer_ms, preprocess_ms + infer_ms, output

    for index in range(args.warmup_iters):
        run_one_iteration()
        print(f"warmup {index + 1}/{args.warmup_iters}")

    preprocess_times = []
    infer_times = []
    total_times = []
    first_output = None
    for index in range(args.iters):
        preprocess_ms, infer_ms, total_ms, output = run_one_iteration(index == 0)
        preprocess_times.append(preprocess_ms)
        infer_times.append(infer_ms)
        total_times.append(total_ms)
        if output is not None:
            first_output = output

    assert first_output is not None
    output_path = args.out_dir / f"{args.backend}_disparity.npz"
    np.savez_compressed(
        output_path,
        names=np.asarray([name for name, _, _ in pairs]),
        disparity=first_output,
    )

    accuracy = None
    accuracy_passed = None
    if args.reference:
        reference = load_named_disparity_reference(
            args.reference,
            [name for name, _, _ in pairs],
            first_output.shape,
        )
        pair_cosines = [
            cosine_similarity(actual, expected) for actual, expected in zip(first_output, reference)
        ]
        absolute_error = np.abs(first_output - reference)
        accuracy = {
            "global_cosine": cosine_similarity(first_output, reference),
            "pair_cosines": pair_cosines,
            "max_abs_error": float(np.max(absolute_error)),
            "mean_abs_error": float(np.mean(absolute_error)),
            "bad_2px_fraction": float(np.mean(absolute_error > 2.0)),
        }
        # Cosine catches structural drift while endpoint error prevents a
        # globally rescaled disparity map from passing a scale-invariant gate.
        accuracy_passed = (
            accuracy["global_cosine"] >= args.min_cosine
            and accuracy["mean_abs_error"] <= args.max_mean_abs_error
            and accuracy["bad_2px_fraction"] <= args.max_bad_2px_fraction
        )

    result = {
        "backend": args.backend,
        "model_root": str(model_root),
        "input_root": str(input_root),
        "num_pairs": args.num_pairs,
        "start_index": args.start_index,
        "warmup_iters": args.warmup_iters,
        "iters": args.iters,
        "scale": args.scale,
        "valid_iters": args.valid_iters,
        "max_disp": args.max_disp,
        "cuda_graphs": args.cuda_graphs,
        "plugin_libraries": loaded_plugin_libraries,
        "model_init_ms": model_init_ms,
        "measurement_scope": {
            "name": "serial_paired_inference",
            "description": (
                f"{args.num_pairs} stereo pairs evaluated serially; inference timing includes "
                "output conversion, device-to-host transfer, unpadding, and clipping"
            ),
        },
        "preprocess_5_pairs": summarize(preprocess_times),
        "infer_5_pairs": summarize(infer_times),
        "total_5_pairs": summarize(total_times),
        "samples_ms": {
            "preprocess": preprocess_times,
            "inference": infer_times,
            "total": total_times,
        },
        "preprocess_per_pair_ms": float(np.mean(preprocess_times) / args.num_pairs),
        "infer_per_pair_ms": float(np.mean(infer_times) / args.num_pairs),
        "total_per_pair_ms": float(np.mean(total_times) / args.num_pairs),
        "output_file": str(output_path),
        "accuracy": accuracy,
        "minimum_cosine": args.min_cosine,
        "maximum_mean_abs_error": args.max_mean_abs_error,
        "maximum_bad_2px_fraction": args.max_bad_2px_fraction,
        "accuracy_passed": accuracy_passed,
        "environment": {
            "torch": torch.__version__,
            "tensorrt": (
                __import__("tensorrt").__version__ if args.backend.startswith("trt") else None
            ),
            "cuda_runtime": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(),
            "gpu_capability": list(torch.cuda.get_device_capability()),
        },
    }
    result["artifacts"] = {
        "inputs": [
            {
                "name": name,
                "left": artifact(left_path),
                "right": artifact(right_path),
            }
            for name, left_path, right_path in pairs
        ],
        "output": artifact(output_path),
        "benchmark_tool": artifact(Path(__file__)),
        "runner_tool": artifact(Path(__file__).with_name("trt_runner.py")),
    }
    if args.reference:
        result["artifacts"]["accuracy_reference"] = artifact(args.reference)
    if checkpoint.is_file():
        result["artifacts"]["checkpoint"] = artifact(checkpoint)
    if args.backend == "trt":
        result["artifacts"]["feature_engine"] = artifact(args.feature_engine)
        result["artifacts"]["post_engine"] = artifact(args.post_engine)
    elif args.backend == "trt-single":
        result["artifacts"]["engine"] = artifact(args.engine)
    result["artifacts"]["plugin_libraries"] = [
        artifact(Path(path)) for path in loaded_plugin_libraries
    ]
    result_path = args.out_dir / "benchmark_result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if accuracy_passed is False:
        raise RuntimeError(f"accuracy gate failed: {accuracy}; minimum cosine={args.min_cosine}")


if __name__ == "__main__":
    main()
