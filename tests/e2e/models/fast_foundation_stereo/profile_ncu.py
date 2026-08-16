# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Emit one real, profiler-delimited stereo inference for Nsight Compute."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import torch

try:
    from .benchmark import load_and_prepare
    from .trt_runner import SplitTensorRTRunner, load_native_plugin_libraries
except ImportError:  # Direct execution: python tests/.../profile_ncu.py
    from benchmark import load_and_prepare
    from trt_runner import SplitTensorRTRunner, load_native_plugin_libraries


_SCRIPT_PATH = Path(__file__).resolve()


@contextmanager
def _nvtx_range(name: str) -> Iterator[None]:
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_root", type=Path)
    parser.add_argument("feature_engine", type=Path)
    parser.add_argument("post_engine", type=Path)
    parser.add_argument(
        "--input-root",
        type=Path,
        help="benchmark input root containing cam_1 (left) and cam_0 (right)",
    )
    parser.add_argument("--left-image", type=Path)
    parser.add_argument("--right-image", type=Path)
    parser.add_argument("--pair-name")
    parser.add_argument(
        "--plugin-library",
        action="append",
        default=[],
        type=Path,
        help="family plugin DSO to load before TensorRT engine deserialization",
    )
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--cuda-graphs", action="store_true")
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="output JSON binding this profile to its inputs and engines",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this profile")
    if args.warmup < 0:
        raise ValueError("warmup cannot be negative")
    if args.scale <= 0:
        raise ValueError("scale must be positive")

    if args.input_root:
        if args.left_image or args.right_image:
            parser.error("--input-root cannot be combined with explicit image paths")
        if not args.pair_name:
            parser.error("--pair-name is required with --input-root")
        input_root = args.input_root.resolve()
        left_image = input_root / "cam_1" / args.pair_name
        right_image = input_root / "cam_0" / args.pair_name
    else:
        if not args.left_image or not args.right_image:
            parser.error(
                "provide --input-root with --pair-name or both --left-image and --right-image"
            )
        input_root = None
        left_image = args.left_image.resolve()
        right_image = args.right_image.resolve()

    root = args.model_root.resolve()
    feature_engine = args.feature_engine.resolve()
    post_engine = args.post_engine.resolve()
    left_image = left_image.resolve()
    right_image = right_image.resolve()
    plugin_libraries = [path.resolve() for path in args.plugin_library]
    manifest_path = args.manifest.resolve()
    for label, path in (
        ("model root", root),
        ("feature engine", feature_engine),
        ("post engine", post_engine),
        ("left image", left_image),
        ("right image", right_image),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    os.chdir(root)
    sys.path.insert(0, str(root))
    import tensorrt as trt
    from core.utils.utils import InputPadder

    loaded_plugin_libraries = load_native_plugin_libraries(plugin_libraries)
    plugin_artifacts = [
        _artifact(Path(path)) for path in loaded_plugin_libraries if Path(path).is_file()
    ]
    if not plugin_artifacts:
        raise RuntimeError(
            "profile manifest requires a fingerprintable native plugin; "
            "pass --plugin-library with an existing DSO"
        )
    model = SplitTensorRTRunner(feature_engine, post_engine)
    left, right, _padder, image_shape = load_and_prepare(
        left_image,
        right_image,
        args.scale,
        "cuda",
        InputPadder,
    )
    if left.shape != right.shape:
        raise RuntimeError(f"prepared stereo shapes differ: {left.shape} != {right.shape}")

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    feature_graph = None
    post_graph = None
    graph_features = None
    graph_outputs = None

    with torch.inference_mode(), torch.cuda.stream(stream):
        # TensorRT requires successful enqueues after shapes and addresses are
        # established and before graph capture. Keep all of them outside NVTX.
        warmup_count = max(args.warmup, 3 if args.cuda_graphs else 0)
        for _ in range(warmup_count):
            warmup_features = model.run_feature(left, right)
            model.run_post(warmup_features)
        stream.synchronize()

        if args.cuda_graphs:
            feature_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(feature_graph, stream=stream):
                graph_features = model.run_feature(left, right)
            feature_graph.replay()
            stream.synchronize()

            post_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(post_graph, stream=stream):
                graph_outputs = model.run_post(graph_features)
            stream.synchronize()

        with _nvtx_range("ffs_full"):
            with _nvtx_range("ffs_feature"):
                if feature_graph is None:
                    profiled_features = model.run_feature(left, right)
                else:
                    feature_graph.replay()
                    profiled_features = graph_features
            with _nvtx_range("ffs_post"):
                if post_graph is None:
                    profiled_outputs = model.run_post(profiled_features)
                else:
                    post_graph.replay()
                    profiled_outputs = graph_outputs
            stream.synchronize()

    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    if not profiled_outputs or "disp" not in profiled_outputs:
        raise RuntimeError(f"unexpected post outputs: {sorted(profiled_outputs or {})}")

    pair_name = args.pair_name or left_image.name
    manifest = {
        "schema_version": 2,
        "profile_scope": "native_feature_and_post",
        "nvtx_ranges": ["ffs_full", "ffs_feature", "ffs_post"],
        "required_profiler_contract": {
            "tool": "ncu",
            "config_file": "off",
            "metric_set": "roofline",
            "nvtx_include": "ffs_full/",
            "replay_mode": "kernel",
            "launch_skip": 0,
            "launch_count": "all",
        },
        "model_root": str(root),
        "input_root": str(input_root) if input_root else None,
        "pair": {
            "name": pair_name,
            "left_image": _artifact(left_image),
            "right_image": _artifact(right_image),
            "unpadded_height": int(image_shape[0]),
            "unpadded_width": int(image_shape[1]),
            "prepared_shape": list(left.shape),
            "scale": args.scale,
        },
        "engines": {
            "feature": _artifact(feature_engine),
            "post": _artifact(post_engine),
        },
        "plugin_libraries": loaded_plugin_libraries,
        "plugin_artifacts": plugin_artifacts,
        "source_artifacts": {
            "python_executable": _artifact(Path(sys.executable)),
            "profile_ncu": _artifact(_SCRIPT_PATH),
            "benchmark": _artifact(_SCRIPT_PATH.with_name("benchmark.py")),
            "trt_runner": _artifact(_SCRIPT_PATH.with_name("trt_runner.py")),
        },
        "cuda_graphs": args.cuda_graphs,
        "requested_warmup_enqueues": args.warmup,
        "actual_warmup_enqueues": warmup_count,
        "environment": {
            "torch": torch.__version__,
            "tensorrt": trt.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(),
            "gpu_capability": list(torch.cuda.get_device_capability()),
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
