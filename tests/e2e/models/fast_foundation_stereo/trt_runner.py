# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small TensorRT runners shared by the stereo benchmark and profiler."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    import torch


_PLUGIN_LIBRARY_NAMES = (
    "libtrtmc_fast_foundation_stereo_native_plugin.so",
    "libtrtmc_model_fast_foundation_stereo.so",
    "trtmc_model_fast_foundation_stereo.dll",
    "libtrtmc_model_fast_foundation_stereo.dylib",
)
_PLUGIN_HANDLES: list[ctypes.CDLL] = []


def _deduplicate(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _automatic_plugin_candidates() -> list[str]:
    candidates: list[str] = []
    for name in (
        "TRTMC_FAST_FOUNDATION_STEREO_NATIVE_PLUGIN_LIBRARY",
        "TRTMC_FAST_FOUNDATION_STEREO_PLUGIN_LIBRARY",
        "TRTMC_PLUGIN_LIBRARY",
    ):
        value = os.environ.get(name, "")
        candidates.extend(part for part in value.split(os.pathsep) if part)

    repository_root = Path(__file__).resolve().parents[4]
    directories = [
        repository_root / "build",
        repository_root / "python/tensorrt_model_connect/bin",
    ]
    configured_plugin_dir = os.environ.get("TRTMC_MODEL_PLUGIN_DIR")
    if configured_plugin_dir:
        directories.insert(0, Path(configured_plugin_dir))
    for directory in directories:
        candidates.extend(str(directory / name) for name in _PLUGIN_LIBRARY_NAMES)
    candidates.extend(_PLUGIN_LIBRARY_NAMES)
    return _deduplicate(candidates)


def load_native_plugin_libraries(explicit: Iterable[Path] = ()) -> list[str]:
    """Load the family DSO before deserializing an engine containing its plugin."""

    explicit_paths = [str(Path(path).resolve()) for path in explicit]
    if not explicit_paths:
        try:
            from tensorrt_model_connect.families.fast_foundation_stereo.native_plugin_builder import (
                load_native_plugin,
            )
        except ImportError:
            pass
        else:
            return [str(load_native_plugin())]

    candidates = explicit_paths if explicit_paths else _automatic_plugin_candidates()
    loaded: list[str] = []
    explicit_set = set(explicit_paths)
    for candidate in candidates:
        path = Path(candidate)
        if candidate not in explicit_set and path.parent != Path(".") and not path.is_file():
            continue
        try:
            handle = ctypes.CDLL(candidate, mode=getattr(ctypes, "RTLD_GLOBAL", 0))
        except OSError as exc:
            if candidate in explicit_set:
                raise RuntimeError(
                    f"failed to load requested TensorRT plugin library {candidate}: {exc}"
                ) from exc
            continue
        _PLUGIN_HANDLES.append(handle)
        loaded.append(candidate)
    return loaded


class SplitTensorRTRunner:
    """Execute the feature and post plans without any framework-side graph work."""

    def __init__(self, feature_engine_path: Path, post_engine_path: Path) -> None:
        import tensorrt as trt
        import torch

        self._trt = trt
        self._torch = torch
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        self.feature_engine = self._deserialize(runtime, feature_engine_path)
        self.post_engine = self._deserialize(runtime, post_engine_path)
        self.feature_context = self.feature_engine.create_execution_context()
        self.post_context = self.post_engine.create_execution_context()
        if self.feature_context is None or self.post_context is None:
            raise RuntimeError("failed to create TensorRT stereo execution contexts")

    @staticmethod
    def _deserialize(runtime: Any, path: Path) -> Any:
        resolved = path.resolve()
        engine = runtime.deserialize_cuda_engine(resolved.read_bytes())
        if engine is None:
            raise RuntimeError(f"failed to deserialize {resolved}")
        return engine

    def _tensor_names(self, engine: Any, mode: Any) -> list[str]:
        return [
            engine.get_tensor_name(index)
            for index in range(engine.num_io_tensors)
            if engine.get_tensor_mode(engine.get_tensor_name(index)) == mode
        ]

    def _torch_dtype(self, dtype: Any) -> torch.dtype:
        mapping = {
            self._trt.DataType.FLOAT: self._torch.float32,
            self._trt.DataType.HALF: self._torch.float16,
            self._trt.DataType.BF16: self._torch.bfloat16,
            self._trt.DataType.INT32: self._torch.int32,
            self._trt.DataType.INT8: self._torch.int8,
            self._trt.DataType.BOOL: self._torch.bool,
        }
        try:
            return mapping[dtype]
        except KeyError as exc:
            raise RuntimeError(f"unsupported TensorRT dtype: {dtype}") from exc

    def _run_engine(self, engine: Any, context: Any, inputs: dict[str, torch.Tensor]):
        input_names = self._tensor_names(engine, self._trt.TensorIOMode.INPUT)
        if set(inputs) != set(input_names):
            raise RuntimeError(
                f"TensorRT input mismatch: expected {input_names}, got {sorted(inputs)}"
            )
        prepared: dict[str, torch.Tensor] = {}
        for name in input_names:
            expected_dtype = self._torch_dtype(engine.get_tensor_dtype(name))
            value = inputs[name].to(expected_dtype).contiguous()
            if not context.set_input_shape(name, tuple(value.shape)):
                raise RuntimeError(f"failed to set TensorRT input shape for {name}")
            prepared[name] = value

        outputs: dict[str, torch.Tensor] = {}
        output_names = self._tensor_names(engine, self._trt.TensorIOMode.OUTPUT)
        for name in output_names:
            outputs[name] = self._torch.empty(
                tuple(context.get_tensor_shape(name)),
                device="cuda",
                dtype=self._torch_dtype(engine.get_tensor_dtype(name)),
            )
        for name, value in {**prepared, **outputs}.items():
            if not context.set_tensor_address(name, int(value.data_ptr())):
                raise RuntimeError(f"failed to bind TensorRT tensor {name}")
        stream = self._torch.cuda.current_stream().cuda_stream
        if not context.execute_async_v3(stream):
            raise RuntimeError("TensorRT stereo enqueue failed")
        return outputs

    def run_feature(self, left: torch.Tensor, right: torch.Tensor):
        input_names = self._tensor_names(self.feature_engine, self._trt.TensorIOMode.INPUT)
        if input_names != ["left", "right"]:
            raise RuntimeError(f"unexpected feature inputs: {input_names}")
        return self._run_engine(
            self.feature_engine,
            self.feature_context,
            {"left": left, "right": right},
        )

    def run_post(self, features: dict[str, torch.Tensor]):
        input_names = self._tensor_names(self.post_engine, self._trt.TensorIOMode.INPUT)
        missing = [name for name in input_names if name not in features]
        if missing:
            raise RuntimeError(f"post engine requested non-feature inputs: {missing}")
        inputs = {name: features[name] for name in input_names}
        return self._run_engine(self.post_engine, self.post_context, inputs)

    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        outputs = self.run_post(self.run_feature(left, right))
        try:
            return outputs["disp"]
        except KeyError as exc:
            raise RuntimeError(f"unexpected post outputs: {sorted(outputs)}") from exc
