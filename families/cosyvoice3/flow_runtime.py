# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned component runtime; not the public C++ TTS pipeline."""

from __future__ import annotations

import json
from pathlib import Path
import threading

from . import trt_compat
from .config import FlowConfig, ShapeProfile

INPUT_NAMES = ("x", "mask", "mu", "t", "spks", "cond")


def validate_input_shapes(shapes, cfg: FlowConfig, profile: ShapeProfile):
    if set(shapes) != set(INPUT_NAMES):
        raise ValueError(f"Expected exactly these inputs: {INPUT_NAMES}")
    if len(shapes["x"]) != 3:
        raise ValueError("x must have shape [2, mel_channels, frames]")
    frames = shapes["x"][-1]
    profile.validate_frames(frames)
    expected = {
        "x": (2, cfg.mel_dim, frames), "mask": (2, 1, frames),
        "mu": (2, cfg.mel_dim, frames), "t": (2,),
        "spks": (2, cfg.spk_dim), "cond": (2, cfg.mel_dim, frames),
    }
    for key, shape in expected.items():
        if tuple(shapes[key]) != shape:
            raise ValueError(f"{key}: expected {shape}, got {shapes[key]}")
    return frames


class FlowEngine:
    """Synchronous, single-context CUDA runner with explicit input contracts.

    Synchronizing before releasing the lock guarantees that neither buffers nor
    the execution context are reused while a previous request is still running.
    No global device changes, implicit CPU/GPU transfers or precision casts.
    """

    def __init__(self, directory: str | Path, *, device=0):
        import torch

        self.torch = torch
        path = Path(directory)
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if (manifest.get("schema_version") != 1 or manifest.get("component") != "cosyvoice3_flow_estimator"
                or manifest.get("precision") != "fp32" or manifest.get("streaming") is not False):
            raise ValueError("Unsupported CosyVoice3 component manifest")
        self.cfg = FlowConfig(**manifest["architecture"])
        self.profile = ShapeProfile(**manifest["profile"])
        plan = (path / "flow.plan").read_bytes()
        self.device = torch.device("cuda", device)
        if not torch.cuda.is_available():
            raise RuntimeError("CosyVoice3 FlowEngine requires a CUDA GPU")
        trt = trt_compat.get_trt()
        self.logger = trt.Logger(trt.Logger.WARNING)
        with torch.cuda.device(self.device):
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(plan)
            if self.engine is None:
                raise RuntimeError("Could not load flow.plan; rebuild for this GPU and TensorRT version")
            names = {self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)}
            if names != set(INPUT_NAMES) | {"positions", "velocity"}:
                raise ValueError(f"Unexpected Flow engine I/O: {sorted(names)}")
            for name in names:
                dtype = trt.int32 if name == "positions" else trt.float32
                mode = trt.TensorIOMode.OUTPUT if name == "velocity" else trt.TensorIOMode.INPUT
                if self.engine.get_tensor_dtype(name) != dtype or self.engine.get_tensor_mode(name) != mode:
                    raise ValueError(f"Unexpected Flow engine I/O type or mode for {name}")
            self.context = self.engine.create_execution_context()
            if self.context is None:
                raise RuntimeError("Could not allocate Flow execution context")
        self.lock = threading.Lock()

    def __call__(self, x, mask, mu, t, spks, cond, *, streaming=False):
        if streaming:
            raise ValueError("This engine is offline-only; streaming requires a different attention mask")
        torch = self.torch
        inputs = dict(zip(INPUT_NAMES, (x, mask, mu, t, spks, cond)))
        frames = validate_input_shapes({k: tuple(v.shape) for k, v in inputs.items()}, self.cfg, self.profile)
        for name, tensor in inputs.items():
            if tensor.device != self.device or tensor.dtype != torch.float32:
                raise ValueError(f"{name} must be FP32 on {self.device}; no implicit transfer/cast")
            if not torch.isfinite(tensor).all().item():
                raise ValueError(f"{name} contains nonfinite values")
        if not ((mask == 0) | (mask == 1)).all().item() or not (mask.sum(dim=-1) > 0).all().item():
            raise ValueError("Each mask row must be binary and contain at least one valid frame")
        with self.lock, torch.cuda.device(self.device), torch.inference_mode():
            stream = torch.cuda.current_stream(self.device)
            inputs = {k: v.contiguous() for k, v in inputs.items()}
            inputs["positions"] = torch.arange(frames, device=self.device, dtype=torch.int32)
            output = torch.empty((2, self.cfg.mel_dim, frames), dtype=torch.float32, device=self.device)
            for name, tensor in inputs.items():
                if not self.context.set_input_shape(name, tuple(tensor.shape)):
                    raise RuntimeError(f"TensorRT rejected {name} shape")
                if not self.context.set_tensor_address(name, tensor.data_ptr()):
                    raise RuntimeError(f"TensorRT rejected {name} address")
            unresolved = self.context.infer_shapes()
            if unresolved or tuple(self.context.get_tensor_shape("velocity")) != tuple(output.shape):
                raise RuntimeError(f"Flow engine shape inference failed: {unresolved}")
            if not self.context.set_tensor_address("velocity", output.data_ptr()):
                raise RuntimeError("TensorRT rejected output address")
            try:
                if not self.context.execute_async_v3(stream.cuda_stream):
                    raise RuntimeError("TensorRT Flow execution failed")
            finally:
                stream.synchronize()
            if not torch.isfinite(output).all().item():
                raise RuntimeError("TensorRT Flow produced NaN/Inf; output rejected")
            return output
