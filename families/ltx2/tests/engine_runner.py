# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a serialized TensorRT plan on torch CUDA tensors (test helper)."""

from __future__ import annotations

import tensorrt as trt
import torch

from families.ltx2.graph import make_logger

_TORCH = {
    trt.float32: torch.float32,
    trt.float16: torch.float16,
    trt.bfloat16: torch.bfloat16,
    trt.int32: torch.int32,
    trt.bool: torch.bool,
}


class Engine:
    """Deserialized plan + execution context, reusable across calls."""

    def __init__(self, plan: bytes, communicator=None):
        self.runtime = trt.Runtime(make_logger())
        self.engine = self.runtime.deserialize_cuda_engine(plan)
        if self.engine is None:
            raise RuntimeError("failed to deserialize plan")
        self.context = self.engine.create_execution_context()
        if communicator is not None and not self.context.set_communicator(communicator):
            raise RuntimeError("set_communicator failed")
        self.stream = torch.cuda.Stream()

    def __call__(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return _execute(self.engine, self.context, inputs, self.stream)


def run_plan(plan: bytes, inputs: dict[str, torch.Tensor], communicator=None) -> dict[str, torch.Tensor]:
    return Engine(plan, communicator)(inputs)


def _execute(engine, context, inputs, stream) -> dict[str, torch.Tensor]:
    keep = []
    outputs: dict[str, torch.Tensor] = {}
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    is_input = {name: engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT for name in names}
    for name in (n for n in names if is_input[n]):  # inputs first: they fix run-time dimensions
        dtype = _TORCH[engine.get_tensor_dtype(name)]
        shape = tuple(engine.get_tensor_shape(name))
        t = inputs[name].to(device="cuda", dtype=dtype).contiguous()
        if -1 in shape:
            context.set_input_shape(name, tuple(t.shape))
        elif tuple(t.shape) != shape:
            raise ValueError(f"{name}: shape {tuple(t.shape)} != engine {shape}")
        keep.append(t)
        context.set_tensor_address(name, t.data_ptr())
    for name in (n for n in names if not is_input[n]):
        t = torch.empty(tuple(context.get_tensor_shape(name)), dtype=_TORCH[engine.get_tensor_dtype(name)],
                        device="cuda")
        outputs[name] = t
        context.set_tensor_address(name, t.data_ptr())
    torch.cuda.current_stream().synchronize()
    if not context.execute_async_v3(stream.cuda_stream):
        raise RuntimeError("execute_async_v3 failed")
    stream.synchronize()
    return outputs


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().double().flatten()
    b = b.detach().double().flatten()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().double()
    b = b.detach().double()
    return float((a - b).norm() / (b.norm() + 1e-30))
