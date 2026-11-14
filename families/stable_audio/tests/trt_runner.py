# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run one family engine from Python for the GPU tests: from a plan, or from a bundle section."""

from __future__ import annotations

import json
import struct
from pathlib import Path

_BUNDLE_MAGIC = b"BUNDLE\x01\x00"


def bundle_section(path: Path, name: str) -> bytes:
    """The bytes of one section of a bundle written by ``BundleWriter``."""
    with Path(path).open("rb") as handle:
        assert handle.read(len(_BUNDLE_MAGIC)) == _BUNDLE_MAGIC, f"{path} is not a bundle"
        (header_size,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(header_size))
        entry = header["sections"][name]
        handle.seek(len(_BUNDLE_MAGIC) + 8 + header_size + int(entry["offset"]))
        data = handle.read(int(entry["length"]))
    assert len(data) == int(entry["length"]), f"{path}: section {name} is truncated"
    return data


class EngineRunner:
    """One deserialized engine and context; float32 outputs as numpy arrays."""

    def __init__(self, plan: bytes) -> None:
        import tensorrt as trt

        self._runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self._engine = self._runtime.deserialize_cuda_engine(plan)
        assert self._engine is not None, "the plan does not deserialize"
        self._context = self._engine.create_execution_context()

    def __call__(self, inputs: dict, outputs: list[str]) -> dict:
        import torch

        tensors = {}
        for name, value in inputs.items():
            tensor = torch.as_tensor(value).contiguous().cuda()
            tensors[name] = tensor
            self._context.set_input_shape(name, tuple(tensor.shape))
            self._context.set_tensor_address(name, tensor.data_ptr())
        results = {}
        for name in outputs:
            shape = tuple(self._context.get_tensor_shape(name))
            results[name] = torch.empty(shape, dtype=torch.float32, device="cuda")
            self._context.set_tensor_address(name, results[name].data_ptr())
        stream = torch.cuda.current_stream()
        assert self._context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        return {name: value.cpu().numpy() for name, value in results.items()}
