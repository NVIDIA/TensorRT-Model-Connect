# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Add an explicitly named TVM-FFI kernel to a family-owned TensorRT graph."""

from __future__ import annotations

import ctypes
import json
import re
from pathlib import Path
from typing import Any


_KERNEL_NAME = re.compile(r"[A-Za-z0-9_.@-]+\Z")
_LOADED_PLUGIN_LIBRARIES: list[ctypes.CDLL] = []


def add_kernel(
    network: Any,
    *,
    plugin_library: str | Path,
    kernel_name: str,
    inputs: list[Any],
    output_specs: list[dict[str, Any]],
    workspace_bytes: int = 0,
    extra_args: list[dict[str, Any]] | None = None,
) -> list[Any]:
    """Add one TVM-FFI TensorRT plugin layer to the network."""

    path = Path(plugin_library).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"BYOK plugin library does not exist: {path}")
    if _KERNEL_NAME.fullmatch(kernel_name) is None:
        raise ValueError("kernel_name contains unsupported characters")
    if not inputs or not output_specs:
        raise ValueError("BYOK requires at least one input and output")
    if workspace_bytes < 0:
        raise ValueError("workspace_bytes must be non-negative")

    _LOADED_PLUGIN_LIBRARIES.append(ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL))

    import tensorrt as trt

    creator = trt.get_plugin_registry().get_creator("TvmFfiKernel", "1", "")
    if creator is None:
        raise RuntimeError("TvmFfiKernel is not registered by the requested plugin library")

    spec: dict[str, Any] = {
        "num_inputs": len(inputs),
        "num_outputs": len(output_specs),
        "outputs": output_specs,
        "workspace_bytes": workspace_bytes,
    }
    if extra_args:
        spec["extra_args"] = extra_args
    fields = trt.PluginFieldCollection(
        [
            trt.PluginField(
                "kernel_name", kernel_name.encode("utf-8"), trt.PluginFieldType.CHAR
            ),
            trt.PluginField(
                "shape_spec",
                json.dumps(spec, separators=(",", ":")).encode("utf-8"),
                trt.PluginFieldType.CHAR,
            ),
        ]
    )
    plugin = creator.create_plugin("tvm_ffi_kernel", fields)
    if plugin is None:
        raise RuntimeError("TensorRT failed to create the TvmFfiKernel plugin")
    layer = network.add_plugin_v2(inputs, plugin)
    if layer is None:
        raise RuntimeError("TensorRT failed to add the TvmFfiKernel plugin")
    return [layer.get_output(index) for index in range(layer.num_outputs)]
