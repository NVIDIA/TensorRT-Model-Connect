# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small shared builder helper for the TensorRT TVM-FFI plugin."""

from __future__ import annotations

import json
from typing import Any

from .. import trt_compat


def add_tvm_ffi_kernel(
    network: Any,
    *,
    kernel_name: str,
    inputs: list[Any],
    output_specs: list[dict[str, Any]],
    workspace_bytes: int = 0,
    extra_args: list[dict[str, Any]] | None = None,
) -> list[Any]:
    """Add one ``TvmFfiKernel`` V2 plugin layer to ``network``."""

    trt_compat.load_native_backend_plugins()
    trt = trt_compat.get_trt()
    registry = trt.get_plugin_registry()
    get_creator = getattr(registry, "get_creator", None)
    creator = (
        get_creator("TvmFfiKernel", "1", "")
        if get_creator is not None
        else registry.get_plugin_creator("TvmFfiKernel", "1", "")
    )
    if creator is None:
        raise RuntimeError(
            "TvmFfiKernel plugin is unavailable; build Model Connect with TVM-FFI support"
        )

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
                "kernel_name",
                kernel_name.encode("utf-8"),
                trt.PluginFieldType.CHAR,
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
        raise RuntimeError("TensorRT failed to add the TvmFfiKernel plugin layer")
    return [layer.get_output(index) for index in range(layer.num_outputs)]
