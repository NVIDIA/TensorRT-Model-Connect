# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V2 policy for using the native TensorRT MoE layer.

TensorRT 11.1 exposes ``INetworkDefinition.add_moe()`` and ``IMoELayer``. The
layer receives the routed expert ids and their scores as runtime inputs, so it
evaluates only the experts a token was actually routed to. Emitting one
subgraph per expert and gathering the selected outputs afterwards instead
evaluates every expert for every token, because the selection happens after the
expert matmuls have already been placed in the graph.

This module is owned by the DeepSeek-V2 family and is intentionally not shared.
Another family that wants the native layer should carry its own policy, so each
family keeps control of the compute capabilities it has actually qualified.

TensorRT restricts the layer to SM 10.x and SM 11.x and rejects SM 12.x inside
``addMoE`` itself, returning a null layer rather than failing the build. The
decision therefore has to be made before the layer is added, which is what this
module provides.

Only SM 10.x is enabled here. TensorRT also permits SM 11.x (Thor), but
DeepSeek-V2 has not been qualified on that target, so it stays on the portable
per-expert path until it is.

Set ``TRTMC_DISABLE_NATIVE_MOE=1`` to force the per-expert path on a supported
GPU, which is useful when bisecting a numerical difference between the two.
"""

from __future__ import annotations

import os
from typing import Any

from ... import trt_compat


DISABLE_ENV = "TRTMC_DISABLE_NATIVE_MOE"

#: Compute-capability majors on which the native MoE layer is enabled.
SUPPORTED_COMPUTE_MAJORS = (10,)


def native_moe_enabled_for(
    compute_capability: tuple[int, int] | None,
    *,
    layer_available: bool,
    disabled: bool = False,
) -> bool:
    """Return whether a MoE graph should be built with the native TRT layer.

    Kept free of TensorRT and CUDA calls so the policy itself is testable on a
    machine without a GPU.
    """
    if disabled or not layer_available:
        return False
    if compute_capability is None:
        return False
    major, _minor = compute_capability
    return major in SUPPORTED_COMPUTE_MAJORS


def _cuda_runtime() -> Any | None:
    try:
        from cuda.bindings import runtime as cudart
    except ImportError:
        try:
            from cuda import cudart
        except ImportError:
            return None
    return cudart


def current_compute_capability() -> tuple[int, int] | None:
    """Compute capability of the active CUDA device, or ``None`` if unknown.

    An unusable or absent CUDA device is not an error here; it only means the
    native layer cannot be selected and the caller keeps the portable path.
    """
    runtime = _cuda_runtime()
    if runtime is None:
        return None
    success = getattr(getattr(runtime, "cudaError_t", None), "cudaSuccess", 0)
    try:
        status, device = runtime.cudaGetDevice()
        if status not in (success, 0):
            return None
        status, properties = runtime.cudaGetDeviceProperties(int(device))
        if status not in (success, 0):
            return None
        major = int(properties.major)
        minor = int(properties.minor)
    except Exception:
        return None
    if major <= 0 or minor < 0:
        return None
    return (major, minor)


def native_moe_layer_available() -> bool:
    """Whether the bound TensorRT Python module exposes the MoE layer."""
    trt = trt_compat.get_trt()
    return hasattr(trt.INetworkDefinition, "add_moe") and hasattr(trt, "MoEActType")


def use_native_moe() -> bool:
    """Resolve the native-MoE decision for the current build environment."""
    return native_moe_enabled_for(
        current_compute_capability(),
        layer_available=native_moe_layer_available(),
        disabled=os.environ.get(DISABLE_ENV) == "1",
    )
