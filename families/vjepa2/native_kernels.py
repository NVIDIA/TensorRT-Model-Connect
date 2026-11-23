# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load V-JEPA2's native FP32 primitives through the public TVM-FFI bridge."""

import ctypes
from importlib import metadata
from pathlib import Path

from tensorrt_model_connect.byok import add_kernel


_LIBRARIES: dict[Path, ctypes.CDLL] = {}


def _model_library():
    try:
        distribution = metadata.distribution("tensorrt-model-connect")
    except metadata.PackageNotFoundError as error:
        raise RuntimeError(
            "Install V-JEPA2's matching native package before building bundles"
        ) from error
    path = Path(distribution.locate_file("tensorrt_model_connect/bin/libtrtmc_model_vjepa2.so"))
    if not path.is_file():
        raise FileNotFoundError(f"V-JEPA2 native model library was not built: {path}")
    return path.resolve()


def load_native_kernels():
    path = _model_library()
    if path not in _LIBRARIES:
        library = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        try:
            initialize = library.trtmc_vjepa2_initialize_fp32_kernels
        except AttributeError as error:
            raise RuntimeError("V-JEPA2 requires a matching native model library") from error
        initialize.argtypes = []
        initialize.restype = ctypes.c_char_p
        error = initialize()
        if error:
            raise RuntimeError(error.decode("utf-8", errors="replace"))
        _LIBRARIES[path] = library
    return path.parent / "libtrtmc_byok_tvm_ffi.so"


def add_fp32_kernel(network, name, inputs, shape, extra_args=None):
    bridge = load_native_kernels()
    return add_kernel(
        network,
        plugin_library=bridge,
        kernel_name=f"vjepa2.{name}_cuda_fp32_v1",
        inputs=inputs,
        output_specs=[{"dims": list(shape), "dtype": "float32"}],
        extra_args=extra_args,
        workspace_bytes=32 * 1024 * 1024 if name == "linear" else 0,
    )[0]


def add_norm(network, source, scale, bias, epsilon):
    return add_fp32_kernel(
        network,
        "norm",
        [source, scale, bias],
        source.shape,
        [{"type": "float", "value": float(epsilon)}],
    )
