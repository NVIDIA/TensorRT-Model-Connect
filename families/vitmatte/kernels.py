# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Family-owned FP32 operations exposed through the TVM-FFI BYOK contract."""

import ctypes
from functools import cache
from pathlib import Path
import sysconfig

import tensorrt_model_connect.byok as byok
from tensorrt_model_connect.build import cmake_prefixes
from tensorrt_model_connect.byok import add_kernel


def _runtime_root():
    roots = [*cmake_prefixes(), Path(byok.__file__).resolve().parent / "bin"]
    roots.append(Path(sysconfig.get_path("platlib")) / "tensorrt_model_connect/bin")
    for root in roots:
        if (root / "libtrtmc_model_vitmatte.so").is_file():
            return root.resolve()
    raise FileNotFoundError(
        "ViTMatte requires its native family library; install the native wheel or "
        "include the native build directory in CMAKE_PREFIX_PATH"
    )


@cache
def _libraries(root):
    family = root / "libtrtmc_model_vitmatte.so"
    bridge = root / "libtrtmc_byok_tvm_ffi.so"
    if not family.is_file() or not bridge.is_file():
        raise FileNotFoundError(
            f"ViTMatte requires its native family library and TVM-FFI BYOK bridge in {root}; "
            "install the native wheel or include the native build directory in CMAKE_PREFIX_PATH"
        )
    library = ctypes.CDLL(str(family), mode=ctypes.RTLD_GLOBAL)
    initialize = library.trtmc_vitmatte_register_kernels
    initialize.argtypes = []
    initialize.restype = ctypes.c_char_p
    error = initialize()
    if error is not None:
        raise RuntimeError(error.decode("utf-8", errors="replace"))
    return library, bridge


def mlp_output(network, inp, weight, bias):
    """FP32 flattened projection, with the bias added after the dot product."""
    _, bridge = _libraries(_runtime_root())
    weights = network.add_constant(weight.shape, weight).get_output(0)
    biases = network.add_constant(bias.shape, bias).get_output(0)
    return add_kernel(
        network,
        plugin_library=bridge,
        kernel_name="vitmatte.fp32_mlp_output_v1",
        inputs=[inp, weights, biases],
        output_specs=[{"dims": [*inp.shape[:-1], weight.shape[0]], "dtype": "float32"}],
    )[0]


def gelu(network, inp):
    """Exact FP32 erf GELU using the stream supplied by TensorRT."""
    _, bridge = _libraries(_runtime_root())
    return add_kernel(
        network,
        plugin_library=bridge,
        kernel_name="vitmatte.fp32_gelu_v1",
        inputs=[inp],
        output_specs=[{"dims": list(inp.shape), "dtype": "float32"}],
    )[0]
