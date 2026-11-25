# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pika's native mathematical bindings through the public TVM-FFI interface."""

import ctypes
from importlib import metadata
from pathlib import Path

from tensorrt_model_connect.byok import add_kernel

_LIBRARIES: dict[Path, ctypes.CDLL] = {}


def _bridge():
    package = metadata.distribution("tensorrt-model-connect")
    directory = Path(package.locate_file("tensorrt_model_connect/bin"))
    library_path = (directory / "libtrtmc_model_pika_h3_flex.so").resolve()
    if library_path not in _LIBRARIES:
        library = ctypes.CDLL(str(library_path), mode=ctypes.RTLD_GLOBAL)
        initialize = library.trtmc_pika_h3_flex_initialize_math
        initialize.argtypes = []
        initialize.restype = ctypes.c_char_p
        error = initialize()
        if error:
            raise RuntimeError(error.decode("utf-8", errors="replace"))
        _LIBRARIES[library_path] = library
    return library_path.parent / "libtrtmc_byok_tvm_ffi.so"


def add_math(network, name, inputs, *, shape=None, extra_args=None):
    import tensorrt as trt

    types = {trt.float32: "float32", trt.bfloat16: "bfloat16"}
    if inputs[0].dtype not in types:
        raise ValueError("Pika transformer math supports FP32 and BF16")
    return add_kernel(
        network,
        plugin_library=_bridge(),
        kernel_name=f"pika_h3_flex.{name}_v1",
        inputs=inputs,
        output_specs=[
            {
                "dims": "same_as_input_0" if shape is None else shape,
                "dtype": types[inputs[0].dtype],
            }
        ],
        extra_args=extra_args,
    )[0]
