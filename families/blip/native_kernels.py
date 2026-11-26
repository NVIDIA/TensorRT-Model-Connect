# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Add family-owned FP32 CUDA primitives to BLIP's TensorRT graph."""

import ctypes
from importlib import metadata
from pathlib import Path

import numpy as np
import tensorrt as trt
from tensorrt_model_connect.build import cmake_prefixes


_LIBRARY = "libtrtmc_model_blip.so"
_LIBRARIES: dict[Path, ctypes.CDLL] = {}


def _model_library():
    """The native model library: a native build directory named in CMAKE_PREFIX_PATH (a source build), else the
    installed native package's."""
    roots = list(cmake_prefixes())
    try:
        roots.append(Path(metadata.distribution("tensorrt-model-connect").locate_file("tensorrt_model_connect/bin")))
    except metadata.PackageNotFoundError:
        pass
    for root in roots:
        if (root / _LIBRARY).is_file():
            return (root / _LIBRARY).resolve()
    raise FileNotFoundError(
        f"BLIP requires its native model library {_LIBRARY}: install BLIP's matching native package or include "
        "the native build directory in CMAKE_PREFIX_PATH"
    )


def load_native_kernels():
    path = _model_library()
    if path not in _LIBRARIES:
        library = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        try:
            initialize = library.trtmc_blip_initialize_fp32_kernels
        except AttributeError as error:
            raise RuntimeError("BLIP requires a matching native model library") from error
        initialize.argtypes = []
        initialize.restype = ctypes.c_char_p
        error = initialize()
        if error:
            raise RuntimeError(error.decode("utf-8", errors="replace"))
        _LIBRARIES[path] = library
    return path


def add_fp32_kernel(network, name, inputs, shape, epsilon=1e-5):
    load_native_kernels()
    operations = {"norm": 0, "linear": 1, "softmax": 2, "gelu": 3}
    creator = trt.get_plugin_registry().get_creator("BlipFp32", "1", "")
    if creator is None:
        raise RuntimeError("BLIP native CUDA plugin is not registered")
    operation = np.array([operations[name]], dtype=np.int32)
    epsilon_value = np.array([epsilon], dtype=np.float32)
    fields = trt.PluginFieldCollection(
        [
            trt.PluginField("operation", operation, trt.PluginFieldType.INT32),
            trt.PluginField("epsilon", epsilon_value, trt.PluginFieldType.FLOAT32),
        ]
    )
    plugin = creator.create_plugin("blip_" + name, fields, trt.TensorRTPhase.BUILD)
    if plugin is None:
        raise RuntimeError(f"BLIP could not create its {name} CUDA plugin")
    layer = network.add_plugin_v3(inputs, [], plugin)
    if layer is None:
        raise RuntimeError(f"BLIP could not add its {name} CUDA plugin")
    output = layer.get_output(0)
    if tuple(output.shape) != tuple(shape):
        raise RuntimeError(f"BLIP {name} CUDA plugin returned an unexpected shape")
    return output
