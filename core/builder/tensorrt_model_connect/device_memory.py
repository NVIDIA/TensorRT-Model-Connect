# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Let a TensorRT build use all of an integrated GPU's memory.

TensorRT's default GPU allocator takes builder memory (the device copy of the weights, tactic workspaces) with
cudaMallocAsync from the device's current memory pool. On Jetson AGX Thor (Jetson Linux R39.2.2) a pool created
with maxSize 0, the default pool included, holds about a third of the memory (40.5 of 122.8 GiB) while cudaMalloc
still reaches the rest, so a build that needs a larger device buffer fails with free memory. The native runtime
switches pools the same way before it deserializes an engine (core/runtime/primitives/cuda_common.h).
"""

from __future__ import annotations

import ctypes

_CU_DEVICE_ATTRIBUTE_INTEGRATED = 18
_CU_MEM_ALLOCATION_TYPE_PINNED = 1
_CU_MEM_LOCATION_TYPE_DEVICE = 1


class _Location(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _PoolProps(ctypes.Structure):
    # CUmemPoolProps (CUDA 12.2 and later).
    _fields_ = [
        ("allocType", ctypes.c_int),
        ("handleTypes", ctypes.c_int),
        ("location", _Location),
        ("win32SecurityAttributes", ctypes.c_void_p),
        ("maxSize", ctypes.c_size_t),
        ("usage", ctypes.c_ushort),
        ("reserved", ctypes.c_ubyte * 54),
    ]


def _driver():
    try:
        return ctypes.CDLL("libcuda.so.1")
    except OSError:
        return None


def use_full_size_memory_pool(ordinal: int = 0) -> bool:
    """Make a pool with maxSize = the device memory the current pool of an integrated GPU that still uses its
    default pool. A discrete GPU, a device that already has another current pool, a machine without a CUDA
    driver and any driver error leave everything as it was. Returns True when the pool was replaced; the pool
    lives until the process exits."""
    cuda = _driver()
    if cuda is None:
        return False
    device, integrated = ctypes.c_int(), ctypes.c_int()
    current, default, pool = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    total = ctypes.c_size_t()
    if (
        cuda.cuInit(0)
        or cuda.cuDeviceGet(ctypes.byref(device), ordinal)
        or cuda.cuDeviceGetAttribute(ctypes.byref(integrated), _CU_DEVICE_ATTRIBUTE_INTEGRATED, device)
        or not integrated.value
        or cuda.cuDeviceGetMemPool(ctypes.byref(current), device)
        or cuda.cuDeviceGetDefaultMemPool(ctypes.byref(default), device)
        or current.value != default.value
        or cuda.cuDeviceTotalMem_v2(ctypes.byref(total), device)
    ):
        return False
    props = _PoolProps()
    props.allocType = _CU_MEM_ALLOCATION_TYPE_PINNED
    props.location = _Location(_CU_MEM_LOCATION_TYPE_DEVICE, device.value)
    props.maxSize = total.value
    if cuda.cuMemPoolCreate(ctypes.byref(pool), ctypes.byref(props)):
        return False
    if cuda.cuDeviceSetMemPool(device, pool):
        cuda.cuMemPoolDestroy(pool)
        return False
    return True
