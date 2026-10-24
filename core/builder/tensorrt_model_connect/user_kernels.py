# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Register replacement kernels for op patterns when building a TensorRT engine.

A user supplies, per kernel:
  - pattern: MLIR text of an ``nvg.graph`` that describes the ops to replace,
  - source:  kernel source that replaces them (CUDA C++), or
  - cubin:   a prebuilt kernel, for example a cuTile kernel built by ``UserKernel.from_cutile``,
  - block:   threads per block.

``user_kernels(...)`` writes one registration file and turns the build option on for the
duration of the engine build. TensorRT matches the pattern during the build and emits the kernel.
A TensorRT build that does not know the option ignores it and builds the original ops.

Kernel contract (CUDA source): ``extern "C" __global__ void USER_KERNEL_ENTRY(const T0 *in0, ...,
T *out, long long n)``. Inputs follow the pattern's argument order, one thread per output element,
static shapes only. ``USER_KERNEL_ENTRY`` is defined by the compiler.

Kernel contract (cuTile): one 1-D array parameter per pattern input (in the pattern's argument
order), then one for the output, then constexpr parameters. The arrays are flat views of the
contiguous tensors with int64 shape and stride. Thread block count is ``ceil(n / tile)``.

Usage::

    with user_kernels([UserKernel("user_fma", PATTERN, SOURCE, block=128)]):
        blob = builder.build_serialized_network(network, config)
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import os
import re
import tempfile
from collections.abc import Iterable, Iterator

# Internal option channel that the TensorRT builder reads. It needs no TensorRT API change.
_ENV = "__LUNOWUD"
_OPTION = "-nvm:user_kernels="


_NAME = re.compile(r"[A-Za-z0-9_.-]+")
# The tile compiler fixes the thread count of every kernel.
_TILE_THREADS = 128


@dataclasses.dataclass(frozen=True)
class UserKernel:
    name: str
    pattern: str
    source: str = ""
    block: int = 256
    cubin: bytes = b""
    entry: str = ""
    tile: int = 0

    @classmethod
    def from_cutile(
        cls,
        name: str,
        pattern: str,
        kernel,
        *,
        tile: int,
        num_arrays: int,
        dtype: str = "float32",
        gpu_code: str | None = None,
    ) -> UserKernel:
        """Build a cuTile kernel into a cubin.

        ``kernel`` is a ``@ct.kernel`` function ``(arr_0, ..., arr_{num_arrays-1}, TILE)``: the
        pattern inputs, then the output, as 1-D arrays of ``dtype``, then the tile size.
        """
        entry = f"{name}_tile"
        cubin = cutile_cubin(
            kernel, num_arrays=num_arrays, dtype=dtype, tile=tile, entry=entry, gpu_code=gpu_code
        )
        return cls(name, pattern, block=_TILE_THREADS, cubin=cubin, entry=entry, tile=tile)


def cutile_cubin(
    kernel, *, num_arrays: int, dtype: str, tile: int, entry: str, gpu_code: str | None = None
) -> bytes:
    """Compile ``kernel`` for flat 1-D arrays with int64 shape and stride into a cubin."""
    import cuda.tile as ct
    from cuda.tile.compilation import (
        ArrayConstraint,
        CallingConvention,
        KernelSignature,
        export_kernel,
    )

    if gpu_code is None:
        import torch

        major, minor = torch.cuda.get_device_capability()
        gpu_code = f"sm_{major}{minor}"
    element = getattr(ct, dtype)

    def array():
        return ArrayConstraint(
            element,
            1,
            index_dtype=ct.int64,
            stride_constant=(1,),
            alias_groups=[],
            stride_lower_bound_incl=0,
            may_alias_internally=False,
        )

    signature = KernelSignature(
        [array() for _ in range(num_arrays)] + [tile],
        CallingConvention.cutile_python_v2(),
        symbol=entry,
    )
    buffer = io.BytesIO()
    export_kernel(kernel, [signature], buffer, gpu_code=gpu_code, output_format="cubin")
    return buffer.getvalue()


def _mlir_string(text: str) -> str:
    out = []
    for ch in text:
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\0A")
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


def _tag_pattern(kernel: UserKernel, directory: str | None) -> str:
    """Insert the kernel attributes into the pattern's ``attributes {`` clause."""
    marker = "attributes {"
    pos = kernel.pattern.find(marker)
    if pos < 0:
        raise ValueError(f"pattern of {kernel.name!r} needs an `attributes {{...}}` clause")
    if "nvm.user_kernel" in kernel.pattern:
        raise ValueError(f"pattern of {kernel.name!r} must not set nvm.user_kernel attributes")
    if kernel.block <= 0 or kernel.block > 1024:
        raise ValueError(f"block of {kernel.name!r} must be in 1..1024")
    if bool(kernel.source) == bool(kernel.cubin):
        raise ValueError(f"{kernel.name!r} needs exactly one of source and cubin")
    attrs = (
        f"nvm.user_kernel = {_mlir_string(kernel.name)}, "
        f"nvm.user_kernel_block = {kernel.block} : i64"
    )
    if kernel.source:
        attrs += f", nvm.user_kernel_source = {_mlir_string(kernel.source)}"
    else:
        if not kernel.entry or kernel.tile <= 0:
            raise ValueError(f"cubin kernel {kernel.name!r} needs an entry and a tile")
        if directory is None or not _NAME.fullmatch(kernel.name):
            raise ValueError(f"cubin kernel {kernel.name!r} needs a plain name and a directory")
        path = os.path.join(directory, kernel.name + ".cubin")
        with open(path, "wb") as f:
            f.write(kernel.cubin)
        attrs += (
            f", nvm.user_kernel_cubin = {_mlir_string(path)}"
            f", nvm.user_kernel_entry = {_mlir_string(kernel.entry)}"
            f", nvm.user_kernel_tile = {kernel.tile} : i64"
        )
    end = pos + len(marker)
    rest = kernel.pattern[end:].lstrip()
    sep = "" if rest.startswith("}") else ", "
    return kernel.pattern[:end] + attrs + sep + kernel.pattern[end:]


def registration_text(kernels: Iterable[UserKernel], directory: str | None = None) -> str:
    """Return the registration file content. Each pattern must be a module-less ``nvg.graph``.

    Cubin kernels are written to ``directory`` and referenced by path.
    """
    kernels = list(kernels)
    names = [k.name for k in kernels]
    if len(set(names)) != len(names):
        raise ValueError("kernel names must be unique")
    body = "\n".join(_tag_pattern(k, directory) for k in kernels)
    return "module {\n" + body + "\n}\n"


def option_string(path: str, previous: str = "") -> str:
    """Return the builder option string. The builder splits options on spaces."""
    if any(c.isspace() for c in path):
        raise ValueError(f"registration path must not contain spaces: {path!r}")
    option = _OPTION + path
    return f"{previous} {option}".strip() if previous else option


@contextlib.contextmanager
def user_kernels(kernels: Iterable[UserKernel]) -> Iterator[str | None]:
    """Enable the kernels for engine builds inside the ``with`` block."""
    kernels = list(kernels)
    if not kernels:
        yield None
        return
    with tempfile.TemporaryDirectory(prefix="user_kernels_") as tmp:
        path = os.path.join(tmp, "registration.mlir")
        with open(path, "w") as f:
            f.write(registration_text(kernels, tmp))
        previous = os.environ.get(_ENV)
        os.environ[_ENV] = option_string(path, previous or "")
        try:
            yield path
        finally:
            if previous is None:
                os.environ.pop(_ENV, None)
            else:
                os.environ[_ENV] = previous
