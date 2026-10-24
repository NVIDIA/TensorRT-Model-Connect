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

Kernel contract (cuTile, N-D): each array keeps its own rank and shape (give ``ranks``), so an
input can be smaller than the output, for example the operand of a broadcast that the pattern
names. The last stride of every array is 1, unless the kernel is built with ``strided=True``: then
the arrays can have any element strides, for example a channel-last input, and the kernel indexes
them through their strides. ``tiles`` has one entry per output dimension (at most
3) and is passed to the kernel as constexpr parameters after the arrays. ``ct.bid(d)`` is the block
index along output dimension ``d``, and the grid is ``ceil(dim / tile)`` per dimension.

A cubin kernel records the architecture it was built for. ``user_kernels`` raises if the current
device has another one, because the engine build would otherwise fail with a module load error.

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
from collections.abc import Iterable, Iterator, Sequence

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
    # GPU architecture the cubin was built for, for example "sm_90". Empty means unknown, and the
    # check in ``user_kernels`` is skipped.
    arch: str = ""
    tiles: tuple[int, ...] = ()
    strided: bool = False

    @classmethod
    def from_cutile(
        cls,
        name: str,
        pattern: str,
        kernel,
        *,
        tile: int | None = None,
        tiles: Sequence[int] | None = None,
        num_arrays: int | None = None,
        ranks: Sequence[int] | None = None,
        dtype: str | Sequence[str] = "float32",
        strided: bool = False,
        gpu_code: str | None = None,
    ) -> UserKernel:
        """Build a cuTile kernel into a cubin.

        Flat mode (``tile`` and ``num_arrays``): ``kernel`` is a ``@ct.kernel`` function
        ``(arr_0, ..., arr_{num_arrays-1}, TILE)``, the pattern inputs then the output as 1-D
        arrays, then the tile size.

        N-D mode (``tiles`` and ``ranks``): ``ranks[i]`` is the rank of array ``i`` (pattern
        inputs, then the output), and ``tiles`` has one entry per output dimension. The kernel is
        ``(arr_0, ..., arr_k, TILE_0, ..., TILE_{rank-1})``.

        ``dtype`` is one cuTile dtype name for all arrays or one per array. With ``strided`` the
        cubin makes no assumption about the strides (N-D mode only), so inputs with another layout
        than row-major work. Without it the compiler rejects such inputs.
        """
        if (tile is None) == (tiles is None):
            raise ValueError("give exactly one of tile (flat mode) and tiles (N-D mode)")
        if tiles is None:
            if num_arrays is None:
                raise ValueError("flat mode needs num_arrays")
            ranks = [1] * num_arrays
            tile_args: tuple[int, ...] = (tile,)
        else:
            if ranks is None:
                raise ValueError("N-D mode needs ranks")
            tile_args = tuple(tiles)
            if not 1 <= len(tile_args) <= 3 or ranks[-1] != len(tile_args):
                raise ValueError("tiles needs one entry (at most 3) per output dimension")
        if strided and tiles is None:
            raise ValueError("strided needs the N-D mode (tiles and ranks)")
        entry = f"{name}_tile"
        gpu_code = gpu_code or device_arch()
        if gpu_code is None:
            raise RuntimeError("no CUDA device found; pass gpu_code to build for a given GPU")
        cubin = cutile_cubin(
            kernel,
            ranks=ranks,
            dtype=dtype,
            tiles=tile_args,
            entry=entry,
            strided=strided,
            gpu_code=gpu_code,
        )
        return cls(
            name,
            pattern,
            block=_TILE_THREADS,
            cubin=cubin,
            entry=entry,
            tile=tile or 0,
            tiles=tuple(tiles) if tiles is not None else (),
            strided=strided,
            arch=gpu_code,
        )


def cutile_cubin(
    kernel,
    *,
    ranks: Sequence[int],
    dtype: str | Sequence[str],
    tiles: Sequence[int],
    entry: str,
    strided: bool = False,
    gpu_code: str | None = None,
) -> bytes:
    """Compile ``kernel`` for arrays of the given ranks (int64 shape and stride) into a cubin."""
    import cuda.tile as ct
    from cuda.tile.compilation import (
        ArrayConstraint,
        CallingConvention,
        KernelSignature,
        export_kernel,
    )

    gpu_code = gpu_code or device_arch()
    if gpu_code is None:
        raise RuntimeError("no CUDA device found; pass gpu_code to build for a given GPU")
    dtypes = [dtype] * len(ranks) if isinstance(dtype, str) else list(dtype)
    if len(dtypes) != len(ranks):
        raise ValueError("dtype needs one entry per array")

    def array(element_name: str, rank: int):
        return ArrayConstraint(
            getattr(ct, element_name),
            rank,
            index_dtype=ct.int64,
            stride_constant=None if strided else (None,) * (rank - 1) + (1,),
            alias_groups=[],
            stride_lower_bound_incl=0,
            may_alias_internally=False,
        )

    signature = KernelSignature(
        [array(d, r) for d, r in zip(dtypes, ranks)] + list(tiles),
        CallingConvention.cutile_python_v2(),
        symbol=entry,
    )
    buffer = io.BytesIO()
    export_kernel(kernel, [signature], buffer, gpu_code=gpu_code, output_format="cubin")
    return buffer.getvalue()


def device_arch() -> str | None:
    """Architecture of the current CUDA device, for example ``sm_90``. None without a device."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    return f"sm_{major}{minor}"


def _same_arch(built: str, current: str) -> bool:
    # "sm_100a" and "sm_100f" are variants of sm_100.
    return built.rstrip("af") == current.rstrip("af")


def check_architecture(kernels: Iterable[UserKernel], current: str | None = None) -> None:
    """Raise if a cubin kernel was built for another GPU architecture than the current device.

    The check is skipped for kernels without ``arch`` and when there is no CUDA device.
    """
    current = current or device_arch()
    if current is None:
        return
    for kernel in kernels:
        if kernel.cubin and kernel.arch and not _same_arch(kernel.arch, current):
            raise ValueError(
                f"kernel {kernel.name!r} was built for {kernel.arch}, but the current device is "
                f"{current}; rebuild it on this device or build on the matching device"
            )


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
        if not kernel.entry or (kernel.tile <= 0 and not kernel.tiles):
            raise ValueError(f"cubin kernel {kernel.name!r} needs an entry and a tile")
        if kernel.tiles and (len(kernel.tiles) > 3 or min(kernel.tiles) <= 0):
            raise ValueError(f"tiles of {kernel.name!r} must be 1 to 3 positive numbers")
        if directory is None or not _NAME.fullmatch(kernel.name):
            raise ValueError(f"cubin kernel {kernel.name!r} needs a plain name and a directory")
        path = os.path.join(directory, kernel.name + ".cubin")
        with open(path, "wb") as f:
            f.write(kernel.cubin)
        attrs += (
            f", nvm.user_kernel_cubin = {_mlir_string(path)}"
            f", nvm.user_kernel_entry = {_mlir_string(kernel.entry)}"
        )
        if kernel.strided:
            attrs += ", nvm.user_kernel_strided = true"
        if kernel.tiles:
            attrs += f", nvm.user_kernel_tiles = array<i64: {', '.join(map(str, kernel.tiles))}>"
        else:
            attrs += f", nvm.user_kernel_tile = {kernel.tile} : i64"
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
    check_architecture(kernels)
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
