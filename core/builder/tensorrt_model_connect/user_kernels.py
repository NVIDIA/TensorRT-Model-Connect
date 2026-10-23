# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Register replacement kernels for op patterns when building a TensorRT engine.

A user supplies, per kernel:
  - pattern: MLIR text of an ``nvg.graph`` that describes the ops to replace,
  - source:  kernel source that replaces them (CUDA C++ today),
  - block:   threads per block.

``user_kernels(...)`` writes one registration file and turns the build option on for the
duration of the engine build. TensorRT matches the pattern during the build and emits the kernel.
A TensorRT build that does not know the option ignores it and builds the original ops.

Kernel contract (CUDA source): ``extern "C" __global__ void USER_KERNEL_ENTRY(const T0 *in0, ...,
T *out, long long n)``. Inputs follow the pattern's argument order, one thread per output element,
static shapes only. ``USER_KERNEL_ENTRY`` is defined by the compiler.

Usage::

    with user_kernels([UserKernel("user_fma", PATTERN, SOURCE, block=128)]):
        blob = builder.build_serialized_network(network, config)
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import tempfile
from collections.abc import Iterable, Iterator

# Internal option channel that the TensorRT builder reads. It needs no TensorRT API change.
_ENV = "__LUNOWUD"
_OPTION = "-nvm:user_kernels="


@dataclasses.dataclass(frozen=True)
class UserKernel:
    name: str
    pattern: str
    source: str
    block: int = 256


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


def _tag_pattern(kernel: UserKernel) -> str:
    """Insert the kernel attributes into the pattern's ``attributes {`` clause."""
    marker = "attributes {"
    pos = kernel.pattern.find(marker)
    if pos < 0:
        raise ValueError(f"pattern of {kernel.name!r} needs an `attributes {{...}}` clause")
    if "nvm.user_kernel" in kernel.pattern:
        raise ValueError(f"pattern of {kernel.name!r} must not set nvm.user_kernel attributes")
    if kernel.block <= 0 or kernel.block > 1024:
        raise ValueError(f"block of {kernel.name!r} must be in 1..1024")
    attrs = (
        f"nvm.user_kernel = {_mlir_string(kernel.name)}, "
        f"nvm.user_kernel_block = {kernel.block} : i64, "
        f"nvm.user_kernel_source = {_mlir_string(kernel.source)}"
    )
    end = pos + len(marker)
    rest = kernel.pattern[end:].lstrip()
    sep = "" if rest.startswith("}") else ", "
    return kernel.pattern[:end] + attrs + sep + kernel.pattern[end:]


def registration_text(kernels: Iterable[UserKernel]) -> str:
    """Return the registration file content. Each pattern must be a module-less ``nvg.graph``."""
    kernels = list(kernels)
    names = [k.name for k in kernels]
    if len(set(names)) != len(names):
        raise ValueError("kernel names must be unique")
    body = "\n".join(_tag_pattern(k) for k in kernels)
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
            f.write(registration_text(kernels))
        previous = os.environ.get(_ENV)
        os.environ[_ENV] = option_string(path, previous or "")
        try:
            yield path
        finally:
            if previous is None:
                os.environ.pop(_ENV, None)
            else:
                os.environ[_ENV] = previous
