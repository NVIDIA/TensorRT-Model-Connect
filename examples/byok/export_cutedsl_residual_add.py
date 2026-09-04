#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile a fixed-shape FP16 residual add into a TVM-FFI kernel DSO."""

import argparse
from importlib import metadata
from pathlib import Path
import shutil
import subprocess
import tempfile


ROWS = 256
COLS = 768
THREADS = 256


def _compile():
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    @cute.kernel
    def residual_add_kernel(
        hidden: cute.Tensor,
        attention_projection: cute.Tensor,
        output: cute.Tensor,
    ):
        thread_x, _, _ = cute.arch.thread_idx()
        block_x, _, _ = cute.arch.block_idx()
        block_size, _, _ = cute.arch.block_dim()
        linear_index = block_x * block_size + thread_x
        row = linear_index // COLS
        column = linear_index % COLS
        output[row, column] = hidden[row, column] + attention_projection[row, column]

    @cute.jit
    def run(
        hidden: cute.Tensor,
        attention_projection: cute.Tensor,
        output: cute.Tensor,
        stream: cuda.CUstream,
    ):
        residual_add_kernel(hidden, attention_projection, output).launch(
            grid=((ROWS * COLS) // THREADS, 1, 1),
            block=(THREADS, 1, 1),
            stream=stream,
        )

    def tensor():
        return cute.runtime.make_fake_compact_tensor(
            cutlass.Float16,
            (ROWS, COLS),
            stride_order=(1, 0),
            assumed_align=16,
        )

    return cute.compile(
        run,
        tensor(),
        tensor(),
        tensor(),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi --opt-level 3",
    )


def _runtime_archive() -> Path:
    distribution = metadata.distribution("nvidia-cutlass-dsl-libs-cu12")
    archive = Path(
        distribution.locate_file(
            "nvidia_cutlass_dsl/cu12/lib/libcuda_dialect_runtime_static.a"
        )
    ).resolve()
    if not archive.is_file():
        raise SystemExit(f"CuTe DSL runtime archive not found: {archive}")
    return archive


def _link(compiled, output: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="trtmc-cutedsl-") as temporary:
        root = Path(temporary)
        object_file = root / "kernel.o"
        exports = root / "exports.map"
        compiled.export_to_c(
            str(object_file),
            function_name="run",
            enable_pic=True,
            export_only_tvm_ffi_symbols=True,
        )
        exports.write_text("{\n  global: __tvm_ffi_*;\n  local: *;\n};\n")
        subprocess.run(
            [
                shutil.which("gcc") or "gcc",
                "-shared",
                "-o",
                str(output),
                str(object_file),
                "-Wl,--whole-archive",
                str(_runtime_archive()),
                "-Wl,--no-whole-archive",
                f"-Wl,--version-script={exports}",
                "-L/usr/local/cuda/lib64",
                "-lcudart",
                "-ldl",
                "-lpthread",
            ],
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()

    import tvm_ffi  # noqa: F401

    if arguments.output.exists():
        raise SystemExit(f"refusing to overwrite {arguments.output}")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    _link(_compile(), arguments.output)
    print(arguments.output.resolve())


if __name__ == "__main__":
    main()
