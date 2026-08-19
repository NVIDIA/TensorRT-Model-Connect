#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export a FlashInfer linear-decode DSO for Qwen's attention Recipe."""

import argparse
from importlib import metadata
import inspect
from pathlib import Path
import shutil
import subprocess
import tempfile


HEAD_DIM = 128
NUM_QUERY_HEADS = 32
NUM_KV_HEADS = 8
KV_CAPACITY = 40960


def _runtime_archive() -> Path:
    distribution = metadata.distribution("nvidia-cutlass-dsl-libs-base")
    archive = Path(
        distribution.locate_file("nvidia_cutlass_dsl/lib/libcuda_dialect_runtime_static.a")
    ).resolve()
    if not archive.is_file():
        raise SystemExit(f"CUTLASS DSL runtime archive not found: {archive}")
    return archive


def _link(compiled, output: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="trtmc-byok-") as temporary:
        root = Path(temporary)
        obj = root / "kernel.o"
        exports = root / "exports.map"
        compiled.export_to_c(
            str(obj),
            function_name="run",
            enable_pic=True,
            export_only_tvm_ffi_symbols=True,
        )
        exports.write_text(
            "{\n  global: __tvm_ffi_*;\n  local: *;\n};\n",
            encoding="utf-8",
        )
        subprocess.run(
            [
                shutil.which("gcc") or "gcc",
                "-shared",
                "-o",
                str(output),
                str(obj),
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


def _compile():
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.typing import Float32, Int32
    from flashinfer.cute_dsl.attention.fusion.mask import DenseMask
    from flashinfer.cute_dsl.attention.gqa_decode import (
        GroupedQueryAttentionDecode,
    )

    if "kv_lens" not in inspect.signature(GroupedQueryAttentionDecode.__call__).parameters:
        raise SystemExit(
            "FlashInfer is missing the optional device KV-length API; "
            "apply flashinfer_device_kv_length.patch first"
        )

    attention = GroupedQueryAttentionDecode(
        HEAD_DIM,
        grouped_head_tile=NUM_QUERY_HEADS // NUM_KV_HEADS,
        prediction_tile=1,
        sequence_tile=256,
        reduction_mode="none",
    )

    @cute.jit
    def run(
        query: cute.Tensor,
        key: cute.Tensor,
        value: cute.Tensor,
        key_value_lengths: cute.Tensor,
        context: cute.Tensor,
        stream: cuda.CUstream,
    ):
        query_layout = cute.make_layout(
            (query.shape[0], query.shape[2], query.shape[1], query.shape[3]),
            stride=(
                query.stride[0],
                query.stride[2],
                query.stride[1],
                query.stride[3],
            ),
        )
        key_value_layout = cute.make_layout(
            (key.shape[0], key.shape[2], key.shape[1], key.shape[3]),
            stride=(
                key.stride[0],
                key.stride[2],
                key.stride[1],
                key.stride[3],
            ),
        )
        context_layout = cute.make_layout(
            (
                context.shape[0],
                context.shape[2],
                context.shape[1],
                context.shape[3],
            ),
            stride=(
                context.stride[0],
                context.stride[2],
                context.stride[1],
                context.stride[3],
            ),
        )
        attention(
            Int32(1),
            cute.make_tensor(query.iterator, query_layout),
            cute.make_tensor(key.iterator, key_value_layout),
            cute.make_tensor(value.iterator, key_value_layout),
            cute.make_tensor(context.iterator, context_layout),
            None,
            None,
            None,
            None,
            None,
            None,
            DenseMask(),
            Float32(1.0),
            Float32(1.0),
            stream,
            False,
            key_value_lengths,
        )

    query = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (1, NUM_QUERY_HEADS, 1, HEAD_DIM),
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )
    key = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (1, NUM_KV_HEADS, KV_CAPACITY, HEAD_DIM),
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )
    value = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (1, NUM_KV_HEADS, KV_CAPACITY, HEAD_DIM),
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )
    lengths = cute.runtime.make_fake_compact_tensor(Int32, (1,), assumed_align=4)
    context = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (1, NUM_QUERY_HEADS, 1, HEAD_DIM),
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )
    return cute.compile(
        run,
        query,
        key,
        value,
        lengths,
        context,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi --opt-level 3",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    import torch
    import tvm_ffi  # noqa: F401 - required by the generated ABI

    torch.cuda.set_device(args.device)
    if torch.cuda.get_device_capability(args.device) != (10, 3):
        raise SystemExit("This POC kernel requires an SM 10.3 GPU")
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    _link(_compile(), args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
