#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export the FlashInfer CuTe Qwen3-8B decode kernel as a TVM-FFI DSO."""

import argparse
from importlib import metadata
from pathlib import Path
import shutil
import subprocess
import tempfile


HEAD_DIM = 128
NUM_HEADS = 32
NUM_KV_HEADS = 8
KV_CAPACITY = 40960
PAGE_SIZE = 64
NUM_PAGES = KV_CAPACITY // PAGE_SIZE


def _compile():
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.typing import Float32, Int32
    from flashinfer.cute_dsl.attention.fusion.mask import DenseMask
    from flashinfer.cute_dsl.attention.gqa_decode_paged import (
        GroupedQueryAttentionDecodePaged,
    )

    attention = GroupedQueryAttentionDecodePaged(
        PAGE_SIZE,
        HEAD_DIM,
        grouped_head_tile=NUM_HEADS // NUM_KV_HEADS,
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
        page_offsets: cute.Tensor,
        page_table: cute.Tensor,
        context: cute.Tensor,
        sm_scale: Float32,
        stream: cuda.CUstream,
    ):
        q_batch = query.shape[0] * query.stride[0]
        o_batch = context.shape[0] * context.stride[0]
        q_layout = cute.make_layout(
            (1, 1, query.shape[0], query.shape[1]),
            stride=(q_batch, q_batch, query.stride[0], query.stride[1]),
        )
        k_layout = cute.make_layout(
            (NUM_PAGES, PAGE_SIZE, key.shape[1], key.shape[3]),
            stride=(
                PAGE_SIZE * key.stride[2],
                key.stride[2],
                key.stride[1],
                key.stride[3],
            ),
        )
        v_layout = cute.make_layout(
            (NUM_PAGES, PAGE_SIZE, value.shape[1], value.shape[3]),
            stride=(
                PAGE_SIZE * value.stride[2],
                value.stride[2],
                value.stride[1],
                value.stride[3],
            ),
        )
        o_layout = cute.make_layout(
            (1, 1, context.shape[0], context.shape[1]),
            stride=(o_batch, o_batch, context.stride[0], context.stride[1]),
        )
        attention(
            Int32(1),
            key_value_lengths,
            page_offsets,
            page_table,
            cute.make_tensor(key.iterator, k_layout),
            cute.make_tensor(value.iterator, v_layout),
            cute.make_tensor(query.iterator, q_layout),
            cute.make_tensor(context.iterator, o_layout),
            None,
            None,
            None,
            None,
            None,
            None,
            DenseMask(),
            sm_scale,
            Float32(1.0),
            None,
            stream,
            False,
        )

    query = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (NUM_HEADS, HEAD_DIM),
        stride_order=(1, 0),
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
    key_value_lengths = cute.runtime.make_fake_compact_tensor(
        Int32,
        (1,),
        assumed_align=4,
    )
    page_offsets = cute.runtime.make_fake_compact_tensor(
        Int32,
        (1,),
        assumed_align=4,
    )
    page_table = cute.runtime.make_fake_compact_tensor(
        Int32,
        (NUM_PAGES,),
        assumed_align=4,
    )
    context = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (NUM_HEADS, HEAD_DIM),
        stride_order=(1, 0),
        assumed_align=16,
    )
    return cute.compile(
        run,
        query,
        key,
        value,
        key_value_lengths,
        page_offsets,
        page_table,
        context,
        Float32(1.0 / HEAD_DIM**0.5),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi --opt-level 3",
    )


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
        exports.write_text("{\n  global: __tvm_ffi_*;\n  local: *;\n};\n")
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    import torch
    import tvm_ffi  # noqa: F401 - required by the generated ABI

    torch.cuda.set_device(args.device)
    if torch.cuda.get_device_capability(args.device) != (10, 3):
        raise SystemExit("This example kernel requires an SM 10.3 GPU")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")
    _link(_compile(), args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
