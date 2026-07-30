#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export a fixed-shape FP16 residual add as a TVM-FFI DSO."""

import argparse
from importlib import metadata
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


ROWS = 256
COLS = 768
THREADS = 256
SUM_OP = "LayerType.ELEMENTWISE/ElementWiseOperation.SUM"


def _verify_contract(snapshot_path: Path, selection_path: Path) -> dict:
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("graph_fingerprint") != snapshot.get("fingerprint"):
        raise SystemExit("selection and graph snapshot fingerprints differ")
    if selection.get("workspace_bytes") != 0 or selection.get("extra_args") != []:
        raise SystemExit("this kernel requires zero workspace and no extra arguments")
    if selection.get("output_shape_input") is not None:
        raise SystemExit("this fixed-shape kernel does not accept a dynamic output")

    input_ids = selection.get("input_tensor_ids")
    output_ids = selection.get("output_tensor_ids")
    node_ids = selection.get("node_ids")
    if not isinstance(input_ids, list) or len(input_ids) != 2:
        raise SystemExit("this kernel requires exactly two inputs")
    if not isinstance(output_ids, list) or len(output_ids) != 1:
        raise SystemExit("this kernel requires exactly one output")
    if not isinstance(node_ids, list) or len(node_ids) != 1:
        raise SystemExit("this example replaces exactly one TensorRT node")

    nodes = {node["id"]: node for node in snapshot["nodes"]}
    node = nodes.get(node_ids[0])
    if node is None or node.get("op") != SUM_OP:
        raise SystemExit(f"selected node must be {SUM_OP}")
    if node.get("inputs") != input_ids or node.get("outputs") != output_ids:
        raise SystemExit("selection boundary does not match the selected node")

    tensors = {tensor["id"]: tensor for tensor in snapshot["tensors"]}
    for tensor_id in [*input_ids, *output_ids]:
        tensor = tensors.get(tensor_id)
        if tensor is None:
            raise SystemExit(f"snapshot does not contain {tensor_id}")
        if tensor.get("dtype") != "DataType.HALF":
            raise SystemExit(f"{tensor_id} must use FP16")
        if tensor.get("shape") != [ROWS, COLS]:
            raise SystemExit(f"{tensor_id} must have shape [{ROWS}, {COLS}]")
        if tensor.get("location") != "TensorLocation.DEVICE":
            raise SystemExit(f"{tensor_id} must be a device tensor")
        if tensor.get("is_shape_tensor"):
            raise SystemExit(f"{tensor_id} must not be a shape tensor")
    return selection


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
    distribution = metadata.distribution("nvidia-cutlass-dsl-libs-base")
    archive = Path(
        distribution.locate_file("nvidia_cutlass_dsl/lib/libcuda_dialect_runtime_static.a")
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
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()

    import tvm_ffi  # noqa: F401 - required by the generated ABI

    selection = _verify_contract(arguments.snapshot, arguments.selection)
    if arguments.output.exists():
        raise SystemExit(f"refusing to overwrite {arguments.output}")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    _link(_compile(), arguments.output)
    print(f"binding_id={selection['binding_id']}")
    print(f"abi_sha256={selection['abi_sha256']}")
    print(arguments.output.resolve())


if __name__ == "__main__":
    main()
