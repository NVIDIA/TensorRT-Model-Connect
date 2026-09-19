# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile a native host launch of the pinned, unchanged original CUDA kernel."""

import importlib.metadata
import json
from pathlib import Path
import shlex
import shutil
import subprocess

from tensorrt_model_connect.build import content_cache_key


HERE = Path(__file__).resolve().parent


def export_attention(output: Path, source: Path, gpu_arch: str, attention_mode="paged"):
    from .native_attention_build import compile_target, cutlass_directory, verify_source

    if attention_mode not in ("paged", "dense"):
        raise ValueError("HSTU attention mode must be paged or dense")
    verify_source(source)
    nvcc = shutil.which("nvcc")
    if not nvcc:
        raise RuntimeError("The original CUDA attention export requires NVCC")
    codes = subprocess.check_output([nvcc, "--list-gpu-code"], text=True).split()
    if not gpu_arch.startswith("sm_") or not gpu_arch[3:].isdigit():
        raise ValueError("Expected a native CUDA compiler target such as sm_80")
    capability = int(gpu_arch[3:-1]), int(gpu_arch[-1])
    if compile_target(capability, codes) != gpu_arch:
        raise ValueError("Invalid original CUDA attention target")
    cutlass = cutlass_directory(source)
    # The original headers contain ATen host declarations. Locate build headers
    # without importing Torch; no ATen operation or Torch runtime is linked.
    torch = importlib.metadata.distribution("torch")
    torch_include = Path(torch.locate_file("torch/include"))
    if not (torch_include / "ATen/ATen.h").is_file():
        raise RuntimeError("Building the original HSTU headers requires the Torch development headers")
    output.mkdir(parents=True, exist_ok=False)
    obj = output / "hstu_attention.o"
    depfile = output / "hstu_attention.d"
    command = [nvcc, "-O3", "-std=c++20", "--expt-relaxed-constexpr",
               "--expt-extended-lambda", "--use_fast_math", "-DNDEBUG", "-c",
               "-Xcompiler=-fPIC,-fvisibility=hidden", "-MD", "-MF", str(depfile),
               "-DHSTU_DENSE_ATTENTION=" + ("1" if attention_mode == "dense" else "0"),
               "-I" + str(source), "-I" + str(cutlass / "include"),
               "-I" + str(torch_include), "-I" + str(torch_include / "torch/csrc/api/include"),
               "-gencode", "arch=compute_" + gpu_arch[3:] + ",code=" + gpu_arch,
               str(HERE / "native_attention_kernel.cu"), "-o", str(obj)]
    (output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (output / "compile.log").write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"Original HSTU CUDA export failed ({result.returncode})\n{result.stdout}")
    # Bind every actual compiler-consumed header, including CUDA/CUTLASS/ATen.
    # Absolute paths are useful in local build receipts, but must not enter the
    # manifest packaged with the model. Preserve duplicate content identities so
    # the path-free record still accounts for every distinct dependency file.
    dependencies = shlex.split(depfile.read_text().replace("\\\n", "").split(":", 1)[1])
    inputs = {str(Path(path).resolve()): content_cache_key("hstu-native-compile-input-v1",
                                                         Path(path).read_bytes())
              for path in sorted(set(dependencies))}
    receipt = {
        "provider": "original_cuda_m64", "attention_mode": attention_mode,
        "gpu_arch": gpu_arch, "tile_m": 64, "tile_n": 128, "warps": 4,
        "torch_build_headers_version": torch.version,
        "compile_input_content_keys": sorted(inputs.values()),
        "object_content_key": content_cache_key("hstu-native-kernel-object-v1", obj.read_bytes()),
        "source_manifest_content_key": content_cache_key(
            "hstu-native-source-manifest-v1", (HERE / "native_attention_source.json").read_bytes()
        ),
    }
    verify_source(source)
    (output / "compile.json").write_text(
        json.dumps({**receipt, "compile_inputs": inputs}, indent=2) + "\n"
    )
    return obj, receipt


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("gpu_arch")
    parser.add_argument("--attention-mode", choices=("paged", "dense"), default="paged")
    args = parser.parse_args()
    export_attention(args.output, args.source, args.gpu_arch, args.attention_mode)
