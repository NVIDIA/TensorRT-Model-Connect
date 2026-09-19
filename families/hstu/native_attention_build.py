# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-only dependency handling for the original NVIDIA HSTU attention.

This module produces a self-contained native plugin for a Model Connect bundle.
The serving process loads that library through its model-local TRT registry.
"""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys

from tensorrt_model_connect.build import content_cache_key


HERE = Path(__file__).resolve().parent


def unsupported_reason(config, precision):
    """Model semantics only: GPU target never selects a different graph."""
    if precision != "bf16":
        return "the original native adapter requires BF16"
    if (config["num_heads"], config["head_dim"]) != (4, 64):
        return "the qualified original adapter currently requires H4/D64"
    if config["mode"] != "ranking" or not config["is_causal"] or config["target_group_size"] != 1:
        return "the original adapter requires causal ranking with target_group_size=1"
    if config["scaling_seqlen"] != 1024 or config["max_sequence_length"] > 1024:
        return "the original adapter requires scaling_seqlen=1024 and capacity<=1024"
    if config["time_buckets"] or any(t["role"] == "context" for t in config["embedding_tables"]):
        return "time/context embeddings require the ordinary TensorRT attention path"
    return None


def source_directory(configured=None):
    """Resolve an explicitly supplied original source; never download at runtime."""
    if configured:
        source = Path(configured).expanduser().resolve()
        if not source.is_dir():
            raise ValueError("native_kernel_source must name an existing original source directory")
        candidates = (source, source / "fbgemm_gpu/experimental/hstu/src/hstu_ampere")
        if source.name == "hstu_blackwell":
            candidates += (source.parent / "hstu_ampere",)
        for candidate in candidates:
            if (candidate / "hstu_fwd.h").is_file():
                return candidate.resolve()
        raise ValueError("native_kernel_source must contain the pinned original hstu_ampere headers")
    # The standalone original HSTU wheel is a build dependency, not a serving
    # dependency. Find its source without running the Torch package initializer.
    for entry in sys.path:
        candidate = Path(entry) / "hstu" / "hstu_ampere"
        if (candidate / "hstu_fwd.h").is_file():
            return candidate.resolve()
    return None


def cutlass_directory(source):
    """Find the original repository's pinned CUTLASS dependency."""
    for parent in source.parents:
        candidate = parent / "external/cutlass"
        if (candidate / "include/cute/tensor.hpp").is_file():
            return candidate
    raise ValueError("The original HSTU source requires its pinned external/cutlass headers")


def cutlass_content_key(directory):
    include = directory / "include"
    files = sorted(path for path in include.rglob("*") if path.is_file())
    return content_cache_key("hstu-original-cutlass-include-v1",
                             *(path.relative_to(include).as_posix().encode() + b"\0" + path.read_bytes()
                               for path in files))


def verify_source(source):
    manifest = json.loads((HERE / "native_attention_source.json").read_text())
    domain = "hstu-original-kernel-source-v1"
    if manifest.get("content_key_domain") != domain:
        raise ValueError("Unsupported original HSTU source identity format")
    for name, expected in manifest["files"].items():
        path = source / name
        if not path.is_file() or content_cache_key(domain, name.encode(), path.read_bytes()) != expected:
            raise ValueError(f"Original HSTU source differs from pinned {manifest['revision']}: {name}")
    cutlass = cutlass_directory(source)
    if cutlass_content_key(cutlass) != manifest["cutlass"]["include_content_key"]:
        raise ValueError("Original HSTU CUTLASS headers differ from the pinned dependency")
    for path, entry in manifest["licenses"].items():
        _license_bytes(cutlass.parent.parent / path, entry["content_key"])
    return manifest


def _license_bytes(path, expected):
    if not path.is_file():
        raise ValueError(f"Original HSTU license differs from its pinned source: {path.name}")
    data = path.read_bytes()
    if content_cache_key("hstu-native-license-v1", data) != expected:
        raise ValueError(f"Original HSTU license differs from its pinned source: {path.name}")
    return data


def native_attention_notices(manifest):
    """Complete, path-free notices accompanying the compiled native provider."""
    sections = [(HERE / "third_party/NOTICE.txt").read_bytes()]
    entries = [*manifest["licenses"].values(), manifest["cccl_notice"]]
    for entry in entries:
        name = entry["notice_file"]
        license_text = _license_bytes(HERE / "third_party" / name, entry["content_key"])
        sections.extend((f"\n{'-' * 72}\n{name}\n{'-' * 72}\n\n".encode(), license_text))
    return b"".join(sections)


def compile_target(capability, supported):
    """Parameterize the same original kernel for a compiler-supported target."""
    if (len(capability) != 2 or any(type(value) is not int for value in capability)
            or capability[0] < 8 or not 0 <= capability[1] <= 9):
        raise ValueError("The original BF16 CUDA kernel requires a compute capability >= 8.0")
    target = f"sm_{capability[0]}{capability[1]}"
    if target not in supported:
        raise ValueError(f"The installed CUDA compiler does not support {target}")
    return target


def _current_capability():
    from cuda.bindings import runtime as cuda

    status, device = cuda.cudaGetDevice()
    if status != cuda.cudaError_t.cudaSuccess:
        raise RuntimeError("Could not determine the HSTU build device")
    status, properties = cuda.cudaGetDeviceProperties(device)
    if status != cuda.cudaError_t.cudaSuccess:
        raise RuntimeError("Could not determine the HSTU compiler target")
    return properties.major, properties.minor


def _checked(command, *, verbose):
    result = subprocess.run(command, cwd=HERE.parents[1], stdout=None if verbose else subprocess.PIPE,
                            stderr=None if verbose else subprocess.STDOUT, text=True)
    if result.returncode:
        raise RuntimeError(f"HSTU native dependency build failed ({result.returncode})\n"
                           f"{result.stdout or ''}")


def build_attention_library(output: Path, source: Path, *, attention_mode="paged",
                            compute_capability=None, verbose=False):
    """Return (library path, specialization manifest); caller owns output dir."""
    from . import native_attention_export

    if attention_mode not in ("paged", "dense"):
        raise ValueError("Native HSTU attention mode must be paged or dense")
    dense = attention_mode == "dense"
    original = verify_source(source)
    notices = native_attention_notices(original)
    nvcc = shutil.which("nvcc")
    compiler = shutil.which("c++")
    if not nvcc or not compiler:
        raise RuntimeError("Native HSTU export requires the C++ compiler and CUDA toolkit")
    cuda = Path(nvcc).resolve().parents[1]
    nvcc_version = subprocess.check_output([str(cuda / "bin/nvcc"), "--version"], text=True)
    compiler_version = subprocess.check_output([compiler, "--version"], text=True).splitlines()[0]
    capability = _current_capability() if compute_capability is None else compute_capability
    supported = subprocess.check_output([nvcc, "--list-gpu-code"], text=True).split()
    target = compile_target(capability, supported)
    output.mkdir(parents=True, exist_ok=True)
    (output / "attention_native.NOTICE").write_bytes(notices)
    kernel_object, kernel = native_attention_export.export_attention(
        output / "attention_export", source, target, attention_mode
    )
    verify_source(source)
    host_sources = ("native_attention_build.py", "native_attention_export.py",
                    "native_attention_plugin.cpp", "native_linear_plugin.cpp",
                    "native_attention_kernel.h", "native_attention_kernel.cu",
                    "runtime/attention_metadata.h")
    if not dense:
        host_sources += ("native_projection_barrier.cpp",)
    specialization = {
        "schema_version": 2, "adapter_abi": 2, "provider": "original_cuda_m64",
        "gpu_arch": target, "compute_capability": list(capability),
        "dtype": "bf16", "heads": 4, "head_dim": 64, "tile_m": 64, "tile_n": 128, "warps": 4,
        "attention_mode": attention_mode,
        "page_size": 0 if dense else 128, "scaling_seqlen": 1024, "max_capacity": 1024,
        "causal": True, "target_group_size": 1, "context": False,
        "uvqk": {"creator": "HstuUvqk", "input_width": 256, "output_width": 1024,
                 "input_dtype": "bf16", "compute_dtype": "fp32", "tf32": False},
        "kernel": kernel,
        "tensorrt": {"distribution": "tensorrt", "version": importlib.metadata.version("tensorrt")},
        "cuda_compiler": nvcc_version, "host_compiler": compiler_version,
        "original": original,
        "notices_content_key": content_cache_key("hstu-native-notices-v1", notices),
        "host_sources": {name: content_cache_key("hstu-native-host-source-v1", name.encode(),
                                                (HERE / name).read_bytes()) for name in host_sources},
    }
    if not dense:
        specialization["projection_barrier"] = {
            "creator": "HstuProjectionBarrier", "creator_version": "1",
            "dtype": "bf16", "width": 1024, "input_output_alias": True,
            "preview_feature": "ALIASED_PLUGIN_IO_10_03",
        }
    digest = content_cache_key("hstu-native-specialization-v1",
                               json.dumps(specialization, sort_keys=True).encode())
    namespace = "trtmc_hstu_" + digest
    export_map = output / "exports.map"
    export_map.write_text("{ global: getCreators; setLoggerFinder; local: *; };\n")
    library = output / (namespace + ".so")
    command = [compiler, "-std=c++17", "-O3", "-fPIC", "-shared", "-Wall", "-Wextra", "-Werror",
               "-isystem", str(cuda / "include"), "-Wl,--no-undefined",
               '-DHSTU_PLUGIN_NAMESPACE="' + namespace + '"',
               "-DHSTU_DENSE_ATTENTION=" + ("1" if dense else "0"),
               str(HERE / "native_attention_plugin.cpp"),
               str(HERE / "native_linear_plugin.cpp"),
               *([] if dense else [str(HERE / "native_projection_barrier.cpp")]),
               str(kernel_object),
               "-L" + str(cuda / "lib64"), "-L" + str(cuda / "lib64/stubs"),
               "-lnvinfer", "-lcudart", "-lcuda", "-lcublasLt", "-ldl", "-lpthread",
               "-Wl,--version-script," + str(export_map), "-o", str(library)]
    _checked(command, verbose=verbose)
    needed = subprocess.check_output(["readelf", "-d", str(library)], text=True).lower()
    if any(name in needed for name in ("libtorch", "libc10", "libpython", "tvm", "ffi")):
        raise RuntimeError("HSTU native serving library acquired a Python/Torch/FFI dependency")
    specialization.update(namespace=namespace,
                          creator="HstuDenseAttention" if dense else "HstuPagedAttention", creator_version="1",
                          digest=digest,
                          library_content_key=content_cache_key("hstu-native-library-v1", library.read_bytes()))
    (output / "attention_native.json").write_text(json.dumps(specialization, indent=2) + "\n")
    return library, specialization
