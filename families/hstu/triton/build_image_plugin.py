# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a public plugin against the supplied 26.10 prototype SDK ABI.

Run INSIDE tritonserver:26.10-py3-trtmc-0.2.0. Its release-coupled C++
DType enum predates main's FP8 addition: INT64=5 rather than 6. Only the
plugin/test bridge are recompiled with that ABI; all image SDK/backend DSOs
are copied byte for byte. Never load this plugin with a main-built SDK.
"""

import argparse
import json
from pathlib import Path
import shutil
import subprocess


def prototype_headers(root, out):
    header = root / "core/runtime/include/trtmc/runtime/tensor.h"
    text = header.read_text()
    expected = "    kFloat8E4M3,\n    kInt64,"
    if expected not in text:
        raise RuntimeError("review SDK compatibility after tensor enum changes")
    compat = out / "sdk-compat/trtmc/runtime/tensor.h"
    compat.parent.mkdir(parents=True)
    compat.write_text(text.replace(expected, "    kInt64,\n    kFloat8E4M3,"))
    factory = root / "core/runtime/include/trtmc/runtime/family_factory.h"
    compat_factory = out / "sdk-compat/trtmc/runtime/family_factory.h"
    marker = "    std::uint64_t kv_cache_size_bytes{0};"
    if factory.read_text().count(marker) != 1:
        raise RuntimeError("review prototype FactoryContext compatibility")
    compat_factory.write_text(
        factory.read_text().replace(marker, marker + "\n    double kv_cache_memory_ratio{0.0};")
    )
    loader = root / "core/runtime/include/trtmc/runtime/family_loader.h"
    compat_loader = out / "sdk-compat/trtmc/runtime/family_loader.h"
    marker = "bool cuda_graphs = false);"
    if loader.read_text().count(marker) != 2:
        raise RuntimeError("review prototype loader function ABI")
    compat_loader.write_text(
        loader.read_text().replace(
            marker, "bool cuda_graphs = false, double kv_cache_memory_ratio = 0.0);"
        )
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--json-include", type=Path, required=True)
    a = p.parse_args()
    backend = Path("/opt/tritonserver/backends/tensorrt/libtriton_tensorrt.so")
    sdk = Path("/usr/lib/x86_64-linux-gnu")
    if not backend.is_file() or not (sdk / "libtrtmc_runtime.so").is_file():
        raise RuntimeError("run inside the supplied Triton 26.10/TRT-MC 0.2.0 image")
    out = a.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("deployment output must be empty")
    out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[3]
    prototype_headers(root, out)
    sdk = Path("/usr/lib/x86_64-linux-gnu")
    receipts = {}
    for stem in ("core", "runtime", "backend_trt", "c"):
        for source in sdk.glob("libtrtmc_" + stem + ".so*"):
            target = out / source.name
            shutil.copyfile(source, target)
            receipts[source.name] = source.stat().st_size
    # Retain the image's other families when this directory becomes the loader root.
    for source in sdk.glob("libtrtmc_model_*.so*"):
        shutil.copyfile(source, out / source.name)
        receipts[source.name] = source.stat().st_size
    include = [
        out / "sdk-compat",
        root,
        root / "core/runtime/include",
        a.json_include,
        Path("/usr/local/cuda/include"),
    ]
    common = [
        "g++",
        "-std=c++17",
        "-DHSTU_TRITON_026_ABI",
        "-shared",
        "-fPIC",
        "-O2",
        "-Wall",
        "-Wextra",
        *[f"-I{x}" for x in include],
        f"-L{out}",
        "-L/usr/local/cuda/lib64",
        "-Wl,-rpath,$ORIGIN",
        "-Wl,--no-undefined",
    ]
    family = root / "families/hstu"
    subprocess.run(
        [
            *common,
            *[
                str(family / "runtime" / name)
                for name in (
                    "validation.cpp",
                    "staged_pipeline.cpp",
                    "history_snapshot.cpp",
                    "plugin.cpp",
                    "sdk.cpp",
                )
            ],
            "-ltrtmc_core",
            "-lcudart",
            "-o",
            str(out / "libtrtmc_model_hstu.so"),
        ],
        check=True,
    )
    subprocess.run(
        [
            *common,
            str(family / "triton/bridge.cpp"),
            "-ltrtmc_runtime",
            "-ltrtmc_core",
            "-lcudart",
            "-o",
            str(out / "libhstu_triton.so"),
        ],
        check=True,
    )
    backend = Path("/opt/tritonserver/backends/tensorrt/libtriton_tensorrt.so")
    (out / "deployment.json").write_text(
        json.dumps(
            dict(
                mode="26.10 prototype ABI; INT64 enum ordinal 5; main uses ordinal 6",
                stock_sdk_bytes=receipts,
                backend=str(backend),
                family_context="prototype 0.2.0 four-field cache ABI",
            ),
            indent=2,
        )
        + "\n"
    )
    print(out)


if __name__ == "__main__":
    main()
