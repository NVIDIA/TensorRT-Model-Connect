# SPDX-License-Identifier: Apache-2.0
"""Build a public plugin against the supplied 26.10 prototype SDK ABI.

Run INSIDE tritonserver:26.10-py3-trtmc-0.2.0. Its release-coupled C++
DType enum predates main's FP8 addition: INT64=5 rather than 6. Only the
plugin/test bridge are recompiled with that ABI; all image SDK/backend DSOs
are copied byte for byte. Never load this plugin with a main-built SDK.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--json-include", type=Path, required=True)
    a = p.parse_args()
    backend = Path("/opt/tritonserver/backends/tensorrt/libtriton_tensorrt.so")
    expected_backend = "8e9d55aaf5b359049f3ea8745d032d10642aac1c9587ace4b629856756ffdca9"
    sdk = Path("/usr/lib/x86_64-linux-gnu")
    expected_sdk = {
        "libtrtmc_core.so": "25cddc8d8b03c609fa524ba01fed394242e5e2d97af04f826f2d07d5a52a7a38",
        "libtrtmc_runtime.so": "44d50ef2c52763802ee0c318a3e464b5cf9125bb51d7a8638f0ed9d1e25a937c",
        "libtrtmc_backend_trt.so": "2122be6b4265702dbe13c6a2d7203fa83c7c731cd76fa5e81dc5f76a92c4ec54",
        "libtrtmc_c.so.1": "87f54868997ce89690ab8fedbc127848e0b1ed0b4ff07422dceff0a827396017",
    }
    if hashlib.sha256(backend.read_bytes()).hexdigest() != expected_backend or any(
        hashlib.sha256((sdk / name).read_bytes()).hexdigest() != digest
        for name, digest in expected_sdk.items()
    ):
        raise RuntimeError("SDK compatibility helper requires the exact tested 26.10/0.2.0 image")
    out = a.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("deployment output must be empty")
    out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[3]
    header = root / "core/runtime/include/trtmc/runtime/tensor.h"
    text = header.read_text()
    expected = "    kFloat8E4M3,\n    kInt64,"
    if expected not in text:
        raise RuntimeError("review SDK compatibility after tensor enum changes")
    compat = out / "sdk-compat/trtmc/runtime/tensor.h"
    compat.parent.mkdir(parents=True)
    compat.write_text(text.replace(expected, "    kInt64,\n    kFloat8E4M3,"))
    sdk = Path("/usr/lib/x86_64-linux-gnu")
    receipts = {}
    for stem in ("core", "runtime", "backend_trt", "c"):
        for source in sdk.glob("libtrtmc_" + stem + ".so*"):
            target = out / source.name
            shutil.copyfile(source, target)
            receipts[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
    # Retain the image's other families when this directory becomes the loader root.
    for source in sdk.glob("libtrtmc_model_*.so*"):
        shutil.copyfile(source, out / source.name)
        receipts[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
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
                    "pipeline.cpp",
                    "staged_pipeline.cpp",
                    "history_snapshot.cpp",
                    "plugin.cpp",
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
                stock_sdk_sha256=receipts,
                backend_sha256=hashlib.sha256(backend.read_bytes()).hexdigest(),
                compat_header_sha256=hashlib.sha256(compat.read_bytes()).hexdigest(),
            ),
            indent=2,
        )
        + "\n"
    )
    print(out)


if __name__ == "__main__":
    main()
