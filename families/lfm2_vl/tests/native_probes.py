# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build family test clients against either installed or build-tree runtime libraries."""

import json
from pathlib import Path
import shutil
import subprocess


def build_native_probe(name, runtime, directory):
    libraries = {
        "lfm2_vl_probe": ["trtmc_model_lfm2_vl", "trtmc_c", "trtmc_core"],
        "lfm25_vl_logits_probe": ["trtmc_model_lfm2_vl", "trtmc_backend_trt", "trtmc_core"],
    }[name]
    runtime, directory = Path(runtime), Path(directory)
    for library in libraries:
        if not (runtime / f"lib{library}.so").is_file():
            raise FileNotFoundError(f"runtime library is missing: {library}")
    from torch.utils.cpp_extension import CUDA_HOME

    if CUDA_HOME is None:
        raise RuntimeError("native LFM2-VL test probes require CUDA toolkit headers")
    compiler = shutil.which("c++")
    if compiler is None:
        raise RuntimeError("native LFM2-VL test probes require a C++ compiler")
    here = Path(__file__).resolve().parent
    root = here.parents[2]
    executable = directory / name
    command = [compiler, "-std=c++17", "-O2"]
    for include in (
        root,
        root / "core/runtime/include",
        root / "core/api/include",
        Path(CUDA_HOME) / "include",
    ):
        command += ["-I", str(include)]
    command += [str(here / "cpp" / (name + ".cpp")), "-L", str(runtime)]
    command += ["-l" + library for library in libraries]
    command += [
        "-L",
        str(Path(CUDA_HOME) / "lib64"),
        "-lcudart",
        "-Wl,-rpath," + str(runtime) + ":" + str(Path(CUDA_HOME) / "lib64"),
        "-o",
        str(executable),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    (directory / (name + "-build.json")).write_text(
        json.dumps(
            {
                "argv": command,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
            indent=2,
        )
        + "\n"
    )
    result.check_returncode()
    return executable
