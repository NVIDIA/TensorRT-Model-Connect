# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configure and build the model-owned A100 qualification runners."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


_CACHE_VARIABLES = (
    "TRTMC_EDGE_LLM_SOURCE_DIR",
    "TRTMC_EDGE_LLM_BUILD_DIR",
    "TRTMC_EDGE_LLM_PLUGIN_LIBRARY",
    "TRTMC_TENSORRT_INCLUDE_DIR",
    "TRTMC_TENSORRT_LIBRARY",
    "TRTMC_TENSORRT_VERSION",
    "TRTMC_CUDA_INCLUDE_DIR",
    "TRTMC_CUDART_LIBRARY",
    "TRTMC_CUDA_DRIVER_LIBRARY",
    "TRTMC_CUDA_VERSION",
    "TRTMC_NLOHMANN_JSON_INCLUDE_DIR",
    "TRTMC_MC_INCLUDE_DIR",
    "TRTMC_MC_CORE_LIBRARY",
)


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--parallel", type=int, default=4)
    args = parser.parse_args()

    missing = [name for name in _CACHE_VARIABLES if not os.environ.get(name, "").strip()]
    if missing:
        parser.error("missing environment variables: " + ", ".join(missing))
    source = Path(__file__).resolve().parent
    build = args.build_dir.expanduser().resolve()
    configure = [
        "cmake",
        "-S",
        str(source),
        "-B",
        str(build),
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    configure.extend(f"-D{name}={os.environ[name]}" for name in _CACHE_VARIABLES)
    for environment_name, cmake_name in (
        ("CC", "CMAKE_C_COMPILER"),
        ("CXX", "CMAKE_CXX_COMPILER"),
        ("CUDAHOSTCXX", "CMAKE_CUDA_HOST_COMPILER"),
    ):
        if os.environ.get(environment_name, "").strip():
            configure.append(f"-D{cmake_name}={os.environ[environment_name]}")
    if os.environ.get("TRTMC_CUDA_COMPILER", "").strip():
        configure.append(f"-DCMAKE_CUDA_COMPILER={os.environ['TRTMC_CUDA_COMPILER']}")
    _run(configure)
    _run(
        [
            "cmake",
            "--build",
            str(build),
            "--parallel",
            str(args.parallel),
            "--target",
            "trtmc_edgellm_direct_runner",
            "trtmc_edgellm_mc_runner",
        ]
    )
    print(
        json.dumps(
            {
                "TRTMC_EDGELLM_DIRECT_RUNNER": str(build / "trtmc_edgellm_direct_runner"),
                "TRTMC_EDGELLM_MC_RUNNER": str(build / "trtmc_edgellm_mc_runner"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
