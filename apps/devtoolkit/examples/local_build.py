# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build and run TRTMC directly in a local host environment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY / "apps/devtoolkit"))

from trtmc_devtoolkit import (  # noqa: E402
    DevToolkit,
    EnvironmentRequest,
    TrtmcBuildRecipe,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the TRTMC CLI and TensorRT backend directly on the host."
    )
    parser.add_argument(
        "--python",
        required=True,
        help="Existing Python 3.12 interpreter adopted by the local execution target.",
    )
    parser.add_argument(
        "--tensorrt",
        required=True,
        help="Exact four-part TensorRT version to adopt or provision.",
    )
    parser.add_argument("--gpu", default="0", help="Local GPU ordinal used by build probes.")
    parser.add_argument(
        "--cmake-python",
        help="Interpreter or wrapper containing Torch for the current top-level CMake configure.",
    )
    parser.add_argument(
        "--cmake-prefix-path",
        action="append",
        type=Path,
        default=[],
        help="Additional CMake package prefix; repeat this option for multiple prefixes.",
    )
    parser.add_argument(
        "--state-root",
        type=Path,
        default=Path(".devtoolkit/examples/local-build"),
        help="Receipt, managed toolchain, and build directory; relative paths are checkout-relative.",
    )
    parser.add_argument("--jobs", type=int, default=None, help="Parallel build job count.")
    return parser


def _checkout_path(path: Path) -> Path:
    expanded = path.expanduser()
    return (expanded if expanded.is_absolute() else REPOSITORY / expanded).resolve()


def main(arguments: list[str] | None = None) -> int:
    args = _parser().parse_args(arguments)
    state_root = _checkout_path(args.state_root)
    toolkit = DevToolkit.from_checkout(REPOSITORY, state_root=state_root)
    prepared = toolkit.prepare_local(python=args.python, family=None)
    lock = toolkit.resolve(
        EnvironmentRequest(
            tensorrt=args.tensorrt,
            target=prepared.execution_target(gpu=args.gpu),
        )
    )
    environment = toolkit.provision(lock)

    cmake_defines: dict[str, str] = {
        "Python3_EXECUTABLE": args.cmake_python or prepared.python,
    }
    if args.cmake_prefix_path:
        cmake_defines["CMAKE_PREFIX_PATH"] = ";".join(
            str(_checkout_path(path)) for path in args.cmake_prefix_path
        )
    build = toolkit.build(
        environment,
        TrtmcBuildRecipe(
            targets=("trtmc", "trtmc_backend_trt"),
            cmake_defines=cmake_defines,
            jobs=args.jobs,
            outputs={"trtmc": "trtmc"},
        ),
    )
    result = toolkit.run_trtmc(environment, ("version",), build=build, capture_output=True)
    print(
        json.dumps(
            {
                "artifact_sha256": build.artifact("trtmc").sha256,
                "build_receipt": str(build.receipt),
                "cuda": environment.observation.cuda_version,
                "run_receipt": str(result.receipt),
                "source_revision": build.source.revision,
                "tensorrt": environment.observation.tensorrt_python_version,
                "toolchain_origin": lock.toolchain.origin,
                "trtmc": result.stdout.strip(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
