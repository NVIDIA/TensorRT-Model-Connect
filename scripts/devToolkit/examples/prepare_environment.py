# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create a Docker target, resolve its toolchain, build TRTMC, and run it."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path, PurePosixPath


REPOSITORY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY / "scripts" / "devToolkit"))

from trtmc_devtoolkit import (  # noqa: E402
    DevToolkit,
    DockerGpuRequest,
    DockerImageRef,
    DockerMount,
    DockerTarget,
    EnvironmentRequest,
    DockerTargetPolicy,
    TrtmcBuildRecipe,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--container", default="trtmc-devtoolkit")
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--gpu",
        action="append",
        default=[],
        help="Docker GPU device ID; repeat to select several (default: all)",
    )
    parser.add_argument("--tensorrt", required=True)
    parser.add_argument("--architecture", choices=("x86_64", "aarch64"))
    parser.add_argument(
        "--workspace",
        default="/workspace/TensorRT-Model-Connect",
        help="Path to this checkout inside the running container",
    )
    arguments = parser.parse_args()

    toolkit = DevToolkit.from_checkout(REPOSITORY)
    target = toolkit.targets.ensure(
        DockerTarget(
            name=arguments.container,
            image=DockerImageRef(arguments.image),
            gpus=(
                DockerGpuRequest.devices(*arguments.gpu)
                if arguments.gpu
                else DockerGpuRequest.all()
            ),
            mounts=(DockerMount(REPOSITORY.resolve(), PurePosixPath(arguments.workspace)),),
            workspace=PurePosixPath(arguments.workspace),
        ),
        policy=DockerTargetPolicy.ENSURE,
    )
    lock = toolkit.resolve(
        EnvironmentRequest(
            tensorrt=arguments.tensorrt,
            architecture=arguments.architecture,
            target=target.execution_target,
        )
    )
    environment = toolkit.provision(lock)
    build = toolkit.build(
        environment,
        TrtmcBuildRecipe(
            targets=("trtmc", "trtmc_backend_trt"),
            outputs={"trtmc": "trtmc"},
        ),
    )

    result = toolkit.run_trtmc(
        environment,
        ("version",),
        build=build,
        capture_output=True,
    )
    print(result.stdout, end="")
    print(f"environment receipt: {environment.receipt}")
    print(f"build receipt: {build.receipt}")


if __name__ == "__main__":
    main()
