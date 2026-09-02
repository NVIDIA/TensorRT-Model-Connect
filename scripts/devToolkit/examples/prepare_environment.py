# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve, adopt, build, and use an existing campaign container."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY / "scripts" / "devToolkit"))

from trtmc_devtoolkit import (  # noqa: E402
    BuildSpec,
    CommandSpec,
    DevToolkit,
    EnvironmentRequest,
    ExecutionTarget,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--container", required=True)
    parser.add_argument("--tensorrt", required=True)
    parser.add_argument("--architecture", choices=("x86_64", "aarch64"))
    parser.add_argument(
        "--workspace",
        default="/workspace/TensorRT-Model-Connect",
        help="Path to this checkout inside the running container",
    )
    arguments = parser.parse_args()

    toolkit = DevToolkit.from_checkout(REPOSITORY)
    lock = toolkit.resolve(
        EnvironmentRequest(
            tensorrt=arguments.tensorrt,
            architecture=arguments.architecture,
            target=ExecutionTarget.docker(
                container=arguments.container,
                workspace=arguments.workspace,
            ),
        )
    )
    environment = toolkit.provision(lock)
    build = toolkit.build(
        environment,
        BuildSpec(
            targets=("trtmc", "trtmc_backend_trt"),
            outputs={"trtmc": "trtmc"},
        ),
    )

    result = toolkit.run(
        environment,
        CommandSpec((build.artifacts[0].path, "version")),
        capture_output=True,
    )
    print(result.stdout, end="")
    print(f"environment receipt: {environment.receipt}")
    print(f"build receipt: {build.receipt}")


if __name__ == "__main__":
    main()
