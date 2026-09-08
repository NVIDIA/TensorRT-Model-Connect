# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build and run TRTMC in a checkout-owned Docker environment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path, PurePosixPath


REPOSITORY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY / "apps/devtoolkit"))

from trtmc_devtoolkit import (  # noqa: E402
    DevToolkit,
    DockerMount,
    DockerTargetPolicy,
    EnvironmentRequest,
    TrtmcBuildRecipe,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the TRTMC CLI and TensorRT backend in a development container."
    )
    parser.add_argument("--gpu", default="0", help="Host GPU identifier passed to Docker.")
    parser.add_argument(
        "--tensorrt",
        default="11.1.0.106",
        help="Exact TensorRT version expected in the development image.",
    )
    parser.add_argument("--image", default="trtmc-dev:build-example")
    parser.add_argument("--container", default="trtmc-devtoolkit-build-example")
    parser.add_argument(
        "--state-root",
        type=Path,
        default=Path(".devtoolkit/examples/docker-build"),
        help="Receipt and build state directory; relative paths are checkout-relative.",
    )
    parser.add_argument("--jobs", type=int, default=None, help="Parallel build job count.")
    return parser


def _checkout_path(path: Path) -> Path:
    expanded = path.expanduser()
    return (expanded if expanded.is_absolute() else REPOSITORY / expanded).resolve()


def main(arguments: list[str] | None = None) -> int:
    args = _parser().parse_args(arguments)
    state_root = _checkout_path(args.state_root)
    state_root.mkdir(parents=True, exist_ok=True)

    mounts: tuple[DockerMount, ...] = ()
    target_state = str(state_root)
    if not state_root.is_relative_to(REPOSITORY):
        target_state = "/trtmc-devtoolkit-state"
        mounts = (DockerMount(state_root, PurePosixPath(target_state)),)

    toolkit = DevToolkit.from_checkout(REPOSITORY, state_root=state_root)
    prepared = toolkit.prepare_docker(
        family=None,
        gpu=args.gpu,
        image=args.image,
        container=args.container,
        policy=DockerTargetPolicy.ENSURE,
        mounts=mounts,
        ipc="host",
    )
    lock = toolkit.resolve(
        EnvironmentRequest(
            tensorrt=args.tensorrt,
            target=prepared.execution_target(state=target_state),
        )
    )
    environment = toolkit.provision(lock)
    build = toolkit.build(
        environment,
        TrtmcBuildRecipe(
            targets=("trtmc", "trtmc_backend_trt"),
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
                "container_id": prepared.container_id,
                "cuda": environment.observation.cuda_version,
                "run_receipt": str(result.receipt),
                "source_revision": build.source.revision,
                "tensorrt": environment.observation.tensorrt_python_version,
                "trtmc": result.stdout.strip(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
