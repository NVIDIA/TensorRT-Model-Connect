# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal command line for the active CI stages."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .process import CiError


STAGES = (
    "impact",
    "family-coverage",
    "complexity",
    "lint",
    "source-quality",
    "premerge-unit",
    "package",
    "setup",
    "selective-e2e",
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="python3 -m tools.ci")
    commands = result.add_subparsers(dest="command", required=True)
    image = commands.add_parser("image")
    image_commands = image.add_subparsers(dest="image_command", required=True)
    image_commands.add_parser("ensure")
    container = commands.add_parser("container")
    container_commands = container.add_subparsers(dest="container_command", required=True)
    container_commands.add_parser("start")
    stage = commands.add_parser("stage")
    stage.add_argument("name", choices=STAGES)
    pipeline = commands.add_parser("pipeline")
    pipeline.add_argument("name", choices=STAGES)
    return result


def main() -> int:
    arguments = parser().parse_args()
    try:
        if arguments.command == "image":
            from .docker_image import DockerImageManager

            manager = DockerImageManager(Path.cwd(), dict(os.environ))
            manager.ensure()
            return 0
        if arguments.command == "container":
            from .container import CiContainer

            CiContainer(dict(os.environ)).start()
            return 0
        if arguments.command == "stage":
            from .stage import ContainerStageRunner

            return ContainerStageRunner(arguments.name, dict(os.environ)).run()
        from .context import CiContext
        from .pipeline import CiPipeline

        CiPipeline(CiContext(env=dict(os.environ))).run(arguments.name)
        return 0
    except CiError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
