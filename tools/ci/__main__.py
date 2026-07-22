# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Translate the public ``python -m tools.ci`` commands into one responsible class.

Boundary: argument parsing and dispatch only; workflow behavior belongs to the called class.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .process import CiError


class CiCommand:
    """Parse a small, explicit command tree and dispatch to the matching class."""

    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser(prog="python3 -m tools.ci")
        commands = self.parser.add_subparsers(dest="command", required=True)
        image = commands.add_parser("image", help="Manage the immutable CI Docker image")
        image_commands = image.add_subparsers(dest="image_command", required=True)
        image_commands.add_parser("ensure", help="Build or verify the CI Docker image")
        container = commands.add_parser("container", help="Manage the run-owned CI container")
        container_commands = container.add_subparsers(dest="container_command", required=True)
        container_commands.add_parser("start", help="Start a clean CI container")
        container_commands.add_parser(
            "cleanup", help="Remove the exact run-owned container and secret residue"
        )
        container_run = container_commands.add_parser(
            "run", help="Run one bounded command in a clean CI container"
        )
        container_run.add_argument("container_arguments", nargs=argparse.REMAINDER)
        stage = commands.add_parser("stage", help="Run a stage inside the CI container")
        stage.add_argument("name", help="Pipeline stage name")
        pipeline = commands.add_parser("pipeline", help="Execute a named pipeline stage")
        pipeline.add_argument("name", help="Pipeline stage name")
        commands.add_parser("e2e", help="Run the parallel E2E scheduler")
        coverage = commands.add_parser("coverage", help="Run a standalone coverage wrapper")
        coverage.add_argument("language", choices=("all", "cpp", "python"))
        proof = commands.add_parser("model-proof", help="Run one hermetic model certification")
        proof.add_argument("--model", required=True)
        proof.add_argument("--suite", default="premerge")
        proof.add_argument("--revision", default="HEAD")
        proof.add_argument("--output-dir", type=Path)
        proof.add_argument("--inner", action="store_true")
        proof.add_argument("--cleanup-containers", action="store_true")

    def run(self) -> int:
        arguments, remaining = self.parser.parse_known_args()
        if (arguments.command, getattr(arguments, "image_command", None)) == ("image", "ensure"):
            from .docker_image import DockerImageManager

            DockerImageManager(Path.cwd(), dict(os.environ)).ensure()
            return 0
        if arguments.command == "container" and arguments.container_command == "start":
            from .container import CiContainer

            CiContainer(dict(os.environ)).start()
            return 0
        if arguments.command == "container" and arguments.container_command == "cleanup":
            from .container import CiContainer

            CiContainer(dict(os.environ)).cleanup()
            return 0
        if arguments.command == "container" and arguments.container_command == "run":
            from .container import CiContainer

            command = arguments.container_arguments
            if command[:1] == ["--"]:
                command = command[1:]
            if not command:
                self.parser.error("container run requires '-- <command>'")
            CiContainer(dict(os.environ)).run_once(command)
            return 0
        if arguments.command == "stage":
            from .stage import ContainerStageRunner

            return ContainerStageRunner(arguments.name, dict(os.environ)).run()
        if arguments.command == "pipeline":
            from .context import CiContext
            from .pipeline import CiPipeline

            CiPipeline(CiContext(env=dict(os.environ))).run(arguments.name)
            return 0
        if arguments.command == "e2e":
            from .context import CiContext
            from .e2e_scheduler import E2EParallelConfig, E2EParallelRunner

            config = E2EParallelConfig.parse(remaining, dict(os.environ))
            return E2EParallelRunner(CiContext(env=dict(os.environ)), config).run()
        if arguments.command == "coverage":
            from .context import CiContext
            from .coverage import CoverageRunner

            coverage_runner = CoverageRunner(CiContext(env=dict(os.environ)))
            if arguments.language == "cpp":
                coverage_runner.cpp_report(remaining)
            elif arguments.language == "python":
                coverage_runner.python_report(remaining or None)
            else:
                if remaining:
                    self.parser.error(f"unrecognized combined coverage arguments: {remaining}")
                coverage_runner.all_reports()
            return 0
        if arguments.command == "model-proof":
            from .context import CiContext
            from .model_proof import (
                ModelProofContainerCleaner,
                ModelProofRequest,
                ModelProofRunner,
            )

            request = ModelProofRequest(
                model=arguments.model,
                suite=arguments.suite,
                revision=arguments.revision,
                output_dir=arguments.output_dir,
            )
            if arguments.cleanup_containers:
                if arguments.inner:
                    self.parser.error("--cleanup-containers cannot be combined with --inner")
                ModelProofContainerCleaner(CiContext(env=dict(os.environ)), request.model).cleanup()
                return 0
            runner = ModelProofRunner(CiContext(env=dict(os.environ)), request)
            if arguments.inner:
                runner.run_inner()
            else:
                runner.run_host()
            return 0
        if remaining:
            self.parser.error(f"unrecognized arguments: {' '.join(remaining)}")
        self.parser.error("unsupported command")
        return 2


def main() -> int:
    try:
        return CiCommand().run()
    except CiError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
