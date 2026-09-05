#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the contributor-visible CPU gate used by local hooks and public CI.

The host owns diff selection and Docker lifecycle. Source-only C++ and Python
units run in the same hardened, GPU-free container boundary used by premerge.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

from tools.ci.container import CiContainer
from tools.ci.context import CiContext
from tools.ci.process import CiError, CommandRunner, GitHubFiles
from tools.ci.quality import SourceQualityChecks
from tools.ci.stage import ContainerStageRunner
from tools import test_impact


class CommunityCI:
    """Coordinate the public, source-only validation contract."""

    def __init__(
        self,
        repository: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self.repository = (repository or Path.cwd()).resolve()
        self.env = dict(env or os.environ)
        self.commands = CommandRunner(cwd=self.repository, env=self.env)
        self.github = GitHubFiles(self.env)

    def source_quality(self, base: str | None) -> None:
        resolved_base = self.resolve_base(base)
        context = CiContext(
            repository=self.repository,
            env={**self.env, "CI_BASE_REF": resolved_base},
        )
        quality = SourceQualityChecks(context)
        name = "New-architecture source quality"
        print(f"::group::{name}")
        try:
            quality.run()
        except CiError as error:
            print(f"::error title={name}::{error}")
            raise
        finally:
            print("::endgroup::")

    def impact(self, base: str | None) -> dict[str, object]:
        resolved_base = self.resolve_base(base)
        try:
            test_impact.validate(self.repository)
            paths = test_impact.changed_files(self.repository, resolved_base, "HEAD")
            result = test_impact.classify(self.repository, paths)
        except (OSError, ValueError, subprocess.CalledProcessError) as error:
            raise CiError(f"Impact analysis failed: {error}") from error
        summary = {
            "scope": result.scope,
            "families": list(result.families),
            "changed_paths": list(result.changed_files),
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        self.github.output(
            "families",
            json.dumps(summary["families"], separators=(",", ":")),
        )
        self.github.summary("### Community CPU ownership and impact")
        self.github.summary(f"- Families: `{len(result.families)}`")
        self.github.summary(
            "- Changed paths: "
            + (", ".join(f"`{path}`" for path in result.changed_files) or "none")
        )
        return summary

    def unit(self) -> None:
        image = self._ensure_cpu_image()
        runner_temp = Path(self.env.get("RUNNER_TEMP", tempfile.gettempdir())).resolve()
        runner_temp.mkdir(parents=True, exist_ok=True)
        container_name = f"trtmc-community-{os.getpid()}"
        with tempfile.TemporaryDirectory(
            prefix="trtmc-community-unit-",
            dir=runner_temp,
        ) as scratch:
            container_env = {
                **self.env,
                "TRTMC_CI_WORKSPACE": str(self.repository),
                "TRTMC_CI_IMAGE": image,
                "TRTMC_CI_CONTAINER_NAME": container_name,
                "TRTMC_CI_HARDENED": "true",
                "TRTMC_CI_SCRATCH_HOST": scratch,
                "GITHUB_RUN_ID": self.env.get("GITHUB_RUN_ID", f"local-{os.getpid()}"),
                "GITHUB_RUN_ATTEMPT": self.env.get("GITHUB_RUN_ATTEMPT", "1"),
            }
            try:
                CiContainer(container_env).start()
                return_code = ContainerStageRunner("premerge-unit", container_env).run()
                if return_code:
                    raise CiError(f"Community CPU unit stage failed with exit code {return_code}")
            finally:
                self.commands.run(
                    ["docker", "rm", "-f", container_name],
                    check=False,
                    capture_output=True,
                )

    def resolve_base(self, explicit: str | None) -> str:
        configured = explicit or self.env.get("TRTMC_COMMUNITY_BASE_REF", "")
        if not configured:
            raise CiError("--base or TRTMC_COMMUNITY_BASE_REF is required")
        result = self.commands.run(
            ["git", "rev-parse", "--verify", f"{configured}^{{commit}}"],
            check=False,
            capture_output=True,
        )
        if result.returncode:
            raise CiError(f"Community CPU base does not resolve: {configured}")
        revision = result.stdout.strip()
        print(f"Community CPU base: {configured} ({revision})")
        return revision

    def _ensure_cpu_image(self) -> str:
        image = "trtmc-community-cpu:local"
        self.commands.run(
            [
                "docker",
                "build",
                "--file",
                "Dockerfile.community-cpu",
                "--tag",
                image,
                "requirements",
            ]
        )
        return image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    for name in ("source-quality", "impact"):
        command = commands.add_parser(name)
        command.add_argument("--base")

    commands.add_parser("unit", help="Run the complete source-only unit suite")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    runner = CommunityCI()
    try:
        if arguments.command == "source-quality":
            runner.source_quality(arguments.base)
        elif arguments.command == "impact":
            runner.impact(arguments.base)
        elif arguments.command == "unit":
            runner.unit()
    except CiError as error:
        print(f"ERROR: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
