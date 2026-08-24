#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the contributor-visible CPU gate used by local hooks and public CI.

The host owns diff selection and Docker lifecycle. Source-only C++ and Python
units run in the same hardened, GPU-free container boundary used by premerge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

from tools.ci.container import CiContainer
from tools.ci.context import CiContext
from tools.ci.process import CiError, CommandRunner, GitHubFiles
from tools.ci.quality import SourceQualityChecks
from tools.ci.stage import ContainerStageRunner
from tools.model_ci import ModelCIError, calculate_impact, discover_catalog


UNIT_SCOPES = ("none", "builder", "cli", "all")


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
        failures = self._collect(
            (
                ("DCO sign-offs", quality.dco),
                ("Diff hygiene", quality.diff_hygiene),
                ("Structured file syntax", quality.structured_files),
                ("Legal source headers", quality.legal_headers),
                ("Family coverage", quality.family_coverage),
                ("Cyclomatic complexity", quality.complexity),
                ("Changed-file lint and formatting", quality.lint_changed_files),
                ("Model architecture contracts", quality.architecture_contracts),
            )
        )
        self._raise_failures("Source quality", failures)

    def impact(self, base: str | None) -> dict[str, object]:
        resolved_base = self.resolve_base(base)
        discover_catalog(self.repository, "HEAD")
        result = calculate_impact(
            self.repository,
            resolved_base,
            "HEAD",
            platform_change_policy="all",
        )
        scope = str(result["unit_scope"])
        if scope not in UNIT_SCOPES:
            raise CiError(f"Impact analysis returned an invalid unit scope: {scope}")

        paths = sorted(
            {
                str(classification["path"])
                for change in result.get("changes", [])
                for classification in change.get("classifications", [])
            }
        )
        kinds = sorted(
            {
                str(classification["kind"])
                for change in result.get("changes", [])
                for classification in change.get("classifications", [])
            }
        )
        summary = {
            "mode": result["mode"],
            "run_unit_tests": bool(result["run_unit_tests"]),
            "unit_scope": scope,
            "changed_paths": paths,
            "classifications": kinds,
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        self.github.output("run_unit_tests", str(summary["run_unit_tests"]).lower())
        self.github.output("unit_scope", scope)
        self.github.summary("### Community CPU ownership and impact")
        self.github.summary(f"- Unit scope: `{scope}`")
        self.github.summary(
            "- Changed paths: " + (", ".join(f"`{path}`" for path in paths) or "none")
        )
        return result

    def unit(self, scope: str) -> None:
        if scope not in UNIT_SCOPES:
            raise CiError(f"Unit scope must be one of: {', '.join(UNIT_SCOPES)}")
        if scope == "none":
            print("No source-only C++ or Python unit tests are required for this change.")
            return

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
                "TRTMC_PREMERGE_UNIT_SCOPE": (
                    "community-all" if scope == "all" else scope
                ),
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
        candidates = [configured] if configured else []
        candidates.extend(("upstream/main", "github/main", "origin/main"))

        seen: set[str] = set()
        for candidate in candidates:
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            result = self.commands.run(
                ["git", "rev-parse", "--verify", f"{candidate}^{{commit}}"],
                check=False,
                capture_output=True,
            )
            if result.returncode == 0:
                revision = result.stdout.strip()
                print(f"Community CPU base: {candidate} ({revision})")
                return revision
        raise CiError(
            "Could not resolve the contribution base. Fetch upstream/main or set "
            "TRTMC_COMMUNITY_BASE_REF to the target branch revision."
        )

    def _ensure_cpu_image(self) -> str:
        inputs = (
            self.repository / "Dockerfile.community-cpu",
            self.repository / "requirements" / "community-ci.txt",
        )
        digest = hashlib.sha256()
        digest.update(b"trtmc-community-cpu-v1\0")
        for path in inputs:
            if not path.is_file():
                raise CiError(f"Community CPU image input is missing: {path}")
            digest.update(path.name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
        image = f"trtmc-community-cpu:{digest.hexdigest()[:12]}"
        inspect = self.commands.run(
            ["docker", "image", "inspect", image],
            check=False,
            capture_output=True,
        )
        if inspect.returncode:
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
        else:
            print(f"Reusing Community CPU image: {image}")
        return image

    @staticmethod
    def _collect(checks: Sequence[tuple[str, Callable[[], None]]]) -> list[str]:
        failures = []
        for name, operation in checks:
            print(f"::group::{name}")
            try:
                operation()
            except CiError as error:
                print(f"::error title={name}::{error}")
                failures.append(f"{name}: {error}")
            finally:
                print("::endgroup::")
        return failures

    @staticmethod
    def _raise_failures(gate: str, failures: Sequence[str]) -> None:
        if failures:
            details = "\n".join(f"- {failure}" for failure in failures)
            raise CiError(f"{gate} failed:\n{details}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    for name in ("source-quality", "impact"):
        command = commands.add_parser(name)
        command.add_argument("--base")

    unit = commands.add_parser("unit", help="Run one source-only unit scope")
    unit.add_argument("--scope", choices=UNIT_SCOPES, required=True)
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
            runner.unit(arguments.scope)
    except (CiError, ModelCIError) as error:
        print(f"ERROR: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
