# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""New-architecture source checks, physical impact, and CPU core units."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from tools import test_impact

from .context import CiContext
from .process import CiError


class EnvironmentVerifier:
    def __init__(self, context: CiContext):
        self.context = context

    def verify(self) -> None:
        self.context.run(["python", "--version"])
        self.context.run(["cmake", "--version"])


class ImpactAnalyzer:
    def __init__(self, context: CiContext):
        self.context = context

    def run(self) -> dict[str, object]:
        base = self.context.env.get("CI_BASE_REF", "")
        if not base:
            raise CiError("CI_BASE_REF is required")
        try:
            test_impact.validate(self.context.repository)
            changed = test_impact.changed_files(self.context.repository, base, "HEAD")
            selected = test_impact.classify(self.context.repository, changed)
        except (OSError, ValueError) as error:
            raise CiError(f"impact analysis failed: {error}") from error
        payload = {
            "scope": selected.scope,
            "families": list(selected.families),
            "changed_files": list(selected.changed_files),
            "run_core_tests": selected.run_core_tests,
            "run_docs": selected.run_docs,
        }
        (self.context.repository / "impact.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return payload


class SourceQualityChecks:
    def __init__(self, context: CiContext):
        self.context = context

    def run(self) -> None:
        self.family_coverage()
        self.complexity()
        self.lint_changed_files()
        self.architecture_contracts()

    def family_coverage(self) -> None:
        self.context.run(["python", "tools/model_ci.py", "validate"])

    def complexity(self) -> None:
        self.context.run(
            [
                "python",
                "tools/check_cyclomatic_complexity.py",
                "core/runtime",
                "--max-ccn",
                "10",
                "--top",
                "20",
            ]
        )

    def lint_changed_files(self) -> None:
        missing = [name for name in ("ruff", "clang-format") if shutil.which(name) is None]
        if missing:
            raise CiError("missing source formatter: " + ", ".join(missing))
        base = self.context.env.get("CI_BASE_REF", "")
        if not base:
            raise CiError("CI_BASE_REF is required")
        python_files = self._changed_files(base, "*.py")
        if python_files:
            self.context.run(["ruff", "check", "--config", "ruff.toml", *python_files])
        native_files = self._changed_files(
            base,
            "*.c",
            "*.cc",
            "*.cpp",
            "*.cu",
            "*.cuh",
            "*.h",
            "*.hpp",
        )
        if native_files:
            self.context.run(["clang-format", "--dry-run", "--Werror", *native_files])

    def architecture_contracts(self) -> None:
        self.context.run(
            [
                "python",
                "-m",
                "pytest",
                "tools/tests/test_architecture.py",
                "tools/tests/test_family_impact.py",
                "tools/tests/test_community_ci.py",
                "tools/tests/test_public_source_hygiene.py",
                "tools/tests/test_new_ci.py",
                "tools/tests/test_pr_metadata.py",
                "-q",
                "-p",
                "no:cacheprovider",
            ],
            updates={
                "PYTHONPATH": (
                    f"{self.context.repository / 'core/builder'}:"
                    f"{self.context.repository / 'apps/benchmark'}:"
                    f"{self.context.repository}"
                ),
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            limit=self.context.env.get("SOURCE_QUALITY_TIMEOUT", "10m"),
        )

    def _changed_files(self, base: str, *patterns: str) -> list[str]:
        result = self.context.run(
            [
                "git",
                "diff",
                "--name-only",
                "--diff-filter=ACMRT",
                f"{base}...HEAD",
                "--",
                *patterns,
            ],
            capture_output=True,
        )
        return [
            line
            for line in result.stdout.splitlines()
            if line and (self.context.repository / line).is_file()
        ]


class UnitTestRunner:
    def __init__(self, context: CiContext):
        self.context = context

    def premerge(self) -> None:
        scope = self.context.env.get("TRTMC_PREMERGE_UNIT_SCOPE", "all")
        if scope != "all":
            raise CiError("the CPU unit stage supports only scope=all")
        EnvironmentVerifier(self.context).verify()
        self.context.run(
            [
                "python",
                "-m",
                "pytest",
                "core/builder/tests",
                "apps/benchmark/trtmc_benchmark/tests",
                "examples/models/cosmos3/dual_spark/test_cosmos3_dual_spark_source.py",
                (
                    "examples/models/nemotron_voicechat/full_duplex/"
                    "test_voicechat_full_duplex_source.py"
                ),
                "tools/tests",
                "-q",
                "-x",
                "-m",
                "not gpu and not trt",
                "-p",
                "no:cacheprovider",
            ],
            updates={
                "PYTHONPATH": (
                    f"{self.context.repository / 'core/builder'}:"
                    f"{self.context.repository / 'apps/benchmark'}:"
                    f"{self.context.repository}"
                ),
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            limit=self.context.env.get("PYTHON_UNIT_TIMEOUT", "20m"),
        )
        build = Path(
            self.context.env.get(
                "TRTMC_CORE_BUILD_DIR",
                "/tmp/trtmc-core-unit-build",
            )
        )
        if build.exists():
            shutil.rmtree(build)
        self.context.run(
            [
                "cmake",
                "-S",
                self.context.repository,
                "-B",
                build,
                "-G",
                "Ninja",
                "-DCMAKE_BUILD_TYPE=Release",
                "-DTRTMC_BUILD_TESTS=ON",
            ]
        )
        self.context.run(
            [
                "cmake",
                "--build",
                build,
                "--parallel",
                "8",
            ],
            limit=self.context.env.get("CPP_BUILD_TIMEOUT", "30m"),
        )
        self.context.run(
            [
                "ctest",
                "--test-dir",
                build,
                "--output-on-failure",
            ],
            limit=self.context.env.get("CPP_UNIT_TIMEOUT", "20m"),
        )
