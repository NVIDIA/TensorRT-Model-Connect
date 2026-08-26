# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Define the readable stage graph executed inside the long-lived CI container.

Boundary: ordering only; each step delegates its behavior to a focused class.
"""

from __future__ import annotations

from collections.abc import Callable

from .context import CiContext
from .coverage import CoverageRunner
from .e2e import E2ERunner
from .package import WheelPackageManager
from .process import CiError
from .quality import EnvironmentVerifier, ImpactAnalyzer, SourceQualityChecks, UnitTestRunner


class CiPipeline:
    """Map every public CI stage to a short sequence of named class methods."""

    def __init__(self, context: CiContext):
        self.context = context
        self.environment = EnvironmentVerifier(context)
        self.impact = ImpactAnalyzer(context)
        self.quality = SourceQualityChecks(context)
        self.units = UnitTestRunner(context)
        self.package = WheelPackageManager(context)
        self.coverage = CoverageRunner(context)
        self.e2e = E2ERunner(context)
        self.stages: dict[str, list[tuple[str, Callable[[], None]]]] = {
            "setup": [
                ("Install trtmc pip package", self.package.install_once),
            ],
            "impact": [
                ("Setup TensorRT-Model-Connect source checks", self.environment.verify),
                ("Impact analysis", self.impact.run),
            ],
            "build": [
                ("Setup TensorRT-Model-Connect source checks", self.environment.verify),
                ("Build C++ test executables", self.units.build_cpp_tests),
            ],
            "family-coverage": [
                ("Setup TensorRT-Model-Connect source checks", self.environment.verify),
                ("Check family coverage", self.quality.family_coverage),
            ],
            "complexity": [
                ("Setup TensorRT-Model-Connect source checks", self.environment.verify),
                ("Check cyclomatic complexity", self.quality.complexity),
            ],
            "lint": [
                ("Setup TensorRT-Model-Connect source checks", self.environment.verify),
                ("Lint changed files", self.quality.lint_changed_files),
            ],
            "source-quality": [
                ("Check cyclomatic complexity", self.quality.complexity),
                ("Lint changed files", self.quality.lint_changed_files),
                ("Check model architecture contracts", self.quality.architecture_contracts),
            ],
            "cpp-unit": [
                ("Setup TensorRT-Model-Connect source checks", self.environment.verify),
                ("C++ unit tests", self.units.cpp),
            ],
            "python-builder": [
                ("Setup TensorRT-Model-Connect wheel runtime", self._verify_wheel_runtime),
                ("Python builder and tools tests", self.coverage.python_builder_tests),
            ],
            "premerge-unit": [
                ("Source-only C++ and Python unit tests", self.units.premerge),
            ],
            "cpp-coverage": [
                ("Setup TensorRT-Model-Connect source checks", self.environment.verify),
                ("C++ coverage", self.coverage.cpp),
            ],
            "graph-ops": [
                ("Setup TensorRT-Model-Connect wheel runtime", self._verify_wheel_runtime),
                ("Graph-op GPU tests", self.units.graph_ops),
            ],
            "selective-e2e": [
                ("Setup TensorRT-Model-Connect wheel runtime", self._verify_wheel_runtime),
                ("Selective E2E tests", self.e2e.selective),
            ],
            "full-e2e": [
                ("Setup TensorRT-Model-Connect wheel runtime", self._verify_wheel_runtime),
                ("Full E2E tests", self.e2e.full),
            ],
            "coverage-map": [
                ("Setup TensorRT-Model-Connect wheel runtime", self._verify_wheel_runtime),
                ("Generate coverage map", self.coverage.map),
            ],
            "package": [
                ("Setup TensorRT-Model-Connect package build environment", self.environment.verify),
                ("Build trtmc pip package", self.package.build),
                ("Install trtmc pip package", self.package.install_once),
            ],
            "package-preflight": [
                ("Setup TensorRT-Model-Connect package preflight", self.environment.verify),
                ("Validate pre-install package metadata", self.package.preflight),
            ],
            "wheel-model-smoke": [
                ("Setup TensorRT-Model-Connect source checks", self.environment.verify),
                ("Model smoke test from trtmc pip package", self.package.model_smoke),
            ],
        }

    def run(self, stage: str) -> None:
        steps = self.stages.get(stage)
        if steps is None:
            raise CiError(f"Unknown CI stage: {stage}")
        self.context.prepare_shared_directories()
        try:
            for name, operation in steps:
                self._step(name, operation)
        finally:
            self._generate_e2e_report()

    def _verify_wheel_runtime(self) -> None:
        self.environment.verify()
        self.package.verify_installed()

    @staticmethod
    def _step(name: str, operation: Callable[[], None]) -> None:
        print(f"::group::{name}")
        try:
            operation()
        finally:
            print("::endgroup::")

    def _generate_e2e_report(self) -> None:
        artifacts = self.context.repository / "e2e_artifacts"
        if not artifacts.is_dir():
            return
        self.context.run(
            [
                "python",
                "scripts/generate_e2e_report.py",
                "--artifacts-dir",
                artifacts / "artifacts",
                "-o",
                artifacts / "e2e_report.html",
                "--manifest-dir",
                "tests/e2e/models",
                "--project-dir",
                ".",
                "--title",
                f"GitHub Actions E2E Report - {self.context.env.get('GITHUB_RUN_ID', 'local')}",
            ],
            check=False,
        )
