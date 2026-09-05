# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The complete, intentionally small CI stage graph."""

from __future__ import annotations

from collections.abc import Callable

from .context import CiContext
from .e2e import E2ERunner
from .package import WheelPackageManager
from .process import CiError
from .quality import ImpactAnalyzer, SourceQualityChecks, UnitTestRunner


class CiPipeline:
    def __init__(self, context: CiContext):
        package = WheelPackageManager(context)
        quality = SourceQualityChecks(context)
        self.stages: dict[str, tuple[tuple[str, Callable[[], object]], ...]] = {
            "impact": (("Resolve physical family impact", ImpactAnalyzer(context).run),),
            "family-coverage": (("Validate physical family coverage", quality.family_coverage),),
            "complexity": (("Check shared native complexity", quality.complexity),),
            "lint": (("Lint changed source files", quality.lint_changed_files),),
            "source-quality": (("Run new-architecture source checks", quality.run),),
            "premerge-unit": (("Run all CPU units", UnitTestRunner(context).premerge),),
            "package": (
                ("Build native wheel", package.build),
                ("Install and verify native wheel", package.install_once),
            ),
            "setup": (("Install and verify native wheel", package.install_once),),
            "selective-e2e": (("Run impacted family E2E", E2ERunner(context).selective),),
        }

    def run(self, stage: str) -> None:
        steps = self.stages.get(stage)
        if steps is None:
            raise CiError(f"unknown CI stage: {stage}")
        for name, operation in steps:
            print(f"::group::{name}")
            try:
                operation()
            finally:
                print("::endgroup::")
