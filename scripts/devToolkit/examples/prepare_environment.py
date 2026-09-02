# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare one persistent Docker development environment from a checkout."""

from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY / "scripts" / "devToolkit"))

from trtmc_devtoolkit import DevToolkit, DockerTarget, PrepareRequest  # noqa: E402


toolkit = DevToolkit.from_checkout(REPOSITORY)
plan = toolkit.plan(
    PrepareRequest(
        tensorrt="11.1.0.106",
        cuda="13.3",
        target=DockerTarget(gpu="0"),
        mode="development",
    )
)

print("Preparation plan:")
for step in plan.steps:
    print(f"  {step.id}: {step.description}")

result = toolkit.apply(plan)
print(f"Ready: {result.environment.activate_command}")
print(f"Receipt: {result.receipt}")
