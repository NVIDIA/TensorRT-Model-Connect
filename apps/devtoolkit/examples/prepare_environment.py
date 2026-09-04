# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare one persistent Docker environment from the current checkout."""

from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY / "apps/devtoolkit"))

from trtmc_devtoolkit import DevToolkit, DockerTargetPolicy  # noqa: E402


environment = DevToolkit.from_checkout(REPOSITORY).prepare_docker(
    family="timm_resnet",
    gpu="0",
    policy=DockerTargetPolicy.ENSURE,
)
print(" ".join(environment.command("bash")))
