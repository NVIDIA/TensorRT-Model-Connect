# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Diffusion schedulers — pure numpy implementations."""

from .base import Scheduler
from .flow_match_euler import FlowMatchEulerScheduler

__all__ = ["Scheduler", "FlowMatchEulerScheduler"]


def get_scheduler(name: str, **kwargs) -> Scheduler:
    """Create a scheduler by name."""
    schedulers = {
        "flow_match_euler": FlowMatchEulerScheduler,
    }
    if name not in schedulers:
        raise ValueError(
            f"Unknown scheduler: {name!r}. Available: {list(schedulers.keys())}")
    return schedulers[name](**kwargs)
