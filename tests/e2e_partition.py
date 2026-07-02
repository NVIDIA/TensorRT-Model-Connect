# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Round-robin partitioning of E2E models across N agents.

Usage:
    from tests.e2e_partition import partition_models

    # Returns list of model names for agent 0 of 4:
    my_models = partition_models(all_cases, num_agents=4, agent_id=0)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tests.e2e_harness.contracts import E2ECase


def partition_models(
    cases: list[E2ECase],
    num_agents: int,
    agent_id: int,
) -> list[str]:
    """Round-robin partition of E2E cases across N agents.

    Distributes models evenly across agents using simple round-robin
    on the sorted case list. Each test runs until it passes or fails
    with no time estimation.

    Args:
        cases: All E2ECase instances to partition.
        num_agents: Total number of parallel agents.
        agent_id: This agent's index (0-based).

    Returns:
        List of case names assigned to this agent.
    """
    if num_agents <= 0 or agent_id < 0 or agent_id >= num_agents:
        raise ValueError(
            f"Invalid partition params: num_agents={num_agents}, agent_id={agent_id}")

    if not cases:
        return []

    # Stable sort by name, then round-robin assign
    sorted_cases = sorted(cases, key=lambda c: c.name)
    return [c.name for i, c in enumerate(sorted_cases) if i % num_agents == agent_id]
