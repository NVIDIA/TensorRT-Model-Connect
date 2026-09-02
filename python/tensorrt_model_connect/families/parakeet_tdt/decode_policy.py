# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure greedy TDT state transition used by tests and native parity review."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class TDTGreedyDecision:
    emit_token: bool
    frame_advance: int


def make_tdt_greedy_decision(
    token_id: int,
    duration_index: int,
    durations: Sequence[int],
    blank_id: int,
) -> TDTGreedyDecision:
    if not 0 <= duration_index < len(durations):
        raise ValueError(f"duration_index {duration_index} is outside {len(durations)} durations")
    duration = int(durations[duration_index])
    if duration < 0:
        raise ValueError("TDT durations must be non-negative")

    emit = token_id != blank_id
    if duration == 0 and not emit:
        duration = 1
    return TDTGreedyDecision(emit_token=emit, frame_advance=duration)
