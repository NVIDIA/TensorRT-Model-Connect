# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GLM-ASR tensor-parallel build configuration.

The decoder builder does not shard weights or emit collective ops, so this
family only accepts tp_size=1; build() rejects anything else before this
config is ever used.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParallelConfig:
    tp_size: int = 1

    def validate(self) -> None:
        if self.tp_size != 1:
            raise ValueError("glmasr does not support tensor_parallel_size > 1")
