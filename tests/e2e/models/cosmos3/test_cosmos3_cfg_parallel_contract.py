# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qualification contracts for Cosmos3 B200 classifier-free parallelism."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorrt_model_connect.families.cosmos3.plugin import Cosmos3Plugin
from tensorrt_model_connect.families.cosmos3.transformer_builder import (
    select_cp_execution_sizes,
)


@pytest.mark.parametrize(
    ("requested_cp_size", "denoiser_cp_size"),
    [(4, 2), (8, 4)],
)
def test_b200_distributed_runs_two_classifier_free_branches(
    requested_cp_size: int,
    denoiser_cp_size: int,
) -> None:
    assert select_cp_execution_sizes(
        (10, 0),
        requested_cp_size=requested_cp_size,
    ) == (denoiser_cp_size, 2)


@pytest.mark.parametrize(
    ("compute_capability", "requested_cp_size"),
    [
        ((10, 0), 1),
        ((10, 0), 2),
        ((8, 0), 2),
        ((8, 0), 4),
        ((8, 0), 8),
        ((9, 0), 8),
        ((10, 3), 8),
        ((11, 0), 8),
    ],
)
def test_classifier_free_parallelism_preserves_unqualified_topologies(
    compute_capability: tuple[int, int], requested_cp_size: int
) -> None:
    assert select_cp_execution_sizes(
        compute_capability,
        requested_cp_size=requested_cp_size,
    ) == (requested_cp_size, 1)


def test_candidate_bundle_records_both_execution_dimensions() -> None:
    config = SimpleNamespace(raw={})
    bundle = Cosmos3Plugin().diffusion_bundle_config(
        config,
        components={
            "negative_prompt": "{}",
            "denoiser_context_parallel_size": 4,
            "classifier_free_parallel_size": 2,
        },
    )
    assert bundle["denoiser_context_parallel_size"] == 4
    assert bundle["classifier_free_parallel_size"] == 2


def test_non_candidate_bundle_does_not_add_parallel_metadata() -> None:
    config = SimpleNamespace(raw={})
    bundle = Cosmos3Plugin().diffusion_bundle_config(
        config,
        components={"negative_prompt": "{}"},
    )
    assert "denoiser_context_parallel_size" not in bundle
    assert "classifier_free_parallel_size" not in bundle


def test_single_rank_context_subgroups_still_bind_their_global_cuda_device() -> None:
    root = Path(__file__).resolve().parents[4]
    source = (root / "src/runtime/core/distributed_runtime.cpp").read_text(encoding="utf-8")
    bind = source.index("bind_cuda_device_for_rank(group.global_rank);")
    singleton_return = source.index("if (group_size == 1)", bind)
    assert bind < singleton_return
