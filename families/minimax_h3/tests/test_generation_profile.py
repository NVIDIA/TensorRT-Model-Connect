# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

from families.minimax_h3.config import (
    BASE_GENERATION_PROFILE,
    FASTH3_DENSE_4STEP_GENERATION_PROFILE,
    FASTH3_DENSE_4STEP_MODEL_ID,
    FASTH3_VSA_4STEP_GENERATION_PROFILE,
    FASTH3_VSA_4STEP_MODEL_ID,
)
from families.minimax_h3.model import _generation_profile


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_fasth3_contract(root: Path, *, vsa: bool = False, **overrides) -> None:
    profile = (
        FASTH3_VSA_4STEP_GENERATION_PROFILE
        if vsa
        else FASTH3_DENSE_4STEP_GENERATION_PROFILE
    )
    contract = {
        "schema_version": "fasth3-inference-contract-v1",
        "model_id": FASTH3_VSA_4STEP_MODEL_ID if vsa else FASTH3_DENSE_4STEP_MODEL_ID,
        "attention_backend": profile.attention_backend,
        "checkpoint_step": 1300 if vsa else 1000,
        "guidance_scale": 1.0,
        "num_inference_steps": 5,
        "transformer_forwards": 4,
        "task": "t2av",
        "dmd_denoising_steps": [999, 749, 500, 250],
    }
    if vsa:
        contract.update(
            {
                "vsa_tile_size": 64,
                "vsa_sparsity": 0.9,
                "vsa_kernel": "sm100a",
            }
        )
    contract.update(overrides)
    _write_json(root / "fastvideo_inference.json", contract)
    _write_json(
        root / "scheduler" / "scheduler_config.json",
        {"_class_name": "MiniMaxH3Scheduler", "shift": 12.0},
    )
    _write_json(
        root / "audio_scheduler" / "scheduler_config.json",
        {"_class_name": "MiniMaxH3Scheduler", "shift": 3.0},
    )


def test_checkpoint_without_fastvideo_contract_uses_base_schedule(tmp_path: Path) -> None:
    assert _generation_profile(tmp_path) == BASE_GENERATION_PROFILE


def test_fasth3_dense_contract_selects_exact_four_step_schedule(tmp_path: Path) -> None:
    _write_fasth3_contract(tmp_path)

    profile = _generation_profile(tmp_path)

    assert profile == FASTH3_DENSE_4STEP_GENERATION_PROFILE
    assert profile.num_inference_steps == 5
    assert profile.transformer_forwards == 4
    assert profile.dmd_denoising_steps == (999, 749, 500, 250)


def test_fasth3_vsa_contract_selects_exact_sparse_recipe(tmp_path: Path) -> None:
    _write_fasth3_contract(tmp_path, vsa=True)

    profile = _generation_profile(tmp_path)

    assert profile == FASTH3_VSA_4STEP_GENERATION_PROFILE
    assert profile.uses_vsa


@pytest.mark.parametrize(
    ("override", "match"),
    (
        ({"vsa_tile_size": 256}, "vsa_tile_size"),
        ({"vsa_sparsity": 0.8}, "vsa_sparsity"),
        ({"vsa_kernel": "triton"}, "vsa_kernel"),
        ({"checkpoint_step": 1000}, "checkpoint_step"),
    ),
)
def test_fasth3_vsa_contract_rejects_untrained_recipe(
    tmp_path: Path, override: dict, match: str
) -> None:
    _write_fasth3_contract(tmp_path, vsa=True, **override)

    with pytest.raises(ValueError, match=match):
        _generation_profile(tmp_path)


@pytest.mark.parametrize(
    ("override", "match"),
    (
        ({"attention_backend": "VIDEO_SPARSE_ATTN_H3"}, "attention_backend"),
        ({"dmd_denoising_steps": [999, 750, 500, 250]}, "dmd_denoising_steps"),
        ({"num_inference_steps": 4}, "num_inference_steps"),
    ),
)
def test_fasth3_dense_contract_rejects_untrained_recipe(
    tmp_path: Path, override: dict, match: str
) -> None:
    _write_fasth3_contract(tmp_path, **override)

    with pytest.raises(ValueError, match=match):
        _generation_profile(tmp_path)


def test_fasth3_dense_contract_rejects_scheduler_shift_mismatch(tmp_path: Path) -> None:
    _write_fasth3_contract(tmp_path)
    _write_json(
        tmp_path / "scheduler" / "scheduler_config.json",
        {"_class_name": "MiniMaxH3Scheduler", "shift": 10.0},
    )

    with pytest.raises(ValueError, match="video scheduler shift must be 12.0"):
        _generation_profile(tmp_path)
