# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from tensorrt_model_connect.models.elf_flow.validate_replay_artifact import (
    validate_artifact,
)


def _write_f32(path: Path, values: list[float]) -> None:
    path.write_bytes(struct.pack(f"{len(values)}f", *values))


def test_validate_conditional_replay_artifact_resolves_relative_files(tmp_path: Path) -> None:
    _write_f32(tmp_path / "initial.f32", [0.0, 1.0, 2.0, 3.0])
    _write_f32(tmp_path / "steps.f32", [0.0, 0.5, 1.0])
    _write_f32(tmp_path / "sde.f32", [0.1, 0.2, 0.3, 0.4])
    _write_f32(tmp_path / "cond.f32", [4.0, 5.0, 0.0, 0.0])
    _write_f32(tmp_path / "mask.f32", [1.0, 0.0])
    (tmp_path / "expected.jsonl").write_text(
        '{"id": 0, "generated": "A replayed sentence.", "token_ids": [7, 8]}\n',
        encoding="utf-8",
    )
    artifact = tmp_path / "elf_replay.json"
    artifact.write_text(
        json.dumps(
            {
                "generation_mode": "conditional",
                "max_length": 2,
                "text_encoder_dim": 2,
                "files": {
                    "initial_latents_raw": "initial.f32",
                    "sampling_steps_raw": "steps.f32",
                    "sde_noise_raw": "sde.f32",
                    "condition_latents_raw": "cond.f32",
                    "condition_mask_raw": "mask.f32",
                    "expected_generated_jsonl_path": "expected.jsonl",
                },
            }
        ),
        encoding="utf-8",
    )

    resolved = validate_artifact(artifact)

    assert resolved["generation_mode"] == "conditional"
    assert resolved["initial_latents"] == str(tmp_path / "initial.f32")
    assert resolved["sampling_step_count"] == 3
    assert resolved["condition_mask_float_count"] == 2
    assert resolved["sde_noise_float_count"] == 4
    assert resolved["expected_sample_count"] == 1


def test_validate_multi_sample_replay_artifact_resolves_each_sample(tmp_path: Path) -> None:
    _write_f32(tmp_path / "initial0.f32", [0.0])
    _write_f32(tmp_path / "initial1.f32", [1.0])
    _write_f32(tmp_path / "steps.f32", [0.0, 1.0])
    (tmp_path / "expected.jsonl").write_text(
        "\n".join(
            [
                '{"id": 0, "generated": "Replay zero.", "token_ids": [1]}',
                '{"id": 1, "generated": "Replay one.", "token_ids": [2]}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    artifact = tmp_path / "elf_replay.json"
    artifact.write_text(
        json.dumps(
            {
                "generation_mode": "unconditional",
                "num_samples": 2,
                "max_length": 1,
                "text_encoder_dim": 1,
                "files": {
                    "sampling_steps_raw": "steps.f32",
                    "expected_generated_jsonl_path": "expected.jsonl",
                },
                "samples": [
                    {"files": {"initial_latents_raw": "initial0.f32"}},
                    {"files": {"initial_latents_raw": "initial1.f32"}},
                ],
            }
        ),
        encoding="utf-8",
    )

    resolved = validate_artifact(artifact)

    assert resolved["replay_sample_count"] == 2
    assert resolved["expected_sample_count"] == 2
    assert resolved["samples"][0]["initial_latents"] == str(tmp_path / "initial0.f32")
    assert resolved["samples"][1]["initial_latents"] == str(tmp_path / "initial1.f32")


def test_validate_multi_sample_replay_artifact_requires_samples_list(tmp_path: Path) -> None:
    _write_f32(tmp_path / "initial.f32", [0.0])
    _write_f32(tmp_path / "steps.f32", [0.0, 1.0])
    artifact = tmp_path / "elf_replay.json"
    artifact.write_text(
        json.dumps(
            {
                "generation_mode": "unconditional",
                "num_samples": 2,
                "files": {
                    "initial_latents_raw": "initial.f32",
                    "sampling_steps_raw": "steps.f32",
                },
                "expected_generated_samples": [
                    {"id": 0, "generated": "Replay zero.", "token_ids": [1]},
                    {"id": 1, "generated": "Replay one.", "token_ids": [2]},
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="samples list"):
        validate_artifact(artifact)


def test_validate_replay_artifact_requires_expected_token_ids(tmp_path: Path) -> None:
    _write_f32(tmp_path / "initial.f32", [0.0])
    _write_f32(tmp_path / "steps.f32", [0.0, 1.0])
    artifact = tmp_path / "elf_replay.json"
    artifact.write_text(
        json.dumps(
            {
                "generation_mode": "unconditional",
                "files": {
                    "initial_latents_raw": "initial.f32",
                    "sampling_steps_raw": "steps.f32",
                },
                "expected_generated_samples": [{"id": 0, "generated": "Text only"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="token_ids"):
        validate_artifact(artifact)


def test_validate_replay_artifact_checks_declared_latent_shape(tmp_path: Path) -> None:
    _write_f32(tmp_path / "initial.f32", [0.0, 1.0, 2.0])
    _write_f32(tmp_path / "steps.f32", [0.0, 1.0])
    artifact = tmp_path / "elf_replay.json"
    artifact.write_text(
        json.dumps(
            {
                "generation_mode": "unconditional",
                "max_length": 2,
                "text_encoder_dim": 2,
                "files": {
                    "initial_latents_raw": "initial.f32",
                    "sampling_steps_raw": "steps.f32",
                },
                "expected_generated_samples": [
                    {"id": 0, "generated": "Shape mismatch", "token_ids": [1]}
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="initial latents"):
        validate_artifact(artifact)
