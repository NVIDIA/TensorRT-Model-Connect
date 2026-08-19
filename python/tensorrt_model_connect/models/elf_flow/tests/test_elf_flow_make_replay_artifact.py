# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

from tensorrt_model_connect.models.elf_flow.make_replay_artifact import build_artifact
from tensorrt_model_connect.models.elf_flow.validate_replay_artifact import (
    validate_artifact,
)


def _write_f32(path: Path, values: list[float]) -> None:
    path.write_bytes(struct.pack(f"{len(values)}f", *values))


def _args(tmp_path: Path, **overrides):
    values = {
        "output": str(tmp_path / "elf_replay.json"),
        "generation_mode": "unconditional",
        "model_id": "",
        "variant": "",
        "max_length": None,
        "max_input_length": None,
        "text_encoder_dim": None,
        "num_samples": None,
        "num_sampling_steps": None,
        "self_cond_cfg_scale": None,
        "cfg_scale": None,
        "sde_gamma": None,
        "seed": None,
        "initial_latents_raw": "",
        "condition_latents_raw": "",
        "condition_mask_raw": "",
        "sampling_steps_raw": "",
        "sde_noise_raw": "",
        "expected_generated_jsonl_path": "",
        "samples_jsonl": "",
        "sample": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_make_single_sample_replay_artifact(tmp_path: Path) -> None:
    _write_f32(tmp_path / "initial.f32", [0.0, 1.0])
    _write_f32(tmp_path / "steps.f32", [0.0, 1.0])
    (tmp_path / "expected.jsonl").write_text(
        '{"id": 0, "generated": "Replay.", "token_ids": [1]}\n',
        encoding="utf-8",
    )
    artifact_path = tmp_path / "elf_replay.json"
    artifact = build_artifact(
        _args(
            tmp_path,
            output=str(artifact_path),
            max_length=1,
            text_encoder_dim=2,
            num_samples=1,
            initial_latents_raw=str(tmp_path / "initial.f32"),
            sampling_steps_raw=str(tmp_path / "steps.f32"),
            expected_generated_jsonl_path=str(tmp_path / "expected.jsonl"),
        )
    )
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    resolved = validate_artifact(artifact_path)

    assert artifact["files"]["initial_latents_raw"] == "initial.f32"
    assert artifact["files"]["expected_generated_jsonl_path"] == "expected.jsonl"
    assert resolved["initial_float_count"] == 2
    assert resolved["expected_sample_count"] == 1


def test_make_multi_sample_replay_artifact_from_samples_jsonl(tmp_path: Path) -> None:
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
    (tmp_path / "samples.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"id": 0, "initial": str(tmp_path / "initial0.f32")}),
                json.dumps({"id": 1, "initial": str(tmp_path / "initial1.f32")}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    artifact_path = tmp_path / "elf_replay.json"
    artifact = build_artifact(
        _args(
            tmp_path,
            output=str(artifact_path),
            max_length=1,
            text_encoder_dim=1,
            sampling_steps_raw=str(tmp_path / "steps.f32"),
            expected_generated_jsonl_path=str(tmp_path / "expected.jsonl"),
            samples_jsonl=str(tmp_path / "samples.jsonl"),
        )
    )
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    resolved = validate_artifact(artifact_path)

    assert artifact["num_samples"] == 2
    assert artifact["samples"][0]["files"]["initial_latents_raw"] == "initial0.f32"
    assert artifact["samples"][1]["files"]["initial_latents_raw"] == "initial1.f32"
    assert resolved["replay_sample_count"] == 2


def test_make_multi_sample_replay_artifact_from_sample_flags(tmp_path: Path) -> None:
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
    artifact_path = tmp_path / "elf_replay.json"
    artifact = build_artifact(
        _args(
            tmp_path,
            output=str(artifact_path),
            max_length=1,
            text_encoder_dim=1,
            sampling_steps_raw=str(tmp_path / "steps.f32"),
            expected_generated_jsonl_path=str(tmp_path / "expected.jsonl"),
            sample=[
                f"id=0,initial={tmp_path / 'initial0.f32'}",
                f"id=1,initial={tmp_path / 'initial1.f32'}",
            ],
        )
    )
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    resolved = validate_artifact(artifact_path)

    assert artifact["samples"][0]["id"] == 0
    assert artifact["samples"][1]["id"] == 1
    assert resolved["replay_sample_count"] == 2
