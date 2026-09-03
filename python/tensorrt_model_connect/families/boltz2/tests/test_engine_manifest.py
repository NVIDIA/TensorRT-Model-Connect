# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import replace

import pytest

from tensorrt_model_connect.families.boltz2.engine_manifest import (
    ALL_ENGINE_SPECS,
    PAIRFORMER_ENGINE_SPECS,
    atom_attention_fence_outputs,
    graph_manifest_json,
    validate_engine_specs,
)


def test_family_uses_no_onnx_interchange_path() -> None:
    family = Path(__file__).resolve().parents[1]
    repository = Path(__file__).resolve().parents[5]
    runtime = repository / "src" / "runtime" / "models" / "boltz2"
    production_sources = [path for path in family.glob("*.py") if path.name != "__init__.py"]
    production_sources.extend(
        path for path in runtime.rglob("*") if path.suffix in {".cpp", ".h"}
    )
    forbidden = ("torch.onnx", "onnxparser", ".onnx", "import onnx")
    for source in production_sources:
        text = source.read_text(encoding="utf-8").lower()
        assert not any(token in text for token in forbidden), source


def test_pairformer_engine_specs_cover_all_blocks_once() -> None:
    validate_engine_specs()
    assert len(PAIRFORMER_ENGINE_SPECS) == 8
    assert PAIRFORMER_ENGINE_SPECS[0].first_block == 0
    assert PAIRFORMER_ENGINE_SPECS[-1].first_block == 56


def test_pairformer_engine_specs_reject_gap() -> None:
    broken = (
        PAIRFORMER_ENGINE_SPECS[0],
        replace(PAIRFORMER_ENGINE_SPECS[1], first_block=9),
        *PAIRFORMER_ENGINE_SPECS[2:],
    )
    with pytest.raises(ValueError, match="cover blocks"):
        validate_engine_specs(broken)


def test_graph_manifest_is_stable_and_self_describing() -> None:
    document = json.loads(graph_manifest_json(token_count=117, tensorrt_version="11.2.1.2"))
    assert document["family"] == "boltz2"
    assert document["precision"] == "bf16-mixed"
    assert document["token_count"] == 117
    assert document["atom_count"] == 928
    assert document["recycling_passes"] == 4
    assert document["sampling_steps"] == 200
    pairformer = next(engine for engine in document["engines"] if engine["role"] == "pairformer")
    assert pairformer["inputs"] == ["s", "z", "token_mask"]
    score_input = next(
        engine for engine in document["engines"] if engine["role"] == "diffusion_score_input"
    )
    assert score_input["outputs"][-36:] == list(
        atom_attention_fence_outputs("encoder_fence")
    )


def test_graph_manifest_covers_every_runtime_stage() -> None:
    roles = {spec.role for spec in ALL_ENGINE_SPECS}
    assert roles == {
        "input_embedder",
        "trunk_init",
        "msa",
        "pairformer",
        "diffusion_conditioning",
        "diffusion_score_input",
        "diffusion_token_transformer",
        "diffusion_score_output",
        "confidence",
    }
    assert len(ALL_ENGINE_SPECS) == 19
