# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

from tensorrt_model_connect.models import (
    family_hf_allow_patterns,
    family_hf_required_files_by_id,
    resolve_family_model_dir,
)


def test_lance_nested_checkpoint_files_are_downloaded() -> None:
    patterns = family_hf_allow_patterns()

    assert "Lance_3B/**" in patterns
    assert "Qwen2.5-VL-ViT/**" in patterns

    assert set(family_hf_required_files_by_id()["bytedance-research/Lance"]) == {
        "Lance_3B/llm_config.json",
        "Lance_3B/model.safetensors",
        "Qwen2.5-VL-ViT/vit.safetensors",
    }


def test_lance_model_dir_adapter_stages_nested_checkpoint(
    tmp_path, monkeypatch,
) -> None:
    source = tmp_path / "source"
    llm = source / "Lance_3B"
    vision = source / "Qwen2.5-VL-ViT"
    llm.mkdir(parents=True)
    vision.mkdir(parents=True)
    (llm / "llm_config.json").write_text(json.dumps({
        "model_type": "qwen2_5_vl",
        "hidden_size": 16,
    }))
    for name in (
        "model.safetensors",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "generation_config.json",
    ):
        (llm / name).write_text(name)
    (vision / "vit.safetensors").write_text("vision")
    staging_root = tmp_path / "staging"
    monkeypatch.setenv("TRTMC_FAMILY_MODEL_ROOT", str(staging_root))

    staged = resolve_family_model_dir(source)

    assert staged is not None
    staged_path = staging_root / next(staging_root.iterdir()).name
    config = json.loads((staged_path / "config.json").read_text())
    assert config["model_type"] == "lance"
    assert (staged_path / "model.safetensors").is_symlink()
    assert (staged_path / "vision" / "model.safetensors").is_symlink()
