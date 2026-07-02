# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from tensorrt_model_connect.config import ModelConfig
from tensorrt_model_connect.families.elf_flow.prepare_model_dir import (
    prepare_model_dir,
)


def _args(tmp_path: Path, **overrides):
    values = {
        "config": str(tmp_path / "train_owt_ELF-B.yml"),
        "checkpoint_path": str(tmp_path / "ELF-B-owt"),
        "tokenizer": str(tmp_path / "tokenizer"),
        "encoder_checkpoint": "",
        "output": str(tmp_path / "prepared"),
        "copy": True,
        "force": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _write_config(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "model: ELF-B",
                "max_length: 1024",
                "encoder_model_name: t5-small",
                "denoiser_noise_scale: 2.0",
                "self_cond_prob: 0.5",
            ]
        ),
        encoding="utf-8",
    )


def test_prepare_elf_model_dir_combines_config_checkpoint_and_tokenizer(tmp_path: Path) -> None:
    _write_config(tmp_path / "train_owt_ELF-B.yml")
    checkpoint_dir = tmp_path / "ELF-B-owt"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "checkpoint_42").write_bytes(b"checkpoint")
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    (tokenizer_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tokenizer_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    encoder_checkpoint = tmp_path / "t5_small_encoder_jax.pkl"
    encoder_checkpoint.write_bytes(b"encoder")

    result = prepare_model_dir(_args(tmp_path, encoder_checkpoint=str(encoder_checkpoint)))
    output = Path(result["output_dir"])
    cfg = ModelConfig.from_dir(output)

    assert result["config"] == "train_owt_ELF-B.yml"
    assert result["checkpoints"] == ["checkpoint_42"]
    assert result["tokenizer_files"] == ["tokenizer.json", "tokenizer_config.json"]
    assert result["encoder_checkpoint"] == "t5_small_encoder_jax.pkl"
    assert (output / "train_owt_ELF-B.yml").exists()
    assert (output / "checkpoint_42").read_bytes() == b"checkpoint"
    assert (output / "tokenizer.json").exists()
    assert (output / "t5_small_encoder_jax.pkl").read_bytes() == b"encoder"
    assert cfg.model_type == "elf"
    assert cfg.raw["model"] == "ELF-B"


def test_prepare_elf_model_dir_rejects_checkpoint_dir_without_checkpoint(tmp_path: Path) -> None:
    _write_config(tmp_path / "train_owt_ELF-B.yml")
    (tmp_path / "ELF-B-owt").mkdir()

    with pytest.raises(FileNotFoundError, match="no ELF checkpoint"):
        prepare_model_dir(_args(tmp_path, tokenizer=""))


def test_prepare_elf_model_dir_rejects_non_empty_output_without_force(tmp_path: Path) -> None:
    _write_config(tmp_path / "train_owt_ELF-B.yml")
    checkpoint_dir = tmp_path / "ELF-B-owt"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "checkpoint_42").write_bytes(b"checkpoint")
    output = tmp_path / "prepared"
    output.mkdir()
    (output / "existing.txt").write_text("x", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        prepare_model_dir(_args(tmp_path, tokenizer=""))
