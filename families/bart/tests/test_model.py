# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for the bart Task SDK migration."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

try:
    import tensorrt  # noqa: F401
except ModuleNotFoundError:
    sys.modules["tensorrt"] = SimpleNamespace()

try:
    import ml_dtypes  # noqa: F401
except ModuleNotFoundError:
    sys.modules["ml_dtypes"] = SimpleNamespace(bfloat16=object())

try:
    import safetensors  # noqa: F401
except ModuleNotFoundError:
    sys.modules["safetensors"] = SimpleNamespace(safe_open=None)

from families.bart import model  # noqa: E402
from tensorrt_model_connect import BuildRequest  # noqa: E402


def _checkpoint(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "bart",
                "vocab_size": 100,
                "hidden_size": 32,
                "encoder_layers": 1,
                "decoder_layers": 1,
                "encoder_attention_heads": 2,
                "decoder_attention_heads": 2,
                "encoder_ffn_dim": 64,
                "decoder_ffn_dim": 64,
                "max_position_embeddings": 256,
                "decoder_start_token_id": 2,
                "bos_token_id": 0,
                "eos_token_id": 2,
                "pad_token_id": 1,
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


class _Writer:
    def __init__(self) -> None:
        self.sections: dict[str, object] = {}

    def set_header(self, **header) -> None:
        self.sections["header"] = header

    def add_bytes(self, name: str, value: bytes) -> None:
        self.sections[name] = value

    def add_json(self, name: str, value: object) -> None:
        self.sections[name] = value


def _patch_build(monkeypatch) -> None:
    monkeypatch.setattr(model._BartModel, "load_weights", lambda self, *a, **k: {})
    monkeypatch.setattr(model._BartModel, "build_engine", lambda self, *a, **k: b"plan")
    monkeypatch.setattr(
        model._BartModel, "build_vision_engine", lambda self, *a, **k: b"encoder_plan"
    )


@pytest.mark.parametrize("task", ["conditional_text_generation", "text_generation"])
def test_build_accepts_every_supported_task(tmp_path: Path, monkeypatch, task: str) -> None:
    _patch_build(monkeypatch)
    writer = _Writer()

    model.build(
        BuildRequest(
            model_dir=_checkpoint(tmp_path),
            output_path=tmp_path / "unused.bundle",
            family="bart",
            task=task,
            precision="fp32",
        ),
        writer,
    )

    assert writer.sections["header"]["task"] == task
    assert writer.sections["engine.plan"] == b"plan"
    assert writer.sections["encoder.plan"] == b"encoder_plan"
    assert writer.sections["runtime.json"]["tensor_parallel_size"] == 1


@pytest.mark.parametrize(
    "task",
    [
        "classification",
        "text_to_audio",
        "text_to_embedding",
        "unconditional_text_generation",
    ],
)
def test_build_rejects_unsupported_tasks(tmp_path: Path, monkeypatch, task: str) -> None:
    _patch_build(monkeypatch)

    with pytest.raises(ValueError, match="bart supports only task="):
        model.build(
            BuildRequest(
                model_dir=_checkpoint(tmp_path),
                output_path=tmp_path / "unused.bundle",
                family="bart",
                task=task,
                precision="fp32",
            ),
            _Writer(),
        )
