# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen-Image condition-image build contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from families.qwen_image import model as qwen_image_model
from families.qwen_image.tests import test_e2e
from tensorrt_model_connect import BuildRequest


TEST_IMAGE = Path(__file__).resolve().parent / "data/test_img.jpeg"


class _Writer:
    def set_header(self, **kwargs):
        pass

    def add_bytes(self, name, value):
        pass

    def add_json(self, name, value):
        pass


def _request(
    tmp_path: Path,
    task: str,
    *,
    image_height: int | None = None,
    image_width: int | None = None,
) -> BuildRequest:
    model_dir = tmp_path / "model"
    (model_dir / "tokenizer").mkdir(parents=True)
    (model_dir / "model_index.json").write_text("{}", encoding="utf-8")
    (model_dir / "tokenizer/tokenizer.json").write_text("{}", encoding="utf-8")
    return BuildRequest(
        model_dir=model_dir,
        output_path=tmp_path / "model.bundle",
        family="qwen_image",
        task=task,
        precision="bf16",
        image_height=image_height,
        image_width=image_width,
    )


def test_image_edit_build_fails_closed_without_condition_image(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="image_height and image_width"):
        qwen_image_model.build(_request(tmp_path, "image_edit"), object())


def test_image_edit_build_passes_condition_geometry(monkeypatch, tmp_path: Path) -> None:
    observed = {}

    def fake_build_components(self, model_dir, config, weights, **kwargs):
        observed["raw"] = config.raw
        observed["condition"] = kwargs["edit_condition_image_size"]
        return {
            "config_json": json.dumps({"task_mode": "edit"}).encode(),
            "text_encoders": [("text", b"text")],
            "denoiser": b"denoiser",
            "vae_decoder": b"vae",
            "preprocessor_weights": b"preprocessor",
            "vision_engine": b"vision",
            "vae_encoder": b"vae-encoder",
        }

    monkeypatch.setattr(qwen_image_model._QwenImageModel, "build_components", fake_build_components)

    qwen_image_model.build(
        _request(tmp_path, "image_edit", image_height=382, image_width=640), _Writer()
    )

    assert observed == {
        "raw": {},
        "condition": (382, 640),
    }


def test_text_to_image_build_does_not_read_condition_image(monkeypatch, tmp_path: Path) -> None:
    def fake_build_components(self, model_dir, config, weights, **kwargs):
        assert config.raw == {"image_height": 1024, "image_width": 1024}
        assert kwargs["edit_condition_image_size"] is None
        return {
            "config_json": json.dumps({"task_mode": "t2i"}).encode(),
            "text_encoders": [("text", b"text")],
            "denoiser": b"denoiser",
            "vae_decoder": b"vae",
            "preprocessor_weights": b"preprocessor",
        }

    monkeypatch.setattr(qwen_image_model._QwenImageModel, "build_components", fake_build_components)

    qwen_image_model.build(_request(tmp_path, "image_generation"), _Writer())


def test_direct_edit_e2e_passes_real_condition_image_geometry(monkeypatch, tmp_path: Path) -> None:
    observed = {}

    def fake_build(request):
        observed["geometry"] = (request.image_height, request.image_width)

    monkeypatch.setattr(test_e2e, "build", fake_build)
    test_e2e._build(
        tmp_path,
        tmp_path / "model.bundle",
        {
            "task": "image_edit",
            "precision": "bf16",
            "tensor_parallel_size": 1,
            "testcases": [{"test_image": "data/test_img.jpeg"}],
        },
    )

    assert TEST_IMAGE.is_file()
    assert observed["geometry"] == (382, 640)
