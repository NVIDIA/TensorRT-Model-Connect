# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks of the Laguna builder's request handling and chat-template decision."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

pytest.importorskip("tensorrt")

from families.laguna import model  # noqa: E402
from tensorrt_model_connect import BuildRequest  # noqa: E402


def _request(tmp_path: Path, **overrides) -> BuildRequest:
    request = BuildRequest(
        model_dir=tmp_path,
        output_path=tmp_path / "laguna.bundle",
        family="laguna",
        task="text_continuation",
        precision="bf16",
    )
    return dataclasses.replace(request, **overrides)


@pytest.mark.parametrize(
    ("override", "error"),
    [
        ({"dynamic_kv_cache": True}, NotImplementedError),
        ({"tensor_parallel_size": 2}, NotImplementedError),
        ({"max_batch_size": 2}, NotImplementedError),
        ({"quantization": "nvfp4"}, NotImplementedError),
        ({"fp32_layers": (1,)}, NotImplementedError),
        ({"mtp_seq_len": 2}, NotImplementedError),
        ({"image_height": 224}, NotImplementedError),
        ({"video_num_frames": 4}, NotImplementedError),
        ({"context_parallel_size": 2}, ValueError),
        ({"task": "text_generation"}, ValueError),
        ({"backend": "trt_rtx"}, ValueError),
        ({"precision": "fp16"}, ValueError),
    ],
)
def test_build_rejects_options_it_cannot_honour(tmp_path: Path, override: dict, error) -> None:
    with pytest.raises(error):
        model.build(_request(tmp_path, **override), writer=None)


def test_chat_template_is_enabled_only_for_the_laguna_framing(tmp_path: Path) -> None:
    assert model._chat_template(tmp_path) is None
    template = "\n".join(model._CHAT_TEMPLATE_FRAGMENTS)
    (tmp_path / "chat_template.jinja").write_text(template, encoding="utf-8")
    assert model._chat_template(tmp_path) == "laguna"
    altered = template.replace('{{- "<assistant>\\n" -}}', '{{- "<|assistant|>" -}}')
    (tmp_path / "chat_template.jinja").write_text(altered, encoding="utf-8")
    assert model._chat_template(tmp_path) is None
