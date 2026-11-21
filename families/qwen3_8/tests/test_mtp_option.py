# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MTP engines are built only when the user passes --mtp-seq-len."""

import json
from types import SimpleNamespace

import pytest

from families.qwen3_8 import model
from families.qwen3_8.build_request import BuildRequest


def _source(tmp_path):
    source = tmp_path / "target"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5", "text_config": {"output_gate_type": "sigmoid"},
    }))
    return source


def test_mtp_is_off_by_default_even_when_the_checkpoint_has_mtp_weights() -> None:
    assert model._mtp_enabled(SimpleNamespace(mtp_seq_len=None), {"mtp_layer.fc": object()}) is False


def test_mtp_seq_len_turns_mtp_on() -> None:
    assert model._mtp_enabled(SimpleNamespace(mtp_seq_len=2), {"mtp_layer.fc": object()}) is True


def test_mtp_seq_len_without_mtp_weights_is_an_error() -> None:
    with pytest.raises(ValueError, match="no MTP weights"):
        model._mtp_enabled(SimpleNamespace(mtp_seq_len=2), {})


def test_mtp_seq_len_must_be_at_least_two(tmp_path) -> None:
    with pytest.raises(ValueError, match="mtp_seq_len"):
        BuildRequest(model_dir=tmp_path, output_path=tmp_path / "o", family="qwen3_8", task="text_generation",
                     precision="fp16", mtp_seq_len=1)


@pytest.mark.parametrize("options, expected", [([], None), (["--mtp-seq-len", "3"], 3)])
def test_cli_passes_mtp_seq_len_to_the_request(tmp_path, monkeypatch, options, expected) -> None:
    from tensorrt_model_connect import family_cli as build_cli
    from families.qwen3_8 import cli as owner

    seen = []
    monkeypatch.setattr(owner, "build_bundle", lambda request, output: seen.append(request))
    assert build_cli.main(["qwen3_8", "build", str(_source(tmp_path)), "-o", str(tmp_path / "out"), *options]) == 0
    assert seen[0].mtp_seq_len == expected
