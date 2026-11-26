# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The builder rejects a checkpoint it cannot bundle before it builds any engine."""

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("numpy")
pytest.importorskip("tensorrt")

from families.minicpmv4_6 import model  # noqa: E402
from families.minicpmv4_6.tests.tokenizer_fixture import minimal_tokenizer  # noqa: E402


def _request(model_dir):
    return SimpleNamespace(
        task="images_text_to_text", backend="trt", precision="fp16", tensor_parallel_size=1,
        context_parallel_size=1, max_batch_size=1, verbose=False, quantization=None, fp32_layers=(),
        dynamic_kv_cache=False, mtp_seq_len=None, image_height=None, image_width=None, video_num_frames=None,
        model_dir=model_dir, max_sequence_length=None,
    )


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    """A checkpoint whose configuration passes; every step from the image metadata on must not run.

    Its tokenizer has 259 tokens (the last one ends a sequence) and its embedding table 320 rows.
    """
    text = SimpleNamespace(max_position_embeddings=4096, rms_norm_eps=1e-6, rotary_dim=64, rope_theta=1e7,
                           vocab_size=320)
    config = SimpleNamespace(text=text, eos_token_id=258)
    monkeypatch.setattr(model.CheckpointConfig, "from_dir", staticmethod(lambda path: config))

    def too_late(*args, **kwargs):
        raise AssertionError("the build went past its checks")

    for name in ("image_metadata", "_open_safetensors", "build_text_engine", "build_vision_engine"):
        monkeypatch.setattr(model, name, too_late)
    for name in model._BUNDLE_FILES:
        (tmp_path / name).write_text("{}", encoding="utf-8")
    (tmp_path / "tokenizer.json").write_text(json.dumps(minimal_tokenizer()), encoding="utf-8")
    return tmp_path


def test_invalid_end_of_sequence_ids_stop_the_build_before_any_engine(checkpoint):
    (checkpoint / "generation_config.json").write_text(json.dumps({"eos_token_id": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="eos_token_id"):
        model.build(_request(checkpoint), writer=SimpleNamespace())


def test_an_end_id_beyond_the_tokenizer_stops_the_build_before_any_engine(checkpoint):
    # Row 300 exists in the embedding table, but generation samples only the tokenizer's 259 ids.
    (checkpoint / "generation_config.json").write_text(json.dumps({"eos_token_id": [258, 300]}), encoding="utf-8")
    with pytest.raises(ValueError, match="eos_token_id"):
        model.build(_request(checkpoint), writer=SimpleNamespace())


@pytest.mark.parametrize("name", model._BUNDLE_FILES)
def test_a_missing_bundle_file_stops_the_build_before_any_engine(checkpoint, name):
    (checkpoint / name).unlink()
    with pytest.raises(FileNotFoundError, match=name):
        model.build(_request(checkpoint), writer=SimpleNamespace())


def test_a_checkpoint_that_passes_reaches_the_image_metadata(checkpoint):
    with pytest.raises(AssertionError, match="went past its checks"):
        model.build(_request(checkpoint), writer=SimpleNamespace())
