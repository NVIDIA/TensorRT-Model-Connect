# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-side unit tests for grounding_dino: config, request contract, prompt derivation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from families.grounding_dino import config as config_module
from families.grounding_dino import model as model_module
from families.grounding_dino import text_builder

_MINIMAL = {
    "model_type": "grounding-dino",
    "architectures": ["GroundingDinoForObjectDetection"],
    "two_stage": True,
    "embedding_init_target": True,
    "backbone_config": {"model_type": "swin", "depths": [2, 2, 6, 2],
                        "num_heads": [3, 6, 12, 24], "out_indices": [2, 3, 4]},
    "text_config": {"num_hidden_layers": 12, "num_attention_heads": 12},
    "num_queries": 900,
    "max_text_len": 256,
}


def _model_dir(tmp_path: Path, **overrides) -> Path:
    raw = json.loads(json.dumps(_MINIMAL))
    raw.update(overrides)
    (tmp_path / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    return tmp_path


def test_feature_shapes_start_at_stride_eight() -> None:
    # Starting from the stem instead of stride 8 is what turns a 100x100
    # leading map into a 200x200 one, so the first level is pinned here.
    assert config_module.feature_shapes(800, 4) == [(100, 100), (50, 50), (25, 25), (13, 13)]


def test_resolve_reads_the_checkpoint(tmp_path: Path) -> None:
    cfg = config_module.resolve(_model_dir(tmp_path), image_size=800)
    assert cfg["shapes"] == [(100, 100), (50, 50), (25, 25), (13, 13)]
    assert cfg["max_text_len"] == 256
    assert cfg["num_queries"] == 900
    # Swin-T's geometry is not in the published backbone_config; the nulls there
    # must not become zeros.
    assert cfg["swin"]["patch_size"] == 4
    assert cfg["swin"]["window_size"] == 7


@pytest.mark.parametrize(
    "overrides",
    [
        {"two_stage": False},
        {"embedding_init_target": False},
        {"position_embedding_type": "learned"},
        {"activation_function": "gelu"},
        {"backbone_config": {"model_type": "resnet"}},
    ],
)
def test_resolve_rejects_unsupported_checkpoints(tmp_path: Path, overrides: dict) -> None:
    with pytest.raises(NotImplementedError):
        config_module.resolve(_model_dir(tmp_path, **overrides), image_size=800)


class _Request:
    def __init__(self, **kwargs):
        self.model_dir = kwargs.pop("model_dir", ".")
        self.task = kwargs.pop("task", "image_text_to_boxes")
        self.backend = kwargs.pop("backend", "trt")
        self.precision = kwargs.pop("precision", "fp16")
        self.dynamic_kv_cache = False
        self.max_batch_size = 1
        self.tensor_parallel_size = 1
        self.context_parallel_size = 1
        self.video_num_frames = None
        self.max_sequence_length = None
        self.quantization = None
        self.fp32_layers = ()
        self.mtp_seq_len = None
        self.image_height = 800
        self.image_width = 800
        self.verbose = False
        for key, value in kwargs.items():
            setattr(self, key, value)


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"task": "object_detection"}, ValueError),
        ({"backend": "onnx"}, ValueError),
        ({"precision": "int8"}, ValueError),
        ({"max_batch_size": 4}, NotImplementedError),
        ({"tensor_parallel_size": 2}, NotImplementedError),
        ({"context_parallel_size": 2}, NotImplementedError),
        ({"video_num_frames": 8}, NotImplementedError),
        ({"quantization": "fp8"}, NotImplementedError),
        ({"fp32_layers": (1,)}, NotImplementedError),
        ({"mtp_seq_len": 2}, NotImplementedError),
        ({"dynamic_kv_cache": True}, NotImplementedError),
        ({"image_height": 800, "image_width": 1333}, NotImplementedError),
        ({"image_height": 801, "image_width": 801}, ValueError),
    ],
)
def test_build_rejects_unsupported_requests(overrides: dict, error: type) -> None:
    with pytest.raises(error):
        model_module.build(_Request(**overrides), writer=None)


def test_phrase_structure_matches_the_reference() -> None:
    # [CLS] a cat . a remote control . [SEP]
    ids = [101, 1037, 4937, 1012, 1037, 6556, 2491, 1012, 102]
    block, positions = text_builder.block_masks_and_positions(ids)
    assert positions.tolist() == [0, 0, 1, 0, 0, 1, 2, 0, 0]
    # Inside a phrase, yes; across phrases, no; special tokens stand alone.
    assert block[1][2] and block[2][1]
    assert not block[1][5] and not block[5][1]
    assert block[0].tolist() == [True] + [False] * 8
    assert block[8].tolist() == [False] * 8 + [True]


def test_padding_is_masked_and_the_diagonal_survives() -> None:
    ids = [101, 1037, 4937, 1012, 102]
    block, _ = text_builder.block_masks_and_positions(ids)
    padded = np.zeros((8, 8), dtype=bool)
    padded[:5, :5] = block
    attention = np.array([1, 1, 1, 1, 1, 0, 0, 0])
    additive = text_builder.self_attention_mask(padded, attention)
    assert additive.shape == (1, 8, 8)
    # A fully masked softmax row is NaN; every row keeps its own diagonal.
    assert np.all(np.diagonal(additive[0]) == 0.0)
    assert additive[0][0][5] < 0.0 and additive[0][5][0] < 0.0
    keys = text_builder.key_mask(attention)
    assert keys.shape == (1, 1, 8)
    assert np.all(keys[0, 0, :5] == 0.0) and np.all(keys[0, 0, 5:] < 0.0)
