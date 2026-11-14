# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks of the configuration contract and of the builder's request policy."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from families.oneformer.config import load_config


def _config() -> dict:
    """The fields of shi-labs/oneformer_cityscapes_swin_large that the family reads."""
    return {
        "model_type": "oneformer",
        "architectures": ["OneFormerForUniversalSegmentation"],
        "backbone_config": {
            "model_type": "swin",
            "embed_dim": 192,
            "depths": [2, 2, 18, 2],
            "num_heads": [6, 12, 24, 48],
            "window_size": 12,
            "patch_size": 4,
            "mlp_ratio": 4.0,
            "hidden_act": "gelu",
            "qkv_bias": True,
            "use_absolute_embeddings": False,
            "layer_norm_eps": 1e-5,
            "out_features": ["stage1", "stage2", "stage3", "stage4"],
        },
        "conv_dim": 256,
        "mask_dim": 256,
        "hidden_dim": 256,
        "num_attention_heads": 8,
        "encoder_layers": 6,
        "encoder_feedforward_dim": 1024,
        "decoder_layers": 10,
        "dim_feedforward": 2048,
        "query_dec_layers": 2,
        "num_queries": 250,
        "num_classes": 3,
        "task_seq_len": 77,
        "layer_norm_eps": 1e-5,
        "strides": [4, 8, 16, 32],
        "common_stride": 4,
        "norm": "GN",
        "pre_norm": False,
        "use_task_norm": True,
        "enforce_input_proj": False,
        "is_training": False,
        "id2label": {"0": "road", "1": "sidewalk", "2": "building"},
    }


def _preprocessor() -> dict:
    return {
        "do_resize": True,
        "do_rescale": True,
        "do_normalize": True,
        "resample": 2,
        "size": {"shortest_edge": 1024, "longest_edge": 2048},
        "image_mean": [0.48500001430511475, 0.4560000002384186, 0.4059999883174896],
        "image_std": [0.2290000021457672, 0.2239999920129776, 0.22499999403953552],
        "rescale_factor": 0.00392156862745098,
    }


def _write(root: Path, config: dict, preprocessor: dict | None = None) -> Path:
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "preprocessor_config.json").write_text(
        json.dumps(preprocessor or _preprocessor()), encoding="utf-8"
    )
    return root


def test_reads_the_cityscapes_swin_large_layout(tmp_path: Path) -> None:
    config = load_config(_write(tmp_path, _config()))
    assert config.swin.stage_dims() == (192, 384, 768, 1536)
    assert config.swin.window_size == 12 and config.swin.depths == (2, 2, 18, 2)
    assert (
        config.num_queries == 250 and config.decoder_layers == 10 and config.query_dec_layers == 2
    )
    assert config.class_names == ("road", "sidewalk", "building")
    assert (config.preprocess.shortest_edge, config.preprocess.longest_edge) == (1024, 2048)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("backbone_config", "model_type"), "dinat", "only Swin backbones"),
        (("backbone_config", "use_absolute_embeddings"), True, "absolute position"),
        (("pre_norm",), True, "pre-norm"),
        (("is_training",), True, "training checkpoints"),
        (("hidden_dim",), 512, "input projections must be identities"),
        (("common_stride",), 8, "common_stride"),
        (("num_classes",), 19, "num_classes must match"),
    ],
)
def test_rejects_layouts_the_graph_does_not_build(tmp_path: Path, path, value, message) -> None:
    raw = _config()
    target = raw
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=message):
        load_config(_write(tmp_path, raw))


def test_rejects_a_non_bilinear_processor(tmp_path: Path) -> None:
    preprocessor = _preprocessor() | {"resample": 3}
    with pytest.raises(ValueError, match="bilinearly"):
        load_config(_write(tmp_path, _config(), preprocessor))


def test_builder_rejects_unsupported_requests_before_reading_weights(tmp_path: Path) -> None:
    pytest.importorskip("tensorrt")
    from families.oneformer.model import build
    from tensorrt_model_connect import BuildRequest

    base = BuildRequest(
        model_dir=_write(tmp_path, _config()),
        output_path=tmp_path / "x.bundle",
        family="oneformer",
        task="image_to_semantic_segmentation",
        precision="fp32",
    )
    for request, message in (
        (replace(base, task="image_to_boxes"), "only task"),
        (replace(base, precision="fp16"), "precision"),
        (replace(base, precision="int8"), "precision"),
        (replace(base, max_batch_size=2), "max_batch_size"),
        (replace(base, tensor_parallel_size=2), "parallelism"),
        (replace(base, quantization="fp8"), "quantization"),
        (replace(base, image_height=16), "at least"),
    ):
        with pytest.raises((ValueError, NotImplementedError), match=message):
            build(request, writer=None)
