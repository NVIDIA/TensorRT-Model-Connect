# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit checks for the Deformable DETR builders that need no GPU."""

from __future__ import annotations

import json

import numpy as np
import pytest


def test_feature_shapes_start_at_stride_eight():
    from families.deformable_detr import config

    shapes = config.feature_shapes(800, 4)
    # Measured against the reference pyramid. Starting from the stem would put
    # a 200x200 map at the front and give the encoder 40000 extra tokens.
    assert shapes == [(100, 100), (50, 50), (25, 25), (13, 13)]
    assert sum(h * w for h, w in shapes) == 13294


def test_feature_shapes_round_the_extra_level_up():
    from families.deformable_detr import config

    # 25 halves to 13, not 12: the extra level is a stride-2 convolution with
    # padding, so it rounds up.
    assert config.feature_shapes(800, 4)[-1] == (13, 13)


def test_feature_shapes_round_every_strided_convolution_up():
    from families.deformable_detr import config

    assert config.feature_shapes(1333, 4) == [(167, 167), (84, 84), (42, 42), (21, 21)]


@pytest.mark.parametrize("size,expected", [
    ({"shortest_edge": 800, "longest_edge": 1333}, (800, 1333)),
    (800, (800, 1333)),
    ({"shortest_edge": 600, "longest_edge": 1000}, (800, 1333)),
])
def test_resolve_preserves_checkpoint_resize_ratio(tmp_path, size, expected):
    from families.deformable_detr import config

    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "preprocessor_config.json").write_text(json.dumps({"size": size}))
    resolved = config.resolve(tmp_path, image_size=800)
    assert (resolved["resize_shortest_edge"], resolved["resize_longest_edge"]) == expected


@pytest.mark.parametrize("size", [
    {"shortest_edge": 0, "longest_edge": 1333},
    {"shortest_edge": 800, "longest_edge": 700},
])
def test_resolve_rejects_invalid_resize_bounds(tmp_path, size):
    from families.deformable_detr import config

    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "preprocessor_config.json").write_text(json.dumps({"size": size}))
    with pytest.raises(ValueError, match="resize bounds"):
        config.resolve(tmp_path, image_size=800)


def test_sine_embedding_matches_its_declared_shape():
    from families.deformable_detr import position_builder

    embedding = position_builder.sine_embedding(4, 6)
    assert embedding.shape == (1, 24, 256)
    # The y half comes first, so two tokens sharing a row share their first
    # 128 features and differ in the rest.
    assert np.allclose(embedding[0, 0, :128], embedding[0, 1, :128], atol=1e-6)
    assert not np.allclose(embedding[0, 0, 128:], embedding[0, 1, 128:], atol=1e-6)


def test_reference_points_are_level_centres():
    from families.deformable_detr import encoder_builder

    reference = encoder_builder.reference_points([(2, 2), (1, 1)])
    assert reference.shape == (1, 5, 2, 2)
    # First token of a 2x2 level sits at (0.25, 0.25); the single token of a
    # 1x1 level sits at the middle.
    assert np.allclose(reference[0, 0, 0], [0.25, 0.25], atol=1e-6)
    assert np.allclose(reference[0, 4, 0], [0.5, 0.5], atol=1e-6)
    # Every level shares the point, which is what valid ratios of 1 reduce to.
    assert np.allclose(reference[0, 0, 0], reference[0, 0, 1], atol=1e-6)


def test_resolve_rejects_two_stage(tmp_path):
    from families.deformable_detr import config

    (tmp_path / "config.json").write_text(json.dumps({"two_stage": True}), encoding="utf-8")
    with pytest.raises(NotImplementedError):
        config.resolve(tmp_path, image_size=800)


def test_resolve_rejects_box_refinement(tmp_path):
    from families.deformable_detr import config

    (tmp_path / "config.json").write_text(
        json.dumps({"with_box_refine": True}), encoding="utf-8")
    with pytest.raises(NotImplementedError):
        config.resolve(tmp_path, image_size=800)


def test_resolve_reads_the_normalisation_statistics(tmp_path):
    from families.deformable_detr import config

    (tmp_path / "config.json").write_text(json.dumps({"num_labels": 91}), encoding="utf-8")
    (tmp_path / "preprocessor_config.json").write_text(
        json.dumps({"do_normalize": True, "image_mean": [0.1, 0.2, 0.3],
                    "image_std": [0.4, 0.5, 0.6]}),
        encoding="utf-8",
    )
    resolved = config.resolve(tmp_path, image_size=800)
    # Unlike RT-DETR's, this checkpoint's preprocessor really does normalise.
    assert resolved["do_normalize"] is True
    assert resolved["image_mean"] == [0.1, 0.2, 0.3]
    assert resolved["image_std"] == [0.4, 0.5, 0.6]


def test_inverse_sigmoid_round_trips():
    from families.deformable_detr import decoder_builder

    values = np.array([[0.1, 0.5, 0.9]])
    recovered = 1.0 / (1.0 + np.exp(-decoder_builder._inverse_sigmoid(values)))
    assert np.allclose(recovered, values, atol=1e-6)


def test_checkpoint_renames_cover_every_builder_name():
    from families.deformable_detr import checkpoint

    # Measured by diffing the checkpoint file's keys against a loaded model's.
    # transformers renames tensors on load, so a graph verified against
    # state_dict() proves nothing about whether a bundle can be built.
    cases = {
        "model.backbone.conv_encoder.model.conv1.weight":
            "model.backbone.model.conv1.weight",
        "model.backbone.conv_encoder.model.layer2.0.downsample.0.weight":
            "model.backbone.model.layer2.0.downsample.0.weight",
        "model.encoder.layers.3.fc1.weight": "model.encoder.layers.3.mlp.fc1.weight",
        "model.decoder.layers.5.fc2.bias": "model.decoder.layers.5.mlp.fc2.bias",
        "model.decoder.layers.0.self_attn.out_proj.weight":
            "model.decoder.layers.0.self_attn.o_proj.weight",
    }
    for stored, expected in cases.items():
        assert checkpoint.canonical_name(stored) == expected


def test_checkpoint_leaves_unrenamed_names_alone():
    from families.deformable_detr import checkpoint

    for name in ("model.level_embed", "class_embed.0.weight",
                 "model.encoder.layers.0.self_attn.value_proj.weight",
                 "model.input_proj.3.0.weight"):
        assert checkpoint.canonical_name(name) == name


def test_checkpoint_rejects_a_rename_collision():
    from families.deformable_detr import checkpoint

    with pytest.raises(KeyError):
        checkpoint.normalise({
            "model.encoder.layers.0.fc1.weight": 1,
            "model.encoder.layers.0.mlp.fc1.weight": 2,
        })
