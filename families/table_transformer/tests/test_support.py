# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free exact checkpoint ownership and default Task."""

import json
import pytest
from families.table_transformer.config import ModelConfig
from families.table_transformer.support import describe, STRUCTURE_LABELS
from tensorrt_model_connect.model_support import ModelMetadata


def structure_config():
    return {
        "model_type": "table-transformer",
        "architectures": ["TableTransformerForObjectDetection"],
        "backbone": "resnet18",
        "num_queries": 125,
        "id2label": dict(STRUCTURE_LABELS),
    }


def test_structure_checkpoint_identity():
    config = structure_config()
    support = describe(ModelMetadata(config=config, model_index={}))
    assert support is not None
    assert support.tasks == ("image_to_boxes",)
    assert support.default_task == "image_to_boxes"
    assert ModelConfig.from_json(json.dumps(config)).model_type == "table-transformer"


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"model_type": "detr"},
        {"model_type": "table-transformer-other"},
        {"architectures": ["DetrForObjectDetection"]},
        {"model_type": "table-transformer"},
        {"architectures": ["TableTransformerForObjectDetection"]},
        {**structure_config(), "num_queries": 15, "id2label": {"0": "table", "1": "table rotated"}},
        {**structure_config(), "backbone": "resnet50"},
        {**structure_config(), "dilation": True},
    ],
)
def test_other_checkpoints_not_claimed(config):
    assert describe(ModelMetadata(config=config, model_index={})) is None
    with pytest.raises(ValueError):
        ModelConfig.from_json(json.dumps(config))


def test_detection_checkpoint_rejected_before_weights(tmp_path, monkeypatch):
    from tensorrt_model_connect import BuildRequest
    from families.table_transformer.model import build, _TableTransformerModel

    config = {
        **structure_config(),
        "num_queries": 15,
        "id2label": {"0": "table", "1": "table rotated"},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "preprocessor_config.json").write_text(json.dumps({"size": 800, "max_size": 800}))
    monkeypatch.setattr(
        _TableTransformerModel, "load_weights", lambda *a, **kw: pytest.fail("read weights")
    )
    assert describe(ModelMetadata(config=config, model_index={})) is None
    with pytest.raises(ValueError, match="structure checkpoint"):
        build(
            BuildRequest(
                model_dir=tmp_path,
                output_path=tmp_path / "model.bundle",
                family="table_transformer",
                task="image_to_boxes",
                precision="fp32",
            ),
            None,
        )
