# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from families.pp_doclayout_v3.model import build, resolve
from tensorrt_model_connect import BuildRequest


@pytest.mark.parametrize(
    "options",
    [
        {"precision": "fp16"},
        {"max_batch_size": 2},
        {"quantization": "fp8"},
        {"task": "object_detection"},
        {"backend": "trt_rtx"},
        {"image_width": 640},
        {"context_parallel_size": 2},
        {"dynamic_kv_cache": True},
    ],
)
def test_unsupported_requests_fail_before_reading_weights(options):
    request = {
        "model_dir": Path("missing"),
        "output_path": Path("unused.bundle"),
        "family": "pp_doclayout_v3",
        "task": "document_layout",
        "precision": "fp32",
    }
    with pytest.raises((ValueError, NotImplementedError)):
        build(BuildRequest(**{**request, **options}), None)


def test_changed_mask_architecture_cannot_silently_use_default_graph(tmp_path):
    config = {"model_type": "pp_doclayout_v3", "mask_enhanced": False}
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(NotImplementedError, match="mask_enhanced"):
        resolve(tmp_path)


@pytest.fixture
def checkpoint(tmp_path):
    config = {
        "model_type": "pp_doclayout_v3",
        "backbone_config": {"model_type": "hgnet_v2"},
        "id2label": {str(i): str(i) for i in range(25)},
    }
    processor = {
        "size": {"height": 800, "width": 800},
        "resample": 3,
        "image_mean": [0, 0, 0],
        "image_std": [1, 1, 1],
        "rescale_factor": 1 / 255,
        "do_resize": True,
        "do_rescale": True,
        "do_normalize": True,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "preprocessor_config.json").write_text(json.dumps(processor))
    return tmp_path


def test_published_topology_and_processor_are_supported(checkpoint):
    assert resolve(checkpoint)["tie_word_embeddings"] is True


def test_untied_decoder_heads_are_rejected(checkpoint):
    path = checkpoint / "config.json"
    config = json.loads(path.read_text())
    config["tie_word_embeddings"] = False
    path.write_text(json.dumps(config))
    with pytest.raises(NotImplementedError, match="tie_word_embeddings"):
        resolve(checkpoint)


@pytest.mark.parametrize("operation", ["do_center_crop", "do_pad"])
def test_unsupported_processor_geometry_is_rejected(checkpoint, operation):
    path = checkpoint / "preprocessor_config.json"
    processor = json.loads(path.read_text())
    processor.update(
        {
            operation: True,
            "crop_size": {"height": 640, "width": 640},
            "pad_size": {"height": 1024, "width": 1024},
        }
    )
    path.write_text(json.dumps(processor))
    with pytest.raises(NotImplementedError, match="preprocessor"):
        resolve(checkpoint)


def test_disabled_processor_geometry_is_supported(checkpoint):
    path = checkpoint / "preprocessor_config.json"
    processor = json.loads(path.read_text())
    processor.update({"do_center_crop": False, "do_pad": False})
    path.write_text(json.dumps(processor))
    assert resolve(checkpoint)["image_size"] == 800
