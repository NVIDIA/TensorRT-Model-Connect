# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import shutil
import json
import pytest
from tensorrt_model_connect.build import BuildRequest
from families.d3.config import read_config


@pytest.fixture
def build_request(tmp_path):
    fixtures = Path(__file__).parent / "fixtures"
    for name in ("config.json", "decision_config.json", "preprocessor_config.json"):
        shutil.copy(fixtures / name, tmp_path / name)
    (tmp_path / "readout.safetensors").touch()
    return BuildRequest(tmp_path, tmp_path / "test.bundle", "d3", "structured_decision", "bf16")


def test_checkpoint_defaults(build_request):
    c, d, maximum, height, width = read_config(build_request)
    assert (maximum, height, width) == (2048, 256, 256)
    assert d["readout_dtype"] == "float32"
    assert c["text_config"]["layer_types"].count("full_attention") == 6


@pytest.mark.parametrize("field,value", [("precision", "fp16"), ("max_batch_size", 9),
    ("tensor_parallel_size", 2), ("context_parallel_size", 2), ("quantization", "fp8"),
    ("dynamic_kv_cache", True), ("video_num_frames", 2), ("fp32_layers", (0,)),
    ("mtp_seq_len", 2), ("image_height", 250)])
def test_reject_unimplemented_options(build_request, field, value):
    from dataclasses import replace
    with pytest.raises(ValueError):
        read_config(replace(build_request, **{field: value}))


@pytest.mark.parametrize("field,value", [("attention_mode", "prefix_bidir"),
    ("prompt_layout", "header-v1"), ("temperature", 0), ("readout_dtype", "bfloat16")])
def test_reject_different_decision_semantics(build_request, field, value):
    path = build_request.model_dir / "decision_config.json"
    d = json.loads(path.read_text())
    d[field] = value
    path.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        read_config(build_request)


@pytest.mark.parametrize("batch", [1, 2, 4, 8])
def test_batch_profiles(build_request, batch):
    from dataclasses import replace
    read_config(replace(build_request, max_batch_size=batch))


@pytest.mark.parametrize("name", ["edge", "lite", "nano", "mini", "flash"])
def test_released_family_profiles(build_request, name):
    from families.d3.support import profile_name
    fixtures = Path(__file__).parent / "fixtures/profiles" / name
    for path in fixtures.iterdir():
        shutil.copy(path, build_request.model_dir / path.name)
    config, decision, *_ = read_config(build_request)
    assert profile_name(config) == name
    assert config["vision_config"]["out_hidden_size"] == config["text_config"]["hidden_size"]
    assert decision["format_id"] == "d3-code-readout-v1"


def test_unknown_hybrid_topology_is_rejected(build_request):
    path = build_request.model_dir / "config.json"
    config = json.loads(path.read_text())
    config["text_config"]["linear_num_value_heads"] = 32
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        read_config(build_request)
