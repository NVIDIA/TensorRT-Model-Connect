# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unsupported public requests must fail before assets or GPU graphs are touched."""

from dataclasses import replace
import json
from pathlib import Path

import pytest

from tensorrt_model_connect.build import BuildRequest

from families.lighton_ocr import model
from families.lighton_ocr.config import Config


def request(tmp_path):
    return BuildRequest(
        model_dir=tmp_path / "unavailable-checkpoint", output_path=tmp_path / "model.bundle",
        family="lighton_ocr", task="images_text_to_text", precision="fp32",
    )


@pytest.mark.parametrize("changes", [
    {"family": "other"},
    {"task": "text_generation"},
    {"backend": "trt_rtx"},
    {"precision": "fp16"},
    {"precision": "bf16"},
    {"max_batch_size": 2},
    {"max_batch_size": True},
    {"tensor_parallel_size": 2},
    {"context_parallel_size": 2},
    {"quantization": "int8"},
    {"fp32_layers": (0,)},
    {"image_height": 28},
    {"image_width": 28},
    {"video_num_frames": 1},
    {"mtp_seq_len": 2},
    {"graph_transform": lambda graph: graph},
])
def test_unsupported_request_fails_without_opening_the_checkpoint(tmp_path, monkeypatch, changes):
    def assets_must_not_be_opened(*_args):
        raise AssertionError("Unsupported requests must fail before checkpoint loading")

    monkeypatch.setattr(model.Config, "from_dir", assets_must_not_be_opened)
    with pytest.raises(ValueError):
        model.build(replace(request(tmp_path), **changes), object())


def test_runtime_sized_kv_cache_is_rejected_before_loading_any_assets(tmp_path, monkeypatch):
    def assets_must_not_be_opened(*_args):
        raise AssertionError("Unsupported runtime-sized caches must fail before checkpoint loading")

    monkeypatch.setattr(model.Config, "from_dir", assets_must_not_be_opened)
    with pytest.raises(NotImplementedError, match="^lighton_ocr does not support dynamic_kv_cache$"):
        model.build(replace(request(tmp_path), dynamic_kv_cache=True), object())


@pytest.mark.parametrize("capacity", [1, True, 512.5, 16385])
def test_invalid_capacity_fails_before_tokenizer_or_weight_loading(tmp_path, monkeypatch, capacity):
    config = Config.from_dict(json.loads((Path(__file__).parent / "data/checkpoint-config.json").read_text()))
    monkeypatch.setattr(model.Config, "from_dir", lambda _directory: config)

    def assets_must_not_be_opened(*_args):
        raise AssertionError("Invalid capacity must fail before runtime assets are loaded")

    monkeypatch.setattr(model, "runtime_config", assets_must_not_be_opened)
    with pytest.raises(ValueError, match="sequence capacity"):
        model.build(replace(request(tmp_path), max_sequence_length=capacity), object())


@pytest.mark.parametrize("tokenizer_kind", ["missing", "directory"])
def test_missing_tokenizer_fails_before_runtime_assets_weights_or_bundle_output(tmp_path, monkeypatch, tokenizer_kind):
    directory = tmp_path / "checkpoint"
    directory.mkdir()
    data = Path(__file__).parent / "data"
    (directory / "config.json").write_bytes((data / "checkpoint-config.json").read_bytes())
    if tokenizer_kind == "directory":
        (directory / "tokenizer.json").mkdir()

    def assets_must_not_be_opened(*_args):
        raise AssertionError("A missing tokenizer must fail before runtime assets or weights are loaded")

    monkeypatch.setattr(model, "runtime_config", assets_must_not_be_opened)
    monkeypatch.setattr(model, "Checkpoint", assets_must_not_be_opened)
    with pytest.raises(ValueError, match="^lighton_ocr requires tokenizer.json in the checkpoint$"):
        model.build(replace(request(tmp_path), model_dir=directory), object())


@pytest.mark.parametrize(("capacity", "output_budget"), [
    (512, 512), (1023, 1023), (1024, 1024), (1025, 1024), (4096, 1024),
])
def test_runtime_metadata_retains_original_preprocessing_and_generation(tmp_path, monkeypatch, capacity, output_budget):
    data = Path(__file__).parent / "data"
    config = Config.from_dict(json.loads((data / "checkpoint-config.json").read_text()))
    (tmp_path / "processor_config.json").write_bytes((data / "processor-config.json").read_bytes())
    (tmp_path / "generation_config.json").write_bytes((data / "generation-config.json").read_bytes())
    monkeypatch.setattr(model, "_prompt_tokens", lambda _directory, _config: ([1, 3], [4, 5]))
    runtime = model.runtime_config(tmp_path, config, capacity)
    assert runtime["precision"] == "fp32" and runtime["max_sequence_length"] == capacity
    assert runtime["max_new_tokens"] == output_budget
    assert 0 < runtime["max_new_tokens"] <= runtime["max_sequence_length"]
    assert runtime["num_key_value_heads"] == 8 and runtime["head_dim"] == 128
    assert runtime["prompt_prefix_ids"] == [1, 3] and runtime["prompt_suffix_ids"] == [4, 5]
    assert runtime["image_processor"]["patch_size"] == 14
    assert runtime["image_processor"]["spatial_merge_size"] == 2
    assert runtime["image_processor"]["max_image_size"] == 1540
    assert runtime["image_processor"]["image_mean"] == [0.48145466, 0.4578275, 0.40821073]
    assert runtime["image_processor"]["image_std"] == [0.26862954, 0.26130258, 0.27577711]


@pytest.mark.parametrize("mutation", ["interpolation", "geometry", "normalization"])
def test_changed_processor_semantics_are_rejected(tmp_path, mutation):
    data = Path(__file__).parent / "data"
    config = Config.from_dict(json.loads((data / "checkpoint-config.json").read_text()))
    raw = json.loads((data / "processor-config.json").read_text())
    processor = raw["image_processor"]
    if mutation == "interpolation":
        processor["resample"] = 2
    elif mutation == "geometry":
        processor["patch_size"] = 16
    else:
        processor["image_std"][0] = 0
    (tmp_path / "processor_config.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        model._processor_config(tmp_path, config)
