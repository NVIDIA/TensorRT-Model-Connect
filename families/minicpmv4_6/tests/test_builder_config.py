# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks of the builder's checkpoint-configuration rules (no TensorRT, no checkpoint)."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from families.minicpmv4_6 import config as cfg
from families.minicpmv4_6 import prompt

_PROCESSOR = {
    "image_processor_type": "MiniCPMV4_6ImageProcessor",
    "processor_class": "MiniCPMV4_6Processor",
    "max_slice_nums": 9,
    "scale_resolution": 448,
    "patch_size": 14,
    "use_image_id": True,
    "slice_mode": True,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
}


_VISION = {
    "model_type": "minicpmv4_6_vision", "hidden_act": "gelu_pytorch_tanh", "num_channels": 3, "image_size": 980,
    "patch_size": 14, "hidden_size": 1152, "intermediate_size": 4304, "num_hidden_layers": 27,
    "num_attention_heads": 16, "layer_norm_eps": 1e-06,
}


def _write(tmp_path, name, value):
    path = tmp_path / name
    path.write_text(value if isinstance(value, str) else json.dumps(value), encoding="utf-8")
    return path


@pytest.mark.parametrize(("value", "expected"), [
    (True, 1), (1, True), (0, False), (3.0, 3), ([2.0, 2], [2, 2]), ([2, 2, 2], [2, 2]), (None, "16x"),
])
def test_a_required_value_must_also_have_the_json_type(value, expected):
    with pytest.raises(ValueError, match="requires"):
        cfg._require({"key": value}, "key", expected)


def test_a_required_value_may_be_restated_exactly_or_left_out():
    cfg._require({"key": [2, 2]}, "key", [2, 2])
    cfg._require({}, "key", "16x", default="16x")


@pytest.mark.parametrize("value", [math.nan, math.inf, 1e-50, 1e39, 0, -1e-6, True, "1e-6", None])
def test_a_config_number_must_be_positive_and_fit_fp32(value):
    with pytest.raises(ValueError, match="FP32"):
        cfg._number({"eps": value}, "eps")


def test_a_config_number_is_accepted_as_a_float():
    assert cfg._number({"eps": 1e-6}, "eps") == 1e-6
    assert cfg._number({"theta": 10000000}, "theta") == 1e7


def test_a_special_token_id_may_be_zero():
    assert cfg._token_id({"id": 0}, "id", 10) == 0
    assert cfg._token_id({"id": 9}, "id", 10) == 9


@pytest.mark.parametrize("value", [-1, 10, 1.0, True, None])
def test_a_special_token_id_must_address_the_vocabulary(value):
    with pytest.raises(ValueError, match="token id"):
        cfg._token_id({"id": value}, "id", 10)


@pytest.mark.parametrize("value", [2**31, 1.0, True, 0, None])
def test_a_config_integer_must_be_a_positive_int32(value):
    with pytest.raises(ValueError, match="32-bit"):
        cfg._integer({"size": value}, "size")


@pytest.mark.parametrize("text", ['{"a": 1, "a": 2}', '{"a": NaN}', '{"a": Infinity}', '{"a": -Infinity}'])
def test_checkpoint_json_rejects_repeated_keys_and_non_numbers(tmp_path, text):
    with pytest.raises(ValueError):
        cfg.load_json(_write(tmp_path, "config.json", text))


def test_an_fp16_epsilon_must_not_vanish():
    assert cfg.positive_when_stored(1e-6, "e")
    assert not cfg.positive_when_stored(1e-8, "e")
    assert cfg.positive_when_stored(1e-8, "f")
    assert not cfg.positive_when_stored(1e39, "f")


@pytest.mark.parametrize("requested", [0, 1, True, 4096.0, 262145])
def test_an_invalid_capacity_is_rejected_not_replaced_by_the_default(requested):
    # 0 and other falsy values are not "unset"; only None is.
    with pytest.raises(ValueError, match="max_sequence_length"):
        cfg.capacity_rows(requested, 262144)


def test_a_defaulted_capacity_must_hold_two_rows():
    # A checkpoint of one position would default to a one-row cache, which the runtime refuses.
    with pytest.raises(ValueError, match="max_sequence_length"):
        cfg.capacity_rows(None, 1)


def test_the_capacity_defaults_only_when_unset():
    assert cfg.capacity_rows(None, 262144) == cfg.DEFAULT_CAPACITY
    assert cfg.capacity_rows(None, 1024) == 1024
    assert cfg.capacity_rows(2, 262144) == 2


def _checkpoint(vocab_size=248320, eos_token_id=248044):
    return SimpleNamespace(text=SimpleNamespace(vocab_size=vocab_size), eos_token_id=eos_token_id)


@pytest.mark.parametrize("generation", [
    {"eos_token_id": [248046, 248046]}, {"eos_token_id": None}, {"eos_token_id": [248320]},
    {"eos_token_id": [1.0]}, {"eos_token_id": True}, {"eos_token_id": []}, [248046],
])
def test_generation_end_ids_must_be_distinct_token_ids(tmp_path, generation):
    _write(tmp_path, "generation_config.json", generation)
    with pytest.raises(ValueError, match="eos_token_id"):
        cfg.generation_ids(tmp_path, _checkpoint())


def test_generation_end_ids_default_to_the_config(tmp_path):
    assert cfg.generation_ids(tmp_path, _checkpoint()) == [248044]
    _write(tmp_path, "generation_config.json", {"eos_token_id": 248046})
    assert cfg.generation_ids(tmp_path, _checkpoint()) == [248046]
    _write(tmp_path, "generation_config.json", {"eos_token_id": [248046, 248044]})
    assert cfg.generation_ids(tmp_path, _checkpoint()) == [248046, 248044]


def test_the_checkpoint_processor_settings_are_accepted(tmp_path):
    _write(tmp_path, "preprocessor_config.json", {**_PROCESSOR, "do_resize": True, "rescale_factor": 1 / 255,
                                                   "resample": 3, "downsample_mode": "16x"})
    assert prompt.processor_settings(tmp_path, 14, 4096) == {
        "scale_resolution": 448, "max_slice_nums": 9, "mean": [0.5] * 3, "stddev": [0.5] * 3}


@pytest.mark.parametrize("change", [
    {"do_rescale": False}, {"do_normalize": False}, {"do_resize": False}, {"do_convert_rgb": False},
    {"rescale_factor": 0.5}, {"resample": 2}, {"resample": 3.0}, {"slice_mode": 1}, {"use_image_id": False},
    {"downsample_mode": "4x"}, {"crop_size": 448}, {"patch_size": 16}, {"max_slice_nums": 0},
    {"max_slice_nums": prompt.MAX_SLICE_NUMS + 1}, {"max_slice_nums": 2**31 - 1},
    {"scale_resolution": 28}, {"scale_resolution": 448.0}, {"scale_resolution": 1000000},
    {"image_mean": [0.5, 0.5]},
    {"image_std": [0.5, 0.5, 0.0]}, {"image_std": [0.5, 0.5, True]}, {"image_std": [1e-40, 0.5, 0.5]},
    {"image_mean": [1e38, 0.5, 0.5]}, {"image_std": [0.5, 0.5, -0.5]},
])
def test_a_processor_option_the_runtime_cannot_honor_is_rejected(tmp_path, change):
    _write(tmp_path, "preprocessor_config.json", {**_PROCESSOR, **change})
    with pytest.raises(ValueError, match="MiniCPM-V 4.6"):
        prompt.processor_settings(tmp_path, 14, 4096)


def test_the_largest_supported_slice_count_is_accepted(tmp_path):
    _write(tmp_path, "preprocessor_config.json", {**_PROCESSOR, "max_slice_nums": prompt.MAX_SLICE_NUMS})
    assert prompt.processor_settings(tmp_path, 14, 4096)["max_slice_nums"] == 4096


def test_a_missing_processor_setting_is_rejected(tmp_path):
    _write(tmp_path, "preprocessor_config.json", {k: v for k, v in _PROCESSOR.items() if k != "image_std"})
    with pytest.raises(ValueError, match="must set image_std"):
        prompt.processor_settings(tmp_path, 14, 4096)


@pytest.mark.parametrize(("vocabulary", "total"), [
    ({"a": 0, "b": 248320}, 2), ({"a": 0, "b": 1}, 248321), ({"a": 0, "b": 0}, 2), ({}, 0), ({"a": -1}, 1),
])
def test_token_ids_must_address_embedding_rows(vocabulary, total):
    with pytest.raises(ValueError, match="embedding table"):
        prompt.check_vocabulary(vocabulary, total, 248320)


def test_a_vocabulary_within_the_embedding_table_is_accepted():
    prompt.check_vocabulary({"a": 0, "b": 248319}, 248077, 248320)


def test_the_checkpoint_vision_config_is_accepted():
    assert cfg.VisionConfig.from_dict(dict(_VISION), 6).positions_per_side == 70


def test_a_position_grid_of_one_row_is_rejected():
    # The runtime buckets patch positions over at least two rows of the position grid.
    with pytest.raises(ValueError, match="at least two patches"):
        cfg.VisionConfig.from_dict({**_VISION, "image_size": 14}, 6)


def test_same_value_compares_json_types_inside_objects():
    assert cfg.same_value({"a": [True, "x"]}, {"a": [True, "x"]})
    assert not cfg.same_value({"a": 1}, {"a": True})
    assert not cfg.same_value({"a": True, "b": None}, {"a": True})


def _tokenizer():
    """The smallest tokenizer.json of the layout the native tokenizer loads (as the host test's)."""
    vocab = {character: index for index, character in enumerate(prompt._byte_alphabet())}
    vocab.update({"xy": 256, "xyx": 257})
    byte_level = {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False}
    split = {"type": "Split", "pattern": {"Regex": prompt._SPLIT_PATTERN}, "behavior": "Isolated", "invert": False}
    return {
        "version": "1.0", "truncation": None, "padding": None,
        "added_tokens": [{"id": 258, "content": "<|end|>", "single_word": False, "lstrip": False, "rstrip": False,
                          "normalized": False, "special": True}],
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [split, dict(byte_level)]},
        "post_processor": dict(byte_level), "decoder": dict(byte_level),
        "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": "",
                  "end_of_word_suffix": "", "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
                  "vocab": vocab, "merges": [["xy", "x"], ["x", "y"]]},
    }


def test_the_supported_tokenizer_layout_is_accepted():
    prompt.check_tokenizer_layout(_tokenizer())


def _set(root, path, value):
    for key in path[:-1]:
        root = root[key]
    root[path[-1]] = value


@pytest.mark.parametrize(("path", "value"), [
    (("pre_tokenizer", "pretokenizers", 1, "use_regex"), True),
    (("pre_tokenizer", "pretokenizers", 0, "pattern", "Regex"), r"\s+|\S+"),
    (("normalizer", "type"), "NFKC"),
    (("decoder", "trim_offsets"), True),
    (("post_processor", "add_prefix_space"), 0),
    (("truncation",), {"max_length": 8}),
    (("model", "byte_fallback"), True),
    (("model", "dropout"), 0.1),
    (("model", "vocab", "xyx"), 300),
    (("model", "vocab", "xyx"), 257.0),
    (("model", "merges", 0), "xy x"),
    (("model", "merges", 1), ["x", "x"]),
    (("added_tokens", 0, "id"), 259),
    (("added_tokens", 0, "normalized"), True),
    (("added_tokens", 0, "special"), 1),
    (("added_tokens", 0, "content"), "xy"),
    (("extra",), 1),
])
def test_a_tokenizer_layout_the_runtime_cannot_load_is_rejected(path, value):
    root = _tokenizer()
    _set(root, path, value)
    with pytest.raises(ValueError, match="native tokenizer cannot load"):
        prompt.check_tokenizer_layout(root)


def test_a_vocabulary_without_every_byte_token_or_with_a_repeated_merge_is_rejected():
    missing = _tokenizer()
    del missing["model"]["vocab"]["x"]
    with pytest.raises(ValueError, match="native tokenizer cannot load"):
        prompt.check_tokenizer_layout(missing)
    repeated = _tokenizer()
    repeated["model"]["merges"].append(["x", "y"])
    with pytest.raises(ValueError, match="native tokenizer cannot load"):
        prompt.check_tokenizer_layout(repeated)


def test_the_builder_and_the_runtime_pin_the_same_split_pattern():
    source = (Path(prompt.__file__).parent / "runtime" / "bpe_tokenizer.cpp").read_text(encoding="utf-8")
    pinned = source.split('kPretokenizerPattern =\n    R"regex(', 1)[1].split(')regex"', 1)[0]
    assert pinned == prompt._SPLIT_PATTERN


def test_checkpoint_json_with_many_keys_loads_in_linear_time(tmp_path):
    # tokenizer.json's vocabulary has about 248,000 keys; a quadratic duplicate check would take hours on it.
    path = _write(tmp_path, "tokenizer.json", {"vocab": {f"t{index}": index for index in range(60000)}})
    start = time.perf_counter()
    assert len(cfg.load_json(path)["vocab"]) == 60000
    assert time.perf_counter() - start < 2.0


_TEXT = {
    "model_type": "qwen3_5_text", "hidden_act": "silu", "attention_bias": False, "attn_output_gate": True,
    "mamba_ssm_dtype": "float32", "tie_word_embeddings": True, "mlp_only_layers": [],
    "rope_parameters": {"rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25},
    "layer_types": ["linear_attention", "full_attention"], "num_hidden_layers": 2,
}


@pytest.mark.parametrize("layer_types", [
    {"linear_attention": 1, "full_attention": 2}, "linear_attention", [["linear_attention"], "full_attention"],
    None,
])
def test_layer_types_must_be_a_json_array_of_names(layer_types):
    with pytest.raises(ValueError, match="layer_types"):
        cfg.TextConfig.from_dict({**_TEXT, "layer_types": layer_types})
