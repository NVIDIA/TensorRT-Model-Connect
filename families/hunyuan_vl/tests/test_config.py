# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU contracts: checkpoint metadata validation, build-option ownership and weight layout."""

import copy
import json

import numpy as np
import pytest

from families.hunyuan_vl.config import ImageProcessor, ModelConfig, load_json
from families.hunyuan_vl.model import (
    TOKENIZER_MODEL,
    TOKENIZER_PIPELINE,
    _added_tokens,
    _check_runtime_limits,
    _check_tokenizer,
    build,
)

# The released HunyuanOCR-1.5 config.json (revision 47644ecc), reduced to the fields read here.
RELEASED = {
    "architectures": ["HunYuanVLForConditionalGeneration"],
    "model_type": "hunyuan_vl",
    "image_token_id": 120120,
    "im_start_id": 120118,
    "im_end_id": 120119,
    "image_start_token_id": 120118,
    "image_end_token_id": 120119,
    "tie_word_embeddings": True,
    "text_config": {
        "model_type": "hunyuan_vl_text",
        "hidden_act": "silu",
        "attention_bias": False,
        "hidden_size": 1024,
        "intermediate_size": 3584,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 120818,
        "max_position_embeddings": 131072,
        "rms_norm_eps": 1e-5,
        "use_qk_norm": True,
        "tie_word_embeddings": True,
        "rope_parameters": {
            "alpha": 1000.0,
            "rope_theta": 10000.0,
            "rope_type": "xdrope",
            "type": "xdrope",
            "factor": 1.0,
            "xdrope_section": [16, 16, 16, 16],
        },
    },
    "vision_config": {
        "model_type": "hunyuan_vl_vision",
        "hidden_act": "gelu",
        "hidden_size": 1152,
        "intermediate_size": 4304,
        "num_hidden_layers": 27,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "num_channels": 3,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 1,
        "max_image_size": 2048,
        "interpolate_mode": "bilinear",
        "rms_norm_eps": 1e-5,
        "text_hidden_size": 1024,
    },
}

# The released preprocessor_config.json (same revision), reduced to the fields read here.
PROCESSOR = {
    "image_processor_type": "HunYuanVLImageProcessor",
    "image_mean": [0.48145466, 0.4578275, 0.40821073],
    "image_std": [0.26862954, 0.26130258, 0.27577711],
    "min_pixels": 262144,
    "max_pixels": 16777216,
    "patch_size": 16,
    "merge_size": 2,
    "temporal_patch_size": 1,
}


def test_released_config_is_accepted():
    config = ModelConfig.from_dict(RELEASED)
    assert (
        config.hidden_size,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
    ) == (1024, 16, 8, 128)
    assert config.rope_sections == (16, 16, 16, 16) and config.rope_alpha == 1000.0
    assert (config.position_edge, config.vision_heads, config.merge_size) == (128, 16, 2)


@pytest.mark.parametrize(
    "path, value",
    [
        (("text_config", "hidden_act"), "gelu"),
        (("text_config", "use_qk_norm"), False),
        (("text_config", "rope_parameters", "xdrope_section"), [16, 16, 16]),
        (("text_config", "rope_parameters", "rope_type"), "yarn"),
        (("vision_config", "hidden_act"), "gelu_pytorch_tanh"),
        (("vision_config", "interpolate_mode"), "bicubic"),
        (("vision_config", "spatial_merge_size"), 3),
        (("tie_word_embeddings",), False),
        # Epsilons are float32 constants in the engines: zero or infinite there is unusable.
        (("text_config", "rms_norm_eps"), 1e-50),
        (("text_config", "rms_norm_eps"), 1e40),
        (("vision_config", "rms_norm_eps"), 1e-50),
        # Fixed values must have their JSON type: no null or 0 for false, no 3.0 for 3.
        (("text_config", "attention_bias"), None),
        (("text_config", "mlp_bias"), 0),
        (("vision_config", "num_channels"), 3.0),
        (("vision_config", "num_key_value_heads"), 16.0),
        (("vision_config", "text_hidden_size"), 1024.0),
        (("vision_config", "temporal_patch_size"), True),
        (("image_start_token_id",), 120118.0),
    ],
)
def test_unsupported_metadata_is_rejected(path, value):
    raw = copy.deepcopy(RELEASED)
    target = raw
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    if path == ("tie_word_embeddings",):
        raw["text_config"]["tie_word_embeddings"] = False
    with pytest.raises(ValueError):
        ModelConfig.from_dict(raw)


def test_inverse_frequencies_use_the_fixed_ntk_alpha_base():
    pytest.importorskip("torch")
    frequencies = ModelConfig.from_dict(RELEASED).inverse_frequencies()
    base = 10000.0 * 1000.0 ** (128 / 126)
    assert len(frequencies) == 64
    assert frequencies[0] == 1.0
    assert abs(frequencies[1] - base ** (-2 / 128)) < 1e-6 * base ** (-2 / 128)


@pytest.mark.parametrize(
    "rope",
    [
        # Equal in value, not in type: the float list would go unchecked.
        {"mrope_section": [16, 16, 16, 16], "xdrope_section": [16.0, 16, 16, 16]},
        # Default RoPE takes no alpha, not even a falsy one.
        {"rope_type": "default", "type": "default", "alpha": 0},
    ],
)
def test_rope_settings_of_another_json_type_are_rejected(rope):
    raw = copy.deepcopy(RELEASED)
    raw["text_config"]["rope_parameters"].update(rope)
    with pytest.raises(ValueError):
        ModelConfig.from_dict(raw)


@pytest.mark.parametrize("field, value", [("rope_theta", 1e39), ("alpha", 1e306)])
def test_rope_settings_outside_the_float32_range_are_rejected(field, value):
    pytest.importorskip("torch")
    raw = copy.deepcopy(RELEASED)
    raw["text_config"]["rope_parameters"][field] = value
    with pytest.raises(ValueError, match="float32 range"):
        ModelConfig.from_dict(raw).inverse_frequencies()


def _bundle_metadata(**changes):
    """The runtime.json fields the runtime bounds, as build() writes them for the release."""
    config = ModelConfig.from_dict(RELEASED)
    metadata = {
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "vocab_size": config.vocab_size,
        "max_cache_length": 8192,
        "patch_size": config.patch_size,
        "max_patches": config.position_edge**2,
        "position_edge": config.position_edge,
        "rope_sections": list(config.rope_sections),
        "rope_inverse_frequencies": [1.0, 1e-4],
        "tokens": {"eos": [120007, 120020]},
        "precision": "bf16",
        "image": {
            "mean": PROCESSOR["image_mean"],
            "std": PROCESSOR["image_std"],
            "rescale_factor": 1 / 255,
        },
    }
    metadata.update(changes)
    return metadata


@pytest.mark.parametrize(
    "change",
    [
        {"hidden_size": (1 << 16) + 1},
        {"num_hidden_layers": (1 << 10) + 1},
        {"head_dim": (1 << 12) + 2},
        {"vocab_size": (1 << 24) + 1},
        {"max_cache_length": (1 << 22) + 1},
        {"max_patches": (1 << 24) + 4},
        {"position_edge": 1},
        {"num_key_value_heads": 1024, "head_dim": 4096, "max_cache_length": 1 << 20},
        {"rope_sections": [8] * 9},
        {"tokens": {"eos": list(range(17))}},
        {"rope_inverse_frequencies": [1.0, 1e37]},
        # White pixels at 3.4e38: finite in float32, infinite once packed as bfloat16.
        {"image": {"mean": [0, 0, 0], "std": [7.5e-37, 0.5, 0.5], "rescale_factor": 1.0}},
    ],
)
def test_bundle_metadata_the_runtime_would_refuse_is_rejected(change):
    _check_runtime_limits(_bundle_metadata())
    with pytest.raises(ValueError):
        _check_runtime_limits(_bundle_metadata(**change))


def test_bfloat16_packing_bounds_only_bfloat16_bundles():
    image = {"mean": [0, 0, 0], "std": [7.5e-37, 0.5, 0.5], "rescale_factor": 1.0}
    _check_runtime_limits(_bundle_metadata(image=image, precision="fp32"))
    with pytest.raises(ValueError, match="overflow bfloat16"):
        _check_runtime_limits(_bundle_metadata(image=image))


def test_rope_angles_that_overflow_float32_within_the_cache_are_rejected():
    pytest.importorskip("torch")
    raw = copy.deepcopy(RELEASED)
    raw["text_config"]["rope_parameters"] = {
        "rope_type": "default",
        "rope_theta": 1e-38,
        "xdrope_section": [16, 16, 16, 16],
    }
    # Each frequency is a normal float32 value; their products with cache positions are not.
    frequencies = ModelConfig.from_dict(raw).inverse_frequencies()
    with pytest.raises(ValueError, match="overflow float32 within the cache"):
        _check_runtime_limits(_bundle_metadata(rope_inverse_frequencies=frequencies))
    _check_runtime_limits(
        _bundle_metadata(rope_inverse_frequencies=frequencies, max_cache_length=1)
    )


def test_json_that_repeats_a_key_is_rejected(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(RELEASED)[:-1] + ', "model_type": "hunyuan_vl"}', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="repeats the key 'model_type'"):
        ModelConfig.from_dir(tmp_path)
    nested = tmp_path / "nested.json"
    nested.write_text('{"a": [{"b": 1}, {"b": 2}], "c": {"b": 1, "b": 2}}', encoding="utf-8")
    with pytest.raises(ValueError, match="repeats the key 'b'"):
        load_json(nested)
    nested.write_text('{"a": [{"b": 1}, {"b": 2}], "c": {"b": 1}}', encoding="utf-8")
    assert load_json(nested) == {"a": [{"b": 1}, {"b": 2}], "c": {"b": 1}}


@pytest.mark.parametrize(
    "field, value",
    [
        ("tensor_parallel_size", 2),
        ("max_batch_size", 2),
        ("context_parallel_size", 2),
        ("video_num_frames", 4),
        ("dynamic_kv_cache", True),
        ("quantization", "fp8"),
        ("fp32_layers", (0,)),
        ("image_height", 1024),
        ("precision", "fp16"),
        ("task", "text_generation"),
    ],
)
def test_unowned_build_options_are_rejected(field, value, tmp_path):
    from tensorrt_model_connect import BuildRequest

    arguments = dict(
        model_dir=tmp_path,
        output_path=tmp_path / "out.bundle",
        family="hunyuan_vl",
        task="images_text_to_text",
        precision="bf16",
    )
    arguments[field] = value
    with pytest.raises(NotImplementedError if field == "dynamic_kv_cache" else ValueError):
        build(BuildRequest(**arguments), writer=None)


def _square_bound_request(tmp_path, edge):
    from tensorrt_model_connect import BuildRequest

    (tmp_path / "config.json").write_text(json.dumps(RELEASED), encoding="utf-8")
    (tmp_path / "preprocessor_config.json").write_text(json.dumps(PROCESSOR), encoding="utf-8")
    return BuildRequest(
        model_dir=tmp_path,
        output_path=tmp_path / "out.bundle",
        family="hunyuan_vl",
        task="images_text_to_text",
        precision="bf16",
        image_height=edge,
        image_width=edge,
    )


@pytest.mark.parametrize("edge", [224, 511])
def test_image_area_bound_below_min_pixels_is_rejected(edge, tmp_path):
    with pytest.raises(ValueError, match="min_pixels"):
        build(_square_bound_request(tmp_path, edge), writer=None)


def test_rope_frequencies_are_checked_before_the_engines_are_built(tmp_path):
    pytest.importorskip("torch")
    request = _square_bound_request(tmp_path, 512)
    raw = copy.deepcopy(RELEASED)
    raw["text_config"]["rope_parameters"]["rope_theta"] = 1e39
    (tmp_path / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="float32 range"):
        build(request, writer=None)


def test_image_area_bound_at_min_pixels_is_accepted(tmp_path):
    # The fixture holds no chat template, which build() reads only after the area bound passes.
    with pytest.raises(FileNotFoundError, match="chat_template"):
        build(_square_bound_request(tmp_path, 512), writer=None)


def test_merge_projection_matches_the_strided_convolution():
    torch = pytest.importorskip("torch")
    from families.hunyuan_vl.graph import merge_projection

    rng = np.random.default_rng(3)
    weight = rng.normal(size=(6, 4, 2, 2)).astype(np.float32)
    grid = rng.normal(size=(1, 4, 4, 6)).astype(np.float32)  # channels, rows, cols
    expected = torch.nn.functional.conv2d(
        torch.from_numpy(grid), torch.from_numpy(weight), stride=2
    )
    blocks = [
        np.concatenate([grid[0, :, 2 * by + dy, 2 * bx + dx] for dy in (0, 1) for dx in (0, 1)])
        for by in range(2)
        for bx in range(3)
    ]
    actual = np.stack(blocks) @ merge_projection(weight).T
    np.testing.assert_allclose(
        actual, expected[0].permute(1, 2, 0).reshape(6, 6).numpy(), atol=1e-5
    )


@pytest.mark.parametrize(
    "change",
    [
        {"image_mean": ["0.5", 0.5, 0.5]},
        {"image_mean": [True, 0.5, 0.5]},
        {"image_mean": [float("nan"), 0.5, 0.5]},
        {"image_std": [1e-45, 0.5, 0.5]},
        {"image_mean": [1e38, 0.5, 0.5]},
        {"rescale_factor": 1e-320},
        {"image_std": [1e-37, 0.5, 0.5], "rescale_factor": 1.0},
        {"max_pixels": 1 << 41},
        # Outside float32 although the folded values would be finite: the runtime rejects them.
        {"rescale_factor": 1e40, "image_mean": [0, 0, 0], "image_std": [1e37, 1e37, 1e37]},
        {"image_mean": [3.4028235e38, 0.5, 0.5], "image_std": [1, 1, 1], "rescale_factor": 1.0},
        {"image_std": [3.4028235e38, 1, 1], "rescale_factor": 1.0},
        # Fixed values must have their JSON type.
        {"patch_size": 16.0},
        {"merge_size": 2.0},
        {"temporal_patch_size": True},
        {"size": []},
        {"image_processor_type": None},
    ],
)
def test_image_settings_the_runtime_cannot_honor_are_rejected(change, tmp_path):
    (tmp_path / "preprocessor_config.json").write_text(json.dumps({**PROCESSOR, **change}))
    with pytest.raises(ValueError):
        ImageProcessor.from_dir(tmp_path, ModelConfig.from_dict(RELEASED))


def test_released_image_settings_are_accepted(tmp_path):
    (tmp_path / "preprocessor_config.json").write_text(json.dumps(PROCESSOR))
    processor = ImageProcessor.from_dir(tmp_path, ModelConfig.from_dict(RELEASED))
    assert processor.min_pixels == 262144 and processor.std == tuple(PROCESSOR["image_std"])


def _released_tokenizer():
    """A small tokenizer.json with the released pipeline and BPE options."""
    document = copy.deepcopy(TOKENIZER_PIPELINE)
    document["model"] = {
        **copy.deepcopy(TOKENIZER_MODEL),
        "vocab": {"a": 0, "b": 1, "ab": 2},
        "merges": [["a", "b"]],
    }
    flags = {"single_word": False, "lstrip": False, "rstrip": False, "normalized": False}
    document["added_tokens"] = [{"id": 3, "content": "<x>", **flags, "special": True}]
    return document


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d["model"].update(dropout=0.1),
        lambda d: d["model"].update(unk_token="a"),
        lambda d: d["model"].update(byte_fallback=0),
        lambda d: d["pre_tokenizer"]["pretokenizers"][-1].update(use_regex=True),
        lambda d: d["pre_tokenizer"]["pretokenizers"][0].update(invert=0),
        lambda d: d.update(normalizer={"type": "NFC"}),
        lambda d: d["added_tokens"][0].update(lstrip=True),
        lambda d: d["model"]["merges"].append(["a", "b", "ab"]),
        lambda d: d["model"]["merges"].append("ab a"),
        lambda d: d.update(post_processor={"type": "TemplateProcessing", "single": [], "pair": []}),
        lambda d: d.update(truncation={"max_length": 8, "strategy": "LongestFirst", "stride": 0}),
        lambda d: d.update(padding={"strategy": "BatchLongest", "pad_id": 0, "pad_token": "a"}),
        # The runtime reads the special flag as a boolean, and the added tokens as a list.
        lambda d: d["added_tokens"][0].update(special=None),
        lambda d: d.update(added_tokens={}),
    ],
)
def test_tokenizer_the_runtime_does_not_implement_is_rejected(change):
    document = _released_tokenizer()
    _check_tokenizer(document)
    change(document)
    with pytest.raises(ValueError):
        _check_tokenizer(document)


def _tokenizer(vocab, added):
    return {
        "model": {"type": "BPE", "vocab": vocab},
        "added_tokens": [{"id": i, "content": c} for c, i in added.items()],
    }


@pytest.mark.parametrize(
    "vocab, added",
    [
        ({"a": 0, "b": 4}, {}),
        ({"a": 0, "b": 0}, {}),
        ({"a": 0, "b": True}, {}),
        ({"a": 0, "b": 1}, {"<x>": 4}),
        ({"a": 0, "b": 1}, {"<x>": 1}),
        ({"a": 0, "b": 1}, {"a": 2}),
        # Not the ids tokenizers assigns: the base entry count (2) for the first new token, then
        # one more each, not one past the highest base id.
        ({"a": 0, "b": 1}, {"<x>": 3}),
        ({"a": 0, "b": 1}, {"a": 0, "<x>": 3}),
        ({"a": 0, "c": 2}, {"<x>": 3}),
    ],
)
def test_token_ids_outside_the_embedding_table_or_conflicting_are_rejected(vocab, added):
    with pytest.raises(ValueError):
        _added_tokens(_tokenizer(vocab, added), rows=4)


def test_token_ids_inside_the_embedding_table_are_accepted():
    tokenizer = _tokenizer({"a": 0, "b": 1}, {"<x>": 2, "a": 0, "<y>": 3})
    assert _added_tokens(tokenizer, rows=4) == {"<x>": 2, "a": 0, "<y>": 3}


@pytest.mark.parametrize("ids", [(3, 4), (40, 41), (3, 5)])
def test_added_token_ids_are_those_tokenizers_assigns(ids):
    tokenizers = pytest.importorskip("tokenizers")
    # A sparse base vocabulary: three entries, the highest id 7.
    document = {
        "version": "1.0",
        "normalizer": None,
        "pre_tokenizer": None,
        "post_processor": None,
        "decoder": None,
        "model": {"type": "BPE", "dropout": None, "vocab": {"a": 0, "b": 1, "c": 7}, "merges": []},
        "added_tokens": [
            {
                "id": i,
                "content": c,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
            for c, i in zip(("<x>", "<y>"), ids)
        ],
    }
    reference = tokenizers.Tokenizer.from_str(json.dumps(document))
    assigned = {c: reference.token_to_id(c) for c in ("<x>", "<y>")}
    assert assigned == {"<x>": 3, "<y>": 4}
    if dict(zip(("<x>", "<y>"), ids)) == assigned:
        assert _added_tokens(document, rows=8) == assigned
    else:
        with pytest.raises(ValueError, match="tokenizers assigns"):
            _added_tokens(document, rows=64)


def test_released_config_names_every_tensor_the_builder_reads():
    shapes = ModelConfig.from_dict(RELEASED).tensor_shapes()
    # The released checkpoint holds these 713 tensors plus the unused vit.perceive.image_sep.
    assert len(shapes) == 713
    assert shapes["model.layers.23.mlp.down_proj.weight"] == (1024, 3584)
    assert shapes["model.layers.0.self_attn.k_proj.weight"] == (1024, 1024)
    assert shapes["vit.embeddings.position_embedding.weight"] == (16385, 1152)
    assert shapes["vit.perceive.proj.0.weight"] == (2304, 1152, 2, 2)
    assert shapes["vit.perceive.mlp.weight"] == (1024, 4608)


def test_checkpoint_tensors_must_have_their_configured_shape_and_a_float_dtype(tmp_path):
    pytest.importorskip("torch")
    from safetensors.numpy import save_file

    from families.hunyuan_vl.checkpoint import CheckpointWeights

    save_file(
        {"a": np.zeros((2, 3), np.float32), "b": np.zeros(4, np.int32)},
        str(tmp_path / "model.safetensors"),
    )
    weights = CheckpointWeights(tmp_path)
    weights.check({"a": (2, 3), "c": (1,)}, optional={"c"})
    with pytest.raises(ValueError, match="shape"):
        weights.check({"a": (3, 2)})
    with pytest.raises(ValueError, match="no tensor"):
        weights.check({"c": (1,)})
    with pytest.raises(ValueError, match="I32"):
        weights.check({"b": (4,)})
