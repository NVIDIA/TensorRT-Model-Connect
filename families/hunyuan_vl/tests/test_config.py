# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU contracts: checkpoint metadata validation, build-option ownership and weight layout."""

import copy
import json

import numpy as np
import pytest

from families.hunyuan_vl.config import ModelConfig
from families.hunyuan_vl.model import build

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
        (("tie_word_embeddings",), False),
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
