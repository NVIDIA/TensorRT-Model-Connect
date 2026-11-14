# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-side unit tests for the transformers Swin Transformer V2.

Two subjects. ``window.py`` computes, on the host, everything an attention block
adds to its scores; it is checked against the reference's own modules. And
``config.resolve`` must give every block the window and the shift the reference
gives it, and refuse what this family does not reproduce.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from families.swinv2 import config as config_module, model as model_module, window
from tensorrt_model_connect import BuildRequest

_PREPROCESSOR = {
    "feature_extractor_type": "ViTFeatureExtractor",
    "do_resize": True,
    "size": 64,
    "resample": 3,
    "do_normalize": True,
    "image_mean": [0.485, 0.456, 0.406],
    "image_std": [0.229, 0.224, 0.225],
}


def _config(**overrides) -> dict:
    config = {
        "model_type": "swinv2",
        "architectures": ["Swinv2ForImageClassification"],
        "image_size": 64,
        "patch_size": 4,
        "num_channels": 3,
        "embed_dim": 8,
        "depths": [2, 2, 2],
        "num_heads": [1, 2, 4],
        "window_size": 8,
        "pretrained_window_sizes": [0, 0, 0],
        "mlp_ratio": 2.0,
        "qkv_bias": True,
        "hidden_act": "gelu",
        "use_absolute_embeddings": False,
        "layer_norm_eps": 1e-5,
        "id2label": {str(index): f"class {index}" for index in range(5)},
    }
    config.update(overrides)
    return config


def _weights(config: dict) -> dict:
    """Zero tensors under the checkpoint's names, shaped by ``config``."""
    dim, ratio = config["embed_dim"], config["mlp_ratio"]
    weights = {
        "swinv2.embeddings.patch_embeddings.projection.weight": np.zeros(
            (dim, 3, config["patch_size"], config["patch_size"]), np.float32),
        "classifier.weight": np.zeros(
            (len(config["id2label"]), dim * 2 ** (len(config["depths"]) - 1)), np.float32),
    }
    for stage, (depth, heads) in enumerate(zip(config["depths"], config["num_heads"])):
        width = dim * 2**stage
        for block in range(depth):
            prefix = f"swinv2.encoder.layers.{stage}.blocks.{block}"
            weights[f"{prefix}.attention.self.logit_scale"] = np.zeros((heads, 1, 1), np.float32)
            weights[f"{prefix}.attention.self.query.weight"] = np.zeros((width, width), np.float32)
            weights[f"{prefix}.intermediate.dense.weight"] = np.zeros(
                (int(ratio * width), width), np.float32)
        if stage < len(config["depths"]) - 1:
            weights[f"swinv2.encoder.layers.{stage}.downsample.reduction.weight"] = np.zeros(
                (2 * width, 4 * width), np.float32)
    return weights


def _model_dir(tmp_path: Path, config: dict, preprocessor: dict | None = None) -> Path:
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "preprocessor_config.json").write_text(
        json.dumps(_PREPROCESSOR if preprocessor is None else preprocessor))
    return tmp_path


def _reference_windows(config: dict) -> list[list[tuple[int, int]]]:
    """(window, shift) of every block, as the reference model builds them."""
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    model = transformers.Swinv2Model(transformers.Swinv2Config(**{
        key: value for key, value in config.items() if key not in ("architectures", "model_type")
    }))
    return [[(block.window_size, block.shift_size) for block in stage.blocks]
            for stage in model.encoder.layers]


@pytest.mark.parametrize(
    "image_size, window_size",
    [
        # Stages of 16, 8 and 4 tokens a side: larger than the window (odd
        # blocks shift by half of it), equal to it (no block shifts) and smaller
        # than it (the window is cut down to the stage and nothing shifts).
        (64, 4),
        (64, 8),
        (64, 16),
        (96, 12),
    ],
)
def test_every_block_gets_the_reference_window_and_shift(tmp_path, image_size, window_size):
    config = _config(image_size=image_size, window_size=window_size)
    preprocessor = dict(_PREPROCESSOR, size=image_size)
    cfg = config_module.resolve(_model_dir(tmp_path, config, preprocessor), _weights(config))
    resolved = [[(stage["window"], shift) for shift in stage["shifts"]] for stage in cfg["stages"]]
    assert resolved == _reference_windows(config)


def test_the_published_geometry(tmp_path):
    # microsoft/swinv2-tiny-patch4-window16-256, scaled to embed_dim 8.
    config = _config(image_size=256, window_size=16, depths=[2, 2, 6, 2], num_heads=[1, 2, 4, 8],
                     pretrained_window_sizes=[0, 0, 0, 0])
    cfg = config_module.resolve(
        _model_dir(tmp_path, config, dict(_PREPROCESSOR, size=256)), _weights(config))
    assert [stage["side"] for stage in cfg["stages"]] == [64, 32, 16, 8]
    assert [stage["window"] for stage in cfg["stages"]] == [16, 16, 16, 8]
    assert [stage["shifts"] for stage in cfg["stages"]] == [[0, 8], [0, 8], [0] * 6, [0, 0]]
    assert cfg["labels"] == [f"class {index}" for index in range(5)]
    assert cfg["vocabulary_id"] == ""
    assert cfg["resample"] == 3


@pytest.mark.parametrize(
    "overrides, error",
    [
        ({"architectures": ["Swinv2ForMaskedImageModeling"]}, NotImplementedError),
        ({"architectures": ["Swinv2Model"]}, NotImplementedError),
        ({"hidden_act": "gelu_pytorch_tanh"}, NotImplementedError),
        ({"use_absolute_embeddings": True}, NotImplementedError),
        ({"qkv_bias": False}, NotImplementedError),
        ({"embed_dim": 16}, ValueError),
        ({"num_heads": [1, 2, 8]}, ValueError),
        ({"mlp_ratio": 4.0}, ValueError),
    ],
)
def test_what_the_family_does_not_reproduce_is_refused(tmp_path, overrides, error):
    weights = _weights(_config())
    with pytest.raises(error):
        config_module.resolve(_model_dir(tmp_path, _config(**overrides)), weights)


def test_a_stage_the_reference_would_pad_is_refused(tmp_path):
    # 56 x 56 tokens do not divide into 16 x 16 windows; the reference pads.
    config = _config(image_size=224, window_size=16)
    with pytest.raises(NotImplementedError, match="pad"):
        config_module.resolve(
            _model_dir(tmp_path, config, dict(_PREPROCESSOR, size=224)), _weights(config))


@pytest.mark.parametrize(
    "preprocessor",
    [
        dict(_PREPROCESSOR, resample=2),
        dict(_PREPROCESSOR, do_center_crop=True),
        dict(_PREPROCESSOR, size={"shortest_edge": 64}),
        dict(_PREPROCESSOR, size=72),
        dict(_PREPROCESSOR, feature_extractor_type="BitImageProcessor"),
        # The fast processor resizes with torchvision, not with Pillow.
        dict(_PREPROCESSOR, feature_extractor_type="ViTImageProcessorFast"),
        dict(_PREPROCESSOR, do_normalize=False),
        # Left out, these take the processor's own defaults - bilinear and
        # 224 x 224 - not the model's bicubic and 64.
        {key: value for key, value in _PREPROCESSOR.items() if key != "resample"},
        {key: value for key, value in _PREPROCESSOR.items() if key != "size"},
    ],
)
def test_a_preprocessing_the_runtime_does_not_reproduce_is_refused(tmp_path, preprocessor):
    config = _config()
    with pytest.raises(NotImplementedError):
        config_module.resolve(_model_dir(tmp_path, config, preprocessor), _weights(config))


def test_missing_image_statistics_take_the_processor_defaults(tmp_path):
    config = _config()
    preprocessor = {key: value for key, value in _PREPROCESSOR.items()
                    if key not in ("image_mean", "image_std")}
    cfg = config_module.resolve(_model_dir(tmp_path, config, preprocessor), _weights(config))
    assert cfg["image_mean"] == [0.5, 0.5, 0.5]
    assert cfg["image_std"] == [0.5, 0.5, 0.5]


def test_processor_defaults_are_the_references():
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("PIL")
    processor = transformers.ViTImageProcessor()
    assert int(processor.resample) == config_module._PROCESSOR_DEFAULTS["resample"]
    assert dict(processor.size) == config_module._PROCESSOR_DEFAULTS["size"]
    assert list(processor.image_mean) == config_module._PROCESSOR_DEFAULTS["image_mean"]
    assert list(processor.image_std) == config_module._PROCESSOR_DEFAULTS["image_std"]


def test_a_partial_vocabulary_is_refused(tmp_path):
    config = _config(id2label={"0": "a", "1": "b"})
    weights = _weights(_config())
    with pytest.raises(ValueError, match="labels"):
        config_module.resolve(_model_dir(tmp_path, config), weights)


@pytest.mark.parametrize("size, pretrained", [(16, 0), (8, 0), (24, 12), (1, 0)])
def test_coordinate_table_and_index_match_the_reference(size, pretrained):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from transformers.models.swinv2 import modeling_swinv2

    attention = modeling_swinv2.Swinv2SelfAttention(
        transformers.Swinv2Config(), dim=8, num_heads=2, window_size=size,
        pretrained_window_size=[pretrained, pretrained])
    table, index = attention.create_coords_table_and_index()
    np.testing.assert_allclose(
        window.relative_coords_table(size, pretrained), table[0].to(torch.float64).numpy(),
        rtol=0, atol=1e-6)
    np.testing.assert_array_equal(window.relative_position_index(size), index.numpy())


def _attention_module(heads: int, size: int, dim: int, seed: int):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from transformers.models.swinv2 import modeling_swinv2

    torch.manual_seed(seed)
    attention = modeling_swinv2.Swinv2SelfAttention(
        transformers.Swinv2Config(), dim=dim, num_heads=heads, window_size=size).eval()
    with torch.no_grad():
        for parameter in attention.parameters():
            parameter.normal_(0.0, 0.5)
        # One head above the clamp at log(100): the reference uses exp(log(100)).
        attention.logit_scale[0] = math.log(1.0 / 0.01) + 1.5
    return attention


def test_position_bias_is_the_reference_squashed_mlp():
    torch = pytest.importorskip("torch")
    attention = _attention_module(heads=3, size=6, dim=12, seed=0)
    mlp = attention.continuous_position_bias_mlp
    with torch.no_grad():
        table = mlp(attention.relative_coords_table).view(-1, 3)
        expected = table[attention.relative_position_index.view(-1)].view(36, 36, -1)
        expected = 16 * torch.sigmoid(expected.permute(2, 0, 1))
    actual = window.position_bias(
        mlp[0].weight.detach().numpy(), mlp[0].bias.detach().numpy(),
        mlp[2].weight.detach().numpy(), window=6,
        pretrained_window=0)
    np.testing.assert_allclose(actual, expected.double().numpy(), rtol=0, atol=2e-5)


def test_shift_mask_is_the_reference_mask_applied_twice():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from transformers.models.swinv2 import modeling_swinv2

    config = transformers.Swinv2Config(window_size=4)
    layer = modeling_swinv2.Swinv2Layer(config, dim=8, input_resolution=(12, 12), num_heads=2,
                                        shift_size=2)
    assert (layer.window_size, layer.shift_size) == (4, 2)
    reference = layer.get_attn_mask(12, 12, dtype=torch.float32).double().numpy()
    assert set(np.unique(reference)) == {0.0, -100.0}
    np.testing.assert_array_equal(window.shift_mask(12, 12, 4, 2), 2 * reference)


def test_folded_attention_reproduces_the_reference_attention():
    """The engine's decomposition gives the reference's attention output.

    The engine normalises query and key, multiplies the normalised query by the
    clamped temperature, and adds ``window.attention_bias`` (position bias plus
    the shifted-window mask) to the product. One head is above the clamp.
    """
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from transformers.models.swinv2 import modeling_swinv2

    heads, size, dim, side, shift = 3, 4, 12, 8, 2
    attention = _attention_module(heads=heads, size=size, dim=dim, seed=1)
    layer = modeling_swinv2.Swinv2Layer(
        transformers.Swinv2Config(window_size=size), dim=dim, input_resolution=(side, side),
        num_heads=heads, shift_size=shift)
    mask = layer.get_attn_mask(side, side, dtype=torch.float32)
    count, area = (side // size) ** 2, size * size
    tokens = torch.randn(count, area, dim, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        expected = attention(tokens, mask)[0].double().numpy()

    def heads_first(values):
        return values.reshape(count, area, heads, dim // heads).transpose(0, 2, 1, 3)

    x = tokens.double().numpy()
    query = heads_first(x @ attention.query.weight.detach().double().numpy().T
                        + attention.query.bias.detach().double().numpy())
    key = heads_first(x @ attention.key.weight.detach().double().numpy().T)
    value = heads_first(x @ attention.value.weight.detach().double().numpy().T
                        + attention.value.bias.detach().double().numpy())
    query = query / np.maximum(np.linalg.norm(query, axis=-1, keepdims=True), 1e-12)
    key = key / np.maximum(np.linalg.norm(key, axis=-1, keepdims=True), 1e-12)
    temperature = window.logit_scale(attention.logit_scale.detach().numpy())
    assert temperature[0] == pytest.approx(100.0)
    mlp = attention.continuous_position_bias_mlp
    bias = window.attention_bias(
        mlp[0].weight.detach().numpy(), mlp[0].bias.detach().numpy(),
        mlp[2].weight.detach().numpy(), height=side, width=side, window=size, shift=shift,
        pretrained_window=0)
    scores = (query * temperature.reshape(1, heads, 1, 1)) @ key.transpose(0, 1, 3, 2) + bias
    scores = np.exp(scores - scores.max(-1, keepdims=True))
    context = (scores / scores.sum(-1, keepdims=True)) @ value
    actual = context.transpose(0, 2, 1, 3).reshape(count, area, dim)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-4)


def test_the_torch_checkpoint_is_read_at_float32(tmp_path):
    torch = pytest.importorskip("torch")
    torch.save({"a": torch.ones(2, 3, dtype=torch.float16), "b": torch.arange(4.0)},
               tmp_path / "pytorch_model.bin")
    weights = model_module._open_torch_checkpoint(tmp_path)
    assert set(weights) == {"a", "b"}
    assert all(value.dtype == np.float32 for value in weights.values())
    np.testing.assert_array_equal(weights["b"], np.arange(4.0, dtype=np.float32))


def _request(tmp_path: Path, **overrides) -> BuildRequest:
    values = {
        "model_dir": tmp_path,
        "output_path": tmp_path / "out.bundle",
        "family": "swinv2",
        "task": "image_to_class_scores",
        "precision": "fp32",
    }
    values.update(overrides)
    return BuildRequest(**values)


@pytest.mark.parametrize(
    "overrides, error",
    [
        ({"precision": "fp16"}, NotImplementedError),
        ({"precision": "bf16"}, NotImplementedError),
        ({"quantization": "fp8"}, NotImplementedError),
        ({"backend": "trt_rtx"}, ValueError),
        ({"task": "image_to_pooled_features"}, ValueError),
        ({"max_batch_size": 2}, NotImplementedError),
        ({"tensor_parallel_size": 2}, NotImplementedError),
        ({"max_sequence_length": 64}, NotImplementedError),
        ({"video_num_frames": 4}, NotImplementedError),
        ({"fp32_layers": (0,)}, NotImplementedError),
        ({"dynamic_kv_cache": True}, NotImplementedError),
    ],
)
def test_unsupported_build_options_are_refused_before_any_work(tmp_path, overrides, error):
    # No checkpoint exists in tmp_path: each option must be refused before it is read.
    with pytest.raises(error):
        model_module.build(_request(tmp_path, **overrides), writer=None)


def test_an_input_size_other_than_the_checkpoints_is_refused(tmp_path):
    torch = pytest.importorskip("torch")
    config = _config()
    _model_dir(tmp_path, config)
    torch.save({name: torch.from_numpy(value) for name, value in _weights(config).items()},
               tmp_path / "pytorch_model.bin")
    with pytest.raises(NotImplementedError, match="image_height"):
        model_module.build(_request(tmp_path, image_height=128), writer=None)
