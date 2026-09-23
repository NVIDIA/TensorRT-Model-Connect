# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DINOv2 family-owned configuration, preprocessing and checkpoint contracts."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

pytest.importorskip("tensorrt", reason="TensorRT is required for DINOv2 builder imports")

from families.dinov2 import model as dinov2_model  # noqa: E402
from tensorrt_model_connect import BuildRequest  # noqa: E402

# Published facebook/dinov2-small and facebook/dinov2-with-registers-small configs.
DINOV2_SMALL = {
    "architectures": ["Dinov2Model"],
    "hidden_act": "gelu",
    "hidden_size": 384,
    "image_size": 518,
    "layer_norm_eps": 1e-06,
    "layerscale_value": 1.0,
    "mlp_ratio": 4,
    "model_type": "dinov2",
    "num_attention_heads": 6,
    "num_channels": 3,
    "num_hidden_layers": 12,
    "patch_size": 14,
    "qkv_bias": True,
    "use_swiglu_ffn": False,
}
DINOV2_REGISTERS_SMALL = {
    **DINOV2_SMALL,
    "architectures": ["Dinov2WithRegistersModel"],
    "model_type": "dinov2_with_registers",
    "num_register_tokens": 4,
}
PROCESSOR = {
    "crop_size": {"height": 224, "width": 224},
    "do_center_crop": True,
    "do_convert_rgb": True,
    "do_normalize": True,
    "do_rescale": True,
    "do_resize": True,
    "image_mean": [0.485, 0.456, 0.406],
    "image_processor_type": "BitImageProcessor",
    "image_std": [0.229, 0.224, 0.225],
    "resample": 3,
    "rescale_factor": 0.00392156862745098,
    "size": {"shortest_edge": 256},
}


def test_published_configs_resolve_to_the_transformers_encoder() -> None:
    small = dinov2_model.resolve_model_config(DINOV2_SMALL)
    assert small["head_dim"] == 64
    assert small["intermediate_size"] == 1536
    assert small["num_register_tokens"] == 0
    assert small["antialias"] is False

    registers = dinov2_model.resolve_model_config(DINOV2_REGISTERS_SMALL)
    assert registers["num_register_tokens"] == 4
    assert registers["antialias"] is True


def test_swiglu_width_matches_transformers_rounding() -> None:
    giant = {
        **DINOV2_SMALL,
        "hidden_size": 1536,
        "num_attention_heads": 24,
        "num_hidden_layers": 40,
        "use_swiglu_ffn": True,
    }
    assert dinov2_model.resolve_model_config(giant)["intermediate_size"] == 4096


@pytest.mark.parametrize(
    "override",
    [
        {"architectures": ["Dinov2ForImageClassification"]},
        {"model_type": "dinov3_vit"},
        {"hidden_act": "relu"},
        {"hidden_size": 385},
        {"num_channels": 1},
    ],
)
def test_unsupported_configs_fail_before_building(override: dict) -> None:
    with pytest.raises(ValueError):
        dinov2_model.resolve_model_config({**DINOV2_SMALL, **override})


def test_registers_config_requires_a_register_count() -> None:
    config = dict(DINOV2_REGISTERS_SMALL)
    del config["num_register_tokens"]
    with pytest.raises(ValueError):
        dinov2_model.resolve_model_config(config)


def test_processor_contract_is_recorded_for_the_runtime() -> None:
    assert dinov2_model.resolve_preprocess_config(PROCESSOR, 14) == {
        "input_image_h": 224,
        "input_image_w": 224,
        "resize_shortest_edge": 256,
        "image_mean": [0.485, 0.456, 0.406],
        "image_std": [0.229, 0.224, 0.225],
    }


@pytest.mark.parametrize(
    "override",
    [
        {"resample": 2},
        {"do_center_crop": False},
        {"size": {"height": 224, "width": 224}},
        {"crop_size": {"height": 225, "width": 225}},
        {"crop_size": {"height": 266, "width": 266}},
        {"rescale_factor": 1.0},
        {"image_processor_type": "ViTImageProcessor"},
        {"image_processor_type": "BitImageProcessorFast"},
    ],
)
def test_unsupported_processors_fail_before_building(override: dict) -> None:
    with pytest.raises(ValueError):
        dinov2_model.resolve_preprocess_config({**PROCESSOR, **override}, 14)


@pytest.mark.parametrize("antialias", [False, True])
@pytest.mark.parametrize("grid", [(16, 16), (16, 20), (40, 40)])
def test_position_interpolation_matches_torch(antialias: bool, grid: tuple[int, int]) -> None:
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(3)
    table = rng.standard_normal((1, 1 + 37 * 37, 8)).astype(np.float32)

    actual = dinov2_model.interpolate_position_embeddings(
        table, grid[0], grid[1], antialias=antialias
    )

    patches = torch.from_numpy(table[:, 1:]).reshape(1, 37, 37, 8).permute(0, 3, 1, 2)
    expected = torch.nn.functional.interpolate(
        patches, size=grid, mode="bicubic", align_corners=False, antialias=antialias
    )
    expected = expected.permute(0, 2, 3, 1).reshape(1, -1, 8).numpy()
    assert actual.shape == (1, 1 + grid[0] * grid[1], 8)
    np.testing.assert_array_equal(actual[:, :1], table[:, :1])
    np.testing.assert_allclose(actual[:, 1:], expected, rtol=0, atol=2e-5)


def test_matching_grid_keeps_the_learned_table() -> None:
    table = np.arange(1 * 5 * 3, dtype=np.float32).reshape(1, 5, 3)
    actual = dinov2_model.interpolate_position_embeddings(table, 2, 2, antialias=False)
    np.testing.assert_array_equal(actual, table)


def _write_tiny_checkpoint(root: Path, *, registers: int, swiglu: bool) -> dict[str, np.ndarray]:
    hidden, patch, grid = 4, 2, 3
    config = {
        **(DINOV2_REGISTERS_SMALL if registers else DINOV2_SMALL),
        "hidden_size": hidden,
        "num_attention_heads": 2,
        "num_hidden_layers": 1,
        "image_size": grid * patch,
        "patch_size": patch,
        "mlp_ratio": 2,
        "use_swiglu_ffn": swiglu,
    }
    if registers:
        config["num_register_tokens"] = registers
    width = dinov2_model.resolve_model_config(config)["intermediate_size"]
    rng = np.random.default_rng(11)

    def tensor(*shape):
        return rng.standard_normal(shape).astype(np.float32)

    tensors = {
        "embeddings.cls_token": tensor(1, 1, hidden),
        "embeddings.mask_token": tensor(1, hidden),
        "embeddings.position_embeddings": tensor(1, 1 + grid * grid, hidden),
        "embeddings.patch_embeddings.projection.weight": tensor(hidden, 3, patch, patch),
        "embeddings.patch_embeddings.projection.bias": tensor(hidden),
        "layernorm.weight": tensor(hidden),
        "layernorm.bias": tensor(hidden),
    }
    if registers:
        tensors["embeddings.register_tokens"] = tensor(1, registers, hidden)
    layer = "encoder.layer.0"
    for name in ("query", "key", "value"):
        tensors[f"{layer}.attention.attention.{name}.weight"] = tensor(hidden, hidden)
        tensors[f"{layer}.attention.attention.{name}.bias"] = tensor(hidden)
    tensors[f"{layer}.attention.output.dense.weight"] = tensor(hidden, hidden)
    tensors[f"{layer}.attention.output.dense.bias"] = tensor(hidden)
    for name in ("norm1", "norm2"):
        tensors[f"{layer}.{name}.weight"] = tensor(hidden)
        tensors[f"{layer}.{name}.bias"] = tensor(hidden)
    for name in ("layer_scale1", "layer_scale2"):
        tensors[f"{layer}.{name}.lambda1"] = tensor(hidden)
    if swiglu:
        tensors[f"{layer}.mlp.weights_in.weight"] = tensor(2 * width, hidden)
        tensors[f"{layer}.mlp.weights_in.bias"] = tensor(2 * width)
        tensors[f"{layer}.mlp.weights_out.weight"] = tensor(hidden, width)
        tensors[f"{layer}.mlp.weights_out.bias"] = tensor(hidden)
    else:
        tensors[f"{layer}.mlp.fc1.weight"] = tensor(width, hidden)
        tensors[f"{layer}.mlp.fc1.bias"] = tensor(width)
        tensors[f"{layer}.mlp.fc2.weight"] = tensor(hidden, width)
        tensors[f"{layer}.mlp.fc2.bias"] = tensor(hidden)
    save_file(tensors, str(root / "model.safetensors"))
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return tensors


@pytest.mark.parametrize(("registers", "swiglu"), [(0, False), (2, True)])
def test_checkpoint_mapping_preserves_transformers_layout(
    tmp_path: Path, registers: int, swiglu: bool
) -> None:
    tensors = _write_tiny_checkpoint(tmp_path, registers=registers, swiglu=swiglu)
    cfg = dinov2_model.resolve_model_config(
        json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    )

    weights = dinov2_model.load_weights(tmp_path, cfg)

    layer = "encoder.layer.0.attention.attention"
    np.testing.assert_array_equal(
        weights["layer.0.qkv.weight"],
        np.concatenate(
            [tensors[f"{layer}.{name}.weight"].T for name in ("query", "key", "value")], 1
        ),
    )
    np.testing.assert_array_equal(
        weights["layer.0.qkv.bias"],
        np.concatenate([tensors[f"{layer}.{name}.bias"] for name in ("query", "key", "value")]),
    )
    source = "weights_in" if swiglu else "fc1"
    np.testing.assert_array_equal(
        weights["layer.0.mlp_in.weight"], tensors[f"encoder.layer.0.mlp.{source}.weight"].T
    )
    assert ("register_tokens" in weights) is bool(registers)
    assert "mask_token" not in weights


def test_undeclared_register_tokens_are_rejected(tmp_path: Path) -> None:
    tensors = _write_tiny_checkpoint(tmp_path, registers=0, swiglu=False)
    tensors["embeddings.register_tokens"] = np.zeros((1, 2, 4), dtype=np.float32)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    cfg = dinov2_model.resolve_model_config(
        json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    )
    with pytest.raises(ValueError, match="register tokens"):
        dinov2_model.load_weights(tmp_path, cfg)


@pytest.mark.parametrize(
    "change",
    [
        {"task": "image_features"},
        {"backend": "trt_rtx"},
        {"precision": "bf16"},
        {"image_height": 518},
        {"image_width": 518},
        {"max_batch_size": 2},
        {"max_sequence_length": 8},
        {"tensor_parallel_size": 2},
        {"context_parallel_size": 2},
        {"quantization": "fp8"},
        {"fp32_layers": (0,)},
        {"video_num_frames": 4},
        {"dynamic_kv_cache": True},
    ],
)
def test_unsupported_build_options_are_rejected(tmp_path: Path, change: dict) -> None:
    request = BuildRequest(
        model_dir=tmp_path,
        output_path=tmp_path / "unused.bundle",
        family="dinov2",
        task="image_to_token_and_pooled_features",
        precision="fp16",
    )
    with pytest.raises((ValueError, NotImplementedError)):
        dinov2_model.build(replace(request, **change), writer=None)
