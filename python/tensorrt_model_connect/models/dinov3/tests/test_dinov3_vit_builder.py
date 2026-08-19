# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused native-TRT construction and semantic tests for DINOv3 ViT."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file


trt = pytest.importorskip("tensorrt", reason="TensorRT is required for DINOv3 builders")

from tensorrt_model_connect.models import resolve_family_id  # noqa: E402
import tensorrt_model_connect.models.dinov3.model as model  # noqa: E402
from tensorrt_model_connect.models.dinov3.config import ModelConfig  # noqa: E402
from tensorrt_model_connect.models.dinov3.model import (  # noqa: E402
    build_vit_engine,
    load_vit_weights,
    resolve_vit_config,
)


def _write_tiny_vit(
    root: Path,
    *,
    layer_prefix: str = "layer",
    use_gated_mlp: bool = False,
    seed: int = 17,
) -> tuple[dict, dict[str, np.ndarray]]:
    hidden = 8
    intermediate = 16
    registers = 2
    patch_size = 16
    config = {
        "model_type": "dinov3_vit",
        "architectures": ["DINOv3ViTModel"],
        "image_size": 32,
        "patch_size": patch_size,
        "num_channels": 3,
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "hidden_act": "silu" if use_gated_mlp else "gelu",
        "attention_dropout": 0.0,
        "layer_norm_eps": 1.0e-5,
        "rope_theta": 100.0,
        "query_bias": True,
        "key_bias": False,
        "value_bias": True,
        "proj_bias": True,
        "mlp_bias": True,
        "layerscale_value": 1.0,
        "drop_path_rate": 0.0,
        "use_gated_mlp": use_gated_mlp,
        "num_register_tokens": registers,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    rng = np.random.default_rng(seed)

    def values(*shape: int) -> np.ndarray:
        return (0.1 * rng.standard_normal(shape)).astype(np.float32)

    tensors = {
        "embeddings.cls_token": values(1, 1, hidden),
        "embeddings.mask_token": np.zeros((1, 1, hidden), dtype=np.float32),
        "embeddings.register_tokens": values(1, registers, hidden),
        "embeddings.patch_embeddings.weight": values(hidden, 3, patch_size, patch_size),
        "embeddings.patch_embeddings.bias": values(hidden),
        "norm.weight": values(hidden),
        "norm.bias": values(hidden),
    }
    prefix = f"{layer_prefix}.0"
    tensors.update(
        {
            f"{prefix}.norm1.weight": values(hidden),
            f"{prefix}.norm1.bias": values(hidden),
            f"{prefix}.attention.q_proj.weight": values(hidden, hidden),
            f"{prefix}.attention.q_proj.bias": values(hidden),
            f"{prefix}.attention.k_proj.weight": values(hidden, hidden),
            f"{prefix}.attention.v_proj.weight": values(hidden, hidden),
            f"{prefix}.attention.v_proj.bias": values(hidden),
            f"{prefix}.attention.o_proj.weight": values(hidden, hidden),
            f"{prefix}.attention.o_proj.bias": values(hidden),
            f"{prefix}.layer_scale1.lambda1": values(hidden),
            f"{prefix}.norm2.weight": values(hidden),
            f"{prefix}.norm2.bias": values(hidden),
            f"{prefix}.mlp.up_proj.weight": values(intermediate, hidden),
            f"{prefix}.mlp.up_proj.bias": values(intermediate),
            f"{prefix}.mlp.down_proj.weight": values(hidden, intermediate),
            f"{prefix}.mlp.down_proj.bias": values(hidden),
            f"{prefix}.layer_scale2.lambda1": values(hidden),
        }
    )
    if use_gated_mlp:
        tensors[f"{prefix}.mlp.gate_proj.weight"] = values(intermediate, hidden)
        tensors[f"{prefix}.mlp.gate_proj.bias"] = values(intermediate)
    save_file(tensors, str(root / "model.safetensors"), metadata={"format": "pt"})
    return config, tensors


def _write_tiny_timm_vit(root: Path) -> tuple[dict, dict[str, np.ndarray]]:
    hidden = 8
    intermediate = 16
    registers = 2
    patch_size = 16
    config = {
        "model_type": "dinov3_vit",
        "image_size": 32,
        "patch_size": patch_size,
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "hidden_act": "gelu",
        "layer_norm_eps": 1.0e-5,
        "rope_theta": 100.0,
        "num_register_tokens": registers,
        "query_bias": True,
        "key_bias": False,
        "value_bias": True,
        "proj_bias": True,
        "mlp_bias": True,
        "use_gated_mlp": False,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")

    def values(*shape: int, start: int = 0) -> np.ndarray:
        size = int(np.prod(shape))
        return np.arange(start, start + size, dtype=np.float32).reshape(shape)

    prefix = "blocks.0"
    tensors = {
        "cls_token": values(1, 1, hidden),
        "reg_token": values(1, registers, hidden, start=10),
        "patch_embed.proj.weight": values(hidden, 3, patch_size, patch_size),
        "patch_embed.proj.bias": values(hidden),
        "norm.weight": values(hidden, start=1),
        "norm.bias": values(hidden, start=2),
        f"{prefix}.norm1.weight": values(hidden, start=3),
        f"{prefix}.norm1.bias": values(hidden, start=4),
        f"{prefix}.gamma_1": values(hidden, start=5),
        f"{prefix}.norm2.weight": values(hidden, start=6),
        f"{prefix}.norm2.bias": values(hidden, start=7),
        f"{prefix}.gamma_2": values(hidden, start=8),
        f"{prefix}.attn.qkv.weight": values(3 * hidden, hidden, start=100),
        f"{prefix}.attn.q_bias": values(hidden, start=200),
        f"{prefix}.attn.v_bias": values(hidden, start=300),
        f"{prefix}.attn.proj.weight": values(hidden, hidden, start=400),
        f"{prefix}.attn.proj.bias": values(hidden, start=500),
        f"{prefix}.mlp.fc1.weight": values(intermediate, hidden, start=600),
        f"{prefix}.mlp.fc1.bias": values(intermediate, start=700),
        f"{prefix}.mlp.fc2.weight": values(hidden, intermediate, start=800),
        f"{prefix}.mlp.fc2.bias": values(hidden, start=900),
    }
    save_file(tensors, str(root / "model.safetensors"), metadata={"format": "pt"})
    return config, tensors


def test_model_owned_build_parses_the_concrete_model_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert model.ModelConfig is ModelConfig

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "dinov3_vit",
                "architectures": ["DINOv3ViTModel"],
            }
        ),
        encoding="utf-8",
    )

    class ReachedWeightLoading(Exception):
        pass

    def stop_after_config_parse(
        model_dir: str,
        config: ModelConfig,
        *,
        precision: str,
    ) -> None:
        assert model_dir == str(tmp_path)
        assert type(config) is ModelConfig
        assert config.model_type == "dinov3_vit"
        assert precision == "fp32"
        raise ReachedWeightLoading

    monkeypatch.setattr(model, "load_weights", stop_after_config_parse)

    with pytest.raises(ReachedWeightLoading):
        model.build(str(tmp_path), str(tmp_path / "model.bundle"))


@pytest.mark.parametrize("layer_prefix", ["layer", "model.layer"])
@pytest.mark.parametrize("use_gated_mlp", [False, True])
def test_vit_loader_accepts_hf_namespaces_and_mlp_variants(
    tmp_path: Path,
    layer_prefix: str,
    use_gated_mlp: bool,
) -> None:
    _write_tiny_vit(
        tmp_path,
        layer_prefix=layer_prefix,
        use_gated_mlp=use_gated_mlp,
    )
    config = ModelConfig.from_dir(tmp_path)

    weights = load_vit_weights(str(tmp_path), config, precision="fp16")

    assert weights["patch.weight"].shape == (8, 3, 16, 16)
    assert weights["register_tokens"].shape == (1, 2, 8)
    assert weights["layer.0.attention.q_proj.weight"].shape == (8, 8)
    assert "layer.0.attention.k_proj.bias" not in weights
    assert weights["layer.0.mlp.up_proj.weight"].shape == (8, 16)
    assert weights["layer.0.mlp.down_proj.weight"].shape == (16, 8)
    assert ("layer.0.mlp.gate_proj.weight" in weights) is use_gated_mlp
    assert all(value.dtype == np.float16 for value in weights.values())


def test_vit_loader_maps_timm_dinov3_qkvb_checkpoint(tmp_path: Path) -> None:
    _, checkpoint = _write_tiny_timm_vit(tmp_path)
    config = ModelConfig.from_dir(tmp_path)

    weights = load_vit_weights(str(tmp_path), config, precision="fp32")

    np.testing.assert_array_equal(weights["cls_token"], checkpoint["cls_token"])
    np.testing.assert_array_equal(weights["register_tokens"], checkpoint["reg_token"])
    qkv = checkpoint["blocks.0.attn.qkv.weight"]
    query, key, value = np.split(qkv, 3, axis=0)
    np.testing.assert_array_equal(weights["layer.0.attention.q_proj.weight"], query.T)
    np.testing.assert_array_equal(weights["layer.0.attention.k_proj.weight"], key.T)
    np.testing.assert_array_equal(weights["layer.0.attention.v_proj.weight"], value.T)
    np.testing.assert_array_equal(
        weights["layer.0.attention.q_proj.bias"], checkpoint["blocks.0.attn.q_bias"]
    )
    np.testing.assert_array_equal(
        weights["layer.0.attention.v_proj.bias"], checkpoint["blocks.0.attn.v_bias"]
    )
    assert "layer.0.attention.k_proj.bias" not in weights
    np.testing.assert_array_equal(
        weights["layer.0.attention.o_proj.weight"],
        checkpoint["blocks.0.attn.proj.weight"].T,
    )
    np.testing.assert_array_equal(
        weights["layer.0.mlp.up_proj.weight"],
        checkpoint["blocks.0.mlp.fc1.weight"].T,
    )
    np.testing.assert_array_equal(
        weights["layer.0.mlp.down_proj.weight"],
        checkpoint["blocks.0.mlp.fc2.weight"].T,
    )
    np.testing.assert_array_equal(
        weights["layer.0.layer_scale1.lambda1"], checkpoint["blocks.0.gamma_1"]
    )
    np.testing.assert_array_equal(
        weights["layer.0.layer_scale2.lambda1"], checkpoint["blocks.0.gamma_2"]
    )


def test_timm_dinov3_qkvb_config_normalizes_before_metadata(tmp_path: Path) -> None:
    sparse = {
        "architecture": "vit_small_patch16_dinov3_qkvb",
        "num_classes": 0,
        "num_features": 384,
        "global_pool": "avg",
        "pretrained_cfg": {"input_size": [3, 256, 256]},
    }
    (tmp_path / "config.json").write_text(json.dumps(sparse), encoding="utf-8")
    config = ModelConfig.from_dir(tmp_path)

    assert resolve_family_id(config) == "dinov3"
    assert model.matches(config.model_type)
    assert not model.matches("vit_small_patch16_dinov3_qkvb_classifier")
    metadata = model.get_bundle_config_overrides(config)

    expected = {
        "model_type": "dinov3_vit",
        "architectures": ["DINOv3ViTModel"],
        "image_size": 224,
        "patch_size": 16,
        "num_channels": 3,
        "hidden_size": 384,
        "intermediate_size": 1536,
        "num_hidden_layers": 12,
        "num_attention_heads": 6,
        "num_key_value_heads": 6,
        "hidden_act": "gelu",
        "layer_norm_eps": 1.0e-5,
        "rope_theta": 100.0,
        "num_register_tokens": 4,
        "query_bias": True,
        "key_bias": False,
        "value_bias": True,
        "proj_bias": True,
        "mlp_bias": True,
        "use_gated_mlp": False,
    }
    assert config.model_type == "dinov3_vit"
    assert config.architectures == ["DINOv3ViTModel"]
    assert {key: config.raw[key] for key in expected} == expected
    assert config.hidden_size == 384
    assert config.intermediate_size == 1536
    assert config.num_hidden_layers == 12
    assert config.num_attention_heads == 6
    assert config.num_key_value_heads == 6
    assert metadata["model_type"] == "dinov3_vit"
    assert metadata["dinov3_architecture"] == "vit"
    assert metadata["sequence_length"] == 201
    assert metadata["image_size"] == 224
    assert metadata["intermediate_size"] == 1536
    assert metadata["num_hidden_layers"] == 12
    assert metadata["num_attention_heads"] == 6


def test_real_tensorrt_timm_mapped_vit_build(tmp_path: Path) -> None:
    _write_tiny_timm_vit(tmp_path)
    config = ModelConfig.from_dir(tmp_path)
    weights = load_vit_weights(str(tmp_path), config, precision="fp16")

    plan = build_vit_engine(config, weights, precision="fp16", verbose=False)
    engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(plan)

    assert engine is not None
    assert tuple(engine.get_tensor_shape("last_hidden_state")) == (1, 7, 8)
    assert tuple(engine.get_tensor_shape("pooler_output")) == (1, 8)


def test_vit_config_and_bundle_metadata_preserve_hf_contract(tmp_path: Path) -> None:
    raw, _ = _write_tiny_vit(tmp_path)
    resolved = resolve_vit_config(raw)
    config = ModelConfig.from_dir(tmp_path)
    metadata = model.get_bundle_config_overrides(config)

    assert resolved["head_dim"] == 4
    assert resolved["num_register_tokens"] == 2
    assert metadata["runtime_strategy"] == "dinov3_image_feature_extraction"
    assert metadata["dinov3_architecture"] == "vit"
    assert metadata["sequence_length"] == 7
    assert metadata["input_image_h"] == 32
    assert metadata["input_image_w"] == 32
    assert metadata["image_mean"] == [0.485, 0.456, 0.406]
    assert metadata["image_std"] == [0.229, 0.224, 0.225]
    assert metadata["interpolation"] == "bilinear"
    assert metadata["do_center_crop"] is False
    assert model.default_max_cache_length(config) == 1


@pytest.mark.parametrize("precision", ["fp32", "fp16"])
def test_real_tensorrt_vit_build_marks_hf_outputs(tmp_path: Path, precision: str) -> None:
    _write_tiny_vit(tmp_path)
    config = ModelConfig.from_dir(tmp_path)
    weights = load_vit_weights(str(tmp_path), config, precision=precision)

    plan = build_vit_engine(config, weights, precision=precision, verbose=False)
    engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(plan)

    assert engine is not None
    names = {engine.get_tensor_name(index) for index in range(engine.num_io_tensors)}
    assert names == {"pixel_values", "last_hidden_state", "pooler_output"}
    assert tuple(engine.get_tensor_shape("last_hidden_state")) == (1, 7, 8)
    assert tuple(engine.get_tensor_shape("pooler_output")) == (1, 8)
    assert engine.get_tensor_dtype("last_hidden_state") == trt.float32
    assert engine.get_tensor_dtype("pooler_output") == trt.float32


def test_real_tensorrt_vit_fp32_matches_transformers(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for TensorRT semantic parity")

    raw, checkpoint = _write_tiny_vit(tmp_path, layer_prefix="layer", seed=23)
    config = ModelConfig.from_dir(tmp_path)
    weights = load_vit_weights(str(tmp_path), config, precision="fp32")
    plan = build_vit_engine(config, weights, precision="fp32", verbose=False)

    hf_config = transformers.DINOv3ViTConfig(**raw)
    model = transformers.DINOv3ViTModel(hf_config).eval().cuda()
    expected_names = set(model.state_dict())
    state = {}
    for name, value in checkpoint.items():
        target_name = name
        if name.startswith("layer.") and target_name not in expected_names:
            target_name = "model." + name
        if target_name in expected_names:
            state[target_name] = torch.from_numpy(value)
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected
    assert not [name for name in missing if name != "rope_embeddings.inv_freq"]

    rng = np.random.default_rng(29)
    pixels = torch.from_numpy(
        rng.standard_normal((1, 3, 32, 32), dtype=np.float32)
    ).cuda()
    with torch.inference_mode():
        reference = model(pixel_values=pixels).last_hidden_state

    engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(plan)
    context = engine.create_execution_context()
    output = torch.empty((1, 7, 8), dtype=torch.float32, device="cuda")
    pooled = torch.empty((1, 8), dtype=torch.float32, device="cuda")
    for name, tensor in {
        "pixel_values": pixels,
        "last_hidden_state": output,
        "pooler_output": pooled,
    }.items():
        assert context.set_tensor_address(name, tensor.data_ptr())
    assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()

    torch.testing.assert_close(output, reference, atol=2.0e-5, rtol=2.0e-5)
    torch.testing.assert_close(pooled, output[:, 0, :], atol=0.0, rtol=0.0)
    cosine = torch.nn.functional.cosine_similarity(
        output.reshape(-1), reference.reshape(-1), dim=0
    )
    assert float(cosine) >= 0.999999
