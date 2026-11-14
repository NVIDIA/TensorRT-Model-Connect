# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default precision and the fp32 parts of the SAM mask decoder."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_the_default_precision_is_fp16() -> None:
    from tensorrt_model_connect.model_support import ModelMetadata

    from families.sam.support import describe

    support = describe(ModelMetadata(config={"model_type": "sam"}, model_index={}))
    assert support is not None
    assert support.default_precision == "fp16"


def _tiny_checkpoint(directory: Path) -> Path:
    """A random SAM checkpoint with a small mask decoder (the encoder is not built)."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    config = transformers.SamConfig(
        vision_config={
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "mlp_dim": 64,
            "output_channels": 32,
            "num_pos_feats": 16,
            "global_attn_indexes": [0],
        },
        prompt_encoder_config={"hidden_size": 32},
        mask_decoder_config={
            "hidden_size": 32,
            "num_attention_heads": 2,
            "mlp_dim": 64,
            "iou_head_hidden_dim": 32,
        },
    )
    torch.manual_seed(0)
    transformers.SamModel(config).save_pretrained(directory, safe_serialization=True)
    return directory


def _decoder_network(model_dir: Path, precision: str, monkeypatch) -> list[tuple[str, str, str]]:
    """(layer type, layer name, first output type) of the mask-decoder network."""
    trt = pytest.importorskip("tensorrt")
    from families.sam.config import ModelConfig
    from families.sam.model import _SamModel

    captured: list[tuple[str, str, str]] = []

    def capture(builder, network, config):
        del builder, config
        for index in range(network.num_layers):
            layer = network.get_layer(index)
            dtype = layer.get_output(0).dtype if layer.num_outputs else None
            captured.append((str(layer.type), layer.name, str(dtype)))
        return b"plan"

    try:
        trt.Builder(trt.Logger(trt.Logger.ERROR))
    except Exception as error:  # noqa: BLE001 - no GPU or driver on this machine
        pytest.skip(f"TensorRT builder unavailable: {error}")
    monkeypatch.setattr(trt.Builder, "build_serialized_network", capture)
    model = _SamModel()
    config = ModelConfig.from_dir(model_dir)
    weights = model.load_weights(str(model_dir), config)
    assert (
        model.build_vision_engine(str(model_dir), config, weights, precision=precision) == b"plan"
    )
    return captured


def test_fp16_decoder_keeps_the_two_way_transformer_in_fp32(tmp_path, monkeypatch) -> None:
    model_dir = _tiny_checkpoint(tmp_path / "sam")
    layers = _decoder_network(model_dir, "fp16", monkeypatch)
    attention = [layer for layer in layers if "ATTENTION" in layer[0]]
    deconvolution = [layer for layer in layers if "DECONVOLUTION" in layer[0]]
    assert attention and deconvolution
    assert all(layer[2] == "DataType.FLOAT" for layer in attention)
    assert all(layer[2] == "DataType.HALF" for layer in deconvolution)


def test_fp32_decoder_has_no_fp16_layer(tmp_path, monkeypatch) -> None:
    model_dir = _tiny_checkpoint(tmp_path / "sam")
    layers = _decoder_network(model_dir, "fp32", monkeypatch)
    assert layers
    assert all(layer[2] != "DataType.HALF" for layer in layers)
