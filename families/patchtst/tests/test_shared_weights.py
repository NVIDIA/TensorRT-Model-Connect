# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PatchTST adds one weight constant per linear, shared by all channel rows."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")
safetensors_numpy = pytest.importorskip("safetensors.numpy")

from .. import model as model_module  # noqa: E402
from ..config import ModelConfig  # noqa: E402

D_MODEL, FFN, LAYERS, HEADS, PATCH, CONTEXT, HORIZON = 32, 64, 2, 4, 8, 48, 4
NUM_PATCHES = CONTEXT // PATCH
# Weight constants of one encoder layer: q, k, v, out ([d, d]), ff.0 ([d, ffn]), ff.3 ([ffn, d]).
LINEAR_WEIGHT_SHAPES = {(D_MODEL, D_MODEL), (D_MODEL, FFN), (FFN, D_MODEL)}


def _rand(rng, *shape):
    return (rng.standard_normal(shape) * 0.05).astype(np.float32)


def _write_checkpoint(path: Path, channels: int) -> None:
    """A small PatchTST forecast checkpoint in the HF layout."""
    rng = np.random.default_rng(0)
    config = {
        "architectures": ["PatchTSTForPrediction"],
        "model_type": "patchtst",
        "d_model": D_MODEL,
        "ffn_dim": FFN,
        "num_attention_heads": HEADS,
        "num_hidden_layers": LAYERS,
        "context_length": CONTEXT,
        "patch_length": PATCH,
        "patch_stride": PATCH,
        "num_input_channels": channels,
        "prediction_length": HORIZON,
        "use_cls_token": True,
        "norm_type": "batchnorm",
        "loss": "mse",
        "scaling": "std",
        "activation_function": "gelu",
        "pre_norm": True,
        "share_embedding": True,
        "share_projection": True,
    }
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    tensors = {
        "model.encoder.embedder.input_embedding.weight": _rand(rng, D_MODEL, PATCH),
        "model.encoder.embedder.input_embedding.bias": _rand(rng, D_MODEL),
        "model.encoder.positional_encoder.position_enc": _rand(rng, NUM_PATCHES + 1, D_MODEL),
        "model.encoder.positional_encoder.cls_token": _rand(rng, 1, 1, 1, D_MODEL),
        "head.projection.weight": _rand(rng, HORIZON, D_MODEL),
        "head.projection.bias": _rand(rng, HORIZON),
    }
    for layer in range(LAYERS):
        prefix = f"model.encoder.layers.{layer}"
        for norm in ("norm_sublayer1", "norm_sublayer3"):
            tensors[f"{prefix}.{norm}.batchnorm.weight"] = np.ones(D_MODEL, np.float32)
            tensors[f"{prefix}.{norm}.batchnorm.bias"] = np.zeros(D_MODEL, np.float32)
            tensors[f"{prefix}.{norm}.batchnorm.running_mean"] = np.zeros(D_MODEL, np.float32)
            tensors[f"{prefix}.{norm}.batchnorm.running_var"] = np.ones(D_MODEL, np.float32)
        for part in ("q_proj", "k_proj", "v_proj", "out_proj"):
            tensors[f"{prefix}.self_attn.{part}.weight"] = _rand(rng, D_MODEL, D_MODEL)
            tensors[f"{prefix}.self_attn.{part}.bias"] = _rand(rng, D_MODEL)
        tensors[f"{prefix}.ff.0.weight"] = _rand(rng, FFN, D_MODEL)
        tensors[f"{prefix}.ff.0.bias"] = _rand(rng, FFN)
        tensors[f"{prefix}.ff.3.weight"] = _rand(rng, D_MODEL, FFN)
        tensors[f"{prefix}.ff.3.bias"] = _rand(rng, D_MODEL)
    safetensors_numpy.save_file(tensors, str(path / "model.safetensors"))


def _network_constants(path: Path, precision: str, monkeypatch) -> list[tuple[int, ...]]:
    """Shapes of every constant in the network the real build path defines (no engine build)."""
    config = ModelConfig.from_dir(path)
    config.raw["_fp32_layers"] = []
    weights = model_module._load_all_tensors(path, precision=precision, depth=LAYERS)
    weights["_task_type"] = model_module._normalize_task_type(config.raw)
    seen = {}

    def capture(builder, network, **_):
        seen["shapes"] = [
            tuple(layer.get_output(0).shape)
            for layer in (network.get_layer(index) for index in range(network.num_layers))
            if layer.type == trt.LayerType.CONSTANT
        ]
        return b""

    monkeypatch.setattr(model_module, "build_serialized_network", capture)
    model_module._build_patchtst_network(config, weights, precision=precision, verbose=False)
    return seen["shapes"]


@pytest.mark.gpu
@pytest.mark.trt
@pytest.mark.parametrize("precision", ["fp16", "fp32"])
def test_one_weight_constant_per_linear_for_any_channel_count(
    tmp_path: Path, monkeypatch, precision: str
) -> None:
    """The encoder and head weight constants do not grow with the number of channels."""
    counts = {}
    for channels in (1, 7):
        path = tmp_path / f"ch{channels}"
        path.mkdir()
        _write_checkpoint(path, channels)
        shapes = _network_constants(path, precision, monkeypatch)
        counts[channels] = (
            sum(1 for shape in shapes if shape in LINEAR_WEIGHT_SHAPES),
            sum(1 for shape in shapes if shape == (D_MODEL, HORIZON)),
        )
    # 6 encoder linears per layer (q, k, v, out, ff.0, ff.3) and one head projection.
    assert counts[1] == counts[7] == (6 * LAYERS, 1)
