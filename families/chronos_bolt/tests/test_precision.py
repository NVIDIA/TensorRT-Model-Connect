# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chronos-Bolt build precision: fp32 (default) and fp16 are accepted, other precisions are not."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors.numpy import save_file

from .. import model as model_module

D_MODEL, D_FF, LAYERS, HEADS, PATCH, CONTEXT, HORIZON = 64, 128, 2, 4, 16, 256, 16
QUANTILES = [0.1, 0.5, 0.9]


def _rand(rng, *shape):
    return (rng.standard_normal(shape) * 0.05).astype(np.float32)


def _write_tiny_chronos_bolt(path: Path, seed: int = 0) -> None:
    """Write a small random Chronos-Bolt checkpoint (HF layout) and its config."""
    rng = np.random.default_rng(seed)
    config = {
        "architectures": ["ChronosBoltModelForForecasting"],
        "model_type": "t5",
        "d_model": D_MODEL,
        "d_ff": D_FF,
        "d_kv": D_MODEL // HEADS,
        "num_heads": HEADS,
        "num_layers": LAYERS,
        "num_decoder_layers": LAYERS,
        "dense_act_fn": "relu",
        "layer_norm_epsilon": 1e-6,
        "relative_attention_num_buckets": 32,
        "relative_attention_max_distance": 128,
        "vocab_size": 2,
        "chronos_config": {
            "context_length": CONTEXT,
            "input_patch_size": PATCH,
            "input_patch_stride": PATCH,
            "prediction_length": HORIZON,
            "quantiles": QUANTILES,
            "use_reg_token": True,
        },
    }
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    tensors = {
        "shared.weight": _rand(rng, 2, D_MODEL),
        "encoder.final_layer_norm.weight": np.ones(D_MODEL, np.float32),
        "decoder.final_layer_norm.weight": np.ones(D_MODEL, np.float32),
        "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight": _rand(
            rng, 32, HEADS
        ),
    }
    for prefix, inputs, outputs in (
        ("input_patch_embedding", 2 * PATCH, D_MODEL),
        ("output_patch_embedding", D_MODEL, HORIZON * len(QUANTILES)),
    ):
        tensors[f"{prefix}.hidden_layer.weight"] = _rand(rng, D_FF, inputs)
        tensors[f"{prefix}.hidden_layer.bias"] = _rand(rng, D_FF)
        tensors[f"{prefix}.output_layer.weight"] = _rand(rng, outputs, D_FF)
        tensors[f"{prefix}.output_layer.bias"] = _rand(rng, outputs)
        tensors[f"{prefix}.residual_layer.weight"] = _rand(rng, outputs, inputs)
        tensors[f"{prefix}.residual_layer.bias"] = _rand(rng, outputs)
    for side, attention in (
        ("encoder", ("SelfAttention",)),
        ("decoder", ("SelfAttention", "EncDecAttention")),
    ):
        for block in range(LAYERS):
            for index, kind in enumerate(attention):
                prefix = f"{side}.block.{block}.layer.{index}"
                tensors[f"{prefix}.layer_norm.weight"] = np.ones(D_MODEL, np.float32)
                for part in ("q", "k", "v", "o"):
                    tensors[f"{prefix}.{kind}.{part}.weight"] = _rand(rng, D_MODEL, D_MODEL)
            prefix = f"{side}.block.{block}.layer.{len(attention)}"
            tensors[f"{prefix}.layer_norm.weight"] = np.ones(D_MODEL, np.float32)
            tensors[f"{prefix}.DenseReluDense.wi.weight"] = _rand(rng, D_FF, D_MODEL)
            tensors[f"{prefix}.DenseReluDense.wo.weight"] = _rand(rng, D_MODEL, D_FF)
    save_file(tensors, str(path / "model.safetensors"))


class _Writer:
    """Collects what ``build`` writes."""

    def __init__(self):
        self.files = {}

    def set_header(self, **_):
        pass

    def add_bytes(self, name, data):
        self.files[name] = data

    def add_json(self, name, data):
        self.files[name] = data


def _request(model_dir: Path, **overrides):
    values = dict(
        dynamic_kv_cache=False,
        image_height=None,
        image_width=None,
        video_num_frames=None,
        max_batch_size=1,
        context_parallel_size=1,
        task="series_to_quantile_forecast",
        model_dir=str(model_dir),
        precision="fp32",
        max_sequence_length=None,
        quantization=None,
        fp32_layers=(),
        mtp_seq_len=None,
        tensor_parallel_size=1,
        verbose=False,
        backend="trt",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.fixture
def tiny(tmp_path: Path) -> Path:
    _write_tiny_chronos_bolt(tmp_path)
    return tmp_path


def _capture_precision(monkeypatch) -> list[str]:
    seen: list[str] = []

    def fake_network(config, weights, *, precision, verbose):
        seen.append(precision)
        return b"plan"

    monkeypatch.setattr(model_module, "_build_chronos_network", fake_network)
    return seen


@pytest.mark.parametrize("precision", ["fp32", "fp16", "FP16"])
def test_fp32_and_fp16_reach_the_network_builder(tiny, monkeypatch, precision) -> None:
    """The requested work type is passed to the network builder; the bundle layout is unchanged."""
    seen = _capture_precision(monkeypatch)
    writer = _Writer()
    model_module.build(_request(tiny, precision=precision), writer)
    assert seen == [precision.lower()]
    assert writer.files["engine.plan"] == b"plan"
    assert writer.files["runtime.json"]["context_length"] == CONTEXT
    assert writer.files["runtime.json"]["prediction_length"] == HORIZON


@pytest.mark.parametrize("precision", ["bf16", "fp8", "int8"])
def test_other_precisions_are_refused(tiny, monkeypatch, precision) -> None:
    """Only fp32 and fp16 builds are measured; anything else is refused before building."""
    seen = _capture_precision(monkeypatch)
    with pytest.raises(ValueError, match="'fp32' or 'fp16'"):
        model_module.build(_request(tiny, precision=precision), _Writer())
    assert seen == []


@pytest.mark.gpu
@pytest.mark.trt
def test_an_fp16_engine_keeps_the_io_and_the_values(tiny) -> None:
    """fp32 and fp16 engines of one checkpoint have the same float32 I/O and close forecasts."""
    trt = pytest.importorskip("tensorrt")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    rng = np.random.default_rng(1)
    context = np.full((1, CONTEXT), np.nan, np.float32)
    context[0, -100:] = 50.0 + np.cumsum(rng.standard_normal(100)).astype(np.float32)
    outputs, io = {}, {}
    for precision in ("fp32", "fp16"):
        writer = _Writer()
        model_module.build(_request(tiny, precision=precision), writer)
        engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(
            writer.files["engine.plan"]
        )
        names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        io[precision] = [
            (name, engine.get_tensor_dtype(name), tuple(engine.get_tensor_shape(name)))
            for name in names
        ]
        buffers = {
            "context": torch.tensor(context, device="cuda"),
            "quantile_preds": torch.zeros(
                tuple(engine.get_tensor_shape("quantile_preds")), device="cuda"
            ),
        }
        execution = engine.create_execution_context()
        for name, tensor in buffers.items():
            execution.set_tensor_address(name, tensor.data_ptr())
        stream = torch.cuda.Stream()
        execution.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        outputs[precision] = buffers["quantile_preds"].cpu().numpy()
    assert io["fp16"] == io["fp32"]
    assert io["fp32"] == [
        ("context", trt.float32, (1, CONTEXT)),
        ("quantile_preds", trt.float32, (1, len(QUANTILES), HORIZON)),
    ]
    assert np.isfinite(outputs["fp16"]).all()
    reference = outputs["fp32"]
    error = np.linalg.norm(outputs["fp16"] - reference) / np.linalg.norm(reference)
    assert error < 1e-2
