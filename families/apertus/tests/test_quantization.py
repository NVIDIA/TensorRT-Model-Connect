# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Apertus FP8 self-quantization: recipe, scales, layer selection and refusals."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("tensorrt")
ml_dtypes = pytest.importorskip("ml_dtypes")

from families.apertus import calibrate_fp8_scales as calibration  # noqa: E402
from families.apertus import quantization as Q  # noqa: E402
from families.apertus.config import parse_config  # noqa: E402
from families.apertus.tests.test_builder import (  # noqa: E402
    CHECKPOINT_CONFIG,
    CHECKPOINT_GENERATION,
    TINY,
    _request,
    _tiny_tensors,
    _write_checkpoint,
)

KEY_8B = "apertus-h4096-l32-i21504-kv8-v131072"


def _layer(rng, hidden=32, mlp=64):
    def w(out, inp):
        return rng.standard_normal((out, inp)).astype(ml_dtypes.bfloat16)

    return SimpleNamespace(
        q_proj=w(hidden, hidden),
        k_proj=w(hidden // 2, hidden),
        v_proj=w(hidden // 2, hidden),
        o_proj=w(hidden, hidden),
        up_proj=w(mlp, hidden),
        down_proj=w(hidden, mlp),
    )


def _scales(layers, value=0.5, **overrides):
    table = {Q.projection_name(i, p): value for i in range(layers) for p in Q.PROJECTIONS}
    table.update(overrides)
    return table


def test_scale_key_of_the_8b_checkpoint():
    assert Q.scale_key(parse_config(CHECKPOINT_CONFIG, CHECKPOINT_GENERATION)) == KEY_8B


def test_committed_scale_file_covers_every_projection_of_the_8b_model():
    table = json.loads((Path(Q.__file__).parent / Q.ACTIVATION_SCALES_FILENAME).read_text())
    assert set(table) == {KEY_8B}
    entries = table[KEY_8B]
    expected = {Q.projection_name(i, p) for i in range(32) for p in Q.PROJECTIONS}
    assert set(entries) == expected
    nulls = sorted(name for name, value in entries.items() if value is None)
    assert nulls == [
        "layers.2.mlp.down_proj",
        "layers.28.mlp.down_proj",
        "layers.29.mlp.down_proj",
        "layers.30.self_attn.o_proj",
        "layers.31.mlp.down_proj",
        "layers.31.self_attn.o_proj",
    ]
    assert all(value > 0 and np.isfinite(value) for value in entries.values() if value is not None)
    # q, k and v read the same input, so the calibration gives them one scale.
    for i in range(32):
        q = entries[f"layers.{i}.self_attn.q_proj"]
        assert (
            entries[f"layers.{i}.self_attn.k_proj"] == q == entries[f"layers.{i}.self_attn.v_proj"]
        )
    assert sum(value is not None for value in entries.values()) == 186


def test_unknown_model_config_asks_for_calibration():
    config = parse_config(TINY, CHECKPOINT_GENERATION | {"eos_token_id": [2]})
    with pytest.raises(NotImplementedError, match="calibrate_fp8_scales"):
        Q.load_activation_scales(config)


def test_fp8_weight_scale_and_bytes():
    rng = np.random.default_rng(0)
    weight = rng.standard_normal((8, 32)).astype(np.float32)
    packed, scale = Q.quantize_fp8_weight(weight)
    assert scale == pytest.approx(np.abs(weight).max() / 448.0)
    assert packed.dtype == np.uint8 and packed.shape == weight.shape
    restored = packed.view(ml_dtypes.float8_e4m3fn).astype(np.float32) * scale
    assert np.abs(restored - weight).max() <= np.abs(weight).max() / 16
    with pytest.raises(ValueError):
        Q.quantize_fp8_weight(np.zeros((4, 16), np.float32))


def _context(monkeypatch, fmt, table, layers=2):
    monkeypatch.setattr(Q, "load_activation_scales", lambda config: table)
    rng = np.random.default_rng(2)
    weights = SimpleNamespace(layers=[_layer(rng) for _ in range(layers)])
    return Q.build_quant_context(SimpleNamespace(), weights, fmt)


def test_fp8_context_quantizes_every_projection_with_the_stored_scale(monkeypatch):
    ctx = _context(monkeypatch, "fp8", _scales(2, 0.25))
    assert set(ctx.weights) == {Q.projection_name(i, p) for i in range(2) for p in Q.PROJECTIONS}
    assert all(isinstance(w, Q.FP8Weight) and w.input_scale == 0.25 for w in ctx.weights.values())


def test_null_scale_keeps_the_projection_bf16(monkeypatch):
    table = _scales(2, **{"layers.1.mlp.down_proj": None})
    ctx = _context(monkeypatch, "fp8", table)
    assert "layers.1.mlp.down_proj" not in ctx.weights
    assert len(ctx.weights) == 11


@pytest.mark.parametrize(
    "table",
    [
        _scales(2, **{"layers.0.self_attn.o_proj": 0.0}),
        _scales(2, **{"layers.0.self_attn.o_proj": float("nan")}),
        {k: v for k, v in _scales(2).items() if k != "layers.1.mlp.up_proj"},
    ],
)
def test_bad_or_missing_scales_are_refused(monkeypatch, table):
    with pytest.raises(ValueError):
        _context(monkeypatch, "fp8", table)


@pytest.mark.parametrize("fmt", ["int8", "nvfp4"])
def test_unknown_format_is_refused(monkeypatch, fmt):
    with pytest.raises(ValueError, match="quantization must be one of"):
        _context(monkeypatch, fmt, _scales(2))


@pytest.mark.parametrize(
    ("update", "error"),
    [
        ({"quantization": "int4"}, NotImplementedError),
        ({"quantization": "nvfp4"}, NotImplementedError),
        ({"quantization": "fp8", "precision": "fp32"}, ValueError),
    ],
)
def test_build_refuses_unsupported_quantization_requests(update, error):
    from families.apertus.model import build

    class Writer:
        def __getattr__(self, name):
            raise AssertionError(f"writer.{name} must not be reached")

    with pytest.raises(error):
        build(_request(**update), Writer())


def test_quantized_checkpoint_is_refused(tmp_path):
    pytest.importorskip("safetensors")
    from families.apertus.weights import load_weights

    config = parse_config(TINY, CHECKPOINT_GENERATION | {"eos_token_id": [2]})
    tensors = _tiny_tensors(ml_dtypes)
    model_dir = _write_checkpoint(tmp_path / "plain", tensors)
    load_weights(model_dir, config, "bf16", quantized=True)  # the BF16 checkpoint is accepted

    scaled = dict(tensors)
    scaled["model.layers.0.mlp.up_proj.weight_scale"] = np.ones((1,), np.float32)
    with pytest.raises(ValueError, match="already quantized"):
        load_weights(_write_checkpoint(tmp_path / "scaled", scaled), config, "bf16", quantized=True)

    marked = _write_checkpoint(tmp_path / "marked", tensors)
    (marked / "config.json").write_text(
        json.dumps(TINY | {"quantization_config": {"quant_method": "compressed-tensors"}})
    )
    with pytest.raises(ValueError, match="already quantized"):
        load_weights(marked, config, "bf16", quantized=True)


def test_calibration_names_and_texts():
    assert calibration.module_projection("model.layers.7.self_attn.o_proj") == (
        "layers.7.self_attn.o_proj"
    )
    assert calibration.module_projection("model.layers.7.mlp.up_proj") == "layers.7.mlp.up_proj"
    assert calibration.module_projection("model.layers.7.mlp.act_fn") is None
    assert calibration.module_projection("lm_head") is None
    assert (
        calibration.conversation_text(
            [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
        )
        == "a\n\nb"
    )
    assert calibration.conversation_text([{"role": "user", "content": "a"}]) is None
    kept, excluded = calibration.split_outlier_scales(
        {f"layers.{i}.mlp.down_proj": 1.0 for i in range(9)} | {"layers.9.mlp.down_proj": 30.0}
    )
    assert excluded == ["layers.9.mlp.down_proj"] and len(kept) == 9


def _gpu_config():
    # 4 heads of 16; every projection is wide enough for FP8 GEMM kernels.
    raw = TINY | {
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 64,
    }
    return parse_config(raw, CHECKPOINT_GENERATION | {"eos_token_id": [2]})


def _gpu_weights(config):
    from families.apertus.weights import ApertusWeights, LayerWeights

    rng = np.random.default_rng(3)

    def bf16(*shape):
        return (0.1 * rng.standard_normal(shape)).astype(ml_dtypes.bfloat16)

    hidden, attention, kv = config.hidden_size, config.attention_size, config.kv_size
    layers = tuple(
        LayerWeights(
            attention_norm=bf16(hidden) + 1,
            q_proj=bf16(attention, hidden),
            k_proj=bf16(kv, hidden),
            v_proj=bf16(kv, hidden),
            o_proj=bf16(hidden, attention),
            q_norm=bf16(config.head_dim) + 1,
            k_norm=bf16(config.head_dim) + 1,
            feedforward_norm=bf16(hidden) + 1,
            up_proj=bf16(config.intermediate_size, hidden),
            down_proj=bf16(hidden, config.intermediate_size),
            xielu_alpha_p=0.8,
            xielu_alpha_n=0.8,
            xielu_beta=0.5,
            xielu_eps=-1e-6,
        )
        for _ in range(config.num_hidden_layers)
    )
    return ApertusWeights(
        embedding=bf16(config.vocab_size, hidden),
        layers=layers,
        final_norm=bf16(hidden) + 1,
        lm_head=bf16(config.vocab_size, hidden),
    )


def _tiny_plan(monkeypatch, fmt):
    import tensorrt as trt

    from families.apertus.graph import build_engine

    config = _gpu_config()
    weights = _gpu_weights(config)
    quant = None
    if fmt is not None:
        table = _scales(config.num_hidden_layers, 0.05)
        monkeypatch.setattr(Q, "load_activation_scales", lambda c: table)
        quant = Q.build_quant_context(config, weights, fmt)
    plan = build_engine(
        config, weights, precision="bf16", max_cache_length=64, verbose=False, quant=quant
    )
    engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(plan)
    names = sorted(engine.get_tensor_name(i) for i in range(engine.num_io_tensors))
    layers = engine.create_engine_inspector().get_engine_information(
        trt.LayerInformationFormat.JSON
    )
    return plan, names, layers.lower()


def test_tiny_fp8_engine_keeps_the_tensor_contract(monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("FP8 engines need an Ada, Hopper or Blackwell GPU")
    _, names16, layers16 = _tiny_plan(monkeypatch, None)
    _, names8, layers8 = _tiny_plan(monkeypatch, "fp8")
    assert names8 == names16
    assert "e4m3" not in layers16 and "fp8" not in layers16
    assert "e4m3" in layers8 or "fp8" in layers8
