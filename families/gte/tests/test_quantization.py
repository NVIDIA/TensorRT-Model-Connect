# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GTE FP8 quantization: weight scaling, projection coverage, calibration tool, engine build."""

from __future__ import annotations

import json
import re
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")

from .. import calibrate_fp8_scales, config as config_module, model  # noqa: E402
from .. import quantization as quantization_module  # noqa: E402
from ..quantization import (  # noqa: E402
    ACTIVATION_SCALES_FILENAME,
    FP8_E4M3_MAX,
    calibrate_gte,
    projections,
    quantize_fp8_weight,
    scale_key,
)

D, FFN, LAYERS, HEADS, VOCAB, SEQ = 128, 256, 2, 4, 64, 16


def _tiny_weights(seed: int = 0) -> dict[str, np.ndarray]:
    """Random float32 weights with every tensor the family reads, at a tiny geometry."""
    rng = np.random.default_rng(seed)

    def normal(*shape, std=0.05):
        return rng.normal(0.0, std, shape).astype(np.float32)

    def norm(prefix):
        return {f"{prefix}.weight": 1.0 + normal(D), f"{prefix}.bias": normal(D)}

    weights = {"embeddings.word_embeddings.weight": normal(VOCAB, D, std=1.0)}
    weights.update(norm("embeddings.LayerNorm"))
    for layer in range(LAYERS):
        prefix = f"encoder.layer.{layer}"
        weights[f"{prefix}.attention.qkv_proj.weight"] = normal(3 * D, D)
        weights[f"{prefix}.attention.qkv_proj.bias"] = normal(3 * D)
        weights[f"{prefix}.attention.o_proj.weight"] = normal(D, D)
        weights[f"{prefix}.attention.o_proj.bias"] = normal(D)
        weights[f"{prefix}.mlp.up_gate_proj.weight"] = normal(2 * FFN, D)
        weights[f"{prefix}.mlp.down_proj.weight"] = normal(D, FFN)
        weights[f"{prefix}.mlp.down_proj.bias"] = normal(D)
        weights.update(norm(f"{prefix}.attn_ln"))
        weights.update(norm(f"{prefix}.mlp_ln"))
    return weights


def _config(tmp_path: Path, weights: dict) -> dict:
    """Resolve the tiny geometry the way ``build`` does."""
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "new",
                "hidden_act": "gelu",
                "layer_norm_type": "layer_norm",
                "position_embedding_type": "rope",
                "num_attention_heads": HEADS,
                "type_vocab_size": 0,
                "max_position_embeddings": 8192,
                "rope_theta": 500000,
                "rope_scaling": {"type": "ntk", "factor": 2.0},
                "pack_qkv": True,
            }
        )
    )
    return config_module.resolve(tmp_path, weights)


def _all_scales(cfg: dict, value: float = 0.05) -> dict[str, float]:
    return {name: value for name in projections(cfg)}


@pytest.fixture
def tiny(tmp_path: Path, monkeypatch):
    """Tiny weights, their resolved geometry, and activation scales for every projection."""
    weights = _tiny_weights()
    cfg = _config(tmp_path, weights)
    scales = _all_scales(cfg)
    monkeypatch.setattr(
        quantization_module, "load_activation_scales", lambda _cfg, _weights: scales
    )
    return weights, cfg


def test_fp8_weight_scale_is_amax_over_448() -> None:
    """The scale is max|w| / 448 and the stored bytes round-trip to the weight."""
    weight = np.random.default_rng(1).normal(0.0, 0.1, (16, 32)).astype(np.float32)
    packed, scale = quantize_fp8_weight(weight)
    assert scale == pytest.approx(float(np.abs(weight).max()) / 448.0)
    assert packed.dtype == np.uint8 and packed.shape == weight.shape
    restored = packed.view(ml_dtypes.float8_e4m3fn).astype(np.float32) * scale
    assert np.abs(restored - weight).max() <= 0.07 * np.abs(weight).max()


def test_fp8_weight_rejects_all_zero() -> None:
    """A weight with no positive amax has no scale."""
    with pytest.raises(ValueError, match="amax"):
        quantize_fp8_weight(np.zeros((4, 4), np.float32))


def test_every_projection_is_quantized(tiny) -> None:
    """Four projections per layer get an FP8 weight of the checkpoint's shape."""
    weights, cfg = tiny
    ctx = calibrate_gte(weights, cfg, "fp8")
    assert len(ctx.scales) == LAYERS * 4 == len(projections(cfg))
    assert ctx.scales["encoder.layer.1.mlp.up_gate_proj"].weight.packed.shape == (2 * FFN, D)
    assert ctx.scales["encoder.layer.1.mlp.down_proj"].weight.packed.shape == (D, FFN)
    assert ctx.scales["encoder.layer.0.attention.qkv_proj"].weight.packed.shape == (3 * D, D)
    assert ctx.has_fp8_weights
    assert not ctx.should_quantize("embeddings.word_embeddings")
    assert ctx.matmul("embeddings.word_embeddings") is None


def test_qkv_parts_are_row_blocks_with_the_packed_scales(tiny) -> None:
    """q, k and v are the contiguous row blocks of qkv_proj and keep its two scales."""
    weights, cfg = tiny
    ctx = calibrate_gte(weights, cfg, "fp8")
    entry = ctx.scales["encoder.layer.0.attention.qkv_proj"]
    packed = weights["encoder.layer.0.attention.qkv_proj.weight"]
    assert entry.weight.weight_scale == pytest.approx(float(np.abs(packed).max()) / 448.0)
    for index, part in enumerate(quantization_module.QKV_PARTS):
        block = entry.weight.rows(index * D, (index + 1) * D)
        assert block.packed.shape == (D, D) and block.packed.flags["C_CONTIGUOUS"]
        assert np.array_equal(block.packed, entry.weight.packed[index * D : (index + 1) * D])
        assert block.weight_scale == entry.weight.weight_scale
        assert ctx.matmul("encoder.layer.0.attention.qkv_proj", part) is not None
    with pytest.raises(ValueError, match="qkv part"):
        ctx.matmul("encoder.layer.0.attention.qkv_proj", "x")


def test_null_scale_keeps_the_projection_unquantized(tmp_path: Path, monkeypatch) -> None:
    """An outlier projection is stored as null and stays in the work precision."""
    weights = _tiny_weights()
    cfg = _config(tmp_path, weights)
    scales = _all_scales(cfg)
    outlier = "encoder.layer.0.mlp.down_proj"
    scales[outlier] = None
    monkeypatch.setattr(quantization_module, "load_activation_scales", lambda _c, _w: scales)
    ctx = calibrate_gte(weights, cfg, "fp8")
    assert not ctx.should_quantize(outlier)
    assert ctx.matmul(outlier) is None
    assert ctx.should_quantize("encoder.layer.0.mlp.up_gate_proj")


def test_missing_scale_raises_and_names_the_tool(tmp_path: Path, monkeypatch) -> None:
    """A table without an entry for a projection is an error, not a silent fallback."""
    weights = _tiny_weights()
    cfg = _config(tmp_path, weights)
    scales = _all_scales(cfg)
    del scales["encoder.layer.0.attention.o_proj"]
    monkeypatch.setattr(quantization_module, "load_activation_scales", lambda _c, _w: scales)
    with pytest.raises(ValueError, match="calibrate_fp8_scales"):
        calibrate_gte(weights, cfg, "fp8")


def test_scale_without_projection_raises(tmp_path: Path, monkeypatch) -> None:
    """A table entry that matches no projection means the table is for another model."""
    weights = _tiny_weights()
    cfg = _config(tmp_path, weights)
    scales = _all_scales(cfg)
    scales["encoder.layer.9.mlp.down_proj"] = 0.05
    monkeypatch.setattr(quantization_module, "load_activation_scales", lambda _c, _w: scales)
    with pytest.raises(ValueError, match="without a projection"):
        calibrate_gte(weights, cfg, "fp8")


def test_already_quantized_checkpoint_is_rejected(tiny) -> None:
    """A checkpoint that already stores weight scales is not quantized a second time."""
    weights, cfg = tiny
    weights["encoder.layer.0.mlp.down_proj.weight_scale"] = np.ones((1,), np.float32)
    with pytest.raises(ValueError, match="already quantized"):
        calibrate_gte(weights, cfg, "fp8")


def test_unsupported_format_is_rejected(tiny) -> None:
    """Only fp8 is implemented."""
    weights, cfg = tiny
    with pytest.raises(NotImplementedError, match="nvfp4"):
        calibrate_gte(weights, cfg, "nvfp4")


def test_unknown_model_config_names_the_calibration_tool(tmp_path: Path) -> None:
    """A checkpoint with no committed scales fails with the available keys and the tool."""
    weights = _tiny_weights()
    cfg = _config(tmp_path, weights)
    with pytest.raises(NotImplementedError, match="calibrate_fp8_scales") as error:
        calibrate_gte(weights, cfg, "fp8")
    assert "no calibrated FP8 activation scales" in str(error.value)


def test_scale_key_holds_the_model_config_and_the_weights(tmp_path: Path) -> None:
    """The key names the geometry and changes when the weights change (a fine-tune)."""
    weights = _tiny_weights()
    cfg = _config(tmp_path, weights)
    key = scale_key(cfg, weights)
    assert key.startswith(f"gte-h{D}-l{LAYERS}-i{FFN}-n{HEADS}-w")
    assert scale_key(cfg, _tiny_weights(seed=1)) != key


def test_committed_scales_cover_every_projection() -> None:
    """Every committed checkpoint has an entry per projection; non-null scales are positive."""
    table = json.loads(
        (Path(quantization_module.__file__).parent / ACTIVATION_SCALES_FILENAME).read_text()
    )
    assert table
    for key, scales in table.items():
        match = re.fullmatch(r"gte-h(\d+)-l(\d+)-i(\d+)-n(\d+)-w[-+.e\d]+", key)
        assert match, key
        layers = int(match.group(2))
        assert len(scales) == layers * len(quantization_module.LAYER_PROJECTIONS), key
        for layer in range(layers):
            for stem in quantization_module.LAYER_PROJECTIONS:
                assert f"encoder.layer.{layer}.{stem}" in scales, (key, layer, stem)
        values = [v for v in scales.values() if v is not None]
        assert all(np.isfinite(v) and v > 0 for v in values), key
        # Outliers are the exception: at most a few projections stay unquantized.
        assert len(values) >= 0.9 * len(scales), key


def test_calibration_tool_maps_module_names_to_checkpoint_names() -> None:
    """Only the four projections of the encoder layers are calibrated."""
    name = calibrate_fp8_scales.projection_name
    assert name("encoder.layer.3.mlp.down_proj") == "encoder.layer.3.mlp.down_proj"
    assert name("encoder.layer.0.mlp.up_gate_proj") == "encoder.layer.0.mlp.up_gate_proj"
    assert name("encoder.layer.11.attention.qkv_proj") == "encoder.layer.11.attention.qkv_proj"
    assert name("new.encoder.layer.2.attention.o_proj") == "encoder.layer.2.attention.o_proj"
    for other in (
        "embeddings.word_embeddings",
        "embeddings.LayerNorm",
        "encoder.layer.0.attention",
        "encoder.layer.0.attn_ln",
        "encoder.layer.0.mlp.down_proj.input_quantizer",
        "pooler.dense",
    ):
        assert name(other) is None, other


def test_outliers_are_more_than_twenty_times_the_median_of_their_kind() -> None:
    """The outlier rule compares a projection with the same projection of the other layers."""
    scales = {}
    for layer in range(5):
        for stem in quantization_module.LAYER_PROJECTIONS:
            scales[f"encoder.layer.{layer}.{stem}"] = 0.01
    scales["encoder.layer.2.mlp.down_proj"] = 0.21
    scales["encoder.layer.3.mlp.up_gate_proj"] = 0.19
    marked = calibrate_fp8_scales.mark_outliers(scales)
    assert [name for name, value in marked.items() if value is None] == [
        "encoder.layer.2.mlp.down_proj"
    ]


def test_calibration_tool_cli_parses_both_commands(monkeypatch, tmp_path: Path) -> None:
    """The fetch and calibrate commands reach their functions with the parsed arguments."""
    calls = []
    monkeypatch.setattr(calibrate_fp8_scales, "fetch_texts", lambda *args: calls.append(args))
    monkeypatch.setattr(calibrate_fp8_scales, "calibrate", lambda *args: calls.append(args))
    monkeypatch.setattr("sys.argv", ["calibrate", "fetch", "--out", str(tmp_path / "t.jsonl")])
    calibrate_fp8_scales.main()
    monkeypatch.setattr(
        "sys.argv",
        ["calibrate", "calibrate", "--model-dir", str(tmp_path), "--texts", str(tmp_path)],
    )
    calibrate_fp8_scales.main()
    assert calls == [(tmp_path / "t.jsonl",), (tmp_path, tmp_path, 8192)]


def test_calibration_texts_round_trip(tmp_path: Path) -> None:
    """``load_texts`` reads the ``text`` field of every line ``fetch`` writes."""
    path = tmp_path / "texts.jsonl"
    path.write_text(
        json.dumps({"source": "a", "text": "first"})
        + "\n\n"
        + json.dumps({"source": "b", "text": "second"})
        + "\n"
    )
    assert calibrate_fp8_scales.load_texts(path) == ["first", "second"]


class _Engine:
    """A deserialized plan that can be run on CUDA tensors."""

    def __init__(self, plan: bytes) -> None:
        self.engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(plan)
        assert self.engine is not None
        self.context = self.engine.create_execution_context()

    def run(self, torch, ids, mask):
        out = torch.empty(tuple(self.engine.get_tensor_shape("last_hidden_state")), device="cuda")
        self.context.set_tensor_address("input_ids", ids.data_ptr())
        self.context.set_tensor_address("attention_mask", mask.data_ptr())
        self.context.set_tensor_address("last_hidden_state", out.data_ptr())
        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        return out


@pytest.mark.gpu
@pytest.mark.trt
def test_quantized_engine_builds_and_matches_fp16(tiny) -> None:
    """The FP8 engine builds on the fp16 graph, is smaller, and tracks the fp16 engine."""
    torch = pytest.importorskip("torch")
    weights, cfg = tiny
    ctx = calibrate_gte(weights, cfg, "fp8")
    generator = torch.Generator().manual_seed(0)
    ids = torch.randint(0, VOCAB, (SEQ,), generator=generator, dtype=torch.int32).cuda()
    mask = torch.ones(SEQ, dtype=torch.int32, device="cuda")
    mask[SEQ - 4 :] = 0

    plain = model.build_engine(weights, cfg, seq_len=SEQ, precision="fp16")
    quant = model.build_engine(weights, cfg, seq_len=SEQ, precision="fp16", quant_ctx=ctx)
    assert len(quant) < len(plain)

    reference = _Engine(plain).run(torch, ids, mask)[: SEQ - 4].reshape(-1)
    actual = _Engine(quant).run(torch, ids, mask)[: SEQ - 4].reshape(-1)
    assert torch.isfinite(actual).all()
    cosine = torch.nn.functional.cosine_similarity(reference, actual, dim=0).item()
    assert cosine > 0.9, cosine


@pytest.mark.gpu
@pytest.mark.trt
def test_fp8_gemm_accumulates_beyond_the_fp16_range() -> None:
    """With fp16 Dequantize the FP8 GEMM keeps a raw accumulator far above 65504.

    Activation and weight both quantize to 448, so the raw sum is 448 * 448 * 768 (1.5e8)
    and the scaled output is 768. An fp16 accumulator would give inf.
    """
    torch = pytest.importorskip("torch")
    rows, width, out_features = 64, 768, 256
    scale = 1.0 / FP8_E4M3_MAX
    packed, weight_scale = quantize_fp8_weight(np.full((out_features, width), 1.0, np.float32))
    assert weight_scale == pytest.approx(scale)
    entry = quantization_module._LayerScales(
        scale, quantization_module._FP8Weight(packed, weight_scale, out_features, width)
    )

    builder = trt.Builder(trt.Logger(trt.Logger.ERROR))
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    activation = network.add_input("x", trt.float16, (rows, width))
    keep_alive: list = []
    output = quantization_module._wrap_fp8_matmul(network, activation, entry, keep_alive=keep_alive)
    assert output.dtype == trt.float16
    output.name = "y"
    network.mark_output(output)
    plan = builder.build_serialized_network(network, builder.create_builder_config())
    engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(plan)
    context = engine.create_execution_context()
    x = torch.ones((rows, width), dtype=torch.float16, device="cuda")
    y = torch.empty((rows, out_features), dtype=torch.float16, device="cuda")
    context.set_tensor_address("x", x.data_ptr())
    context.set_tensor_address("y", y.data_ptr())
    context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    assert torch.isfinite(y).all()
    # 1/448 is not exact in fp16; the two scales round by about 3e-4 each.
    assert torch.allclose(y.float(), torch.full_like(y.float(), float(width)), rtol=2e-3)


@pytest.mark.gpu
@pytest.mark.trt
def test_fp8_build_options_keep_layer_names() -> None:
    """FP8 builds ask for detailed profiling so the plan inspector shows kernel names."""
    ctx = quantization_module.GteQuantContext(scales={"x": object()})
    _, _, config = model._builder(False, ctx)
    assert config.profiling_verbosity == trt.ProfilingVerbosity.DETAILED
    _, _, plain = model._builder(False)
    assert plain.profiling_verbosity != trt.ProfilingVerbosity.DETAILED


@pytest.mark.parametrize(
    ("precision", "message"), [("fp32", "fp8 with --precision fp16 only"), ("bf16", "fp32 or fp16")]
)
def test_build_requires_fp16_for_fp8(tmp_path: Path, precision: str, message: str) -> None:
    """fp8 is built on the fp16 graph: other precisions are refused before the checkpoint is read."""
    from tensorrt_model_connect import BuildRequest

    request = BuildRequest(
        model_dir=tmp_path,
        output_path=tmp_path / "b",
        family="gte",
        task="text_to_embedding",
        precision=precision,
        quantization="fp8",
        max_sequence_length=SEQ,
    )
    with pytest.raises(NotImplementedError, match=message):
        model.build(request, None)


def test_build_refuses_other_quantization_modes(tmp_path: Path) -> None:
    """Only fp8 and none are accepted."""
    from tensorrt_model_connect import BuildRequest

    request = BuildRequest(
        model_dir=tmp_path,
        output_path=tmp_path / "b",
        family="gte",
        task="text_to_embedding",
        precision="fp32",
        quantization="nvfp4",
        max_sequence_length=SEQ,
    )
    with pytest.raises(NotImplementedError, match="fp8 or none"):
        model.build(request, None)
