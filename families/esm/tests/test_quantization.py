# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ESM-2 FP8 self-quantization: weights, scales, graph wiring and the calibration tool."""

from __future__ import annotations

import json
import re
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")

from tensorrt_model_connect import BuildRequest  # noqa: E402

from .. import calibrate_fp8_scales, graph, model  # noqa: E402
from .. import quantization as quantization_module  # noqa: E402
from ..quantization import (  # noqa: E402
    ACTIVATION_SCALES_FILENAME,
    HF_SUFFIX,
    EsmQuantContext,
    calibrate_esm,
    configure_quantized_build,
    projection_names,
    quantize_fp8_weight,
    scale_key,
)

D, FFN, LAYERS, HEADS, VOCAB, SEQ = 64, 128, 2, 4, 33, 16
PREFIX = "esm."
CFG = {
    "prefix": PREFIX,
    "vocab_size": VOCAB,
    "hidden_size": D,
    "intermediate_size": FFN,
    "num_layers": LAYERS,
    "num_heads": HEADS,
    "head_dim": D // HEADS,
    "eps": 1e-5,
    "max_positions": 1026,
    "token_dropout": True,
    "mask_token_id": 32,
    "pad_token_id": 1,
}


def _tiny_weights(seed: int = 0) -> dict:
    """A small random ESM-2 checkpoint with every tensor ``model.build_engine`` reads."""
    rng = np.random.default_rng(seed)

    def mat(*shape):
        return (rng.standard_normal(shape) * 0.05).astype(np.float32)

    w = {
        f"{PREFIX}embeddings.word_embeddings.weight": mat(VOCAB, D),
        f"{PREFIX}encoder.emb_layer_norm_after.weight": np.ones(D, np.float32),
        f"{PREFIX}encoder.emb_layer_norm_after.bias": mat(D),
        "lm_head.dense.weight": mat(D, D),
        "lm_head.dense.bias": mat(D),
        "lm_head.layer_norm.weight": np.ones(D, np.float32),
        "lm_head.layer_norm.bias": mat(D),
        "lm_head.bias": mat(VOCAB),
    }
    for i in range(LAYERS):
        layer = f"{PREFIX}encoder.layer.{i}"
        for stem, suffix in HF_SUFFIX.items():
            out, inner = (FFN, D) if stem == "fc1" else (D, FFN) if stem == "fc2" else (D, D)
            w[f"{layer}.{suffix}.weight"] = mat(out, inner)
            w[f"{layer}.{suffix}.bias"] = mat(out)
        for name in ("attention.LayerNorm", "LayerNorm"):
            w[f"{layer}.{name}.weight"] = np.ones(D, np.float32)
            w[f"{layer}.{name}.bias"] = mat(D)
    return w


def _all_scales(value: float = 0.02) -> dict[str, float]:
    return {name: value for name in projection_names(CFG)}


@pytest.fixture
def tiny(monkeypatch):
    """Tiny weights with a scale for every projection in place of the committed table."""
    weights = _tiny_weights()
    scales = _all_scales()
    monkeypatch.setattr(quantization_module, "load_activation_scales", lambda key: scales)
    return weights, scales


def test_fp8_weight_scale_is_amax_over_448() -> None:
    """The weight scale is amax / 448 and the bytes decode back to the weight within e4m3 error."""
    weight = np.random.default_rng(1).standard_normal((32, 16)).astype(np.float32)
    packed, scale = quantize_fp8_weight(weight)
    assert scale == pytest.approx(float(np.abs(weight).max()) / 448.0)
    assert packed.dtype == np.uint8 and packed.shape == weight.shape
    decoded = packed.view(ml_dtypes.float8_e4m3fn).astype(np.float32) * scale
    np.testing.assert_allclose(decoded, weight, rtol=0.07, atol=scale)


def test_fp8_weight_rejects_all_zero() -> None:
    """A weight with no positive amax cannot be scaled."""
    with pytest.raises(ValueError, match="amax"):
        quantize_fp8_weight(np.zeros((4, 4), dtype=np.float32))


def test_every_encoder_projection_is_quantized(tiny) -> None:
    """The six projections of every layer get a scale; the lm_head does not."""
    weights, scales = tiny
    ctx = calibrate_esm(weights, CFG, graph, "fp8")
    assert set(ctx.scales) == set(scales)
    assert len(ctx.scales) == LAYERS * 6
    # The attention is explicit matmul and softmax, so its output projection is quantized too.
    assert ctx.should_quantize("layer.0.attn_out") and ctx.should_quantize("layer.1.fc2")
    assert ctx.has_fp8_weights
    assert not ctx.should_quantize("lm_head.dense") and not ctx.should_quantize("lm_head")


def test_projection_without_scale_is_rejected(tiny, monkeypatch) -> None:
    """A projection missing from the table fails loudly and names the calibration tool."""
    weights, scales = tiny
    scales = dict(scales)
    del scales["layer.0.fc1"]
    monkeypatch.setattr(quantization_module, "load_activation_scales", lambda key: scales)
    with pytest.raises(ValueError, match=r"layer\.0\.fc1.*calibrate_fp8_scales"):
        calibrate_esm(weights, CFG, graph, "fp8")


def test_null_scale_keeps_the_projection_unquantized(tiny, monkeypatch) -> None:
    """A stored null (calibration outlier) leaves that one projection unquantized."""
    weights, scales = tiny
    scales = {**scales, "layer.1.fc2": None}
    monkeypatch.setattr(quantization_module, "load_activation_scales", lambda key: scales)
    ctx = calibrate_esm(weights, CFG, graph, "fp8")
    assert not ctx.should_quantize("layer.1.fc2")
    assert len(ctx.scales) == LAYERS * 6 - 1


def test_quantized_checkpoint_is_rejected(tiny) -> None:
    """An already quantized checkpoint raises instead of being quantized twice."""
    weights, _ = tiny
    weights[f"{PREFIX}encoder.layer.0.intermediate.dense.weight_scale"] = np.ones(1, np.float32)
    with pytest.raises(ValueError, match="already quantized"):
        calibrate_esm(weights, CFG, graph, "fp8")


def test_unsupported_format_is_rejected(tiny) -> None:
    """Only fp8 is implemented."""
    weights, _ = tiny
    with pytest.raises(NotImplementedError, match="nvfp4"):
        calibrate_esm(weights, CFG, graph, "nvfp4")


def test_unknown_model_config_is_reported() -> None:
    """A checkpoint with no committed scales raises and names the calibration tool."""
    with pytest.raises(NotImplementedError, match="model config.*calibrate_fp8_scales"):
        calibrate_esm(_tiny_weights(), CFG, graph, "fp8")


def test_scale_key_separates_checkpoints_of_one_model_config() -> None:
    """A fine-tune with the same model config gets a different key through the query bias."""
    weights = _tiny_weights()
    key = scale_key(CFG, weights)
    assert re.fullmatch(rf"esm-h{D}-l{LAYERS}-i{FFN}-a{HEADS}-[-+0-9._]+", key)
    bias = f"{PREFIX}encoder.layer.0.attention.self.query.bias"
    assert scale_key(CFG, {**weights, bias: weights[bias] + 1.0}) != key


def test_committed_scales_cover_every_projection() -> None:
    """Every committed checkpoint has an entry (a scale or null) for each projection."""
    path = Path(quantization_module.__file__).parent / ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8"))
    assert table, "no committed activation scales"
    for key, scales in table.items():
        match = re.fullmatch(r"esm-h(\d+)-l(\d+)-i(\d+)-a(\d+)-[-+0-9._]+", key)
        assert match, key
        assert set(scales) == set(projection_names({"num_layers": int(match[2])})), key
        assert all(value is None or value > 0 for value in scales.values()), key
        assert any(value is not None for value in scales.values()), key


def test_calibration_tool_maps_module_names_to_short_names() -> None:
    """The calibration tool maps module paths of the reference model to short names."""
    name = calibrate_fp8_scales.weight_name
    layer = "esm.encoder.layer.3"
    assert name(f"{layer}.attention.self.key") == "layer.3.key"
    assert name(f"{layer}.attention.output.dense") == "layer.3.attn_out"
    assert name(f"{layer}.intermediate.dense") == "layer.3.fc1"
    assert name(f"{layer}.output.dense") == "layer.3.fc2"
    assert name("lm_head.dense") is None
    assert name(f"{layer}.attention.self") is None
    assert name("esm.contact_head.regression") is None


def test_calibration_masks_fifteen_percent_of_residues() -> None:
    """Calibration inputs carry masks like the task input, never on the framing tokens."""
    ids = [0] + list(range(4, 24)) * 5 + [2]
    masked = calibrate_fp8_scales.mask_residues(ids, 32, seed=3)
    assert masked[0] == 0 and masked[-1] == 2
    assert sum(1 for value in masked if value == 32) == round(0.15 * (len(ids) - 2))
    assert masked == calibrate_fp8_scales.mask_residues(ids, 32, seed=3)


def test_calibration_drops_outlier_scales() -> None:
    """A scale above 20x the median of its kind is stored as null; the others are kept."""
    scales = {f"layer.{i}.fc2": 1.0 for i in range(5)}
    scales["layer.2.fc2"] = 25.0
    scales["layer.0.fc1"] = 30.0
    stored = calibrate_fp8_scales.drop_outliers(scales)
    assert stored["layer.2.fc2"] is None
    assert stored["layer.0.fc2"] == 1.0 and stored["layer.0.fc1"] == 30.0


def test_calibration_tool_cli_parses_both_commands(monkeypatch, tmp_path: Path) -> None:
    """The calibration tool command line accepts fetch and calibrate."""
    seen = {}
    monkeypatch.setattr(
        calibrate_fp8_scales,
        "fetch_sequences",
        lambda out, n, skip: seen.update(fetch=(out, n, skip)),
    )
    monkeypatch.setattr(
        calibrate_fp8_scales,
        "calibrate",
        lambda d, s, length: seen.update(calibrate=(d, s, length)),
    )
    monkeypatch.setattr("sys.argv", ["x", "fetch", "--out", str(tmp_path / "s.txt")])
    calibrate_fp8_scales.main()
    monkeypatch.setattr(
        "sys.argv", ["x", "calibrate", "--model-dir", str(tmp_path), "--sequences", "s.txt"]
    )
    calibrate_fp8_scales.main()
    assert seen["fetch"] == (tmp_path / "s.txt", 512, 0)
    assert seen["calibrate"] == (tmp_path, Path("s.txt"), 1022)


def _request(tmp_path: Path, precision: str, quantization: str) -> BuildRequest:
    return BuildRequest(
        model_dir=tmp_path,
        output_path=tmp_path / "b.bundle",
        family="esm",
        task="masked_text_to_token_scores",
        precision=precision,
        max_sequence_length=SEQ,
        quantization=quantization,
    )


def test_fp8_build_requires_fp16(tmp_path: Path) -> None:
    """FP8 runs around an fp16 graph; fp32 with fp8 is refused before the checkpoint is read."""
    with pytest.raises(ValueError, match="fp16"):
        model.build(_request(tmp_path, "fp32", "fp8"), object())


def test_other_quantization_formats_are_refused_early(tmp_path: Path) -> None:
    """Only fp8 is implemented; the error comes before the checkpoint is read."""
    with pytest.raises(NotImplementedError, match="nvfp4"):
        model.build(_request(tmp_path, "fp16", "nvfp4"), object())


class _FakeBuilderConfig:
    """Stands in for IBuilderConfig; only the attributes the build options touch."""

    def __init__(self, all_build_routes: str) -> None:
        self.all_build_routes = all_build_routes
        self.build_route = ""
        self.profiling_verbosity = None


@pytest.mark.parametrize(
    ("routes", "expected"),
    [('{"peep:match_dual_gemm": {}}', "-peep:match_dual_gemm=off"), ("", "")],
)
def test_dual_gemm_route_is_set_only_when_listed(routes: str, expected: str) -> None:
    """The FP8 build route is applied only when the TensorRT build lists the knob."""
    ctx = EsmQuantContext(scales={"layer.0.fc1": 0.02}, graph=graph)
    config = _FakeBuilderConfig(routes)
    configure_quantized_build(config, ctx)
    assert config.build_route == expected
    assert config.profiling_verbosity is not None
    unquantized = _FakeBuilderConfig(routes)
    configure_quantized_build(unquantized, None)
    assert unquantized.profiling_verbosity is None


def _network():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("creating a TensorRT builder needs a GPU")
    builder = trt.Builder(trt.Logger(trt.Logger.ERROR))
    return builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))


def test_query_key_value_share_one_quantize() -> None:
    """q, k and v read the same rows with the same scale, so one Q/DQ pair feeds all three."""
    network = _network()
    inp = network.add_input("x", trt.float16, (SEQ, D))
    weights = _tiny_weights()
    ctx = EsmQuantContext(
        scales={f"layer.0.{s}": 0.02 for s in ("query", "key", "value")}, graph=graph
    )
    for stem in ("query", "key", "value"):
        module = f"{PREFIX}encoder.layer.0.{HF_SUFFIX[stem]}"
        ctx.linear(
            network,
            inp,
            f"layer.0.{stem}",
            weights[f"{module}.weight"],
            weights[f"{module}.bias"],
            np.float16,
        )
    layers = [network.get_layer(i) for i in range(network.num_layers)]
    assert sum(layer.type == trt.LayerType.QUANTIZE for layer in layers) == 1
    # One Dequantize for the shared activation, one per FP8 weight.
    assert sum(layer.type == trt.LayerType.DEQUANTIZE for layer in layers) == 4


def test_dequantize_and_matmul_run_in_fp16() -> None:
    """Both Dequantize layers and the matmul are fp16 (the FP8 work type); no cast is added."""
    network = _network()
    inp = network.add_input("x", trt.float16, (SEQ, D))
    ctx = EsmQuantContext(scales={"layer.0.fc1": 0.02}, graph=graph)
    weight = _tiny_weights()[f"{PREFIX}encoder.layer.0.intermediate.dense.weight"]
    out = ctx.linear(network, inp, "layer.0.fc1", weight, None, np.float16)
    layers = [network.get_layer(i) for i in range(network.num_layers)]
    dequant = [layer for layer in layers if layer.type == trt.LayerType.DEQUANTIZE]
    assert len(dequant) == 2
    assert all(layer.get_output(0).dtype == trt.float16 for layer in dequant)
    matmul = [layer for layer in layers if layer.type == trt.LayerType.MATRIX_MULTIPLY]
    assert len(matmul) == 1 and matmul[0].get_output(0).dtype == trt.float16
    assert not any(layer.type == trt.LayerType.CAST for layer in layers)
    assert out.dtype == trt.float16


def test_unquantized_layers_have_no_quantize() -> None:
    """Without a quant context the encoder graph has no Q/DQ layer at all."""
    network = _network()
    hidden = network.add_input("h", trt.float32, (SEQ, D))
    mask = network.add_input("m", trt.float32, (1, SEQ, SEQ))
    cos, sin = graph.rotary_tables(SEQ, D // HEADS)
    from .. import encoder_builder

    encoder_builder.build_layers(
        network,
        hidden,
        _tiny_weights(),
        CFG,
        mask=mask,
        cos=graph.add_constant(network, cos.shape, cos),
        sin=graph.add_constant(network, sin.shape, sin),
    )
    kinds = {network.get_layer(i).type for i in range(network.num_layers)}
    assert trt.LayerType.QUANTIZE not in kinds and trt.LayerType.DEQUANTIZE not in kinds


def test_quantized_engine_builds(tiny) -> None:
    """The engine builds with FP8 tensors and keeps its IO.

    The plan size is not compared: at this tiny model config the kernels, not
    the weights, set the plan size, and on B200 the FP8 plan is the larger one.
    """
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("engine build needs a GPU")
    weights, _ = tiny
    ctx = calibrate_esm(weights, CFG, graph, "fp8")
    fp8 = model.build_engine(weights, CFG, seq_len=SEQ, precision="fp16", quant_ctx=ctx)
    engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(fp8)
    names = {engine.get_tensor_name(i) for i in range(engine.num_io_tensors)}
    assert names == {"input_ids", "attention_mask", "logits"}
    assert engine.get_tensor_dtype("logits") == trt.float32
    info = engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON)
    assert "FP8" in info
