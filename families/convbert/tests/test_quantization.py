# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ConvBERT FP8 self-quantization: weight quantizer, scale lookup, context, calibration tool, engine."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import ml_dtypes
import numpy as np
import pytest

torch = pytest.importorskip("torch")
trt = pytest.importorskip("tensorrt")

from .. import builder, calibrate_fp8_scales, model as model_module, quantization  # noqa: E402
from ..config import ModelConfig  # noqa: E402
from ..quantization import (  # noqa: E402
    PROJECTIONS,
    calibrate_convbert,
    load_activation_scales,
    module_names,
    quantize_fp8_weight,
    scale_key,
)

# Wide enough that FP8 weight savings dominate the per-layer Q/DQ plan overhead.
# 8 heads -> 4 attention heads of size 32 (head_ratio 2), so q/k/v/conv_out have 128 outputs.
HIDDEN, INTER, LAYERS, HEADS, VOCAB, SEQ, KERNEL = 256, 512, 3, 8, 64, 32, 9
ALL_HEAD = 128


def _hf_config():
    from transformers import ConvBertConfig

    return ConvBertConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        embedding_size=HIDDEN,
        intermediate_size=INTER,
        num_hidden_layers=LAYERS,
        num_attention_heads=HEADS,
        max_position_embeddings=SEQ,
        conv_kernel_size=KERNEL,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )


def _write_tiny_convbert(path: Path, seed: int = 0) -> None:
    """Write a small random ConvBERT checkpoint (``pytorch_model.bin``) and its config."""
    from transformers import ConvBertModel

    torch.manual_seed(seed)
    hf_config = _hf_config()
    torch.save(ConvBertModel(hf_config).eval().state_dict(), path / "pytorch_model.bin")
    (path / "config.json").write_text(hf_config.to_json_string(), encoding="utf-8")


@pytest.fixture
def tiny(tmp_path: Path):
    """The tiny model's config and weights as the build loads them."""
    _write_tiny_convbert(tmp_path)
    config = ModelConfig.from_dir(tmp_path)
    weights = model_module._ConvBertModel().load_weights(str(tmp_path), config)
    return config, weights, tmp_path


def _all_scales(config, value: float = 0.02, null: tuple[str, ...] = ()):
    return {
        name: (None if name in null else value) for name in module_names(config.num_hidden_layers)
    }


def _patch_scales(monkeypatch, scales) -> None:
    monkeypatch.setattr(quantization, "load_activation_scales", lambda *_: scales)


def test_quantize_fp8_weight_scale_and_bytes() -> None:
    """The weight scale is amax / 448 and the bytes decode back to the weight."""
    weight = np.random.default_rng(0).standard_normal((64, 128)).astype(np.float32)
    packed, scale = quantize_fp8_weight(weight)
    assert scale == pytest.approx(float(np.abs(weight).max()) / 448.0)
    assert packed.dtype == np.uint8 and packed.shape == weight.shape
    restored = packed.view(ml_dtypes.float8_e4m3fn).astype(np.float32) * scale
    assert np.abs(restored - weight).max() <= np.abs(weight).max() * 0.07  # e4m3 half-step


def test_quantize_fp8_weight_never_overflows_to_nan() -> None:
    """Values at the scale boundary clip to the largest finite fp8 value."""
    packed, _ = quantize_fp8_weight(np.full((4, 4), 3.0, dtype=np.float32))
    assert not np.isnan(packed.view(ml_dtypes.float8_e4m3fn).astype(np.float32)).any()


def test_quantize_fp8_weight_rejects_zero_tensor() -> None:
    """An all-zero weight has no usable scale and is refused."""
    with pytest.raises(ValueError, match="amax"):
        quantize_fp8_weight(np.zeros((2, 2), dtype=np.float32))


def test_module_names_cover_the_linear_projections_only() -> None:
    """Six projections per layer; the attention output and the kernel generator stay 16-bit."""
    names = module_names(4)
    assert len(names) == 4 * len(PROJECTIONS) == 24
    assert {
        "layer.3.attention.self.query",
        "layer.3.attention.self.key",
        "layer.3.attention.self.value",
        "layer.3.attention.self.conv_out_layer",
        "layer.3.intermediate.dense",
        "layer.3.output.dense",
    } == {name for name in names if name.startswith("layer.3.")}
    assert not any("attention.output" in name for name in names)
    assert not any("conv_kernel_layer" in name or "key_conv_attn" in name for name in names)


def test_scale_key_names_the_model_config_and_weights(tiny, tmp_path_factory) -> None:
    """The key holds the config values and weight sums, so other weights get another key."""
    config, weights, _ = tiny
    key = scale_key(config, weights)
    assert key.startswith(f"convbert-h{HIDDEN}-l{LAYERS}-i{INTER}-n{HEADS}-k{KERNEL}-v{VOCAB}-w")
    other = tmp_path_factory.mktemp("other")
    _write_tiny_convbert(other, seed=1)
    other_weights = model_module._ConvBertModel().load_weights(str(other), config)
    assert scale_key(config, other_weights) != key


def test_unknown_model_config_is_refused(tiny) -> None:
    """A model config without committed scales fails and names the calibration tool."""
    config, weights, _ = tiny
    with pytest.raises(NotImplementedError, match=r"calibrate_fp8_scales\.py"):
        load_activation_scales(config, weights)


def test_committed_scales_cover_every_projection() -> None:
    """Every committed entry has a positive scale, or null, for every projection of its model."""
    path = Path(quantization.__file__).parent / quantization.ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8"))
    assert table
    for key, scales in table.items():
        layers = int(key.split("-l")[1].split("-")[0])
        assert set(scales) == set(module_names(layers)), key
        assert all(value is None or value > 0 for value in scales.values()), key
        assert any(value is not None for value in scales.values()), key


def test_context_quantizes_the_loaded_weights(tiny, monkeypatch) -> None:
    """Each projection gets its own FP8 weight in [in, out] layout and weight scale."""
    config, weights, _ = tiny
    _patch_scales(monkeypatch, _all_scales(config))
    ctx = calibrate_convbert(config, weights, "fp8", precision="fp16")

    assert len(ctx.scales) == LAYERS * len(PROJECTIONS)
    for stem, logical in PROJECTIONS.items():
        entry = ctx.scales[f"layer.1.{stem}"]
        matrix = weights[f"layer.1.{logical}"]
        assert entry.input_scale == pytest.approx(0.02)
        assert (entry.weight.in_features, entry.weight.out_features) == matrix.shape
        assert entry.weight.weight_scale == pytest.approx(float(np.abs(matrix).max()) / 448.0)
        assert np.array_equal(entry.weight.packed, quantize_fp8_weight(matrix)[0])
    assert ctx.scales["layer.0.attention.self.conv_out_layer"].weight.out_features == ALL_HEAD
    assert ctx.scales["layer.0.intermediate.dense"].weight.out_features == INTER
    assert ctx.scales["layer.0.output.dense"].weight.in_features == INTER
    assert not ctx.should_quantize("layer.0.attention.output.dense")  # 16-bit, see quantization.py
    assert not ctx.should_quantize("layer.0.attention.self.conv_kernel_layer")


def test_null_entry_stays_unquantized(tiny, monkeypatch) -> None:
    """A null scale (massive activation) leaves that projection 16-bit on purpose."""
    config, weights, _ = tiny
    name = "layer.1.output.dense"
    _patch_scales(monkeypatch, _all_scales(config, null=(name,)))
    ctx = calibrate_convbert(config, weights, "fp8", precision="fp16")
    assert not ctx.should_quantize(name)
    assert ctx.should_quantize("layer.0.output.dense")
    assert len(ctx.scales) == LAYERS * len(PROJECTIONS) - 1


def test_projection_without_a_scale_is_refused(tiny, monkeypatch) -> None:
    """A projection missing from the file fails the build and names the calibration tool."""
    config, weights, _ = tiny
    scales = _all_scales(config)
    del scales["layer.0.attention.self.conv_out_layer"]
    _patch_scales(monkeypatch, scales)
    with pytest.raises(
        ValueError, match=r"layer\.0\.attention\.self\.conv_out_layer.*calibrate_fp8_scales\.py"
    ):
        calibrate_convbert(config, weights, "fp8", precision="fp16")


def test_scale_for_an_unknown_projection_is_refused(tiny, monkeypatch) -> None:
    """A table with more layers than the model is a mismatch, not a silent partial build."""
    config, weights, _ = tiny
    scales = _all_scales(config)
    scales["layer.9.output.dense"] = 0.02
    _patch_scales(monkeypatch, scales)
    with pytest.raises(ValueError, match=r"layer\.9\.output\.dense"):
        calibrate_convbert(config, weights, "fp8", precision="fp16")


@pytest.mark.parametrize("value", [0.0, -0.1, float("nan")])
def test_non_positive_scale_is_refused(tiny, monkeypatch, value) -> None:
    """A stored scale must be finite and positive."""
    config, weights, _ = tiny
    scales = _all_scales(config)
    scales["layer.2.attention.self.key"] = value
    _patch_scales(monkeypatch, scales)
    with pytest.raises(ValueError, match="finite and positive"):
        calibrate_convbert(config, weights, "fp8", precision="fp16")


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_fp8_needs_fp16_precision(tiny, monkeypatch, precision) -> None:
    """FP8 is built on the fp16 engine only."""
    config, weights, _ = tiny
    _patch_scales(monkeypatch, _all_scales(config))
    with pytest.raises(ValueError, match="precision fp16"):
        calibrate_convbert(config, weights, "fp8", precision=precision)


def test_unsupported_quantization_is_refused(tiny) -> None:
    """Only fp8 is implemented."""
    config, weights, _ = tiny
    with pytest.raises(NotImplementedError, match="fp8"):
        calibrate_convbert(config, weights, "nvfp4", precision="fp16")


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
        task="encoding",
        model_dir=str(model_dir),
        precision="fp16",
        max_sequence_length=SEQ,
        quantization="fp8",
        fp32_layers=None,
        mtp_seq_len=None,
        tensor_parallel_size=1,
        verbose=False,
        backend="tensorrt",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.fixture
def no_tokenizer(monkeypatch):
    """The tiny checkpoint has no tokenizer files."""
    monkeypatch.setattr(model_module, "_tokenizer_runtime_contract", lambda _: {})


@pytest.mark.parametrize("marker", ["scale-tensor", "quantization-config", "hf-quant-config"])
def test_already_quantized_checkpoint_is_refused(tiny, no_tokenizer, marker) -> None:
    """A checkpoint that carries FP8 scales or a quantization config is refused."""
    *_, model_dir = tiny
    if marker == "scale-tensor":
        state = torch.load(model_dir / "pytorch_model.bin", weights_only=True)
        state["encoder.layer.0.output.dense.weight_scale"] = torch.ones(1)
        torch.save(state, model_dir / "pytorch_model.bin")
    elif marker == "quantization-config":
        raw = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        raw["quantization_config"] = {"quant_method": "modelopt"}
        (model_dir / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    else:
        (model_dir / "hf_quant_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="already quantized"):
        model_module.build(_request(model_dir), _Writer())


def test_build_refuses_an_unknown_model_config(tiny, no_tokenizer) -> None:
    """The committed table has no entry for the tiny model, so the FP8 build raises."""
    *_, model_dir = tiny
    with pytest.raises(NotImplementedError, match=r"calibrate_fp8_scales\.py"):
        model_module.build(_request(model_dir), _Writer())


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_build_refuses_fp8_without_fp16(tiny, no_tokenizer, monkeypatch, precision) -> None:
    """FP8 on the fp32 or bf16 engine is rejected before the checkpoint is read."""
    *_, model_dir = tiny
    monkeypatch.setattr(
        model_module._ConvBertModel,
        "load_weights",
        lambda *_: pytest.fail("FP8 without fp16 must fail before loading weights"),
    )
    with pytest.raises(ValueError, match="precision fp16"):
        model_module.build(_request(model_dir, precision=precision), _Writer())


def test_build_refuses_fp8_with_tensor_parallel(tiny, no_tokenizer) -> None:
    """Tensor-parallel plans stay unquantized."""
    *_, model_dir = tiny
    with pytest.raises(ValueError, match="tensor-parallel"):
        model_module.build(_request(model_dir, tensor_parallel_size=2), _Writer())


def test_build_refuses_other_quantization(tiny, no_tokenizer) -> None:
    """A quantization other than fp8 or none is rejected."""
    *_, model_dir = tiny
    with pytest.raises(NotImplementedError, match="fp8 or none"):
        model_module.build(_request(model_dir, quantization="nvfp4"), _Writer())


class _FakeBuilderConfig:
    """Stands in for IBuilderConfig; only the attribute the build option touches."""

    profiling_verbosity = None


def test_fp8_build_keeps_layer_details() -> None:
    """A quantized build asks for detailed profiling."""
    builder_config = _FakeBuilderConfig()
    quantization.ConvBertQuantContext(scales={}).configure_build(builder_config)
    assert builder_config.profiling_verbosity == trt.ProfilingVerbosity.DETAILED


# --- calibration tool ---------------------------------------------------------------------


def test_projection_name_matches_only_quantized_projections() -> None:
    """Only the six quantized projections are calibrated, under any model root."""
    name = calibrate_fp8_scales.projection_name
    for stem in PROJECTIONS:
        assert name(f"encoder.layer.11.{stem}") == f"layer.11.{stem}"
        assert name(f"convbert.encoder.layer.3.{stem}") == f"layer.3.{stem}"
        assert name(f"model.convbert.encoder.layer.0.{stem}") == f"layer.0.{stem}"
    assert name("encoder.layer.2.attention.output.dense") is None
    assert name("encoder.layer.2.attention.self.conv_kernel_layer") is None
    assert name("encoder.layer.2.attention.self.key_conv_attn_layer.pointwise") is None
    assert name("generator_predictions.dense") is None
    assert name("encoder.layer.0.output.LayerNorm") is None
    assert name("encoder.layer.0.output.dense.input_quantizer") is None


def test_outlier_projections_become_null() -> None:
    """A scale above 20x the median of its kind is stored as null, the others are kept."""
    scales = {f"layer.{layer}.output.dense": 0.01 for layer in range(6)}
    scales.update({f"layer.{layer}.intermediate.dense": 0.02 for layer in range(6)})
    scales["layer.3.output.dense"] = 0.5
    table, excluded = calibrate_fp8_scales.split_outlier_scales(scales)
    assert excluded == ["layer.3.output.dense"]
    assert table["layer.3.output.dense"] is None
    assert table["layer.2.output.dense"] == 0.01
    assert table["layer.3.intermediate.dense"] == 0.02


def test_passages_are_disjoint_runs_of_the_requested_lengths() -> None:
    """Passages take consecutive tokens across lines and never reuse a token."""
    lines = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    assert calibrate_fp8_scales.passages(lines, [2, 4, 2]) == [[1, 2], [3, 4, 5, 6], [7, 8]]
    assert calibrate_fp8_scales.passages(lines, [5, 5]) == [[1, 2, 3, 4, 5]]


def test_calibration_lengths_mix_short_and_long() -> None:
    """Short passages have 16 to the longest body; the long ones come last."""
    lengths = calibrate_fp8_scales.calibration_lengths(100, 3, 510)
    assert len(lengths) == 103
    assert all(16 <= length <= 510 for length in lengths[:100])
    assert lengths[100:] == [510] * 3
    assert lengths == calibrate_fp8_scales.calibration_lengths(100, 3, 510)


def test_calibration_cli(monkeypatch) -> None:
    """The command line passes the model directory and passage counts."""
    seen = {}
    monkeypatch.setattr(calibrate_fp8_scales, "calibrate", lambda *args: seen.update(c=args))
    monkeypatch.setattr(
        "sys.argv", ["x", "--model-dir", "m", "--short-passages", "5", "--long-passages", "1"]
    )
    calibrate_fp8_scales.main()
    assert seen["c"] == (Path("m"), 5, 1)


# --- engine -------------------------------------------------------------------------------


def _layer_network(weights, ctx):
    network = trt.Builder(trt.Logger(trt.Logger.ERROR)).create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    hidden = network.add_input("x", trt.float16, (SEQ, HIDDEN))
    attn_mask = network.add_input("m", trt.float16, (1, 1, SEQ))
    sequence_mask = network.add_input("s", trt.float16, (SEQ, 1))
    out = builder._add_convbert_layer(
        network=network,
        hidden=hidden,
        weights=weights,
        prefix="layer.0",
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        new_num_heads=HEADS // 2,
        head_size=ALL_HEAD // (HEADS // 2),
        all_head_size=ALL_HEAD,
        conv_kernel_size=KERNEL,
        seq_length=SEQ,
        attn_mask=attn_mask,
        sequence_mask=sequence_mask,
        hidden_act="gelu",
        eps=1e-12,
        dtype=np.float16,
        quant_ctx=ctx,
    )
    return network, out


def test_layer_shares_one_quantize_for_the_hidden_state_projections(tiny, monkeypatch) -> None:
    """One layer has three Quantize layers (q/k/v/conv_out shared, fc1, fc2), all fp16 out."""
    config, weights, _ = tiny
    scales = _all_scales(config)
    scales["layer.0.attention.self.conv_out_layer"] = 0.03  # the shared input takes the largest
    _patch_scales(monkeypatch, scales)
    ctx = calibrate_convbert(config, weights, "fp8", precision="fp16")
    network, out = _layer_network(weights, ctx)
    layers = [network.get_layer(i) for i in range(network.num_layers)]
    quantize = [layer for layer in layers if layer.type == trt.LayerType.QUANTIZE]
    dequantize = [layer for layer in layers if layer.type == trt.LayerType.DEQUANTIZE]
    assert len(quantize) == 3
    assert len(dequantize) == 3 + len(PROJECTIONS)
    assert all(layer.get_output(0).dtype == trt.float16 for layer in dequantize)
    assert out.dtype == trt.float16


def test_layer_without_context_has_no_quantize(tiny) -> None:
    """The unquantized layer graph has no Q/DQ layer."""
    _, weights, _ = tiny
    network, _ = _layer_network(weights, None)
    kinds = {network.get_layer(i).type for i in range(network.num_layers)}
    assert trt.LayerType.QUANTIZE not in kinds and trt.LayerType.DEQUANTIZE not in kinds


@pytest.mark.gpu
@pytest.mark.trt
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_engine_builds_with_fp8_weights(tiny, no_tokenizer, monkeypatch) -> None:
    """The quantized engine builds from the real build path, is smaller, finite and close."""
    config, _, model_dir = tiny
    _patch_scales(monkeypatch, _all_scales(config, value=0.01))
    plain, quantized = _Writer(), _Writer()
    model_module.build(_request(model_dir, quantization=None), plain)
    model_module.build(_request(model_dir), quantized)

    plain_plan, fp8_plan = plain.files["engine.plan"], quantized.files["engine.plan"]
    assert len(fp8_plan) < 0.8 * len(plain_plan)
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    outputs = []
    for plan in (plain_plan, fp8_plan):
        engine = runtime.deserialize_cuda_engine(plan)
        assert engine is not None
        assert {engine.get_tensor_name(i) for i in range(engine.num_io_tensors)} == {
            "input_ids",
            "attention_mask",
            "hidden_states",
        }
        context = engine.create_execution_context()
        ids = torch.arange(1, SEQ + 1, dtype=torch.int32, device="cuda") % VOCAB
        mask = torch.ones(SEQ, dtype=torch.int32, device="cuda")
        out = torch.empty((SEQ, HIDDEN), dtype=torch.float32, device="cuda")
        for name, value in (("input_ids", ids), ("attention_mask", mask), ("hidden_states", out)):
            context.set_tensor_address(name, value.data_ptr())
        assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        outputs.append(out)
    assert torch.isfinite(outputs[1]).all()
    cosine = torch.nn.functional.cosine_similarity(outputs[0], outputs[1], dim=1)
    assert float(cosine.mean()) > 0.9
