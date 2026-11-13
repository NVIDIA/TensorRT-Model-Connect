# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T5 FP8 self-quantization: weight quantizer, scale lookup, context build, engine build."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import ml_dtypes
import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")
safetensors_numpy = pytest.importorskip("safetensors.numpy")

from .. import calibrate_fp8_scales, graph_ops, quantization  # noqa: E402
from ..config import ModelConfig  # noqa: E402
from ..model import _T5Model, build  # noqa: E402
from ..quantization import (  # noqa: E402
    DECODER_STEMS,
    ENCODER_STEMS,
    T5QuantContext,
    calibrate_t5_fp8,
    configure_fp8_build,
    load_activation_scales,
    projection_names,
    quantize_fp8_weight,
    reject_quantized_checkpoint,
    scale_key,
    weights_fingerprint,
)

# Wide enough that FP8 weight savings dominate the per-layer Q/DQ plan overhead.
D, FF, HEADS, DKV, VOCAB, LAYERS, SEQ = 256, 1024, 4, 64, 64, 2, 16


def _write_checkpoint(path: Path, seed: int = 0) -> ModelConfig:
    """Write a random tiny T5 checkpoint (HF tensor names) and return its config."""
    rng = np.random.default_rng(seed)
    inner = HEADS * DKV

    def w(*shape):
        return (rng.standard_normal(shape) * 0.05).astype(np.float32)

    t: dict[str, np.ndarray] = {
        "shared.weight": w(VOCAB, D),
        "encoder.final_layer_norm.weight": np.ones(D, np.float32),
        "decoder.final_layer_norm.weight": np.ones(D, np.float32),
        "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight": w(32, HEADS),
        "decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight": w(32, HEADS),
    }

    def attention(prefix: str) -> None:
        for proj in ("q", "k", "v"):
            t[f"{prefix}.{proj}.weight"] = w(inner, D)
        t[f"{prefix}.o.weight"] = w(D, inner)

    def mlp(prefix: str) -> None:
        t[f"{prefix}.DenseReluDense.wi.weight"] = w(FF, D)
        t[f"{prefix}.DenseReluDense.wo.weight"] = w(D, FF)

    for i in range(LAYERS):
        enc = f"encoder.block.{i}.layer"
        attention(f"{enc}.0.SelfAttention")
        mlp(f"{enc}.1")
        dec = f"decoder.block.{i}.layer"
        attention(f"{dec}.0.SelfAttention")
        attention(f"{dec}.1.EncDecAttention")
        mlp(f"{dec}.2")
        for norm in (f"{enc}.0", f"{enc}.1", f"{dec}.0", f"{dec}.1", f"{dec}.2"):
            t[f"{norm}.layer_norm.weight"] = np.ones(D, np.float32)
    safetensors_numpy.save_file(t, str(path / "model.safetensors"))
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "t5",
                "architectures": ["T5ForConditionalGeneration"],
                "d_model": D,
                "d_ff": FF,
                "d_kv": DKV,
                "num_heads": HEADS,
                "num_layers": LAYERS,
                "num_decoder_layers": LAYERS,
                "vocab_size": VOCAB,
                "n_positions": 512,
                "relative_attention_num_buckets": 32,
                "eos_token_id": 1,
                "pad_token_id": 0,
                "decoder_start_token_id": 0,
            }
        ),
        encoding="utf-8",
    )
    return ModelConfig.from_dir(path)


def _activation_scales(config, skip: tuple[str, ...] = (), null: tuple[str, ...] = ()) -> dict:
    """Return one activation scale per quantized projection of the tiny model."""
    return {
        name: (None if name in null else 0.02)
        for name in projection_names(config)
        if name not in skip
    }


@pytest.fixture
def tiny(tmp_path: Path):
    """Tiny checkpoint, its config and its loaded weights."""
    config = _write_checkpoint(tmp_path)
    config.raw["_model_dir"] = str(tmp_path)
    return tmp_path, config, _T5Model().load_weights(str(tmp_path), config)


def _patch_scales(monkeypatch, scales) -> None:
    monkeypatch.setattr(quantization, "load_activation_scales", lambda config, weights: scales)


def test_quantize_fp8_weight_scale_and_bytes() -> None:
    """The weight scale is amax / 448 and the bytes decode back to the weight."""
    weight = np.random.RandomState(0).randn(64, 128).astype(np.float32)
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


def test_scale_key_identifies_model_config_and_weights(tmp_path: Path, tiny) -> None:
    """The scale key names the model config and tells checkpoints of one config apart."""
    _, config, weights = tiny
    assert scale_key(config, weights) == (
        f"t5-h{D}-e{LAYERS}-d{LAYERS}-f{FF}-a{HEADS}x{DKV}-{weights_fingerprint(weights)}"
    )
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    other_config = _write_checkpoint(other_dir, seed=1)
    other = _T5Model().load_weights(str(other_dir), other_config)
    assert weights_fingerprint(other) != weights_fingerprint(weights)


def test_unknown_model_config_is_refused(tiny) -> None:
    """A checkpoint without committed scales fails and names the calibration tool."""
    _, config, weights = tiny
    with pytest.raises(NotImplementedError, match="calibrate_fp8_scales"):
        load_activation_scales(config, weights)
    with pytest.raises(NotImplementedError, match="no calibrated FP8 activation scales"):
        calibrate_t5_fp8(config, weights, graph_ops)


def test_committed_scales_cover_every_quantized_projection() -> None:
    """Every committed checkpoint has a positive scale, or null, for every quantized projection."""
    path = Path(quantization.__file__).parent / quantization.ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8"))
    assert table
    for key, scales in table.items():
        match = re.fullmatch(r"t5-h(\d+)-e(\d+)-d(\d+)-f(\d+)-a(\d+)x(\d+)-[0-9a-f]{12}", key)
        assert match, key
        hidden, enc, dec, ffn, heads, dkv = (int(g) for g in match.groups())
        config = ModelConfig(
            model_type="t5",
            hidden_size=hidden,
            raw={
                "d_model": hidden,
                "num_layers": enc,
                "num_decoder_layers": dec,
                "d_ff": ffn,
                "num_heads": heads,
                "d_kv": dkv,
            },
        )
        assert set(scales) == set(projection_names(config)), key
        assert all(value is None or value > 0 for value in scales.values()), key
        # q, k and v of one attention read one input; so do cross k and v (the encoder output).
        for layer in range(enc):
            qkv = {scales[f"enc_layer.{layer}.{stem}"] for stem in ("w_q", "w_k", "w_v")}
            assert len(qkv) == 1, (key, layer)
        for layer in range(dec):
            qkv = {scales[f"layer.{layer}.{stem}"] for stem in ("w_q", "w_k", "w_v")}
            assert len(qkv) == 1, (key, layer)
            cross = {scales[f"layer.{layer}.{stem}"] for stem in ("cross_w_k", "cross_w_v")}
            assert len(cross) == 1, (key, layer)


def test_attention_output_projections_are_not_quantized() -> None:
    """Attention output projections stay 16 bit (TensorRT 11.4 cannot compile FP8 after them)."""
    assert "w_o" not in ENCODER_STEMS and "w_o" not in DECODER_STEMS
    assert "cross_w_o" not in DECODER_STEMS


def test_context_quantizes_the_loaded_weights(monkeypatch, tiny) -> None:
    """The context quantizes the transposed loaded weights and keeps the committed scales."""
    _, config, weights = tiny
    _patch_scales(monkeypatch, _activation_scales(config))
    ctx = calibrate_t5_fp8(config, weights, graph_ops)

    assert len(ctx.scales) == LAYERS * (len(ENCODER_STEMS) + len(DECODER_STEMS))
    entry = ctx.scales["layer.1.w_fc2"]
    assert (entry.weight.out_features, entry.weight.in_features) == (D, FF)
    assert entry.input_scale == pytest.approx(0.02)
    packed, scale = quantize_fp8_weight(np.asarray(weights["layer.1.w_fc2"], np.float32).T)
    assert np.array_equal(entry.weight.packed, packed)
    assert entry.weight.weight_scale == pytest.approx(scale)
    assert ctx.should_quantize("enc_layer.0.w_q") and ctx.should_quantize("layer.0.cross_w_k")
    assert not ctx.should_quantize("layer.0.w_o") and not ctx.should_quantize("enc_layer.0.w_o")
    assert not ctx.should_quantize("layer.0.cross_w_o")


def test_projection_without_a_scale_is_refused(monkeypatch, tiny) -> None:
    """A projection without a scale fails the build and names the projection and the tool."""
    _, config, weights = tiny
    scales = _activation_scales(config)
    del scales["layer.0.cross_w_v"]
    _patch_scales(monkeypatch, scales)
    with pytest.raises(ValueError, match=r"layer\.0\.cross_w_v.*calibrate_fp8_scales\.py"):
        calibrate_t5_fp8(config, weights, graph_ops)


def test_null_scale_leaves_the_projection_unquantized(monkeypatch, tiny) -> None:
    """A null entry is an intentional exclusion: that projection stays fp16, the rest is FP8."""
    _, config, weights = tiny
    _patch_scales(monkeypatch, _activation_scales(config, null=("enc_layer.1.w_fc2",)))
    ctx = calibrate_t5_fp8(config, weights, graph_ops)
    assert not ctx.should_quantize("enc_layer.1.w_fc2")
    assert ctx.should_quantize("enc_layer.0.w_fc2") and ctx.should_quantize("layer.1.w_fc2")


def test_scale_for_a_projection_that_is_not_quantized_is_refused(monkeypatch, tiny) -> None:
    """A stored scale for w_o (not a quantized projection) means a stale or wrong scale file."""
    _, config, weights = tiny
    scales = _activation_scales(config)
    scales["enc_layer.0.w_o"] = 0.02
    _patch_scales(monkeypatch, scales)
    with pytest.raises(ValueError, match=r"enc_layer\.0\.w_o.*calibrate_fp8_scales\.py"):
        calibrate_t5_fp8(config, weights, graph_ops)


def test_already_quantized_checkpoint_is_refused(tmp_path: Path, tiny) -> None:
    """A checkpoint with a quantization config or FP8 scale tensors is refused."""
    path, config, weights = tiny
    config.raw["quantization_config"] = {"quant_method": "fp8"}
    with pytest.raises(ValueError, match="already quantized"):
        reject_quantized_checkpoint(path, config)
    del config.raw["quantization_config"]
    reject_quantized_checkpoint(path, config)

    tensors = safetensors_numpy.load_file(str(path / "model.safetensors"))
    tensors["encoder.block.0.layer.1.DenseReluDense.wi.weight_scale"] = np.ones(1, np.float32)
    safetensors_numpy.save_file(tensors, str(path / "model.safetensors"))
    with pytest.raises(ValueError, match="already quantized"):
        reject_quantized_checkpoint(path, config)

    weights["enc_layer.0.w_q.weight_scale"] = np.ones(1, np.float32)
    with pytest.raises(ValueError, match="already quantized"):
        calibrate_t5_fp8(config, weights, graph_ops)


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
    config = _FakeBuilderConfig(routes)
    configure_fp8_build(config, T5QuantContext(scales={}, graph_ops=graph_ops))
    assert config.build_route == expected
    assert config.profiling_verbosity is not None
    unquantized = _FakeBuilderConfig(routes)
    configure_fp8_build(unquantized, None)
    assert unquantized.build_route == "" and unquantized.profiling_verbosity is None


def _io(plan: bytes) -> dict[str, tuple]:
    """Return the I/O tensor names and shapes of a plan."""
    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(plan)
    assert engine is not None
    return {
        engine.get_tensor_name(i): tuple(engine.get_tensor_shape(engine.get_tensor_name(i)))
        for i in range(engine.num_io_tensors)
    }


def _fp8_layers(plan: bytes) -> int:
    """Count the engine layers that run an FP8 kernel."""
    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(plan)
    info = engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON)
    layers = json.loads(info)["Layers"]
    return sum(1 for layer in layers if "e4m3" in json.dumps(layer).lower())


@pytest.mark.gpu
@pytest.mark.trt
@pytest.mark.parametrize("precision", ["fp16", "fp32"])
def test_engines_build_with_fp8_weights(monkeypatch, tiny, precision) -> None:
    """The quantized encoder and decoder build, keep the I/O, are smaller and run FP8 kernels."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("engine build needs a GPU")
    path, config, weights = tiny
    _patch_scales(monkeypatch, _activation_scales(config))
    ctx = calibrate_t5_fp8(config, weights, graph_ops)
    model = _T5Model()

    plain_decoder = model.build_engine(config, weights, SEQ, precision=precision)
    plain_encoder = model.build_vision_engine(str(path), config, weights, precision=precision)
    fp8_decoder = model.build_engine(config, weights, SEQ, precision=precision, quant_ctx=ctx)
    fp8_encoder = model.build_vision_engine(
        str(path), config, weights, precision=precision, quant_ctx=ctx
    )
    assert _io(fp8_decoder) == _io(plain_decoder)
    assert _io(fp8_encoder) == _io(plain_encoder)
    assert len(fp8_decoder) < len(plain_decoder)
    assert len(fp8_encoder) < len(plain_encoder)
    assert _fp8_layers(fp8_decoder) > 0 and _fp8_layers(fp8_encoder) > 0


def _build_request(**updates):
    """Return a minimal build request for the build-time refusal tests."""
    request = SimpleNamespace(
        dynamic_kv_cache=False,
        image_height=None,
        image_width=None,
        video_num_frames=None,
        max_batch_size=1,
        context_parallel_size=1,
        task="text_generation",
        model_dir=Path("."),
        precision="fp16",
        max_sequence_length=16,
        tensor_parallel_size=1,
        quantization="fp8",
        fp32_layers=None,
        mtp_seq_len=None,
        verbose=False,
        backend="trt",
    )
    for key, value in updates.items():
        setattr(request, key, value)
    return request


def test_build_requires_calibrated_scales_for_fp8(tiny) -> None:
    """quantization=fp8 on a checkpoint without committed scales fails before any engine build."""
    path, _, _ = tiny
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    with pytest.raises(NotImplementedError, match="no calibrated FP8 activation scales"):
        build(_build_request(model_dir=path), writer=None)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"quantization": "nvfp4"}, "quantization='nvfp4'"),
        ({"tensor_parallel_size": 2, "precision": "fp32"}, "does not support tensor parallelism"),
        ({"tensor_parallel_size": 2}, "tensor-parallel builds require fp32"),
    ],
)
def test_build_refuses_unsupported_fp8_requests(tiny, updates, message) -> None:
    """Other quantization modes and tensor parallelism are refused loudly."""
    path, _, _ = tiny
    with pytest.raises(NotImplementedError, match=message):
        build(_build_request(model_dir=path, **updates), writer=None)


def test_tensor_parallel_builder_refuses_a_quant_context(tiny) -> None:
    """The tensor-parallel decoder path does not quantize and says so."""
    from ..parallel import ParallelConfig

    path, config, weights = tiny
    ctx = T5QuantContext(scales={}, graph_ops=graph_ops)
    with pytest.raises(NotImplementedError, match="tensor parallelism"):
        _T5Model().build_engine(
            config,
            weights,
            SEQ,
            parallel_config=ParallelConfig(tp_size=2).for_rank(0),
            quant_ctx=ctx,
        )


def test_calibration_tool_maps_module_names_to_weight_names() -> None:
    """The calibration tool maps HF module paths to internal weight names."""
    name = calibrate_fp8_scales.weight_name
    assert name("encoder.block.3.layer.0.SelfAttention.q") == "enc_layer.3.w_q"
    assert name("encoder.block.3.layer.0.SelfAttention.o") == "enc_layer.3.w_o"
    assert name("encoder.block.0.layer.1.DenseReluDense.wi") == "enc_layer.0.w_fc1"
    assert name("decoder.block.2.layer.0.SelfAttention.k") == "layer.2.w_k"
    assert name("decoder.block.2.layer.1.EncDecAttention.v") == "layer.2.cross_w_v"
    assert name("decoder.block.2.layer.1.EncDecAttention.o") == "layer.2.cross_w_o"
    assert name("decoder.block.2.layer.2.DenseReluDense.wo") == "layer.2.w_fc2"
    assert name("decoder.block.2.layer.2.DenseReluDense.wo.base") is None
    assert name("encoder.block.0.layer.0.SelfAttention.relative_attention_bias") is None
    assert name("lm_head") is None


def test_outlier_projection_is_split_off_per_side() -> None:
    """A scale above 20x the median of the same projection on the same side is left out."""
    scales = {f"enc_layer.{i}.w_fc2": 0.01 for i in range(5)}
    scales["enc_layer.5.w_fc2"] = 0.5
    # A decoder projection of the same stem is judged against the decoder median only.
    scales.update({f"layer.{i}.w_fc2": 0.4 for i in range(6)})
    kept, excluded = calibrate_fp8_scales.split_outlier_scales(scales)
    assert excluded == ["enc_layer.5.w_fc2"]
    assert len(kept) == len(scales) - 1


def test_calibration_tool_cli_parses_both_commands(monkeypatch, tmp_path: Path) -> None:
    """The calibration tool command line accepts fetch and calibrate."""
    seen = {}
    monkeypatch.setattr(
        calibrate_fp8_scales, "fetch_prompts", lambda out, n: seen.update(fetch=(out, n))
    )
    monkeypatch.setattr(
        calibrate_fp8_scales,
        "calibrate",
        lambda d, p, n, g: seen.update(calibrate=(d, p, n, g)),
    )
    monkeypatch.setattr("sys.argv", ["x", "fetch", "--out", str(tmp_path / "p.jsonl")])
    calibrate_fp8_scales.main()
    monkeypatch.setattr(
        "sys.argv", ["x", "calibrate", "--model-dir", str(tmp_path), "--prompts", "p.jsonl"]
    )
    calibrate_fp8_scales.main()
    assert seen["fetch"] == (tmp_path / "p.jsonl", 256)
    assert seen["calibrate"] == (tmp_path, Path("p.jsonl"), 256, 64)
