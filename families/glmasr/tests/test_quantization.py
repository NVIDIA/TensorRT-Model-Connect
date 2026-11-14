# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GLM-ASR FP8 quantization: weight scaling, projection coverage, calibration tool, engine build."""

from __future__ import annotations

import json
import re
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")
safetensors_numpy = pytest.importorskip("safetensors.numpy")

from families.glmasr import calibrate_fp8_scales, graph_ops, model  # noqa: E402
from families.glmasr import quantization as quantization_module  # noqa: E402
from families.glmasr.config import ModelConfig  # noqa: E402
from families.glmasr.quantization import (  # noqa: E402
    ACTIVATION_SCALES_FILENAME,
    GlmAsrQuantContext,
    _projections,
    calibrate_glmasr,
    configure_fp8_build,
    encoder_weight_name,
    quantize_fp8_weight,
    scale_key,
)

# Decoder: hidden D, GQA with 4 query and 2 key/value heads. Encoder: hidden ED with the
# projector input width fixed at ED * MERGE_FACTOR (= 4 * ED) by the checkpoint contract.
D, FFN, VOCAB, LAYERS, HEADS, KV_HEADS = 128, 256, 100, 2, 4, 2
ED, EFFN, EHEADS, ELAYERS, MEL, FRAMES = 128, 512, 4, 2, 16, 32


def _write_checkpoint(path: Path) -> ModelConfig:
    """Write a random tiny GLM-ASR checkpoint (HF tensor names) and return its config."""
    rng = np.random.default_rng(0)
    head_dim = D // HEADS

    def w(*shape):
        return rng.standard_normal(shape).astype(np.float32) * 0.05

    def norm(prefix: str, size: int, bias: bool) -> None:
        t[f"{prefix}.weight"] = np.ones(size, np.float32)
        if bias:
            t[f"{prefix}.bias"] = np.zeros(size, np.float32)

    t: dict[str, np.ndarray] = {
        "language_model.model.embed_tokens.weight": w(VOCAB, D),
        "language_model.lm_head.weight": w(VOCAB, D),
        "audio_tower.conv1.weight": w(ED, MEL, 3),
        "audio_tower.conv1.bias": w(ED),
        "audio_tower.conv2.weight": w(ED, ED, 3),
        "audio_tower.conv2.bias": w(ED),
    }
    norm("language_model.model.norm", D, bias=False)
    norm("audio_tower.norm", ED, bias=True)
    for i in range(LAYERS):
        layer = f"language_model.model.layers.{i}"
        norm(f"{layer}.input_layernorm", D, bias=False)
        norm(f"{layer}.post_attention_layernorm", D, bias=False)
        t[f"{layer}.self_attn.q_proj.weight"] = w(HEADS * head_dim, D)
        t[f"{layer}.self_attn.k_proj.weight"] = w(KV_HEADS * head_dim, D)
        t[f"{layer}.self_attn.v_proj.weight"] = w(KV_HEADS * head_dim, D)
        t[f"{layer}.self_attn.o_proj.weight"] = w(D, HEADS * head_dim)
        t[f"{layer}.mlp.gate_proj.weight"] = w(FFN, D)
        t[f"{layer}.mlp.up_proj.weight"] = w(FFN, D)
        t[f"{layer}.mlp.down_proj.weight"] = w(D, FFN)
    for i in range(ELAYERS):
        layer = f"audio_tower.layers.{i}"
        norm(f"{layer}.input_layernorm", ED, bias=True)
        norm(f"{layer}.post_attention_layernorm", ED, bias=True)
        for proj in ("q", "k", "v", "o"):
            t[f"{layer}.self_attn.{proj}_proj.weight"] = w(ED, ED)
            if proj != "k":
                t[f"{layer}.self_attn.{proj}_proj.bias"] = w(ED)
        t[f"{layer}.mlp.fc1.weight"], t[f"{layer}.mlp.fc1.bias"] = w(EFFN, ED), w(EFFN)
        t[f"{layer}.mlp.fc2.weight"], t[f"{layer}.mlp.fc2.bias"] = w(ED, EFFN), w(ED)
    t["multi_modal_projector.linear_1.weight"] = w(2 * D, EFFN)
    t["multi_modal_projector.linear_1.bias"] = w(2 * D)
    t["multi_modal_projector.linear_2.weight"] = w(D, 2 * D)
    t["multi_modal_projector.linear_2.bias"] = w(D)

    safetensors_numpy.save_file(t, str(path / "model.safetensors"))
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "glmasr",
                "architectures": ["GlmAsrForConditionalGeneration"],
                "audio_token_id": 7,
                "projector_hidden_act": "gelu",
                "audio_config": {
                    "model_type": "glmasr_encoder",
                    "hidden_size": ED,
                    "num_hidden_layers": ELAYERS,
                    "num_attention_heads": EHEADS,
                    "intermediate_size": EFFN,
                    "num_mel_bins": MEL,
                    "max_position_embeddings": FRAMES,
                    "partial_rotary_factor": 0.5,
                    "hidden_act": "gelu",
                },
                "text_config": {
                    "model_type": "llama",
                    "hidden_size": D,
                    "intermediate_size": FFN,
                    "num_hidden_layers": LAYERS,
                    "num_attention_heads": HEADS,
                    "num_key_value_heads": KV_HEADS,
                    "vocab_size": VOCAB,
                    "max_position_embeddings": 128,
                    "eos_token_id": 3,
                },
            }
        ),
        encoding="utf-8",
    )
    return ModelConfig.from_dir(path)


def _all_scales(config: ModelConfig, value: float = 0.02) -> dict[str, float]:
    return {name: value for _, name in _projections(config)}


@pytest.fixture
def tiny(tmp_path: Path, monkeypatch):
    """Tiny checkpoint plus an activation-scale table that covers every projection."""
    config = _write_checkpoint(tmp_path)
    monkeypatch.setattr(quantization_module, "load_activation_scales", lambda cfg: _all_scales(cfg))
    return tmp_path, config


def test_fp8_weight_scale_is_amax_over_448() -> None:
    """The scale is amax / 448 and the bytes decode to the original weight within e4m3 precision."""
    weight = np.random.default_rng(1).standard_normal((8, 32)).astype(np.float32)
    packed, scale = quantize_fp8_weight(weight)
    assert scale == pytest.approx(float(np.abs(weight).max()) / 448.0)
    assert packed.dtype == np.uint8 and packed.shape == weight.shape
    decoded = packed.view(ml_dtypes.float8_e4m3fn).astype(np.float32) * scale
    assert np.abs(decoded - weight).max() <= 0.07 * float(np.abs(weight).max())


def test_fp8_weight_rejects_all_zero() -> None:
    """A weight with no positive amax cannot be scaled."""
    with pytest.raises(ValueError, match="amax"):
        quantize_fp8_weight(np.zeros((4, 4), dtype=np.float32))


def test_every_projection_is_quantized(tiny) -> None:
    """Decoder and encoder projections get FP8 weights; the encoder attention output and the projector do not."""
    path, config = tiny
    ctx = calibrate_glmasr(path, graph_ops, "fp8", config)
    assert len(ctx.scales) == LAYERS * 7 + ELAYERS * 5
    assert ctx.should_quantize("layer.0.w_o")
    assert ctx.should_quantize("layer.1.w_down")
    assert ctx.should_quantize("audio.layer.0.w_q")
    # encoder self-attention output projections stay fp16 (TensorRT 11.4 cannot compile them in FP8)
    assert not ctx.should_quantize("audio.layer.0.w_o")
    assert ctx.should_quantize("audio.layer.1.w_fc2")
    # the audio-to-decoder projector stays unquantized
    assert not any(name.startswith("audio.proj") for name in ctx.scales)
    entry = ctx.scales["layer.0.w_k"]
    assert (entry.weight.out_features, entry.weight.in_features) == (KV_HEADS * (D // HEADS), D)
    assert entry.weight.packed.dtype == np.uint8


def test_projection_without_scale_stays_unquantized(tmp_path: Path, monkeypatch) -> None:
    """A projection with no activation scale is left in fp16."""
    config = _write_checkpoint(tmp_path)
    scales = _all_scales(config)
    del scales["layer.0.w_down"]
    monkeypatch.setattr(quantization_module, "load_activation_scales", lambda cfg: scales)
    ctx = calibrate_glmasr(tmp_path, graph_ops, "fp8", config)
    assert not ctx.should_quantize("layer.0.w_down")
    assert ctx.should_quantize("layer.0.w_up")


def test_already_quantized_checkpoint_is_rejected(tiny) -> None:
    """A checkpoint that already stores weight scales is not quantized a second time."""
    path, config = tiny
    tensors = safetensors_numpy.load_file(str(path / "model.safetensors"))
    tensors["audio_tower.layers.0.mlp.fc1.weight_scale"] = np.ones((1,), np.float32)
    safetensors_numpy.save_file(tensors, str(path / "model.safetensors"))
    with pytest.raises(ValueError, match="already quantized"):
        calibrate_glmasr(path, graph_ops, "fp8", config)


def test_unsupported_format_is_rejected(tiny) -> None:
    """Only fp8 is implemented."""
    path, config = tiny
    with pytest.raises(NotImplementedError, match="nvfp4"):
        calibrate_glmasr(path, graph_ops, "nvfp4", config)


def test_unknown_model_shape_is_reported(tmp_path: Path) -> None:
    """A model shape with no committed scales raises and names the calibration tool."""
    config = _write_checkpoint(tmp_path)
    with pytest.raises(NotImplementedError, match="calibrate_fp8_scales"):
        calibrate_glmasr(tmp_path, graph_ops, "fp8", config)


def test_committed_scales_cover_every_projection() -> None:
    """Every committed model shape has a scale for each projection of every layer."""
    path = Path(quantization_module.__file__).parent / ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8"))
    assert table, "no committed activation scales"
    for key, scales in table.items():
        match = re.fullmatch(r"glmasr-h(\d+)-l(\d+)-i(\d+)-e(\d+)-eh(\d+)", key)
        assert match, key
        hidden, layers, inter, enc_layers, enc_hidden = (int(g) for g in match.groups())
        config = ModelConfig.from_json(
            json.dumps(
                {
                    "model_type": "glmasr",
                    "audio_config": {
                        "hidden_size": enc_hidden,
                        "num_hidden_layers": enc_layers,
                        "num_attention_heads": 1,
                    },
                    "text_config": {
                        "hidden_size": hidden,
                        "num_hidden_layers": layers,
                        "intermediate_size": inter,
                    },
                }
            )
        )
        assert scale_key(config) == key
        assert {name for _, name in _projections(config)} == set(scales), key
        assert all(value > 0 for value in scales.values()), key


def test_encoder_weight_names() -> None:
    """Encoder engine module paths map to the internal projection names used by the scale file."""
    assert encoder_weight_name("audio.audio_tower.layers.3.self_attn.o_proj") == "audio.layer.3.w_o"
    assert encoder_weight_name("audio.audio_tower.layers.0.mlp.fc1") == "audio.layer.0.w_fc1"
    assert encoder_weight_name("projector.linear_2") == "projector.linear_2"


def test_load_weights_skips_quantized_projections(tiny) -> None:
    """Quantized decoder projections become shape-only placeholders; the encoder keeps no weight."""
    path, config = tiny
    ctx = calibrate_glmasr(path, graph_ops, "fp8", config)
    plugin = model.GlmAsrPlugin()
    weights = plugin.load_weights(str(path), config, precision="fp16", quant_ctx=ctx)
    placeholder = weights["layer.0.w_q"]
    assert placeholder.shape == (D, HEADS * (D // HEADS))
    assert placeholder.strides == (0, 0)  # broadcast zero, no storage
    assert "audio.audio_tower.layers.0.self_attn.q_proj.weight" not in weights
    assert "audio.audio_tower.layers.0.self_attn.q_proj.bias" in weights
    assert "projector.linear_1.weight" in weights
    assert weights["w_out"].shape == (D, VOCAB)
    assert weights["_attention_size"] == HEADS * (D // HEADS)


def test_calibration_tool_maps_module_names_to_weight_names() -> None:
    """The calibration tool maps HF module paths to internal weight names."""
    name = calibrate_fp8_scales.weight_name
    assert name("language_model.model.layers.4.self_attn.o_proj") == "layer.4.w_o"
    assert name("language_model.model.layers.0.mlp.gate_proj") == "layer.0.w_gate"
    assert name("audio_tower.layers.3.self_attn.q_proj") == "audio.layer.3.w_q"
    assert name("audio_tower.layers.1.mlp.fc2") == "audio.layer.1.w_fc2"
    assert name("multi_modal_projector.linear_1") is None
    assert name("audio_tower.layers.1.mlp.fc2.base") is None
    assert name("language_model.lm_head") is None


def test_calibration_tool_cli_parses_both_commands(monkeypatch, tmp_path: Path) -> None:
    """The calibration tool command line accepts fetch and calibrate."""
    seen = {}
    monkeypatch.setattr(
        calibrate_fp8_scales, "fetch_clips", lambda out, n: seen.update(fetch=(out, n))
    )
    monkeypatch.setattr(
        calibrate_fp8_scales, "calibrate", lambda d, c, n: seen.update(calibrate=(d, c, n))
    )
    monkeypatch.setattr("sys.argv", ["x", "fetch", "--out", str(tmp_path / "clips")])
    calibrate_fp8_scales.main()
    monkeypatch.setattr(
        "sys.argv", ["x", "calibrate", "--model-dir", str(tmp_path), "--clips", "clips"]
    )
    calibrate_fp8_scales.main()
    assert seen["fetch"] == (tmp_path / "clips", 96)
    assert seen["calibrate"] == (tmp_path, Path("clips"), 96)


def test_quantized_engines_build(tiny) -> None:
    """The encoder and decoder engines build with FP8 projections and are smaller than fp16."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("engine build needs a GPU")
    path, config = tiny
    plugin = model.GlmAsrPlugin()
    config.raw["_model_dir"] = str(path)
    config.raw["_fp32_layers"] = ()
    plain_weights = plugin.load_weights(str(path), config, precision="fp16")
    ctx = calibrate_glmasr(path, graph_ops, "fp8", config)
    fp8_weights = plugin.load_weights(str(path), config, precision="fp16", quant_ctx=ctx)

    plain_decoder = plugin.build_engine(config, plain_weights, 64, precision="fp16")
    plain_encoder = plugin.build_vision_engine(str(path), config, plain_weights, precision="fp16")
    fp8_decoder = plugin.build_engine(config, fp8_weights, 64, precision="fp16", quant_ctx=ctx)
    fp8_encoder = plugin.build_vision_engine(
        str(path), config, fp8_weights, precision="fp16", quant_ctx=ctx
    )

    runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    decoder = runtime.deserialize_cuda_engine(fp8_decoder)
    decoder_names = {decoder.get_tensor_name(i) for i in range(decoder.num_io_tensors)}
    assert "logits" in decoder_names
    encoder = runtime.deserialize_cuda_engine(fp8_encoder)
    encoder_names = {encoder.get_tensor_name(i) for i in range(encoder.num_io_tensors)}
    assert "audio_embeddings" in encoder_names
    assert len(fp8_decoder) < len(plain_decoder)
    assert len(fp8_encoder) < len(plain_encoder)


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
    ctx = GlmAsrQuantContext(scales={}, hf_names={}, graph_ops=graph_ops)
    config = _FakeBuilderConfig(routes)
    configure_fp8_build(config, ctx)
    assert config.build_route == expected
    assert config.profiling_verbosity is not None
    unquantized = _FakeBuilderConfig(routes)
    configure_fp8_build(unquantized, None)
    assert unquantized.build_route == "" and unquantized.profiling_verbosity is None
