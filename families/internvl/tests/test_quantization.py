# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""InternVL FP8 quantization: weight scaling, projection coverage, calibration tool, engine build."""

from __future__ import annotations

import json
import re
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")
safetensors_numpy = pytest.importorskip("safetensors.numpy")

from families.internvl import calibrate_fp8_scales, graph_ops, model  # noqa: E402
from families.internvl import quantization as quantization_module  # noqa: E402
from families.internvl.build_request import BuildRequest, coerce_request  # noqa: E402
from families.internvl.config import ModelConfig  # noqa: E402
from families.internvl.quantization import (  # noqa: E402
    ACTIVATION_SCALES_FILENAME,
    PROJECTIONS,
    InternVLQuantContext,
    calibrate_internvl,
    configure_fp8_build,
    quantize_fp8_weight,
    scale_key,
)

# GQA with 4 query and 2 key/value heads (4 heads: TensorRT 11.1 can segfault on 2 heads of 64).
D, FFN, VOCAB, LAYERS, HEADS, KV_HEADS = 128, 256, 100, 2, 4, 2


def _write_checkpoint(path: Path) -> ModelConfig:
    """Write a random tiny InternVL language model (HF tensor names) and return its config."""
    rng = np.random.default_rng(0)
    head_dim = D // HEADS

    def w(*shape):
        return rng.standard_normal(shape).astype(np.float32) * 0.05

    t: dict[str, np.ndarray] = {
        "language_model.model.embed_tokens.weight": w(VOCAB, D),
        "language_model.lm_head.weight": w(VOCAB, D),
        "language_model.model.norm.weight": np.ones(D, np.float32),
    }
    for i in range(LAYERS):
        layer = f"language_model.model.layers.{i}"
        t[f"{layer}.input_layernorm.weight"] = np.ones(D, np.float32)
        t[f"{layer}.post_attention_layernorm.weight"] = np.ones(D, np.float32)
        t[f"{layer}.self_attn.q_proj.weight"] = w(HEADS * head_dim, D)
        t[f"{layer}.self_attn.q_proj.bias"] = w(HEADS * head_dim)
        t[f"{layer}.self_attn.k_proj.weight"] = w(KV_HEADS * head_dim, D)
        t[f"{layer}.self_attn.k_proj.bias"] = w(KV_HEADS * head_dim)
        t[f"{layer}.self_attn.v_proj.weight"] = w(KV_HEADS * head_dim, D)
        t[f"{layer}.self_attn.v_proj.bias"] = w(KV_HEADS * head_dim)
        t[f"{layer}.self_attn.o_proj.weight"] = w(D, HEADS * head_dim)
        t[f"{layer}.mlp.gate_proj.weight"] = w(FFN, D)
        t[f"{layer}.mlp.up_proj.weight"] = w(FFN, D)
        t[f"{layer}.mlp.down_proj.weight"] = w(D, FFN)
    safetensors_numpy.save_file(t, str(path / "model.safetensors"))
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "internvl",
                "architectures": ["InternVLForConditionalGeneration"],
                "text_config": {
                    "model_type": "qwen2",
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
    return {
        f"layer.{layer}.{stem}": value
        for layer in range(config.num_hidden_layers)
        for _, stem in PROJECTIONS
    }


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


def test_projections_are_quantized_except_attention_output(tiny) -> None:
    """q/k/v and the MLP projections get FP8 weights; the attention output projection does not."""
    path, config = tiny
    ctx = calibrate_internvl(path, graph_ops, "fp8", config)
    assert len(ctx.scales) == LAYERS * 6
    assert ctx.should_quantize("layer.0.w_q")
    assert ctx.should_quantize("layer.1.w_down")
    # TensorRT 11.4 cannot compile an FP8 Quantize right after the fused attention
    assert not ctx.should_quantize("layer.0.w_o")
    entry = ctx.scales["layer.0.w_k"]
    assert (entry.weight.out_features, entry.weight.in_features) == (KV_HEADS * (D // HEADS), D)
    assert entry.weight.packed.dtype == np.uint8


def test_projection_without_scale_stays_unquantized(tmp_path: Path, monkeypatch) -> None:
    """A projection with no activation scale is left in fp16."""
    config = _write_checkpoint(tmp_path)
    scales = _all_scales(config)
    del scales["layer.0.w_down"]
    monkeypatch.setattr(quantization_module, "load_activation_scales", lambda cfg: scales)
    ctx = calibrate_internvl(tmp_path, graph_ops, "fp8", config)
    assert not ctx.should_quantize("layer.0.w_down")
    assert ctx.should_quantize("layer.0.w_up")


def test_already_quantized_checkpoint_is_rejected(tiny) -> None:
    """A checkpoint that already stores weight scales is not quantized a second time."""
    path, config = tiny
    tensors = safetensors_numpy.load_file(str(path / "model.safetensors"))
    tensors["language_model.model.layers.0.mlp.up_proj.weight_scale"] = np.ones((1,), np.float32)
    safetensors_numpy.save_file(tensors, str(path / "model.safetensors"))
    with pytest.raises(ValueError, match="already quantized"):
        calibrate_internvl(path, graph_ops, "fp8", config)


def test_unsupported_format_is_rejected(tiny) -> None:
    """Only fp8 is implemented."""
    path, config = tiny
    with pytest.raises(NotImplementedError, match="nvfp4"):
        calibrate_internvl(path, graph_ops, "nvfp4", config)


def test_unknown_model_shape_is_reported(tmp_path: Path) -> None:
    """A model shape with no committed scales raises and names the calibration tool."""
    config = _write_checkpoint(tmp_path)
    with pytest.raises(NotImplementedError, match="calibrate_fp8_scales"):
        calibrate_internvl(tmp_path, graph_ops, "fp8", config)


def test_committed_scales_cover_every_projection() -> None:
    """Every committed model shape has a scale for each quantized projection of every layer."""
    path = Path(quantization_module.__file__).parent / ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8"))
    assert table, "no committed activation scales"
    for key, scales in table.items():
        match = re.fullmatch(r"internvl-h(\d+)-l(\d+)-i(\d+)", key)
        assert match, key
        hidden, layers, inter = (int(g) for g in match.groups())
        config = ModelConfig.from_json(
            json.dumps(
                {
                    "model_type": "internvl",
                    "text_config": {
                        "hidden_size": hidden,
                        "num_hidden_layers": layers,
                        "intermediate_size": inter,
                    },
                }
            )
        )
        assert scale_key(config) == key
        expected = {
            f"layer.{layer}.{stem}"
            for layer in range(layers)
            for _, stem in PROJECTIONS
            if stem != "w_o"
        }
        assert set(scales) == expected, key
        assert all(value > 0 for value in scales.values()), key


def test_load_weights_skips_quantized_projections(tiny) -> None:
    """Quantized projections become shape-only placeholders; the attention output stays loaded."""
    path, config = tiny
    ctx = calibrate_internvl(path, graph_ops, "fp8", config)
    weights = model._InternVLModel().load_weights(str(path), config, quant_ctx=ctx)
    placeholder = weights["layer.0.w_q"]
    assert placeholder.shape == (D, HEADS * (D // HEADS))
    assert placeholder.strides == (0, 0)  # broadcast zero, no storage
    assert weights["layer.0.w_o"].strides != (0, 0)
    assert weights["layer.1.w_gate"].shape == (D, FFN)
    assert weights["w_out"].shape == (D, VOCAB)
    assert weights["_attention_size"] == HEADS * (D // HEADS)
    assert weights["_mlp_size"] == FFN


def test_calibration_tool_maps_module_names_to_weight_names() -> None:
    """The calibration tool maps HF module paths to internal weight names."""
    name = calibrate_fp8_scales.weight_name
    assert name("model.language_model.layers.4.self_attn.o_proj") == "layer.4.w_o"
    assert name("language_model.model.layers.0.mlp.gate_proj") == "layer.0.w_gate"
    assert name("model.vision_tower.encoder.layer.3.attention.q_proj") is None
    assert name("model.multi_modal_projector.linear_1") is None
    assert name("model.language_model.layers.1.mlp.down_proj.base") is None
    assert name("lm_head") is None


def test_calibration_tool_cli_parses_calibrate(monkeypatch, tmp_path: Path) -> None:
    """The calibration tool command line accepts calibrate."""
    seen = {}
    monkeypatch.setattr(
        calibrate_fp8_scales, "calibrate", lambda d, i, c, n: seen.update(calibrate=(d, i, c, n))
    )
    monkeypatch.setattr(
        "sys.argv", ["x", "calibrate", "--model-dir", str(tmp_path), "--images", "imgs"]
    )
    calibrate_fp8_scales.main()
    assert seen["calibrate"] == (tmp_path, Path("imgs"), 128, 64)


def test_build_request_carries_quantization(tmp_path: Path) -> None:
    """The typed request keeps the quantization choice and a legacy request converts to it."""
    request = BuildRequest(
        model_dir=tmp_path,
        output_path=tmp_path / "out",
        family="internvl",
        task="vision_language_generation",
        precision="fp16",
        quantization="fp8",
    )
    assert coerce_request(request).quantization == "fp8"
    assert (
        BuildRequest(
            model_dir=tmp_path,
            output_path=tmp_path / "out",
            family="internvl",
            task="vision_language_generation",
            precision="fp16",
        ).quantization
        is None
    )


def test_quantized_decoder_engine_builds(tiny) -> None:
    """The decoder engine builds with FP8 projections and is smaller than the fp16 engine."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("engine build needs a GPU")
    path, config = tiny
    plugin = model._InternVLModel()
    config.raw["_model_dir"] = str(path)
    plain_weights = plugin.load_weights(str(path), config)
    ctx = calibrate_internvl(path, graph_ops, "fp8", config)
    fp8_weights = plugin.load_weights(str(path), config, quant_ctx=ctx)

    plain = plugin.build_engine(config, plain_weights, 64, precision="fp16")
    fp8 = plugin.build_engine(config, fp8_weights, 64, precision="fp16", quant_ctx=ctx)

    engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(fp8)
    names = {engine.get_tensor_name(i) for i in range(engine.num_io_tensors)}
    assert "logits" in names
    assert len(fp8) < len(plain)


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
    ctx = InternVLQuantContext(scales={}, hf_names={}, graph_ops=graph_ops)
    config = _FakeBuilderConfig(routes)
    configure_fp8_build(config, ctx)
    assert config.build_route == expected
    assert config.profiling_verbosity is not None
    unquantized = _FakeBuilderConfig(routes)
    configure_fp8_build(unquantized, None)
    assert unquantized.build_route == "" and unquantized.profiling_verbosity is None
