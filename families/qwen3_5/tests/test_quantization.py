# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.5 NVFP4 and FP8 quantization: checkpoint reading, query/gate split, placeholders, engine build."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import ml_dtypes
import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")
torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

from .. import calibrate_fp8_scales, graph_ops, model  # noqa: E402
from .. import quantization as quantization_module  # noqa: E402
from ..build_request import BuildRequest, coerce_request  # noqa: E402
from ..config import ModelConfig  # noqa: E402
from ..quantization import (  # noqa: E402
    ACTIVATION_SCALES_FILENAME,
    Qwen35QuantContext,
    _NVFP4Weight,
    _split_query_gate,
    calibrate_qwen35,
    is_nvfp4_checkpoint,
    quantize_fp8_weight,
)

# Every quantized input dimension is a multiple of the 16-element NVFP4 block.
VOCAB, HIDDEN, HEADS, KV_HEADS, HEAD_DIM, MLP = 32, 256, 4, 2, 64, 128
ATTENTION, KV = HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM
DELTA_HEADS, DELTA_KV_HEADS, DELTA_DIM = 4, 2, 64
D_INNER = DELTA_HEADS * DELTA_DIM
CONV_DIM = 2 * DELTA_KV_HEADS * DELTA_DIM + D_INNER
LAYER_TYPES = ["deltanet", "attention"]
PREFIX = "model.language_model.layers"

_DELTANET = {
    "linear_attn.in_proj_qkv": (CONV_DIM, HIDDEN),
    "linear_attn.in_proj_z": (D_INNER, HIDDEN),
    "linear_attn.in_proj_a": (DELTA_HEADS, HIDDEN),
    "linear_attn.in_proj_b": (DELTA_HEADS, HIDDEN),
    "linear_attn.out_proj": (HIDDEN, D_INNER),
}
_ATTENTION = {
    "self_attn.q_proj": (HEADS * 2 * HEAD_DIM, HIDDEN),  # query and gate rows, interleaved
    "self_attn.k_proj": (KV, HIDDEN),
    "self_attn.v_proj": (KV, HIDDEN),
    "self_attn.o_proj": (HIDDEN, ATTENTION),
}
_MLP = {
    "mlp.gate_proj": (MLP, HIDDEN),
    "mlp.up_proj": (MLP, HIDDEN),
    "mlp.down_proj": (HIDDEN, MLP),
}


def _config_json() -> dict:
    return {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": {
            "model_type": "qwen3_5_text",
            "vocab_size": VOCAB,
            "hidden_size": HIDDEN,
            "num_hidden_layers": len(LAYER_TYPES),
            "num_attention_heads": HEADS,
            "num_key_value_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "intermediate_size": MLP,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 64,
            "layer_types": ["linear_attention", "full_attention"],
            "linear_num_value_heads": DELTA_HEADS,
            "linear_num_key_heads": DELTA_KV_HEADS,
            "linear_value_head_dim": DELTA_DIM,
            "linear_key_head_dim": DELTA_DIM,
            "linear_conv_kernel_dim": 4,
            "rope_parameters": {"rope_theta": 10000.0, "partial_rotary_factor": 0.25},
        },
    }


def _write_checkpoint(
    path: Path, *, quantized: bool, input_scale: float = 0.02
) -> dict[str, torch.Tensor]:
    """Write a tiny hybrid Qwen3.5 checkpoint, ModelOpt NVFP4 style when ``quantized``."""
    gen = torch.Generator().manual_seed(0)
    tensors: dict[str, torch.Tensor] = {
        "model.language_model.embed_tokens.weight": torch.randn(
            VOCAB, HIDDEN, generator=gen
        ).bfloat16(),
        "model.language_model.norm.weight": torch.zeros(HIDDEN).bfloat16(),
        "lm_head.weight": torch.randn(VOCAB, HIDDEN, generator=gen).bfloat16(),
    }

    def add_projection(name: str, out_f: int, in_f: int) -> None:
        if not quantized:
            tensors[f"{name}.weight"] = torch.randn(out_f, in_f, generator=gen).bfloat16()
            return
        tensors[f"{name}.weight"] = torch.randint(
            0, 256, (out_f, in_f // 2), generator=gen, dtype=torch.uint8
        )
        scale = (torch.rand(out_f, in_f // 16, generator=gen) + 0.5).to(torch.float8_e4m3fn)
        tensors[f"{name}.weight_scale"] = scale
        tensors[f"{name}.weight_scale_2"] = torch.tensor(0.004)
        tensors[f"{name}.input_scale"] = torch.tensor(input_scale)

    for layer, kind in enumerate(LAYER_TYPES):
        base = f"{PREFIX}.{layer}"
        tensors[f"{base}.input_layernorm.weight"] = torch.zeros(HIDDEN).bfloat16()
        tensors[f"{base}.post_attention_layernorm.weight"] = torch.zeros(HIDDEN).bfloat16()
        if kind == "deltanet":
            mixer = _DELTANET
            tensors[f"{base}.linear_attn.A_log"] = torch.zeros(DELTA_HEADS).bfloat16()
            tensors[f"{base}.linear_attn.dt_bias"] = torch.ones(DELTA_HEADS).bfloat16()
            tensors[f"{base}.linear_attn.conv1d.weight"] = torch.randn(
                CONV_DIM, 1, 4, generator=gen
            ).bfloat16()
            tensors[f"{base}.linear_attn.norm.weight"] = torch.ones(DELTA_DIM).bfloat16()
        else:
            mixer = _ATTENTION
            tensors[f"{base}.self_attn.q_norm.weight"] = torch.zeros(HEAD_DIM).bfloat16()
            tensors[f"{base}.self_attn.k_norm.weight"] = torch.zeros(HEAD_DIM).bfloat16()
        for stem, (out_f, in_f) in {**mixer, **_MLP}.items():
            add_projection(f"{base}.{stem}", out_f, in_f)
    path.mkdir(parents=True, exist_ok=True)
    safetensors_torch.save_file(tensors, str(path / "model.safetensors"))
    (path / "config.json").write_text(json.dumps(_config_json()))
    return tensors


def _calibrate(path: Path) -> Qwen35QuantContext:
    return calibrate_qwen35(
        path, graph_ops, "nvfp4", layer_types=LAYER_TYPES, num_heads=HEADS, head_dim=HEAD_DIM
    )


def test_calibrate_reads_every_projection_bit_exact(tmp_path: Path) -> None:
    """Every projection of both layer kinds is read, and its bytes are the checkpoint's."""
    tensors = _write_checkpoint(tmp_path, quantized=True)
    ctx = _calibrate(tmp_path)

    # deltanet: 5 mixer + 3 mlp; attention: q split into 2, k, v, o + 3 mlp
    assert len(ctx.scales) == 8 + 8
    qkv = ctx.scales["layer.0.deltanet_in_proj_qkv"]
    assert isinstance(qkv.weight, _NVFP4Weight)
    assert (qkv.weight.out_features, qkv.weight.in_features) == (CONV_DIM, HIDDEN)
    assert np.array_equal(
        qkv.weight.packed,
        tensors[f"{PREFIX}.0.linear_attn.in_proj_qkv.weight"].view(torch.uint8).numpy(),
    )
    assert qkv.input_scale == pytest.approx(0.02)
    assert qkv.weight.global_scale == pytest.approx(0.004)
    # tiny projection (one output per head) is still read as a quantized weight
    assert ctx.scales["layer.0.deltanet_a_proj"].weight.out_features == DELTA_HEADS
    assert ctx.should_quantize("layer.1.w_down")
    assert not ctx.should_quantize("layer.5.w_q")


def test_query_gate_split_follows_the_per_head_row_layout(tmp_path: Path) -> None:
    """q_proj rows are [head0 query, head0 gate, head1 query, ...]; each half is bit-exact."""
    tensors = _write_checkpoint(tmp_path, quantized=True)
    ctx = _calibrate(tmp_path)
    packed = tensors[f"{PREFIX}.1.self_attn.q_proj.weight"].view(torch.uint8).numpy()
    scale = tensors[f"{PREFIX}.1.self_attn.q_proj.weight_scale"].view(torch.uint8).numpy()

    query, gate = ctx.scales["layer.1.w_q"].weight, ctx.scales["layer.1.w_gate_attn"].weight
    assert (query.out_features, gate.out_features) == (ATTENTION, ATTENTION)
    assert query.packed.shape == (ATTENTION, HIDDEN // 2)
    for head in range(HEADS):
        first = head * 2 * HEAD_DIM
        rows = slice(head * HEAD_DIM, (head + 1) * HEAD_DIM)
        assert np.array_equal(query.packed[rows], packed[first : first + HEAD_DIM])
        assert np.array_equal(gate.packed[rows], packed[first + HEAD_DIM : first + 2 * HEAD_DIM])
        assert np.array_equal(query.block_scale_f8[rows], scale[first : first + HEAD_DIM])
        assert np.array_equal(
            gate.block_scale_f8[rows], scale[first + HEAD_DIM : first + 2 * HEAD_DIM]
        )
    with pytest.raises(ValueError, match="rows"):
        _split_query_gate(query, HEADS + 1, HEAD_DIM)


def test_calibrate_rejects_unquantized_checkpoint(tmp_path: Path) -> None:
    """A BF16 checkpoint fails with a format error instead of silently staying unquantized."""
    _write_checkpoint(tmp_path, quantized=False)
    with pytest.raises(ValueError, match="not an nvfp4 ModelOpt tensor"):
        _calibrate(tmp_path)


def test_calibrate_rejects_unknown_format(tmp_path: Path) -> None:
    """Only nvfp4 and fp8 are supported."""
    with pytest.raises(NotImplementedError):
        calibrate_qwen35(
            tmp_path, graph_ops, "int4", layer_types=LAYER_TYPES, num_heads=HEADS, head_dim=HEAD_DIM
        )


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
def test_calibrate_rejects_invalid_input_scale(tmp_path: Path, bad: float) -> None:
    """A zero, negative or NaN input_scale is reported as such."""
    _write_checkpoint(tmp_path, quantized=True, input_scale=bad)
    with pytest.raises(ValueError, match="input_scale must be one positive finite value"):
        _calibrate(tmp_path)


def test_is_nvfp4_checkpoint(tmp_path: Path) -> None:
    """Quantized checkpoints are recognised so a build without the flag can refuse them."""
    _write_checkpoint(tmp_path / "quant", quantized=True)
    _write_checkpoint(tmp_path / "plain", quantized=False)
    assert is_nvfp4_checkpoint(tmp_path / "quant")
    assert not is_nvfp4_checkpoint(tmp_path / "plain")
    assert not is_nvfp4_checkpoint(tmp_path / "missing")


def test_build_request_accepts_quantization(tmp_path: Path) -> None:
    """quantization is a real request field; 'none' and None both mean unquantized."""
    request = BuildRequest(
        model_dir=tmp_path,
        output_path=tmp_path / "out",
        family="qwen3_5",
        task="text_generation",
        precision="fp16",
        quantization="nvfp4",
    )
    assert coerce_request(request).quantization == "nvfp4"
    assert (
        BuildRequest(
            model_dir=tmp_path,
            output_path=tmp_path / "out",
            family="qwen3_5",
            task="text_generation",
            precision="fp16",
        ).quantization
        is None
    )


def test_load_weights_uses_shape_only_placeholders(tmp_path: Path) -> None:
    """Quantized projections get shape-only placeholders; other tensors are still loaded."""
    _write_checkpoint(tmp_path, quantized=True)
    config = ModelConfig.from_dir(tmp_path)
    ctx = _calibrate(tmp_path)
    weights = model._Qwen35Model().load_weights(str(tmp_path), config, ctx)

    assert weights["layer.0.deltanet_in_proj_qkv"].shape == (HIDDEN, CONV_DIM)
    assert weights["layer.0.deltanet_out_proj"].shape == (D_INNER, HIDDEN)
    assert weights["layer.1.w_q"].shape == (HIDDEN, ATTENTION)
    assert weights["layer.1.w_gate_attn"].shape == (HIDDEN, ATTENTION)
    assert weights["layer.1.w_down"].shape == (MLP, HIDDEN)
    # Placeholders must not hold a full-size copy of the weight.
    assert weights["layer.0.deltanet_in_proj_qkv"].strides == (0, 0)
    assert weights["layer.1.w_o"].strides == (0, 0)
    assert weights["layer.0.conv1d_weight"].shape == (CONV_DIM, 4)
    assert weights["w_lm_head"].shape == (HIDDEN, VOCAB)


@pytest.mark.gpu
@pytest.mark.trt
def test_engine_builds_with_packed_weights(tmp_path: Path) -> None:
    """The quantized engine builds and stores packed (not full-precision) weights."""
    plain_dir, quant_dir = tmp_path / "plain", tmp_path / "quant"
    _write_checkpoint(plain_dir, quantized=False)
    _write_checkpoint(quant_dir, quantized=True)
    instance = model._Qwen35Model()

    config = ModelConfig.from_dir(plain_dir)
    plain = instance.build_engine(
        config, instance.load_weights(str(plain_dir), config), 4, precision="fp16"
    )

    config = ModelConfig.from_dir(quant_dir)
    ctx = _calibrate(quant_dir)
    weights = instance.load_weights(str(quant_dir), config, ctx)
    quantized = instance.build_engine(config, weights, 4, precision="fp16", quant_ctx=ctx)

    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(quantized)
    assert engine is not None
    names = {engine.get_tensor_name(i) for i in range(engine.num_io_tensors)}
    assert {"token_id", "logits"} <= names
    # The projections are 4 bits instead of 16; the embedding and lm_head are not.
    assert len(quantized) < len(plain)


# ---------------------------------------------------------------------------
# FP8 self-quantization of the BF16 checkpoint
# ---------------------------------------------------------------------------


def _all_projection_names() -> list[str]:
    """Internal names of every projection that FP8 quantizes in the tiny model."""
    names = []
    for layer, kind in enumerate(LAYER_TYPES):
        stems = (
            [
                "deltanet_in_proj_qkv",
                "deltanet_z_proj",
                "deltanet_a_proj",
                "deltanet_b_proj",
                "deltanet_out_proj",
            ]
            if kind == "deltanet"
            else ["w_q", "w_k", "w_v", "w_o"]
        )
        names += [f"layer.{layer}.{stem}" for stem in (*stems, "w_gate", "w_up", "w_down")]
    return names


def _fp8_context(path: Path, monkeypatch, names=None) -> Qwen35QuantContext:
    """FP8 context for the tiny BF16 checkpoint with fabricated activation scales."""
    names = _all_projection_names() if names is None else names
    monkeypatch.setattr(
        quantization_module, "load_activation_scales", lambda config: {n: 0.05 for n in names}
    )
    config = ModelConfig.from_dir(path)
    return calibrate_qwen35(
        path,
        graph_ops,
        "fp8",
        layer_types=LAYER_TYPES,
        num_heads=HEADS,
        head_dim=HEAD_DIM,
        config=config,
    )


def test_quantize_fp8_weight_scale_is_amax_over_448() -> None:
    """The scale maps the largest weight to the largest finite e4m3 value."""
    weight = np.random.default_rng(0).standard_normal((8, 32)).astype(np.float32)
    packed, scale = quantize_fp8_weight(weight)
    assert scale == pytest.approx(float(np.abs(weight).max()) / 448.0)
    assert packed.dtype == np.uint8 and packed.shape == weight.shape
    restored = packed.view(ml_dtypes.float8_e4m3fn).astype(np.float32) * scale
    # e4m3 keeps 3 mantissa bits, so the relative error is below 2**-4 for normal values.
    assert np.abs(restored - weight).max() <= float(np.abs(weight).max()) * 2**-4


def test_quantize_fp8_weight_rejects_all_zero_weight() -> None:
    with pytest.raises(ValueError, match="amax"):
        quantize_fp8_weight(np.zeros((4, 16), dtype=np.float32))


def test_fp8_calibrate_quantizes_every_projection(tmp_path: Path, monkeypatch) -> None:
    """Each projection is quantized from the BF16 weight with its own scale."""
    tensors = _write_checkpoint(tmp_path, quantized=False)
    ctx = _fp8_context(tmp_path, monkeypatch)
    assert ctx.has_fp8_weights
    assert set(ctx.scales) == set(_all_projection_names()) | {"layer.1.w_gate_attn"}

    raw = tensors[f"{PREFIX}.0.mlp.down_proj.weight"].float().numpy()
    entry = ctx.scales["layer.0.w_down"]
    packed, scale = quantize_fp8_weight(raw)
    assert entry.input_scale == pytest.approx(0.05)
    assert entry.weight.weight_scale == pytest.approx(scale)
    np.testing.assert_array_equal(entry.weight.packed, packed)


def test_fp8_query_and_gate_halves_share_the_input_scale(tmp_path: Path, monkeypatch) -> None:
    """q_proj rows are split per head; each half has its own weight scale and the same input scale."""
    tensors = _write_checkpoint(tmp_path, quantized=False)
    ctx = _fp8_context(tmp_path, monkeypatch)
    raw = tensors[f"{PREFIX}.1.self_attn.q_proj.weight"].float().numpy()
    per_head = raw.reshape(HEADS, 2 * HEAD_DIM, HIDDEN)
    for name, part in (
        ("layer.1.w_q", slice(0, HEAD_DIM)),
        ("layer.1.w_gate_attn", slice(HEAD_DIM, 2 * HEAD_DIM)),
    ):
        half = per_head[:, part, :].reshape(ATTENTION, HIDDEN)
        packed, scale = quantize_fp8_weight(half)
        np.testing.assert_array_equal(ctx.scales[name].weight.packed, packed)
        assert ctx.scales[name].weight.weight_scale == pytest.approx(scale)
        assert ctx.scales[name].input_scale == pytest.approx(0.05)


def test_fp8_projection_without_scale_stays_unquantized(tmp_path: Path, monkeypatch) -> None:
    """A projection with no calibrated scale is built unquantized."""
    _write_checkpoint(tmp_path, quantized=False)
    names = [n for n in _all_projection_names() if n != "layer.0.deltanet_a_proj"]
    ctx = _fp8_context(tmp_path, monkeypatch, names)
    assert not ctx.should_quantize("layer.0.deltanet_a_proj")
    assert ctx.should_quantize("layer.0.deltanet_b_proj")


def test_fp8_rejects_checkpoint_that_is_already_quantized(tmp_path: Path, monkeypatch) -> None:
    """FP8 self-quantization needs the BF16 weights; an NVFP4 checkpoint is refused."""
    _write_checkpoint(tmp_path, quantized=True)
    with pytest.raises(ValueError, match="already quantized"):
        _fp8_context(tmp_path, monkeypatch)


def test_fp8_without_scales_for_the_model_shape_is_refused(tmp_path: Path) -> None:
    """A model shape with no committed activation scales fails with the available keys."""
    _write_checkpoint(tmp_path, quantized=False)
    config = ModelConfig.from_dir(tmp_path)
    with pytest.raises(NotImplementedError, match="no calibrated FP8 activation scales"):
        calibrate_qwen35(
            tmp_path,
            graph_ops,
            "fp8",
            layer_types=LAYER_TYPES,
            num_heads=HEADS,
            head_dim=HEAD_DIM,
            config=config,
        )


def test_committed_activation_scales_cover_every_projection() -> None:
    """Every committed model shape has a positive finite scale for each projection."""
    table = json.loads(
        (Path(quantization_module.__file__).parent / ACTIVATION_SCALES_FILENAME).read_text()
    )
    assert table
    for key, scales in table.items():
        assert all(np.isfinite(v) and v > 0 for v in scales.values()), key
        layers = sorted({int(name.split(".")[1]) for name in scales})
        assert layers == list(range(int(re.search(r"-l(\d+)-", key).group(1)))), key
        for layer in layers:
            names = {n.split(".", 2)[2] for n in scales if n.startswith(f"layer.{layer}.")}
            assert {"w_gate", "w_up", "w_down"} <= names, (key, layer)
            attention = {"w_q", "w_k", "w_v", "w_o"} <= names
            deltanet = {"deltanet_in_proj_qkv", "deltanet_z_proj", "deltanet_out_proj"} <= names
            assert attention or deltanet, (key, layer)


@pytest.mark.parametrize("model_type", ["qwen3_5", "qwen3.5", "Qwen3.5"])
def test_scale_key_normalizes_model_type_spelling(model_type: str) -> None:
    """Both accepted model_type spellings select the same committed scale table."""
    config = SimpleNamespace(
        model_type=model_type, hidden_size=8, num_hidden_layers=2, intermediate_size=16
    )
    assert quantization_module.scale_key(config) == "qwen3_5-h8-l2-i16"


@pytest.mark.gpu
@pytest.mark.trt
def test_fp8_engine_builds_with_fp8_weights(tmp_path: Path, monkeypatch) -> None:
    """The FP8 engine builds, keeps FP8 weights, and is smaller than the plain engine."""
    plain_dir, quant_dir = tmp_path / "plain", tmp_path / "quant"
    _write_checkpoint(plain_dir, quantized=False)
    _write_checkpoint(quant_dir, quantized=False)
    instance = model._Qwen35Model()

    config = ModelConfig.from_dir(plain_dir)
    plain = instance.build_engine(
        config, instance.load_weights(str(plain_dir), config), 4, precision="fp16"
    )

    config = ModelConfig.from_dir(quant_dir)
    ctx = _fp8_context(quant_dir, monkeypatch)
    weights = instance.load_weights(str(quant_dir), config, ctx)
    quantized = instance.build_engine(config, weights, 4, precision="fp16", quant_ctx=ctx)

    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(quantized)
    assert engine is not None
    names = {engine.get_tensor_name(i) for i in range(engine.num_io_tensors)}
    assert {"token_id", "logits"} <= names
    assert len(quantized) < len(plain)


def test_calibration_tool_maps_module_names_to_weight_stems() -> None:
    """The calibration tool maps module names to internal weight names."""
    match = calibrate_fp8_scales._PROJECTION_NAME.search("model.layers.3.linear_attn.in_proj_qkv")
    assert match is not None and match.group(1) == "3"
    assert calibrate_fp8_scales._WEIGHT_STEM[match.group(2)] == "deltanet_in_proj_qkv"
    assert calibrate_fp8_scales._PROJECTION_NAME.search("model.layers.3.mlp.down_proj.base") is None


def test_calibration_tool_cli_parses_both_commands(monkeypatch, tmp_path: Path) -> None:
    """The calibration tool command line accepts fetch and calibrate."""
    seen = {}
    monkeypatch.setattr(
        calibrate_fp8_scales, "fetch_prompts", lambda out, n: seen.update(fetch=(out, n))
    )
    monkeypatch.setattr(
        calibrate_fp8_scales, "calibrate", lambda d, p, n: seen.update(calibrate=(d, p, n))
    )
    monkeypatch.setattr("sys.argv", ["x", "fetch", "--out", str(tmp_path / "p.jsonl")])
    calibrate_fp8_scales.main()
    monkeypatch.setattr(
        "sys.argv", ["x", "calibrate", "--model-dir", str(tmp_path), "--prompts", "p.jsonl"]
    )
    calibrate_fp8_scales.main()
    assert seen["fetch"] == (tmp_path / "p.jsonl", 64)
    assert seen["calibrate"] == (tmp_path, Path("p.jsonl"), 512)
