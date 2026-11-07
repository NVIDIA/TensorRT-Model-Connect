# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeBERTa FP8 quantization: weight scaling, projection coverage, calibration tool, build."""

from __future__ import annotations

import json
import re
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")

from tensorrt_model_connect import BuildRequest  # noqa: E402

from .. import calibrate_fp8_scales, model  # noqa: E402
from .. import quantization as quantization_module  # noqa: E402
from ..config import ModelConfig  # noqa: E402
from ..quantization import (  # noqa: E402
    ACTIVATION_SCALES_FILENAME,
    LAYER_PROJECTIONS,
    create_quant_context,
    projections,
    quantize_fp8_weight,
    scale_key,
)

D, FFN, LAYERS, HEADS, VOCAB, REL, SEQ = 64, 128, 2, 4, 64, 32, 24
_CONFIG = {
    "model_type": "deberta",
    "hidden_size": D,
    "intermediate_size": FFN,
    "num_hidden_layers": LAYERS,
    "num_attention_heads": HEADS,
    "vocab_size": VOCAB,
    "relative_attention": True,
    "position_biased_input": False,
    "pos_att_type": "c2p|p2c",
    "max_relative_positions": -1,
    "max_position_embeddings": REL,
    "type_vocab_size": 0,
    "hidden_act": "gelu",
    "layer_norm_eps": 1e-7,
}


def _tiny_config() -> ModelConfig:
    return ModelConfig.from_json(json.dumps(_CONFIG))


def _tiny_weights(seed: int = 0) -> dict:
    """Random float32 weights in the layout ``load_weights`` returns ([in, out] matrices)."""
    rng = np.random.default_rng(seed)

    def normal(*shape, std=0.05):
        return rng.normal(0.0, std, shape).astype(np.float32)

    weights = {
        "embedding": normal(VOCAB, D, std=1.0),
        "embed_norm": np.ones(D, np.float32),
        "embed_norm_beta": normal(D),
        "rel_embeddings": normal(2 * REL, D, std=1.0),
        "_deberta_config": {
            "position_biased_input": False,
            "type_vocab_size": 0,
            "max_relative_positions": REL,
            "pos_att_type": ["c2p", "p2c"],
        },
    }
    for layer in range(LAYERS):
        p = f"layer.{layer}"
        for name in ("w_q", "w_k", "w_v", "w_o", "pos_proj", "pos_q_proj"):
            weights[f"{p}.{name}"] = normal(D, D)
        for name in ("q_bias", "v_bias", "o_bias", "pos_q_proj_bias", "fc2_bias"):
            weights[f"{p}.{name}"] = normal(D)
        weights[f"{p}.w_fc1"] = normal(D, FFN)
        weights[f"{p}.fc1_bias"] = normal(FFN)
        weights[f"{p}.w_fc2"] = normal(FFN, D)
        for norm in ("post_attn_norm", "output_norm"):
            weights[f"{p}.{norm}"] = np.ones(D, np.float32)
            weights[f"{p}.{norm}_beta"] = normal(D)
    return weights


def _all_scales(value: float = 0.05) -> dict[str, float | None]:
    return {name: value for name in projections(LAYERS)}


@pytest.fixture
def tiny(monkeypatch):
    """Tiny weights, their config, and activation scales for every projection."""
    weights = _tiny_weights()
    scales = _all_scales()
    monkeypatch.setattr(quantization_module, "load_activation_scales", lambda _key: scales)
    return weights, _tiny_config(), scales


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
    """Four projections per layer; in_proj holds the query, key and value weights."""
    weights, config, _ = tiny
    ctx = create_quant_context(weights, config, "fp8")
    assert len(ctx.projections) == LAYERS * len(LAYER_PROJECTIONS) == len(projections(LAYERS))
    in_proj = ctx.projections["encoder.layer.1.attention.self.in_proj"]
    assert len(in_proj.weights) == 3
    fc1 = ctx.projections["encoder.layer.1.intermediate.dense"].weights[0]
    assert fc1.packed.shape == (FFN, D) == (fc1.out_features, fc1.in_features)
    assert ctx.has_fp8_weights
    assert ctx.should_quantize("encoder.layer.0.attention.output.dense")
    assert not ctx.should_quantize("encoder.layer.0.attention.self.pos_proj")
    assert not ctx.should_quantize("embeddings.word_embeddings")


def test_query_key_value_keep_their_own_weight_scales(tiny) -> None:
    """The three parts of in_proj share the input scale, not the weight scale."""
    weights, config, _ = tiny
    weights["layer.0.w_k"] = weights["layer.0.w_k"] * 3.0
    ctx = create_quant_context(weights, config, "fp8")
    q, k, v = ctx.projections["encoder.layer.0.attention.self.in_proj"].weights
    assert k.weight_scale == pytest.approx(float(np.abs(weights["layer.0.w_k"]).max()) / 448.0)
    assert k.weight_scale > 2.0 * max(q.weight_scale, v.weight_scale)
    # The [in, out] loaded weight is stored as the checkpoint's [out, in].
    restored = q.packed.view(ml_dtypes.float8_e4m3fn).astype(np.float32) * q.weight_scale
    np.testing.assert_allclose(restored.T, weights["layer.0.w_q"], atol=0.07 * 0.25)


def test_outlier_projection_stays_unquantized(tiny) -> None:
    """A null entry is an explicit decision: that projection keeps the work type."""
    weights, config, scales = tiny
    scales["encoder.layer.1.output.dense"] = None
    ctx = create_quant_context(weights, config, "fp8")
    assert not ctx.should_quantize("encoder.layer.1.output.dense")
    assert ctx.project(None, None, "encoder.layer.1.output.dense") is None
    assert ctx.should_quantize("encoder.layer.0.output.dense")


def test_projection_without_an_entry_is_an_error(tiny) -> None:
    """A projection missing from the table raises; there is no silent fallback."""
    weights, config, scales = tiny
    del scales["encoder.layer.0.attention.self.in_proj"]
    with pytest.raises(ValueError, match="calibrate_fp8_scales.py"):
        create_quant_context(weights, config, "fp8")


def test_entry_without_a_projection_is_an_error(tiny) -> None:
    """A table written for another depth raises."""
    weights, config, scales = tiny
    scales["encoder.layer.7.intermediate.dense"] = 0.05
    with pytest.raises(ValueError, match="without a projection"):
        create_quant_context(weights, config, "fp8")


def test_non_positive_scale_is_an_error(tiny) -> None:
    """A zero scale would divide by zero in Quantize."""
    weights, config, scales = tiny
    scales["encoder.layer.0.intermediate.dense"] = 0.0
    with pytest.raises(ValueError, match="positive"):
        create_quant_context(weights, config, "fp8")


def test_unsupported_format_is_rejected(tiny) -> None:
    """Only fp8 is implemented."""
    weights, config, _ = tiny
    with pytest.raises(NotImplementedError, match="nvfp4"):
        create_quant_context(weights, config, "nvfp4")


def test_unknown_checkpoint_names_the_calibration_tool() -> None:
    """A checkpoint with no committed scales fails and says how to add them."""
    with pytest.raises(NotImplementedError, match="calibrate_fp8_scales.py"):
        create_quant_context(_tiny_weights(), _tiny_config(), "fp8")


def test_scale_key_names_the_config_and_the_weights() -> None:
    """The key holds the model config and two weight sums, so a fine-tune gets its own key."""
    weights = _tiny_weights()
    key = scale_key(_tiny_config(), weights)
    assert key.startswith(f"deberta-h{D}-l{LAYERS}-i{FFN}-n{HEADS}-v{VOCAB}-r{REL}-w")
    tuned = dict(weights)
    tuned["layer.0.w_q"] = weights["layer.0.w_q"] * 1.01
    assert scale_key(_tiny_config(), tuned) != key


def test_committed_scales_cover_every_projection() -> None:
    """Every committed checkpoint has an entry per projection; outliers are null."""
    table = json.loads(
        (Path(quantization_module.__file__).parent / ACTIVATION_SCALES_FILENAME).read_text()
    )
    assert table
    for key, scales in table.items():
        match = re.fullmatch(
            r"deberta-h(\d+)-l(\d+)-i(\d+)-n(\d+)-v(\d+)-r(\d+)-w[0-9.e+]+-[0-9.e+]+", key
        )
        assert match, key
        layers = int(match.group(2))
        assert set(scales) == set(projections(layers)), key
        values = [v for v in scales.values() if v is not None]
        assert all(np.isfinite(v) and v > 0 for v in values), key
        assert len(values) >= 0.9 * len(scales), f"{key}: too many projections left unquantized"


def test_calibration_tool_maps_module_names_to_table_names() -> None:
    """Only the encoder projections that read the hidden rows are calibrated."""
    name = calibrate_fp8_scales.projection_name
    assert name("encoder.layer.3.intermediate.dense") == "encoder.layer.3.intermediate.dense"
    assert name("encoder.layer.0.attention.self.in_proj") is not None
    assert name("encoder.layer.0.attention.output.dense") is not None
    assert name("encoder.layer.11.output.dense") is not None
    assert name("encoder.layer.0.attention.self.pos_proj") is None
    assert name("encoder.layer.0.attention.self.pos_q_proj") is None
    assert name("encoder.layer.0.output.LayerNorm") is None
    assert name("encoder.rel_embeddings") is None
    assert name("embeddings.word_embeddings") is None
    assert name("pooler.dense") is None


def test_calibration_tool_frames_like_the_runtime() -> None:
    """One text is ``[CLS] a [SEP]``; a pair adds ``b [SEP]``; both stay within the limit."""

    class _Tokenizer:
        cls_token_id, sep_token_id = 1, 2

        def __call__(self, text, add_special_tokens):
            assert not add_special_tokens
            return {"input_ids": [10 + len(word) for word in text.split()]}

    tokenizer = _Tokenizer()
    assert calibrate_fp8_scales.frame_ids(tokenizer, "a bb") == [1, 11, 12, 2]
    assert calibrate_fp8_scales.frame_ids(tokenizer, "a", "bb") == [1, 11, 2, 12, 2]
    long = " ".join(["a"] * 600)
    assert len(calibrate_fp8_scales.frame_ids(tokenizer, long)) == 512
    framed = calibrate_fp8_scales.frame_ids(tokenizer, long, long)
    assert len(framed) == 512 and framed.count(2) == 2


def test_calibration_tool_marks_outliers_against_the_median_of_their_kind() -> None:
    """A scale above 20x the median of the same projection in other layers becomes None."""
    scales = {name: 0.01 for name in projections(5)}
    scales["encoder.layer.4.output.dense"] = 0.5
    scales["encoder.layer.4.intermediate.dense"] = 0.15
    marked = calibrate_fp8_scales.mark_outliers(scales)
    assert marked["encoder.layer.4.output.dense"] is None
    assert marked["encoder.layer.4.intermediate.dense"] == 0.15
    assert marked["encoder.layer.0.output.dense"] == 0.01
    assert sum(v is None for v in marked.values()) == 1


def test_calibration_tool_cli_parses_both_commands(monkeypatch, tmp_path: Path) -> None:
    """The fetch and calibrate commands reach their functions with the parsed arguments."""
    calls = []
    monkeypatch.setattr(calibrate_fp8_scales, "fetch_texts", lambda *args: calls.append(args))
    monkeypatch.setattr(calibrate_fp8_scales, "calibrate", lambda *args: calls.append(args))
    out = str(tmp_path / "t.json")
    monkeypatch.setattr(
        "sys.argv", ["calibrate", "fetch", "--out", out, "--paragraphs", "10", "--pairs", "5"]
    )
    calibrate_fp8_scales.main()
    monkeypatch.setattr(
        "sys.argv",
        ["calibrate", "calibrate", "--model-dir", str(tmp_path), "--texts", str(tmp_path)],
    )
    calibrate_fp8_scales.main()
    assert calls == [(tmp_path / "t.json", 10, 5), (tmp_path, tmp_path, 800)]


def _run(plan: bytes, torch, ids, mask):
    engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(plan)
    assert engine is not None
    context = engine.create_execution_context()
    out = torch.empty(tuple(engine.get_tensor_shape("hidden_states")), device="cuda")
    token_types = torch.zeros_like(ids)
    context.set_tensor_address("input_ids", ids.data_ptr())
    context.set_tensor_address("token_type_ids", token_types.data_ptr())
    context.set_tensor_address("attention_mask", mask.data_ptr())
    context.set_tensor_address("hidden_states", out.data_ptr())
    context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    return out


@pytest.mark.gpu
@pytest.mark.trt
def test_quantized_engine_builds_and_tracks_the_unquantized_one(tiny) -> None:
    """The FP8 engine builds, is smaller, is finite and tracks the unquantized engine."""
    torch = pytest.importorskip("torch")
    weights, config, _ = tiny
    ctx = create_quant_context(weights, config, "fp8")
    real = SEQ - 5
    ids = torch.randint(0, VOCAB, (SEQ,), dtype=torch.int32, device="cuda")
    mask = (torch.arange(SEQ, device="cuda") < real).to(torch.int32)

    builder = model._DebertaModel()
    plain = builder.build_engine(config, weights, SEQ, precision="fp16")
    quant = builder.build_engine(config, weights, SEQ, precision="fp16", quant_ctx=ctx)
    assert len(quant) < len(plain)

    reference = _run(plain, torch, ids, mask)[:real].reshape(-1)
    actual = _run(quant, torch, ids, mask)[:real].reshape(-1)
    assert torch.isfinite(actual).all()
    cosine = torch.nn.functional.cosine_similarity(reference, actual, dim=0).item()
    assert cosine > 0.9, cosine


def _checkpoint(tmp_path: Path, **extra) -> Path:
    (tmp_path / "config.json").write_text(json.dumps({**_CONFIG, **extra}), encoding="utf-8")
    return tmp_path


def _request(tmp_path: Path, **overrides) -> BuildRequest:
    fields = {
        "model_dir": _checkpoint(tmp_path),
        "output_path": tmp_path / "b",
        "family": "deberta",
        "task": "text_to_pooled_features",
        "precision": "fp16",
        "max_sequence_length": 16,
        "quantization": "fp8",
    }
    fields.update(overrides)
    return BuildRequest(**fields)


def test_build_refuses_other_quantization_formats(tmp_path: Path) -> None:
    """Only fp8 is accepted."""
    with pytest.raises(NotImplementedError, match="'fp8' or none"):
        model.build(_request(tmp_path, quantization="nvfp4"), None)


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_build_requires_fp16_for_fp8(tmp_path: Path, monkeypatch, precision: str) -> None:
    """FP8 runs with the fp16 work type only; the refusal comes before the checkpoint is read."""

    def fail(*args, **kwargs):
        raise AssertionError("checkpoint read before the precision check")

    monkeypatch.setattr(model._DebertaModel, "load_weights", fail)
    with pytest.raises(ValueError, match="requires precision=fp16"):
        model.build(_request(tmp_path, precision=precision), None)


def test_build_refuses_fp8_with_tensor_parallelism(tmp_path: Path) -> None:
    """The tensor-parallel path has no quantized build."""
    with pytest.raises(ValueError, match="tensor-parallel"):
        model.build(_request(tmp_path, tensor_parallel_size=2), None)


def test_build_refuses_an_already_quantized_checkpoint(tmp_path: Path) -> None:
    """Self-quantization needs the original checkpoint."""
    request = _request(tmp_path)
    (tmp_path / "config.json").write_text(
        json.dumps({**_CONFIG, "quantization_config": {"quant_method": "modelopt"}})
    )
    with pytest.raises(ValueError, match="already quantized"):
        model.build(request, None)
    (tmp_path / "config.json").write_text(json.dumps(_CONFIG))
    (tmp_path / "hf_quant_config.json").write_text("{}")
    with pytest.raises(ValueError, match="already quantized"):
        model.build(request, None)
