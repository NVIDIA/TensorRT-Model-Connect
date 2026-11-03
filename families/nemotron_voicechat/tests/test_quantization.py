# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VoiceChat thinker FP8 self-quantization: quantizer, scale lookup, loader, tiny engine."""

from __future__ import annotations

import json
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")
torch = pytest.importorskip("torch")
safetensors_numpy = pytest.importorskip("safetensors.numpy")

from .. import calibrate_fp8_scales, native_core, quantization  # noqa: E402
from ..config import ModelConfig  # noqa: E402
from ..model import _THINKER_CONFIG  # noqa: E402

# Tiny hybrid thinker: Mamba-2, MLP, attention, MLP. Every projection is a multiple of 16
# wide, which the FP8 GEMM needs, and wide enough that the FP8 weights dominate the plan.
HIDDEN, MLP, VOCAB, HEADS, KV_HEADS, HEAD_DIM = 128, 256, 64, 4, 2, 32
M_HEADS, M_HEAD_DIM, N_GROUPS, D_STATE, D_CONV = 16, 16, 2, 16, 4
D_INNER = M_HEADS * M_HEAD_DIM
CONV_DIM = D_INNER + 2 * N_GROUPS * D_STATE
PATTERN = "M-*-"
LAYER_TYPES = native_core._parse_layer_types(PATTERN)
MAX_CACHE = 16


def _config() -> ModelConfig:
    raw = {
        "vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "intermediate_size": MLP,
        "num_hidden_layers": len(LAYER_TYPES),
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "rms_norm_eps": 1e-5,
        "hybrid_override_pattern": PATTERN,
        "mamba_num_heads": M_HEADS,
        "mamba_head_dim": M_HEAD_DIM,
        "n_groups": N_GROUPS,
        "ssm_state_size": D_STATE,
        "conv_kernel": D_CONV,
    }
    return ModelConfig.from_json(json.dumps(raw))


def _write_checkpoint(directory: Path, extra: dict | None = None) -> None:
    """Write a tiny random fp32 checkpoint with the VoiceChat tensor names."""
    rng = np.random.RandomState(0)

    def rand(*shape: int) -> np.ndarray:
        return (rng.randn(*shape) * 0.05).astype(np.float32)

    tensors = {"stt_model.embed_tokens.weight": rand(VOCAB, HIDDEN)}
    for layer, kind in enumerate(LAYER_TYPES):
        prefix = f"stt_model.llm.layers.{layer}"
        tensors[f"{prefix}.norm.weight"] = np.ones(HIDDEN, dtype=np.float32)
        if kind == "mamba2":
            tensors[f"{prefix}.mixer.in_proj.weight"] = rand(D_INNER + CONV_DIM + M_HEADS, HIDDEN)
            tensors[f"{prefix}.mixer.conv1d.weight"] = rand(CONV_DIM, 1, D_CONV)
            tensors[f"{prefix}.mixer.conv1d.bias"] = rand(CONV_DIM)
            tensors[f"{prefix}.mixer.out_proj.weight"] = rand(HIDDEN, D_INNER)
            tensors[f"{prefix}.mixer.A_log"] = rand(M_HEADS)
            tensors[f"{prefix}.mixer.D"] = rand(M_HEADS)
            tensors[f"{prefix}.mixer.dt_bias"] = rand(M_HEADS)
            tensors[f"{prefix}.mixer.norm.weight"] = np.ones(D_INNER, dtype=np.float32)
        elif kind == "mlp":
            tensors[f"{prefix}.mixer.up_proj.weight"] = rand(MLP, HIDDEN)
            tensors[f"{prefix}.mixer.down_proj.weight"] = rand(HIDDEN, MLP)
        else:
            tensors[f"{prefix}.mixer.q_proj.weight"] = rand(HEADS * HEAD_DIM, HIDDEN)
            tensors[f"{prefix}.mixer.k_proj.weight"] = rand(KV_HEADS * HEAD_DIM, HIDDEN)
            tensors[f"{prefix}.mixer.v_proj.weight"] = rand(KV_HEADS * HEAD_DIM, HIDDEN)
            tensors[f"{prefix}.mixer.o_proj.weight"] = rand(HIDDEN, HEADS * HEAD_DIM)
    tensors["stt_model.llm.norm_f.weight"] = np.ones(HIDDEN, dtype=np.float32)
    tensors["stt_model.lm_head.weight"] = rand(VOCAB, HIDDEN)
    tensors["stt_model.function_head.weight"] = rand(VOCAB, HIDDEN)
    tensors.update(extra or {})
    safetensors_numpy.save_file(tensors, str(directory / "model.safetensors"))


def _scales(skip: tuple[str, ...] = ()) -> dict[str, float]:
    return {n: 0.02 for n in quantization.projection_names(LAYER_TYPES) if n not in skip}


def _context(scales: dict[str, float] | None = None) -> quantization.VoiceChatQuantContext:
    return quantization.VoiceChatQuantContext(activation_scales=scales or _scales())


def test_quantize_fp8_weight_scale_and_bytes() -> None:
    """The weight scale is amax / 448 and the bytes decode back to the weight."""
    weight = np.random.RandomState(0).randn(64, 128).astype(np.float32)
    packed, scale = quantization.quantize_fp8_weight(weight)

    assert scale == pytest.approx(float(np.abs(weight).max()) / 448.0)
    assert packed.dtype == np.uint8 and packed.shape == weight.shape
    restored = packed.view(ml_dtypes.float8_e4m3fn).astype(np.float32) * scale
    assert np.abs(restored - weight).max() <= np.abs(weight).max() * 0.07  # e4m3 half-step


def test_quantize_fp8_weight_never_overflows_to_nan() -> None:
    """Values at the scale boundary clip to the largest finite fp8 value."""
    packed, _ = quantization.quantize_fp8_weight(np.full((4, 4), 3.0, dtype=np.float32))
    assert not np.isnan(packed.view(ml_dtypes.float8_e4m3fn).astype(np.float32)).any()


@pytest.mark.parametrize("bad", [0.0, np.nan, np.inf])
def test_quantize_fp8_weight_rejects_unscalable_tensor(bad: float) -> None:
    """An all-zero or non-finite weight has no usable scale and is rejected."""
    weight = np.zeros((4, 4), dtype=np.float32)
    weight[0, 0] = bad
    with pytest.raises(ValueError, match="amax"):
        quantization.quantize_fp8_weight(weight)


def test_projection_names_cover_the_recipe() -> None:
    """Mamba in/out, MLP up/down and attention q/k/v are quantized; the output projection is not."""
    names = quantization.projection_names(LAYER_TYPES)
    assert names == [
        "layer.0.mamba_in_proj",
        "layer.0.mamba_out_proj",
        "layer.1.w_up",
        "layer.1.w_down",
        "layer.2.w_q",
        "layer.2.w_k",
        "layer.2.w_v",
        "layer.3.w_up",
        "layer.3.w_down",
    ]
    assert not any(name.endswith("w_o") for name in names)


def test_committed_scales_cover_every_projection_of_the_model() -> None:
    """The committed scales file has an entry for every quantized projection of the model."""
    layer_types = native_core._parse_layer_types(str(_THINKER_CONFIG["hybrid_override_pattern"]))
    table = quantization.load_activation_scales(
        _THINKER_CONFIG["hidden_size"],
        _THINKER_CONFIG["num_hidden_layers"],
        _THINKER_CONFIG["intermediate_size"],
    )
    assert set(table) == set(quantization.projection_names(layer_types))
    assert all(np.isfinite(v) and v > 0 for v in table.values())


def test_create_context_for_the_real_model_config() -> None:
    """The committed scales build a context that wants every recipe projection."""
    layer_types = native_core._parse_layer_types(str(_THINKER_CONFIG["hybrid_override_pattern"]))
    ctx = quantization.create_fp8_context(
        _THINKER_CONFIG["hidden_size"],
        _THINKER_CONFIG["num_hidden_layers"],
        _THINKER_CONFIG["intermediate_size"],
        layer_types,
    )
    assert all(ctx.wants(name) for name in quantization.projection_names(layer_types))
    assert not ctx.wants("layer.1.w_o")


def test_unknown_model_config_raises() -> None:
    """A model config with no committed scales raises and names the calibration tool."""
    with pytest.raises(NotImplementedError, match="calibrate_fp8_scales"):
        quantization.create_fp8_context(HIDDEN, len(LAYER_TYPES), MLP, LAYER_TYPES)


def test_missing_projection_scale_raises_with_name_and_tool(monkeypatch) -> None:
    """A projection with no scale entry raises; it never stays fp32 silently."""
    table = _scales(skip=("layer.2.w_k",))
    monkeypatch.setattr(quantization, "load_activation_scales", lambda *args: table)
    with pytest.raises(ValueError, match=r"layer\.2\.w_k") as error:
        quantization.create_fp8_context(HIDDEN, len(LAYER_TYPES), MLP, LAYER_TYPES)
    assert "calibrate_fp8_scales" in str(error.value)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
def test_invalid_scale_raises(monkeypatch, bad: float) -> None:
    """A zero, negative or NaN activation scale is rejected."""
    table = _scales()
    table["layer.1.w_up"] = bad
    monkeypatch.setattr(quantization, "load_activation_scales", lambda *args: table)
    with pytest.raises(ValueError, match=r"layer\.1\.w_up"):
        quantization.create_fp8_context(HIDDEN, len(LAYER_TYPES), MLP, LAYER_TYPES)


def test_loader_quantizes_recipe_and_keeps_the_rest_fp32(tmp_path: Path) -> None:
    """Quantized projections get placeholders and FP8 bytes; o_proj and the heads stay fp32."""
    _write_checkpoint(tmp_path)
    ctx = _context()
    weights = native_core.VoiceChatThinkerBuilder().load_weights(
        str(tmp_path), _config(), quant=ctx
    )
    assert set(ctx.weights) == set(quantization.projection_names(LAYER_TYPES))
    for name in ctx.weights:
        assert weights[name].shape == (
            ctx.weights[name].in_features,
            ctx.weights[name].out_features,
        )
        assert weights[name].strides == (0, 0)  # shape-only placeholder, no memory
    assert weights["layer.2.w_o"].dtype == np.float32 and weights["layer.2.w_o"].strides != (0, 0)
    assert weights["w_lm_head"].shape == (HIDDEN, VOCAB)
    assert weights["w_function_head"].shape == (HIDDEN, VOCAB)


def test_loader_without_context_is_unchanged(tmp_path: Path) -> None:
    """With no context every projection is a real fp32 ``[in, out]`` array."""
    _write_checkpoint(tmp_path)
    weights = native_core.VoiceChatThinkerBuilder().load_weights(str(tmp_path), _config())
    assert weights["layer.0.mamba_in_proj"].shape == (HIDDEN, D_INNER + CONV_DIM + M_HEADS)
    assert weights["layer.0.mamba_in_proj"].strides != (0, 0)


def test_quantized_checkpoint_is_rejected(tmp_path: Path) -> None:
    """A checkpoint that already carries quantization scales is refused."""
    _write_checkpoint(
        tmp_path,
        {"stt_model.llm.layers.1.mixer.up_proj.weight_scale": np.ones(1, dtype=np.float32)},
    )
    with pytest.raises(ValueError, match="already quantized"):
        native_core.VoiceChatThinkerBuilder().load_weights(
            str(tmp_path), _config(), quant=_context()
        )


def test_wants_only_projections_with_scales() -> None:
    """A projection without an activation scale is not quantized."""
    ctx = _context(_scales(skip=("layer.1.w_down",)))
    assert ctx.wants("layer.1.w_up") and not ctx.wants("layer.1.w_down")
    with pytest.raises(ValueError, match="not a quantized projection"):
        ctx.quantize_projection("layer.1.w_down", np.ones((HIDDEN, MLP), dtype=np.float32))


def _build_and_run(tmp_path: Path, quant) -> tuple[bytes, np.ndarray]:
    """Build the tiny thinker (optionally FP8) and return the plan and the first-step logits."""
    config = _config()
    weights = native_core.VoiceChatThinkerBuilder().load_weights(str(tmp_path), config, quant=quant)
    plan = native_core.build_thinker_engine(config, weights, MAX_CACHE, quant=quant)
    thinker = calibrate_fp8_scales.ThinkerRunner(
        plan, LAYER_TYPES.count("mamba2"), LAYER_TYPES.count("attention"), MAX_CACHE
    )
    audio = torch.tensor(np.random.RandomState(1).randn(HIDDEN).astype(np.float32), device="cuda")
    logits, _ = thinker.step(3, 5, 7, audio, True)
    return plan, logits.float().cpu().numpy()


def test_tiny_fp8_engine_matches_fp32_and_is_smaller(tmp_path: Path) -> None:
    """The FP8 engine builds, is much smaller than the fp32 plan and tracks its logits."""
    _write_checkpoint(tmp_path)
    plan32, logits32 = _build_and_run(tmp_path, None)
    ctx = _context()
    plan8, logits8 = _build_and_run(tmp_path, ctx)

    assert len(plan8) < 0.7 * len(plan32)
    assert np.isfinite(logits8).all()
    cosine = float(logits32 @ logits8 / (np.linalg.norm(logits32) * np.linalg.norm(logits8)))
    assert cosine > 0.98


def test_calibration_context_records_every_projection_input(tmp_path: Path) -> None:
    """The instrumented plan has one amax output per quantized projection and the same logits."""
    _write_checkpoint(tmp_path)
    recorder = calibrate_fp8_scales.CalibrationContext()
    config = _config()
    weights = native_core.VoiceChatThinkerBuilder().load_weights(str(tmp_path), config)
    plan = native_core.build_thinker_engine(config, weights, MAX_CACHE, quant=recorder)
    assert set(recorder.outputs) == set(quantization.projection_names(LAYER_TYPES))
    thinker = calibrate_fp8_scales.ThinkerRunner(
        plan, LAYER_TYPES.count("mamba2"), LAYER_TYPES.count("attention"), MAX_CACHE
    )
    thinker.step(3, 5, 7, None, False)
    values = [float(thinker.t[name].max()) for name in recorder.outputs.values()]
    assert all(np.isfinite(v) and v > 0 for v in values)
    _, plain_logits = _build_and_run(tmp_path, None)
    thinker.reset()
    audio = torch.tensor(np.random.RandomState(1).randn(HIDDEN).astype(np.float32), device="cuda")
    logits, _ = thinker.step(3, 5, 7, audio, True)
    assert np.allclose(logits.float().cpu().numpy(), plain_logits, atol=1e-4)


class _FakeThinker:
    """Stands in for ThinkerRunner: records inputs and emits a scripted text-token stream."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def reset(self) -> None:
        self.calls.clear()

    def step(self, text, timeline, function, audio, use_audio):
        index = len(self.calls)
        self.calls.append((text, timeline, function, use_audio))
        logits = torch.zeros(32)
        # Prompt steps predict nothing; audio steps follow the script.
        target = self.script[index - 3] if index >= 3 and index - 3 < len(self.script) else 12
        logits[target % 32] = 1.0
        return logits, torch.zeros(32)


def test_run_conversation_follows_the_native_loop() -> None:
    """Prompt on the timeline channel, BOS forced after the speech, stop at EOS."""
    frames = np.zeros((10, 8), dtype=np.float32)
    thinker = _FakeThinker([12, 1, 12, 12, 5, 6, 2, 12, 12, 12])  # a stray BOS in the speech
    inputs, produced = calibrate_fp8_scales.run_conversation(thinker, frames, 4, [1, 20, 2])

    assert [call[1] for call in thinker.calls[:3]] == [1, 20, 2]  # timeline = prompt ids
    assert all(call[3] is False for call in thinker.calls[:3])
    assert all(call[3] is True for call in thinker.calls[3:])
    assert produced[0][0] == 12 and produced[1][0] == 12  # stray BOS suppressed while listening
    assert produced[3][0] == 1  # BOS forced after the last speech frame
    assert thinker.calls[3 + 4][0] == 1  # and fed back as the next previous text token
    assert produced[-1][0] == 2 and len(produced) <= 10  # stops at EOS


def test_mel_spectrogram_shape() -> None:
    """The mel front end returns ``[mel_bins, samples // 160]`` log features."""
    samples = np.random.RandomState(0).randn(16000).astype(np.float32) * 0.1
    filters = np.abs(np.random.RandomState(1).randn(257, 128)).astype(np.float32)
    mel = calibrate_fp8_scales.mel_spectrogram(samples, filters, np.hanning(400).astype(np.float32))
    assert mel.shape == (128, 100) and np.isfinite(mel).all()
