# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LFM2 FP8 / INT4-AWQ self-quantization: weight quantizers, scale lookup, context and engine build."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import ml_dtypes
import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")

from .. import calibrate_awq_scales, calibrate_fp8_scales, quantization  # noqa: E402
from .. import model as lfm2_model  # noqa: E402
from ..checkpoint_mapper import WeightDict, _open_safetensors  # noqa: E402
from ..config import validate_dense_lfm2_config  # noqa: E402
from ..model import build_lfm2_engine  # noqa: E402
from ..quantization import (  # noqa: E402
    Lfm2QuantContext,
    _AWQWeight,
    _LayerScales,
    awq_identity,
    calibrate_lfm2_awq,
    calibrate_lfm2_fp8,
    configure_quantized_build,
    load_activation_scales,
    quantize_awq_weight,
    quantize_fp8_weight,
    scale_key,
)

# Wide enough that FP8 weight savings dominate the per-layer Q/DQ plan overhead.
HIDDEN, MLP, VOCAB, HEADS, KV_HEADS = 512, 2048, 64, 8, 4
LAYER_TYPES = ["conv", "full_attention", "conv", "full_attention"]
_WEIGHT_STEMS = dict(quantization.PROJECTIONS)


def _raw_config() -> dict:
    """Return a small dense LFM2 ``config.json`` for the tests."""
    return {
        "architectures": ["Lfm2ForCausalLM"],
        "model_type": "lfm2",
        "vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "block_dim": HIDDEN,
        "block_ff_dim": MLP,
        "intermediate_size": MLP,
        "block_auto_adjust_ff_dim": False,
        "num_hidden_layers": len(LAYER_TYPES),
        "num_attention_heads": HEADS,
        "num_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "layer_types": LAYER_TYPES,
        "conv_L_cache": 3,
        "norm_eps": 1e-5,
        "rope_theta": 1_000_000.0,
        "max_position_embeddings": 256,
        "conv_dim": HIDDEN,
        "conv_dim_out": HIDDEN,
    }


def _config():
    """Return the raw tiny config wrapped the way the family loader passes it."""
    return SimpleNamespace(raw=_raw_config())


def _resolved():
    """Return the validated dense LFM2 config of the tiny model."""
    return validate_dense_lfm2_config(_config())


def _weights() -> WeightDict:
    """Return random weights in the loaded ``[in, out]`` layout of the tiny model."""
    cfg = _resolved()
    rng = np.random.RandomState(0)

    def rand(*shape: int) -> np.ndarray:
        return (rng.randn(*shape) * 0.05).astype(np.float32)

    weights = WeightDict()
    weights["embedding"] = rand(VOCAB, HIDDEN)
    for layer, layer_type in enumerate(cfg.layer_types):
        prefix = f"layer.{layer}"
        weights[f"{prefix}.operator_norm"] = np.ones(HIDDEN, dtype=np.float32)
        weights[f"{prefix}.ffn_norm"] = np.ones(HIDDEN, dtype=np.float32)
        if layer_type == "conv":
            weights[f"{prefix}.conv_in"] = rand(HIDDEN, 3 * HIDDEN)
            weights[f"{prefix}.conv_weight"] = rand(HIDDEN, 3)
            weights[f"{prefix}.conv_out"] = rand(HIDDEN, HIDDEN)
        else:
            weights[f"{prefix}.w_q"] = rand(HIDDEN, cfg.attention_size)
            weights[f"{prefix}.w_k"] = rand(HIDDEN, cfg.kv_attention_size)
            weights[f"{prefix}.w_v"] = rand(HIDDEN, cfg.kv_attention_size)
            weights[f"{prefix}.w_o"] = rand(cfg.attention_size, HIDDEN)
            weights[f"{prefix}.q_norm"] = np.ones(cfg.head_dim, dtype=np.float32)
            weights[f"{prefix}.k_norm"] = np.ones(cfg.head_dim, dtype=np.float32)
        weights[f"{prefix}.w1"] = rand(HIDDEN, MLP)
        weights[f"{prefix}.w3"] = rand(HIDDEN, MLP)
        weights[f"{prefix}.w2"] = rand(MLP, HIDDEN)
    weights["final_norm"] = np.ones(HIDDEN, dtype=np.float32)
    weights["w_lm_head"] = np.ascontiguousarray(weights["embedding"].T)
    weights["_layer_types"] = list(cfg.layer_types)
    weights["_num_attention_layers"] = cfg.num_attention_layers
    weights["_num_conv_layers"] = cfg.num_conv_layers
    weights["_hidden_size"] = HIDDEN
    weights["_intermediate_size"] = MLP
    weights["_head_dim"] = cfg.head_dim
    weights["_conv_l_cache"] = cfg.conv_l_cache
    return weights


def _activation_scales(skip: tuple[str, ...] = ()) -> dict[str, float]:
    """Return one activation scale per existing quantized projection of the tiny model."""
    weights = _weights()
    return {
        f"layer.{layer}.{stem}": 0.02
        for layer in range(len(LAYER_TYPES))
        for stem in _WEIGHT_STEMS.values()
        if stem not in skip and f"layer.{layer}.{stem}" in weights
    }


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
    weight = np.full((4, 4), 3.0, dtype=np.float32)
    packed, _ = quantize_fp8_weight(weight)
    assert not np.isnan(packed.view(ml_dtypes.float8_e4m3fn).astype(np.float32)).any()


def test_quantize_fp8_weight_rejects_zero_tensor() -> None:
    """An all-zero weight has no usable scale and is refused."""
    with pytest.raises(ValueError):
        quantize_fp8_weight(np.zeros((2, 2), dtype=np.float32))


def test_scale_key_identifies_model_shape_and_checkpoint() -> None:
    """The scale key names the model shape and tells checkpoints of the same shape apart."""
    base = scale_key(_resolved(), np.ones(HIDDEN, dtype=np.float32))
    assert base == f"lfm2-h{HIDDEN}-l{len(LAYER_TYPES)}-i{MLP}-n{float(HIDDEN):.4f}"
    assert base == scale_key(_resolved(), np.ones(HIDDEN, dtype=np.float32))
    assert base != scale_key(_resolved(), np.full(HIDDEN, 1.5, dtype=np.float32))


def test_missing_scale_entry_is_refused() -> None:
    """A checkpoint without committed scales fails with a clear error."""
    with pytest.raises(NotImplementedError, match="no calibrated FP8 activation scales"):
        load_activation_scales(_resolved(), np.ones(HIDDEN, dtype=np.float32))


def test_committed_scales_cover_every_quantized_projection() -> None:
    """Each committed shape has a positive scale for every projection its layers carry."""
    path = Path(quantization.__file__).parent / quantization.ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8"))
    # LFM2-350M, 700M, 1.2B, 2.6B and LFM2.5-1.2B-Instruct (same shape as LFM2-1.2B).
    shapes = sorted(key.rsplit("-n", 1)[0] for key in table)
    assert shapes == sorted(
        [
            "lfm2-h1024-l16-i4608",
            "lfm2-h1536-l16-i6912",
            "lfm2-h2048-l16-i8192",
            "lfm2-h2048-l16-i8192",
            "lfm2-h2048-l30-i10752",
        ]
    )
    for key, scales in table.items():
        assert re.fullmatch(r"lfm2-h\d+-l\d+-i\d+-n-?\d+\.\d{4}", key), key
        assert all(value > 0 for value in scales.values()), key
        layers = {int(name.split(".")[1]) for name in scales}
        # Every layer has the SwiGLU projections plus either conv or q/k/v projections.
        for layer in layers:
            names = {name.split(".", 2)[2] for name in scales if name.startswith(f"layer.{layer}.")}
            assert {"w1", "w2", "w3"} <= names, (key, layer)
            assert names - {"w1", "w2", "w3"} in ({"conv_in", "conv_out"}, {"w_q", "w_k", "w_v"})


def test_unsupported_projections_are_not_in_the_quantized_set() -> None:
    """The attention output projection and the depthwise conv stay unquantized."""
    assert "w_o" not in _WEIGHT_STEMS.values()
    assert "conv_weight" not in _WEIGHT_STEMS.values()


def test_context_quantizes_the_loaded_weights(monkeypatch) -> None:
    """The context quantizes the transposed loaded weights and keeps the committed scales."""
    weights = _weights()
    monkeypatch.setattr(
        quantization, "load_activation_scales", lambda config, final_norm: _activation_scales()
    )
    ctx = calibrate_lfm2_fp8(_resolved(), weights)

    # conv layers: conv_in, conv_out + 3 MLP; attention layers: q, k, v + 3 MLP.
    assert len(ctx.scales) == 2 * 5 + 2 * 6
    entry = ctx.scales["layer.1.w2"]
    assert (entry.weight.out_features, entry.weight.in_features) == (HIDDEN, MLP)
    assert entry.input_scale == pytest.approx(0.02)
    conv_in = ctx.scales["layer.0.conv_in"]
    assert (conv_in.weight.out_features, conv_in.weight.in_features) == (3 * HIDDEN, HIDDEN)

    packed, scale = quantize_fp8_weight(np.asarray(weights["layer.1.w2"], dtype=np.float32).T)
    assert np.array_equal(entry.weight.packed, packed)
    assert entry.weight.weight_scale == pytest.approx(scale)
    assert not ctx.should_quantize("layer.1.w_o")


def test_projection_without_a_scale_stays_unquantized(monkeypatch) -> None:
    """A projection without a scale is left unquantized."""
    monkeypatch.setattr(
        quantization,
        "load_activation_scales",
        lambda config, final_norm: _activation_scales(("w2",)),
    )
    ctx = calibrate_lfm2_fp8(_resolved(), _weights())
    assert not ctx.should_quantize("layer.0.w2")
    assert ctx.should_quantize("layer.0.w1")


def test_already_quantized_checkpoint_is_refused(monkeypatch) -> None:
    """A checkpoint that carries FP8 scales is refused instead of quantized twice."""
    weights = _weights()
    weights["layer.0.q_proj.weight_scale"] = np.ones(1, dtype=np.float32)
    monkeypatch.setattr(
        quantization, "load_activation_scales", lambda config, final_norm: _activation_scales()
    )
    with pytest.raises(ValueError, match="already quantized"):
        calibrate_lfm2_fp8(_resolved(), weights)


class _FakeBuilderConfig:
    """Stands in for IBuilderConfig; only the attributes the build options touch."""

    def __init__(self, all_build_routes: str) -> None:
        self.all_build_routes = all_build_routes
        self.build_route = ""
        self.profiling_verbosity = None
        self.builder_optimization_level = 1


@pytest.mark.parametrize(
    ("routes", "expected"),
    [('{"peep:match_dual_gemm": {}}', "-peep:match_dual_gemm=off"), ("", "")],
)
def test_dual_gemm_route_is_set_only_when_listed(routes: str, expected: str) -> None:
    """The FP8 build route is applied only when the TensorRT build lists the knob."""
    fp8 = quantization._FP8Weight(np.zeros((1, 1), dtype=np.uint8), 1.0, 1, 1)
    ctx = Lfm2QuantContext(scales={"w": _LayerScales(0.02, fp8)})
    config = _FakeBuilderConfig(routes)
    configure_quantized_build(config, ctx)
    assert config.build_route == expected
    assert config.profiling_verbosity is not None
    assert config.builder_optimization_level == 1
    unquantized = _FakeBuilderConfig(routes)
    configure_quantized_build(unquantized, None)
    assert unquantized.build_route == "" and unquantized.profiling_verbosity is None


@pytest.mark.gpu
@pytest.mark.trt
@pytest.mark.parametrize("precision", ["fp16", "bf16"])
def test_engine_builds_with_fp8_weights(monkeypatch, precision: str) -> None:
    """The quantized hybrid engine builds, keeps its I/O contract and stores FP8 weights."""
    monkeypatch.setattr(
        quantization, "load_activation_scales", lambda config, final_norm: _activation_scales()
    )
    ctx = calibrate_lfm2_fp8(_resolved(), _weights())

    plain = build_lfm2_engine(_config(), _weights(), 64, precision=precision)
    quantized = build_lfm2_engine(_config(), _weights(), 64, precision=precision, quant_ctx=ctx)

    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(quantized)
    assert engine is not None
    names = {engine.get_tensor_name(i) for i in range(engine.num_io_tensors)}
    assert {"token_id", "logits", "present_k_0", "present_conv_0", "conv_state_0"} <= names
    # FP8 halves the bytes of every quantized projection.
    assert len(quantized) < len(plain)


@pytest.mark.gpu
@pytest.mark.trt
def test_engine_attends_over_a_runtime_sized_key_row_slice(monkeypatch) -> None:
    """Attention reads ``len(key_positions)`` cache rows, from one row to the capacity."""
    capacity, optimum = 64, 32
    monkeypatch.setattr(lfm2_model, "ACTIVE_KEY_ROWS_OPTIMUM", optimum)
    plan = build_lfm2_engine(_config(), _weights(), capacity, precision="bf16")

    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(plan)
    assert engine is not None
    assert engine.get_tensor_mode("key_positions") == trt.TensorIOMode.INPUT
    assert engine.get_tensor_dtype("key_positions") == trt.int32
    assert tuple(engine.get_tensor_shape("key_positions")) == (-1,)
    assert [tuple(shape) for shape in engine.get_tensor_profile_shape("key_positions", 0)] == [
        (1,),
        (optimum,),
        (capacity,),
    ]
    # The cache itself keeps its full static capacity.
    assert tuple(engine.get_tensor_shape("cache_k_0")) == (1, KV_HEADS, capacity, HIDDEN // HEADS)


def test_build_requires_calibrated_scales_for_fp8(monkeypatch) -> None:
    """quantization=fp8 on a model shape without committed scales fails before any engine build."""
    from ..model import build

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
        max_sequence_length=64,
        tensor_parallel_size=1,
        quantization="fp8",
        fp32_layers=None,
        mtp_seq_len=None,
        backend="trt",
    )
    monkeypatch.setattr("families.lfm2.model._load_config", lambda model_dir: _config())
    monkeypatch.setattr("families.lfm2.model.load_lfm2_weights", lambda *a, **k: _weights())
    with pytest.raises(NotImplementedError, match="no calibrated FP8 activation scales"):
        build(request, writer=None)


def test_build_rejects_other_quantization_modes(monkeypatch) -> None:
    """Only fp8 and int4_awq are supported; other modes are refused loudly."""
    from ..model import build

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
        max_sequence_length=64,
        tensor_parallel_size=1,
        quantization="nvfp4",
        fp32_layers=None,
        mtp_seq_len=None,
    )
    monkeypatch.setattr("families.lfm2.model._load_config", lambda model_dir: _config())
    with pytest.raises(NotImplementedError, match="quantization='nvfp4'"):
        build(request, writer=None)


def test_calibration_tool_maps_module_names_to_weight_stems() -> None:
    """The calibration tool maps module names to internal weight names."""
    pattern = calibrate_fp8_scales._PROJECTION_NAME
    stems = calibrate_fp8_scales._WEIGHT_STEM
    for module, weight in (
        ("model.layers.3.feed_forward.w2", "w2"),
        ("model.layers.3.conv.in_proj", "conv_in"),
        ("model.layers.3.conv.out_proj", "conv_out"),
        ("model.layers.3.self_attn.q_proj", "w_q"),
    ):
        match = pattern.search(module)
        assert match is not None and match.group(1) == "3"
        assert stems[match.group(2)] == weight
    assert pattern.search("model.layers.3.feed_forward.w2.base") is None
    assert pattern.search("model.layers.3.self_attn.out_proj") is None
    assert pattern.search("model.layers.3.conv.conv") is None


def test_calibration_tool_cli_parses_both_commands(monkeypatch, tmp_path: Path) -> None:
    """The calibration tool command line accepts fetch and calibrate."""
    seen = {}
    monkeypatch.setattr(
        calibrate_fp8_scales, "fetch_prompts", lambda out, n: seen.update(fetch=(out, n))
    )
    monkeypatch.setattr(
        calibrate_fp8_scales, "calibrate", lambda d, p, n, o: seen.update(calibrate=(d, p, n, o))
    )
    monkeypatch.setattr("sys.argv", ["x", "fetch", "--out", str(tmp_path / "p.jsonl")])
    calibrate_fp8_scales.main()
    monkeypatch.setattr(
        "sys.argv", ["x", "calibrate", "--model-dir", str(tmp_path), "--prompts", "p.jsonl"]
    )
    calibrate_fp8_scales.main()
    assert seen["fetch"] == (tmp_path / "p.jsonl", 64)
    assert seen["calibrate"] == (tmp_path, Path("p.jsonl"), 512, None)


def test_render_prompt_falls_back_to_plain_text() -> None:
    """A tokenizer without a chat template gets the messages joined as plain text."""

    class _NoTemplate:
        chat_template = None

    text = calibrate_fp8_scales.render_prompt(
        _NoTemplate(), [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    )
    assert text == "a\n\nb"


class _EngineStepper:
    """Run the hybrid engine in one optimization profile over shared device state."""

    def __init__(self, engine, profile: int, state: dict, torch) -> None:
        self.torch = torch
        self.state = state
        self.context = engine.create_execution_context()
        stream = torch.cuda.current_stream().cuda_stream
        assert self.context.set_optimization_profile_async(profile, stream)
        self.logits = torch.empty((1, VOCAB), dtype=torch.float32, device="cuda")

    def step(self, tokens: list[int], position: int):
        """Launch ``tokens`` at ``position``; commit the convolution state; return logits."""
        torch = self.torch
        length = position + len(tokens)
        inputs = {
            "token_id": torch.tensor(tokens, dtype=torch.int32, device="cuda"),
            "position_id": torch.arange(position, length, dtype=torch.int32, device="cuda"),
            "cache_write_indices": torch.tensor([position], dtype=torch.int32, device="cuda"),
            "key_value_lengths": torch.tensor([length], dtype=torch.int32, device="cuda"),
            "key_positions": torch.arange(length, dtype=torch.int32, device="cuda"),
        }
        for name, value in inputs.items():
            if name in ("token_id", "position_id", "key_positions"):
                assert self.context.set_input_shape(name, tuple(value.shape))
            self.context.set_tensor_address(name, value.data_ptr())
        for name, value in self.state.items():
            self.context.set_tensor_address(name, value.data_ptr())
        self.context.set_tensor_address("logits", self.logits.data_ptr())
        assert self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        for name in [name for name in self.state if name.startswith("conv_state_")]:
            self.state[name].copy_(self.state[name.replace("conv_state_", "present_conv_")])
        return self.logits.clone()


def _engine_state(engine, torch) -> dict:
    """Allocate zeroed KV and convolution buffers; present KV aliases the cache."""
    state = {}
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        if name.startswith(("cache_k_", "cache_v_", "conv_state_", "present_conv_")):
            shape = tuple(engine.get_tensor_shape(name))
            state[name] = torch.zeros(shape, dtype=torch.float16, device="cuda")
    for name in [name for name in state if name.startswith(("cache_k_", "cache_v_"))]:
        state[name.replace("cache_", "present_")] = state[name]
    return state


@pytest.mark.gpu
@pytest.mark.trt
def test_chunked_prefill_matches_token_by_token_prefill() -> None:
    """A multi-token prefill launch leaves the same logits and state as one token per launch."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    plan = build_lfm2_engine(_config(), _weights(), 64, precision="fp16")
    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(plan)
    assert engine is not None and engine.num_optimization_profiles == 2
    assert [tuple(shape) for shape in engine.get_tensor_profile_shape("token_id", 0)] == [
        (1,),
        (64,),
        (64,),
    ]
    assert [tuple(shape) for shape in engine.get_tensor_profile_shape("token_id", 1)] == [(1,)] * 3

    prompt, next_token = [5, 17, 3, 42, 9, 21, 33], 11
    chunked_state = _engine_state(engine, torch)
    prefill = _EngineStepper(engine, 0, chunked_state, torch)
    decode = _EngineStepper(engine, 1, chunked_state, torch)
    chunked = [prefill.step(prompt[:4], 0), prefill.step(prompt[4:], 4)]
    chunked.append(decode.step([next_token], len(prompt)))

    stepped_state = _engine_state(engine, torch)
    single = _EngineStepper(engine, 1, stepped_state, torch)
    stepped = [single.step([token], position) for position, token in enumerate(prompt)]
    stepped.append(single.step([next_token], len(prompt)))

    for chunk_logits, step_logits in ((chunked[0], stepped[3]), (chunked[1], stepped[6])):
        torch.testing.assert_close(chunk_logits, step_logits, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(chunked[2], stepped[7], atol=2e-2, rtol=2e-2)
    for name, value in chunked_state.items():
        if name.startswith("conv_state_"):
            torch.testing.assert_close(value, stepped_state[name], atol=2e-2, rtol=2e-2)


# ---------------------------------------------------------------- INT4-AWQ

_HF_STEMS = {stem: hf_stem for hf_stem, stem in quantization.ALL_PROJECTIONS}


def _write_checkpoint(directory: Path, weights: WeightDict, **config_updates) -> Path:
    """Write the identity tensors and projections of ``weights`` as an HF checkpoint."""
    from safetensors.numpy import save_file

    tensors = {
        "model.embedding_norm.weight": weights["final_norm"].astype(np.float16),
        "model.layers.0.operator_norm.weight": weights["layer.0.operator_norm"].astype(np.float16),
    }
    for key, value in weights.items():
        match = re.fullmatch(r"layer\.(\d+)\.(\w+)", str(key))
        if match and match.group(2) in _HF_STEMS:
            name = f"model.layers.{match.group(1)}.{_HF_STEMS[match.group(2)]}.weight"
            tensors[name] = np.ascontiguousarray(np.asarray(value).T, dtype=np.float16)
    directory.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(directory / "model.safetensors"))
    (directory / "config.json").write_text(json.dumps({**_raw_config(), **config_updates}))
    return directory


def _write_awq_scales(checkpoint: Path, scales_dir: Path, skip: str | None = None) -> None:
    """Write a smoothing-scale table keyed to ``checkpoint``'s identity, like the calibration tool."""
    gen = np.random.default_rng(0)
    table = {}
    for name, value in _weights().items():
        match = re.fullmatch(r"layer\.\d+\.(\w+)", str(name))
        if match and match.group(1) in _HF_STEMS and name != skip:
            table[name] = gen.uniform(0.5, 2.0, np.asarray(value).shape[0]).astype(np.float16)
    scales_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        scales_dir / "tiny.npz",
        hf_id=np.array("tiny/lfm2"),
        hf_revision=np.array("0" * 40),
        algorithm=np.array("test"),
        **awq_identity(_open_safetensors(checkpoint)),
        **table,
    )


def _dequantize_awq(weight: _AWQWeight) -> np.ndarray:
    """Unpack the INT4 codes and apply the block scales: the weight the engine multiplies by."""
    low = (weight.packed & 15).astype(np.int8)
    high = (weight.packed >> 4).astype(np.int8)
    codes = np.empty((weight.out_features, weight.in_features), dtype=np.int8)
    codes[:, 0::2] = np.where(low > 7, low - 16, low)
    codes[:, 1::2] = np.where(high > 7, high - 16, high)
    scale = np.repeat(weight.block_scale.astype(np.float32), 128, axis=1)
    return codes.astype(np.float32) * scale


def _awq_context(tmp_path: Path, **kwargs) -> Lfm2QuantContext:
    """Quantize the tiny model's weights with a scale table written for its checkpoint."""
    checkpoint = _write_checkpoint(tmp_path / "ckpt", _weights())
    _write_awq_scales(checkpoint, tmp_path / "scales")
    return calibrate_lfm2_awq(
        _resolved(), _weights(), checkpoint, scales_dir=tmp_path / "scales", **kwargs
    )


def test_quantize_awq_weight_reproduces_group_int4() -> None:
    """Codes round the smoothed weight with fp16 amax/7 block scales and stay in [-8, 7]."""
    gen = np.random.default_rng(1)
    weight = gen.standard_normal((8, 256)).astype(np.float32)
    weight[3, :128] = 0.0  # an all-zero block
    smooth = gen.uniform(0.5, 2.0, 256).astype(np.float16)
    quantized = quantize_awq_weight(weight, smooth)

    assert quantized.packed.shape == (8, 128)
    assert quantized.block_scale.shape == (8, 2)
    smoothed = weight / smooth.astype(np.float32)
    expected_scale = (np.abs(smoothed.reshape(8, 2, 128)).max(axis=2) / 7).astype(np.float16)
    expected_scale[3, 0] = 1.0
    assert np.array_equal(quantized.block_scale, expected_scale)

    restored = _dequantize_awq(quantized)
    step = np.repeat(expected_scale.astype(np.float32), 128, axis=1)
    # Round-to-nearest on the grid: at most half a step away.
    assert np.all(np.abs(restored - smoothed) <= 0.5 * step + 1e-6)
    assert np.all(restored[3, :128] == 0.0)


@pytest.mark.parametrize(
    ("in_features", "smooth", "message"),
    [(200, np.ones(200), "divisible by 128"), (128, np.zeros(128), "positive finite")],
)
def test_quantize_awq_weight_rejects_invalid_input(in_features, smooth, message) -> None:
    """Shapes that do not split into 128-wide blocks and non-positive scales are refused."""
    with pytest.raises(ValueError, match=message):
        quantize_awq_weight(np.ones((4, in_features), dtype=np.float32), smooth)


def test_awq_context_quantizes_the_loaded_weights(tmp_path: Path) -> None:
    """The FP8 projection set is quantized at level 3; the attention out_proj stays 16 bit."""
    ctx = _awq_context(tmp_path)

    # conv layers: conv_in, conv_out + 3 MLP; attention layers: q, k, v + 3 MLP.
    assert len(ctx.scales) == 2 * 5 + 2 * 6
    assert not ctx.should_quantize("layer.1.w_o")
    entry = ctx.scales["layer.1.w2"]
    assert entry.input_scale is None and isinstance(entry.weight, _AWQWeight)
    expected = quantize_awq_weight(_weights()["layer.1.w2"].T, entry.weight.pre_quant_scale)
    assert np.array_equal(entry.weight.packed, expected.packed)
    conv_in = ctx.scales["layer.0.conv_in"].weight
    assert (conv_in.out_features, conv_in.in_features) == (3 * HIDDEN, HIDDEN)
    assert ctx.builder_optimization_level == 3 and not ctx.has_fp8_weights


def test_awq_context_quantizes_only_the_selected_projections(tmp_path: Path) -> None:
    """``projections`` selects the quantized subset; out_proj can be added (the ablation path)."""
    ctx = _awq_context(tmp_path, projections=("w1", "w2", "w3"))
    assert sorted({name.split(".")[2] for name in ctx.scales}) == ["w1", "w2", "w3"]
    assert not ctx.should_quantize("layer.0.conv_in")
    with_out = _awq_context(tmp_path / "out", projections=quantization.AWQ_PROJECTIONS + ("w_o",))
    assert with_out.should_quantize("layer.1.w_o") and len(with_out.scales) == 2 * 5 + 2 * 7


def test_awq_build_options_keep_dual_gemm_fusion(tmp_path: Path) -> None:
    """INT4-AWQ builds at level 3 with DETAILED profiling and leaves the dual-GEMM route alone."""
    config = _FakeBuilderConfig('{"peep:match_dual_gemm": {}}')
    configure_quantized_build(config, _awq_context(tmp_path))
    assert config.builder_optimization_level == 3
    assert config.profiling_verbosity is not None
    assert config.build_route == ""


def test_awq_rejects_scales_of_another_checkpoint(tmp_path: Path) -> None:
    """A scale table calibrated for different weights is never applied."""
    _write_awq_scales(_write_checkpoint(tmp_path / "ckpt", _weights()), tmp_path / "scales")
    other = _weights()
    other["final_norm"] = other["final_norm"] * 2
    with pytest.raises(ValueError, match="no INT4-AWQ scales"):
        calibrate_lfm2_awq(
            _resolved(), other, _write_checkpoint(tmp_path / "other", other),
            scales_dir=tmp_path / "scales",
        )


def test_awq_rejects_checkpoint_with_changed_projection(tmp_path: Path) -> None:
    """Matching norms are not enough: a checkpoint with one changed projection is refused."""
    _write_awq_scales(_write_checkpoint(tmp_path / "ckpt", _weights()), tmp_path / "scales")
    tuned = _weights()
    tuned["layer.2.conv_out"] = tuned["layer.2.conv_out"] + 0.01
    with pytest.raises(ValueError, match="no INT4-AWQ scales"):
        calibrate_lfm2_awq(
            _resolved(), tuned, _write_checkpoint(tmp_path / "tuned", tuned),
            scales_dir=tmp_path / "scales",
        )


def test_awq_refuses_an_already_quantized_checkpoint(tmp_path: Path) -> None:
    """A prepared INT4 checkpoint fails with a format error before any scale lookup."""
    checkpoint = _write_checkpoint(
        tmp_path / "ckpt", _weights(), quantization_config={"quant_method": "awq", "bits": 4}
    )
    _write_awq_scales(checkpoint, tmp_path / "scales")
    with pytest.raises(ValueError, match="already quantized"):
        calibrate_lfm2_awq(_resolved(), _weights(), checkpoint, scales_dir=tmp_path / "scales")
    plain = _write_checkpoint(tmp_path / "plain", _weights())
    (plain / "hf_quant_config.json").write_text("{}")
    with pytest.raises(ValueError, match="already quantized"):
        calibrate_lfm2_awq(_resolved(), _weights(), plain, scales_dir=tmp_path / "scales")


def test_awq_rejects_missing_projection_scale(tmp_path: Path) -> None:
    """A projection without a calibrated scale fails the build instead of staying unquantized."""
    checkpoint = _write_checkpoint(tmp_path / "ckpt", _weights())
    _write_awq_scales(checkpoint, tmp_path / "scales", skip="layer.1.w3")
    with pytest.raises(ValueError, match="no layer.1.w3"):
        calibrate_lfm2_awq(_resolved(), _weights(), checkpoint, scales_dir=tmp_path / "scales")


def test_build_requires_committed_awq_scales(monkeypatch, tmp_path: Path) -> None:
    """quantization=int4_awq on a checkpoint without committed scales fails before the engine build."""
    from ..model import build

    checkpoint = _write_checkpoint(tmp_path / "ckpt", _weights())
    request = SimpleNamespace(
        dynamic_kv_cache=False,
        image_height=None,
        image_width=None,
        video_num_frames=None,
        max_batch_size=1,
        context_parallel_size=1,
        task="text_generation",
        model_dir=checkpoint,
        precision="fp16",
        max_sequence_length=64,
        tensor_parallel_size=1,
        quantization="int4_awq",
        fp32_layers=None,
        mtp_seq_len=None,
        backend="trt",
    )
    monkeypatch.setattr("families.lfm2.model.load_lfm2_weights", lambda *a, **k: _weights())
    with pytest.raises(ValueError, match="no INT4-AWQ scales"):
        build(request, writer=None)


def test_awq_calibration_tool_maps_projection_names() -> None:
    """The tool keys every block projection, out_proj included, by the graph's weight stem."""
    pattern = calibrate_awq_scales._PROJECTION_NAME
    for module, weight in (
        ("model.layers.3.conv.in_proj", "conv_in"),
        ("model.layers.3.self_attn.out_proj", "w_o"),
        ("model.layers.3.feed_forward.w2", "w2"),
    ):
        match = pattern.search(module)
        assert match is not None and match.group(1) == "3"
        assert calibrate_awq_scales._WEIGHT_STEM[match.group(2)] == weight
    assert pattern.search("model.layers.3.conv.conv") is None


def test_awq_calibration_tool_detects_block_amax_drift() -> None:
    """The tool accepts ModelOpt's plain block amax and rejects a clipped or mismatched one."""
    torch = pytest.importorskip("torch")
    gen = np.random.default_rng(3)
    weight = gen.standard_normal((4, 256)).astype(np.float32)
    smooth = gen.uniform(0.5, 2.0, 256).astype(np.float16)
    amax = np.abs((weight / smooth.astype(np.float32)).reshape(4, 2, 128)).max(axis=2)
    calibrate_awq_scales._check_block_amax("w", weight, smooth, torch.from_numpy(amax))
    with pytest.raises(RuntimeError, match="recomputed block amax differs"):
        calibrate_awq_scales._check_block_amax("w", weight, smooth, torch.from_numpy(amax * 0.9))


def test_awq_calibration_config_leaves_the_short_convolution_unquantized() -> None:
    """The ModelOpt recipe is INT4_AWQ_CFG plus a rule that skips the depthwise convolution."""
    pytest.importorskip("modelopt.torch.quantization")
    config = calibrate_awq_scales.awq_config()
    assert config["algorithm"]["method"] == "awq_lite"
    rules = config["quant_cfg"]
    if isinstance(rules, list):
        assert {"quantizer_name": "*conv.conv.*", "enable": False} in rules
    else:
        assert rules["*conv.conv.*"] == {"enable": False}


def test_committed_awq_scales_are_complete() -> None:
    """Every committed table names its checkpoint and covers each projection of its layers."""
    paths = sorted(quantization.AWQ_SCALES_DIR.glob("*.npz"))
    # Conv and attention layer counts of each table's checkpoint.
    expected_layers = {"lfm2-1.2b-40f3da0d.npz": (10, 6)}
    assert [path.name for path in paths] == sorted(expected_layers)
    conv_names = {"conv_in", "conv_out", "w1", "w2", "w3"}
    attention_names = {"w_q", "w_k", "w_v", "w_o", "w1", "w2", "w3"}
    for path in paths:
        with np.load(path, allow_pickle=False) as table:
            for key in ("hf_id", "hf_revision", "algorithm"):
                assert str(table[key]), (path.name, key)
            for key in quantization.AWQ_IDENTITY_TENSORS:
                assert f"identity.{key}" in table.files, (path.name, key)
            # The table identifies its checkpoint by two float64 sums per region, no weights.
            for name in table.files:
                if name.startswith("identity."):
                    assert table[name].dtype == np.float64 and table[name].shape == (2,), name
            num_conv, num_attention = expected_layers[path.name]
            layers = {int(name.split(".")[1]) for name in table.files if name.startswith("layer.")}
            assert layers == set(range(num_conv + num_attention)), path.name
            kinds = []
            for layer in sorted(layers):
                names = {
                    name.split(".", 2)[2] for name in table.files if name.startswith(f"layer.{layer}.")
                }
                assert names in (conv_names, attention_names), (path.name, layer)
                kinds.append(names == conv_names)
                for stem in names:
                    smooth = table[f"layer.{layer}.{stem}"]
                    assert smooth.dtype == np.float16 and smooth.ndim == 1
                    assert np.isfinite(smooth).all() and np.all(smooth > 0)
            assert (sum(kinds), len(kinds) - sum(kinds)) == (num_conv, num_attention), path.name


@pytest.mark.gpu
@pytest.mark.trt
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_awq_matmul_matches_dequantized_reference(dtype: str) -> None:
    """The INT4 constant, block Dequantize and smoothing compute the dequantized weight's matmul."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    gen = np.random.default_rng(2)
    rows, n, k = 5, 64, 256
    weight = quantize_awq_weight(
        gen.standard_normal((n, k)).astype(np.float32) * 0.05,
        gen.uniform(0.5, 2.0, k).astype(np.float16),
    )
    ctx = Lfm2QuantContext(scales={"w": _LayerScales(None, weight)})
    trt_dtype = {"fp16": trt.float16, "bf16": trt.bfloat16}[dtype]
    torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype]

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    x = network.add_input("x", trt_dtype, (rows, k))
    out = ctx.quantized_matmul(network, x, "w", name="w")
    out.name = "y"
    network.mark_output(out)
    plan = builder.build_serialized_network(network, builder.create_builder_config())
    assert plan is not None
    engine = trt.Runtime(logger).deserialize_cuda_engine(plan)
    context = engine.create_execution_context()

    activation = torch.from_numpy(gen.standard_normal((rows, k)).astype(np.float32))
    activation = activation.to(torch_dtype).cuda()
    result = torch.empty((rows, n), dtype=torch_dtype, device="cuda")
    context.set_tensor_address("x", activation.data_ptr())
    context.set_tensor_address("y", result.data_ptr())
    stream = torch.cuda.current_stream()
    assert context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()

    smoothed = activation.float().cpu().numpy() * weight.pre_quant_scale.astype(np.float32)
    expected = smoothed @ _dequantize_awq(weight).T
    tolerance = {"fp16": 2e-2, "bf16": 1e-1}[dtype]
    np.testing.assert_allclose(result.float().cpu().numpy(), expected, rtol=0, atol=tolerance)


@pytest.mark.gpu
@pytest.mark.trt
@pytest.mark.parametrize("precision", ["fp16", "bf16"])
def test_engine_builds_with_awq_weights(tmp_path: Path, precision: str) -> None:
    """The INT4-AWQ hybrid engine builds, keeps its I/O contract and stores 4-bit weights."""
    ctx = _awq_context(tmp_path)
    plain = build_lfm2_engine(_config(), _weights(), 64, precision=precision)
    quantized = build_lfm2_engine(_config(), _weights(), 64, precision=precision, quant_ctx=ctx)

    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(quantized)
    assert engine is not None
    names = {engine.get_tensor_name(i) for i in range(engine.num_io_tensors)}
    assert {"token_id", "logits", "present_k_0", "present_conv_0", "conv_state_0"} <= names
    # INT4 quarters the fp16 projection bytes; level 3 adds kernels, so the bound is loose.
    assert len(quantized) < 0.6 * len(plain), (len(quantized), len(plain))
