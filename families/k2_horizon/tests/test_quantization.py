# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""K2-Horizon FP8 self-quantization: weight quantizer, scale lookup, context build, engine build."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import ml_dtypes
import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")

from .. import calibrate_fp8_scales, model, quantization  # noqa: E402
from ..checkpoint_mapper import WeightDict  # noqa: E402
from ..quantization import (  # noqa: E402
    K2HorizonQuantContext,
    calibrate_k2_horizon_fp8,
    configure_fp8_build,
    load_activation_scales,
    projection_names,
    quantize_fp8_weight,
    scale_key,
)

# Wide enough that FP8 weight savings dominate the per-layer Q/DQ plan overhead.
HIDDEN, MLP, VOCAB, LAYERS, HEADS, KV_HEADS, HEAD_DIM = 512, 2048, 64, 2, 4, 2, 128

_SHAPES = {
    "w_q": (HIDDEN, HEADS * HEAD_DIM),
    "w_k": (HIDDEN, KV_HEADS * HEAD_DIM),
    "w_v": (HIDDEN, KV_HEADS * HEAD_DIM),
    "w_o": (HEADS * HEAD_DIM, HIDDEN),
    "w_gate": (HIDDEN, MLP),
    "w_up": (HIDDEN, MLP),
    "w_down": (MLP, HIDDEN),
}
_WEIGHT_STEMS = dict(quantization.PROJECTIONS)
_SHIPPED_KEY = "k2_horizon-h4096-l36-i12288-n"


def _config(**overrides) -> SimpleNamespace:
    """Build a small K2-Horizon model config for the tests."""
    raw = {
        "model_type": "k2_horizon",
        "architectures": ["K2HorizonForCausalLM"],
        "vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "intermediate_size": MLP,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10_000_000.0,
        "max_position_embeddings": 4096,
        "hidden_act": "silu",
        "layernorm_num_groups": 4,
        "tie_word_embeddings": False,
        "bos_token_id": 0,
        "eos_token_id": 1,
    }
    raw.update(overrides)
    return SimpleNamespace(raw=raw, **raw)


def _bf16(array: np.ndarray) -> np.ndarray:
    """Return BF16 bit patterns (``uint16``), the storage the checkpoint mapper produces."""
    return np.ascontiguousarray(array.astype(ml_dtypes.bfloat16).view(np.uint16))


def _weights() -> WeightDict:
    """Return random weights in the loaded ``[in, out]`` BF16 layout of the tiny model."""
    rng = np.random.RandomState(0)
    weights = WeightDict()
    weights["embedding"] = _bf16(rng.randn(VOCAB, HIDDEN) * 0.1)
    for layer in range(LAYERS):
        prefix = f"layer.{layer}"
        for norm in ("input_norm", "post_attn_norm"):
            weights[f"{prefix}.{norm}"] = np.ones(HIDDEN, dtype=np.float32)
        for stem, shape in _SHAPES.items():
            weights[f"{prefix}.{stem}"] = _bf16(rng.randn(*shape) * 0.05)
    weights["final_norm"] = np.ones(HIDDEN, dtype=np.float32)
    weights["w_out"] = _bf16(rng.randn(HIDDEN, VOCAB) * 0.05)
    weights["_attention_size"] = HEADS * HEAD_DIM
    weights["_kv_attention_size"] = KV_HEADS * HEAD_DIM
    weights["_mlp_size"] = MLP
    return weights


def _activation_scales(skip: tuple[str, ...] = ()) -> dict[str, float | None]:
    """Return one activation scale per projection; skipped stems are ``None`` (left in bf16)."""
    return {
        f"layer.{layer}.{stem}": (None if stem in skip else 0.02)
        for layer in range(LAYERS)
        for stem in _WEIGHT_STEMS.values()
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


def test_scale_key_identifies_model_config_and_weights() -> None:
    """The scale key names the model config and carries the final-norm weight sum."""
    ones = np.ones(HIDDEN, dtype=np.float32)
    assert scale_key(_config(), ones) == f"k2_horizon-h{HIDDEN}-l{LAYERS}-i{MLP}-n{HIDDEN}.0000"
    # Two checkpoints with one config (for example a fine-tune) must not share scales.
    assert scale_key(_config(), ones) != scale_key(_config(), ones * 1.01)


def test_missing_scale_entry_is_refused() -> None:
    """A model config without committed scales fails with a clear error naming the tool."""
    with pytest.raises(NotImplementedError, match="calibrate_fp8_scales.py"):
        load_activation_scales(_config(), np.ones(HIDDEN, dtype=np.float32))


def _committed_scales() -> dict:
    """Return the committed activation-scale table."""
    path = Path(quantization.__file__).parent / quantization.ACTIVATION_SCALES_FILENAME
    return json.loads(path.read_text(encoding="utf-8"))


def test_committed_scales_are_the_7b_checkpoint() -> None:
    """Scales ship for K2-Horizon-7B only."""
    keys = sorted(_committed_scales())
    assert len(keys) == 1 and keys[0].startswith(_SHIPPED_KEY)


def test_committed_scales_cover_every_projection() -> None:
    """Every committed model config has a scale or an explicit null for every projection."""
    for key, scales in _committed_scales().items():
        assert all(value is None or value > 0 for value in scales.values()), key
        layers = int(key.split("-l", 1)[1].split("-", 1)[0])
        expected = {
            f"layer.{layer}.{stem}" for layer in range(layers) for stem in _WEIGHT_STEMS.values()
        }
        assert set(scales) == expected, key
        assert any(value is not None for value in scales.values()), key


def test_committed_recipe_keeps_the_first_two_layers_in_bf16() -> None:
    """The shipped table leaves layers 0 and 1 and one massive-activation projection unquantized."""
    for scales in _committed_scales().values():
        nulls = {name for name, value in scales.items() if value is None}
        leading = {f"layer.{layer}.{stem}" for layer in (0, 1) for stem in _WEIGHT_STEMS.values()}
        assert nulls == leading | {"layer.35.w_down"}


def test_projection_names_cover_seven_projections_per_layer() -> None:
    """The recipe quantizes q, k, v, o, gate, up and down of every layer."""
    names = projection_names(_config())
    assert len(names) == LAYERS * 7
    assert any(name.endswith("w_o") for name in names)


def test_context_quantizes_the_loaded_bf16_weights(monkeypatch) -> None:
    """The context quantizes the transposed BF16 weights and keeps the committed scales."""
    weights = _weights()
    monkeypatch.setattr(
        quantization, "load_activation_scales", lambda config, final_norm: _activation_scales()
    )
    ctx = calibrate_k2_horizon_fp8(_config(), weights)

    assert len(ctx.scales) == LAYERS * len(_WEIGHT_STEMS)
    entry = ctx.scales["layer.1.w_down"]
    in_f, out_f = _SHAPES["w_down"]
    assert (entry.weight.out_features, entry.weight.in_features) == (out_f, in_f)
    assert entry.input_scale == pytest.approx(0.02)

    source = weights["layer.1.w_down"].view(ml_dtypes.bfloat16).astype(np.float32).T
    packed, scale = quantize_fp8_weight(np.ascontiguousarray(source))
    assert np.array_equal(entry.weight.packed, packed)
    assert entry.weight.weight_scale == pytest.approx(scale)
    assert ctx.should_quantize("layer.0.w_o")
    assert not ctx.should_quantize("w_out")


def test_projection_with_a_null_scale_stays_unquantized(monkeypatch) -> None:
    """A projection stored as null (massive activation) is left in bf16 on purpose."""
    monkeypatch.setattr(
        quantization,
        "load_activation_scales",
        lambda config, final_norm: _activation_scales(("w_down",)),
    )
    ctx = calibrate_k2_horizon_fp8(_config(), _weights())
    assert not ctx.should_quantize("layer.0.w_down")
    assert ctx.should_quantize("layer.0.w_up")


def test_projection_missing_from_the_scale_file_is_an_error(monkeypatch) -> None:
    """A projection with no entry at all is an error, never a silent bf16 fallback."""
    scales = _activation_scales()
    del scales["layer.1.w_up"]
    monkeypatch.setattr(quantization, "load_activation_scales", lambda config, final_norm: scales)
    with pytest.raises(ValueError, match="layer.1.w_up has no activation scale"):
        calibrate_k2_horizon_fp8(_config(), _weights())


def test_scale_for_a_projection_that_is_not_quantized_is_an_error(monkeypatch) -> None:
    """A stored scale for a tensor that is not a quantized projection does not match this recipe."""
    scales = {**_activation_scales(), "w_out": 0.02}
    monkeypatch.setattr(quantization, "load_activation_scales", lambda config, final_norm: scales)
    with pytest.raises(ValueError, match="not quantized"):
        calibrate_k2_horizon_fp8(_config(), _weights())


def test_already_quantized_checkpoint_is_refused(monkeypatch) -> None:
    """A checkpoint that carries FP8 scales is refused instead of quantized twice."""
    monkeypatch.setattr(
        quantization, "load_activation_scales", lambda config, final_norm: _activation_scales()
    )
    weights = _weights()
    weights["layer.0.q_proj.weight_scale"] = np.ones(1, dtype=np.float32)
    with pytest.raises(ValueError, match="already quantized"):
        calibrate_k2_horizon_fp8(_config(), weights)
    with pytest.raises(ValueError, match="already quantized"):
        calibrate_k2_horizon_fp8(
            _config(quantization_config={"quant_method": "compressed-tensors"}), _weights()
        )


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
    ctx = K2HorizonQuantContext(scales={})
    config = _FakeBuilderConfig(routes)
    configure_fp8_build(config, ctx)
    assert config.build_route == expected
    assert config.profiling_verbosity is not None
    unquantized = _FakeBuilderConfig(routes)
    configure_fp8_build(unquantized, None)
    assert unquantized.build_route == "" and unquantized.profiling_verbosity is None


class _Writer:
    """Collects the bundle sections written by ``model.build``."""

    def __init__(self) -> None:
        self.header = None
        self.sections: dict = {}

    def set_header(self, **value) -> None:
        self.header = value

    def add_bytes(self, name, value) -> None:
        self.sections[name] = value

    def add_json(self, name, value) -> None:
        self.sections[name] = value


def _request(tmp_path: Path, quantization_value) -> SimpleNamespace:
    """Return a build request for the tiny model."""
    return SimpleNamespace(
        model_dir=tmp_path,
        backend="trt",
        dynamic_kv_cache=False,
        task="text_generation",
        precision="bf16",
        max_sequence_length=8,
        image_height=None,
        image_width=None,
        video_num_frames=None,
        max_batch_size=1,
        tensor_parallel_size=1,
        context_parallel_size=1,
        quantization=quantization_value,
        fp32_layers=(),
        mtp_seq_len=None,
        verbose=False,
    )


def test_build_passes_the_fp8_context_and_keeps_the_runtime_contract(
    monkeypatch, tmp_path: Path
) -> None:
    """``quantization=fp8`` builds the context; runtime.json is the one of the BF16 bundle."""
    (tmp_path / "config.json").write_text(json.dumps(_config().raw), encoding="utf-8")
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(model, "load_standard_weights", lambda *_args, **_kwargs: _weights())
    monkeypatch.setattr(
        quantization, "load_activation_scales", lambda config, final_norm: _activation_scales()
    )
    seen = {}

    def fake_build_engine(*_args, quant_ctx=None, **_kwargs):
        seen.setdefault("ctx", []).append(quant_ctx)
        return b"plan"

    monkeypatch.setattr(model, "build_engine", fake_build_engine)
    plain, quantized = _Writer(), _Writer()
    model.build(_request(tmp_path, None), plain)
    model.build(_request(tmp_path, "fp8"), quantized)

    assert seen["ctx"][0] is None
    assert len(seen["ctx"][1].scales) == LAYERS * 7
    assert quantized.sections["runtime.json"] == plain.sections["runtime.json"]


@pytest.mark.parametrize("value", ["nvfp4", "int8"])
def test_other_quantization_formats_are_refused(tmp_path: Path, value: str) -> None:
    """Only FP8 is implemented."""
    with pytest.raises(NotImplementedError, match="does not support quantization"):
        model.build(_request(tmp_path, value), _Writer())


@pytest.mark.gpu
@pytest.mark.trt
def test_engine_builds_with_fp8_weights(monkeypatch) -> None:
    """The quantized engine builds, keeps the engine I/O and stores FP8 (not BF16) weights."""
    config = _config()
    monkeypatch.setattr(
        quantization, "load_activation_scales", lambda config, final_norm: _activation_scales()
    )
    ctx = calibrate_k2_horizon_fp8(config, _weights())

    plain = model.build_engine(config, _weights(), 8)
    quantized = model.build_engine(config, _weights(), 8, quant_ctx=ctx)

    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engines = [runtime.deserialize_cuda_engine(plan) for plan in (plain, quantized)]
    assert all(engine is not None for engine in engines)
    names = [
        {engine.get_tensor_name(i) for i in range(engine.num_io_tensors)} for engine in engines
    ]
    assert names[0] == names[1]
    assert {"token_id", "logits", "cache_k_0", "present_v_1"} <= names[1]
    # 7 projections x 2 layers: FP8 halves the BF16 projection bytes.
    assert len(quantized) < len(plain)


def test_calibration_tool_maps_module_names_to_weight_stems() -> None:
    """The calibration tool maps module names to internal weight names."""
    match = calibrate_fp8_scales._PROJECTION_NAME.search("model.layers.3.mlp.down_proj")
    assert match is not None and match.group(1) == "3"
    assert calibrate_fp8_scales._WEIGHT_STEM[match.group(2)] == "w_down"
    assert calibrate_fp8_scales._PROJECTION_NAME.search("model.layers.3.mlp.down_proj.base") is None
    out = calibrate_fp8_scales._PROJECTION_NAME.search("model.layers.3.self_attn.o_proj")
    assert out is not None and calibrate_fp8_scales._WEIGHT_STEM[out.group(2)] == "w_o"
    assert calibrate_fp8_scales._PROJECTION_NAME.search("lm_head") is None


def test_calibration_tool_splits_outlier_scales() -> None:
    """A scale more than 20 times the median of its projection is left out."""
    scales = {f"layer.{i}.w_down": 0.01 for i in range(7)} | {"layer.7.w_down": 0.5}
    scales |= {f"layer.{i}.w_up": 0.01 for i in range(8)}
    kept, excluded = calibrate_fp8_scales.split_outlier_scales(scales)
    assert excluded == ["layer.7.w_down"]
    assert "layer.7.w_down" not in kept and len(kept) == 15


def test_recipe_scales_nulls_the_leading_layers_and_the_outliers() -> None:
    """The stored table has null for layers 0 and 1 and for outliers, scales elsewhere."""
    scales = {f"layer.{i}.{stem}": 0.01 for i in range(6) for stem in ("w_q", "w_down")}
    scales["layer.4.w_down"] = 0.5
    table, excluded = calibrate_fp8_scales.recipe_scales(scales)
    assert excluded == ["layer.4.w_down"]
    assert list(table) == sorted(scales)
    assert {name for name, value in table.items() if value is None} == {
        "layer.0.w_q",
        "layer.0.w_down",
        "layer.1.w_q",
        "layer.1.w_down",
        "layer.4.w_down",
    }
    assert table["layer.2.w_q"] == 0.01


def test_calibration_tool_cli_parses_both_commands(monkeypatch, tmp_path: Path) -> None:
    """The calibration tool command line accepts fetch and calibrate."""
    seen = {}
    monkeypatch.setattr(
        calibrate_fp8_scales,
        "fetch_prompts",
        lambda out, n, news, start: seen.update(fetch=(out, n, news, start)),
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
    assert seen["fetch"] == (tmp_path / "p.jsonl", 96, 128, 0)
    assert seen["calibrate"] == (tmp_path, Path("p.jsonl"), 1024)


def test_calibration_rows_are_picked_with_a_stride() -> None:
    """Nemotron rows are taken every few rows and short answers are skipped."""
    rows = [
        {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a" * n}]}
        for n in (300, 10, 300, 300) * 40
    ]
    picked = calibrate_fp8_scales._nemotron_rows(rows, "math", 3, 0)
    assert len(picked) == 3 and all(row["split"] == "math" for row in picked)
    assert all(len(row["messages"][1]["content"]) >= 200 for row in picked)


def test_render_prompt_keeps_the_answer_as_plain_text() -> None:
    """The user turn goes through the template; the answer and raw text follow BOS as plain text."""

    class _Tokenizer:
        bos_token = "<s>"
        chat_template = "template"

        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            assert not tokenize and add_generation_prompt
            return "<s>[user]" + messages[0]["content"] + "[assistant]"

    tokenizer = _Tokenizer()
    chat = {
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "<think>r</think>a"},
        ]
    }
    assert (
        calibrate_fp8_scales.render_prompt(tokenizer, chat)
        == "<s>[user]q[assistant]<think>r</think>a"
    )
    no_user = {
        "messages": [{"role": "user", "content": "-"}, {"role": "assistant", "content": "a"}]
    }
    assert calibrate_fp8_scales.render_prompt(tokenizer, no_user) == "<s>a"
    assert calibrate_fp8_scales.render_prompt(tokenizer, {"text": "news"}) == "<s>news"
