# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-OCR FP8 quantization: weight scaling, projection coverage, calibration tool, engine build."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import ml_dtypes
import numpy as np
import pytest

torch = pytest.importorskip("torch")
trt = pytest.importorskip("tensorrt")
safetensors_numpy = pytest.importorskip("safetensors.numpy")

from families.deepseek_ocr import calibrate_fp8_scales, graph_ops, model  # noqa: E402
from families.deepseek_ocr import quantization as quantization_module  # noqa: E402
from families.deepseek_ocr.config import ModelConfig  # noqa: E402
from families.deepseek_ocr.quantization import (  # noqa: E402
    ACTIVATION_SCALES_FILENAME,
    ATTENTION_PROJECTIONS,
    DENSE_MLP_PROJECTIONS,
    SHARED_EXPERT_PROJECTIONS,
    DeepseekOcrQuantContext,
    calibrate_deepseek_ocr,
    configure_fp8_build,
    dequantize_fp8_weight,
    quantize_fp8_weight,
    scale_key,
)

# A tiny DeepSeek-OCR language model: layer 0 is dense, layers 1 and 2 are MoE with shared experts.
D, HEADS, LAYERS, VOCAB = (
    256,
    2,
    3,
    100,
)  # head size 128 like the real model (TensorRT 11.1 can crash on small heads)
EXPERTS, TOP_K, MOE_FFN, SHARED_FFN, DENSE_FFN = 4, 2, 64, 128, 192
ATTN = D  # heads * head_dim, no GQA


def _config_json() -> dict:
    fields = {
        "hidden_size": D,
        "intermediate_size": DENSE_FFN,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "num_key_value_heads": HEADS,
        "vocab_size": VOCAB,
        "max_position_embeddings": 128,
        "first_k_dense_replace": 1,
        "moe_intermediate_size": MOE_FFN,
        "n_routed_experts": EXPERTS,
        "n_shared_experts": 2,
        "num_experts_per_tok": TOP_K,
        "bos_token_id": 0,
        "eos_token_id": 1,
    }
    return {"model_type": "deepseek_vl_v2", "architectures": ["DeepseekOCR2ForCausalLM"], **fields}


def _write_checkpoint(path: Path) -> ModelConfig:
    """Write a random tiny checkpoint (HF tensor names) and return its config."""
    rng = np.random.default_rng(0)

    def w(*shape):
        return rng.standard_normal(shape).astype(np.float32) * 0.05

    t: dict[str, np.ndarray] = {
        "model.embed_tokens.weight": w(VOCAB, D),
        "lm_head.weight": w(VOCAB, D),
        "model.norm.weight": np.ones(D, np.float32),
    }
    for i in range(LAYERS):
        layer = f"model.layers.{i}"
        t[f"{layer}.input_layernorm.weight"] = np.ones(D, np.float32)
        t[f"{layer}.post_attention_layernorm.weight"] = np.ones(D, np.float32)
        for stem in ("q_proj", "k_proj", "v_proj", "o_proj"):
            t[f"{layer}.self_attn.{stem}.weight"] = w(ATTN, D)
        if i == 0:
            t[f"{layer}.mlp.gate_proj.weight"] = w(DENSE_FFN, D)
            t[f"{layer}.mlp.up_proj.weight"] = w(DENSE_FFN, D)
            t[f"{layer}.mlp.down_proj.weight"] = w(D, DENSE_FFN)
            continue
        t[f"{layer}.mlp.gate.weight"] = w(EXPERTS, D)
        for e in range(EXPERTS):
            t[f"{layer}.mlp.experts.{e}.gate_proj.weight"] = w(MOE_FFN, D)
            t[f"{layer}.mlp.experts.{e}.up_proj.weight"] = w(MOE_FFN, D)
            t[f"{layer}.mlp.experts.{e}.down_proj.weight"] = w(D, MOE_FFN)
        t[f"{layer}.mlp.shared_experts.gate_proj.weight"] = w(SHARED_FFN, D)
        t[f"{layer}.mlp.shared_experts.up_proj.weight"] = w(SHARED_FFN, D)
        t[f"{layer}.mlp.shared_experts.down_proj.weight"] = w(D, SHARED_FFN)
    safetensors_numpy.save_file(t, str(path / "model.safetensors"))
    (path / "config.json").write_text(json.dumps(_config_json()), encoding="utf-8")
    return ModelConfig.from_dir(path)


def _all_scales(config: ModelConfig, value: float = 0.02) -> dict[str, float]:
    scales = {}
    for layer in range(config.num_hidden_layers):
        mlp = DENSE_MLP_PROJECTIONS if layer < 1 else SHARED_EXPERT_PROJECTIONS
        for _, stem in (*ATTENTION_PROJECTIONS, *mlp):
            scales[f"layer.{layer}.{stem}"] = value
    return scales


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
    decoded = dequantize_fp8_weight(packed, scale).numpy()
    assert np.abs(decoded - weight).max() <= 0.07 * float(np.abs(weight).max())


def test_fp8_rounding_matches_ml_dtypes() -> None:
    """The torch cast rounds exactly like the reference e4m3 implementation."""
    weight = np.random.default_rng(2).standard_normal((16, 64)).astype(np.float32)
    packed, scale = quantize_fp8_weight(weight)
    expected = np.clip(weight / scale, -448.0, 448.0).astype(ml_dtypes.float8_e4m3fn).view(np.uint8)
    np.testing.assert_array_equal(packed, expected)


def test_fp8_weight_rejects_all_zero() -> None:
    """A weight with no positive amax cannot be scaled."""
    with pytest.raises(ValueError, match="amax"):
        quantize_fp8_weight(np.zeros((4, 4), dtype=np.float32))


def test_dense_projections_and_experts_are_quantized_except_attention_output(tiny) -> None:
    """q/k/v, the dense MLP, the shared experts and every routed expert get FP8 weights."""
    path, config = tiny
    ctx = calibrate_deepseek_ocr(path, graph_ops, "fp8", config)
    # layer 0: q, k, v, gate, up, down; layers 1 and 2: q, k, v, shared gate, up, down
    assert len(ctx.scales) == LAYERS * 6
    assert ctx.should_quantize("layer.0.w_q")
    assert ctx.should_quantize("layer.0.w_down")
    assert ctx.should_quantize("layer.2.shared.w_up")
    assert ctx.should_quantize("layer.1.experts.w_gate")
    assert not ctx.should_quantize("layer.0.experts.w_gate")  # layer 0 is dense
    # TensorRT 11.4 cannot compile an FP8 Quantize right after the fused attention
    assert not ctx.should_quantize("layer.0.w_o")
    entry = ctx.scales["layer.1.shared.w_gate"]
    assert (entry.weight.out_features, entry.weight.in_features) == (SHARED_FFN, D)
    assert entry.weight.packed.dtype == np.uint8
    assert set(ctx.experts) == {"layer.1", "layer.2"}


def test_expert_stacks_share_one_scale_and_use_the_engine_layout(tiny) -> None:
    """Experts are stored as [experts, in, out] bytes with the layer-wide amax / 448 scale."""
    path, config = tiny
    ctx = calibrate_deepseek_ocr(path, graph_ops, "fp8", config)
    tensors = safetensors_numpy.load_file(str(path / "model.safetensors"))
    stack = ctx.experts["layer.1"]
    assert stack.gate.shape == (EXPERTS, D, MOE_FFN)
    assert stack.up.shape == (EXPERTS, D, MOE_FFN)
    assert stack.down.shape == (EXPERTS, MOE_FFN, D)
    gate = np.stack(
        [tensors[f"model.layers.1.mlp.experts.{e}.gate_proj.weight"] for e in range(EXPERTS)]
    )  # [E, out, in]
    assert stack.gate.weight_scale == pytest.approx(float(np.abs(gate).max()) / 448.0)
    decoded = dequantize_fp8_weight(stack.gate.packed, stack.gate.weight_scale).numpy()
    np.testing.assert_allclose(
        decoded, gate.transpose(0, 2, 1), atol=0.07 * float(np.abs(gate).max())
    )


def test_projection_without_scale_stays_unquantized(tmp_path: Path, monkeypatch) -> None:
    """A projection with no activation scale is left in fp16."""
    config = _write_checkpoint(tmp_path)
    scales = _all_scales(config)
    del scales["layer.0.w_down"]
    monkeypatch.setattr(quantization_module, "load_activation_scales", lambda cfg: scales)
    ctx = calibrate_deepseek_ocr(tmp_path, graph_ops, "fp8", config)
    assert not ctx.should_quantize("layer.0.w_down")
    assert ctx.should_quantize("layer.0.w_up")


def test_already_quantized_checkpoint_is_rejected(tiny) -> None:
    """A checkpoint that already stores weight scales is not quantized a second time."""
    path, config = tiny
    tensors = safetensors_numpy.load_file(str(path / "model.safetensors"))
    tensors["model.layers.1.mlp.experts.0.up_proj.weight_scale"] = np.ones((1,), np.float32)
    safetensors_numpy.save_file(tensors, str(path / "model.safetensors"))
    with pytest.raises(ValueError, match="already quantized"):
        calibrate_deepseek_ocr(path, graph_ops, "fp8", config)


def test_quantization_config_in_checkpoint_is_rejected(tiny) -> None:
    """A config.json with a quantization_config is a quantized checkpoint (for example FP8)."""
    path, config = tiny
    config.raw["quantization_config"] = {"quant_method": "compressed-tensors"}
    with pytest.raises(ValueError, match="quantization_config"):
        calibrate_deepseek_ocr(path, graph_ops, "fp8", config)


def test_fp8_checkpoint_weights_are_rejected(tiny) -> None:
    """Weights that are already stored as one byte per element are not re-quantized."""
    path, config = tiny
    tensors = safetensors_numpy.load_file(str(path / "model.safetensors"))
    packed = tensors["model.layers.0.self_attn.q_proj.weight"].astype(ml_dtypes.float8_e4m3fn)
    tensors["model.layers.0.self_attn.q_proj.weight"] = packed.view(np.uint8)
    safetensors_numpy.save_file(tensors, str(path / "model.safetensors"))
    with pytest.raises(ValueError, match="original"):
        calibrate_deepseek_ocr(path, graph_ops, "fp8", config)


def test_unsupported_format_is_rejected(tiny) -> None:
    """Only fp8 is implemented."""
    path, config = tiny
    with pytest.raises(NotImplementedError, match="nvfp4"):
        calibrate_deepseek_ocr(path, graph_ops, "nvfp4", config)


def test_unknown_model_shape_is_reported(tmp_path: Path) -> None:
    """A model shape with no committed scales raises and names the calibration tool."""
    config = _write_checkpoint(tmp_path)
    with pytest.raises(NotImplementedError, match="calibrate_fp8_scales"):
        calibrate_deepseek_ocr(tmp_path, graph_ops, "fp8", config)


def test_committed_scales_cover_every_projection() -> None:
    """Every committed model shape has a scale for each quantized projection of every layer."""
    path = Path(quantization_module.__file__).parent / ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8"))
    assert table, "no committed activation scales"
    for key, scales in table.items():
        match = re.fullmatch(r"deepseek_vl_v2-h(\d+)-l(\d+)-i(\d+)", key)
        assert match, key
        hidden, layers, inter = (int(g) for g in match.groups())
        config = ModelConfig.from_json(
            json.dumps(
                {
                    "model_type": "deepseek_vl_v2",
                    "hidden_size": hidden,
                    "num_hidden_layers": layers,
                    "intermediate_size": inter,
                }
            )
        )
        assert scale_key(config) == key
        expected = set()
        for layer in range(layers):
            mlp = DENSE_MLP_PROJECTIONS if layer < 1 else SHARED_EXPERT_PROJECTIONS
            expected |= {f"layer.{layer}.{stem}" for _, stem in (*ATTENTION_PROJECTIONS, *mlp)}
        assert set(scales) == expected, key
        assert all(value > 0 for value in scales.values()), key


def test_load_weights_skips_quantized_projections(tiny, monkeypatch) -> None:
    """Quantized projections and expert stacks are never read; the rest loads as before."""
    path, config = tiny
    ctx = calibrate_deepseek_ocr(path, graph_ops, "fp8", config)
    loaded: list[str] = []
    original = model._load_tensor

    def recording_load(readers, name):
        loaded.append(name)
        return original(readers, name)

    monkeypatch.setattr(model, "_load_tensor", recording_load)
    weights = model._DeepseekOcrModel().load_weights(
        str(path), config, precision="fp16", quant_ctx=ctx
    )

    placeholder = weights["layer.0.w_q"]
    assert placeholder.shape == (D, ATTN)
    assert placeholder.strides == (0, 0)  # broadcast zero, no storage
    assert weights["layer.0.w_o"].strides != (0, 0)  # the attention output stays loaded
    assert weights["layer.0.w_gate"].shape == (D, DENSE_FFN)
    assert weights["layer.1.shared.w_gate"].shape == (D, SHARED_FFN)
    assert weights["layer.1.experts.w_gate"].shape == (EXPERTS, D, MOE_FFN)
    assert weights["layer.1.experts.w_gate"].strides == (0, 0, 0)
    assert weights["layer.1.router"].dtype == np.float16
    assert weights["w_out"].shape == (D, VOCAB)
    assert weights["_attention_size"] == ATTN
    assert not any(
        re.search(r"(q_proj|k_proj|v_proj|gate_proj|up_proj|down_proj)\.weight$", name)
        for name in loaded
    )
    assert "model.layers.0.self_attn.o_proj.weight" in loaded
    assert "model.layers.1.mlp.gate.weight" in loaded


def test_unquantized_load_still_reads_every_projection(tiny, monkeypatch) -> None:
    """Without a quantization context the loader reads all projections and the real experts."""
    path, config = tiny
    weights = model._DeepseekOcrModel().load_weights(str(path), config, precision="fp16")
    assert weights["layer.0.w_q"].strides != (0, 0)
    assert weights["layer.1.experts.w_gate"].shape == (EXPERTS, D, MOE_FFN)
    assert weights["layer.1.experts.w_gate"].strides != (0, 0, 0)


def test_calibration_tool_maps_module_names_to_weight_names() -> None:
    """The calibration tool maps decoder projections only; vision, experts and router are skipped."""
    name = calibrate_fp8_scales.weight_name
    assert name("model.layers.4.self_attn.q_proj") == "layer.4.w_q"
    assert name("model.layers.0.mlp.gate_proj") == "layer.0.w_gate"
    assert name("model.layers.7.mlp.shared_experts.down_proj") == "layer.7.shared.w_down"
    assert name("language.model.layers.2.self_attn.v_proj") == "layer.2.w_v"
    assert name("model.layers.4.self_attn.o_proj") is None  # stays fp16
    assert name("model.layers.4.mlp.experts.3.gate_proj") is None  # weight-only
    assert name("model.layers.4.mlp.gate") is None  # router
    assert name("model.qwen2_model.model.model.layers.0.self_attn.q_proj") is None  # vision
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
    assert seen["calibrate"] == (tmp_path, Path("imgs"), 128, 128)


def _request(tmp_path: Path, **overrides) -> SimpleNamespace:
    values = {
        "model_dir": tmp_path,
        "backend": "trt",
        "dynamic_kv_cache": False,
        "mtp_seq_len": None,
        "task": "vision_language_generation",
        "precision": "fp16",
        "max_sequence_length": 64,
        "tensor_parallel_size": 1,
        "context_parallel_size": 1,
        "quantization": "fp8",
        "fp32_layers": (),
        "image_height": None,
        "image_width": None,
        "video_num_frames": None,
        "max_batch_size": 1,
        "verbose": False,
    }
    return SimpleNamespace(**{**values, **overrides})


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"quantization": "nvfp4"}, "does not support quantization"),
        ({"precision": "fp32"}, "requires precision fp16"),
        ({"tensor_parallel_size": 2}, "tensor parallelism"),
    ],
)
def test_build_rejects_unsupported_quantized_requests(tmp_path: Path, overrides, message) -> None:
    """Only fp16, single-device fp8 builds are accepted."""
    with pytest.raises(NotImplementedError, match=message):
        model.build(_request(tmp_path, **overrides), writer=None)


def test_build_passes_the_context_to_load_weights_and_both_decoder_engines(
    tiny, monkeypatch
) -> None:
    """A quantized build threads one context through the weight loader and every decoder engine."""
    path, config = tiny
    seen = {"engines": []}

    class FakeModel:
        def load_weights(self, _model_dir, _config, *, precision, quant_ctx=None):
            seen["load"] = (precision, quant_ctx)
            return {"weights": True}

        def build_engine(self, _config, _weights, _length, **options):
            seen["engines"].append(options["quant_ctx"])
            return b"plan"

        def build_vision_engine(self, *_args, **_options):
            return b"vision"

        def get_vl_config(self, _config):
            return {"image_token_id": 7}

    class Writer:
        def set_header(self, **_values):
            pass

        def add_bytes(self, *_args):
            pass

        def add_json(self, *_args):
            pass

    monkeypatch.setattr(model, "_DeepseekOcrModel", FakeModel)
    model.build(_request(path), Writer())

    precision, ctx = seen["load"]
    assert precision == "fp16" and isinstance(ctx, DeepseekOcrQuantContext)
    assert seen["engines"] == [ctx, ctx]  # prefill and decode


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
    ctx = DeepseekOcrQuantContext(scales={}, experts={}, graph_ops=graph_ops)
    config = _FakeBuilderConfig(routes)
    configure_fp8_build(config, ctx)
    assert config.build_route == expected
    assert config.profiling_verbosity is not None
    unquantized = _FakeBuilderConfig(routes)
    configure_fp8_build(unquantized, None)
    assert unquantized.build_route == "" and unquantized.profiling_verbosity is None


# ---------------------------------------------------------------------------
# Tiny engines: the FP8 graph (dense Q/DQ and the gathered FP8 experts) builds and runs.
# ---------------------------------------------------------------------------

_MAX_CACHE = 64
_TORCH_DTYPE = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT32: torch.int32,
}


def _run_engine(plan: bytes, tokens: list[int]) -> np.ndarray:
    """Run a decoder plan on ``tokens`` with an empty cache and return the last-row logits."""
    engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(plan)
    context = engine.create_execution_context()
    count = len(tokens)
    mask = torch.full((count, _MAX_CACHE + count), -1e4)
    for row in range(count):
        mask[row, _MAX_CACHE : _MAX_CACHE + row + 1] = 0
    feeds = {
        "token_id": torch.tensor(tokens, dtype=torch.int32),
        "position_id": torch.arange(count, dtype=torch.int32),
        "attention_mask": mask if count > 1 else mask[:1],
        "input_embed": torch.zeros(count, D),
        "use_input_embed": torch.zeros(count, 1),
    }
    outputs = {}
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        dtype = _TORCH_DTYPE[engine.get_tensor_dtype(name)]
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            tensor = feeds.get(name)
            if tensor is None:  # KV cache inputs
                tensor = torch.zeros(_MAX_CACHE, ATTN)
            tensor = tensor.to(dtype).cuda()
            feeds[name] = tensor
            if -1 in tuple(engine.get_tensor_shape(name)):
                context.set_input_shape(name, tuple(tensor.shape))
            context.set_tensor_address(name, tensor.data_ptr())
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            shape = tuple(context.get_tensor_shape(name))
            outputs[name] = torch.empty(shape, dtype=_TORCH_DTYPE[engine.get_tensor_dtype(name)])
            outputs[name] = outputs[name].cuda()
            context.set_tensor_address(name, outputs[name].data_ptr())
    assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    return outputs["logits"].float().cpu().numpy().reshape(-1)


@pytest.mark.parametrize("role", ["decode", "prefill"])
@pytest.mark.parametrize("fp32_layers", [(), (1,)], ids=["fp16", "fp32-layer"])
def test_quantized_decoder_engine_builds_and_matches_the_plain_engine(
    tiny, role: str, fp32_layers: tuple[int, ...]
) -> None:
    """Both decoder engines build with FP8 projections and gathered FP8 experts, and agree with fp16."""
    if not torch.cuda.is_available():
        pytest.skip("engine build needs a GPU")
    path, config = tiny
    plugin = model._DeepseekOcrModel()
    config.raw["_model_dir"] = str(path)
    config.raw["_fp32_layers"] = fp32_layers
    config.raw["_decoder_engine_role"] = role
    plain_weights = plugin.load_weights(str(path), config, precision="fp16")
    ctx = calibrate_deepseek_ocr(path, graph_ops, "fp8", config)
    fp8_weights = plugin.load_weights(str(path), config, precision="fp16", quant_ctx=ctx)

    plain = plugin.build_engine(config, plain_weights, _MAX_CACHE, precision="fp16")
    fp8 = plugin.build_engine(config, fp8_weights, _MAX_CACHE, precision="fp16", quant_ctx=ctx)
    assert len(fp8) < len(plain)

    tokens = [5, 17, 42, 9] if role == "prefill" else [5]
    reference, quantized = _run_engine(plain, tokens), _run_engine(fp8, tokens)
    assert np.isfinite(quantized).all()
    cosine = float(
        np.dot(reference, quantized) / (np.linalg.norm(reference) * np.linalg.norm(quantized))
    )
    assert cosine > 0.95, cosine


# ---------------------------------------------------------------------------
# Prefill runs every expert as three plain matmuls instead of gathering per-token weights.
# ---------------------------------------------------------------------------


def test_dense_expert_layout_matches_the_gathered_formulation() -> None:
    """The all-experts matmuls with zero routing weights equal the per-token top-k sum."""
    from families.deepseek_ocr import graph_blocks

    rng = np.random.default_rng(3)
    tokens, experts, hidden, inter, top_k = 5, 8, 16, 12, 3
    w_gate = rng.standard_normal((experts, hidden, inter)).astype(np.float32)
    w_up = rng.standard_normal((experts, hidden, inter)).astype(np.float32)
    w_down = rng.standard_normal((experts, inter, hidden)).astype(np.float32)
    x = rng.standard_normal((tokens, hidden)).astype(np.float32)
    indices = np.stack([rng.permutation(experts)[:top_k] for _ in range(tokens)])
    weights = rng.random((tokens, top_k)).astype(np.float32)

    def silu(v):
        return v / (1.0 + np.exp(-v))

    expected = np.zeros((tokens, hidden), np.float32)
    for t in range(tokens):
        for k in range(top_k):
            e = indices[t, k]
            expected[t] += weights[t, k] * (silu(x[t] @ w_gate[e]) * (x[t] @ w_up[e])) @ w_down[e]

    gate, up, down = graph_blocks.dense_expert_layout(w_gate, w_up, w_down)
    assert gate.shape == up.shape == (hidden, experts * inter)
    assert down.shape == (experts * inter, hidden)
    dense_weights = np.zeros((tokens, experts), np.float32)
    for t in range(tokens):
        dense_weights[t, indices[t]] = weights[t]
    gated = (silu(x @ gate) * (x @ up)).reshape(tokens, experts, inter)
    actual = (gated * dense_weights[:, :, None]).reshape(tokens, experts * inter) @ down
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("quantized", [False, True], ids=["fp16", "fp8"])
def test_prefill_engine_with_all_expert_matmuls_matches_the_gathered_engine(
    tiny, monkeypatch, quantized: bool
) -> None:
    """The prefill engine's logits do not change when the experts are not gathered per token."""
    if not torch.cuda.is_available():
        pytest.skip("engine build needs a GPU")
    path, config = tiny
    plugin = model._DeepseekOcrModel()
    config.raw["_model_dir"] = str(path)
    config.raw["_fp32_layers"] = ()
    config.raw["_decoder_engine_role"] = "prefill"
    ctx = calibrate_deepseek_ocr(path, graph_ops, "fp8", config) if quantized else None
    load_options = {"quant_ctx": ctx} if quantized else {}
    weights = plugin.load_weights(str(path), config, precision="fp16", **load_options)

    dense = plugin.build_engine(config, weights, _MAX_CACHE, precision="fp16", quant_ctx=ctx)
    original = model._add_moe_with_shared_experts
    monkeypatch.setattr(
        model,
        "_add_moe_with_shared_experts",
        lambda *args, **kwargs: original(*args, **{**kwargs, "dense_experts": False}),
    )
    gathered = plugin.build_engine(config, weights, _MAX_CACHE, precision="fp16", quant_ctx=ctx)

    tokens = [5, 17, 42, 9, 3, 60]
    reference, actual = _run_engine(gathered, tokens), _run_engine(dense, tokens)
    assert np.isfinite(actual).all()
    cosine = float(np.dot(reference, actual) / (np.linalg.norm(reference) * np.linalg.norm(actual)))
    assert cosine > 0.999, cosine
