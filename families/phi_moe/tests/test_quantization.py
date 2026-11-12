# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phi-MoE FP8 quantization: weight scaling, projection coverage, calibration tool, engine build."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
trt = pytest.importorskip("tensorrt")
safetensors_numpy = pytest.importorskip("safetensors.numpy")

from .. import calibrate_fp8_scales, graph_ops, model  # noqa: E402
from .. import quantization as quantization_module  # noqa: E402
from ..config import ModelConfig  # noqa: E402
from ..quantization import (  # noqa: E402
    ACTIVATION_SCALES_FILENAME,
    ATTENTION_PROJECTIONS,
    PhiMoeQuantContext,
    calibrate_phi_moe,
    configure_fp8_build,
    dequantize_fp8_weight,
    quantize_fp8_weight,
    scale_key,
)

# A tiny Phi-MoE: head size 128 like the real model (TensorRT 11.1 can crash on small heads).
D, HEADS, KV_HEADS, LAYERS, VOCAB = 256, 2, 2, 2, 100
EXPERTS, TOP_K, FFN = 4, 2, 128
HEAD_DIM = D // HEADS
KV = KV_HEADS * HEAD_DIM


def _config_json() -> dict:
    return {
        "model_type": "phimoe",
        "architectures": ["PhiMoEForCausalLM"],
        "hidden_size": D,
        "intermediate_size": FFN,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "vocab_size": VOCAB,
        "num_local_experts": EXPERTS,
        "num_experts_per_tok": TOP_K,
        "router_jitter_noise": 0.01,
        "max_position_embeddings": 128,
    }


def _tensors() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)

    def w(*shape):
        return (rng.standard_normal(shape) * 0.05).astype(np.float32)

    t = {
        "model.embed_tokens.weight": w(VOCAB, D),
        "model.norm.weight": np.ones(D, np.float32),
        "model.norm.bias": np.zeros(D, np.float32),
        "lm_head.weight": w(VOCAB, D),
        "lm_head.bias": np.zeros(VOCAB, np.float32),
    }
    for i in range(LAYERS):
        layer = f"model.layers.{i}"
        for norm in ("input_layernorm", "post_attention_layernorm"):
            t[f"{layer}.{norm}.weight"] = np.ones(D, np.float32)
            t[f"{layer}.{norm}.bias"] = np.zeros(D, np.float32)
        for stem, rows in (("q_proj", D), ("k_proj", KV), ("v_proj", KV), ("o_proj", D)):
            t[f"{layer}.self_attn.{stem}.weight"] = w(rows, D)
            t[f"{layer}.self_attn.{stem}.bias"] = w(rows)
        t[f"{layer}.block_sparse_moe.gate.weight"] = w(EXPERTS, D) * 20
        for e in range(EXPERTS):
            expert = f"{layer}.block_sparse_moe.experts.{e}"
            t[f"{expert}.w1.weight"] = w(FFN, D)
            t[f"{expert}.w3.weight"] = w(FFN, D)
            t[f"{expert}.w2.weight"] = w(D, FFN)
    return t


def _write_checkpoint(path: Path, tensors: dict | None = None) -> ModelConfig:
    """Write a random tiny checkpoint with HF tensor names and return its config."""
    safetensors_numpy.save_file(tensors or _tensors(), str(path / "model.safetensors"))
    (path / "config.json").write_text(json.dumps(_config_json()), encoding="utf-8")
    return ModelConfig.from_dir(path)


def _all_scales(value: float = 0.02) -> dict[str, float]:
    return {
        f"layer.{layer}.{stem}": value
        for layer in range(LAYERS)
        for _, stem in ATTENTION_PROJECTIONS
    }


@pytest.fixture
def tiny(tmp_path: Path, monkeypatch):
    """Tiny checkpoint plus an activation-scale table that covers every quantized projection."""
    config = _write_checkpoint(tmp_path)
    monkeypatch.setattr(quantization_module, "load_activation_scales", lambda cfg: _all_scales())
    return tmp_path, config


def test_fp8_weight_scale_is_amax_over_448() -> None:
    """The scale is amax / 448 and the bytes decode to the original weight within e4m3 precision."""
    weight = np.random.default_rng(1).standard_normal((8, 32)).astype(np.float32)
    packed, scale = quantize_fp8_weight(weight)
    assert scale == pytest.approx(float(np.abs(weight).max()) / 448.0)
    assert packed.dtype == np.uint8 and packed.shape == weight.shape
    decoded = dequantize_fp8_weight(packed, scale).numpy()
    assert np.abs(decoded - weight).max() <= 0.07 * float(np.abs(weight).max())


def test_fp8_weight_never_overflows_to_nan() -> None:
    """Values at the scale boundary clip to the largest finite e4m3 value."""
    packed, scale = quantize_fp8_weight(np.full((4, 4), 3.0, dtype=np.float32))
    assert not torch.isnan(dequantize_fp8_weight(packed, scale)).any()


def test_fp8_weight_rejects_all_zero() -> None:
    """A weight with no positive amax cannot be scaled."""
    with pytest.raises(ValueError, match="amax"):
        quantize_fp8_weight(np.zeros((4, 4), dtype=np.float32))


def test_scale_key_identifies_model_config() -> None:
    """The scale key names the model type and shape."""
    config = ModelConfig.from_json(json.dumps(_config_json()))
    assert scale_key(config) == f"phimoe-h{D}-l{LAYERS}-i{FFN}"


def test_attention_and_every_expert_are_quantized_except_attention_output(tiny) -> None:
    """q/k/v and every routed expert get FP8 weights; the attention output stays unquantized."""
    path, config = tiny
    ctx = calibrate_phi_moe(path, config, graph_ops, "fp8")
    assert len(ctx.scales) == LAYERS * 3
    for layer in range(LAYERS):
        assert ctx.should_quantize(f"layer.{layer}.w_q")
        assert ctx.should_quantize(f"layer.{layer}.experts.w_gate")
        assert ctx.should_quantize(f"layer.{layer}.experts.w_up")
        assert ctx.should_quantize(f"layer.{layer}.experts.w_down")
        # TensorRT 11.4 cannot compile an FP8 Quantize right after the fused attention.
        assert not ctx.should_quantize(f"layer.{layer}.w_o")
    entry = ctx.scales["layer.1.w_k"]
    assert (entry.weight.out_features, entry.weight.in_features) == (KV, D)
    assert entry.input_scale == pytest.approx(0.02)
    assert entry.weight.packed.dtype == np.uint8
    assert set(ctx.experts) == {"layer.0", "layer.1"}


def test_expert_stacks_share_one_scale_and_use_the_engine_layout(tiny) -> None:
    """Experts are stored as [experts, in, out] bytes with the layer-wide amax / 448 scale."""
    path, config = tiny
    ctx = calibrate_phi_moe(path, config, graph_ops, "fp8")
    tensors = safetensors_numpy.load_file(str(path / "model.safetensors"))
    stack = ctx.experts["layer.1"]
    assert stack.gate.shape == (EXPERTS, D, FFN)
    assert stack.up.shape == (EXPERTS, D, FFN)
    assert stack.down.shape == (EXPERTS, FFN, D)
    base = "model.layers.1.block_sparse_moe.experts"
    gate = np.stack([tensors[f"{base}.{e}.w1.weight"] for e in range(EXPERTS)])  # [E, out, in]
    down = np.stack([tensors[f"{base}.{e}.w2.weight"] for e in range(EXPERTS)])
    assert stack.gate.weight_scale == pytest.approx(float(np.abs(gate).max()) / 448.0)
    decoded = dequantize_fp8_weight(stack.gate.packed, stack.gate.weight_scale).numpy()
    np.testing.assert_allclose(
        decoded, gate.transpose(0, 2, 1), atol=0.07 * float(np.abs(gate).max())
    )
    decoded = dequantize_fp8_weight(stack.down.packed, stack.down.weight_scale).numpy()
    np.testing.assert_allclose(
        decoded, down.transpose(0, 2, 1), atol=0.07 * float(np.abs(down).max())
    )


def test_projection_without_scale_is_an_error(tmp_path: Path, monkeypatch) -> None:
    """A projection with no activation scale raises instead of silently staying fp16."""
    config = _write_checkpoint(tmp_path)
    scales = _all_scales()
    del scales["layer.0.w_v"]
    monkeypatch.setattr(quantization_module, "load_activation_scales", lambda cfg: scales)
    with pytest.raises(ValueError, match=r"layer\.0\.w_v.*calibrate_fp8_scales\.py"):
        calibrate_phi_moe(tmp_path, config, graph_ops, "fp8")


def test_already_quantized_checkpoint_is_rejected(tiny) -> None:
    """A checkpoint that already stores weight scales is not quantized a second time."""
    path, config = tiny
    tensors = safetensors_numpy.load_file(str(path / "model.safetensors"))
    tensors["model.layers.1.block_sparse_moe.experts.0.w3.weight_scale"] = np.ones(1, np.float32)
    safetensors_numpy.save_file(tensors, str(path / "model.safetensors"))
    with pytest.raises(ValueError, match="already quantized"):
        calibrate_phi_moe(path, config, graph_ops, "fp8")


def test_quantization_config_in_checkpoint_is_rejected(tiny) -> None:
    """A config.json with a quantization_config is a quantized checkpoint (for example FP8)."""
    path, config = tiny
    config.raw["quantization_config"] = {"quant_method": "compressed-tensors"}
    with pytest.raises(ValueError, match="quantization_config"):
        calibrate_phi_moe(path, config, graph_ops, "fp8")


def test_fp8_checkpoint_weights_are_rejected(tiny) -> None:
    """Weights that are already stored as one byte per element are not re-quantized."""
    path, config = tiny
    tensors = safetensors_numpy.load_file(str(path / "model.safetensors"))
    tensors["model.layers.0.self_attn.q_proj.weight"] = np.zeros((D, D), dtype=np.uint8)
    safetensors_numpy.save_file(tensors, str(path / "model.safetensors"))
    with pytest.raises(ValueError, match="original"):
        calibrate_phi_moe(path, config, graph_ops, "fp8")


def test_unsupported_format_is_rejected(tiny) -> None:
    """Only fp8 is implemented."""
    path, config = tiny
    with pytest.raises(NotImplementedError, match="nvfp4"):
        calibrate_phi_moe(path, config, graph_ops, "nvfp4")


def test_unknown_model_config_is_reported(tmp_path: Path) -> None:
    """A model config with no committed scales raises and names the calibration tool."""
    config = _write_checkpoint(tmp_path)
    with pytest.raises(NotImplementedError, match="calibrate_fp8_scales"):
        calibrate_phi_moe(tmp_path, config, graph_ops, "fp8")


def test_committed_scales_cover_every_quantized_projection() -> None:
    """Every committed model config has a positive scale for q/k/v of every layer."""
    path = Path(quantization_module.__file__).parent / ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8"))
    assert table, "no committed activation scales"
    for key, scales in table.items():
        match = re.fullmatch(r"phimoe-h(\d+)-l(\d+)-i(\d+)", key)
        assert match, key
        layers = int(match.group(2))
        expected = {
            f"layer.{layer}.{stem}" for layer in range(layers) for _, stem in ATTENTION_PROJECTIONS
        }
        assert set(scales) == expected, key
        assert all(np.isfinite(v) and v > 0 for v in scales.values()), key


def test_load_weights_uses_shape_only_placeholders(tiny) -> None:
    """Quantized attention projections and experts are not read in 16 bits."""
    path, config = tiny
    ctx = calibrate_phi_moe(path, config, graph_ops, "fp8")
    weights = model._PhiMoEModel().load_weights(str(path), config, quant_ctx=ctx)

    assert weights["layer.0.w_q"].shape == (D, D)
    assert weights["layer.0.w_k"].shape == (D, KV)
    # Placeholders must not hold a full-size copy of the weight.
    assert weights["layer.0.w_q"].strides == (0, 0)
    assert weights["_attention_size"] == D
    # The unquantized attention output, the router and the head are still loaded.
    assert weights["layer.0.w_o"].shape == (D, D)
    assert weights["layer.0.w_o"].strides != (0, 0)
    assert weights["layer.0.router"].shape == (D, EXPERTS)
    assert weights["w_out"].shape == (D, VOCAB)
    # The experts are not loaded at all.
    assert not any(".expert." in name for name in weights)


def test_load_weights_without_context_is_unchanged(tiny) -> None:
    """An unquantized build still loads every projection and expert."""
    path, config = tiny
    weights = model._PhiMoEModel().load_weights(str(path), config)
    assert weights["layer.0.w_q"].shape == (D, D) and weights["layer.0.w_q"].strides != (0, 0)
    assert weights["layer.1.expert.3.w_down"].shape == (FFN, D)


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
    ctx = PhiMoeQuantContext(scales={}, experts={}, graph_ops=graph_ops)
    config = _FakeBuilderConfig(routes)
    configure_fp8_build(config, ctx)
    assert config.build_route == expected
    assert config.profiling_verbosity is not None
    unquantized = _FakeBuilderConfig(routes)
    configure_fp8_build(unquantized, None)
    assert unquantized.build_route == "" and unquantized.profiling_verbosity is None


class _Writer:
    """Collects what build() writes, without a bundle file."""

    def __init__(self) -> None:
        self.sections: dict[str, bytes] = {}

    def set_header(self, **_kwargs) -> None:
        pass

    def add_bytes(self, name: str, data: bytes) -> None:
        self.sections[name] = data

    def add_json(self, name: str, data) -> None:
        self.sections[name] = json.dumps(data).encode()


def _request(path: Path, **overrides):
    fields = {
        "model_dir": path,
        "output_path": path / "out.bundle",
        "family": "phi_moe",
        "task": "text_generation",
        "precision": "fp16",
        "max_sequence_length": 4,
        "quantization": "fp8",
        "dynamic_kv_cache": False,
        "image_height": None,
        "image_width": None,
        "video_num_frames": None,
        "max_batch_size": 1,
        "context_parallel_size": 1,
        "tensor_parallel_size": 1,
        "fp32_layers": (),
        "mtp_seq_len": None,
        "backend": "tensorrt",
        "verbose": False,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"quantization": "nvfp4"}, "does not support quantization"),
        ({"precision": "fp32"}, "requires precision fp16"),
        ({"tensor_parallel_size": 2}, "tensor parallelism"),
    ],
)
def test_build_refuses_unsupported_fp8_requests(tiny, overrides: dict, message: str) -> None:
    """FP8 needs fp16 and no tensor parallelism; other formats are refused before any work."""
    path, _ = tiny
    with pytest.raises(NotImplementedError, match=message):
        model.build(_request(path, **overrides), _Writer())


def test_fp8_build_gives_the_quantization_context_to_both_engines(tiny, monkeypatch) -> None:
    """The decode and the batched prefill engine both get the context.

    The loader leaves quantized projections as shape-only placeholders, so a prefill engine built
    without the context would read zero attention weights.
    """
    path, _ = tiny
    (path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ messages }}"}), encoding="utf-8"
    )
    received: dict[str, object] = {}

    def fake_decode(self, config, weights, max_cache_length, **options):
        received["decode"] = options.get("quant_ctx")
        return b"decode"

    def fake_prefill(self, config, weights, max_cache_length, **options):
        received["prefill"] = options.get("quant_ctx")
        return b"prefill"

    monkeypatch.setattr(model._PhiMoEModel, "build_engine", fake_decode)
    monkeypatch.setattr(model._PhiMoEModel, "build_prefill_engine", fake_prefill)
    writer = _Writer()
    model.build(_request(path), writer)
    assert writer.sections["engine.plan"] == b"decode"
    assert writer.sections["prefill.plan"] == b"prefill"
    assert received["decode"] is not None
    assert received["decode"] is received["prefill"]


def _run_decode(plan: bytes, steps: list[tuple[int, int]]) -> torch.Tensor:
    """Run the single-token decode engine for ``(token, position)`` steps; return the last logits."""
    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(plan)
    context = engine.create_execution_context()
    cache_length = 4
    dev = {"device": "cuda"}
    caches = {
        f"cache_{kind}_{i}": torch.zeros(cache_length, KV, dtype=torch.float16, **dev)
        for kind in "kv"
        for i in range(LAYERS)
    }
    present = {
        f"present_{kind}_{i}": torch.empty(1, KV, dtype=torch.float16, **dev)
        for kind in "kv"
        for i in range(LAYERS)
    }
    token = torch.zeros(1, dtype=torch.int32, **dev)
    position = torch.zeros(1, dtype=torch.int32, **dev)
    mask = torch.zeros(1, cache_length + 1, dtype=torch.float32, **dev)
    logits = torch.empty(1, VOCAB, dtype=torch.float32, **dev)
    buffers = {
        "token_id": token,
        "position_id": position,
        "attention_mask": mask,
        "logits": logits,
        **caches,
        **present,
    }
    for name, tensor in buffers.items():
        context.set_tensor_address(name, tensor.data_ptr())
    for value, pos in steps:
        token.fill_(value)
        position.fill_(pos)
        mask.fill_(-1e4)
        mask[0, :pos] = 0
        mask[0, cache_length] = 0
        context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        for i in range(LAYERS):
            caches[f"cache_k_{i}"][pos] = present[f"present_k_{i}"][0]
            caches[f"cache_v_{i}"][pos] = present[f"present_v_{i}"][0]
    return logits[0].cpu()


@pytest.mark.gpu
@pytest.mark.trt
def test_engine_builds_with_fp8_weights_and_matches_the_fp16_engine(tiny) -> None:
    """The FP8 engine stores FP8 (not fp16) weights, gathers the routed experts and tracks fp16."""
    path, config = tiny
    instance = model._PhiMoEModel()
    plain = instance.build_engine(
        config, instance.load_weights(str(path), config), 4, precision="fp16"
    )
    ctx = calibrate_phi_moe(path, config, graph_ops, "fp8")
    weights = instance.load_weights(str(path), config, quant_ctx=ctx)
    quantized = instance.build_engine(config, weights, 4, precision="fp16", quant_ctx=ctx)

    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(quantized)
    assert engine is not None
    names = {engine.get_tensor_name(i) for i in range(engine.num_io_tensors)}
    assert {"token_id", "logits", "present_k_0"} <= names
    # Experts, q, k and v are 8 bits instead of 16. A plan that stored the experts at 16 bits
    # again (a Dequantize folded into the constants) would miss this bound.
    assert len(quantized) < 0.7 * len(plain), (len(quantized), len(plain))

    steps = [(3, 0), (17, 1), (42, 2)]
    reference = _run_decode(plain, steps)
    result = _run_decode(quantized, steps)
    assert torch.isfinite(result).all()
    cosine = torch.nn.functional.cosine_similarity(reference, result, dim=0).item()
    assert cosine > 0.98, cosine


@pytest.mark.gpu
@pytest.mark.trt
def test_prefill_engine_keeps_fp8_experts_and_quantized_attention(tiny) -> None:
    """The batched prefill engine of an FP8 bundle stores the experts as FP8 and builds."""
    path, config = tiny
    instance = model._PhiMoEModel()
    plain = instance.build_prefill_engine(
        config, instance.load_weights(str(path), config), 4, precision="fp16"
    )
    ctx = calibrate_phi_moe(path, config, graph_ops, "fp8")
    weights = instance.load_weights(str(path), config, quant_ctx=ctx)
    quantized = instance.build_prefill_engine(config, weights, 4, precision="fp16", quant_ctx=ctx)

    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(quantized)
    assert engine is not None
    names = {engine.get_tensor_name(i) for i in range(engine.num_io_tensors)}
    assert {"token_id", "logits", "present_k_0"} <= names
    # A Dequantize of the expert stack that TensorRT folded into constants would store 16 bits
    # (a plan about as large as the fp16 one); the tiny model keeps more 16-bit weights than a
    # real one, so the bound is looser than for the decode engine.
    assert len(quantized) < 0.8 * len(plain), (len(quantized), len(plain))


def test_calibration_tool_maps_module_names_to_weight_stems() -> None:
    """The calibration tool maps module names to internal weight names."""
    match = calibrate_fp8_scales._PROJECTION_NAME.search("model.layers.3.self_attn.v_proj")
    assert match is not None and match.group(1) == "3"
    assert calibrate_fp8_scales._WEIGHT_STEM[match.group(2)] == "w_v"
    assert (
        calibrate_fp8_scales._PROJECTION_NAME.search("model.layers.3.self_attn.v_proj.base") is None
    )
    assert calibrate_fp8_scales._PROJECTION_NAME.search("model.layers.3.self_attn.o_proj") is None
    assert (
        calibrate_fp8_scales._PROJECTION_NAME.search("model.layers.3.block_sparse_moe.gate") is None
    )


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


def test_render_prompt_falls_back_to_plain_text() -> None:
    """A tokenizer without a chat template gets the messages joined as plain text."""

    class _NoTemplate:
        chat_template = None

    text = calibrate_fp8_scales.render_prompt(
        _NoTemplate(), [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    )
    assert text == "a\n\nb"
