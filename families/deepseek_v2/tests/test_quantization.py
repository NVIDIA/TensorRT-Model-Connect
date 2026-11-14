# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V2 FP8 routed experts (weight-only): quantizer, context, loader, tiny engine."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import ml_dtypes
import numpy as np
import pytest
from safetensors.numpy import save_file

from .. import quantization
from ..config import ModelConfig
from ..quantization import (
    DeepseekV2QuantContext,
    configure_fp8_build,
    dequantize_fp8_weight,
    quantize_deepseek_v2_experts,
    quantize_fp8_weight,
)

BF16 = ml_dtypes.bfloat16

# Wide experts so that the FP8 weight savings dominate the plan size.
HIDDEN, DENSE, VOCAB, LAYERS, HEADS = 256, 512, 64, 3, 4
NOPE, ROPE, VDIM, KV_RANK = 32, 16, 32, 64
EXPERTS, TOP_K, MOE_INTER, SHARED = 8, 2, 256, 1
K_DIM = NOPE + ROPE

_RAW = {
    "qk_nope_head_dim": NOPE,
    "qk_rope_head_dim": ROPE,
    "v_head_dim": VDIM,
    "kv_lora_rank": KV_RANK,
    "q_lora_rank": None,
    "n_routed_experts": EXPERTS,
    "n_shared_experts": SHARED,
    "num_experts_per_tok": TOP_K,
    "first_k_dense_replace": 1,
    "moe_layer_freq": 1,
    "moe_intermediate_size": MOE_INTER,
    "norm_topk_prob": False,
    "routed_scaling_factor": 1.0,
    "scoring_func": "softmax",
    "topk_method": "greedy",
}
_CONFIG = dict(
    hidden_size=HIDDEN,
    vocab_size=VOCAB,
    intermediate_size=DENSE,
    num_hidden_layers=LAYERS,
    num_attention_heads=HEADS,
    num_key_value_heads=HEADS,
    **_RAW,
)


def _config(model_type: str = "deepseek_v2") -> ModelConfig:
    """A small model config: one dense layer, then MoE layers."""
    return ModelConfig.create_tiny(model_type, **_CONFIG)


def _checkpoint(path: Path) -> dict[str, np.ndarray]:
    """Write a tiny BF16 safetensors checkpoint and a config.json; return the tensors."""
    rng = np.random.RandomState(0)

    def rand(*shape: int, scale: float = 0.05) -> np.ndarray:
        return (rng.randn(*shape) * scale).astype(BF16)

    def ones(n: int) -> np.ndarray:
        return np.ones(n, dtype=BF16)

    tensors = {"model.embed_tokens.weight": rand(VOCAB, HIDDEN, scale=1.0)}
    tensors["model.norm.weight"] = ones(HIDDEN)
    tensors["lm_head.weight"] = rand(VOCAB, HIDDEN)
    for layer in range(LAYERS):
        p = f"model.layers.{layer}"
        tensors[f"{p}.input_layernorm.weight"] = ones(HIDDEN)
        tensors[f"{p}.post_attention_layernorm.weight"] = ones(HIDDEN)
        tensors[f"{p}.self_attn.q_proj.weight"] = rand(HEADS * K_DIM, HIDDEN)
        tensors[f"{p}.self_attn.kv_a_proj_with_mqa.weight"] = rand(KV_RANK + ROPE, HIDDEN)
        tensors[f"{p}.self_attn.kv_a_layernorm.weight"] = ones(KV_RANK)
        tensors[f"{p}.self_attn.kv_b_proj.weight"] = rand(HEADS * (NOPE + VDIM), KV_RANK)
        tensors[f"{p}.self_attn.o_proj.weight"] = rand(HIDDEN, HEADS * VDIM)
        if layer == 0:
            tensors[f"{p}.mlp.gate_proj.weight"] = rand(DENSE, HIDDEN)
            tensors[f"{p}.mlp.up_proj.weight"] = rand(DENSE, HIDDEN)
            tensors[f"{p}.mlp.down_proj.weight"] = rand(HIDDEN, DENSE)
            continue
        tensors[f"{p}.mlp.gate.weight"] = rand(EXPERTS, HIDDEN, scale=0.2)
        for expert in range(EXPERTS):
            e = f"{p}.mlp.experts.{expert}"
            scale = 0.05 * (1 + expert)  # the layer-wide scale must come from the largest expert
            tensors[f"{e}.gate_proj.weight"] = rand(MOE_INTER, HIDDEN, scale=scale)
            tensors[f"{e}.up_proj.weight"] = rand(MOE_INTER, HIDDEN, scale=scale)
            tensors[f"{e}.down_proj.weight"] = rand(HIDDEN, MOE_INTER, scale=scale)
        s = f"{p}.mlp.shared_experts"
        tensors[f"{s}.gate_proj.weight"] = rand(MOE_INTER * SHARED, HIDDEN)
        tensors[f"{s}.up_proj.weight"] = rand(MOE_INTER * SHARED, HIDDEN)
        tensors[f"{s}.down_proj.weight"] = rand(HIDDEN, MOE_INTER * SHARED)
    save_file(tensors, str(path / "model.safetensors"))
    config = {"model_type": "deepseek_v2", "max_position_embeddings": 128, **_CONFIG}
    config.update(rms_norm_eps=1e-6, rope_theta=10000.0)
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return tensors


def _model():
    """The family model module (it imports TensorRT, so CPU-only environments skip)."""
    pytest.importorskip("tensorrt")
    from .. import model

    return model


@pytest.fixture
def checkpoint(tmp_path: Path):
    """A tiny BF16 checkpoint directory and its tensors."""
    return tmp_path, _checkpoint(tmp_path)


def test_quantize_fp8_weight_scale_and_bytes() -> None:
    """The weight scale is amax / 448 and the bytes decode back to the weight."""
    weight = np.random.RandomState(0).randn(64, 128).astype(np.float32)
    packed, scale = quantize_fp8_weight(weight)

    assert scale == pytest.approx(float(np.abs(weight).max()) / 448.0)
    assert packed.dtype == np.uint8 and packed.shape == weight.shape
    restored = packed.view(ml_dtypes.float8_e4m3fn).astype(np.float32) * scale
    assert np.abs(restored - weight).max() <= np.abs(weight).max() * 0.07  # e4m3 half-step
    assert np.array_equal(restored, dequantize_fp8_weight(packed, scale))
    # the largest value maps to the largest finite e4m3 value (448 is byte 0x7e)
    assert packed.reshape(-1)[int(np.abs(weight).argmax())] & 0x7F == 0x7E
    # known e4m3 encodings at scale 1
    known, _ = quantize_fp8_weight(np.array([1.0, 0.5, -1.0, 448.0, 0.0], np.float32), scale=1.0)
    assert known.tolist() == [0x38, 0x30, 0xB8, 0x7E, 0x00]


def test_quantize_fp8_weight_never_overflows_to_nan() -> None:
    """Values beyond the given scale clip to the largest finite fp8 value."""
    packed, _ = quantize_fp8_weight(np.full((4, 4), 3.0, dtype=np.float32), scale=0.001)
    values = packed.view(ml_dtypes.float8_e4m3fn).astype(np.float32)
    assert np.isfinite(values).all() and (values == 448.0).all()


def test_quantize_fp8_weight_rejects_zero_tensor() -> None:
    """An all-zero weight has no usable scale and is refused."""
    with pytest.raises(ValueError, match="amax"):
        quantize_fp8_weight(np.zeros((2, 2), dtype=np.float32))


def test_context_quantizes_the_routed_experts_only(checkpoint) -> None:
    """Each expert projection is one FP8 stack with the layer-wide scale, in the engine layout."""
    model_dir, tensors = checkpoint
    ctx = quantize_deepseek_v2_experts(model_dir, "fp8", _config())

    # Layer 0 is dense, layers 1 and 2 are MoE; nothing else is owned by the context.
    assert set(ctx.experts) == {"layer.1", "layer.2"}
    assert ctx.quantizes_experts("layer.1") and not ctx.quantizes_experts("layer.0")
    assert {f.name for f in DeepseekV2QuantContext.__dataclass_fields__.values()} == {
        "experts",
        "keep_alive",
    }

    for layer in (1, 2):
        entry = ctx.experts[f"layer.{layer}"]
        for hf_stem, stack, shape in (
            ("gate_proj", entry.gate, (EXPERTS, HIDDEN, MOE_INTER)),
            ("up_proj", entry.up, (EXPERTS, HIDDEN, MOE_INTER)),
            ("down_proj", entry.down, (EXPERTS, MOE_INTER, HIDDEN)),
        ):
            assert stack.shape == shape  # [experts, in, out]
            assert stack.packed.dtype == np.uint8
            sources = [
                tensors[f"model.layers.{layer}.mlp.experts.{e}.{hf_stem}.weight"].astype(np.float32)
                for e in range(EXPERTS)
            ]
            layer_amax = max(float(np.abs(w).max()) for w in sources)
            assert stack.weight_scale == pytest.approx(layer_amax / 448.0)
            for expert, source in enumerate(sources):
                expected, _ = quantize_fp8_weight(source.T, stack.weight_scale)
                assert np.array_equal(stack.packed[expert], expected)
                restored = dequantize_fp8_weight(stack.packed[expert], stack.weight_scale)
                assert np.abs(restored - source.T).max() <= layer_amax * 0.07


def test_loader_skips_the_quantized_experts_and_nothing_else(checkpoint) -> None:
    """Experts become shape-only placeholders; every other weight loads exactly as before."""
    model_dir, _ = checkpoint
    config = _config()
    net = _model()._DeepSeekV2Model()
    plain = net.load_weights(str(model_dir), config)
    ctx = quantize_deepseek_v2_experts(model_dir, "fp8", config)
    quant = net.load_weights(str(model_dir), config, quant_ctx=ctx)

    for layer in (1, 2):
        for stem, shape in (
            ("w_gate", (EXPERTS, HIDDEN, MOE_INTER)),
            ("w_up", (EXPERTS, HIDDEN, MOE_INTER)),
            ("w_down", (EXPERTS, MOE_INTER, HIDDEN)),
        ):
            placeholder = quant[f"layer.{layer}.experts.{stem}"]
            assert placeholder.shape == shape and not np.asarray(placeholder).any()
        assert not any(".expert." in name for name in quant if name.startswith(f"layer.{layer}."))
    plain_only = {name for name in plain if name not in quant}
    assert plain_only and all(".expert." in name for name in plain_only)
    assert not any(".expert." in name for name in quant)
    for name in quant:
        if ".experts." in name:
            continue
        assert np.array_equal(np.asarray(quant[name]), np.asarray(plain[name])), name
    # attention, dense MLP, shared experts and router are real weights
    for name in ("layer.0.w_q", "layer.0.w_o", "layer.0.w_gate", "layer.1.shared.w_up"):
        assert np.asarray(quant[name]).any(), name
    assert np.asarray(quant["layer.1.router"]).any()


def test_unquantized_loader_is_unchanged(checkpoint) -> None:
    """Without a context the loader reads every expert and has no stacked-expert key."""
    model_dir, tensors = checkpoint
    weights = _model()._DeepSeekV2Model().load_weights(str(model_dir), _config())
    assert not any(".experts." in name for name in weights)
    for e in range(EXPERTS):
        expected = tensors[f"model.layers.1.mlp.experts.{e}.up_proj.weight"].astype(np.float32).T
        assert np.array_equal(weights[f"layer.1.expert.{e}.w_up"], expected)


def test_unsupported_quantization_is_refused(checkpoint) -> None:
    """Only fp8 is supported."""
    model_dir, _ = checkpoint
    with pytest.raises(NotImplementedError, match="supports quantization"):
        quantize_deepseek_v2_experts(model_dir, "nvfp4", _config())


def test_other_model_type_is_refused(checkpoint) -> None:
    """deepseek_v3 shares the family but its experts were not measured."""
    model_dir, _ = checkpoint
    with pytest.raises(NotImplementedError, match="model_type deepseek_v2"):
        quantize_deepseek_v2_experts(model_dir, "fp8", _config("deepseek_v3"))


def test_quantization_config_is_refused(checkpoint) -> None:
    """A checkpoint config with a quantization_config is already quantized."""
    model_dir, _ = checkpoint
    config = _config()
    config.raw["quantization_config"] = {"quant_method": "fp8"}
    with pytest.raises(ValueError, match="quantization_config"):
        quantize_deepseek_v2_experts(model_dir, "fp8", config)


@pytest.mark.parametrize("side", ["weight_scale", "weight_scale_inv", "weight_scale_2"])
def test_checkpoint_with_scale_tensors_is_refused(checkpoint, side: str) -> None:
    """A checkpoint that carries quantization scale tensors is refused, not quantized twice."""
    model_dir, tensors = checkpoint
    tensors = dict(tensors)
    tensors[f"model.layers.1.mlp.experts.0.gate_proj.{side}"] = np.ones(1, dtype=np.float32)
    save_file(tensors, str(model_dir / "model.safetensors"))
    with pytest.raises(ValueError, match="already quantized"):
        quantize_deepseek_v2_experts(model_dir, "fp8", _config())


def test_fp8_checkpoint_weights_are_refused(checkpoint) -> None:
    """Expert weights stored in 8 bits are not the original checkpoint."""
    model_dir, tensors = checkpoint
    tensors = dict(tensors)
    name = "model.layers.2.mlp.experts.3.down_proj.weight"
    tensors[name] = tensors[name].astype(np.int8)  # any 8-bit dtype
    save_file(tensors, str(model_dir / "model.safetensors"))
    with pytest.raises(ValueError, match="needs the original"):
        quantize_deepseek_v2_experts(model_dir, "fp8", _config())


def test_missing_expert_tensor_is_refused(checkpoint) -> None:
    """A checkpoint without an expert weight fails with the tensor name."""
    model_dir, tensors = checkpoint
    tensors = {k: v for k, v in tensors.items() if "experts.5.up_proj" not in k}
    save_file(tensors, str(model_dir / "model.safetensors"))
    with pytest.raises(ValueError, match=r"experts\.5\.up_proj\.weight"):
        quantize_deepseek_v2_experts(model_dir, "fp8", _config())


def test_zero_experts_are_refused(checkpoint) -> None:
    """An all-zero expert projection has no scale to quantize with."""
    model_dir, tensors = checkpoint
    tensors = dict(tensors)
    for e in range(EXPERTS):
        name = f"model.layers.1.mlp.experts.{e}.down_proj.weight"
        tensors[name] = np.zeros_like(tensors[name])
    save_file(tensors, str(model_dir / "model.safetensors"))
    with pytest.raises(ValueError, match="amax"):
        quantize_deepseek_v2_experts(model_dir, "fp8", _config())


def test_config_without_a_moe_layer_is_refused(checkpoint) -> None:
    """A config whose layers are all dense has nothing to quantize."""
    model_dir, _ = checkpoint
    config = ModelConfig.create_tiny("deepseek_v2", **{**_CONFIG, "first_k_dense_replace": LAYERS})
    with pytest.raises(RuntimeError, match="no routed-expert layer"):
        quantize_deepseek_v2_experts(model_dir, "fp8", config)


class _FakeBuilderConfig:
    """Stands in for IBuilderConfig; only the attributes the build option touches."""

    def __init__(self, all_build_routes: str) -> None:
        self.all_build_routes = all_build_routes
        self.build_route = ""


_DUAL = '{"option": "-peep:match_dual_gemm"}'
_PPG = '{"option": "-quantize_ppg"}'
_BOTH = f"[{_DUAL}, {_PPG}]"


@pytest.mark.parametrize(
    ("routes", "expected"),
    [(_DUAL, "-peep:match_dual_gemm=off"), ("", "")],
)
def test_dual_gemm_route_is_set_only_for_fp8_and_only_when_listed(
    routes, expected, monkeypatch
) -> None:
    """FP8 builds turn the dual-GEMM kernel off when TensorRT lists the knob; others are untouched."""
    monkeypatch.setattr(quantization, "_compute_capability", lambda: (10, 0))
    config = _FakeBuilderConfig(routes)
    configure_fp8_build(config, DeepseekV2QuantContext(experts={}))
    assert config.build_route == expected
    unquantized = _FakeBuilderConfig(routes)
    configure_fp8_build(unquantized, None)
    assert unquantized.build_route == ""


@pytest.mark.parametrize(
    ("routes", "capability", "expected"),
    [
        (_BOTH, (10, 3), "-peep:match_dual_gemm=off -quantize_ppg=off"),
        (_PPG, (10, 3), "-quantize_ppg=off"),
        (_BOTH, (10, 0), "-peep:match_dual_gemm=off"),
        (_DUAL, (10, 3), "-peep:match_dual_gemm=off"),
        ('{"option": "-quantize_ppg_conv"}', (10, 3), ""),
    ],
    ids=["sm103", "sm103-ppg-only", "sm100", "sm103-not-listed", "sm103-other-name"],
)
def test_quantize_ppg_route_is_set_only_on_sm103_and_only_when_listed(
    routes, capability, expected, monkeypatch
) -> None:
    """B300 (sm103) FP8 builds also turn quantize propagation off; B200 (sm100) does not."""
    monkeypatch.setattr(quantization, "_compute_capability", lambda: capability)
    config = _FakeBuilderConfig(routes)
    configure_fp8_build(config, DeepseekV2QuantContext(experts={}))
    assert config.build_route == expected
    unquantized = _FakeBuilderConfig(routes)
    configure_fp8_build(unquantized, None)
    assert unquantized.build_route == ""


def _build_request(model_dir: Path = Path("."), **updates):
    """Return a minimal build request."""
    request = SimpleNamespace(
        dynamic_kv_cache=False,
        image_height=None,
        image_width=None,
        video_num_frames=None,
        max_batch_size=1,
        context_parallel_size=1,
        task="text_generation",
        model_dir=model_dir,
        precision="fp16",
        max_sequence_length=8,
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


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"quantization": "nvfp4"}, "quantization='nvfp4'"),
        ({"quantization": "int8"}, "quantization='int8'"),
        ({"precision": "fp32"}, "requires precision fp16"),
        ({"precision": "bf16"}, "requires precision fp16"),
        ({"tensor_parallel_size": 2}, "does not support tensor parallelism"),
    ],
)
def test_build_refuses_unsupported_fp8_requests(checkpoint, updates, message) -> None:
    """Other quantization modes, fp32 / bf16 precision and tensor parallelism are refused."""
    model_dir, _ = checkpoint
    with pytest.raises(NotImplementedError, match=message):
        _model().build(_build_request(model_dir, **updates), writer=None)


class _Writer:
    """Collects what ``build`` writes."""

    def __init__(self) -> None:
        self.blobs: dict[str, bytes] = {}

    def set_header(self, **kwargs) -> None:
        pass

    def add_bytes(self, name: str, data: bytes) -> None:
        self.blobs[name] = data

    def add_json(self, name: str, data) -> None:
        pass

    def __getattr__(self, name: str):
        return lambda *args, **kwargs: None


def _engine_info(plan: bytes):
    """Deserialize a plan and return its engine and the layer list of the inspector."""
    import tensorrt as trt

    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(plan)
    assert engine is not None
    info = json.loads(
        engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON)
    )
    return engine, info.get("Layers", [])


def _io(engine) -> dict[str, tuple]:
    """Return name -> (dtype, shape) of every I/O tensor of an engine."""
    return {
        engine.get_tensor_name(i): (
            engine.get_tensor_dtype(engine.get_tensor_name(i)),
            tuple(engine.get_tensor_shape(engine.get_tensor_name(i))),
        )
        for i in range(engine.num_io_tensors)
    }


class _NetworkSpy:
    """Forwards to a TensorRT network and records constants and Dequantize scales."""

    def __init__(self, network) -> None:
        self._network = network
        self.constants: list[tuple[tuple[int, ...], object, str]] = []
        self.dequantize_scales: list[str] = []

    def add_constant(self, shape, weights):
        layer = self._network.add_constant(shape, weights)
        self.constants.append((tuple(shape), weights.dtype, layer.get_output(0).name))
        return layer

    def add_dequantize(self, tensor, scale, dtype):
        self.dequantize_scales.append(scale.name)
        return self._network.add_dequantize(tensor, scale, dtype)

    def __getattr__(self, name: str):
        return getattr(self._network, name)


def test_routed_experts_graph_has_fp8_constants_and_a_run_time_scale(checkpoint) -> None:
    """Three FP8 constants, each behind a Dequantize whose scale is not a constant, then Gather."""
    trt = pytest.importorskip("tensorrt")
    model_dir, _ = checkpoint
    ctx = quantize_deepseek_v2_experts(model_dir, "fp8", _config())
    network = trt.Builder(trt.Logger(trt.Logger.ERROR)).create_network(0)
    spy = _NetworkSpy(network)
    inp = network.add_input("inp", trt.float16, (1, HIDDEN))
    indices = network.add_input("indices", trt.int32, (1, TOP_K))
    weights = network.add_input("weights", trt.float32, (1, TOP_K))
    out = ctx.routed_experts(spy, inp, indices, weights, "layer.1", HIDDEN, TOP_K)
    assert tuple(out.shape) == (1, HIDDEN) and out.dtype == trt.float16

    fp8 = [(shape, name) for shape, dtype, name in spy.constants if dtype == trt.DataType.FP8]
    assert [shape for shape, _ in fp8] == [
        (EXPERTS, HIDDEN, MOE_INTER),
        (EXPERTS, HIDDEN, MOE_INTER),
        (EXPERTS, MOE_INTER, HIDDEN),
    ]
    # one Dequantize per stack; its scale is computed from the router indices, so TensorRT
    # cannot fold the dequantize of the constant into a 16-bit constant
    constant_outputs = {name for _, _, name in spy.constants}
    assert len(spy.dequantize_scales) == 3
    assert not constant_outputs & set(spy.dequantize_scales)
    layers = [network.get_layer(i) for i in range(network.num_layers)]
    assert sum(layer.type == trt.LayerType.DEQUANTIZE for layer in layers) == 3
    assert sum(layer.type == trt.LayerType.GATHER for layer in layers) == 3


@pytest.mark.gpu
@pytest.mark.trt
def test_engine_builds_with_fp8_experts(checkpoint, monkeypatch) -> None:
    """The FP8 expert engine builds, keeps the I/O, has FP8 constants and a smaller plan."""
    model_dir, _ = checkpoint
    # The fp32-accumulation MLA attention of the family does not compile on TensorRT 11.4 with or
    # without quantization, and it is not what is under test.
    monkeypatch.setattr(_model(), "_use_fp32_mla_attention", lambda dtype, head_dim: False)
    config = _config()
    net = _model()._DeepSeekV2Model()
    plain = net.build_engine(config, net.load_weights(str(model_dir), config), 8, precision="fp16")
    ctx = quantize_deepseek_v2_experts(model_dir, "fp8", config)
    weights = net.load_weights(str(model_dir), config, quant_ctx=ctx)
    quant = net.build_engine(config, weights, 8, precision="fp16", quant_ctx=ctx)
    # An empty context (no quantized layer) builds the unquantized graph.
    empty = net.build_engine(
        config,
        net.load_weights(str(model_dir), config),
        8,
        precision="fp16",
        quant_ctx=DeepseekV2QuantContext(experts={}),
    )

    plain_engine, plain_layers = _engine_info(plain)
    quant_engine, quant_layers = _engine_info(quant)
    _, empty_layers = _engine_info(empty)
    assert _io(quant_engine) == _io(plain_engine)
    # 8-bit experts hold most of the weights: the plan is clearly smaller.
    assert len(quant) < 0.75 * len(plain)
    assert len(empty_layers) == len(plain_layers) and len(empty) == pytest.approx(
        len(plain), rel=0.02
    )


@pytest.mark.gpu
@pytest.mark.trt
def test_build_writes_a_smaller_fp8_bundle_engine(checkpoint, monkeypatch) -> None:
    """``build`` with quantization=fp8 writes one engine plan smaller than the unquantized one."""
    model_dir, _ = checkpoint
    monkeypatch.setattr(_model(), "_use_fp32_mla_attention", lambda dtype, head_dim: False)
    plain, quant = _Writer(), _Writer()
    _model().build(_build_request(model_dir, quantization=None), plain)
    _model().build(_build_request(model_dir), quant)
    assert set(plain.blobs) == set(quant.blobs) == {"engine.plan"}
    assert len(quant.blobs["engine.plan"]) < 0.75 * len(plain.blobs["engine.plan"])
