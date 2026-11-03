# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Llama FP8 / NVFP4 / INT4-AWQ quantization: checkpoint reading, weight loading, engine build."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")
torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

from .. import calibrate_awq_scales, graph_blocks, graph_ops  # noqa: E402
from ..build_request import BuildRequest, coerce_request  # noqa: E402
from ..checkpoint_mapper import load_standard_weights  # noqa: E402
from ..config import ModelConfig  # noqa: E402
from ..quantization import (  # noqa: E402
    LlamaQuantContext,
    _AWQWeight,
    _FP8Weight,
    _LayerScales,
    _NVFP4Weight,
    _open_raw_readers,
    awq_identity,
    calibrate_llama,
    quantize_awq_weight,
)
from ..standard_decoder_builder import build_standard_decoder_engine  # noqa: E402

# Wide enough that packed-weight savings dominate the per-layer Q/DQ plan overhead.
HIDDEN, MLP, VOCAB, LAYERS, HEADS = 512, 2048, 32, 2, 4

_PROJECTIONS = {
    "self_attn.q_proj": (HIDDEN, HIDDEN),
    "self_attn.k_proj": (HIDDEN, HIDDEN),
    "self_attn.v_proj": (HIDDEN, HIDDEN),
    "self_attn.o_proj": (HIDDEN, HIDDEN),
    "mlp.gate_proj": (MLP, HIDDEN),
    "mlp.up_proj": (MLP, HIDDEN),
    "mlp.down_proj": (HIDDEN, MLP),
}


def _config() -> ModelConfig:
    """Tiny Llama config matching the checkpoint written by ``_write_checkpoint``."""
    return ModelConfig(
        model_type="llama",
        hidden_size=HIDDEN,
        vocab_size=VOCAB,
        intermediate_size=MLP,
        num_hidden_layers=LAYERS,
        num_attention_heads=HEADS,
        num_key_value_heads=HEADS,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        max_position_embeddings=64,
    )


def _write_checkpoint(
    path: Path, quantization: str, input_scale: float = 0.02
) -> dict[str, torch.Tensor]:
    """Write a tiny checkpoint: ModelOpt-style for fp8/nvfp4, plain bf16 for "none"."""
    gen = torch.Generator().manual_seed(0)
    tensors: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": torch.randn(VOCAB, HIDDEN, generator=gen).bfloat16(),
        "model.norm.weight": torch.ones(HIDDEN).bfloat16(),
        "lm_head.weight": torch.randn(VOCAB, HIDDEN, generator=gen).bfloat16(),
    }
    for layer in range(LAYERS):
        base = f"model.layers.{layer}"
        tensors[f"{base}.input_layernorm.weight"] = torch.ones(HIDDEN).bfloat16()
        tensors[f"{base}.post_attention_layernorm.weight"] = torch.ones(HIDDEN).bfloat16()
        for stem, (out_f, in_f) in _PROJECTIONS.items():
            name = f"{base}.{stem}"
            if quantization == "none":
                tensors[f"{name}.weight"] = torch.randn(out_f, in_f, generator=gen).bfloat16()
                continue
            tensors[f"{name}.input_scale"] = torch.tensor(input_scale)
            if quantization == "fp8":
                weight = torch.randn(out_f, in_f, generator=gen).to(torch.float8_e4m3fn)
                tensors[f"{name}.weight"] = weight
                tensors[f"{name}.weight_scale"] = torch.tensor(0.01)
            else:
                tensors[f"{name}.weight"] = torch.randint(
                    0, 256, (out_f, in_f // 2), generator=gen, dtype=torch.uint8
                )
                scale = (torch.rand(out_f, in_f // 16, generator=gen) + 0.5).to(torch.float8_e4m3fn)
                tensors[f"{name}.weight_scale"] = scale
                tensors[f"{name}.weight_scale_2"] = torch.tensor(0.003)
    path.mkdir(parents=True, exist_ok=True)
    safetensors_torch.save_file(tensors, str(path / "model.safetensors"))
    (path / "config.json").write_text(json.dumps({"model_type": "llama"}))
    return tensors


_STEMS = (
    ("self_attn.q_proj", "w_q"),
    ("self_attn.k_proj", "w_k"),
    ("self_attn.v_proj", "w_v"),
    ("self_attn.o_proj", "w_o"),
    ("mlp.gate_proj", "w_gate"),
    ("mlp.up_proj", "w_up"),
    ("mlp.down_proj", "w_down"),
)


def _write_awq_scales(checkpoint: Path, scales_dir: Path, skip: str | None = None) -> None:
    """Write a smoothing-scale table keyed to ``checkpoint``'s identity tensors, like the calibration tool."""
    gen = np.random.default_rng(0)
    table = {}
    for layer in range(LAYERS):
        for stem, (_, in_f) in _PROJECTIONS.items():
            name = f"layer.{layer}.{dict(_STEMS)[stem]}"
            if name != skip:
                table[name] = gen.uniform(0.5, 2.0, in_f).astype(np.float16)
    scales_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        scales_dir / "tiny.npz",
        **awq_identity(_open_raw_readers(checkpoint)),
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
    # Round-to-nearest on the grid: at most half a step away, except clamped -8/+7 edges.
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


def test_calibrate_awq_quantizes_original_weights(tmp_path: Path) -> None:
    """INT4-AWQ quantizes the 16-bit checkpoint with the scales keyed to its identity tensors."""
    tensors = _write_checkpoint(tmp_path / "ckpt", "none")
    _write_awq_scales(tmp_path / "ckpt", tmp_path / "scales")
    ctx = calibrate_llama(
        tmp_path / "ckpt", _config(), graph_ops, "int4_awq", awq_scales_dir=tmp_path / "scales"
    )

    assert len(ctx.scales) == LAYERS * len(_PROJECTIONS)
    entry = ctx.scales["layer.1.w_down"]
    assert entry.input_scale is None
    assert isinstance(entry.weight, _AWQWeight)
    source = tensors["model.layers.1.mlp.down_proj.weight"].float().numpy()
    expected = quantize_awq_weight(source, entry.weight.pre_quant_scale)
    assert np.array_equal(entry.weight.packed, expected.packed)
    assert not ctx.disable_dual_gemm_fusion
    assert ctx.builder_optimization_level == 3


def test_calibrate_awq_rejects_scales_of_another_checkpoint(tmp_path: Path) -> None:
    """A scale table calibrated for different weights is never applied."""
    _write_checkpoint(tmp_path / "ckpt", "none")
    _write_awq_scales(tmp_path / "ckpt", tmp_path / "scales")
    other = _write_checkpoint(tmp_path / "other", "none")
    other["model.norm.weight"] = other["model.norm.weight"] * 2
    safetensors_torch.save_file(other, str(tmp_path / "other" / "model.safetensors"))
    with pytest.raises(ValueError, match="no INT4-AWQ scales"):
        calibrate_llama(
            tmp_path / "other", _config(), graph_ops, "int4_awq",
            awq_scales_dir=tmp_path / "scales",
        )


def test_calibrate_awq_rejects_missing_projection_scale(tmp_path: Path) -> None:
    """A projection without a calibrated scale fails the build instead of staying unquantized."""
    _write_checkpoint(tmp_path / "ckpt", "none")
    _write_awq_scales(tmp_path / "ckpt", tmp_path / "scales", skip="layer.1.w_up")
    with pytest.raises(ValueError, match="no layer.1.w_up"):
        calibrate_llama(
            tmp_path / "ckpt", _config(), graph_ops, "int4_awq",
            awq_scales_dir=tmp_path / "scales",
        )


def test_calibration_tool_maps_projection_names() -> None:
    """The tool keys each HF projection module by the weight stem the graph uses."""
    match = calibrate_awq_scales._PROJECTION_NAME.search("model.layers.3.mlp.down_proj")
    assert match is not None and match.group(1) == "3"
    assert calibrate_awq_scales._WEIGHT_STEM[match.group(2)] == "w_down"
    assert calibrate_awq_scales._PROJECTION_NAME.search("model.layers.3.mlp.down_proj.inner") is None


def test_calibration_tool_detects_block_amax_drift() -> None:
    """The tool accepts ModelOpt's plain block amax and rejects a clipped or mismatched one."""
    gen = np.random.default_rng(3)
    weight = gen.standard_normal((4, 256)).astype(np.float32)
    smooth = gen.uniform(0.5, 2.0, 256).astype(np.float16)
    amax = np.abs((weight / smooth.astype(np.float32)).reshape(4, 2, 128)).max(axis=2)
    calibrate_awq_scales._check_block_amax("w", weight, smooth, torch.from_numpy(amax))
    with pytest.raises(RuntimeError, match="recomputed block amax differs"):
        calibrate_awq_scales._check_block_amax("w", weight, smooth, torch.from_numpy(amax * 0.9))


def test_committed_awq_scales_are_complete() -> None:
    """Every committed table names its checkpoint and covers each projection of its shape."""
    from ..quantization import AWQ_IDENTITY_TENSORS, AWQ_SCALES_DIR

    for path in sorted(AWQ_SCALES_DIR.glob("*.npz")):
        with np.load(path, allow_pickle=False) as table:
            for key in ("hf_id", "hf_revision", "algorithm"):
                assert str(table[key]), (path.name, key)
            for key in AWQ_IDENTITY_TENSORS:
                assert table[f"identity.{key}"].size, (path.name, key)
            layers = {name.split(".")[1] for name in table.files if name.startswith("layer.")}
            for layer in layers:
                for _, stem in _STEMS:
                    smooth = table[f"layer.{layer}.{stem}"]
                    assert smooth.dtype == np.float16 and smooth.ndim == 1
                    assert np.isfinite(smooth).all() and np.all(smooth > 0)


@pytest.mark.parametrize("quantization", ["fp8", "nvfp4"])
def test_calibrate_reads_checkpoint_bytes(tmp_path: Path, quantization: str) -> None:
    """Weights and scales come from the checkpoint bytes, with no re-quantization."""
    tensors = _write_checkpoint(tmp_path, quantization)
    ctx = calibrate_llama(tmp_path, _config(), graph_ops, quantization)

    assert len(ctx.scales) == LAYERS * len(_PROJECTIONS)
    assert ctx.builder_optimization_level == 1
    entry = ctx.scales["layer.1.w_down"]
    out_f, in_f = _PROJECTIONS["mlp.down_proj"]
    assert (entry.weight.out_features, entry.weight.in_features) == (out_f, in_f)
    assert entry.input_scale == pytest.approx(0.02)

    source = tensors["model.layers.1.mlp.down_proj.weight"]
    expected = source.view(torch.uint8).numpy()
    assert np.array_equal(entry.weight.packed, expected)  # bit-exact reuse
    if quantization == "fp8":
        assert isinstance(entry.weight, _FP8Weight)
        assert entry.weight.weight_scale == pytest.approx(0.01)
    else:
        assert isinstance(entry.weight, _NVFP4Weight)
        assert entry.weight.global_scale == pytest.approx(0.003)
        assert entry.weight.block_scale_f8.shape == (out_f, in_f // 16)


def test_calibrate_rejects_format_mismatch(tmp_path: Path) -> None:
    """An FP8 checkpoint is refused when nvfp4 is requested."""
    _write_checkpoint(tmp_path, "fp8")
    with pytest.raises(ValueError, match="not a nvfp4 ModelOpt tensor"):
        calibrate_llama(tmp_path, _config(), graph_ops, "nvfp4")


@pytest.mark.parametrize("quantization", ["fp8", "nvfp4"])
@pytest.mark.parametrize("bad_scale", [0.0, -1.0, float("nan")])
def test_calibrate_rejects_invalid_input_scale(
    tmp_path: Path, quantization: str, bad_scale: float
) -> None:
    """A present but invalid input_scale gets its own error, not the format-mismatch one."""
    _write_checkpoint(tmp_path, quantization, input_scale=bad_scale)
    with pytest.raises(ValueError, match="input_scale must be one positive finite value"):
        calibrate_llama(tmp_path, _config(), graph_ops, quantization)


def test_calibrate_rejects_unknown_format(tmp_path: Path) -> None:
    """Formats other than fp8 and nvfp4 are not supported."""
    with pytest.raises(NotImplementedError):
        calibrate_llama(tmp_path, _config(), graph_ops, "int4")


@pytest.mark.parametrize("quantization", ["fp8", "nvfp4"])
def test_load_weights_uses_shape_only_placeholders(tmp_path: Path, quantization: str) -> None:
    """Quantized projections are not loaded in full precision; the rest still are."""
    _write_checkpoint(tmp_path, quantization)
    config = _config()
    ctx = calibrate_llama(tmp_path, config, graph_ops, quantization)
    weights = load_standard_weights(tmp_path, config, precision="fp16", quant_ctx=ctx)

    assert weights["layer.0.w_q"].shape == (HIDDEN, HIDDEN)
    assert weights["layer.0.w_gate"].shape == (HIDDEN, MLP)
    assert weights["layer.0.w_down"].shape == (MLP, HIDDEN)
    # Placeholders must not hold a full-size copy of the weight.
    assert weights["layer.0.w_gate"].strides == (0, 0)
    assert weights["_mlp_size"] == MLP
    assert weights["_kv_attention_size"] == HIDDEN
    # lm_head is not quantized and is still loaded.
    assert weights["w_out"].shape == (HIDDEN, VOCAB)


class _RecordingContext:
    """Records the weight names the matmul helper routes through the context."""

    def __init__(self) -> None:
        """Start with no recorded names."""
        self.names: list[str] = []

    def maybe_quantized_matmul(self, network, lhs, lw, rw, weights, name, dtype=np.float32):
        """Record ``name`` and return the input unchanged."""
        self.names.append(name)
        return lhs


def test_matmul_fn_routes_through_context() -> None:
    """``make_matmul_fn`` hands every projection to the quantization context."""
    ctx = _RecordingContext()
    matmul = graph_blocks.make_matmul_fn(None, np.float16, ctx)
    marker = object()
    assert matmul(marker, 1, 1, None, "layer.0.w_q") is marker
    assert ctx.names == ["layer.0.w_q"]


@pytest.mark.parametrize("value", [None, "none", "fp8", "nvfp4", "int4_awq"])
def test_request_accepts_supported_quantization(value) -> None:
    """``BuildRequest`` carries the quantization setting through ``coerce_request``."""
    legacy = SimpleNamespace(
        model_dir=Path("m"),
        output_path=Path("o"),
        family="llama",
        task="text_generation",
        precision="fp16",
        backend="trt",
        max_sequence_length=None,
        image_height=None,
        image_width=None,
        video_num_frames=None,
        max_batch_size=1,
        tensor_parallel_size=1,
        context_parallel_size=1,
        fp32_layers=(),
        dynamic_kv_cache=False,
        verbose=False,
        graph_transform=None,
        quantization=value,
    )
    request = coerce_request(legacy)
    assert isinstance(request, BuildRequest)
    assert request.quantization == value


@pytest.mark.gpu
@pytest.mark.trt
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_awq_matmul_matches_dequantized_reference(dtype: str) -> None:
    """The INT4 constant, block Dequantize and smoothing compute the dequantized weight's matmul."""
    gen = np.random.default_rng(2)
    rows, n, k = 5, 64, 256
    weight = quantize_awq_weight(
        gen.standard_normal((n, k)).astype(np.float32) * 0.05,
        gen.uniform(0.5, 2.0, k).astype(np.float16),
    )
    ctx = LlamaQuantContext(scales={"w": _LayerScales(None, weight)}, graph_ops=graph_ops)
    trt_dtype = {"fp16": trt.float16, "bf16": trt.bfloat16}[dtype]
    torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype]

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    x = network.add_input("x", trt_dtype, (rows, k))
    out = ctx.maybe_quantized_matmul(network, x, k, n, None, "w")
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
    stream = torch.cuda.Stream()
    assert context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()

    smoothed = activation.float().cpu().numpy() * weight.pre_quant_scale.astype(np.float32)
    expected = smoothed @ _dequantize_awq(weight).T
    tolerance = {"fp16": 2e-2, "bf16": 1e-1}[dtype]
    np.testing.assert_allclose(result.float().cpu().numpy(), expected, rtol=0, atol=tolerance)


@pytest.mark.gpu
@pytest.mark.trt
@pytest.mark.parametrize("quantization", ["fp8", "nvfp4", "int4_awq"])
@pytest.mark.parametrize("decoder_role", ["dual_profile", "decode"])
def test_engine_builds_with_packed_weights(
    tmp_path: Path, quantization: str, decoder_role: str
) -> None:
    """The quantized engine builds and is smaller than the unquantized one.

    Both engines are built from the production weight loader, so the quantized
    one sees the same shape-only placeholders as a real build.
    """
    plain_dir, quant_dir = tmp_path / "plain", tmp_path / "quant"
    _write_checkpoint(plain_dir, "none")
    config = _config()
    config.raw["_decoder_engine_role"] = decoder_role
    if quantization == "int4_awq":
        _write_checkpoint(quant_dir, "none")
        _write_awq_scales(quant_dir, tmp_path / "scales")
        ctx = calibrate_llama(
            quant_dir, config, graph_ops, quantization, awq_scales_dir=tmp_path / "scales"
        )
    else:
        _write_checkpoint(quant_dir, quantization)
        ctx = calibrate_llama(quant_dir, config, graph_ops, quantization)
    assert isinstance(ctx, LlamaQuantContext)

    plain = build_standard_decoder_engine(
        config,
        load_standard_weights(plain_dir, config, precision="fp16"),
        4,
        precision="fp16",
    )
    quantized = build_standard_decoder_engine(
        config,
        load_standard_weights(quant_dir, config, precision="fp16", quant_ctx=ctx),
        4,
        precision="fp16",
        quant_ctx=ctx,
    )

    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(quantized)
    assert engine is not None
    if decoder_role == "decode" and quantization == "nvfp4":
        assert engine.num_aux_streams == 0
    names = {engine.get_tensor_name(i) for i in range(engine.num_io_tensors)}
    assert {"token_id", "logits", "present_k_0"} <= names
    # 7 projections x 2 layers: FP8 halves and FP4/INT4 quarter the fp16 projection bytes,
    # so a plan that stored 16-bit weights again would miss these bounds. INT4-AWQ builds
    # at optimization level 3, whose extra kernels weigh more in a plan this small.
    bound = {"fp8": 0.6, "nvfp4": 0.4, "int4_awq": 0.6}[quantization]
    assert len(quantized) < bound * len(plain), (len(quantized), len(plain))
