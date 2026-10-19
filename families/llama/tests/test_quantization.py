# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Llama FP8 / NVFP4 quantization: checkpoint reading, weight loading, engine build."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")
torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

from .. import graph_blocks, graph_ops  # noqa: E402
from ..build_request import BuildRequest, coerce_request  # noqa: E402
from ..checkpoint_mapper import load_standard_weights  # noqa: E402
from ..config import ModelConfig  # noqa: E402
from ..quantization import (  # noqa: E402
    LlamaQuantContext,
    _FP8Weight,
    _NVFP4Weight,
    calibrate_llama,
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


@pytest.mark.parametrize("quantization", ["fp8", "nvfp4"])
def test_calibrate_reads_checkpoint_bytes(tmp_path: Path, quantization: str) -> None:
    """Weights and scales come from the checkpoint bytes, with no re-quantization."""
    tensors = _write_checkpoint(tmp_path, quantization)
    ctx = calibrate_llama(tmp_path, _config(), graph_ops, quantization)

    assert len(ctx.scales) == LAYERS * len(_PROJECTIONS)
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


@pytest.mark.parametrize("value", [None, "none", "fp8", "nvfp4"])
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
@pytest.mark.parametrize("quantization", ["fp8", "nvfp4"])
def test_engine_builds_with_packed_weights(tmp_path: Path, quantization: str) -> None:
    """The quantized engine builds and is smaller than the unquantized one.

    Both engines are built from the production weight loader, so the quantized
    one sees the same shape-only placeholders as a real build.
    """
    plain_dir, quant_dir = tmp_path / "plain", tmp_path / "quant"
    _write_checkpoint(plain_dir, "none")
    _write_checkpoint(quant_dir, quantization)
    config = _config()
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
    names = {engine.get_tensor_name(i) for i in range(engine.num_io_tensors)}
    assert {"token_id", "logits", "present_k_0"} <= names
    # 7 projections x 2 layers: FP8 halves and FP4 quarters the fp16 projection bytes,
    # so a plan that stored 16-bit weights again would miss these bounds.
    bound = {"fp8": 0.6, "nvfp4": 0.4}[quantization]
    assert len(quantized) < bound * len(plain), (len(quantized), len(plain))
