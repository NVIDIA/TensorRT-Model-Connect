# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Eagle VLM FP8 quantization: checkpoint reading, weight loading, refusals, engine build."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")
torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

from families.eagle_vlm import graph_ops, model  # noqa: E402
from families.eagle_vlm.config import ModelConfig  # noqa: E402
from families.eagle_vlm.quantization import (  # noqa: E402
    EagleQuantContext,
    calibrate_eagle,
    configure_fp8_build,
    is_quantized_checkpoint,
)

# GQA with 4 query and 2 key/value heads (4 heads: TensorRT 11.1 can segfault on 2 heads of 64).
HIDDEN, MLP, VOCAB, LAYERS, HEADS, KV_HEADS = 128, 256, 64, 2, 4, 2
HEAD_DIM = HIDDEN // HEADS

_PROJECTIONS = {
    "self_attn.q_proj": (HEADS * HEAD_DIM, HIDDEN),
    "self_attn.k_proj": (KV_HEADS * HEAD_DIM, HIDDEN),
    "self_attn.v_proj": (KV_HEADS * HEAD_DIM, HIDDEN),
    "self_attn.o_proj": (HIDDEN, HEADS * HEAD_DIM),
    "mlp.gate_proj": (MLP, HIDDEN),
    "mlp.up_proj": (MLP, HIDDEN),
    "mlp.down_proj": (HIDDEN, MLP),
}


def _config_json(rerank: bool) -> dict:
    return {
        "model_type": "llama_nemotron_vl_rerank" if rerank else "llama_nemotron_vl",
        "architectures": [
            "LlamaNemotronVLForSequenceClassification" if rerank else "LlamaNemotronVLModel"
        ],
        "pooling": "avg",
        "vision_config": {"image_size": 56, "patch_size": 14},
        "llm_config": {
            "model_type": "llama_bidirec",
            "hidden_size": HIDDEN,
            "intermediate_size": MLP,
            "num_hidden_layers": LAYERS,
            "num_attention_heads": HEADS,
            "num_key_value_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "vocab_size": VOCAB,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "max_position_embeddings": 128,
        },
    }


def _write_checkpoint(
    path: Path,
    quantization: str,
    *,
    rerank: bool = False,
    input_scale: float = 0.02,
    prefix: str = "language_model",
    exclude: tuple[str, ...] = ("mlp1*", "vision_model*", "language_model.lm_head*"),
    skip_scales_for: tuple[str, ...] = (),
) -> dict[str, torch.Tensor]:
    """Write a tiny checkpoint: ModelOpt-style for fp8, plain bf16 for "none".

    Projections named in ``skip_scales_for`` are stored in bf16 without scales.
    """
    gen = torch.Generator().manual_seed(0)
    tensors: dict[str, torch.Tensor] = {
        f"{prefix}.embed_tokens.weight": torch.randn(VOCAB, HIDDEN, generator=gen).bfloat16(),
        f"{prefix}.norm.weight": torch.ones(HIDDEN).bfloat16(),
        "mlp1.1.weight": torch.randn(HIDDEN, HIDDEN, generator=gen).bfloat16(),
    }
    if rerank:
        tensors["score.weight"] = torch.randn(1, HIDDEN, generator=gen)
    for layer in range(LAYERS):
        base = f"{prefix}.layers.{layer}"
        tensors[f"{base}.input_layernorm.weight"] = torch.ones(HIDDEN).bfloat16()
        tensors[f"{base}.post_attention_layernorm.weight"] = torch.ones(HIDDEN).bfloat16()
        for stem, (out_f, in_f) in _PROJECTIONS.items():
            name = f"{base}.{stem}"
            if quantization == "none" or name in skip_scales_for:
                tensors[f"{name}.weight"] = (
                    torch.randn(out_f, in_f, generator=gen) * 0.05
                ).bfloat16()
                continue
            tensors[f"{name}.input_scale"] = torch.tensor(input_scale)
            weight = (torch.randn(out_f, in_f, generator=gen) * 20).to(torch.float8_e4m3fn)
            tensors[f"{name}.weight"] = weight
            tensors[f"{name}.weight_scale"] = torch.tensor(0.0025)
    path.mkdir(parents=True, exist_ok=True)
    safetensors_torch.save_file(tensors, str(path / "model.safetensors"))
    (path / "config.json").write_text(json.dumps(_config_json(rerank)))
    if quantization != "none":
        quant = {"quant_algo": "FP8", "exclude_modules": list(exclude)}
        (path / "hf_quant_config.json").write_text(json.dumps({"quantization": quant}))
    return tensors


def _calibrate(path: Path, quantization: str = "fp8") -> EagleQuantContext:
    return calibrate_eagle(path, graph_ops, quantization, ModelConfig.from_dir(path))


@pytest.mark.parametrize("prefix", ["language_model", "model.language_model"])
def test_calibrate_reads_checkpoint_bytes(tmp_path: Path, prefix: str) -> None:
    """Weights and scales come from the checkpoint bytes, with no re-quantization."""
    tensors = _write_checkpoint(tmp_path, "fp8", prefix=prefix)
    ctx = _calibrate(tmp_path)

    assert len(ctx.scales) == LAYERS * (len(_PROJECTIONS) - 1)  # o_proj stays fp16
    entry = ctx.scales["layer.1.w_down"]
    assert (entry.weight.out_features, entry.weight.in_features) == _PROJECTIONS["mlp.down_proj"]
    assert entry.input_scale == pytest.approx(0.02)
    assert entry.weight.weight_scale == pytest.approx(0.0025)
    source = tensors[f"{prefix}.layers.1.mlp.down_proj.weight"]
    assert np.array_equal(entry.weight.packed, source.view(torch.uint8).numpy())  # bit-exact


def test_calibrate_rejects_unquantized_checkpoint(tmp_path: Path) -> None:
    """A plain full-precision checkpoint cannot pass as quantized."""
    _write_checkpoint(tmp_path, "none")
    with pytest.raises(ValueError, match="not listed in exclude_modules"):
        _calibrate(tmp_path)


def test_calibrate_allows_excluded_projection(tmp_path: Path) -> None:
    """A projection ModelOpt left unquantized is accepted only when exclude_modules lists it."""
    name = "language_model.layers.0.mlp.down_proj"
    _write_checkpoint(tmp_path, "fp8", skip_scales_for=(name,), exclude=(name, "mlp1*"))
    ctx = _calibrate(tmp_path)
    assert "layer.0.w_down" not in ctx.scales
    assert "layer.1.w_down" in ctx.scales

    other = tmp_path / "unlisted"
    _write_checkpoint(other, "fp8", skip_scales_for=(name,))
    with pytest.raises(ValueError, match="not listed in exclude_modules"):
        _calibrate(other)


def test_calibrate_matches_exclude_modules_as_glob_patterns(tmp_path: Path) -> None:
    """ModelOpt writes exclude_modules as glob patterns, not only exact module names."""
    name = "language_model.layers.0.mlp.down_proj"
    _write_checkpoint(
        tmp_path, "fp8", skip_scales_for=(name,), exclude=("language_model.layers.*.mlp.down_proj",)
    )
    assert "layer.0.w_down" not in _calibrate(tmp_path).scales


@pytest.mark.parametrize("bad_scale", [0.0, -1.0, float("nan")])
def test_calibrate_rejects_invalid_input_scale(tmp_path: Path, bad_scale: float) -> None:
    """A present but invalid input_scale gets its own error."""
    _write_checkpoint(tmp_path, "fp8", input_scale=bad_scale)
    with pytest.raises(ValueError, match="input_scale must be one positive finite value"):
        _calibrate(tmp_path)


def test_calibrate_rejects_unknown_format(tmp_path: Path) -> None:
    """Only fp8 is supported; there is no NVFP4 ModelOpt checkpoint for this family."""
    _write_checkpoint(tmp_path, "fp8")
    with pytest.raises(NotImplementedError):
        _calibrate(tmp_path, "nvfp4")


def test_is_quantized_checkpoint(tmp_path: Path) -> None:
    """Checkpoints with weight_scale tensors are detected without loading torch tensors."""
    _write_checkpoint(tmp_path / "q", "fp8")
    _write_checkpoint(tmp_path / "p", "none")
    assert is_quantized_checkpoint(tmp_path / "q")
    assert not is_quantized_checkpoint(tmp_path / "p")


def test_load_weights_skips_quantized_projections(tmp_path: Path) -> None:
    """Quantized projections become shape-only placeholders; everything else loads."""
    _write_checkpoint(tmp_path, "fp8", rerank=True)
    config = ModelConfig.from_dir(tmp_path)
    ctx = calibrate_eagle(tmp_path, graph_ops, "fp8", config)
    weights = model._EagleModel().load_weights(str(tmp_path), config, quant_ctx=ctx)
    placeholder = weights["layer.0.w_q"]
    assert placeholder.shape == (HIDDEN, HEADS * HEAD_DIM)
    assert placeholder.strides == (0, 0)  # broadcast zero, no storage
    assert weights["layer.1.w_gate"].shape == (HIDDEN, MLP)
    # the attention output projection stays fp16: the engine gets the dequantized checkpoint weight
    assert weights["layer.0.w_o"].shape == (HEADS * HEAD_DIM, HIDDEN)
    assert weights["layer.0.w_o"].strides != (0, 0)
    assert weights["embedding"].shape == (VOCAB, HIDDEN)
    assert weights["score_weight"].shape == (HIDDEN, 1)
    assert weights["_attention_size"] == HEADS * HEAD_DIM
    assert weights["_mlp_size"] == MLP


@pytest.mark.parametrize("rerank", [False, True])
def test_quantized_engine_builds(tmp_path: Path, rerank: bool) -> None:
    """The text engine builds with FP8 projections and is smaller than the fp16 engine."""
    if not torch.cuda.is_available():
        pytest.skip("engine build needs a GPU")
    plain_dir, fp8_dir = tmp_path / "plain", tmp_path / "fp8"
    _write_checkpoint(plain_dir, "none", rerank=rerank)
    _write_checkpoint(fp8_dir, "fp8", rerank=rerank)
    config = ModelConfig.from_dir(fp8_dir)
    plugin = model._EagleModel()
    plain = plugin.build_engine(
        config, plugin.load_weights(str(plain_dir), config), 32, precision="fp16"
    )
    ctx = calibrate_eagle(fp8_dir, graph_ops, "fp8", config)
    fp8 = plugin.build_engine(
        config,
        plugin.load_weights(str(fp8_dir), config, quant_ctx=ctx),
        32,
        precision="fp16",
        quant_ctx=ctx,
    )
    engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(fp8)
    assert engine.num_io_tensors > 0
    assert len(fp8) < len(plain)


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
    ctx = EagleQuantContext(scales={}, graph_ops=graph_ops)
    config = _FakeBuilderConfig(routes)
    configure_fp8_build(config, ctx)
    assert config.build_route == expected
    assert config.profiling_verbosity is not None
    unquantized = _FakeBuilderConfig(routes)
    configure_fp8_build(unquantized, None)
    assert unquantized.build_route == "" and unquantized.profiling_verbosity is None
