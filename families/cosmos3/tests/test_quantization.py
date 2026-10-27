# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FP8 lowering of the official ModelOpt FP8 Cosmos3-Nano checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from families.cosmos3 import checkpoint_mapper, model  # noqa: E402
from families.cosmos3 import quantization as quantization_module  # noqa: E402
from families.cosmos3 import transformer_builder  # noqa: E402
from families.cosmos3.model_config import Cosmos3NanoConfig  # noqa: E402
from families.cosmos3.quantization import (  # noqa: E402
    FP8_E4M3_MAX,
    LAYER_PROJECTIONS,
    configure_build,
    is_fp8_checkpoint,
    is_fp8_state,
    load_fp8_projections,
    projection_names,
)

from .test_model import RecordingWriter, _checkpoint, _request  # noqa: E402

# A tiny profile: 4 heads of size 64 (TensorRT 11.1 cannot build a 2-head tiny attention).
TINY = Cosmos3NanoConfig(
    hidden_size=256,
    intermediate_size=512,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=64,
    vocab_size=64,
    video_height=64,
    video_width=64,
    video_num_frames=5,
    max_text_seq_len=16,
)
FP8_SHARDS = tuple(
    f"diffusion_pytorch_model-{index:05d}-of-00004.safetensors" for index in range(1, 5)
)


def _projection_shapes(profile: Cosmos3NanoConfig) -> dict[str, tuple[int, int]]:
    """Checkpoint ``[out, in]`` shape of every projection stem."""
    hidden, ffn = profile.hidden_size, profile.intermediate_size
    q_out = profile.num_attention_heads * profile.head_dim
    kv_out = profile.num_key_value_heads * profile.head_dim
    shapes = {}
    for stem in LAYER_PROJECTIONS:
        leaf = stem.split(".")[-1]
        if leaf in ("to_q", "add_q_proj"):
            shapes[stem] = (q_out, hidden)
        elif leaf in ("to_k", "to_v", "add_k_proj", "add_v_proj"):
            shapes[stem] = (kv_out, hidden)
        elif leaf in ("to_out", "to_add_out"):
            shapes[stem] = (hidden, q_out)
        elif leaf == "down_proj":
            shapes[stem] = (hidden, ffn)
        else:
            shapes[stem] = (ffn, hidden)
    return shapes


def _fp8_tensor(weight: np.ndarray) -> tuple["torch.Tensor", float]:
    """The weight as a float8_e4m3 tensor with its ModelOpt scale ``amax / 448``."""
    scale = float(np.abs(weight).max()) / FP8_E4M3_MAX
    packed = (weight / scale).astype(ml_dtypes.float8_e4m3fn).view(np.uint8)
    return torch.from_numpy(np.ascontiguousarray(packed)).view(torch.float8_e4m3fn), scale


def _state(profile: Cosmos3NanoConfig, *, fp8: bool, seed: int = 0) -> dict[str, "torch.Tensor"]:
    """A tiny transformer state dict, BF16 or in the ModelOpt FP8 layout."""
    rng = np.random.default_rng(seed)
    hidden, head_dim = profile.hidden_size, profile.head_dim

    def tensor(*shape, scale=0.05):
        return torch.from_numpy((rng.standard_normal(shape) * scale).astype(np.float32)).bfloat16()

    def ones(size):
        return torch.ones(size, dtype=torch.bfloat16)

    state = {
        "embed_tokens.weight": tensor(profile.vocab_size, hidden),
        "proj_in.weight": tensor(hidden, profile.patch_latent_dim),
        "proj_in.bias": tensor(hidden),
        "proj_out.weight": tensor(profile.patch_latent_dim, hidden),
        "proj_out.bias": tensor(profile.patch_latent_dim),
        "time_embedder.linear_1.weight": tensor(hidden, profile.timestep_dim),
        "time_embedder.linear_1.bias": tensor(hidden),
        "time_embedder.linear_2.weight": tensor(hidden, hidden),
        "time_embedder.linear_2.bias": tensor(hidden),
        "norm.weight": ones(hidden),
        "norm_moe_gen.weight": ones(hidden),
    }
    shapes = _projection_shapes(profile)
    for layer in range(profile.num_hidden_layers):
        prefix = f"layers.{layer}"
        for norm in (
            "input_layernorm",
            "input_layernorm_moe_gen",
            "post_attention_layernorm",
            "post_attention_layernorm_moe_gen",
        ):
            state[f"{prefix}.{norm}.weight"] = ones(hidden)
        for norm in ("norm_q", "norm_k", "norm_added_q", "norm_added_k"):
            state[f"{prefix}.self_attn.{norm}.weight"] = ones(head_dim)
        for stem in LAYER_PROJECTIONS:
            name = f"{prefix}.{stem}"
            weight = tensor(*shapes[stem])
            if not fp8:
                state[f"{name}.weight"] = weight
                continue
            q_weight, weight_scale = _fp8_tensor(weight.float().numpy())
            state[f"{name}.weight"] = q_weight
            state[f"{name}.weight_scale"] = torch.tensor(weight_scale)
            state[f"{name}.input_scale"] = torch.tensor(0.05)
            # The official file also keeps ModelOpt's quantizer buffers; they are ignored.
            state[f"{name}.input_quantizer._amax"] = torch.tensor(0.05 * FP8_E4M3_MAX)
    return state


def _write_index(transformer: Path, shards: tuple[str, ...]) -> None:
    transformer.mkdir(parents=True, exist_ok=True)
    weight_map = {f"tensor.{index}": name for index, name in enumerate(shards)}
    weight_map["layers.0.self_attn.to_q.weight_scale"] = shards[-1]
    (transformer / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8"
    )


def test_projection_names_cover_both_streams_of_every_layer() -> None:
    """14 projections per layer: 7 for the understanding stream and 7 for the generation stream."""
    names = projection_names(36)
    assert len(names) == 36 * 14 == 504
    assert "layers.35.self_attn.to_add_out" in names
    assert "layers.0.mlp_moe_gen.down_proj" in names


def test_state_with_weight_scales_is_fp8() -> None:
    assert is_fp8_state(_state(TINY, fp8=True))
    assert not is_fp8_state(_state(TINY, fp8=False))


def test_fp8_checkpoint_is_detected_from_the_index_only(tmp_path: Path) -> None:
    """Detection reads the index, so a build refuses the wrong checkpoint before loading 16 GB."""
    _write_index(tmp_path / "fp8", FP8_SHARDS)
    assert is_fp8_checkpoint(tmp_path / "fp8")
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layers.0.self_attn.to_q.weight": "a.safetensors"}})
    )
    assert not is_fp8_checkpoint(plain)
    assert not is_fp8_checkpoint(tmp_path / "missing")


def test_four_fp8_shards_are_accepted_and_other_sets_are_rejected(tmp_path: Path) -> None:
    fp8 = tmp_path / "fp8"
    _write_index(fp8, FP8_SHARDS)
    for name in FP8_SHARDS:
        (fp8 / name).write_bytes(b"shard")
    assert [p.name for p in checkpoint_mapper.transformer_safetensor_paths(fp8)] == list(FP8_SHARDS)

    odd = tmp_path / "odd"
    _write_index(odd, FP8_SHARDS[:3])
    with pytest.raises(ValueError, match="four official FP8 shards"):
        checkpoint_mapper.transformer_safetensor_paths(odd)


def test_load_fp8_projections_reuses_the_checkpoint_bytes() -> None:
    """Weights and scales come from the file as stored; nothing is re-quantized."""
    state = _state(TINY, fp8=True)
    ctx = load_fp8_projections(state, TINY.num_hidden_layers)
    assert len(ctx.weights) == TINY.num_hidden_layers * 14
    name = "layers.1.mlp_moe_gen.up_proj"
    stored = state[f"{name}.weight"].view(torch.uint8).numpy()
    assert np.array_equal(ctx.weights[name].packed, stored)
    assert ctx.weights[name].weight_scale == pytest.approx(float(state[f"{name}.weight_scale"]))
    assert ctx.input_scales[name] == pytest.approx(0.05)
    assert (ctx.weights[name].out_features, ctx.weights[name].in_features) == stored.shape
    assert ctx.has_fp8_weights and ctx.should_quantize(name)


def test_dequantized_weight_matches_the_fp8_values() -> None:
    state = _state(TINY, fp8=True)
    ctx = load_fp8_projections(state, TINY.num_hidden_layers)
    name = "layers.0.self_attn.to_q"
    expected = state[f"{name}.weight"].float().numpy() * float(state[f"{name}.weight_scale"])
    np.testing.assert_allclose(ctx.dequantized_weight(name), expected, rtol=1e-6)


def test_bf16_state_has_no_fp8_projections() -> None:
    with pytest.raises(ValueError, match="no FP8 projections"):
        load_fp8_projections(_state(TINY, fp8=False), TINY.num_hidden_layers)


@pytest.mark.parametrize("missing", ["weight_scale", "input_scale"])
def test_projection_with_fp8_bytes_but_a_missing_scale_is_rejected(missing: str) -> None:
    state = _state(TINY, fp8=True)
    del state[f"layers.0.self_attn.to_k.{missing}"]
    with pytest.raises(ValueError, match="layers.0.self_attn.to_k is FP8 but lacks"):
        load_fp8_projections(state, TINY.num_hidden_layers)


def test_scales_on_a_bf16_weight_are_rejected() -> None:
    state = _state(TINY, fp8=True)
    state["layers.0.self_attn.to_k.weight"] = torch.zeros(
        TINY.num_key_value_heads * TINY.head_dim, TINY.hidden_size, dtype=torch.bfloat16
    )
    with pytest.raises(ValueError, match="not FP8 but has"):
        load_fp8_projections(state, TINY.num_hidden_layers)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_scales_are_rejected(bad: float) -> None:
    state = _state(TINY, fp8=True)
    state["layers.0.mlp.gate_proj.input_scale"] = torch.tensor(bad)
    with pytest.raises(ValueError, match="finite positive scale"):
        load_fp8_projections(state, TINY.num_hidden_layers)


def test_build_options_set_the_route_only_when_tensorrt_lists_the_knob() -> None:
    class Config:
        def __init__(self, routes: str) -> None:
            self.all_build_routes = routes
            self.build_route = ""
            self.profiling_verbosity = None

    listed, unlisted = Config('{"peep:match_dual_gemm": {}}'), Config("")
    configure_build(listed)
    configure_build(unlisted)
    assert listed.build_route == "-peep:match_dual_gemm=off"
    assert unlisted.build_route == ""
    assert listed.profiling_verbosity is not None and unlisted.profiling_verbosity is not None


def _record_denoiser(monkeypatch: pytest.MonkeyPatch) -> list:
    calls = []
    monkeypatch.setattr(
        model, "_build_denoiser", lambda path, **options: calls.append(options) or b"denoiser"
    )
    monkeypatch.setattr(model, "_load_vae_weights", lambda path: {"weight": object()})
    monkeypatch.setattr(model, "_build_vae", lambda weights, **options: b"vae")
    return calls


def _fp8_checkpoint(root: Path) -> Path:
    model_dir = _checkpoint(root)
    transformer = model_dir / "transformer"
    index = transformer / "diffusion_pytorch_model.safetensors.index.json"
    for old in transformer.glob("*of-00007.safetensors"):
        old.unlink()
    _write_index(transformer, FP8_SHARDS)
    for name in FP8_SHARDS:
        (transformer / name).write_bytes(b"shard")
    assert index.is_file()
    return model_dir


def test_build_passes_fp8_to_the_denoiser_for_the_fp8_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _record_denoiser(monkeypatch)
    model_dir = _fp8_checkpoint(tmp_path / "checkpoint")
    model.build(_request(model_dir, quantization="fp8", context_parallel_size=1), RecordingWriter())
    assert calls == [{"context_parallel_size": 1, "quantization": "fp8", "verbose": False}]


def test_build_refuses_the_fp8_checkpoint_without_the_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _record_denoiser(monkeypatch)
    model_dir = _fp8_checkpoint(tmp_path / "checkpoint")
    with pytest.raises(ValueError, match="build it with --quantization fp8"):
        model.build(_request(model_dir, context_parallel_size=1), RecordingWriter())


def test_build_refuses_fp8_for_the_bf16_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _record_denoiser(monkeypatch)
    model_dir = _checkpoint(tmp_path / "checkpoint")
    with pytest.raises(ValueError, match="revision fp8"):
        model.build(
            _request(model_dir, quantization="fp8", context_parallel_size=1), RecordingWriter()
        )


def test_build_rejects_fp8_with_context_parallelism(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="context_parallel_size=1 only"):
        model.build(
            _request(tmp_path / "missing", quantization="fp8", context_parallel_size=2),
            RecordingWriter(),
        )


def test_build_rejects_other_quantization_formats(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not support quantization"):
        model.build(_request(tmp_path / "missing", quantization="nvfp4"), RecordingWriter())


def test_builder_refuses_an_fp8_state_without_the_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        transformer_builder, "load_transformer_state_dict", lambda path: _state(TINY, fp8=True)
    )
    with pytest.raises(ValueError, match="build it with --quantization fp8"):
        transformer_builder.build_cosmos3_transformer_engine("unused", profile=TINY)


def test_builder_refuses_fp8_for_a_bf16_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        transformer_builder, "load_transformer_state_dict", lambda path: _state(TINY, fp8=False)
    )
    with pytest.raises(ValueError, match="no FP8 projections"):
        transformer_builder.build_cosmos3_transformer_engine(
            "unused", profile=TINY, quantization="fp8"
        )


@pytest.mark.gpu
@pytest.mark.trt
def test_fp8_engine_builds_with_fp8_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tiny FP8 denoiser builds, and its plan is smaller than the BF16 plan of the same weights."""
    fp8_state = _state(TINY, fp8=True)
    # The BF16 reference holds the same values the FP8 file encodes.
    bf16_state = {k: v for k, v in fp8_state.items() if not k.endswith(("_scale", "_amax"))}
    for name in projection_names(TINY.num_hidden_layers):
        scale = float(fp8_state[f"{name}.weight_scale"])
        bf16_state[f"{name}.weight"] = (fp8_state[f"{name}.weight"].float() * scale).bfloat16()

    def build(state, **options) -> bytes:
        monkeypatch.setattr(transformer_builder, "load_transformer_state_dict", lambda path: state)
        return transformer_builder.build_cosmos3_transformer_engine(
            "unused", profile=TINY, **options
        )

    plain = build(bf16_state)
    quantized = build(fp8_state, quantization="fp8")
    assert len(quantized) < len(plain)
    assert quantization_module.is_fp8_state(fp8_state)


def test_attention_output_projections_stay_bf16_with_dequantized_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """to_out and to_add_out use the dequantized checkpoint weight; every other projection is FP8."""
    state = _state(TINY, fp8=True)
    ctx = load_fp8_projections(state, TINY.num_hidden_layers)
    for stem in ("self_attn.to_out", "self_attn.to_add_out"):
        assert not ctx.should_quantize(f"layers.0.{stem}")
    quantized = [
        name for name in projection_names(TINY.num_hidden_layers) if ctx.should_quantize(name)
    ]
    assert len(quantized) == TINY.num_hidden_layers * 12

    seen = []
    monkeypatch.setattr(
        transformer_builder.op,
        "linear",
        lambda network, x, weight, bias=None, **kw: seen.append(np.asarray(weight)) or x,
    )
    name = "layers.0.self_attn.to_out"
    assert transformer_builder._linear(None, "x", state, name, ctx) == "x"
    expected = state[f"{name}.weight"].float().numpy() * float(state[f"{name}.weight_scale"])
    np.testing.assert_allclose(seen[0], expected, rtol=1e-6)
