# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the MiniMax-Music3 FP8 path: scales, weight quantization, wiring, calibration tool."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("ml_dtypes")

from families.minimax_music3 import calibrate_fp8_scales as cal  # noqa: E402
from families.minimax_music3 import model, pipeline_spec  # noqa: E402
from families.minimax_music3 import quantization as quant  # noqa: E402

SCALES = json.loads((Path(quant.__file__).parent / quant.ACTIVATION_SCALES_FILENAME).read_text())


def _decode(packed, scale):
    import ml_dtypes

    return packed.view(ml_dtypes.float8_e4m3fn).astype(np.float32) * scale


# --- weight quantization ------------------------------------------------------------------


def test_weight_scale_is_amax_over_448_and_round_trips() -> None:
    rng = np.random.default_rng(0)
    weight = rng.standard_normal((64, 32)).astype(np.float32)
    packed, scale = quant.quantize_fp8_weight(weight)
    assert packed.dtype == np.uint8 and packed.shape == weight.shape
    assert scale == pytest.approx(np.abs(weight).max() / 448.0)
    error = np.abs(_decode(packed, scale) - weight)
    # e4m3 keeps 3 mantissa bits: relative error at most 2**-4 above the subnormal range
    assert np.all(error <= np.maximum(np.abs(weight) * 2.0**-4, scale * 2.0**-9) + 1e-7)


def test_all_zero_weight_is_rejected() -> None:
    with pytest.raises(ValueError, match="amax"):
        quant.quantize_fp8_weight(np.zeros((4, 4), dtype=np.float32))


# --- scale file and keys ------------------------------------------------------------------


@pytest.mark.parametrize("component", sorted(quant.ENGINE_PROJECTIONS))
def test_scale_file_covers_every_projection_of_each_component(component) -> None:
    config = quant._KEY_PARTS[component][0]
    keys = [key for key in SCALES if key.startswith(f"minimax_music3-{config}-n")]
    assert len(keys) == 1, keys
    entry = SCALES[keys[0]]
    assert sorted(entry) == sorted(quant.ENGINE_PROJECTIONS[component]())
    for name, value in entry.items():
        assert value is None or (np.isfinite(value) and value > 0), name


def test_projection_counts() -> None:
    assert len(quant.ENGINE_PROJECTIONS["language_model"]()) == 36 * 6
    assert len(quant.ENGINE_PROJECTIONS["rvq_depth_decoder"]()) == 4 * 7
    assert len(quant.ENGINE_PROJECTIONS["transformer"]()) == 36 * 2
    # the attention output projection of the language model stays bf16 (TensorRT 11.4)
    assert not any(name.endswith(".w_o") for name in quant.ENGINE_PROJECTIONS["language_model"]())


def test_scale_key_has_model_config_and_weight_fingerprint() -> None:
    tensors = {"model.norm.weight": np.full(4096, 2.0, dtype=np.float32)}
    assert (
        quant.scale_key("language_model", tensors)
        == "minimax_music3-lm-h4096-l36-i12288-n8192.0000"
    )
    tensors["model.norm.weight"][0] = 3.0
    assert quant.scale_key("language_model", tensors).endswith("-n8193.0000")


def test_unknown_checkpoint_names_the_calibration_tool() -> None:
    tensors = {"norm.weight": np.ones(4096, dtype=np.float32)}
    with pytest.raises(NotImplementedError, match="calibrate_fp8_scales.py"):
        quant.load_activation_scales("rvq_depth_decoder", tensors)


def test_missing_and_extra_scales_are_errors(monkeypatch, tmp_path) -> None:
    tensors = {"norm.weight": np.ones(4096, dtype=np.float32)}
    key = quant.scale_key("rvq_depth_decoder", tensors)
    names = quant.ENGINE_PROJECTIONS["rvq_depth_decoder"]()
    path = tmp_path / quant.ACTIVATION_SCALES_FILENAME
    monkeypatch.setattr(quant, "__file__", str(tmp_path / "quantization.py"))

    path.write_text(json.dumps({key: dict.fromkeys(names[1:], 0.01)}))
    with pytest.raises(ValueError, match="no activation scale"):
        quant.load_activation_scales("rvq_depth_decoder", tensors)
    path.write_text(json.dumps({key: {**dict.fromkeys(names, 0.01), "layers.0.projection": 0.01}}))
    with pytest.raises(ValueError, match="not quantized"):
        quant.load_activation_scales("rvq_depth_decoder", tensors)
    path.write_text(json.dumps({key: {**dict.fromkeys(names, 0.01), names[0]: None}}))
    assert quant.load_activation_scales("rvq_depth_decoder", tensors)[names[0]] is None


def test_null_scale_leaves_the_projection_unquantized() -> None:
    block = quant.BlockQuant(scales={"layers.0.gate_proj": None})
    assert block.linear(None, None, np.ones((2, 2), dtype=np.float32), "layers.0.gate_proj") is None
    assert block.linear(None, None, np.ones((2, 2), dtype=np.float32), "layers.0.other") is None
    assert block.quantized == []


def test_quantized_checkpoint_is_rejected() -> None:
    with pytest.raises(ValueError, match="already quantized"):
        quant.reject_quantized_checkpoint({"layers.0.gate_proj.weight_scale": np.ones(1)})


# --- model wiring -------------------------------------------------------------------------


def _request(**overrides):
    values = dict(
        task="audio_generation",
        precision="bf16",
        backend="trt",
        tensor_parallel_size=1,
        context_parallel_size=1,
        max_batch_size=1,
        quantization=None,
        fp32_layers=None,
        mtp_seq_len=None,
        dynamic_kv_cache=False,
        image_height=None,
        image_width=None,
        video_num_frames=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_build_refuses_quantizations_other_than_fp8() -> None:
    with pytest.raises(NotImplementedError, match="fp8 only"):
        model.build(_request(quantization="nvfp4"), None)


def test_condition_encoder_and_vocoder_stay_unquantized() -> None:
    for engine in ("condition_encoder", "vocoder"):
        assert model._quant_context(engine, {}, "fp8") is None
    assert model._quant_context("dit", {}, None) is None


# --- calibration tool ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("module", "expected"),
    [
        ("language_model.model.layers.3.self_attn.q_proj", ("language_model", "layer.3.w_q")),
        ("language_model.model.layers.35.mlp.down_proj", ("language_model", "layer.35.w_down")),
        ("language_model.model.layers.0.self_attn.o_proj", ("language_model", "layer.0.w_o")),
        ("rvq_depth_decoder.layers.2.attn.to_out", ("rvq_depth_decoder", "layers.2.attn.to_out")),
        ("rvq_depth_decoder.layers.1.gate_proj", ("rvq_depth_decoder", "layers.1.gate_proj")),
        (
            "transformer.transformer_blocks.7.attn.to_out.0",
            ("transformer", "transformer_blocks.7.attn.to_out.0"),
        ),
        ("transformer.transformer_blocks.7.ff_in", ("transformer", "transformer_blocks.7.ff_in")),
        ("transformer.proj_in", None),
        ("language_model.lm_head", None),
        ("rvq_depth_decoder.projection", None),
        ("rvq_depth_decoder.audio_heads.0", None),
    ],
)
def test_calibration_name_map(module, expected) -> None:
    assert cal.engine_projection_name(module) == expected


def test_every_quantized_projection_has_a_calibration_source() -> None:
    for component, names in quant.ENGINE_PROJECTIONS.items():
        mapped = set()
        for name in names():
            layer = name.split(".")[1]
            if component == "language_model":
                stem = {v: k for k, v in cal._LM_STEMS.items()}[name.split(".", 2)[2]]
                module = f"language_model.model.layers.{layer}.{stem}"
            elif component == "rvq_depth_decoder":
                module = f"rvq_depth_decoder.{name}"
            else:
                module = f"transformer.{name}"
            mapped.add(cal.engine_projection_name(module))
        assert mapped == {(component, name) for name in names()}


def test_outlier_scales_are_split_off() -> None:
    scales = {f"transformer_blocks.{layer}.ff_out": 1.0 for layer in range(8)}
    scales["transformer_blocks.3.ff_out"] = 25.0
    scales.update({f"transformer_blocks.{layer}.ff_in": 30.0 for layer in range(8)})
    kept, excluded = cal.split_outlier_scales(scales)
    assert excluded == ["transformer_blocks.3.ff_out"]
    assert len(kept) == 15


def test_timesteps_match_the_runtime_schedule() -> None:
    times = cal.timestep_schedule(30)
    assert len(times) == 31 and times[0] == pytest.approx(0.0) and times[-1] == 1.0
    assert times[:-1] == pytest.approx([1.0 - sigma for sigma in pipeline_spec.sigma_schedule(30)])


def test_lyrics_keep_section_tags_in_checkpoint_form() -> None:
    text = "[Verse 1: Someone]\nline one\nline two\n[Chorus]\nla la\n[Produced by X]\nmore\nwords\nhere"
    cleaned = cal.clean_lyrics(text)
    assert cleaned.splitlines()[0] == "[verse]"
    assert "[chorus]" in cleaned and "Produced" not in cleaned
    assert cal.clean_lyrics("no tags\nat all\n") is None


def test_calibration_command_line(monkeypatch, capsys) -> None:
    monkeypatch.setattr("sys.argv", ["calibrate_fp8_scales", "calibrate", "--help"])
    with pytest.raises(SystemExit) as exit_info:
        cal.main()
    assert exit_info.value.code == 0
    assert "--frames" in capsys.readouterr().out


# --- tiny engines (GPU) -------------------------------------------------------------------


def _engine_run(trt, plan, inputs: dict, output: str):
    import torch

    engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(plan)
    context = engine.create_execution_context()
    buffers = {}
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        shape = tuple(engine.get_tensor_shape(name))
        if name in inputs:
            buffers[name] = torch.as_tensor(inputs[name]).reshape(shape).float().cuda().contiguous()
        else:
            buffers[name] = torch.zeros(shape, dtype=torch.float32, device="cuda")
        context.set_tensor_address(name, buffers[name].data_ptr())
    assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    return buffers[output].cpu().numpy()


@pytest.mark.gpu
def test_tiny_dit_builds_with_fp8_block_projections() -> None:
    trt = pytest.importorskip("tensorrt")
    pytest.importorskip("torch")
    dib = importlib.import_module("families.minimax_music3.dit_builder")
    dit = importlib.import_module("families.minimax_music3.dit")
    rng = np.random.default_rng(1)

    def normal(*shape, scale=0.02):
        return (rng.standard_normal(shape, dtype=np.float32) * scale).astype(np.float32)

    p = "transformer_blocks.0"
    weights = {
        "preprocess_conv.weight": normal(dit.CONCAT_CHANNELS, dit.CONCAT_CHANNELS, 1),
        "postprocess_conv.weight": normal(dit.IN_CHANNELS, dit.IN_CHANNELS, 1),
        "proj_in.weight": normal(dit.INNER_DIM, dit.CONCAT_CHANNELS),
        "proj_out.weight": normal(dit.IN_CHANNELS, dit.INNER_DIM),
        f"{p}.norm1.weight": np.ones(dit.INNER_DIM, np.float32),
        f"{p}.norm1.bias": np.zeros(dit.INNER_DIM, np.float32),
        f"{p}.norm2.weight": np.ones(dit.INNER_DIM, np.float32),
        f"{p}.norm2.bias": np.zeros(dit.INNER_DIM, np.float32),
        f"{p}.ff_in.weight": normal(2 * dit.FF_INNER_DIM, dit.INNER_DIM),
        f"{p}.ff_in.bias": normal(2 * dit.FF_INNER_DIM),
        f"{p}.ff_out.weight": normal(dit.INNER_DIM, dit.FF_INNER_DIM),
        f"{p}.ff_out.bias": normal(dit.INNER_DIM),
    }
    for name in ("to_q", "to_k", "to_v", "to_out.0"):
        weights[f"{p}.attn.{name}.weight"] = normal(dit.INNER_DIM, dit.INNER_DIM)
    latent_length = 8
    names = [f"{p}.{stem}" for stem in quant.DIT_PROJECTIONS]

    def build(block):
        builder = trt.Builder(trt.Logger(trt.Logger.ERROR))
        network = builder.create_network()
        config = builder.create_builder_config()
        latents = network.add_input("latents", trt.float32, (1, dit.IN_CHANNELS, latent_length))
        condition = network.add_input(
            "condition", trt.float32, (1, latent_length, dit.CONDITION_DIM)
        )
        temb = network.add_input("temb", trt.float32, (1, 1, dit.INNER_DIM))
        out = dib.add_dit(
            network,
            trt,
            latents,
            condition,
            temb,
            latent_length=latent_length,
            weights=weights,
            num_layers=1,
            dtype=trt.bfloat16,
            quant=block,
        )
        out.name = "velocity"
        network.mark_output(out)
        dib.configure(config, trt)
        if block is not None:
            quant.configure_fp8_build(config)
        return bytes(builder.build_serialized_network(network, config))

    inputs = {
        "latents": rng.standard_normal((1, dit.IN_CHANNELS, latent_length)),
        "condition": rng.standard_normal((1, latent_length, dit.CONDITION_DIM)),
        "temb": rng.standard_normal((1, 1, dit.INNER_DIM)),
    }
    reference = _engine_run(trt, build(None), inputs, "velocity")
    block = quant.BlockQuant(scales=dict.fromkeys(names, 6.0 / 448.0))
    plan = build(block)
    assert sorted(block.quantized) == sorted(names)
    result = _engine_run(trt, plan, inputs, "velocity")
    assert np.isfinite(result).all()
    assert np.linalg.norm(result - reference) / np.linalg.norm(reference) < 0.1


@pytest.mark.gpu
def test_tiny_language_model_mlp_builds_with_fp8() -> None:
    trt = pytest.importorskip("tensorrt")
    pytest.importorskip("torch")
    graph_blocks = importlib.import_module("families.minimax_music3.graph_blocks")
    graph_ops = importlib.import_module("families.minimax_music3.graph_ops")
    rng = np.random.default_rng(2)
    hidden, mlp = 64, 128
    weights = {
        f"layer.0.{stem}": (rng.standard_normal(shape, dtype=np.float32) * 0.05)
        for stem, shape in (
            ("w_gate", (hidden, mlp)),
            ("w_up", (hidden, mlp)),
            ("w_down", (mlp, hidden)),
        )
    }
    context = quant.LanguageModelQuantContext(
        scales={
            "layer.0.w_gate": 4.0 / 448,
            "layer.0.w_up": 4.0 / 448,
            "layer.0.w_down": 1.0 / 448,
        },
        graph_ops=graph_ops,
    )
    builder = trt.Builder(trt.Logger(trt.Logger.ERROR))
    network = builder.create_network()
    config = builder.create_builder_config()
    source = network.add_input("x", trt.float32, (1, hidden))
    as_bf16 = network.add_cast(source, trt.bfloat16).get_output(0)
    out = graph_blocks.add_swiglu_mlp(
        network,
        as_bf16,
        weights=weights,
        prefix="layer.0",
        hidden_size=hidden,
        mlp_size=mlp,
        dtype=np.float32,
        quant_ctx=context,
        layer_prefix="layer.0",
    )
    out = network.add_cast(out, trt.float32).get_output(0)
    out.name = "y"
    network.mark_output(out)
    quant.configure_lm_fp8_build(config)
    plan = builder.build_serialized_network(network, config)
    assert plan is not None
    assert sorted(context.quantized) == ["layer.0.w_down", "layer.0.w_gate", "layer.0.w_up"]
    x = rng.standard_normal((1, hidden)).astype(np.float32)
    y = _engine_run(trt, bytes(plan), {"x": x}, "y")
    gate, up = x @ weights["layer.0.w_gate"], x @ weights["layer.0.w_up"]
    expected = (gate / (1 + np.exp(-gate)) * up) @ weights["layer.0.w_down"]
    assert np.isfinite(y).all()
    assert np.linalg.norm(y - expected) / np.linalg.norm(expected) < 0.15
