# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FP8 self-quantization of the Clef backbone."""

import json

import numpy as np
import pytest

from families.clef import calibrate_fp8_scales
from families.clef.cli import BuildRequest
from families.clef.quantization import (
    ACTIVATION_SCALES_FILENAME,
    FP8_E4M3_MAX,
    LAYER_PREFIX,
    layer_scales,
    load_activation_scales,
    projections,
    scale_key,
)

torch = pytest.importorskip("torch")

TINY = {
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_hidden_layers": 2,
    "layer_types": ["linear_attention", "full_attention"],
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "linear_num_value_heads": 4,
    "linear_num_key_heads": 2,
    "linear_key_head_dim": 64,
    "linear_value_head_dim": 64,
    "linear_conv_kernel_dim": 4,
    "rms_norm_eps": 1e-6,
    "rope_parameters": {"partial_rotary_factor": 0.25, "rope_theta": 10000000},
}
CHECKPOINTS = {
    # Cloudflare/clef and Cloudflare/clef-flash at the pinned revisions.
    (5120, 64, 17408),
    (4096, 32, 12288),
}


def _table():
    from pathlib import Path

    import families.clef.quantization as quantization

    path = Path(quantization.__file__).parent / ACTIVATION_SCALES_FILENAME
    return json.loads(path.read_text())


def test_weight_scale_is_amax_over_448():
    from families.clef.quantization import quantize_fp8_weight

    weight = torch.tensor([[0.5, -2.0, 0.25], [1.0, 0.0, -0.125]], dtype=torch.bfloat16)
    packed, scale = quantize_fp8_weight(weight)
    assert scale == pytest.approx(2.0 / FP8_E4M3_MAX)
    assert packed.dtype == np.uint8 and packed.shape == (2, 3)
    values = torch.from_numpy(packed).view(torch.float8_e4m3fn).float() * scale
    torch.testing.assert_close(values, weight.float(), rtol=0.07, atol=0)
    assert float(values.abs().max()) == pytest.approx(2.0)


def test_zero_weight_is_rejected():
    from families.clef.quantization import quantize_fp8_weight

    with pytest.raises(ValueError, match="amax"):
        quantize_fp8_weight(torch.zeros(4, 4))


def test_scale_key_names_config_and_fingerprint():
    from families.clef.quantization import weights_fingerprint

    weight = torch.tensor([1.0, -2.0, 0.5, 0.0, 9.0]).bfloat16()
    fingerprint = weights_fingerprint(weight)
    assert fingerprint == "3f80c0003f000000"
    assert scale_key(TINY, fingerprint) == "clef-h256-l2-i512-3f80c0003f000000"


def test_unknown_config_raises_and_names_the_tool(tmp_path):
    path = tmp_path / ACTIVATION_SCALES_FILENAME
    path.write_text(json.dumps({"clef-h1-l1-i1-0000": {}}))
    with pytest.raises(NotImplementedError, match="calibrate_fp8_scales.py"):
        load_activation_scales("clef-h256-l2-i512-ffff", path)


def test_null_scale_keeps_projection_in_bf16_and_missing_scale_raises():
    scales = {f"{LAYER_PREFIX}.0.{stem}": 0.01 for stem in projections(TINY, 0)}
    scales[f"{LAYER_PREFIX}.0.mlp.down_proj"] = None
    selected = layer_scales(scales, TINY, 0)
    assert f"{LAYER_PREFIX}.0.mlp.down_proj" not in selected
    assert len(selected) == len(projections(TINY, 0)) - 1
    del scales[f"{LAYER_PREFIX}.0.mlp.up_proj"]
    with pytest.raises(KeyError, match="calibrate_fp8_scales.py"):
        layer_scales(scales, TINY, 0)
    scales[f"{LAYER_PREFIX}.0.mlp.up_proj"] = -1.0
    with pytest.raises(ValueError, match="invalid"):
        layer_scales(scales, TINY, 0)


def test_recipe_keeps_small_deltanet_projections_in_bf16():
    assert projections(TINY, 0) == (
        "linear_attn.in_proj_qkv",
        "linear_attn.in_proj_z",
        "linear_attn.out_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    )
    assert projections(TINY, 1)[:4] == (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
    )


def test_committed_scales_cover_every_projection_of_both_checkpoints():
    table = _table()
    found = set()
    for key, scales in table.items():
        hidden, layers, intermediate = (int(part[1:]) for part in key.split("-")[1:4])
        found.add((hidden, layers, intermediate))
        attention = [index % 4 == 3 for index in range(layers)]
        config = {
            "layer_types": ["full_attention" if full else "linear_attention" for full in attention]
        }
        expected = {
            f"{LAYER_PREFIX}.{layer}.{stem}"
            for layer in range(layers)
            for stem in projections(config, layer)
        }
        assert set(scales) == expected
        values = [value for value in scales.values() if value is not None]
        assert all(np.isfinite(value) and value > 0 for value in values)
        assert len(values) >= 0.95 * len(scales)
    assert found == CHECKPOINTS


def test_outliers_are_stored_as_null():
    scales = {f"{LAYER_PREFIX}.{layer}.mlp.down_proj": 0.01 for layer in range(5)}
    scales[f"{LAYER_PREFIX}.2.mlp.down_proj"] = 0.5
    scales[f"{LAYER_PREFIX}.0.mlp.up_proj"] = 0.5
    marked = calibrate_fp8_scales.mark_outliers(scales)
    assert marked[f"{LAYER_PREFIX}.2.mlp.down_proj"] is None
    assert marked[f"{LAYER_PREFIX}.0.mlp.up_proj"] == 0.5
    assert marked[f"{LAYER_PREFIX}.1.mlp.down_proj"] == 0.01


def test_calibration_cli_requires_a_command():
    with pytest.raises(SystemExit):
        calibrate_fp8_scales.main([])


@pytest.mark.parametrize("quantization", ["int8", "nvfp4", "FP8"])
def test_unsupported_quantization_fails_closed(tmp_path, quantization):
    with pytest.raises(ValueError, match="quantization"):
        BuildRequest(tmp_path, quantization=quantization)


def test_none_and_fp8_are_accepted(tmp_path):
    assert BuildRequest(tmp_path).quantization is None
    assert BuildRequest(tmp_path, quantization="none").quantization == "none"
    assert BuildRequest(tmp_path, quantization="fp8").quantization == "fp8"


def _tiny_weights(index):
    generator = torch.Generator().manual_seed(index)

    def randn(*shape, scale=0.05):
        return (torch.randn(*shape, generator=generator) * scale).bfloat16()

    hidden, inter = TINY["hidden_size"], TINY["intermediate_size"]
    p = f"{LAYER_PREFIX}.{index}"
    weights = {
        f"{p}.input_layernorm.weight": randn(hidden),
        f"{p}.post_attention_layernorm.weight": randn(hidden),
        f"{p}.mlp.gate_proj.weight": randn(inter, hidden),
        f"{p}.mlp.up_proj.weight": randn(inter, hidden),
        f"{p}.mlp.down_proj.weight": randn(hidden, inter),
    }
    if TINY["layer_types"][index] == "linear_attention":
        heads, keys, dim = 4, 2, 64
        conv = 2 * keys * dim + heads * dim
        weights.update(
            {
                f"{p}.linear_attn.in_proj_qkv.weight": randn(conv, hidden),
                f"{p}.linear_attn.conv1d.weight": randn(conv, 1, 4, scale=0.3),
                f"{p}.linear_attn.in_proj_b.weight": randn(heads, hidden),
                f"{p}.linear_attn.in_proj_a.weight": randn(heads, hidden),
                f"{p}.linear_attn.dt_bias": randn(heads, scale=1.0),
                f"{p}.linear_attn.A_log": randn(heads, scale=1.0),
                f"{p}.linear_attn.norm.weight": (1 + randn(dim)).bfloat16(),
                f"{p}.linear_attn.in_proj_z.weight": randn(heads * dim, hidden),
                f"{p}.linear_attn.out_proj.weight": randn(hidden, heads * dim),
            }
        )
    else:
        heads, kv, dim = 4, 2, 64
        weights.update(
            {
                f"{p}.self_attn.q_proj.weight": randn(heads * dim * 2, hidden),
                f"{p}.self_attn.k_proj.weight": randn(kv * dim, hidden),
                f"{p}.self_attn.v_proj.weight": randn(kv * dim, hidden),
                f"{p}.self_attn.o_proj.weight": randn(hidden, heads * dim),
                f"{p}.self_attn.q_norm.weight": randn(dim),
                f"{p}.self_attn.k_norm.weight": randn(dim),
            }
        )
    return weights


def _run(trt, plan, inputs):
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(plan)
    context = engine.create_execution_context()
    for name, value in inputs.items():
        context.set_input_shape(name, tuple(value.shape))
        context.set_tensor_address(name, value.data_ptr())
    output = torch.empty_like(inputs["hidden_states"])
    context.set_tensor_address("output", output.data_ptr())
    assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    return output.float().cpu()


@pytest.mark.trt
@pytest.mark.parametrize("index", [0, 1])
def test_tiny_fp8_layer_builds_with_fp8_weights_and_tracks_bf16(index):
    trt = pytest.importorskip("tensorrt")
    if not torch.cuda.is_available():
        pytest.skip("FP8 engine build requires CUDA")
    if torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("FP8 GEMMs need compute capability 8.9 or newer")
    from families.clef.model import build_decoder_layer

    weights = _tiny_weights(index)
    scales = {f"{LAYER_PREFIX}.{index}.{stem}": 0.01 for stem in projections(TINY, index)}
    base = build_decoder_layer(weights, TINY, index, max_sequence_length=512)
    plan = build_decoder_layer(weights, TINY, index, max_sequence_length=512, fp8_scales=scales)
    assert len(plan) < len(base)
    inspector = (
        trt.Runtime(trt.Logger(trt.Logger.WARNING))
        .deserialize_cuda_engine(plan)
        .create_engine_inspector()
    )
    layers = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    assert "FP8" in layers.upper()

    torch.manual_seed(index)
    sequence = 130
    inputs = {
        "hidden_states": torch.randn(sequence, 256, dtype=torch.bfloat16, device="cuda") * 0.5
    }
    if TINY["layer_types"][index] == "full_attention":
        angles = torch.arange(sequence, device="cuda")[:, None, None] * torch.linspace(
            0.0, 1.0, 16, device="cuda"
        )
        inputs["rope_cos"] = angles.cos().bfloat16()
        inputs["rope_sin"] = angles.sin().bfloat16()
    expected = _run(trt, base, inputs)
    actual = _run(trt, plan, inputs)
    assert torch.isfinite(actual).all()
    cosine = torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0)
    assert cosine > 0.99
