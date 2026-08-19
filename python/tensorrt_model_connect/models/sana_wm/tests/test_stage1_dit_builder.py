# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the native SANA-WM Stage-1 DiT TensorRT builder surface."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tensorrt_model_connect")
safetensors_numpy = pytest.importorskip("safetensors.numpy")

from tensorrt_model_connect.checkpoint_mapper import WeightDict  # noqa: E402
from tensorrt_model_connect.models.sana_wm import stage1_dit_builder  # noqa: E402


SANA_WM_GDN_PLUGIN_NAME = "libtrtmc_sana_wm_gdn_plugin.so"


def _raw_sana_wm_config() -> dict:
    return {
        "video_num_frames": 321,
        "video_height": 704,
        "video_width": 1280,
        "patch_size": [1, 1, 1],
        "model": {
            "attn_type": "BidirectionalGDNTriton",
            "cam_attn_compress": 1,
            "camctrl_type": "BidirectionalGDNUCPESinglePathLiteLABothTriton",
            "cross_norm": True,
            "ffn_type": "GLUMBConvTemp",
            "fp32_attention": True,
            "linear_head_dim": 112,
            "mlp_ratio": 3,
            "mixed_precision": "bf16",
            "pos_embed_type": "wan_rope",
            "qk_norm": True,
            "softmax_every_n": 4,
            "t_kernel_size": 3,
            "use_chunk_plucker_post_attn": True,
            "use_pe": True,
            "y_norm": True,
        },
        "vae": {"vae_latent_dim": 128, "vae_stride": [8, 32, 32]},
        "text_encoder": {"model_max_length": 300},
        "_sana_wm_stage1_dit_summary": {
            "text_embed_dim": 2304,
            "chunk_plucker_channels": 48,
        },
    }


def _stage1_weights() -> WeightDict:
    weights = WeightDict()
    weights["x_embedder.proj.weight"] = np.zeros(
        (2240, 128, 1, 1, 1),
        dtype=np.float16,
    )
    weights["x_embedder.proj.bias"] = np.zeros((2240,), dtype=np.float16)
    weights["plucker_embedder.proj.weight"] = np.zeros(
        (2240, 48, 1, 1, 1),
        dtype=np.float16,
    )
    weights["plucker_embedder.proj.bias"] = np.zeros((2240,), dtype=np.float16)
    weights["t_embedder.mlp.0.weight"] = np.zeros((256, 2240), dtype=np.float16)
    weights["t_embedder.mlp.0.bias"] = np.zeros((2240,), dtype=np.float16)
    weights["t_embedder.mlp.2.weight"] = np.zeros((2240, 2240), dtype=np.float16)
    weights["t_embedder.mlp.2.bias"] = np.zeros((2240,), dtype=np.float16)
    weights["t_block.1.weight"] = np.zeros((2240, 6 * 2240), dtype=np.float16)
    weights["t_block.1.bias"] = np.zeros((6 * 2240,), dtype=np.float16)
    weights["y_embedder.y_proj.fc1.weight"] = np.zeros((2304, 2240), dtype=np.float16)
    weights["y_embedder.y_proj.fc1.bias"] = np.zeros((2240,), dtype=np.float16)
    weights["y_embedder.y_proj.fc2.weight"] = np.zeros((2240, 2240), dtype=np.float16)
    weights["y_embedder.y_proj.fc2.bias"] = np.zeros((2240,), dtype=np.float16)
    weights["attention_y_norm.weight"] = np.ones((2240,), dtype=np.float32) * 0.01
    weights["blocks.0.scale_shift_table"] = np.zeros((6, 2240), dtype=np.float32)
    weights["blocks.0.attn.qkv.weight"] = np.zeros(
        (2240, 3 * 2240),
        dtype=np.float16,
    )
    weights["blocks.0.attn.conv_k.weight"] = np.zeros((2240, 1, 4), dtype=np.float16)
    weights["blocks.0.attn.conv_k.bias"] = np.zeros((2240,), dtype=np.float16)
    weights["blocks.0.attn.q_norm.weight"] = np.ones((2240,), dtype=np.float32)
    weights["blocks.0.attn.k_norm.weight"] = np.ones((2240,), dtype=np.float32)
    weights["blocks.0.attn.beta_proj.weight"] = np.zeros((2240, 20), dtype=np.float16)
    weights["blocks.0.attn.beta_proj.bias"] = np.zeros((20,), dtype=np.float16)
    weights["blocks.0.attn.gate_proj.weight"] = np.zeros((2240, 20), dtype=np.float16)
    weights["blocks.0.attn.gate_proj.bias"] = np.zeros((20,), dtype=np.float16)
    weights["blocks.0.attn.dt_bias"] = np.zeros((20,), dtype=np.float32)
    weights["blocks.0.attn.A_log"] = np.zeros((20,), dtype=np.float32)
    weights["blocks.0.attn.q_proj_cam.weight"] = np.zeros(
        (2240, 2240),
        dtype=np.float16,
    )
    weights["blocks.0.attn.q_proj_cam.bias"] = np.zeros((2240,), dtype=np.float16)
    weights["blocks.0.attn.k_proj_cam.weight"] = np.zeros(
        (2240, 2240),
        dtype=np.float16,
    )
    weights["blocks.0.attn.k_proj_cam.bias"] = np.zeros((2240,), dtype=np.float16)
    weights["blocks.0.attn.v_proj_cam.weight"] = np.zeros(
        (2240, 2240),
        dtype=np.float16,
    )
    weights["blocks.0.attn.v_proj_cam.bias"] = np.zeros((2240,), dtype=np.float16)
    weights["blocks.0.attn.conv_k_cam.weight"] = np.zeros(
        (2240, 1, 4),
        dtype=np.float16,
    )
    weights["blocks.0.attn.conv_k_cam.bias"] = np.zeros((2240,), dtype=np.float16)
    weights["blocks.0.attn.q_norm_cam.weight"] = np.ones((2240,), dtype=np.float32)
    weights["blocks.0.attn.k_norm_cam.weight"] = np.ones((2240,), dtype=np.float32)
    weights["blocks.0.attn.out_proj_cam.weight"] = np.zeros(
        (2240, 2240),
        dtype=np.float16,
    )
    weights["blocks.0.attn.out_proj_cam.bias"] = np.zeros((2240,), dtype=np.float16)
    weights["blocks.0.attn.output_gate.weight"] = np.zeros(
        (2240, 2240),
        dtype=np.float16,
    )
    weights["blocks.0.attn.output_gate.bias"] = np.zeros((2240,), dtype=np.float16)
    weights["blocks.0.attn.proj.weight"] = np.zeros((2240, 2240), dtype=np.float16)
    weights["blocks.0.attn.proj.bias"] = np.zeros((2240,), dtype=np.float16)
    weights["blocks.0.plucker_proj.weight"] = np.zeros((2240, 2240), dtype=np.float16)
    weights["blocks.0.plucker_proj.bias"] = np.zeros((2240,), dtype=np.float16)
    weights["blocks.0.cross_attn.q_linear.weight"] = np.zeros((2240, 2240), dtype=np.float16)
    weights["blocks.0.cross_attn.q_linear.bias"] = np.zeros((2240,), dtype=np.float16)
    weights["blocks.0.cross_attn.kv_linear.weight"] = np.zeros((2240, 4480), dtype=np.float16)
    weights["blocks.0.cross_attn.kv_linear.bias"] = np.zeros((4480,), dtype=np.float16)
    weights["blocks.0.cross_attn.q_norm.weight"] = np.ones((2240,), dtype=np.float32)
    weights["blocks.0.cross_attn.k_norm.weight"] = np.ones((2240,), dtype=np.float32)
    weights["blocks.0.cross_attn.proj.weight"] = np.zeros((2240, 2240), dtype=np.float16)
    weights["blocks.0.cross_attn.proj.bias"] = np.zeros((2240,), dtype=np.float16)
    weights["blocks.0.mlp.inverted_conv.conv.weight"] = np.zeros(
        (13440, 2240, 1, 1),
        dtype=np.float16,
    )
    weights["blocks.0.mlp.inverted_conv.conv.bias"] = np.zeros((13440,), dtype=np.float16)
    weights["blocks.0.mlp.depth_conv.conv.weight"] = np.zeros(
        (13440, 1, 3, 3),
        dtype=np.float16,
    )
    weights["blocks.0.mlp.depth_conv.conv.bias"] = np.zeros((13440,), dtype=np.float16)
    weights["blocks.0.mlp.point_conv.conv.weight"] = np.zeros(
        (2240, 6720, 1, 1),
        dtype=np.float16,
    )
    weights["blocks.0.mlp.t_conv.weight"] = np.zeros((2240, 2240, 3, 1), dtype=np.float16)
    weights["final_layer.scale_shift_table"] = np.zeros((2, 2240), dtype=np.float32)
    weights["final_layer.linear.weight"] = np.zeros((2240, 128), dtype=np.float16)
    weights["final_layer.linear.bias"] = np.zeros((128,), dtype=np.float16)
    return weights


def test_load_sana_wm_stage1_dit_weights_uses_trt_layouts(tmp_path) -> None:
    path = tmp_path / "sana_wm_1600m_720p.safetensors"
    linear = np.arange(6, dtype=np.float32).reshape(2, 3)
    conv = np.arange(12, dtype=np.float32).reshape(4, 3, 1, 1, 1)
    norm = np.arange(4, dtype=np.float32)
    gdn_state = np.arange(4, dtype=np.float32)
    safetensors_numpy.save_file(
        {
            "y_embedder.y_proj.fc1.weight": linear,
            "x_embedder.proj.weight": conv,
            "blocks.0.norm1.weight": norm,
            "blocks.0.attn.gdn.A_log": gdn_state,
            "blocks.0.attn.gdn.dt_bias": gdn_state,
            "final_layer.scale_shift_table": np.zeros((2, 4), dtype=np.float32),
        },
        str(path),
    )

    weights = stage1_dit_builder.load_sana_wm_stage1_dit_weights(
        path,
        precision="fp16",
    )

    assert weights["y_embedder.y_proj.fc1.weight"].shape == (3, 2)
    np.testing.assert_allclose(
        weights["y_embedder.y_proj.fc1.weight"],
        linear.T.astype(np.float16),
    )
    assert weights["y_embedder.y_proj.fc1.weight"].dtype == np.float16
    assert weights["x_embedder.proj.weight"].shape == conv.shape
    assert weights["x_embedder.proj.weight"].dtype == np.float16
    assert weights["blocks.0.norm1.weight"].dtype == np.float32
    assert weights["blocks.0.attn.gdn.A_log"].dtype == np.float32
    assert weights["blocks.0.attn.gdn.dt_bias"].dtype == np.float32
    assert weights["final_layer.scale_shift_table"].dtype == np.float32
    assert weights["_source_path"] == str(path)
    assert weights["_precision"] == "fp16"


def test_stage1_shape_from_config_matches_upstream_contract() -> None:
    shape = stage1_dit_builder.stage1_shape_from_config(
        _raw_sana_wm_config(),
        _stage1_weights(),
    )

    assert shape.batch_size == 2
    assert shape.latent_channels == 128
    assert shape.latent_frames == 41
    assert shape.latent_height == 22
    assert shape.latent_width == 40
    assert shape.text_max_length == 300
    assert shape.text_embed_dim == 2304
    assert shape.chunk_plucker_channels == 48
    assert shape.raymap_width == 20


def test_stage1_precision_helpers_match_public_bf16_config() -> None:
    assert stage1_dit_builder._stage1_norm_eps(_raw_sana_wm_config()) == 1.0e-5
    assert stage1_dit_builder._stage1_attention_eps(_raw_sana_wm_config()) == 1.0e-15
    assert stage1_dit_builder._target_trt_dtype(_FakeTrtWithBf16, "bf16") == "bfloat16"
    with pytest.raises(RuntimeError, match="BF16"):
        stage1_dit_builder._target_trt_dtype(_FakeTrt, "bf16")


def test_stage1_camera_prep_plugin_mode_parses_v_only(monkeypatch) -> None:
    monkeypatch.delenv("TRTMC_SANA_WM_CAMERA_PREP_PLUGIN", raising=False)
    assert stage1_dit_builder._sana_wm_camera_prep_plugin_mode() == "v"

    monkeypatch.setenv("TRTMC_SANA_WM_CAMERA_PREP_PLUGIN", "0")
    assert stage1_dit_builder._sana_wm_camera_prep_plugin_mode() == "off"

    monkeypatch.setenv("TRTMC_SANA_WM_CAMERA_PREP_PLUGIN", "1")
    assert stage1_dit_builder._sana_wm_camera_prep_plugin_mode() == "all"

    monkeypatch.setenv("TRTMC_SANA_WM_CAMERA_PREP_PLUGIN", "v-only")
    assert stage1_dit_builder._sana_wm_camera_prep_plugin_mode() == "v"


def test_sana_wm_bf16_linear_plugin_can_request_matmul_bias(monkeypatch) -> None:
    ml_dtypes = pytest.importorskip("ml_dtypes")
    creator = _FakePluginCreator()
    monkeypatch.setattr(
        stage1_dit_builder,
        "_get_sana_wm_gate_proj_plugin_creator",
        lambda trt_module: creator,
    )
    monkeypatch.setenv("TRTMC_SANA_WM_CAMERA_OUT_PROJ_PLUGIN", "1")
    weights = WeightDict()
    weights["blocks.0.attn.out_proj_cam.weight"] = np.zeros((8, 8), dtype=ml_dtypes.bfloat16)
    weights["blocks.0.attn.out_proj_cam.bias"] = np.zeros((8,), dtype=ml_dtypes.bfloat16)

    output = stage1_dit_builder._add_sana_wm_bf16_linear_plugin(
        _FakeNetwork(),
        _FakeTensor("camera_tokens", dtype=_FakeTrtWithBf16Plugin.bfloat16),
        weights=weights,
        prefix="blocks.0.attn.out_proj_cam",
        input_dim=8,
        output_dim=8,
        trt_module=_FakeTrtWithBf16Plugin,
        dtype=ml_dtypes.bfloat16,
        env_var="TRTMC_SANA_WM_CAMERA_OUT_PROJ_PLUGIN",
        name="blocks.0.attn.out_proj_cam",
        use_matmul_bias=True,
    )

    assert output.name == "blocks.0.attn.out_proj_cam.output"
    assert int(creator.created[0].fields["use_matmul_bias"][0]) == 1


def test_sana_wm_output_gate_plugin_uses_two_inputs_and_fp32_mode(monkeypatch) -> None:
    ml_dtypes = pytest.importorskip("ml_dtypes")
    creator = _FakePluginCreator()
    monkeypatch.setattr(
        stage1_dit_builder,
        "_get_sana_wm_gate_proj_plugin_creator",
        lambda trt_module: creator,
    )
    monkeypatch.setenv("TRTMC_SANA_WM_OUTPUT_GATE_PLUGIN", "1")
    weights = WeightDict()
    weights["blocks.0.attn.output_gate.weight"] = np.zeros((8, 8), dtype=ml_dtypes.bfloat16)
    weights["blocks.0.attn.output_gate.bias"] = np.zeros((8,), dtype=ml_dtypes.bfloat16)
    network = _FakeNetwork()

    output = stage1_dit_builder._add_sana_wm_output_gate_plugin(
        network,
        _FakeTensor("combined", dtype=_FakeTrtWithBf16Plugin.bfloat16),
        _FakeTensor("x_msa_in", dtype=_FakeTrtWithBf16Plugin.bfloat16),
        weights=weights,
        prefix="blocks.0.attn.output_gate",
        hidden_size=8,
        trt_module=_FakeTrtWithBf16Plugin,
        dtype=ml_dtypes.bfloat16,
        name="blocks.0.attn.output_gate.gated",
    )

    assert output.name == "blocks.0.attn.output_gate.gated"
    assert [tensor.name for tensor in network.plugins[0].inputs] == [
        "combined",
        "x_msa_in",
    ]
    assert int(creator.created[0].fields["activation"][0]) == 2


def test_sana_wm_frame_gate_plugin_preserves_bf16_rounding_boundary(monkeypatch) -> None:
    ml_dtypes = pytest.importorskip("ml_dtypes")
    creator = _FakePluginCreator()
    monkeypatch.setattr(
        stage1_dit_builder,
        "_get_sana_wm_frame_gate_plugin_creator",
        lambda trt_module: creator,
    )
    monkeypatch.setenv("TRTMC_SANA_WM_FRAME_GATE_PLUGIN", "1")
    network = _FakeNetwork()
    shape = stage1_dit_builder.SanaWmStage1Shape(
        batch_size=1,
        latent_channels=2,
        latent_frames=2,
        latent_height=1,
        latent_width=2,
        text_max_length=4,
        text_embed_dim=8,
        chunk_plucker_channels=3,
    )

    output = stage1_dit_builder._apply_frame_gate_to_tokens(
        network,
        _FakeTensor("tokens", dtype=_FakeTrtWithBf16Plugin.bfloat16),
        _FakeTensor("gate", dtype=_FakeTrtWithBf16Plugin.bfloat16),
        shape,
        8,
        trt_module=_FakeTrtWithBf16Plugin,
        dtype=ml_dtypes.bfloat16,
        name="blocks.0.attn.gated",
    )

    assert output.name == "blocks.0.attn.gated"
    assert [tensor.name for tensor in network.plugins[0].inputs] == ["tokens", "gate"]
    assert creator.created[0].name == "sana_wm_frame_gate"


def test_stage1_runtime_parameter_values_round_fp16_before_promoting_to_fp32() -> None:
    values = np.asarray([1.001, -0.3333, 0.0, 2.0001], dtype=np.float32)

    rounded = stage1_dit_builder._fp32_parameter_values_for_runtime_dtype(values, np.float16)

    assert rounded.dtype == np.float32
    np.testing.assert_array_equal(rounded, values.astype(np.float16).astype(np.float32))


def test_stage1_runtime_parameter_values_round_bf16_before_promoting_to_fp32() -> None:
    ml_dtypes = pytest.importorskip("ml_dtypes")
    values = np.asarray([1.001, -0.3333, 0.0, 2.0001], dtype=np.float32)

    rounded = stage1_dit_builder._fp32_parameter_values_for_runtime_dtype(
        values,
        ml_dtypes.bfloat16,
    )

    assert rounded.dtype == np.float32
    np.testing.assert_array_equal(
        rounded,
        values.astype(ml_dtypes.bfloat16).astype(np.float32),
    )


def test_stage1_mlp_silu_defaults_to_fp32_roundtrip(monkeypatch) -> None:
    ml_dtypes = pytest.importorskip("ml_dtypes")
    monkeypatch.delenv("TRTMC_SANA_WM_MLP_SILU_FP32", raising=False)
    network = _FakeNetwork()

    out = stage1_dit_builder._add_silu_for_stage1_mlp(
        network,
        _FakeTensor("x", dtype=_FakeTrtWithBf16.bfloat16),
        trt_module=_FakeTrtWithBf16,
        dtype=ml_dtypes.bfloat16,
    )

    assert out.dtype == _FakeTrtWithBf16.bfloat16
    assert [(layer.inp.name, layer.dtype) for layer in network.casts] == [
        ("x", _FakeTrtWithBf16.float32),
        ("elementwise", _FakeTrtWithBf16.bfloat16),
    ]
    assert network.activations[0].inp.name == "cast"

    monkeypatch.setenv("TRTMC_SANA_WM_MLP_SILU_FP32", "0")
    network = _FakeNetwork()
    out = stage1_dit_builder._add_silu_for_stage1_mlp(
        network,
        _FakeTensor("x", dtype=_FakeTrtWithBf16.bfloat16),
        trt_module=_FakeTrtWithBf16,
        dtype=ml_dtypes.bfloat16,
    )

    assert out.dtype == _FakeTrtWithBf16.bfloat16
    assert network.casts == []
    assert network.activations[0].inp.name == "x"


def test_add_rmsnorm_uses_runtime_rounded_weight_values() -> None:
    network = _FakeNetwork()
    weight = np.asarray([1.001, -0.3333, 0.0, 2.0001], dtype=np.float32)

    out = stage1_dit_builder._add_rmsnorm(
        network,
        _FakeTensor("x", dtype=_FakeTrt.float16),
        weight,
        rank=2,
        eps=1.0e-5,
        trt_module=_FakeTrt,
        dtype=np.float16,
        name="rms.output",
    )

    assert out.name == "rms.output"
    np.testing.assert_array_equal(
        network.constants[1].weights.value,
        weight.astype(np.float16).astype(np.float32).reshape((1, 4)),
    )


def test_add_rmsnorm_keep_fp32_output_preserves_fp32_weight_values() -> None:
    network = _FakeNetwork()
    weight = np.asarray([1.001, -0.3333, 0.0, 2.0001], dtype=np.float32)

    out = stage1_dit_builder._add_rmsnorm(
        network,
        _FakeTensor("x", dtype=_FakeTrt.float16),
        weight,
        rank=2,
        eps=1.0e-5,
        trt_module=_FakeTrt,
        dtype=np.float16,
        name="rms.output",
        keep_fp32_output=True,
    )

    assert out.name == "rms.output"
    np.testing.assert_array_equal(
        network.constants[1].weights.value,
        weight.reshape((1, 4)),
    )


def test_add_layernorm_no_affine_uses_registered_bf16_plugin(monkeypatch) -> None:
    ml_dtypes = pytest.importorskip("ml_dtypes")
    monkeypatch.setenv("TRTMC_SANA_WM_LAYER_NORM_PLUGIN", "1")
    network = _FakeNetwork()
    creator = _FakePluginCreator()
    monkeypatch.setattr(
        stage1_dit_builder,
        "_get_sana_wm_layer_norm_plugin_creator",
        lambda trt_module: creator,
    )

    out = stage1_dit_builder._add_layernorm_no_affine(
        network,
        _FakeTensor("x", dtype=_FakeTrtWithBf16Plugin.bfloat16),
        rank=4,
        eps=1.0e-6,
        trt_module=_FakeTrtWithBf16Plugin,
        dtype=ml_dtypes.bfloat16,
        name="layernorm.output",
    )

    assert out.name == "layernorm.output"
    assert len(network.plugins) == 1
    assert network.plugins[0].inputs[0].name == "x"
    assert creator.created[0].name == "sana_wm_layer_norm"
    np.testing.assert_array_equal(
        creator.created[0].fields["eps"],
        np.asarray([1.0e-6], dtype=np.float32),
    )


def test_lower_sana_wm_gdn_frame_gates_uses_runtime_rounded_decay_parameters() -> None:
    network = _FakeNetwork()
    weights = _stage1_weights()
    dt_bias = np.linspace(-5.125, 1.125, 20, dtype=np.float32)
    a_log = np.linspace(-0.875, 0.875, 20, dtype=np.float32)
    weights["blocks.0.attn.dt_bias"] = dt_bias
    weights["blocks.0.attn.A_log"] = a_log
    shape = stage1_dit_builder.stage1_shape_from_config(
        _raw_sana_wm_config(),
        weights,
    )
    frontend = stage1_dit_builder.SanaWmStage1Frontend(
        x_tokens=_FakeTensor("x_tokens", dtype=_FakeTrt.float16),
        token_count=41 * 22 * 40,
        hidden_size=2240,
        plucker_tokens=_FakeTensor("plucker_tokens", dtype=_FakeTrt.float16),
    )

    stage1_dit_builder._lower_sana_wm_gdn_frame_gates(
        network,
        _FakeTensor("x_msa", dtype=_FakeTrt.float16),
        shape,
        frontend,
        weights,
        block_index=0,
        num_heads=20,
        trt_module=_FakeTrt,
        dtype=np.float16,
    )

    constants = [
        np.asarray(layer.weights.value).reshape(-1)
        for layer in network.constants
        if layer.shape == (1, 1, 20)
    ]
    expected_dt = dt_bias.astype(np.float16).astype(np.float32)
    expected_a = np.exp(a_log.astype(np.float16).astype(np.float32))
    assert any(np.array_equal(value, expected_dt) for value in constants)
    assert any(np.array_equal(value, expected_a) for value in constants)


def test_lower_sana_wm_gdn_frame_gates_uses_bf16_plugins_by_default(monkeypatch) -> None:
    ml_dtypes = pytest.importorskip("ml_dtypes")
    for name in (
        "TRTMC_SANA_WM_BETA_PROJ_PLUGIN",
        "TRTMC_SANA_WM_GATE_PROJ_PLUGIN",
        "TRTMC_SANA_WM_FRAME_MEAN_PLUGIN",
        "TRTMC_SANA_WM_DECAY_PLUGIN",
    ):
        monkeypatch.delenv(name, raising=False)
    gate_creator = _FakePluginCreator()
    frame_mean_creator = _FakePluginCreator()
    decay_creator = _FakePluginCreator()
    monkeypatch.setattr(
        stage1_dit_builder,
        "_get_sana_wm_gate_proj_plugin_creator",
        lambda trt_module: gate_creator,
    )
    monkeypatch.setattr(
        stage1_dit_builder,
        "_get_sana_wm_frame_mean_plugin_creator",
        lambda trt_module: frame_mean_creator,
    )
    monkeypatch.setattr(
        stage1_dit_builder,
        "_get_sana_wm_decay_plugin_creator",
        lambda trt_module: decay_creator,
    )
    weights = WeightDict()
    weights["blocks.0.attn.beta_proj.weight"] = np.zeros((8, 2), dtype=np.float16)
    weights["blocks.0.attn.beta_proj.bias"] = np.zeros((2,), dtype=np.float16)
    weights["blocks.0.attn.gate_proj.weight"] = np.zeros((8, 2), dtype=np.float16)
    weights["blocks.0.attn.gate_proj.bias"] = np.zeros((2,), dtype=np.float16)
    weights["blocks.0.attn.dt_bias"] = np.zeros((2,), dtype=np.float32)
    weights["blocks.0.attn.A_log"] = np.zeros((2,), dtype=np.float32)
    shape = stage1_dit_builder.SanaWmStage1Shape(
        batch_size=1,
        latent_channels=2,
        latent_frames=2,
        latent_height=1,
        latent_width=2,
        text_max_length=4,
        text_embed_dim=8,
        chunk_plucker_channels=3,
    )
    frontend = stage1_dit_builder.SanaWmStage1Frontend(
        x_tokens=_FakeTensor("x_tokens"),
        token_count=4,
        hidden_size=8,
    )

    network = _FakeNetwork()
    beta, decay, x_frame, gate, gate_dt = stage1_dit_builder._lower_sana_wm_gdn_frame_gates(
        network,
        _FakeTensor("x_msa", dtype=_FakeTrtWithBf16Plugin.bfloat16),
        shape,
        frontend,
        weights,
        block_index=0,
        num_heads=2,
        trt_module=_FakeTrtWithBf16Plugin,
        dtype=ml_dtypes.bfloat16,
    )

    assert beta.name == "blocks.0.attn.beta"
    assert decay.name == "blocks.0.attn.decay"
    assert x_frame.name == "blocks.0.attn.x_frame"
    assert gate.name == "blocks.0.attn.gate"
    assert gate_dt.name == "blocks.0.attn.gate_dt"
    assert len(network.plugins) == 4
    assert [plugin.name for plugin in gate_creator.created] == [
        "sana_wm_gate_proj",
        "sana_wm_gate_proj",
    ]
    assert frame_mean_creator.created[0].name == "sana_wm_frame_mean"
    assert decay_creator.created[0].name == "sana_wm_decay"
    assert len(network.matrix_multiply) == 0
    assert len(network.reductions) == 0

    for name in (
        "TRTMC_SANA_WM_BETA_PROJ_PLUGIN",
        "TRTMC_SANA_WM_GATE_PROJ_PLUGIN",
        "TRTMC_SANA_WM_FRAME_MEAN_PLUGIN",
        "TRTMC_SANA_WM_DECAY_PLUGIN",
    ):
        monkeypatch.setenv(name, "0")
    fallback_network = _FakeNetwork()
    stage1_dit_builder._lower_sana_wm_gdn_frame_gates(
        fallback_network,
        _FakeTensor("x_msa", dtype=_FakeTrtWithBf16Plugin.bfloat16),
        shape,
        frontend,
        weights,
        block_index=0,
        num_heads=2,
        trt_module=_FakeTrtWithBf16Plugin,
        dtype=ml_dtypes.bfloat16,
    )

    assert len(fallback_network.plugins) == 0
    assert len(fallback_network.matrix_multiply) == 2
    assert len(fallback_network.reductions) == 1
    assert [layer.activation_type for layer in fallback_network.activations] == ["sigmoid"]


def test_wan_rope_angles_match_upstream_fhw_split_order() -> None:
    angles = stage1_dit_builder._wan_rope_angles(
        latent_frames=2,
        latent_height=3,
        latent_width=4,
        head_dim=112,
    )

    assert angles.shape == (2 * 3 * 4, 56)
    np.testing.assert_allclose(angles[0], np.zeros((56,), dtype=np.float64))

    token = 1 * 3 * 4 + 2 * 4 + 3
    t_freq = 1.0 / (10000.0 ** (np.arange(0, 40, 2) / 40.0))
    h_freq = 1.0 / (10000.0 ** (np.arange(0, 36, 2) / 36.0))
    expected = np.concatenate([1.0 * t_freq, 2.0 * h_freq, 3.0 * h_freq])
    np.testing.assert_allclose(angles[token], expected)


class _FakeNetwork:
    def __init__(self, flags: int = 0) -> None:
        self.flags = flags
        self.inputs: list[tuple[str, object, tuple[int, ...]]] = []
        self.convolutions: list[_FakeConvolution] = []
        self.shuffles: list[_FakeShuffle] = []
        self.constants: list[_FakeConstant] = []
        self.matrix_multiply: list[_FakeMatrixMultiply] = []
        self.elementwise: list[_FakeElementwise] = []
        self.activations: list[_FakeActivation] = []
        self.unary: list[_FakeUnary] = []
        self.reductions: list[_FakeReduce] = []
        self.slices: list[_FakeSlice] = []
        self.concatenations: list[_FakeConcatenation] = []
        self.casts: list[_FakeCast] = []
        self.softmax: list[_FakeSoftmax] = []
        self.attentions: list[_FakeAttention] = []
        self.plugins: list[_FakePluginLayer] = []
        self.outputs: list[_FakeTensor] = []

    def add_input(self, name, dtype, shape):  # noqa: ANN001
        record = (name, dtype, tuple(shape))
        self.inputs.append(record)
        return _FakeTensor(name, dtype=dtype)

    def add_convolution_nd(  # noqa: ANN001
        self,
        inp,
        *,
        num_output_maps,
        kernel_shape,
        kernel,
        bias,
    ):
        layer = _FakeConvolution(
            inp=inp,
            num_output_maps=num_output_maps,
            kernel_shape=tuple(kernel_shape),
            kernel=kernel,
            bias=bias,
        )
        self.convolutions.append(layer)
        return layer

    def add_shuffle(self, inp):  # noqa: ANN001
        layer = _FakeShuffle(inp)
        self.shuffles.append(layer)
        return layer

    def add_constant(self, shape, weights):  # noqa: ANN001
        layer = _FakeConstant(tuple(shape), weights)
        self.constants.append(layer)
        return layer

    def add_matrix_multiply(self, lhs, lhs_op, rhs, rhs_op):  # noqa: ANN001
        layer = _FakeMatrixMultiply(lhs, lhs_op, rhs, rhs_op)
        self.matrix_multiply.append(layer)
        return layer

    def add_elementwise(self, lhs, rhs, op):  # noqa: ANN001
        layer = _FakeElementwise(lhs, rhs, op)
        self.elementwise.append(layer)
        return layer

    def add_activation(self, inp, activation_type):  # noqa: ANN001
        layer = _FakeActivation(inp, activation_type)
        self.activations.append(layer)
        return layer

    def add_unary(self, inp, op):  # noqa: ANN001
        layer = _FakeUnary(inp, op)
        self.unary.append(layer)
        return layer

    def add_reduce(self, inp, op, axes, keep_dims):  # noqa: ANN001
        layer = _FakeReduce(inp, op, axes, keep_dims)
        self.reductions.append(layer)
        return layer

    def add_slice(self, inp, *, start, shape, stride):  # noqa: ANN001
        layer = _FakeSlice(inp, tuple(start), tuple(shape), tuple(stride))
        self.slices.append(layer)
        return layer

    def add_concatenation(self, inputs):  # noqa: ANN001
        layer = _FakeConcatenation(list(inputs))
        self.concatenations.append(layer)
        return layer

    def add_cast(self, inp, dtype):  # noqa: ANN001
        layer = _FakeCast(inp, dtype)
        self.casts.append(layer)
        return layer

    def add_softmax(self, inp):  # noqa: ANN001
        layer = _FakeSoftmax(inp)
        self.softmax.append(layer)
        return layer

    def add_attention(self, q, k, v, norm_op, causal):  # noqa: ANN001
        layer = _FakeAttention(q, k, v, norm_op, causal)
        self.attentions.append(layer)
        return layer

    def add_plugin_v2(self, inputs, plugin):  # noqa: ANN001
        layer = _FakePluginLayer(list(inputs), plugin)
        self.plugins.append(layer)
        return layer

    def mark_output(self, tensor):  # noqa: ANN001
        self.outputs.append(tensor)


class _FakeTensor:
    def __init__(self, name: str, *, dtype=None) -> None:  # noqa: ANN001
        self.name = name
        self.dtype = dtype


class _FakeConvolution:
    def __init__(self, *, inp, num_output_maps, kernel_shape, kernel, bias) -> None:  # noqa: ANN001
        self.inp = inp
        self.num_output_maps = num_output_maps
        self.kernel_shape = kernel_shape
        self.kernel = kernel
        self.bias = bias
        self.stride_nd: tuple[int, int, int] | None = None
        self.padding_nd: tuple[int, int, int] | None = None
        self.pre_padding: tuple[int, ...] | None = None
        self.post_padding: tuple[int, ...] | None = None
        self.num_groups: int = 1
        self.output = _FakeTensor(f"conv_{num_output_maps}", dtype=getattr(inp, "dtype", None))

    def get_output(self, index: int) -> _FakeTensor:
        assert index == 0
        return self.output


class _FakeShuffle:
    def __init__(self, inp) -> None:  # noqa: ANN001
        self.inp = inp
        self.first_transpose: object | None = None
        self.second_transpose: object | None = None
        self.reshape_dims: tuple[int, ...] | None = None
        self.output = _FakeTensor("shuffle", dtype=getattr(inp, "dtype", None))

    def get_output(self, index: int) -> _FakeTensor:
        assert index == 0
        return self.output


class _FakeConstant:
    def __init__(self, shape: tuple[int, ...], weights) -> None:  # noqa: ANN001
        self.shape = shape
        self.weights = weights
        self.output = _FakeTensor(
            f"constant_{len(shape)}",
            dtype=_fake_trt_dtype_from_weights(weights),
        )

    def get_output(self, index: int) -> _FakeTensor:
        assert index == 0
        return self.output


class _FakeMatrixMultiply:
    def __init__(self, lhs, lhs_op, rhs, rhs_op) -> None:  # noqa: ANN001
        self.lhs = lhs
        self.lhs_op = lhs_op
        self.rhs = rhs
        self.rhs_op = rhs_op
        self.output = _FakeTensor("matmul", dtype=getattr(lhs, "dtype", None))

    def get_output(self, index: int) -> _FakeTensor:
        assert index == 0
        return self.output


class _FakeElementwise:
    def __init__(self, lhs, rhs, op) -> None:  # noqa: ANN001
        self.lhs = lhs
        self.rhs = rhs
        self.op = op
        self.output = _FakeTensor("elementwise", dtype=getattr(lhs, "dtype", None))

    def get_output(self, index: int) -> _FakeTensor:
        assert index == 0
        return self.output


class _FakeAttention:
    def __init__(self, q, k, v, norm_op, causal) -> None:  # noqa: ANN001
        self.q = q
        self.k = k
        self.v = v
        self.norm_op = norm_op
        self.causal = causal
        self.decomposable: bool | None = None
        self.output = _FakeTensor("attention", dtype=getattr(q, "dtype", None))

    def get_output(self, index: int) -> _FakeTensor:
        assert index == 0
        return self.output


class _FakeActivation:
    def __init__(self, inp, activation_type) -> None:  # noqa: ANN001
        self.inp = inp
        self.activation_type = activation_type
        self.output = _FakeTensor("activation", dtype=getattr(inp, "dtype", None))

    def get_output(self, index: int) -> _FakeTensor:
        assert index == 0
        return self.output


class _FakeUnary:
    def __init__(self, inp, op) -> None:  # noqa: ANN001
        self.inp = inp
        self.op = op
        self.output = _FakeTensor("unary", dtype=getattr(inp, "dtype", None))

    def get_output(self, index: int) -> _FakeTensor:
        assert index == 0
        return self.output


class _FakeReduce:
    def __init__(self, inp, op, axes, keep_dims) -> None:  # noqa: ANN001
        self.inp = inp
        self.op = op
        self.axes = axes
        self.keep_dims = keep_dims
        self.output = _FakeTensor("reduce", dtype=getattr(inp, "dtype", None))

    def get_output(self, index: int) -> _FakeTensor:
        assert index == 0
        return self.output


class _FakeSlice:
    def __init__(
        self,
        inp,
        start: tuple[int, ...],
        shape: tuple[int, ...],
        stride: tuple[int, ...],
    ) -> None:  # noqa: ANN001
        self.inp = inp
        self.start = start
        self.shape = shape
        self.stride = stride
        self.output = _FakeTensor("slice", dtype=getattr(inp, "dtype", None))

    def get_output(self, index: int) -> _FakeTensor:
        assert index == 0
        return self.output


class _FakeConcatenation:
    def __init__(self, inputs: list) -> None:
        self.inputs = inputs
        self.axis: int | None = None
        dtype = getattr(inputs[0], "dtype", None) if inputs else None
        self.output = _FakeTensor("concat", dtype=dtype)

    def get_output(self, index: int) -> _FakeTensor:
        assert index == 0
        return self.output


class _FakeCast:
    def __init__(self, inp, dtype) -> None:  # noqa: ANN001
        self.inp = inp
        self.dtype = dtype
        self.output = _FakeTensor("cast", dtype=dtype)

    def get_output(self, index: int) -> _FakeTensor:
        assert index == 0
        return self.output


class _FakeSoftmax:
    def __init__(self, inp) -> None:  # noqa: ANN001
        self.inp = inp
        self.axes: int | None = None
        self.output = _FakeTensor("softmax", dtype=getattr(inp, "dtype", None))

    def get_output(self, index: int) -> _FakeTensor:
        assert index == 0
        return self.output


class _FakePluginLayer:
    def __init__(self, inputs: list, plugin) -> None:  # noqa: ANN001
        self.inputs = inputs
        self.plugin = plugin
        self.outputs = [
            _FakeTensor(f"plugin_out{index}", dtype=getattr(inputs[0], "dtype", None))
            for index in range(4)
        ]

    def get_output(self, index: int) -> _FakeTensor:
        return self.outputs[index]


class _FakePluginFieldType:
    INT32 = "int32"
    FLOAT32 = "float32"


class _FakePluginField:
    def __init__(self, name: str, data, field_type) -> None:  # noqa: ANN001
        self.name = name
        self.data = data
        self.type = field_type


class _FakePluginFieldCollection:
    def __init__(self, fields) -> None:  # noqa: ANN001
        self.fields = fields


class _FakePlugin:
    def __init__(self, name: str, fields: _FakePluginFieldCollection) -> None:
        self.name = name
        self.fields = {field.name: field.data for field in fields.fields}


class _FakePluginCreator:
    def __init__(self) -> None:
        self.created: list[_FakePlugin] = []

    def create_plugin(self, name: str, fields: _FakePluginFieldCollection) -> _FakePlugin:
        plugin = _FakePlugin(name, fields)
        self.created.append(plugin)
        return plugin


class _FakeBuilderConfig:
    def __init__(self) -> None:
        self.pool_limits: list[tuple[object, int]] = []

    def set_memory_pool_limit(self, pool, limit):  # noqa: ANN001
        self.pool_limits.append((pool, limit))


class _FakeBuilder:
    last: "_FakeBuilder | None" = None

    def __init__(self, logger) -> None:  # noqa: ANN001
        self.logger = logger
        self.config = _FakeBuilderConfig()
        self.network: _FakeNetwork | None = None
        _FakeBuilder.last = self

    def create_builder_config(self) -> _FakeBuilderConfig:
        return self.config

    def create_network(self, flags: int) -> _FakeNetwork:
        self.network = _FakeNetwork(flags)
        return self.network

    def build_serialized_network(self, network, config):  # noqa: ANN001
        assert network is self.network
        assert config is self.config
        return b"fake-sana-wm-stage1-plan"


class _FakeLogger:
    VERBOSE = 2
    WARNING = 1

    def __init__(self, level: int) -> None:
        self.level = level


class _FakeMemoryPoolType:
    WORKSPACE = "workspace"


class _FakeNetworkDefinitionCreationFlag:
    STRONGLY_TYPED = 0


class _FakeTrt:
    float16 = "float16"
    float32 = "float32"
    int32 = "int32"
    Logger = _FakeLogger
    Builder = _FakeBuilder
    MemoryPoolType = _FakeMemoryPoolType
    NetworkDefinitionCreationFlag = _FakeNetworkDefinitionCreationFlag

    class MatrixOperation:
        NONE = "none"

    class ElementWiseOperation:
        SUM = "sum"
        PROD = "prod"
        SUB = "sub"
        DIV = "div"
        MAX = "max"
        MIN = "min"

    class AttentionNormalizationOp:
        SOFTMAX = "softmax"

    class ActivationType:
        SIGMOID = "sigmoid"
        TANH = "tanh"
        RELU = "relu"

    class UnaryOperation:
        COS = "cos"
        SIN = "sin"
        SQRT = "sqrt"
        RECIP = "recip"
        EXP = "exp"
        LOG = "log"

    class ReduceOperation:
        AVG = "avg"
        SUM = "sum"

    class Weights:
        def __init__(self, *args) -> None:  # noqa: ANN001
            self.args = args
            self.value = args[0] if len(args) == 1 else None
            self.dtype = args[0] if len(args) == 3 else None

    class Permutation:
        def __init__(self, values) -> None:  # noqa: ANN001
            self.values = list(values)


class _FakeTrtWithBf16(_FakeTrt):
    bfloat16 = "bfloat16"


class _FakeTrtWithPlugin(_FakeTrt):
    PluginField = _FakePluginField
    PluginFieldCollection = _FakePluginFieldCollection
    PluginFieldType = _FakePluginFieldType


class _FakeTrtWithBf16Plugin(_FakeTrtWithPlugin):
    bfloat16 = "bfloat16"


def _fake_trt_dtype_from_weights(weights) -> str | None:  # noqa: ANN001
    if getattr(weights, "dtype", None) is not None:
        return weights.dtype
    value = getattr(weights, "value", None)
    if value is None:
        return None
    dtype = np.asarray(value).dtype
    if dtype == np.float16:
        return "float16"
    if dtype == np.float32:
        return "float32"
    if dtype == np.int32:
        return "int32"
    return str(dtype)


def test_define_sana_wm_stage1_inputs_names_shapes_and_dtypes() -> None:
    network = _FakeNetwork()
    inputs = stage1_dit_builder.define_sana_wm_stage1_inputs(
        network,
        stage1_dit_builder.stage1_shape_from_config(
            _raw_sana_wm_config(),
            _stage1_weights(),
        ),
        trt_module=_FakeTrt,
        dtype=_FakeTrt.float16,
    )

    assert set(inputs) == {
        "x",
        "timestep",
        "y",
        "mask",
        "camera_conditions",
        "raymats",
        "raymats_inv",
        "chunk_plucker",
    }
    assert network.inputs == [
        ("x", "float16", (2, 128, 41, 22, 40)),
        ("timestep", "float32", (2, 1, 41)),
        ("y", "float16", (2, 1, 300, 2304)),
        ("mask", "int32", (2, 300)),
        ("camera_conditions", "float16", (2, 41, 20)),
        ("raymats", "float32", (2, 41 * 22 * 40, 4, 4)),
        ("raymats_inv", "float32", (2, 41 * 22 * 40, 4, 4)),
        ("chunk_plucker", "float16", (2, 48, 41, 22, 40)),
    ]


def test_lower_sana_wm_stage1_frontend_embeds_latents_and_plucker() -> None:
    network = _FakeNetwork()
    shape = stage1_dit_builder.stage1_shape_from_config(
        _raw_sana_wm_config(),
        _stage1_weights(),
    )
    inputs = stage1_dit_builder.define_sana_wm_stage1_inputs(
        network,
        shape,
        trt_module=_FakeTrt,
        dtype=_FakeTrt.float16,
    )

    frontend = stage1_dit_builder.lower_sana_wm_stage1_frontend(
        network,
        inputs,
        shape,
        _stage1_weights(),
        _raw_sana_wm_config(),
        trt_module=_FakeTrt,
        dtype=np.float16,
    )

    assert frontend.token_count == 41 * 22 * 40
    assert frontend.hidden_size == 2240
    assert frontend.x_tokens.name == "x_embedder.tokens"
    assert frontend.plucker_tokens is not None
    assert frontend.plucker_tokens.name == "plucker_embedder.tokens"
    assert len(network.convolutions) == 2
    assert network.convolutions[0].num_output_maps == 2240
    assert network.convolutions[0].kernel_shape == (1, 1, 1)
    assert network.convolutions[0].stride_nd == (1, 1, 1)
    assert network.convolutions[1].kernel.value.shape == (2240, 48, 1, 1, 1)
    assert len(network.shuffles) == 2
    assert network.shuffles[0].first_transpose.values == [0, 2, 3, 4, 1]
    assert network.shuffles[0].reshape_dims == (2, 41 * 22 * 40, 2240)


def test_lower_sana_wm_stage1_conditioning_embeds_timestep_and_text() -> None:
    network = _FakeNetwork()
    weights = _stage1_weights()
    shape = stage1_dit_builder.stage1_shape_from_config(
        _raw_sana_wm_config(),
        weights,
    )
    inputs = stage1_dit_builder.define_sana_wm_stage1_inputs(
        network,
        shape,
        trt_module=_FakeTrt,
        dtype=_FakeTrt.float16,
    )

    conditioning = stage1_dit_builder.lower_sana_wm_stage1_conditioning(
        network,
        inputs,
        shape,
        weights,
        hidden_size=2240,
        trt_module=_FakeTrt,
        dtype=np.float16,
    )

    assert conditioning.t.name == "t_embedder.output"
    assert conditioning.t0.name == "t_block.output"
    assert conditioning.y.name == "attention_y_norm.output"
    assert conditioning.mask == inputs["mask"]
    assert len(network.matrix_multiply) == 5
    assert len(network.concatenations) == 1
    assert network.concatenations[0].axis == 3
    assert [(layer.dtype, layer.inp.name) for layer in network.casts] == [
        ("float16", "timestep.frequency_embedding"),
        ("float16", "t_embedder.output.fp32"),
        ("float16", "t_block.output.fp32"),
        ("float32", "y_embedder.output"),
        ("float16", "elementwise"),
    ]
    assert [layer.op for layer in network.unary[:2]] == ["cos", "sin"]
    assert [layer.op for layer in network.unary[-2:]] == ["sqrt", "recip"]
    assert network.constants[1].shape == (1, 1, 256, 2240)
    assert network.constants[3].shape == (1, 1, 2240, 2240)
    assert network.constants[5].shape == (1, 1, 2240, 6 * 2240)
    assert any(layer.shape == (1, 1, 2240, 2240) for layer in network.constants)


def test_lower_sana_wm_stage1_conditioning_uses_timestep_plugin(monkeypatch) -> None:
    ml_dtypes = pytest.importorskip("ml_dtypes")
    network = _FakeNetwork()
    weights = _stage1_weights()
    shape = stage1_dit_builder.stage1_shape_from_config(
        _raw_sana_wm_config(),
        weights,
    )
    inputs = stage1_dit_builder.define_sana_wm_stage1_inputs(
        network,
        shape,
        trt_module=_FakeTrtWithBf16Plugin,
        dtype=_FakeTrtWithBf16Plugin.bfloat16,
    )
    creator = _FakePluginCreator()
    monkeypatch.setenv("TRTMC_SANA_WM_TIMESTEP_PLUGIN", "1")
    monkeypatch.setattr(
        stage1_dit_builder,
        "_get_sana_wm_timestep_plugin_creator",
        lambda trt_module: creator,
    )

    conditioning = stage1_dit_builder.lower_sana_wm_stage1_conditioning(
        network,
        inputs,
        shape,
        weights,
        hidden_size=2240,
        trt_module=_FakeTrtWithBf16Plugin,
        dtype=ml_dtypes.bfloat16,
    )

    assert conditioning.t.name == "t_embedder.output"
    assert conditioning.t0.name == "t_block.output"
    assert len(network.plugins) == 1
    assert network.plugins[0].inputs[0].name == "timestep"
    assert creator.created[0].name == "sana_wm_timestep_embed"
    np.testing.assert_array_equal(
        creator.created[0].fields["frequency_dim"],
        np.asarray([256], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        creator.created[0].fields["hidden_size"],
        np.asarray([2240], dtype=np.int32),
    )
    assert creator.created[0].fields["freqs"].shape == (128,)
    assert creator.created[0].fields["w0"].shape == (256 * 2240,)
    assert creator.created[0].fields["w2"].shape == (2240 * 6 * 2240,)


def test_lower_sana_wm_stage1_block_preamble_builds_adaln_and_qkv() -> None:
    network = _FakeNetwork()
    weights = _stage1_weights()
    shape = stage1_dit_builder.stage1_shape_from_config(
        _raw_sana_wm_config(),
        weights,
    )
    frontend = stage1_dit_builder.SanaWmStage1Frontend(
        x_tokens=_FakeTensor("x_tokens", dtype=_FakeTrt.float16),
        token_count=41 * 22 * 40,
        hidden_size=2240,
        plucker_tokens=_FakeTensor("plucker_tokens", dtype=_FakeTrt.float16),
    )
    conditioning = stage1_dit_builder.SanaWmStage1Conditioning(
        t=_FakeTensor("t_embedder.output", dtype=_FakeTrt.float16),
        t0=_FakeTensor("t_block.output", dtype=_FakeTrt.float16),
        y=_FakeTensor("y_embedder.output", dtype=_FakeTrt.float16),
        mask=_FakeTensor("mask"),
    )

    preamble = stage1_dit_builder.lower_sana_wm_stage1_block_preamble(
        network,
        frontend.x_tokens,
        conditioning,
        shape,
        frontend,
        weights,
        _raw_sana_wm_config(),
        block_index=0,
        trt_module=_FakeTrt,
        dtype=np.float16,
    )

    assert preamble.x_msa_in.name == "blocks.0.x_msa_in"
    assert preamble.qkv.name == "blocks.0.attn.qkv.output"
    assert preamble.qkv_heads.name == "blocks.0.attn.qkv.heads"
    assert preamble.q.name == "blocks.0.attn.q_bhdn"
    assert preamble.k.name == "blocks.0.attn.k_bhdn"
    assert preamble.q_rot.name == "blocks.0.attn.q_rot"
    assert preamble.k_rot.name == "blocks.0.attn.k_rot"
    assert preamble.v.name == "blocks.0.attn.v_bhdn"
    assert preamble.beta.name == "blocks.0.attn.beta"
    assert preamble.decay.name == "blocks.0.attn.decay"
    assert preamble.num_heads == 20
    assert preamble.head_dim == 112
    assert preamble.modulation.gate_mlp.name == "blocks.0.modulation.gate_mlp"
    assert len(network.slices) == 15
    assert [layer.start[2] for layer in network.slices[:6]] == [0, 1, 2, 3, 4, 5]
    assert network.slices[0].shape == (2, 41, 1, 2240)
    reverse_strides = [layer.stride for layer in network.slices if -1 in layer.stride]
    assert reverse_strides == [(1, 1, -1), (1, 1, -1)]
    assert len(network.reductions) == 5
    assert [layer.axes for layer in network.reductions] == [8, 8, 4, 4, 4]
    assert [layer.op for layer in network.unary] == [
        "sqrt",
        "recip",
        "sqrt",
        "recip",
        "sqrt",
        "recip",
        "exp",
        "log",
        "exp",
    ]
    assert [layer.activation_type for layer in network.activations] == [
        "relu",
        "relu",
        "sigmoid",
    ]
    assert len(network.matrix_multiply) == 3
    assert any(layer.shape == (1, 2240, 3 * 2240) for layer in network.constants)
    assert any(layer.shape == (1, 1, 112 // 2, 1, 41 * 22 * 40) for layer in network.constants)
    assert network.concatenations[-2].axis == 3
    assert network.concatenations[-1].axis == 3
    assert [(layer.dtype, layer.inp.name) for layer in network.casts] == [
        ("float32", "shuffle"),
        ("float16", "elementwise"),
        ("float32", "elementwise"),
        ("float16", "cast"),
        ("float32", "elementwise"),
        ("float16", "cast"),
        ("float32", "elementwise"),
        ("float16", "cast"),
        ("float32", "blocks.0.attn.q"),
        ("float32", "blocks.0.attn.conv_k.output"),
        ("float32", "blocks.0.attn.gate_proj.output"),
    ]
    assert len(network.convolutions) == 2
    assert all(layer.num_groups == 2240 for layer in network.convolutions)
    assert all(layer.kernel_shape == (4, 1) for layer in network.convolutions)
    assert all(layer.pre_padding == (3, 0) for layer in network.convolutions)
    assert any(layer.reshape_dims == (2, 41, 6, 2240) for layer in network.shuffles)
    assert any(layer.reshape_dims == (2, 41, 22 * 40, 2240) for layer in network.shuffles)
    assert any(layer.reshape_dims == (2, 41 * 22 * 40, 2240) for layer in network.shuffles)
    assert any(layer.reshape_dims == (2, 41 * 22 * 40, 3, 20, 112) for layer in network.shuffles)
    assert any(
        getattr(layer.first_transpose, "values", None) == [0, 2, 3, 1] for layer in network.shuffles
    )
    assert any(
        getattr(layer.first_transpose, "values", None) == [0, 3, 1, 2] for layer in network.shuffles
    )


def test_add_bidirectional_short_conv_uses_registered_bf16_plugin_by_default(
    monkeypatch,
) -> None:
    ml_dtypes = pytest.importorskip("ml_dtypes")
    monkeypatch.delenv("TRTMC_SANA_WM_SHORT_CONV_PLUGIN", raising=False)
    creator = _FakePluginCreator()
    monkeypatch.setattr(
        stage1_dit_builder,
        "_get_sana_wm_short_conv_plugin_creator",
        lambda trt_module: creator,
    )
    shape = stage1_dit_builder.SanaWmStage1Shape(
        batch_size=1,
        latent_channels=2,
        latent_frames=3,
        latent_height=1,
        latent_width=2,
        text_max_length=4,
        text_embed_dim=8,
        chunk_plucker_channels=3,
    )
    frontend = stage1_dit_builder.SanaWmStage1Frontend(
        x_tokens=_FakeTensor("x_tokens"),
        token_count=6,
        hidden_size=4,
    )
    weight = np.arange(12, dtype=np.float32).reshape(4, 1, 3)
    bias = np.arange(4, dtype=np.float32)

    network = _FakeNetwork()
    out = stage1_dit_builder._add_bidirectional_short_conv1d(
        network,
        _FakeTensor("k", dtype=_FakeTrtWithBf16Plugin.bfloat16),
        weight,
        bias,
        shape,
        frontend,
        trt_module=_FakeTrtWithBf16Plugin,
        dtype=ml_dtypes.bfloat16,
        name="blocks.0.attn.conv_k",
    )

    assert out.name == "blocks.0.attn.conv_k.output"
    assert len(network.plugins) == 1
    assert network.plugins[0].inputs[0].name == "k"
    assert len(network.convolutions) == 0
    assert creator.created[0].name == "sana_wm_short_conv"
    assert int(creator.created[0].fields["frames"][0]) == 3
    assert int(creator.created[0].fields["spatial"][0]) == 2
    assert int(creator.created[0].fields["channels"][0]) == 4
    assert int(creator.created[0].fields["kernel_size"][0]) == 3

    monkeypatch.setenv("TRTMC_SANA_WM_SHORT_CONV_PLUGIN", "0")
    fallback_network = _FakeNetwork()
    fallback = stage1_dit_builder._add_bidirectional_short_conv1d(
        fallback_network,
        _FakeTensor("k", dtype=_FakeTrtWithBf16Plugin.bfloat16),
        weight,
        bias,
        shape,
        frontend,
        trt_module=_FakeTrtWithBf16Plugin,
        dtype=ml_dtypes.bfloat16,
        name="blocks.0.attn.conv_k",
    )

    assert fallback.name == "blocks.0.attn.conv_k.output"
    assert len(fallback_network.plugins) == 0
    assert len(fallback_network.convolutions) == 2
    assert all(layer.num_groups == 4 for layer in fallback_network.convolutions)


def test_lower_sana_wm_stage1_gdn_forward_components_unrolls_frames() -> None:
    network = _FakeNetwork()
    shape = stage1_dit_builder.SanaWmStage1Shape(
        batch_size=1,
        latent_channels=2,
        latent_frames=2,
        latent_height=1,
        latent_width=2,
        text_max_length=4,
        text_embed_dim=8,
        chunk_plucker_channels=3,
    )
    frontend = stage1_dit_builder.SanaWmStage1Frontend(
        x_tokens=_FakeTensor("x_tokens"),
        token_count=4,
        hidden_size=8,
    )
    preamble = stage1_dit_builder.SanaWmStage1BlockPreamble(
        x_msa_in=_FakeTensor("x_msa"),
        qkv=_FakeTensor("qkv"),
        qkv_heads=_FakeTensor("qkv_heads"),
        q=_FakeTensor("q"),
        k=_FakeTensor("k"),
        q_rot=_FakeTensor("q_rot"),
        k_rot=_FakeTensor("k_rot"),
        v=_FakeTensor("v"),
        beta=_FakeTensor("beta"),
        decay=_FakeTensor("decay"),
        num_heads=2,
        head_dim=4,
        modulation=stage1_dit_builder.SanaWmStage1BlockModulation(
            shift_msa=_FakeTensor("shift_msa"),
            scale_msa=_FakeTensor("scale_msa"),
            gate_msa=_FakeTensor("gate_msa"),
            shift_mlp=_FakeTensor("shift_mlp"),
            scale_mlp=_FakeTensor("scale_mlp"),
            gate_mlp=_FakeTensor("gate_mlp"),
        ),
    )

    components = stage1_dit_builder.lower_sana_wm_stage1_gdn_forward_components(
        network,
        preamble,
        shape,
        frontend,
        trt_module=_FakeTrt,
        dtype=np.float16,
        name="blocks.0.attn.gdn_fwd",
    )

    assert components.num.name == "blocks.0.attn.gdn_fwd.num"
    assert components.den.name == "blocks.0.attn.gdn_fwd.den"
    assert len(network.matrix_multiply) == 12
    assert len(network.slices) == 14
    assert len(network.concatenations) == 2
    assert network.concatenations[0].axis == 3
    assert network.concatenations[1].axis == 3
    assert [layer.shape for layer in network.constants[:2]] == [
        (1, 1, 4, 4),
        (1, 1, 4, 1),
    ]
    assert [layer.dtype for layer in network.casts] == [
        "float32",
        "float32",
        "float32",
        "float32",
        "float32",
        "float32",
        "float32",
    ]
    frame_squeezes = [layer for layer in network.shuffles if layer.reshape_dims == (1, 2, 4, 2)]
    assert len(frame_squeezes) == 10
    assert any(
        getattr(layer.first_transpose, "values", None) == [0, 1, 3, 2] for layer in network.shuffles
    )


def test_create_sana_wm_gdn_plugin_is_enabled_by_default(monkeypatch) -> None:
    creator = _FakePluginCreator()
    monkeypatch.delenv("TRTMC_SANA_WM_GDN_PLUGIN", raising=False)
    monkeypatch.setattr(
        stage1_dit_builder,
        "_get_sana_wm_gdn_plugin_creator",
        lambda trt_module: creator,
    )

    plugin = stage1_dit_builder._create_sana_wm_gdn_plugin(
        _FakeTrtWithPlugin,
        mode=0,
        reverse_output=False,
    )

    assert plugin is creator.created[0]
    assert creator.created[0].name == "sana_wm_gdn_0_0"


def test_get_sana_wm_gdn_plugin_creator_uses_get_creator_registry(monkeypatch) -> None:
    creator = _FakePluginCreator()

    class _Registry:
        def get_creator(self, name: str, version: str, namespace: str = ""):  # noqa: ANN201
            assert (name, version, namespace) == ("SanaWmGdnScan", "1", "")
            return creator

    class _TrtWithGetCreator(_FakeTrtWithPlugin):
        @staticmethod
        def get_plugin_registry():  # noqa: ANN205
            return _Registry()

    monkeypatch.setattr(
        stage1_dit_builder,
        "_candidate_sana_wm_gdn_plugin_libraries",
        lambda: [],
    )
    monkeypatch.setattr(stage1_dit_builder, "_SANA_WM_GDN_PLUGIN_LOAD_ATTEMPTED", True)

    assert stage1_dit_builder._get_sana_wm_gdn_plugin_creator(_TrtWithGetCreator) is creator


def test_stage1_trt_loop_lowering_is_opt_in(monkeypatch) -> None:
    class _FakeLoopNetwork:
        def add_loop(self):  # noqa: ANN201
            raise AssertionError("capability probe should not construct a loop")

    class _FakeTrtWithLoop(_FakeTrt):
        class TripLimit:
            COUNT = "count"

        class LoopOutput:
            CONCATENATE = "concatenate"
            REVERSE = "reverse"

    network = _FakeLoopNetwork()
    monkeypatch.delenv("TRTMC_SANA_WM_TRT_LOOP", raising=False)

    assert not stage1_dit_builder._can_use_trt_loop(network, _FakeTrtWithLoop)

    monkeypatch.setenv("TRTMC_SANA_WM_TRT_LOOP", "1")

    assert stage1_dit_builder._can_use_trt_loop(network, _FakeTrtWithLoop)


def test_candidate_sana_wm_gdn_plugin_libraries_include_packaged_and_build_paths(
    monkeypatch, tmp_path
) -> None:
    source_root = tmp_path / "repo"
    module_path = (
        source_root
        / "python"
        / "tensorrt_model_connect"
        / "models"
        / "sana_wm"
        / "stage1_dit_builder.py"
    )
    package_plugin = (
        source_root / "python" / "tensorrt_model_connect" / "bin" / SANA_WM_GDN_PLUGIN_NAME
    )
    build_plugin = source_root / "build" / SANA_WM_GDN_PLUGIN_NAME
    package_plugin.parent.mkdir(parents=True)
    build_plugin.parent.mkdir(parents=True)
    package_plugin.write_bytes(b"plugin")
    build_plugin.write_bytes(b"plugin")
    env_plugin = tmp_path / "custom-plugin.so"
    monkeypatch.setattr(stage1_dit_builder, "__file__", str(module_path))
    monkeypatch.setenv("TRTMC_SANA_WM_GDN_PLUGIN_LIBRARY", str(env_plugin))

    candidates = stage1_dit_builder._candidate_sana_wm_gdn_plugin_libraries()

    assert candidates[:3] == [
        str(env_plugin),
        str(package_plugin),
        str(build_plugin),
    ]
    assert SANA_WM_GDN_PLUGIN_NAME in candidates


def test_create_sana_wm_gdn_plugin_can_be_disabled(monkeypatch) -> None:
    creator = _FakePluginCreator()
    monkeypatch.setenv("TRTMC_SANA_WM_GDN_PLUGIN", "0")
    monkeypatch.setattr(
        stage1_dit_builder,
        "_get_sana_wm_gdn_plugin_creator",
        lambda trt_module: creator,
    )

    plugin = stage1_dit_builder._create_sana_wm_gdn_plugin(
        _FakeTrtWithPlugin,
        mode=0,
        reverse_output=False,
    )

    assert plugin is None
    assert creator.created == []


def test_lower_sana_wm_stage1_gdn_forward_components_uses_registered_plugin(
    monkeypatch,
) -> None:
    network = _FakeNetwork()
    creator = _FakePluginCreator()
    monkeypatch.setenv("TRTMC_SANA_WM_GDN_PLUGIN", "1")
    monkeypatch.setattr(
        stage1_dit_builder,
        "_get_sana_wm_gdn_plugin_creator",
        lambda trt_module: creator,
    )
    shape = stage1_dit_builder.SanaWmStage1Shape(
        batch_size=1,
        latent_channels=2,
        latent_frames=2,
        latent_height=1,
        latent_width=2,
        text_max_length=4,
        text_embed_dim=8,
        chunk_plucker_channels=3,
    )
    frontend = stage1_dit_builder.SanaWmStage1Frontend(
        x_tokens=_FakeTensor("x_tokens"),
        token_count=4,
        hidden_size=8,
    )
    preamble = stage1_dit_builder.SanaWmStage1BlockPreamble(
        x_msa_in=_FakeTensor("x_msa"),
        qkv=_FakeTensor("qkv"),
        qkv_heads=_FakeTensor("qkv_heads"),
        q=_FakeTensor("q"),
        k=_FakeTensor("k"),
        q_rot=_FakeTensor("q_rot"),
        k_rot=_FakeTensor("k_rot"),
        v=_FakeTensor("v"),
        beta=_FakeTensor("beta"),
        decay=_FakeTensor("decay"),
        num_heads=2,
        head_dim=4,
        modulation=stage1_dit_builder.SanaWmStage1BlockModulation(
            shift_msa=_FakeTensor("shift_msa"),
            scale_msa=_FakeTensor("scale_msa"),
            gate_msa=_FakeTensor("gate_msa"),
            shift_mlp=_FakeTensor("shift_mlp"),
            scale_mlp=_FakeTensor("scale_mlp"),
            gate_mlp=_FakeTensor("gate_mlp"),
        ),
    )

    components = stage1_dit_builder.lower_sana_wm_stage1_gdn_forward_components(
        network,
        preamble,
        shape,
        frontend,
        trt_module=_FakeTrtWithPlugin,
        dtype=np.float16,
        name="blocks.0.attn.gdn_fwd",
        reverse_output=True,
    )

    assert components.num.name == "blocks.0.attn.gdn_fwd.num"
    assert components.den.name == "blocks.0.attn.gdn_fwd.den"
    assert len(network.plugins) == 1
    assert len(network.plugins[0].inputs) == 7
    assert creator.created[0].name == "sana_wm_gdn_0_1"
    assert int(creator.created[0].fields["mode"][0]) == 0
    assert int(creator.created[0].fields["reverse_output"][0]) == 1
    assert len(network.slices) == 0
    assert len(network.concatenations) == 0


def test_lower_sana_wm_stage1_gdn_core_prefers_combined_plugin(monkeypatch) -> None:
    network = _FakeNetwork()
    creator = _FakePluginCreator()
    monkeypatch.setenv("TRTMC_SANA_WM_GDN_PLUGIN", "1")
    monkeypatch.setattr(
        stage1_dit_builder,
        "_get_sana_wm_gdn_plugin_creator",
        lambda trt_module: creator,
    )
    shape = stage1_dit_builder.SanaWmStage1Shape(
        batch_size=1,
        latent_channels=2,
        latent_frames=2,
        latent_height=1,
        latent_width=2,
        text_max_length=4,
        text_embed_dim=8,
        chunk_plucker_channels=3,
    )
    frontend = stage1_dit_builder.SanaWmStage1Frontend(
        x_tokens=_FakeTensor("x_tokens"),
        token_count=4,
        hidden_size=8,
    )
    preamble = stage1_dit_builder.SanaWmStage1BlockPreamble(
        x_msa_in=_FakeTensor("x_msa"),
        qkv=_FakeTensor("qkv"),
        qkv_heads=_FakeTensor("qkv_heads"),
        q=_FakeTensor("q"),
        k=_FakeTensor("k"),
        q_rot=_FakeTensor("q_rot"),
        k_rot=_FakeTensor("k_rot"),
        v=_FakeTensor("v"),
        beta=_FakeTensor("beta"),
        decay=_FakeTensor("decay"),
        num_heads=2,
        head_dim=4,
        modulation=stage1_dit_builder.SanaWmStage1BlockModulation(
            shift_msa=_FakeTensor("shift_msa"),
            scale_msa=_FakeTensor("scale_msa"),
            gate_msa=_FakeTensor("gate_msa"),
            shift_mlp=_FakeTensor("shift_mlp"),
            scale_mlp=_FakeTensor("scale_mlp"),
            gate_mlp=_FakeTensor("gate_mlp"),
        ),
    )

    core = stage1_dit_builder.lower_sana_wm_stage1_bidirectional_gdn_core(
        network,
        preamble,
        shape,
        frontend,
        _raw_sana_wm_config(),
        trt_module=_FakeTrtWithPlugin,
        dtype=np.float16,
        name="blocks.0.attn.gdn",
    )

    assert core.tokens.name == "blocks.0.attn.gdn.tokens"
    assert len(network.plugins) == 1
    assert len(network.plugins[0].inputs) == 7
    assert creator.created[0].name == "sana_wm_gdn_2_0"
    assert int(creator.created[0].fields["mode"][0]) == 2
    assert int(creator.created[0].fields["reverse_output"][0]) == 0
    assert len(network.slices) == 0
    assert len(network.concatenations) == 0


def test_lower_sana_wm_stage1_gdn_core_uses_raw_fused_plugin(monkeypatch) -> None:
    network = _FakeNetwork()
    creator = _FakePluginCreator()
    monkeypatch.setenv("TRTMC_SANA_WM_GDN_PLUGIN", "1")
    monkeypatch.delenv("TRTMC_SANA_WM_RAW_GDN_PLUGIN", raising=False)
    monkeypatch.setattr(
        stage1_dit_builder,
        "_get_sana_wm_gdn_plugin_creator",
        lambda trt_module: creator,
    )
    shape = stage1_dit_builder.SanaWmStage1Shape(
        batch_size=1,
        latent_channels=2,
        latent_frames=2,
        latent_height=1,
        latent_width=2,
        text_max_length=4,
        text_embed_dim=8,
        chunk_plucker_channels=3,
    )
    frontend = stage1_dit_builder.SanaWmStage1Frontend(
        x_tokens=_FakeTensor("x_tokens"),
        token_count=4,
        hidden_size=8,
    )
    preamble = stage1_dit_builder.SanaWmStage1BlockPreamble(
        x_msa_in=_FakeTensor("x_msa"),
        qkv=_FakeTensor("qkv"),
        qkv_heads=_FakeTensor("qkv_heads"),
        q=_FakeTensor("q"),
        k=_FakeTensor("k"),
        q_rot=_FakeTensor("q_rot"),
        k_rot=_FakeTensor("k_rot"),
        v=_FakeTensor("v"),
        beta=_FakeTensor("beta"),
        decay=_FakeTensor("decay"),
        num_heads=2,
        head_dim=4,
        modulation=stage1_dit_builder.SanaWmStage1BlockModulation(
            shift_msa=_FakeTensor("shift_msa"),
            scale_msa=_FakeTensor("scale_msa"),
            gate_msa=_FakeTensor("gate_msa"),
            shift_mlp=_FakeTensor("shift_mlp"),
            scale_mlp=_FakeTensor("scale_mlp"),
            gate_mlp=_FakeTensor("gate_mlp"),
        ),
        q_raw=_FakeTensor("q_raw"),
        k_conv=_FakeTensor("k_conv"),
        v_raw=_FakeTensor("v_raw"),
    )
    q_norm_weight = np.asarray(
        [1.001, -0.3333, 0.0, 2.0001, 0.875, -1.1251, 3.14159, -4.25],
        dtype=np.float32,
    )
    k_norm_weight = np.asarray(
        [-1.001, 0.3333, 1.5, -2.0001, 0.125, 1.1251, -3.14159, 4.25],
        dtype=np.float32,
    )
    weights = WeightDict(
        {
            "blocks.0.attn.q_norm.weight": q_norm_weight,
            "blocks.0.attn.k_norm.weight": k_norm_weight,
        }
    )

    core = stage1_dit_builder.lower_sana_wm_stage1_bidirectional_gdn_core(
        network,
        preamble,
        shape,
        frontend,
        _raw_sana_wm_config(),
        weights=weights,
        block_index=0,
        trt_module=_FakeTrtWithPlugin,
        dtype=np.float16,
        name="blocks.0.attn.gdn",
    )

    assert core.tokens.name == "blocks.0.attn.gdn.tokens"
    assert len(network.plugins) == 1
    assert len(network.plugins[0].inputs) == 9
    assert creator.created[0].name == "sana_wm_gdn_3_0"
    assert int(creator.created[0].fields["mode"][0]) == 3
    assert int(creator.created[0].fields["frames"][0]) == 2
    assert int(creator.created[0].fields["head_dim"][0]) == 4
    assert np.isclose(float(creator.created[0].fields["eps"][0]), 1.0e-15, rtol=0.0, atol=1.0e-20)
    assert np.isclose(float(creator.created[0].fields["norm_eps"][0]), 1.0e-5)
    np.testing.assert_array_equal(
        network.constants[0].weights.value,
        q_norm_weight,
    )
    np.testing.assert_array_equal(
        network.constants[1].weights.value,
        k_norm_weight,
    )

    monkeypatch.setenv("TRTMC_SANA_WM_RAW_GDN_PLUGIN", "0")
    fallback_network = _FakeNetwork()
    fallback_core = stage1_dit_builder.lower_sana_wm_stage1_bidirectional_gdn_core(
        fallback_network,
        preamble,
        shape,
        frontend,
        _raw_sana_wm_config(),
        weights=weights,
        block_index=0,
        trt_module=_FakeTrtWithPlugin,
        dtype=np.float16,
        name="blocks.0.attn.gdn",
    )

    assert fallback_core.tokens.name == "blocks.0.attn.gdn.tokens"
    assert len(fallback_network.plugins) == 1
    assert len(fallback_network.plugins[0].inputs) == 7
    assert creator.created[1].name == "sana_wm_gdn_2_0"
    assert int(creator.created[1].fields["mode"][0]) == 2


def test_add_t2i_modulate_uses_registered_bf16_plugin(monkeypatch) -> None:
    ml_dtypes = pytest.importorskip("ml_dtypes")
    monkeypatch.setenv("TRTMC_SANA_WM_T2I_MODULATE_PLUGIN", "1")
    network = _FakeNetwork()
    creator = _FakePluginCreator()
    monkeypatch.setattr(
        stage1_dit_builder,
        "_get_sana_wm_t2i_modulate_plugin_creator",
        lambda trt_module: creator,
    )
    inp = _FakeTensor("norm1", dtype=_FakeTrtWithBf16Plugin.bfloat16)
    shift = _FakeTensor("shift_msa", dtype=_FakeTrtWithBf16Plugin.bfloat16)
    scale = _FakeTensor("scale_msa", dtype=_FakeTrtWithBf16Plugin.bfloat16)

    out = stage1_dit_builder._add_t2i_modulate(
        network,
        inp,
        shift,
        scale,
        rank=4,
        trt_module=_FakeTrtWithBf16Plugin,
        dtype=ml_dtypes.bfloat16,
        name="blocks.0.x_msa_4d",
    )

    assert out.name == "blocks.0.x_msa_4d"
    assert len(network.plugins) == 1
    assert network.plugins[0].inputs == [inp, shift, scale]
    assert creator.created[0].name == "sana_wm_t2i_modulate"
    assert len(network.elementwise) == 0
    assert len(network.casts) == 0


def test_lower_sana_wm_stage1_camera_forward_uses_registered_plugin(monkeypatch) -> None:
    network = _FakeNetwork()
    creator = _FakePluginCreator()
    monkeypatch.setenv("TRTMC_SANA_WM_GDN_PLUGIN", "1")
    monkeypatch.setattr(
        stage1_dit_builder,
        "_get_sana_wm_gdn_plugin_creator",
        lambda trt_module: creator,
    )
    shape = stage1_dit_builder.SanaWmStage1Shape(
        batch_size=1,
        latent_channels=2,
        latent_frames=2,
        latent_height=1,
        latent_width=2,
        text_max_length=4,
        text_embed_dim=8,
        chunk_plucker_channels=3,
    )
    frontend = stage1_dit_builder.SanaWmStage1Frontend(
        x_tokens=_FakeTensor("x_tokens"),
        token_count=4,
        hidden_size=8,
    )
    camera = stage1_dit_builder.SanaWmStage1CameraUcpe(
        q_rot=_FakeTensor("q_rot"),
        k_rot=_FakeTensor("k_rot"),
        v=_FakeTensor("v"),
        beta=_FakeTensor("beta"),
        num_heads=2,
        head_dim=4,
    )

    out = stage1_dit_builder.lower_sana_wm_stage1_camera_single_path_forward(
        network,
        camera,
        _FakeTensor("decay"),
        shape,
        frontend,
        trt_module=_FakeTrtWithPlugin,
        dtype=np.float16,
        name="blocks.0.attn.cam_gdn_fwd",
        reverse_output=False,
    )

    assert out.name == "blocks.0.attn.cam_gdn_fwd"
    assert len(network.plugins) == 1
    assert len(network.plugins[0].inputs) == 5
    assert creator.created[0].name == "sana_wm_gdn_1_0"
    assert int(creator.created[0].fields["mode"][0]) == 1
    assert int(creator.created[0].fields["reverse_output"][0]) == 0
    assert len(network.slices) == 0
    assert len(network.concatenations) == 0


def test_lower_sana_wm_stage1_camera_combined_plugin_can_be_disabled(monkeypatch) -> None:
    creator = _FakePluginCreator()
    monkeypatch.setenv("TRTMC_SANA_WM_GDN_PLUGIN", "1")
    monkeypatch.delenv("TRTMC_SANA_WM_CAMERA_COMBINED_PLUGIN", raising=False)
    monkeypatch.setattr(
        stage1_dit_builder,
        "_get_sana_wm_gdn_plugin_creator",
        lambda trt_module: creator,
    )
    shape = stage1_dit_builder.SanaWmStage1Shape(
        batch_size=1,
        latent_channels=2,
        latent_frames=2,
        latent_height=1,
        latent_width=2,
        text_max_length=4,
        text_embed_dim=8,
        chunk_plucker_channels=3,
    )
    frontend = stage1_dit_builder.SanaWmStage1Frontend(
        x_tokens=_FakeTensor("x_tokens"),
        token_count=4,
        hidden_size=8,
    )
    camera = stage1_dit_builder.SanaWmStage1CameraUcpe(
        q_rot=_FakeTensor("q_rot"),
        k_rot=_FakeTensor("k_rot"),
        v=_FakeTensor("v"),
        beta=_FakeTensor("beta"),
        num_heads=2,
        head_dim=4,
    )

    enabled_network = _FakeNetwork()
    out = stage1_dit_builder._lower_sana_wm_stage1_camera_single_path_combined_plugin(
        enabled_network,
        camera,
        _FakeTensor("decay"),
        shape,
        frontend,
        trt_module=_FakeTrtWithPlugin,
        dtype=np.float16,
        name="blocks.0.attn.cam_gdn.combined_scan",
    )

    assert out.name == "blocks.0.attn.cam_gdn.combined_scan"
    assert len(enabled_network.plugins) == 1
    assert len(enabled_network.plugins[0].inputs) == 5
    assert creator.created[0].name == "sana_wm_gdn_4_0"
    assert int(creator.created[0].fields["mode"][0]) == 4

    monkeypatch.setenv("TRTMC_SANA_WM_CAMERA_COMBINED_PLUGIN", "0")
    disabled_network = _FakeNetwork()
    assert (
        stage1_dit_builder._lower_sana_wm_stage1_camera_single_path_combined_plugin(
            disabled_network,
            camera,
            _FakeTensor("decay"),
            shape,
            frontend,
            trt_module=_FakeTrtWithPlugin,
            dtype=np.float16,
            name="blocks.0.attn.cam_gdn.combined_scan",
        )
        is None
    )
    assert len(disabled_network.plugins) == 0

    monkeypatch.setenv("TRTMC_SANA_WM_CAMERA_COMBINED_PLUGIN", "1")
    reenabled_network = _FakeNetwork()
    out = stage1_dit_builder._lower_sana_wm_stage1_camera_single_path_combined_plugin(
        reenabled_network,
        camera,
        _FakeTensor("decay"),
        shape,
        frontend,
        trt_module=_FakeTrtWithPlugin,
        dtype=np.float16,
        name="blocks.0.attn.cam_gdn.combined_scan",
    )

    assert out.name == "blocks.0.attn.cam_gdn.combined_scan"
    assert len(reenabled_network.plugins) == 1


def test_lower_sana_wm_stage1_camera_ucpe_v_mode_uses_fused_v_only(monkeypatch) -> None:
    ml_dtypes = pytest.importorskip("ml_dtypes")
    creator = _FakePluginCreator()
    monkeypatch.setenv("TRTMC_SANA_WM_CAMERA_PREP_PLUGIN", "v")
    monkeypatch.setattr(
        stage1_dit_builder,
        "_get_sana_wm_cam_prep_plugin_creator",
        lambda trt_module: creator,
    )
    shape = stage1_dit_builder.SanaWmStage1Shape(
        batch_size=1,
        latent_channels=2,
        latent_frames=2,
        latent_height=1,
        latent_width=2,
        text_max_length=4,
        text_embed_dim=8,
        chunk_plucker_channels=3,
    )
    preamble = stage1_dit_builder.SanaWmStage1BlockPreamble(
        x_msa_in=_FakeTensor("x_msa"),
        qkv=_FakeTensor("qkv"),
        qkv_heads=_FakeTensor("qkv_heads"),
        q=_FakeTensor("q"),
        k=_FakeTensor("k"),
        q_rot=_FakeTensor("q_rot"),
        k_rot=_FakeTensor("k_rot"),
        v=_FakeTensor("v"),
        beta=_FakeTensor("beta"),
        decay=_FakeTensor("decay"),
        num_heads=1,
        head_dim=8,
        modulation=stage1_dit_builder.SanaWmStage1BlockModulation(
            shift_msa=_FakeTensor("shift_msa"),
            scale_msa=_FakeTensor("scale_msa"),
            gate_msa=_FakeTensor("gate_msa"),
            shift_mlp=_FakeTensor("shift_mlp"),
            scale_mlp=_FakeTensor("scale_mlp"),
            gate_mlp=_FakeTensor("gate_mlp"),
        ),
    )
    camera = stage1_dit_builder.SanaWmStage1CameraPreamble(
        q=_FakeTensor("camera_q"),
        k=_FakeTensor("camera_k"),
        v=_FakeTensor("camera_v"),
        num_heads=1,
        head_dim=8,
        q_raw=_FakeTensor("camera_q_raw", dtype=_FakeTrtWithBf16Plugin.bfloat16),
        k_raw=_FakeTensor("camera_k_raw", dtype=_FakeTrtWithBf16Plugin.bfloat16),
        v_raw=_FakeTensor("camera_v_raw", dtype=_FakeTrtWithBf16Plugin.bfloat16),
        q_norm_weight=np.ones((8,), dtype=np.float32),
        k_norm_weight=np.ones((8,), dtype=np.float32),
    )
    network = _FakeNetwork()

    out = stage1_dit_builder.lower_sana_wm_stage1_camera_ucpe(
        network,
        camera,
        preamble,
        {
            "raymats": _FakeTensor("raymats", dtype=_FakeTrtWithBf16Plugin.bfloat16),
            "raymats_inv": _FakeTensor("raymats_inv", dtype=_FakeTrtWithBf16Plugin.bfloat16),
        },
        shape,
        _raw_sana_wm_config(),
        trt_module=_FakeTrtWithBf16Plugin,
        dtype=ml_dtypes.bfloat16,
        name="blocks.0.attn.cam_ucpe",
    )

    assert out.q_rot.name == "blocks.0.attn.cam_ucpe.q_bhdn"
    assert out.k_rot.name == "blocks.0.attn.cam_ucpe.k_bhdn"
    assert out.v.name == "blocks.0.attn.cam_ucpe.prep_plugin.v"
    assert len(network.plugins) == 1
    assert len(network.plugins[0].inputs) == 9
    assert network.plugins[0].inputs[:3] == [camera.q_raw, camera.k_raw, camera.v_raw]
    assert creator.created[0].name == "sana_wm_cam_prep"
    assert int(creator.created[0].fields["frames"][0]) == 2
    assert int(creator.created[0].fields["spatial"][0]) == 2
    assert int(creator.created[0].fields["heads"][0]) == 1
    assert int(creator.created[0].fields["head_dim"][0]) == 8
    assert np.isclose(float(creator.created[0].fields["norm_eps"][0]), 1.0e-5)


def test_lower_sana_wm_stage1_camera_preamble_reaches_ucpe_inputs() -> None:
    network = _FakeNetwork()
    weights = _stage1_weights()
    shape = stage1_dit_builder.stage1_shape_from_config(
        _raw_sana_wm_config(),
        weights,
    )
    frontend = stage1_dit_builder.SanaWmStage1Frontend(
        x_tokens=_FakeTensor("x_tokens"),
        token_count=41 * 22 * 40,
        hidden_size=2240,
        plucker_tokens=_FakeTensor("plucker_tokens"),
    )

    camera = stage1_dit_builder.lower_sana_wm_stage1_camera_preamble(
        network,
        _FakeTensor("x_msa"),
        shape,
        frontend,
        weights,
        _raw_sana_wm_config(),
        block_index=0,
        trt_module=_FakeTrt,
        dtype=np.float16,
    )

    assert camera.q.name == "blocks.0.attn.q_cam_bhdn"
    assert camera.k.name == "blocks.0.attn.k_cam_bhdn"
    assert camera.v.name == "blocks.0.attn.v_cam_bhdn"
    assert camera.num_heads == 20
    assert camera.head_dim == 112
    assert len(network.matrix_multiply) == 3
    assert len(network.convolutions) == 2
    assert all(layer.num_groups == 2240 for layer in network.convolutions)
    assert len(network.reductions) == 2
    assert [layer.activation_type for layer in network.activations] == [
        "relu",
        "relu",
    ]
    assert any(layer.shape == (1, 2240, 2240) for layer in network.constants)
    assert any(
        getattr(layer.first_transpose, "values", None) == [0, 2, 3, 1] for layer in network.shuffles
    )


def test_lower_sana_wm_stage1_camera_ucpe_stabilizes_and_discounts_beta() -> None:
    network = _FakeNetwork()
    shape = stage1_dit_builder.SanaWmStage1Shape(
        batch_size=1,
        latent_channels=2,
        latent_frames=2,
        latent_height=1,
        latent_width=2,
        text_max_length=4,
        text_embed_dim=8,
        chunk_plucker_channels=3,
    )
    preamble = stage1_dit_builder.SanaWmStage1BlockPreamble(
        x_msa_in=_FakeTensor("x_msa"),
        qkv=_FakeTensor("qkv"),
        qkv_heads=_FakeTensor("qkv_heads"),
        q=_FakeTensor("q"),
        k=_FakeTensor("k"),
        q_rot=_FakeTensor("q_rot"),
        k_rot=_FakeTensor("k_rot"),
        v=_FakeTensor("v"),
        beta=_FakeTensor("beta"),
        decay=_FakeTensor("decay"),
        num_heads=2,
        head_dim=8,
        modulation=stage1_dit_builder.SanaWmStage1BlockModulation(
            shift_msa=_FakeTensor("shift_msa"),
            scale_msa=_FakeTensor("scale_msa"),
            gate_msa=_FakeTensor("gate_msa"),
            shift_mlp=_FakeTensor("shift_mlp"),
            scale_mlp=_FakeTensor("scale_mlp"),
            gate_mlp=_FakeTensor("gate_mlp"),
        ),
    )
    camera = stage1_dit_builder.SanaWmStage1CameraPreamble(
        q=_FakeTensor("q_cam"),
        k=_FakeTensor("k_cam"),
        v=_FakeTensor("v_cam"),
        num_heads=2,
        head_dim=8,
    )

    ucpe = stage1_dit_builder.lower_sana_wm_stage1_camera_ucpe(
        network,
        camera,
        preamble,
        {"raymats": _FakeTensor("raymats"), "raymats_inv": _FakeTensor("raymats_inv")},
        shape,
        _raw_sana_wm_config(),
        trt_module=_FakeTrt,
        dtype=np.float16,
        name="blocks.0.attn.cam_ucpe",
    )

    assert ucpe.q_rot.name == "blocks.0.attn.cam_ucpe.q_bhdn"
    assert ucpe.k_rot.name == "blocks.0.attn.cam_ucpe.k_bhdn"
    assert ucpe.v.name == "blocks.0.attn.cam_ucpe.v_bhdn"
    assert ucpe.beta.name == "blocks.0.attn.cam_ucpe.beta_discounted"
    assert len(network.matrix_multiply) == 4
    assert any(getattr(layer.inp, "name", "") == "raymats" for layer in network.casts)
    assert not any(getattr(layer.inp, "name", "") == "raymats_inv" for layer in network.casts)
    assert [layer.op for layer in network.reductions].count("avg") >= 7
    assert [layer.op for layer in network.reductions].count("sum") == 2
    assert any(layer.op == "max" for layer in network.elementwise)
    assert any(layer.op == "min" for layer in network.elementwise)
    assert any(layer.reshape_dims == (1, 1, 4, 1, 4, 4) for layer in network.shuffles)
    assert any(layer.reshape_dims == (1, 2, 4, 1, 4, 1) for layer in network.shuffles)


def test_apply_ucpe_block_uses_registered_plugin_by_default(monkeypatch) -> None:
    network = _FakeNetwork()
    creator = _FakePluginCreator()
    monkeypatch.setattr(
        stage1_dit_builder,
        "_get_sana_wm_ucpe_plugin_creator",
        lambda trt_module: creator,
    )
    shape = stage1_dit_builder.SanaWmStage1Shape(
        batch_size=1,
        latent_channels=2,
        latent_frames=2,
        latent_height=1,
        latent_width=2,
        text_max_length=4,
        text_embed_dim=8,
        chunk_plucker_channels=3,
    )

    out = stage1_dit_builder._apply_ucpe_block_diagonal_to_bhnd(
        network,
        _FakeTensor("feats", dtype=_FakeTrt.float16),
        _FakeTensor("raymats", dtype=_FakeTrt.float32),
        _FakeTensor("cos", dtype=_FakeTrt.float32),
        _FakeTensor("sin", dtype=_FakeTrt.float32),
        shape,
        num_heads=2,
        head_dim=8,
        inverse_rope=True,
        trt_module=_FakeTrtWithPlugin,
        dtype=np.float16,
        name="blocks.0.attn.cam_ucpe.k",
        rms_downscale=True,
    )

    assert out.name == "blocks.0.attn.cam_ucpe.k"
    assert len(network.plugins) == 1
    assert len(network.plugins[0].inputs) == 4
    assert len(network.matrix_multiply) == 0
    assert len(network.slices) == 0
    assert len(network.concatenations) == 0
    assert len(network.reductions) == 0
    assert creator.created[0].name == "sana_wm_ucpe"
    assert int(creator.created[0].fields["frames"][0]) == 2
    assert int(creator.created[0].fields["spatial"][0]) == 2
    assert int(creator.created[0].fields["heads"][0]) == 2
    assert int(creator.created[0].fields["head_dim"][0]) == 8
    assert int(creator.created[0].fields["inverse"][0]) == 1
    assert int(creator.created[0].fields["downscale"][0]) == 1


def test_lower_sana_wm_stage1_camera_ucpe_can_skip_stabilization_for_both_triton() -> None:
    network = _FakeNetwork()
    shape = stage1_dit_builder.SanaWmStage1Shape(
        batch_size=1,
        latent_channels=2,
        latent_frames=2,
        latent_height=1,
        latent_width=2,
        text_max_length=4,
        text_embed_dim=8,
        chunk_plucker_channels=3,
    )
    preamble = stage1_dit_builder.SanaWmStage1BlockPreamble(
        x_msa_in=_FakeTensor("x_msa"),
        qkv=_FakeTensor("qkv"),
        qkv_heads=_FakeTensor("qkv_heads"),
        q=_FakeTensor("q"),
        k=_FakeTensor("k"),
        q_rot=_FakeTensor("q_rot"),
        k_rot=_FakeTensor("k_rot"),
        v=_FakeTensor("v"),
        beta=_FakeTensor("beta"),
        decay=_FakeTensor("decay"),
        num_heads=2,
        head_dim=8,
        modulation=stage1_dit_builder.SanaWmStage1BlockModulation(
            shift_msa=_FakeTensor("shift_msa"),
            scale_msa=_FakeTensor("scale_msa"),
            gate_msa=_FakeTensor("gate_msa"),
            shift_mlp=_FakeTensor("shift_mlp"),
            scale_mlp=_FakeTensor("scale_mlp"),
            gate_mlp=_FakeTensor("gate_mlp"),
        ),
    )
    camera = stage1_dit_builder.SanaWmStage1CameraPreamble(
        q=_FakeTensor("q_cam"),
        k=_FakeTensor("k_cam"),
        v=_FakeTensor("v_cam"),
        num_heads=2,
        head_dim=8,
    )

    ucpe = stage1_dit_builder.lower_sana_wm_stage1_camera_ucpe(
        network,
        camera,
        preamble,
        {"raymats": _FakeTensor("raymats"), "raymats_inv": _FakeTensor("raymats_inv")},
        shape,
        _raw_sana_wm_config(),
        trt_module=_FakeTrt,
        dtype=np.float16,
        name="blocks.0.attn.cam_ucpe",
        stabilize_transforms=False,
    )

    assert ucpe.beta.name == "blocks.0.attn.cam_ucpe.beta_discounted"
    assert [layer.op for layer in network.reductions].count("avg") == 1
    assert [layer.op for layer in network.reductions].count("sum") == 2
    assert not any(layer.op == "min" for layer in network.elementwise)


def test_stage1_camera_ucpe_softmax_uses_upstream_unstabilized_transforms() -> None:
    assert not stage1_dit_builder._stage1_stabilizes_camera_ucpe(
        _raw_sana_wm_config(),
        softmax_attention=True,
    )


def test_lower_sana_wm_stage1_camera_softmax_uses_hf_padded_scale() -> None:
    network = _FakeNetwork()

    out = stage1_dit_builder._lower_softmax_attention_bhnd(
        network,
        _FakeTensor("q"),
        _FakeTensor("k"),
        _FakeTensor("v"),
        head_dim=112,
        scale_head_dim=128,
        trt_module=_FakeTrt,
        dtype=np.float16,
        name="blocks.3.attn.cam_softmax.out_bhnd",
    )

    scales = [
        float(np.asarray(layer.weights.value).reshape(-1)[0])
        for layer in network.constants
        if layer.shape == (1, 1, 1, 1)
    ]
    assert out.name == "blocks.3.attn.cam_softmax.out_bhnd"
    assert len(network.attentions) == 1
    assert len(network.matrix_multiply) == 0
    assert network.attentions[0].q is network.elementwise[-1].get_output(0)
    assert np.isclose(scales[-1], np.float16(128**-0.5))
    assert not np.isclose(scales[-1], np.float16(112**-0.5))


def test_lower_sana_wm_stage1_block_post_attention_lowers_cross_plucker_and_mlp() -> None:
    network = _FakeNetwork()
    shape = stage1_dit_builder.SanaWmStage1Shape(
        batch_size=1,
        latent_channels=2,
        latent_frames=2,
        latent_height=1,
        latent_width=2,
        text_max_length=4,
        text_embed_dim=8,
        chunk_plucker_channels=3,
    )
    weights = WeightDict()
    weights["blocks.0.plucker_proj.weight"] = np.zeros((8, 8), dtype=np.float16)
    weights["blocks.0.plucker_proj.bias"] = np.zeros((8,), dtype=np.float16)
    weights["blocks.0.cross_attn.q_linear.weight"] = np.zeros((8, 8), dtype=np.float16)
    weights["blocks.0.cross_attn.q_linear.bias"] = np.zeros((8,), dtype=np.float16)
    weights["blocks.0.cross_attn.kv_linear.weight"] = np.zeros((8, 16), dtype=np.float16)
    weights["blocks.0.cross_attn.kv_linear.bias"] = np.zeros((16,), dtype=np.float16)
    weights["blocks.0.cross_attn.q_norm.weight"] = np.ones((8,), dtype=np.float32)
    weights["blocks.0.cross_attn.k_norm.weight"] = np.ones((8,), dtype=np.float32)
    weights["blocks.0.cross_attn.proj.weight"] = np.zeros((8, 8), dtype=np.float16)
    weights["blocks.0.cross_attn.proj.bias"] = np.zeros((8,), dtype=np.float16)
    weights["blocks.0.mlp.inverted_conv.conv.weight"] = np.zeros((48, 8, 1, 1), dtype=np.float16)
    weights["blocks.0.mlp.inverted_conv.conv.bias"] = np.zeros((48,), dtype=np.float16)
    weights["blocks.0.mlp.depth_conv.conv.weight"] = np.zeros((48, 1, 3, 3), dtype=np.float16)
    weights["blocks.0.mlp.depth_conv.conv.bias"] = np.zeros((48,), dtype=np.float16)
    weights["blocks.0.mlp.point_conv.conv.weight"] = np.zeros((8, 24, 1, 1), dtype=np.float16)
    weights["blocks.0.mlp.t_conv.weight"] = np.zeros((8, 8, 3, 1), dtype=np.float16)
    preamble = stage1_dit_builder.SanaWmStage1BlockPreamble(
        x_msa_in=_FakeTensor("x_msa"),
        qkv=_FakeTensor("qkv"),
        qkv_heads=_FakeTensor("qkv_heads"),
        q=_FakeTensor("q"),
        k=_FakeTensor("k"),
        q_rot=_FakeTensor("q_rot"),
        k_rot=_FakeTensor("k_rot"),
        v=_FakeTensor("v"),
        beta=_FakeTensor("beta"),
        decay=_FakeTensor("decay"),
        num_heads=2,
        head_dim=4,
        modulation=stage1_dit_builder.SanaWmStage1BlockModulation(
            shift_msa=_FakeTensor("shift_msa"),
            scale_msa=_FakeTensor("scale_msa"),
            gate_msa=_FakeTensor("gate_msa"),
            shift_mlp=_FakeTensor("shift_mlp"),
            scale_mlp=_FakeTensor("scale_mlp"),
            gate_mlp=_FakeTensor("gate_mlp"),
        ),
    )
    conditioning = stage1_dit_builder.SanaWmStage1Conditioning(
        t=_FakeTensor("t"),
        t0=_FakeTensor("t0"),
        y=_FakeTensor("y"),
        mask=_FakeTensor("mask"),
    )
    frontend = stage1_dit_builder.SanaWmStage1Frontend(
        x_tokens=_FakeTensor("x_tokens"),
        token_count=4,
        hidden_size=8,
        plucker_tokens=_FakeTensor("plucker_tokens"),
    )

    out = stage1_dit_builder.lower_sana_wm_stage1_block_post_attention(
        network,
        _FakeTensor("block_input"),
        _FakeTensor("attn_tokens"),
        preamble,
        conditioning,
        shape,
        frontend,
        weights,
        {
            "model": {
                "cross_norm": True,
                "ffn_type": "GLUMBConvTemp",
                "mlp_ratio": 3,
                "t_kernel_size": 3,
                "use_chunk_plucker_post_attn": True,
            }
        },
        block_index=0,
        trt_module=_FakeTrt,
        dtype=np.float16,
    )

    assert out.name == "blocks.0.output"
    assert len(network.softmax) == 1
    assert network.softmax[0].axes == 1 << 3
    assert len(network.convolutions) == 4
    assert network.convolutions[1].num_groups == 48
    assert network.convolutions[-1].kernel_shape == (3, 1)
    assert len(network.matrix_multiply) == 6
    assert any(layer.reshape_dims == (1, 4, 8) for layer in network.shuffles)


def test_lower_sana_wm_stage1_final_layer_matches_upstream_frame_aware_path() -> None:
    network = _FakeNetwork()
    weights = _stage1_weights()
    shape = stage1_dit_builder.stage1_shape_from_config(
        _raw_sana_wm_config(),
        weights,
    )
    frontend = stage1_dit_builder.SanaWmStage1Frontend(
        x_tokens=_FakeTensor("x_tokens"),
        token_count=41 * 22 * 40,
        hidden_size=2240,
        plucker_tokens=_FakeTensor("plucker_tokens"),
    )
    conditioning = stage1_dit_builder.SanaWmStage1Conditioning(
        t=_FakeTensor("t_embedder.output"),
        t0=_FakeTensor("t_block.output"),
        y=_FakeTensor("y_embedder.output"),
        mask=_FakeTensor("mask"),
    )

    final_output = stage1_dit_builder.lower_sana_wm_stage1_final_layer(
        network,
        _FakeTensor("block_output"),
        conditioning,
        shape,
        frontend,
        weights,
        _raw_sana_wm_config(),
        trt_module=_FakeTrt,
        dtype=np.float16,
    )

    assert final_output.tokens.name == "final_layer.tokens"
    assert final_output.latents.name == "final_layer.latents"
    assert len(network.reductions) == 2
    assert [layer.op for layer in network.reductions] == ["avg", "avg"]
    assert all(layer.axes == 8 for layer in network.reductions)
    assert [layer.op for layer in network.unary] == ["sqrt", "recip"]
    assert network.constants[-2].shape == (1, 2240, 128)
    assert network.constants[-1].shape == (1, 1, 128)
    assert network.shuffles[0].reshape_dims == (2, 41, 22 * 40, 2240)
    assert network.shuffles[1].first_transpose.values == [0, 2, 1, 3]
    assert network.shuffles[-2].reshape_dims == (2, 41, 22, 40, 128)
    assert network.shuffles[-2].second_transpose.values == [0, 4, 1, 2, 3]
    assert network.shuffles[-1].reshape_dims == (2, 128, 41, 22, 40)


def test_build_sana_wm_stage1_dit_engine_starts_direct_trt_network(monkeypatch) -> None:
    _FakeBuilder.last = None
    monkeypatch.setattr(stage1_dit_builder.trt_compat, "get_trt", lambda: _FakeTrt)
    stage1_dit_builder._BF16_WEIGHT_REFS.append(np.zeros((1,), dtype=np.float32))

    plan = stage1_dit_builder.build_sana_wm_stage1_dit_engine(
        _stage1_weights(),
        _raw_sana_wm_config(),
        precision="fp16",
        verbose=True,
    )

    assert plan == b"fake-sana-wm-stage1-plan"
    builder = _FakeBuilder.last
    assert builder is not None
    assert builder.logger.level == _FakeLogger.VERBOSE
    assert stage1_dit_builder._BF16_WEIGHT_REFS == []
    assert builder.config.pool_limits == [("workspace", 64 << 30)]
    assert builder.network is not None
    assert builder.network.flags == 1
    assert builder.network.inputs[0] == ("x", "float16", (2, 128, 41, 22, 40))
    assert len(builder.network.convolutions) == 10
    assert builder.network.convolutions[0].kernel_shape == (1, 1, 1)
    assert len(builder.network.matrix_multiply) == 764
    assert len(builder.network.slices) == 1515
    assert builder.network.inputs[6] == ("raymats_inv", "float32", (2, 41 * 22 * 40, 4, 4))
    assert len(builder.network.softmax) == 1
    assert [output.name for output in builder.network.outputs] == ["output0"]
    assert any(
        layer.reshape_dims == (2, 41 * 22 * 40, 3, 20, 112) for layer in builder.network.shuffles
    )


def test_build_sana_wm_stage1_dit_engine_lowers_hybrid_softmax_block(monkeypatch) -> None:
    _FakeBuilder.last = None
    monkeypatch.setattr(stage1_dit_builder.trt_compat, "get_trt", lambda: _FakeTrt)
    raw_config = _raw_sana_wm_config()
    raw_config["model"] = dict(raw_config["model"])
    raw_config["model"]["softmax_every_n"] = 1

    plan = stage1_dit_builder.build_sana_wm_stage1_dit_engine(
        _stage1_weights(),
        raw_config,
        precision="fp16",
        verbose=False,
    )

    builder = _FakeBuilder.last
    assert plan == b"fake-sana-wm-stage1-plan"
    assert builder is not None
    assert builder.network is not None
    assert len(builder.network.softmax) == 1
    assert len(builder.network.attentions) == 2
    assert [layer.decomposable for layer in builder.network.attentions] == [True, True]
    assert [layer.axes for layer in builder.network.softmax] == [1 << 3]
    assert len(builder.network.matrix_multiply) == 24
    assert len(builder.network.outputs) == 1
