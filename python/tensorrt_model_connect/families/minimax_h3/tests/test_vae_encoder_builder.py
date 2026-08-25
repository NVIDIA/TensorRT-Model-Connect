# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tensorrt_model_connect.families.minimax_h3 import vae_encoder_builder
from tensorrt_model_connect.families.minimax_h3.vae_encoder_builder import (
    DIFFUSERS_ENCODER_REVISION,
    VAE_ENCODER_TILE_FRAMES,
    VAE_ENCODER_DEFAULT_WORKSPACE_BYTES,
    VAE_MOMENT_CHANNELS,
    VideoVaeEncoderConfig,
    VideoVaeEncoderShape,
    VideoVaeEncoderTileShape,
    _build_serialized_engine,
    _make_encoder_module,
    _make_encoder_tile_module,
    build_vae_encoder_engine,
    build_vae_encoder_tile_engines,
    checkpoint_keys,
    latent_frames_for,
    make_spatial_tile_plan,
    make_temporal_chunk_plan,
    raw_latent_frames_for,
    spatial_tile_metadata,
    split_spatial_tiles,
    temporal_chunk_metadata,
    validate_spatial_tile_metadata,
    validate_spatial_tile_plan,
    validate_temporal_chunk_metadata,
    validate_temporal_chunk_plan,
    validate_video_vae_encoder_config,
)


def _official_config() -> dict:
    return {
        "in_channels": 3,
        "out_channels": 3,
        "latent_channels": 24,
        "block_out_channels": [128, 256, 256, 512, 512, 1024],
        "layers_per_block": 2,
        "spatial_downsample_factors": [2, 2, 2, 2, 1, 1],
        "temporal_downsample_factors": [1, 2, 2, 1, 1, 1],
        "norm_num_groups": 32,
        "norm_eps": 1.0e-6,
        "spatial_padding_mode": "reflect",
        "decoder_num_layers": 36,
        "clip_length": 17,
        "token_drop": 3,
        "latents_mean": [float(index) / 24 for index in range(24)],
        "latents_std": [1.0 + float(index) / 24 for index in range(24)],
    }


def _tiny_config(*, token_drop: int = 0) -> VideoVaeEncoderConfig:
    return VideoVaeEncoderConfig(
        in_channels=1,
        latent_channels=1,
        block_out_channels=(2,),
        layers_per_block=1,
        spatial_downsample_factors=(1,),
        temporal_downsample_factors=(1,),
        norm_num_groups=1,
        norm_eps=1.0e-6,
        spatial_padding_mode="reflect",
        clip_length=3,
        token_drop=token_drop,
        latents_mean=(0.0,),
        latents_std=(1.0,),
    )


def test_visual_encoder_config_is_pinned_and_fails_closed() -> None:
    config = validate_video_vae_encoder_config(_official_config())
    assert config.spatial_compression_ratio == 16
    assert config.temporal_compression_ratio == 4
    assert config.moment_channels == VAE_MOMENT_CHANNELS == 48
    assert DIFFUSERS_ENCODER_REVISION == "06e0f2a81caaa6eaf4381e25cef07b0819582160"

    malformed = _official_config()
    malformed["temporal_downsample_factors"] = [1, 2, 1, 1, 1, 1]
    with pytest.raises(ValueError, match="architecture"):
        validate_video_vae_encoder_config(malformed)

    malformed = _official_config()
    malformed["latents_std"][7] = 0.0
    with pytest.raises(ValueError, match="finite positive"):
        validate_video_vae_encoder_config(malformed)


def test_static_shapes_cover_keyframes_and_released_temporal_chunking() -> None:
    assert latent_frames_for(1) == 1
    assert latent_frames_for(5) == 2
    assert latent_frames_for(17) == 2
    assert latent_frames_for(22) == 7
    assert latent_frames_for(39) == 12

    keyframe = VideoVaeEncoderShape(1, 1, 768, 1344)
    keyframe.validate()
    assert keyframe.input_shape == (1, 3, 1, 768, 1344)
    assert keyframe.output_shape == (1, 48, 1, 48, 84)

    reference = VideoVaeEncoderShape(1, 22, 768, 1344)
    reference.validate()
    assert reference.output_shape == (1, 48, 7, 48, 84)

    with pytest.raises(ValueError, match="multiple of 16"):
        VideoVaeEncoderShape(1, 1, 767, 1344).validate()
    with pytest.raises(ValueError, match="positive integers"):
        VideoVaeEncoderShape(True, 1, 768, 1344).validate()


def test_ref2va_tile_shapes_are_two_honest_raw_static_contracts() -> None:
    assert VAE_ENCODER_TILE_FRAMES == (1, 17)
    image = VideoVaeEncoderTileShape(1)
    video = VideoVaeEncoderTileShape(17)
    image.validate()
    video.validate()
    assert image.input_shape == (1, 3, 1, 256, 256)
    assert image.output_shape == (1, 48, 1, 16, 16)
    assert video.input_shape == (1, 3, 17, 256, 256)
    assert video.output_shape == (1, 48, 5, 16, 16)
    assert raw_latent_frames_for(17) == 5

    with pytest.raises(ValueError, match="exactly one of"):
        VideoVaeEncoderTileShape(5).validate()
    with pytest.raises(ValueError, match="exactly one of"):
        VideoVaeEncoderTileShape(True).validate()


def test_released_spatial_tile_layout_is_exact() -> None:
    assert split_spatial_tiles(256) == ([0], [256], [])
    assert split_spatial_tiles(768) == (
        [0, 160, 336, 512],
        [256, 256, 256, 256],
        [96, 80, 80],
    )
    assert split_spatial_tiles(1344) == (
        [0, 176, 352, 528, 704, 896, 1088],
        [256, 256, 256, 256, 256, 256, 256],
        [80, 80, 80, 80, 64, 64],
    )


def test_ref2va_spatial_plan_exposes_exact_latent_stitch_contract() -> None:
    plan = make_spatial_tile_plan(768, 1344)
    assert [tile.input_start for tile in plan.rows] == [0, 160, 336, 512]
    assert [tile.latent_blend_before for tile in plan.rows] == [0, 6, 5, 5]
    assert [tile.latent_crop_after for tile in plan.rows] == [6, 5, 5, 0]
    assert [tile.stitch_start for tile in plan.rows] == [0, 10, 21, 32]
    assert [tile.stitch_length for tile in plan.rows] == [10, 11, 11, 16]
    assert [tile.input_start for tile in plan.columns] == [
        0,
        176,
        352,
        528,
        704,
        896,
        1088,
    ]
    assert [tile.latent_blend_before for tile in plan.columns] == [0, 5, 5, 5, 5, 4, 4]
    assert [tile.latent_crop_after for tile in plan.columns] == [5, 5, 5, 5, 4, 4, 0]
    assert sum(tile.stitch_length for tile in plan.rows) == plan.latent_height == 48
    assert sum(tile.stitch_length for tile in plan.columns) == plan.latent_width == 84
    assert validate_spatial_tile_plan(plan) is plan

    metadata = spatial_tile_metadata(768, 1344)
    assert metadata["stitch"] == {
        "blend_order": ["height", "width"],
        "predecessor": "unstitched_neighbor_tile",
        "index_range": "[0,overlap)",
        "previous_weight": "1-index/overlap",
        "current_weight": "index/overlap",
    }
    assert validate_spatial_tile_metadata(metadata, height=768, width=1344) == metadata

    drifted = replace(
        plan,
        rows=(replace(plan.rows[0], stitch_length=9), *plan.rows[1:]),
    )
    with pytest.raises(ValueError, match="does not match Diffusers"):
        validate_spatial_tile_plan(drifted)
    with pytest.raises(ValueError, match="does not match Diffusers"):
        validate_spatial_tile_metadata({**metadata, "minimum_overlap": 32}, height=768, width=1344)


def test_ref2va_spatial_plan_rejects_approximate_padding_or_fake_alignment() -> None:
    with pytest.raises(ValueError, match="at least 256"):
        make_spatial_tile_plan(224, 1344)
    with pytest.raises(ValueError, match="32-aligned"):
        make_spatial_tile_plan(752, 1344)
    with pytest.raises(ValueError, match="at least 256"):
        make_spatial_tile_plan(True, 1344)


def test_ref2va_temporal_plan_pads_tail_then_drops_once_after_concatenation() -> None:
    still = make_temporal_chunk_plan(1)
    assert still.token_drop == 0
    assert still.output_moment_frames == 1
    assert still.chunks[0].engine_num_frames == 1

    short_video = make_temporal_chunk_plan(5)
    assert short_video.raw_moment_frames == 5
    assert short_video.token_drop == 3
    assert short_video.output_moment_frames == latent_frames_for(5) == 2
    assert short_video.chunks[0].to_metadata() == {
        "input_start": 0,
        "valid_input_frames": 5,
        "repeated_tail_frames": 12,
        "repeat_source_frame": 4,
        "engine_num_frames": 17,
        "raw_moment_start": 0,
        "raw_moment_frames": 5,
    }

    video = make_temporal_chunk_plan(22)
    assert video.raw_moment_frames == 10
    assert video.output_moment_frames == latent_frames_for(22) == 7
    assert [chunk.input_start for chunk in video.chunks] == [0, 17]
    assert [chunk.valid_input_frames for chunk in video.chunks] == [17, 5]
    assert [chunk.repeated_tail_frames for chunk in video.chunks] == [0, 12]
    assert [chunk.repeat_source_frame for chunk in video.chunks] == [None, 21]
    assert [chunk.raw_moment_start for chunk in video.chunks] == [0, 5]
    assert validate_temporal_chunk_plan(video) is video

    metadata = temporal_chunk_metadata(22)
    assert metadata["token_drop_scope"] == "once_from_concatenated_tail"
    assert validate_temporal_chunk_metadata(metadata, num_frames=22) == metadata

    drifted = replace(video, token_drop=2)
    with pytest.raises(ValueError, match="does not match Diffusers"):
        validate_temporal_chunk_plan(drifted)
    with pytest.raises(ValueError, match="does not match Diffusers"):
        validate_temporal_chunk_metadata({**metadata, "token_drop": 2}, num_frames=22)


def test_checkpoint_partition_contains_only_encoder_and_quant_conv() -> None:
    names = checkpoint_keys()
    assert len(names) == len(set(names)) == 118
    assert "encoder.conv_in.weight" in names
    assert "encoder.down_blocks.5.resnets.0.conv_shortcut.weight" in names
    assert "encoder.down_blocks.3.downsamplers.0.conv.weight" in names
    assert "quant_conv.weight" in names
    assert not any(name.startswith("decoder.") for name in names)
    assert not any(name.startswith("post_quant_conv.") for name in names)


def test_local_encoder_is_temporally_causal_and_keeps_fp32_boundary() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    module = _make_encoder_module(torch, _tiny_config())
    prefix = torch.randn((1, 1, 2, 8, 8), dtype=torch.float32)
    first = torch.cat([prefix, torch.zeros((1, 1, 1, 8, 8))], dim=2)
    second = torch.cat([prefix, torch.full((1, 1, 1, 8, 8), 100.0)], dim=2)

    with torch.inference_mode():
        first_moments = module(first.to(torch.bfloat16))
        second_moments = module(second.to(torch.bfloat16))

    assert first_moments.shape == (1, 2, 3, 8, 8)
    assert first_moments.dtype == torch.float32
    assert torch.equal(first_moments[:, :, :2], second_moments[:, :, :2])
    assert not torch.equal(first_moments[:, :, 2:], second_moments[:, :, 2:])


def test_tiny_raw_tile_graph_preserves_keys_and_defers_token_drop() -> None:
    torch = pytest.importorskip("torch")
    config = _tiny_config(token_drop=1)
    module = _make_encoder_tile_module(torch, config)
    normalized_rgb = torch.zeros((1, 1, 3, 8, 8), dtype=torch.bfloat16)

    assert tuple(module.state_dict()) == checkpoint_keys(config)
    with torch.inference_mode():
        raw_moments = module(normalized_rgb)
        dropped_moments = _make_encoder_module(torch, config)(normalized_rgb)

    assert raw_moments.shape == (1, 2, 3, 8, 8)
    assert dropped_moments.shape == (1, 2, 2, 8, 8)
    assert raw_moments.dtype == dropped_moments.dtype == torch.float32


def test_static_full_encoder_export_precomputes_python_tile_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    onnx = pytest.importorskip("onnx")
    reference = pytest.importorskip("onnx.reference")
    config = replace(_tiny_config(), in_channels=3)
    shape = VideoVaeEncoderShape(1, 1, 32, 32)
    observed = []
    original_split = vae_encoder_builder.split_spatial_tiles

    def record_split(length, **kwargs):
        observed.append((length, type(length)))
        return original_split(length, **kwargs)

    monkeypatch.setattr(vae_encoder_builder, "split_spatial_tiles", record_split)
    module = _make_encoder_module(torch, config, build_shape=shape)
    assert observed == [(32, int), (32, int)]

    def reject_trace_geometry(*_args, **_kwargs):
        raise AssertionError("static export must not derive tile geometry from traced tensors")

    monkeypatch.setattr(vae_encoder_builder, "split_spatial_tiles", reject_trace_geometry)
    torch.manual_seed(11)
    normalized_rgb = torch.randn(shape.input_shape, dtype=torch.float32)
    buffer = io.BytesIO()
    torch.onnx.export(
        module,
        normalized_rgb,
        buffer,
        opset_version=17,
        input_names=["normalized_rgb"],
        output_names=["posterior_moments"],
        dynamo=False,
    )
    model = onnx.load_model_from_string(buffer.getvalue())
    onnx.checker.check_model(model)
    assert [dim.dim_value for dim in model.graph.input[0].type.tensor_type.shape.dim] == [
        1,
        3,
        1,
        32,
        32,
    ]

    with torch.inference_mode():
        expected = module(normalized_rgb).numpy()
    (actual,) = reference.ReferenceEvaluator(model).run(
        None, {"normalized_rgb": normalized_rgb.numpy()}
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)


def test_builder_exports_then_builds_one_explicit_static_shape(tmp_path: Path, monkeypatch) -> None:
    vae_dir = tmp_path / "vae"
    vae_dir.mkdir()
    (vae_dir / "config.json").write_text(json.dumps(_official_config()))
    observed = {}

    def export(root, config, shape, verbose):
        observed.update(root=root, config=config, shape=shape, export_verbose=verbose)
        return b"onnx"

    def build(onnx_bytes, *, shape, verbose, workspace_bytes):
        observed.update(
            onnx_bytes=onnx_bytes,
            build_shape=shape,
            build_verbose=verbose,
            workspace_bytes=workspace_bytes,
        )
        return b"visual-encoder-plan"

    monkeypatch.setattr(vae_encoder_builder, "_export_encoder_onnx", export)
    monkeypatch.setattr(vae_encoder_builder, "_build_serialized_engine", build)
    assert (
        build_vae_encoder_engine(
            vae_dir,
            batch_size=1,
            num_frames=22,
            height=768,
            width=1344,
            verbose=True,
            workspace_bytes=12 << 30,
        )
        == b"visual-encoder-plan"
    )
    shape = VideoVaeEncoderShape(1, 22, 768, 1344)
    assert observed == {
        "root": vae_dir,
        "config": validate_video_vae_encoder_config(_official_config()),
        "shape": shape,
        "export_verbose": True,
        "onnx_bytes": b"onnx",
        "build_shape": shape,
        "build_verbose": True,
        "workspace_bytes": 12 << 30,
    }


def test_ref2va_builder_exports_only_the_two_reusable_raw_tile_shapes(
    tmp_path: Path, monkeypatch
) -> None:
    vae_dir = tmp_path / "vae"
    vae_dir.mkdir()
    (vae_dir / "config.json").write_text(json.dumps(_official_config()))
    observed = []

    def export(root, config, shape, verbose):
        observed.append(("export", root, config, shape, verbose))
        return f"onnx-t{shape.num_frames}".encode()

    def build(onnx_bytes, *, shape, verbose, workspace_bytes):
        observed.append(("build", onnx_bytes, shape, verbose, workspace_bytes))
        return f"plan-t{shape.num_frames}".encode()

    monkeypatch.setattr(vae_encoder_builder, "_export_encoder_tile_onnx", export)
    monkeypatch.setattr(vae_encoder_builder, "_build_serialized_engine", build)
    assert build_vae_encoder_tile_engines(vae_dir, verbose=True, workspace_bytes=10 << 30) == {
        1: b"plan-t1",
        17: b"plan-t17",
    }

    config = validate_video_vae_encoder_config(_official_config())
    assert observed == [
        ("export", vae_dir, config, VideoVaeEncoderTileShape(1), True),
        ("build", b"onnx-t1", VideoVaeEncoderTileShape(1), True, 10 << 30),
        ("export", vae_dir, config, VideoVaeEncoderTileShape(17), True),
        ("build", b"onnx-t17", VideoVaeEncoderTileShape(17), True, 10 << 30),
    ]


@pytest.mark.parametrize(
    "shape",
    [
        VideoVaeEncoderShape(1, 22, 768, 1344),
        VideoVaeEncoderTileShape(1),
        VideoVaeEncoderTileShape(17),
    ],
    ids=("full-shape", "image-tile", "video-tile"),
)
def test_onnx_contract_and_workspace_are_fail_closed(
    monkeypatch, shape: VideoVaeEncoderShape | VideoVaeEncoderTileShape
) -> None:
    shape.validate()
    observed = {}
    fp32 = object()

    class Tensor:
        def __init__(self, name, tensor_shape):
            self.name = name
            self.shape = tensor_shape
            self.dtype = fp32

    class Network:
        num_inputs = 1
        num_outputs = 1

        def get_input(self, index):
            assert index == 0
            return Tensor("normalized_rgb", shape.input_shape)

        def get_output(self, index):
            assert index == 0
            return Tensor("posterior_moments", shape.output_shape)

    class Parser:
        num_errors = 0

        def __init__(self, network, logger):
            del network, logger

        def parse(self, payload):
            observed["onnx"] = payload
            return True

    class BuildConfig:
        def set_memory_pool_limit(self, pool, size):
            observed.update(pool=pool, workspace=size)

        def get_memory_pool_limit(self, pool):
            assert pool == "workspace"
            return observed["workspace"]

    class Builder:
        def __init__(self, logger):
            del logger

        def create_network(self, flags):
            observed["flags"] = flags
            return Network()

        def create_builder_config(self):
            return BuildConfig()

        def build_serialized_network(self, network, config):
            del network, config
            return b"visual-plan"

    class Logger:
        INFO = "info"
        WARNING = "warning"

        def __init__(self, level):
            observed["log_level"] = level

    class FakeTrt:
        class MemoryPoolType:
            WORKSPACE = "workspace"

    FakeTrt.Logger = Logger
    FakeTrt.Builder = Builder
    FakeTrt.OnnxParser = Parser
    FakeTrt.float32 = fp32

    monkeypatch.setattr(vae_encoder_builder.trt_compat, "get_trt", lambda: FakeTrt)
    monkeypatch.setattr(
        vae_encoder_builder.trt_compat,
        "network_creation_flags",
        lambda **_kwargs: 13,
    )

    assert (
        _build_serialized_engine(b"onnx", shape=shape, verbose=False, workspace_bytes=None)
        == b"visual-plan"
    )
    assert observed == {
        "onnx": b"onnx",
        "flags": 13,
        "log_level": "warning",
        "pool": "workspace",
        "workspace": VAE_ENCODER_DEFAULT_WORKSPACE_BYTES,
    }

    original_get_output = Network.get_output
    Network.get_output = lambda self, index: Tensor("posterior_mode", shape.output_shape)
    try:
        with pytest.raises(RuntimeError, match="contract mismatch"):
            _build_serialized_engine(b"onnx", shape=shape, verbose=False, workspace_bytes=None)
    finally:
        Network.get_output = original_get_output
