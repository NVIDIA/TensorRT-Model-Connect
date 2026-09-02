# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tensorrt_model_connect.families.minimax_h3.fl2va_contract import (
    LATENTS_MEAN,
    LATENTS_STD,
    MultimodalTextProfile,
    VisionEncoderProfile,
    align_num_frames,
    audio_latent_frames,
    fl2va_mrope_position_ids,
    fl2va_text_rows,
    keyframe_resize_geometry,
    keyframe_tile_count,
    keyframe_vae_encoder_abi,
    last_keyframe_rotary_time,
    latent_tile_axis,
    packed_rows,
    qwen_vision_patch_rows,
    qwen_vision_token_rows,
    resolve_canvas_size,
    resolve_keyframe_anchors,
    split_tile_axis,
    text_encoder_abi,
    video_latent_frames,
    vision_encoder_abi,
)


@pytest.mark.parametrize(
    ("aspect", "canvas"),
    [
        ((21, 9), (672, 1536)),
        ((16, 9), (768, 1344)),
        ((4, 3), (768, 1024)),
        ((1, 1), (768, 768)),
        ((3, 4), (1024, 768)),
        ((9, 16), (1344, 768)),
        ((4, 1), (512, 2016)),
        ((1, 4), (2016, 512)),
    ],
)
def test_public_canvas_resolution_is_exact(aspect, canvas) -> None:
    assert resolve_canvas_size(*aspect) == canvas


@pytest.mark.parametrize("aspect", [(0, 1), (1, 0), (4.01, 1), (1, 4.01)])
def test_public_canvas_resolution_fails_closed(aspect) -> None:
    with pytest.raises(ValueError):
        resolve_canvas_size(*aspect)


def test_first_last_and_both_anchor_order_and_resize_geometry() -> None:
    assert resolve_keyframe_anchors(has_first=True, has_last=False) == ("first",)
    assert resolve_keyframe_anchors(has_first=False, has_last=True) == ("last",)
    assert resolve_keyframe_anchors(has_first=True, has_last=True) == ("first", "last")
    with pytest.raises(ValueError, match="needs a first frame"):
        resolve_keyframe_anchors(has_first=False, has_last=False)

    # A last-only request still packs its last image at index zero, so it is
    # the direct-stretch geometry anchor rather than a cover-crop follower.
    assert keyframe_resize_geometry(1000, 1000, 768, 1344, packed_index=0).mode == "stretch"
    follower = keyframe_resize_geometry(1000, 1000, 768, 1344, packed_index=1)
    assert follower.mode == "cover_crop"
    assert (follower.resized_height, follower.resized_width) == (1344, 1344)
    assert (follower.crop_top, follower.crop_left) == (288, 0)


def test_keyframe_vae_tile_geometry_matches_reference() -> None:
    assert split_tile_axis(768).starts == (0, 160, 336, 512)
    assert split_tile_axis(768).overlaps == (96, 80, 80)
    assert split_tile_axis(1344).starts == (0, 176, 352, 528, 704, 896, 1088)
    assert split_tile_axis(1344).overlaps == (80, 80, 80, 80, 64, 64)
    assert latent_tile_axis(768).starts == (0, 10, 21, 32)
    assert latent_tile_axis(768).overlaps == (6, 5, 5)
    assert latent_tile_axis(1344).starts == (0, 11, 22, 33, 44, 56, 68)
    assert latent_tile_axis(1344).overlaps == (5, 5, 5, 5, 4, 4)
    assert keyframe_tile_count(768, 1344) == 28
    with pytest.raises(ValueError, match="at least 256"):
        keyframe_tile_count(128, 512)
    assert len(LATENTS_MEAN) == len(LATENTS_STD) == 24


def test_qwen_grid_and_fl2va_presentation_rows() -> None:
    assert qwen_vision_patch_rows(768, 1344) == 4032
    assert qwen_vision_token_rows(768, 1344) == 1008
    assert fl2va_text_rows(128, 1, height=768, width=1344) == 1144
    assert fl2va_text_rows(128, 2, height=768, width=1344) == 2160


def test_qwen_fl2va_mrope_runs_are_exact() -> None:
    temporal, height, width = fl2va_mrope_position_ids(2, 1, height=768, width=1344)
    assert len(temporal) == len(height) == len(width) == 1018
    assert temporal[:7] == tuple(range(7))
    assert set(temporal[7 : 7 + 1008]) == {7}
    assert height[7 : 7 + 42] == (7,) * 42
    assert width[7 : 7 + 42] == tuple(range(7, 49))
    assert temporal[-3:] == height[-3:] == width[-3:] == (49, 50, 51)

    temporal, _, _ = fl2va_mrope_position_ids(2, 2, height=768, width=1344)
    assert len(temporal) == 2034
    assert temporal[1015:1023] == tuple(range(49, 57))
    assert set(temporal[1023 : 1023 + 1008]) == {57}
    assert temporal[-3:] == (99, 100, 101)


def test_dynamic_duration_and_packed_row_accounting() -> None:
    assert align_num_frames(124) == 124
    assert align_num_frames(125) == 141
    assert align_num_frames(345) == 345
    with pytest.raises(ValueError, match="124..345"):
        align_num_frames(107)
    with pytest.raises(ValueError, match="124..345"):
        align_num_frames(346)
    assert video_latent_frames(124) == 37
    assert video_latent_frames(345) == 102
    assert audio_latent_frames(124) == 207
    # The reference uses round(), not ceil(), at the 40-Hz audio grid.
    assert audio_latent_frames(209) == 348

    first = packed_rows(
        prompt_tokens=128,
        keyframes=1,
        height=768,
        width=1344,
        num_frames=124,
    )
    assert (first.text, first.condition_video, first.target_audio, first.target_video) == (
        1144,
        1008,
        414,
        37296,
    )
    both = packed_rows(
        prompt_tokens=128,
        keyframes=2,
        height=768,
        width=1344,
        num_frames=345,
    )
    assert (both.text, both.condition_video, both.target_audio, both.target_video) == (
        2160,
        2016,
        1150,
        102816,
    )
    assert last_keyframe_rotary_time(2160, 345) == pytest.approx(2733.3333333333335)


def test_plan_abis_are_dynamic_compact_and_shared() -> None:
    vae = keyframe_vae_encoder_abi()
    assert vae.filename == "fl2va_keyframe_vae_encoder.plan"
    assert vae.inputs[0].name == "pixel_tiles"
    assert vae.inputs[0].min_shape[0] == 1
    assert vae.inputs[0].opt_shape[0] == 28
    assert vae.inputs[0].max_shape[0] == 33
    assert vae.outputs[0].max_shape == (33, 48, 1, 16, 16)

    vision = vision_encoder_abi(VisionEncoderProfile(4, 8, 16))
    assert vision.filename == "vision_encoder.plan"
    assert tuple(binding.name for binding in vision.inputs) == (
        "pixel_values",
        "interp_indices",
        "interp_weights",
        "vision_position_ids",
    )
    assert vision.outputs[0].max_shape == (4, 5120)

    text = text_encoder_abi()
    assert text.filename == "text_encoder.plan"
    assert tuple(binding.name for binding in text.inputs) == (
        "input_ids",
        "mrope_position_ids",
        "vision_mask",
        "vision_count",
        "vision_row_indices",
        "vision_embeds",
        "deepstack_0",
        "deepstack_1",
        "deepstack_2",
    )
    assert text.inputs[3].min_shape == text.inputs[3].max_shape == (1,)
    assert text.inputs[4].min_shape == (1,)
    assert text.inputs[4].max_shape == (2088,)
    assert text.inputs[5].max_shape == (2088, 5120)
    assert text.outputs[0].max_shape == (2641, 5120)


def test_multimodal_text_profile_rejects_incoherent_visual_capacity() -> None:
    with pytest.raises(ValueError, match="maximum visual rows"):
        MultimodalTextProfile(
            opt_sequence_length=100,
            max_sequence_length=100,
            opt_vision_rows=100,
            max_vision_rows=101,
        ).validate()


def test_new_builders_do_not_import_runtime_frameworks_or_process_launchers() -> None:
    family_dir = Path(__file__).resolve().parents[1]
    forbidden = {"torch", "triton", "fastvideo", "subprocess", "ffmpeg"}
    for filename in (
        "fl2va_vae_encoder_builder.py",
        "multimodal_vision_builder.py",
        "multimodal_text_encoder_builder.py",
    ):
        tree = ast.parse((family_dir / filename).read_text(encoding="utf-8"))
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0].lower() for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0].lower())
        assert imports.isdisjoint(forbidden), (filename, imports & forbidden)


def test_checkpoint_partitions_are_exhaustive_and_fail_closed_when_trt_is_available() -> None:
    from tensorrt_model_connect import trt_compat

    if trt_compat.is_available("tensorrt"):
        pass
    elif trt_compat.is_available("tensorrt_rtx"):
        trt_compat.configure_backend(rtx=True)
    else:
        pytest.skip("TensorRT or TensorRT-RTX bindings are unavailable")
    from tensorrt_model_connect.families.minimax_h3.fl2va_vae_encoder_builder import (
        build_keyframe_vae_encoder_engine,
        checkpoint_keys as vae_keys,
    )
    from tensorrt_model_connect.families.minimax_h3.multimodal_text_encoder_builder import (
        build_multimodal_text_encoder_engine,
        checkpoint_keys as text_keys,
    )
    from tensorrt_model_connect.families.minimax_h3.multimodal_vision_builder import (
        build_multimodal_vision_encoder_engine,
        checkpoint_keys as vision_keys,
    )

    assert len(vae_keys()) == len(set(vae_keys())) == 118
    assert len(vision_keys()) == len(set(vision_keys())) == 351
    assert len(text_keys()) == len(set(text_keys())) == 551
    for builder in (
        build_keyframe_vae_encoder_engine,
        build_multimodal_vision_encoder_engine,
        build_multimodal_text_encoder_engine,
    ):
        with pytest.raises(ValueError, match="checkpoint partition mismatch"):
            builder({})
