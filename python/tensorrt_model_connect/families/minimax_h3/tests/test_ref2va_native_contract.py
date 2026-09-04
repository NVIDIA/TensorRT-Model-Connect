# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import numpy as np
import pytest

from tensorrt_model_connect.families.minimax_h3.ref2va_checkpoint import (
    CHECKPOINT_REVISION,
    CONFIG_SHA256,
    INDEX_SHA256,
    REF2VA_ADALN_KEYS,
    REF2VA_ALL_KEYS,
    REF2VA_DENOISER_KEYS,
    SHARDS,
    TOTAL_TENSOR_BYTES,
    download_command,
    validate_transformer_ref_checkpoint,
)
from tensorrt_model_connect.families.minimax_h3.ref2va_contract import (
    H3_AUDIO_TAG,
    H3_TEXT_TAG,
    H3_VIDEO_TAG,
    QWEN_IMAGE_PAD_TOKEN_ID,
    QWEN_VISION_END_TOKEN_ID,
    QWEN_VISION_START_TOKEN_ID,
    REF2VA_MAX_ALL_AUDIO_ROWS,
    REF2VA_MAX_ALL_VIDEO_ROWS,
    REF2VA_MAX_PACKED_ROWS,
    REF2VA_MAX_TEXT_ROWS,
    EncodedReferenceGeometry,
    ReferenceSpec,
    Ref2VADenoiserProfile,
    audio_latent_frames,
    build_ref2va_packed_layout,
    build_row_timestep_indices,
    materialize_ref2va_presentation,
    pad_ref2va_timesteps,
    qwen_mrope_position_ids,
    qwen_merged_rows,
    qwen_video_condition_sample,
    ref2va_denoiser_abi,
    ref2va_presentation_blueprint,
    reference_rng_draw_order,
    reference_video_encode_schedule,
    reference_video_latent_frames,
    resolve_reference_image_size,
    resolve_reference_video_size,
    snap_reference_video_frames_down,
    validate_reference_request,
    video_resample_source_indices,
)


def _image() -> ReferenceSpec:
    return ReferenceSpec("image", width=1024, height=1024)


def _video(duration: float = 5.0, *, audio: bool = False) -> ReferenceSpec:
    return ReferenceSpec(
        "video",
        duration_seconds=duration,
        width=1920,
        height=1080,
        has_audio=audio,
    )


def _audio(duration: float = 5.0) -> ReferenceSpec:
    return ReferenceSpec("audio", duration_seconds=duration)


def test_public_reference_limits_and_soundtrack_identity() -> None:
    references = (_video(5.0, audio=True), _image(), _audio(5.0))
    summary = validate_reference_request(references)
    assert (summary.image_count, summary.video_count, summary.audio_count) == (1, 1, 1)
    assert summary.audio_bearing_count == 2
    assert summary.total_video_seconds == summary.total_audio_seconds == 5.0

    # A soundtrack remains part of its video file; it is not a fourth kind or
    # an explicit audio reference and does not perturb request order.
    assert len(references) == 3
    audio_only = validate_reference_request((_audio(),))
    assert (audio_only.image_count, audio_only.video_count, audio_only.audio_count) == (0, 0, 1)
    assert audio_only.total_video_seconds == 0.0
    assert audio_only.total_audio_seconds == 5.0
    assert audio_only.audio_bearing_count == 1
    with pytest.raises(ValueError, match="at most 9"):
        validate_reference_request(tuple(_image() for _ in range(10)))
    with pytest.raises(ValueError, match="video duration total"):
        validate_reference_request((_image(), _video(8.0), _video(8.0)))
    with pytest.raises(ValueError, match="audio duration total"):
        validate_reference_request((_image(), _audio(8.0), _audio(8.0)))
    with pytest.raises(ValueError, match="2..15"):
        validate_reference_request((_image(), _audio(1.99)))


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ((1000, 1000), (2048, 2048)),
        ((4000, 1000), (2048, 8192)),
        ((1000, 4000), (8192, 2048)),
    ],
)
def test_reference_image_short_edge_2048_has_no_area_cap(source, expected) -> None:
    assert resolve_reference_image_size(*source) == expected


def test_reference_video_uses_its_own_768p_canvas() -> None:
    assert resolve_reference_video_size(1920, 1080) == (768, 1344)
    assert resolve_reference_video_size(4000, 1000) == (512, 2016)


def test_video_rate_normalization_and_vae_snap_are_exact() -> None:
    assert video_resample_source_indices(5, 12.0, target_frames=20) == (
        0,
        0,
        1,
        1,
        2,
        2,
        3,
        3,
        4,
        4,
    )
    assert video_resample_source_indices(10, 30.0, target_frames=20) == (
        0,
        1,
        3,
        4,
        5,
        6,
        8,
        9,
    )
    assert snap_reference_video_frames_down(48) == 39
    assert snap_reference_video_frames_down(124) == 124
    assert snap_reference_video_frames_down(345) == 345
    assert reference_video_latent_frames(48) == 12
    assert reference_video_latent_frames(124) == 37
    assert reference_video_latent_frames(345) == 102
    schedule = reference_video_encode_schedule(48)
    assert (
        schedule.snapped_frames,
        schedule.clip_count,
        schedule.repeated_tail_frames,
        schedule.raw_posterior_frames,
        schedule.dropped_tail_latents,
        schedule.output_latent_frames,
    ) == (39, 3, 12, 15, 3, 12)
    assert audio_latent_frames(64_000) == 80
    assert audio_latent_frames(64_001) == 81


def test_qwen_two_fps_sampling_pair_timestamps_use_half_even_rendering() -> None:
    indices, timestamps = qwen_video_condition_sample(124)
    assert indices == tuple(range(0, 121, 12))
    assert timestamps == (0.25, 1.25, 2.25, 3.25, 4.25, 5.0)
    assert f"<{timestamps[0]:.1f} seconds>" == "<0.2 seconds>"


def test_ordered_presentation_keeps_video_soundtrack_attached() -> None:
    references = (_video(5.0, audio=True), _image(), _audio(5.0))
    pieces = ref2va_presentation_blueprint(
        "prompt",
        references,
        normalized_visual_sizes=((768, 1344), (2048, 2048)),
        normalized_video_frames=(124,),
    )
    text = tuple(piece.text for piece in pieces if piece.modality == "text")
    assert text[:3] == ("<Audio 1>: ", "<Video 1>: ", "<0.2 seconds>")
    assert "<Picture 1>: " in text
    assert text[-2:] == ("<Audio 2>: ", "prompt")
    assert tuple(piece.modality for piece in pieces).count("video") == 6
    assert tuple(piece.modality for piece in pieces).count("image") == 1


def test_materialized_presentation_has_distinct_qwen_and_h3_modality_maps() -> None:
    pieces = ref2va_presentation_blueprint(
        "prompt",
        (_image(),),
        normalized_visual_sizes=((2048, 2048),),
        normalized_video_frames=(),
    )
    # One token per literal isolates the vision-boundary accounting.
    result = materialize_ref2va_presentation(pieces, lambda _text: (42,))
    assert qwen_merged_rows(2048, 2048) == 4096
    assert result.input_ids[1] == QWEN_VISION_START_TOKEN_ID
    assert set(result.input_ids[2 : 2 + 4096]) == {QWEN_IMAGE_PAD_TOKEN_ID}
    assert result.input_ids[4098] == QWEN_VISION_END_TOKEN_ID
    assert result.qwen_token_types[1] == result.qwen_token_types[4098] == 0
    assert set(result.qwen_token_types[2:4098]) == {1}
    assert result.h3_token_tags[0] == H3_TEXT_TAG
    assert set(result.h3_token_tags[1:4099]) == {H3_VIDEO_TAG}
    assert result.h3_token_tags[-1] == H3_TEXT_TAG
    assert result.vision_row_indices == tuple(range(2, 4098))
    assert result.mrope_position_ids.shape == (3, 4100)


def test_qwen_mrope_rejects_pad_grid_disagreement() -> None:
    with pytest.raises(ValueError, match="grid requires"):
        qwen_mrope_position_ids(
            (0, 1, 1, 0),
            image_grids=((1, 4, 4),),
            video_grids=(),
        )


def test_ref2va_interleaved_packed_layout_and_rotary_clock() -> None:
    references = (
        EncodedReferenceGeometry("image", 1, 32, 32),
        EncodedReferenceGeometry("video", 2, 32, 64, audio_latents=3),
    )
    layout = build_ref2va_packed_layout(
        (H3_TEXT_TAG, H3_VIDEO_TAG),
        references,
        target_latent_frames=2,
        target_latent_height=32,
        target_latent_width=32,
        target_audio_latents=4,
    )
    # text[0:2], image-video[2:258], video soundtrack[258:264],
    # reference-video[264:1288], target-audio[1288:1296], target-video[1296:].
    assert layout.sequence_length == 1808
    assert layout.num_condition_video_rows == 1280
    assert layout.num_condition_audio_rows == 6
    assert np.array_equal(layout.text_indices, np.arange(2))
    assert np.array_equal(layout.video_indices[:256], np.arange(2, 258))
    assert np.array_equal(layout.audio_indices[:6], np.arange(258, 264))
    assert layout.video_indices[256] == 264
    assert layout.audio_indices[6] == 1288
    assert layout.video_indices[-512] == 1296
    assert np.all(layout.token_tags[layout.audio_indices] == H3_AUDIO_TAG)
    assert np.all(layout.token_tags[layout.video_indices] == H3_VIDEO_TAG)

    assert np.all(layout.position_ids[2:258, 0] == 2.0)
    assert layout.position_ids[258, 0] == 3.0
    assert layout.position_ids[264, 0] == 3.0
    # image consumes one slot; two reference latent frames consume 25/3.
    assert layout.position_ids[1288, 0] == pytest.approx(3.0 + 25.0 / 3.0)

    unique, timestep_indices, adaln = build_row_timestep_indices(
        layout,
        video_timestep=0.8,
        audio_timestep=0.5,
    )
    assert unique == pytest.approx(np.asarray((0.5, 0.8, 0.999, 1.0), np.float32))
    assert timestep_indices.shape == adaln.shape == (layout.sequence_length,)
    assert np.all(adaln == timestep_indices * 3 + layout.token_tags)
    padded = pad_ref2va_timesteps(unique[:3])
    assert padded.count == 3
    assert padded.values.tolist() == pytest.approx([unique[0], unique[1], unique[2], unique[2]])
    assert reference_rng_draw_order(references) == (
        "condition_0_image",
        "condition_1_video",
        "target_video",
        "target_audio",
    )


def test_full_public_capacity_is_explicit_not_silently_narrowed() -> None:
    profile = Ref2VADenoiserProfile()
    profile.validate()
    assert profile.min_video_rows == 18_870
    assert profile.min_packed_rows == 19_285
    assert profile.min_text_rows == 1
    assert profile.max_video_rows == REF2VA_MAX_ALL_VIDEO_ROWS == 364_608
    assert profile.max_audio_rows == REF2VA_MAX_ALL_AUDIO_ROWS == 3_558
    assert profile.max_text_rows == REF2VA_MAX_TEXT_ROWS == 262_144
    assert profile.max_packed_rows == REF2VA_MAX_PACKED_ROWS == 630_310
    abi = ref2va_denoiser_abi(profile)
    assert abi.filename == "ref2va_denoiser.plan"
    names = tuple(binding.name for binding in abi.inputs)
    assert names[:9] == (
        "video_hidden_states",
        "audio_hidden_states",
        "encoder_hidden_states",
        "position_ids",
        "video_indices",
        "audio_indices",
        "text_indices",
        "adaln_indices",
        "timestep_indices",
    )
    assert names[9:59] == tuple(f"block_modulation_{index}" for index in range(50))
    assert names[-1] == "final_modulation"


def test_transformer_ref_partition_is_distinct_exhaustive_and_pinned() -> None:
    assert len(REF2VA_DENOISER_KEYS) == 532
    assert len(REF2VA_ADALN_KEYS) == 106
    assert len(REF2VA_ALL_KEYS) == len(set(REF2VA_ALL_KEYS)) == 638
    assert not set(REF2VA_DENOISER_KEYS) & set(REF2VA_ADALN_KEYS)
    assert len(SHARDS) == 14
    assert sum(size for _name, size, _sha in SHARDS) > TOTAL_TENSOR_BYTES
    assert len(CONFIG_SHA256) == len(INDEX_SHA256) == 64
    command = download_command("checkpoint")
    assert CHECKPOINT_REVISION in command
    assert '--include "transformer_ref/*"' in command


def test_missing_transformer_ref_never_falls_back_to_transformer(tmp_path: Path) -> None:
    (tmp_path / "transformer").mkdir()
    with pytest.raises(FileNotFoundError, match="not a valid fallback"):
        validate_transformer_ref_checkpoint(tmp_path)


def test_local_base_index_proves_ref_partition_has_same_schema_not_same_values() -> None:
    checkpoint_root = os.environ.get("TRTMC_MINIMAX_H3_CHECKPOINT_ROOT")
    if not checkpoint_root:
        pytest.skip("TRTMC_MINIMAX_H3_CHECKPOINT_ROOT is not configured")
    snapshot = Path(checkpoint_root).expanduser()
    index_path = snapshot / "transformer" / "diffusion_pytorch_model.safetensors.index.json"
    if not index_path.is_file():
        pytest.skip("local MiniMax-H3 checkpoint is unavailable")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert index["metadata"]["total_size"] == TOTAL_TENSOR_BYTES
    assert set(index["weight_map"]) == set(REF2VA_ALL_KEYS)
    # An identical base index/config does not establish identical tensor
    # values.  This remains valid if a developer later downloads the required
    # partition into the same snapshot.
    if (snapshot / "transformer_ref").exists():
        record = validate_transformer_ref_checkpoint(snapshot)
        assert record.tensor_count == 638
    else:
        with pytest.raises(FileNotFoundError, match="transformer_ref"):
            validate_transformer_ref_checkpoint(snapshot)


def test_ref2va_build_tooling_has_no_framework_or_process_runtime_imports() -> None:
    family = Path(__file__).resolve().parents[1]
    forbidden = {"torch", "triton", "fastvideo", "subprocess", "ffmpeg"}
    for filename in (
        "ref2va_contract.py",
        "ref2va_checkpoint.py",
        "ref2va_dit_builder.py",
        "ref2va_audio_encoder_builder.py",
        "ref2va_bundle_contract.py",
        "ref2va_video_encoder_builder.py",
    ):
        tree = ast.parse((family / filename).read_text(encoding="utf-8"))
        imports: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0].lower() for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0].lower())
        assert imports.isdisjoint(forbidden), (filename, imports & forbidden)


def test_ref2va_builder_fails_closed_before_any_large_build_when_trt_is_available() -> None:
    from tensorrt_model_connect import trt_compat

    if trt_compat.is_available("tensorrt"):
        pass
    elif trt_compat.is_available("tensorrt_rtx"):
        trt_compat.configure_backend(rtx=True)
    else:
        pytest.skip("TensorRT or TensorRT-RTX bindings are unavailable")
    from tensorrt_model_connect.families.minimax_h3.ref2va_dit_builder import (
        build_ref2va_adaln_precompute_engine,
        build_ref2va_dit_engine,
    )

    with pytest.raises(ValueError, match="denoiser checkpoint partition mismatch"):
        build_ref2va_dit_engine({})
    with pytest.raises(ValueError, match="AdaLN checkpoint partition mismatch"):
        build_ref2va_adaln_precompute_engine({})


def test_ref2va_dynamic_profile_serializes_and_round_trips() -> None:
    from tensorrt_model_connect import trt_compat

    if trt_compat.is_available("tensorrt"):
        pass
    elif trt_compat.is_available("tensorrt_rtx"):
        trt_compat.configure_backend(rtx=True)
    else:
        pytest.skip("TensorRT or TensorRT-RTX bindings are unavailable")

    from tensorrt_model_connect.families.minimax_h3.ref2va_dit_builder import (
        _add_optimization_profile,
        _set_profile_shape,
    )

    trt = trt_compat.get_trt()
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    profile = Ref2VADenoiserProfile()
    bindings = ref2va_denoiser_abi(profile).inputs[:9]
    dtypes = {"float32": trt.float32, "int32": trt.int32}
    for binding in bindings:
        tensor = network.add_input(
            binding.name,
            dtypes[binding.dtype],
            (-1, *binding.min_shape[1:]),
        )
        output = network.add_identity(tensor).get_output(0)
        output.name = f"{binding.name}_identity"
        network.mark_output(output)

    _add_optimization_profile(builder, config, profile)
    plan = builder.build_serialized_network(network, config)
    assert plan is not None and len(bytes(plan)) > 0

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(bytes(plan))
    assert engine is not None
    for binding in bindings:
        recorded = tuple(
            tuple(int(dimension) for dimension in shape)
            for shape in engine.get_tensor_profile_shape(binding.name, 0)
        )
        assert recorded == (binding.min_shape, binding.opt_shape, binding.max_shape)

    invalid = Ref2VADenoiserProfile(min_video_rows=profile.opt_video_rows + 1)
    with pytest.raises(RuntimeError, match="profile binding video_hidden_states"):
        _add_optimization_profile(builder, builder.create_builder_config(), invalid)

    class RejectingOptimization:
        def set_shape(self, *_args, **_kwargs):
            raise ValueError("invalid profile")

    with pytest.raises(RuntimeError, match="profile binding test") as rejected:
        _set_profile_shape(RejectingOptimization(), "test", ((1,), (2,), (3,)))
    assert isinstance(rejected.value.__cause__, ValueError)


def test_trt_scatter_gather_micrograph_serializes_when_trt_is_available() -> None:
    from tensorrt_model_connect import trt_compat

    if trt_compat.is_available("tensorrt"):
        pass
    elif trt_compat.is_available("tensorrt_rtx"):
        trt_compat.configure_backend(rtx=True)
    else:
        pytest.skip("TensorRT or TensorRT-RTX bindings are unavailable")

    from tensorrt_model_connect.families.minimax_h3 import graph_ops as op
    from tensorrt_model_connect.families.minimax_h3.ref2va_dit_builder import (
        _scatter_rows,
    )

    trt = trt_compat.get_trt()
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 64 << 20)
    base = network.add_input("base", trt.float32, (6, 4))
    compact = network.add_input("compact", trt.float32, (2, 4))
    indices = network.add_input("indices", trt.int32, (2,))
    scattered = _scatter_rows(network, base, indices, compact, label="micro")
    gathered = op.gather_rows(network, scattered, indices)
    gathered.name = "gathered"
    network.mark_output(gathered)
    plan = builder.build_serialized_network(network, config)
    assert plan is not None and len(bytes(plan)) > 0
    op.release_weight_buffers(network)
