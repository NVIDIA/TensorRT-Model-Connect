# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real native AV-guide loop against the upstream 49-forward mixed-BF16 model.

Declared synthetic reduced topology/encoded inputs, not trained Qwen/VAE/DiT
or image/video quality qualification. No engine, RNG or scheduler doubles.
"""

from dataclasses import asdict, replace
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from families.minimax_h3.adaln_builder import build_adaln_precompute_engine
from families.minimax_h3.config import Ref2VAPackedProfile, SOL_ENGINE_1344X768_124F
from families.minimax_h3.dit_builder import build_dit_engine


def _trajectory_profile():
    profile = replace(
        SOL_ENGINE_1344X768_124F,
        hidden_size=8,
        num_layers=1,
        num_refiner_layers=1,
        num_heads=4,
        head_dim=8,
        ffn_dim=16,
        text_dim=8,
        timestep_input_dim=4,
        timestep_hidden_size=8,
        timestep_embed_dim=4,
        rope_freq_dim=1,
        video_rows=12,
        audio_rows=6,
        min_text_rows=15,
        opt_text_rows=15,
        text_rows=15,
        padded_sequence_length=33,
    )
    return profile, Ref2VAPackedProfile((15, 15, 15), (14, 14, 14), (2, 2, 2))


def test_native_guide_fixture_keeps_clean_audio_and_target_budgets_separate():
    profile, reference = _trajectory_profile()
    profile.validate()
    reference.validate(profile)
    assert profile.sequence_length == profile.padded_sequence_length == 33
    assert reference.sequence_rows(profile) == (49, 49, 49)
    assert profile.video_rows == 12 and profile.audio_rows == 6 and profile.text_rows == 15


@pytest.mark.gpu
@pytest.mark.trt
def test_real_native_latent_guide_matches_49_forward_upstream(
    tmp_path, record_property
):
    import torch
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (
        MiniMaxH3SetTimestepsStep,
        patchify_video_latents,
    )
    from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler
    from .test_trt_builders import _weights

    assert importlib.metadata.version("diffusers") == "0.40.0"
    root = Path(os.environ["TRTMC_RUNTIME_ROOT"])
    probe = root / "minimax_h3_reference_denoiser_probe"
    assert probe.is_file() and (root / "libtrtmc_backend_trt.so").is_file()
    profile, reference = _trajectory_profile()
    weights = _weights(profile)
    np.savez(tmp_path / "synthetic-weights.npz", **weights)
    (tmp_path / "case.json").write_text(
        json.dumps(
            dict(
                profile=asdict(profile),
                request_seed=42,
                input_seed=91,
                weight_seed=7,
                rtol=0.016,
                atol=1e-5,
                sample_dtype="float32",
                model_dtype="bfloat16 with original FP32 boundaries",
                trained_weights=False,
                reference="diffusers==0.40.0",
                expected_forwards=49,
                guide_video_shape=[1,24,1,4,6],
                guide_audio_shape=[2,1,32], temporal_offset=17.5, media_time_origin=21,
                upstream_layout_revision="cbe4cc4b7c1126a903553f8e79e44ddfab782cf3",
            ),
            indent=2,
        )
    )
    (tmp_path / "adaln.plan").write_bytes(
        build_adaln_precompute_engine(
            weights, profile, input_timesteps=True, workspace_bytes=1 << 30
        )
    )
    (tmp_path / "denoiser.plan").write_bytes(
        build_dit_engine(
            weights,
            profile,
            reference_profile=reference,
            workspace_bytes=1 << 30,
        )
    )
    model = MiniMaxH3Transformer3DModel(
        num_attention_heads=4,
        attention_head_dim=8,
        hidden_size=8,
        num_layers=1,
        num_refiner_layers=1,
        ffn_dim=16,
        in_channels=24,
        audio_in_channels=32,
        text_dim=8,
        freq_dim=4,
        time_embed_hidden_dim=8,
        time_embed_dim=4,
        rope_freq_dim=1,
    )
    assert set(model.state_dict()) == set(weights)
    model.load_state_dict({name: torch.from_numpy(value) for name, value in weights.items()})
    for name, parameter in model.named_parameters():
        keep = any(part in name for part in model._keep_in_fp32_modules)
        parameter.data = parameter.data.to(torch.float32 if keep else torch.bfloat16)
    # Do not cast the whole module to BF16: its rotary buffer and the original
    # input/output heads and time MLP must remain FP32.
    model = model.cuda().eval()
    rng = np.random.default_rng(91)
    embeddings = rng.normal(0, 0.5, (15, 8)).astype(np.float32)
    references = [
        rng.normal(0, 0.2, (1, 24, 1, h, w)).astype(np.float32) for h, w in ((4, 6), (2, 4))
    ]
    embeddings.tofile(tmp_path / "text.f32.bin")
    for index, value in enumerate(references):
        value.tofile(tmp_path / f"reference-{index}.f32.bin")
    tags = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    guide_video = rng.normal(0, 0.2, (1, 24, 1, 4, 6)).astype(np.float32)
    guide_audio = rng.normal(0, 0.2, (2, 1, 32)).astype(np.float32)
    patchify_video_latents(torch.from_numpy(guide_video), (1, 2, 2)).numpy().tofile(tmp_path / "guide.video.f32.bin")
    guide_audio.reshape(2, 32).tofile(tmp_path / "guide.audio.f32.bin")
    source = Path(os.environ["TRTMC_H3_PACKED_SEQUENCE_SOURCE"])
    raw = source.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == "eb2464f91fdfd0ec8e12057bf53d2727c32678aef02d479b39a95d4f4b6fb6d4"
    tree = ast.parse(raw)
    tree.body = [n for n in tree.body if not (isinstance(n, ast.ImportFrom) and n.module == "vllm_omni.errors")]
    upstream = {"OmniClientError": ValueError}
    exec(compile(tree, str(source), "exec"), upstream)
    packed = upstream["minimax_h3_packed_sequence_ref2va_blocks"](
        text_len=15, latent_t=2, latent_h=4, latent_w=6, audio_t=3,
        ref_blocks=[dict(kind="image", latent_h=4, latent_w=6),
                    dict(kind="image", latent_h=2, latent_w=4),
                    dict(kind="latent_guide", latent_t=1, latent_h=4, latent_w=6, ref_audio_t=1)],
        temporal_offset=17.5, media_time_origin=21, seq_len=49)
    positions, packed_tags = packed["img_position_ids"], packed["token_tags"].clone()
    video_indices, audio_indices, text_indices = packed["img_pos"], packed["audio_pos"], packed["text_pos"]
    packed_tags[text_indices] = tags
    cv, ca = 14, 2
    assert int((~packed["update_mask"]).sum()) == cv
    assert int((~packed["audio_update_mask"]).sum()) == ca
    seed = 42
    generator = torch.Generator().manual_seed(seed)
    video_scheduler = MiniMaxH3Scheduler(shift=12.0)
    audio_scheduler = MiniMaxH3Scheduler(shift=3.0)
    video_scheduler.set_timesteps(50, device="cuda")
    audio_scheduler.set_timesteps(50, device="cuda")
    video = []
    for value in references:
        noise = torch.randn(value.shape, generator=generator).cuda()
        condition = video_scheduler.scale_noise(torch.from_numpy(value).cuda(), 0.999, noise)
        video.append(patchify_video_latents(condition, (1, 2, 2)))
    guide_noise = torch.randn(guide_video.shape, generator=generator).cuda()
    guide_condition = video_scheduler.scale_noise(torch.from_numpy(guide_video).cuda(), 0.999, guide_noise)
    video.append(patchify_video_latents(guide_condition, (1, 2, 2)))
    target = torch.randn((1, 24, 2, 4, 6), generator=generator).cuda()
    video.append(patchify_video_latents(target, (1, 2, 2)))
    video = torch.cat(video)
    audio = torch.cat((torch.from_numpy(guide_audio.reshape(2, 32)).cuda(),
                       torch.randn((6, 32), generator=generator).cuda()))
    fixed_audio = audio[:ca].clone()
    fixed_reference = video[:cv].clone()
    expected_steps = 0
    trace_video, trace_audio = [], []
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.no_grad(), torch.backends.cudnn.flags(allow_tf32=False):
            for t, a in zip(video_scheduler.timesteps, audio_scheduler.timesteps):
                trace_video.append(video.cpu().numpy().copy())
                trace_audio.append(audio.cpu().numpy().copy())
                times, inverse = MiniMaxH3SetTimestepsStep.build_row_timesteps(
                    video_indices,
                    audio_indices,
                    cv,
                    ca,
                    15,
                    float(t),
                    float(a),
                    max(float(t), 0.999),
                    1.0,
                )
                vv, av = model(
                    hidden_states=video[None],
                    audio_hidden_states=audio[None],
                    encoder_hidden_states=torch.from_numpy(embeddings)[None].cuda(),
                    timestep=times.cuda(),
                    timestep_indices=inverse.cuda(),
                    position_ids=positions.cuda(),
                    token_tags=packed_tags.cuda(),
                    video_indices=video_indices.cuda(),
                    audio_indices=audio_indices.cuda(),
                    text_indices=text_indices.cuda(),
                    return_dict=False,
                )
                video[cv:] = video_scheduler.step(
                    vv[0, cv:].float(), t, video[cv:], return_dict=False
                )[0]
                audio[ca:] = audio_scheduler.step(av[0, ca:].float(), a, audio[ca:], return_dict=False)[0]
                assert torch.equal(audio[:ca], fixed_audio)
                assert torch.equal(video[:cv], fixed_reference)
                expected_steps += 1
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    assert expected_steps == 49
    expected_video = video[cv:].cpu().numpy()
    expected_audio = audio[ca:].cpu().numpy()
    expected_video.tofile(tmp_path / "upstream.video.f32.bin")
    expected_audio.tofile(tmp_path / "upstream.audio.f32.bin")
    expected_traces = dict(video=np.stack(trace_video), audio=np.stack(trace_audio))
    for kind, value in expected_traces.items():
        value.tofile(tmp_path / f"upstream.trace.{kind}.f32.bin")
    results = {}
    failures = []
    for mode in ("staged", "resident"):
        prefix = tmp_path / mode
        execution = subprocess.run(
            [str(probe), str(root), str(tmp_path), str(tmp_path), str(prefix), mode, str(seed), "guide"],
            capture_output=True,
            text=True,
            check=False,
            timeout=180,
        )
        (tmp_path / f"{mode}.probe.log").write_text(execution.stdout + execution.stderr)
        if execution.returncode != 0:
            failures.append(mode + ": native probe failed: " + execution.stdout + execution.stderr)
            continue
        report = json.loads(execution.stdout)
        assert report["cold_forwards"] == report["warm_forwards"] == 49
        values = {
            kind: np.fromfile(str(prefix) + f".cold.{kind}.bin", dtype=np.float32).reshape(
                expected.shape
            )
            for kind, expected in (("video", expected_video), ("audio", expected_audio))
        }
        for kind, expected in (("video", expected_video), ("audio", expected_audio)):
            # Freeze the same BF16 model criterion used by the other H3
            # numerical component oracles. Samples stay FP32; never cast them
            # to half merely to obtain a pass. The scheduler has no double.
            try:
                torch.testing.assert_close(
                    torch.from_numpy(values[kind]),
                    torch.from_numpy(expected),
                    rtol=0.016,
                    atol=1e-5,
                )
            except AssertionError as error:
                failures.append(mode + ":" + kind + ":" + str(error))
            warm = np.fromfile(str(prefix) + f".warm.{kind}.bin", dtype=np.float32).reshape(
                expected.shape
            )
            assert values[kind].tobytes() == warm.tobytes()
        results[mode] = values
        for kind, expected in expected_traces.items():
            actual = np.fromfile(str(prefix) + f".cold.trace.{kind}.bin", dtype=np.float32).reshape(
                expected.shape
            )
            warm = np.fromfile(str(prefix) + f".warm.trace.{kind}.bin", dtype=np.float32).reshape(
                expected.shape
            )
            assert actual[0].tobytes() == expected[0].tobytes(), (
                "initial native noise/packing differs from upstream"
            )
            assert actual.tobytes() == warm.tobytes()
            fixed = cv if kind == "video" else ca
            assert all(value[:fixed].tobytes() == actual[0, :fixed].tobytes() for value in actual)
            try:
                torch.testing.assert_close(
                    torch.from_numpy(actual), torch.from_numpy(expected), rtol=0.016, atol=1e-5
                )
            except AssertionError as error:
                failures.append(mode + ":trace:" + kind + ":" + str(error))
            results[mode]["trace_" + kind] = actual
        record_property(
            mode,
            json.dumps(
                dict(
                    **report,
                    max_video_abs_error=float(np.abs(values["video"] - expected_video).max()),
                    max_audio_abs_error=float(np.abs(values["audio"] - expected_audio).max()),
                    trained_model_qualified=False,
                    native_Ref2VA_video_qualified=False,
                )
            ),
        )
    assert not failures, "\n".join(failures)
    for kind in ("video", "audio", "trace_video", "trace_audio"):
        assert results["staged"][kind].tobytes() == results["resident"][kind].tobytes()
