# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Actual native two-window inference; synthetic weights, no trained-video claim."""

import ast
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from .test_native_reference_continuation import _trajectory_profile
from families.minimax_h3.config import Ref2VAPackedProfile


def _window_profile():
    profile, _ = _trajectory_profile()
    profile = replace(profile, video_rows=222, audio_rows=414, padded_sequence_length=651)
    return profile, Ref2VAPackedProfile((15, 15, 15), (8, 50, 50), (0, 74, 74))


def test_window_fixture_admits_first_window_and_complete_av_guide():
    profile, reference = _window_profile()
    reference.validate(profile)
    assert reference.sequence_rows(profile) == (659, 775, 775)
    assert (profile.video_rows, profile.audio_rows) == (37 * 6, 207 * 2)


@pytest.mark.gpu
@pytest.mark.trt
def test_real_native_two_windows_match_98_forward_upstream(tmp_path, record_property):
    import torch
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (
        MiniMaxH3SetTimestepsStep,
        patchify_video_latents,
    )
    from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler
    from families.minimax_h3.adaln_builder import build_adaln_precompute_engine
    from families.minimax_h3.dit_builder import build_dit_engine
    from .test_trt_builders import _weights

    assert importlib.metadata.version("diffusers") == "0.40.0"
    runtime = Path(os.environ["TRTMC_RUNTIME_ROOT"])
    probe = runtime / "minimax_h3_reference_denoiser_probe"
    assert probe.is_file()
    profile, reference = _window_profile()
    weights = _weights(profile)
    np.savez(tmp_path / "synthetic-weights.npz", **weights)
    (tmp_path / "case.json").write_text(
        json.dumps(
            dict(
                profile=asdict(profile),
                reference_profile=asdict(reference),
                seed=42,
                windows=2,
                window_frames=124,
                overlap_frames=22,
                output_frames=226,
                window_latent_frames=37,
                window_audio_latents=207,
                cumulative_latent_frames=67,
                cumulative_audio_latents=377,
                expected_forwards=98,
                rtol=0.016,
                atol=1e-5,
                sample_dtype="float32",
                trained_weights=False,
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
        build_dit_engine(weights, profile, reference_profile=reference, workspace_bytes=1 << 30)
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
    model.load_state_dict({n: torch.from_numpy(v) for n, v in weights.items()})
    for name, parameter in model.named_parameters():
        keep = any(part in name for part in model._keep_in_fp32_modules)
        parameter.data = parameter.data.to(torch.float32 if keep else torch.bfloat16)
    model = model.cuda().eval()
    rng = np.random.default_rng(91)
    embeddings = rng.normal(0, 0.5, (15, 8)).astype(np.float32)
    images = [rng.normal(0, 0.2, (1, 24, 1, h, w)).astype(np.float32) for h, w in [(4, 6), (2, 4)]]
    embeddings.tofile(tmp_path / "text.f32.bin")
    for i, image in enumerate(images):
        image.tofile(tmp_path / f"reference-{i}.f32.bin")
    source = Path(os.environ["TRTMC_H3_PACKED_SEQUENCE_SOURCE"])
    raw = source.read_bytes()
    assert (
        hashlib.sha256(raw).hexdigest()
        == "eb2464f91fdfd0ec8e12057bf53d2727c32678aef02d479b39a95d4f4b6fb6d4"
    )
    tree = ast.parse(raw)
    tree.body = [
        n
        for n in tree.body
        if not (isinstance(n, ast.ImportFrom) and n.module == "vllm_omni.errors")
    ]
    upstream = {"OmniClientError": ValueError}
    exec(compile(tree, str(source), "exec"), upstream)
    text_tags = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1])

    def sample_window(guide=None):
        cv, ca = (8, 0) if guide is None else (50, 74)
        blocks = [
            dict(kind="image", latent_h=4, latent_w=6),
            dict(kind="image", latent_h=2, latent_w=4),
        ]
        if guide is not None:
            blocks.append(
                dict(kind="latent_guide", latent_t=7, latent_h=4, latent_w=6, ref_audio_t=37)
            )
        packed = upstream["minimax_h3_packed_sequence_ref2va_blocks"](
            text_len=15,
            latent_t=37,
            latent_h=4,
            latent_w=6,
            audio_t=207,
            ref_blocks=blocks,
            temporal_offset=0.0 if guide is None else 170.0,
            seq_len=659 if guide is None else 775,
        )
        tags = packed["token_tags"].clone()
        tags[packed["text_pos"]] = text_tags
        gen = torch.Generator().manual_seed(42)
        vs, aus = MiniMaxH3Scheduler(shift=12.0), MiniMaxH3Scheduler(shift=3.0)
        vs.set_timesteps(50, device="cuda")
        aus.set_timesteps(50, device="cuda")
        video = []
        for image in images:
            noise = torch.randn(image.shape, generator=gen).cuda()
            video.append(
                patchify_video_latents(
                    vs.scale_noise(torch.from_numpy(image).cuda(), 0.999, noise), (1, 2, 2)
                )
            )
        if guide is not None:
            noise = patchify_video_latents(
                torch.randn((1, 24, 7, 4, 6), generator=gen).cuda(), (1, 2, 2)
            )
            video.append(vs.scale_noise(guide[0], 0.999, noise))
        video.append(
            patchify_video_latents(torch.randn((1, 24, 37, 4, 6), generator=gen).cuda(), (1, 2, 2))
        )
        video = torch.cat(video)
        audio = torch.randn((414, 32), generator=gen).cuda()
        if guide is not None:
            audio = torch.cat((guide[1].reshape(74, 32), audio))
        fixed_v, fixed_a = video[:cv].clone(), audio[:ca].clone()
        traces = {"video": [], "audio": []}
        for t, a in zip(vs.timesteps, aus.timesteps, strict=True):
            traces["video"].append(video.cpu().numpy().copy())
            traces["audio"].append(audio.cpu().numpy().copy())
            times, inverse = MiniMaxH3SetTimestepsStep.build_row_timesteps(
                packed["img_pos"],
                packed["audio_pos"],
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
                position_ids=packed["img_position_ids"].cuda(),
                token_tags=tags.cuda(),
                video_indices=packed["img_pos"].cuda(),
                audio_indices=packed["audio_pos"].cuda(),
                text_indices=packed["text_pos"].cuda(),
                return_dict=False,
            )
            video[cv:] = vs.step(vv[0, cv:].float(), t, video[cv:], return_dict=False)[0]
            audio[ca:] = aus.step(av[0, ca:].float(), a, audio[ca:], return_dict=False)[0]
            assert torch.equal(video[:cv], fixed_v) and torch.equal(audio[:ca], fixed_a)
        assert len(traces["video"]) == 49
        return (
            video[cv:].clone(),
            audio[ca:].reshape(2, 207, 32).clone(),
            {k: np.stack(v) for k, v in traces.items()},
        )

    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.no_grad(), torch.backends.cudnn.flags(allow_tf32=False):
            first_v, first_a, first_trace = sample_window()
            child_v, child_a, child_trace = sample_window((first_v[-42:], first_a[:, -37:]))
            expected = {
                "video": torch.cat((first_v, child_v[42:])).cpu().numpy(),
                "audio": torch.cat((first_a, child_a[:, 37:]), dim=1)
                .reshape(754, 32)
                .cpu()
                .numpy(),
            }
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    for kind in expected:
        expected[kind].tofile(tmp_path / f"upstream.{kind}.bin")
        first_trace[kind].tofile(tmp_path / f"upstream.first.trace.{kind}.bin")
        child_trace[kind].tofile(tmp_path / f"upstream.child.trace.{kind}.bin")
    results = {}
    for mode in ["staged", "resident"]:
        prefix = tmp_path / mode
        r = subprocess.run(
            [
                str(probe),
                str(runtime),
                str(tmp_path),
                str(tmp_path),
                str(prefix),
                mode,
                "42",
                "windows",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        (tmp_path / f"{mode}.probe.log").write_text(r.stdout + r.stderr)
        assert r.returncode == 0, r.stdout + r.stderr
        report = json.loads(r.stdout)
        assert report["cold_forwards"] == report["warm_forwards"] == 98
        outputs = {}
        for kind, want in expected.items():
            cold = np.fromfile(str(prefix) + f".cold.{kind}.bin", np.float32).reshape(want.shape)
            warm = np.fromfile(str(prefix) + f".warm.{kind}.bin", np.float32).reshape(want.shape)
            assert cold.tobytes() == warm.tobytes()
            torch.testing.assert_close(
                torch.from_numpy(cold), torch.from_numpy(want), rtol=0.016, atol=1e-5
            )
            outputs[kind] = cold
            trace = np.fromfile(str(prefix) + f".cold.trace.{kind}.bin", np.float32)
            warm_trace = np.fromfile(str(prefix) + f".warm.trace.{kind}.bin", np.float32)
            assert trace.tobytes() == warm_trace.tobytes()
            split = first_trace[kind].size
            actual_first = trace[:split].reshape(first_trace[kind].shape)
            actual_child = trace[split:].reshape(child_trace[kind].shape)
            assert actual_first[0].tobytes() == first_trace[kind][0].tobytes()
            for actual, wanted, fixed in [
                (actual_first, first_trace[kind], 8 if kind == "video" else 0),
                (actual_child, child_trace[kind], 50 if kind == "video" else 74),
            ]:
                torch.testing.assert_close(
                    torch.from_numpy(actual), torch.from_numpy(wanted), rtol=0.016, atol=1e-5
                )
                assert all(x[:fixed].tobytes() == actual[0, :fixed].tobytes() for x in actual)
            if kind == "audio":
                retained_tail = cold.reshape(2, 377, 32)[:, 170:207].reshape(74, 32)
                assert actual_child[0, :74].tobytes() == retained_tail.tobytes()
            else:
                # Verify guide augmentation against the retained native parent,
                # rather than assuming the native and HF parent are byte-identical.
                gen = torch.Generator().manual_seed(42)
                for image in images:
                    torch.randn(image.shape, generator=gen)
                noise = patchify_video_latents(
                    torch.randn((1, 24, 7, 4, 6), generator=gen), (1, 2, 2)
                ).numpy()
                augmented = (
                    np.float32(0.999) * cold[180:222] + (np.float32(1) - np.float32(0.999)) * noise
                )
                assert actual_child[0, 8:50].tobytes() == augmented.tobytes()
        results[mode] = outputs
        record_property(
            mode,
            json.dumps(
                dict(
                    **report,
                    max_video_abs_error=float(np.abs(outputs["video"] - expected["video"]).max()),
                    max_audio_abs_error=float(np.abs(outputs["audio"] - expected["audio"]).max()),
                    trained_model_qualified=False,
                    full_video_qualified=False,
                )
            ),
        )
    for kind in expected:
        assert results["staged"][kind].tobytes() == results["resident"][kind].tobytes()
