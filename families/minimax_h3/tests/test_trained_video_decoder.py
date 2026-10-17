# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Full original trained video decoder through the actual native controller.

Synthetic normalized input latents, not a prompt-conditioned H3 result. The
model and 124-frame 1344x768 output are not reduced for this comparison.
"""

import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest


_RTOL = 1e-3
_ATOL = 1e-5
_SHAPE = (124, 1344, 768, 3)
_SHARDS = {
    "diffusion_pytorch_model-00001-of-00003.safetensors": "72f4c6be84ac0674f27398cde991dd9d719762f3952c4921aa66b2ce542f6374",
    "diffusion_pytorch_model-00002-of-00003.safetensors": "2e05e8bc23fa4071043e17fd242be8acd0685e781a43987432b2eae925be4198",
    "diffusion_pytorch_model-00003-of-00003.safetensors": "c05d6ac4b1a33de372799d708531da6320f6a3ce6d1ce6d895e770988e004a39",
}


def _sha(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _compare_frames(actual_path, expected_path, shape):
    import torch

    count = int(np.prod(shape))
    assert actual_path.stat().st_size == expected_path.stat().st_size == count * 4
    actual = np.memmap(actual_path, dtype=np.float32, mode="r", shape=shape)
    expected = np.memmap(expected_path, dtype=np.float32, mode="r", shape=shape)
    maximum, error_sum, failed_frames = 0.0, 0.0, 0
    errors = []
    for frame in range(shape[0]):
        a = torch.from_numpy(np.array(actual[frame], copy=True))
        b = torch.from_numpy(np.array(expected[frame], copy=True))
        assert a.dtype == b.dtype == torch.float32
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        difference = (a - b).abs().numpy()
        maximum = max(maximum, float(difference.max()))
        error_sum += float(difference.sum(dtype=np.float64))
        try:
            # Same predeclared FP16 model criterion, applied to the actual
            # FP32 output contract without any comparison cast or subsampling.
            torch.testing.assert_close(a, b, rtol=_RTOL, atol=_ATOL)
        except AssertionError as error:
            failed_frames += 1
            if len(errors) < 3:
                errors.append(f"frame {frame}: {error}")
    return dict(
        max_abs_error=maximum,
        mean_abs_error=error_sum / count,
        compared_elements=count,
        failed_frames=failed_frames,
    ), errors


def test_full_video_comparison_checks_every_frame_and_keeps_fp32(tmp_path):
    expected = np.arange(36, dtype=np.float32).reshape(2, 2, 3, 3) / 35
    a, b = tmp_path / "actual.bin", tmp_path / "expected.bin"
    expected.tofile(a)
    expected.tofile(b)
    result, errors = _compare_frames(a, b, expected.shape)
    assert not errors and result["compared_elements"] == 36 and result["failed_frames"] == 0
    changed = expected.copy()
    changed[1, 0, 0, 0] += 0.1
    changed.tofile(a)
    result, errors = _compare_frames(a, b, expected.shape)
    assert len(errors) == result["failed_frames"] == 1
    assert errors[0].startswith("frame 1:")


def _large_inventory(root):
    files = {
        str(path.relative_to(root)): dict(bytes=path.stat().st_size, sha256=_sha(path))
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink() and path.stat().st_size > 32 * 1024**2
    }
    with (root / "retained-large-files.json").open("x") as file:
        json.dump(files, file, indent=2, allow_nan=False)


@pytest.mark.gpu
@pytest.mark.trt
def test_trained_video_decoder_full_portrait_matches_native_memory_presets(
    tmp_path, record_property
):
    cached_reference = os.environ.get("TRTMC_MINIMAX_H3_VIDEO_REFERENCE_CACHE")
    if cached_reference:
        from families.minimax_h3.tests.vae_cached_video_case import run_cached_case

        return run_cached_case(tmp_path, record_property, Path(cached_reference))

    import torch
    from diffusers import AutoencoderKLMiniMaxH3
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import MiniMaxH3VideoRotaryPosEmbed
    from diffusers.modular_pipelines.minimax_h3.before_denoise import patchify_video_latents
    from tensorrt_model_connect.bundle_writer import BundleWriter

    from families.minimax_h3.checkpoint import load_selected_component_state_dict, numpy_state
    from families.minimax_h3.vae_builder import build_vae_tile_decoder_engine, checkpoint_keys

    assert importlib.metadata.version("diffusers") == "0.40.0"
    assert shutil.disk_usage(tmp_path).free >= 36 * 1024**3, (
        "full trained video proof needs reserved disk; do not delete old assets"
    )
    torch.set_num_threads(4)
    root = Path(os.environ["TRTMC_RUNTIME_ROOT"])
    probe = root / "minimax_h3_video_decoder_probe"
    assert probe.is_file() and (root / "libtrtmc_backend_trt.so").is_file()
    component = Path(os.environ["TRTMC_MINIMAX_H3_MODEL_DIR"]) / "vae"
    config = json.loads((component / "config.json").read_text())
    architecture = dict(
        latent_channels=24,
        decoder_num_layers=36,
        decoder_num_attention_heads=32,
        decoder_attention_head_dim=64,
        decoder_num_register_tokens=4,
        decoder_ffn_mult=4,
        decoder_rope_theta=100.0,
        decoder_rope_dim_ratio=0.75,
        decoder_norm_eps=1e-5,
        clip_length=17,
        token_drop=3,
    )
    assert all(config[name] == value for name, value in architecture.items())
    assert set(path.name for path in component.glob("*.safetensors")) == set(_SHARDS)
    for name, expected_sha in _SHARDS.items():
        assert _sha(component / name) == expected_sha
    index = json.loads(next(component.glob("*.safetensors.index.json")).read_text())["weight_map"]
    assert {name for name in index if name.startswith(("decoder.", "post_quant_conv."))} == set(
        checkpoint_keys()
    )
    state = load_selected_component_state_dict(component, checkpoint_keys())
    assert len(state) == 585 and all(value.dtype == torch.float32 for value in state.values())
    weights = numpy_state(state)
    (tmp_path / "case.json").write_text(
        json.dumps(
            dict(
                hf_id="MiniMaxAI/MiniMax-H3",
                hf_revision="48d93ede732756e404a3b1b2f3b3a9b5a22f6cfc",
                checkpoint_shards=_SHARDS,
                decoder_tensors=585,
                architecture=architecture,
                output_shape=_SHAPE,
                latent_shape=[1, 24, 37, 84, 48],
                input_seed=831,
                synthetic_normalized_input="N(0,1)",
                trained_weights=True,
                rtol=_RTOL,
                atol=_ATOL,
                comparison_dtype="float32",
                TF32=False,
                upstream_source_sha256=_sha(Path(inspect.getfile(AutoencoderKLMiniMaxH3))),
                native_ref2va_video_qualified=False,
                prompt_or_dialogue_quality_qualified=False,
            ),
            indent=2,
        )
    )
    (tmp_path / "checkpoint-shapes.json").write_text(
        json.dumps({name: list(value.shape) for name, value in state.items()}, indent=2)
    )
    try:
        # Construct unused encoder parameters on meta; decode still invokes the
        # actual upstream VAE methods and the complete original trained decoder.
        with torch.device("meta"):
            model = AutoencoderKLMiniMaxH3.from_config(config)
        all_names = set(model.state_dict())
        assert {
            name for name in all_names if name.startswith(("decoder.", "post_quant_conv."))
        } == set(state)
        assert all(
            tuple(model.state_dict()[name].shape) == tuple(value.shape)
            for name, value in state.items()
        )
        loaded = model.load_state_dict(state, strict=False, assign=True)
        assert not loaded.unexpected_keys and set(loaded.missing_keys) == all_names - set(state)
        # inv_freq is deliberately nonpersistent: restore it with the original
        # upstream constructor instead of leaving an uninitialized meta buffer.
        model.decoder.rope = MiniMaxH3VideoRotaryPosEmbed(48, theta=100.0)
        model.decoder.cuda()
        model.post_quant_conv.cuda()
        model.eval()
        assert all(
            value.device.type == "cuda" and value.dtype == torch.float32
            for value in model.decoder.parameters()
        )
        normalized = np.random.default_rng(831).normal(0, 1, (1, 24, 37, 84, 48)).astype(np.float32)
        packed = patchify_video_latents(torch.from_numpy(normalized), (1, 2, 2)).numpy()
        assert packed.shape == (37296, 96)
        packed.tofile(tmp_path / "rows.f32.bin")
        latent = torch.from_numpy(normalized).cuda()
        mean = torch.tensor(config["latents_mean"], dtype=torch.float32, device="cuda").view(
            1, 24, 1, 1, 1
        )
        std = torch.tensor(config["latents_std"], dtype=torch.float32, device="cuda").view(
            1, 24, 1, 1, 1
        )
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            with torch.no_grad(), torch.backends.cudnn.flags(allow_tf32=False):
                denormalized = latent * std + mean
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    decoded = model.decode(denormalized, return_dict=False)[0]
                assert decoded.shape == (1, 3, 124, 1344, 768) and decoded.dtype == torch.float16
                pixel_std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1, 1)
                pixel_mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1, 1)
                expected = (decoded.float() * pixel_std + pixel_mean).clamp(0, 1)
                expected = expected[0].permute(1, 2, 3, 0).contiguous().cpu().numpy()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        expected_path = tmp_path / "upstream.f32.bin"
        expected.tofile(expected_path)
        del expected, model, latent, mean, std, decoded, denormalized
        failures, hashes = [], {}
        for preset, batch, mode in [("minimal", 1, "staged"), ("original", 28, "resident")]:
            config_path = tmp_path / (preset + ".config.json")
            config_path.write_text(
                json.dumps(
                    dict(
                        latent_frames=37,
                        latent_height=84,
                        latent_width=48,
                        tile_batch=batch,
                        latents_mean=config["latents_mean"],
                        latents_std=config["latents_std"],
                    )
                )
            )
            bundle = tmp_path / (preset + ".component.bundle")
            plan = build_vae_tile_decoder_engine(weights, batch_size=batch, workspace_bytes=8 << 30)
            writer = BundleWriter(bundle)
            writer.set_header(family="minimax_h3", task="image_generation", backend="trt")
            writer.add_bytes("vae.plan", plan)
            writer.finish()
            del plan, writer
            prefix = tmp_path / preset
            run = subprocess.run(
                [
                    str(probe),
                    str(root),
                    str(bundle),
                    str(tmp_path / "rows.f32.bin"),
                    str(config_path),
                    str(prefix),
                    mode,
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=600,
            )
            (tmp_path / (preset + ".probe.log")).write_text(run.stdout + run.stderr)
            if run.returncode:
                failures.append(preset + ": native probe failed: " + run.stdout + run.stderr)
                continue
            report = json.loads(run.stdout)
            assert (report["frames"], report["height"], report["width"], report["tile_batch"]) == (
                *_SHAPE[:3],
                batch,
            )
            assert report["loads"] == (2 if mode == "staged" else 1)
            assert report["cold_forwards"] == report["warm_forwards"] == (196 if batch == 1 else 7)
            cold, warm = Path(str(prefix) + ".cold.f32.bin"), Path(str(prefix) + ".warm.f32.bin")
            assert _sha(cold) == _sha(warm)
            comparison, errors = _compare_frames(cold, expected_path, _SHAPE)
            hashes[preset] = _sha(cold)
            (tmp_path / (preset + ".comparison.json")).write_text(
                json.dumps(
                    dict(**report, **comparison, errors=errors, rtol=_RTOL, atol=_ATOL), indent=2
                )
            )
            record_property(
                preset,
                json.dumps(
                    dict(
                        **report,
                        **comparison,
                        decoder_component_only=True,
                        native_ref2va_video_qualified=False,
                    )
                ),
            )
            failures.extend(preset + ": " + error for error in errors)
        (tmp_path / "cross-preset.json").write_text(
            json.dumps(
                dict(
                    byte_exact=hashes.get("minimal") == hashes.get("original")
                    if len(hashes) == 2
                    else None,
                    passing_requires_both_presets_against_upstream=True,
                ),
                indent=2,
            )
        )
        assert not failures, "\n".join(failures)
    finally:
        # Full videos and multi-GB bundles remain in the owned target result
        # directory. The campaign mirrors bounded logs/metadata and hashes every
        # retained complete artifact; it does not drop data after a mismatch.
        _large_inventory(tmp_path)
