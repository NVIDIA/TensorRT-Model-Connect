# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from families.minimax_h3.audio_vae_builder import (
    BATCH,
    LATENT_FRAMES,
    SAMPLES,
    _weight_norm,
    build_audio_vae_decoder_engine,
    checkpoint_keys,
    checkpoint_shapes,
    validate_decoder_weights,
)


def test_audio_decoder_partition_and_full_stereo_shape():
    shapes = checkpoint_shapes()
    assert len(shapes) == len(checkpoint_keys()) == 914
    assert len(set(checkpoint_keys())) == 914
    assert all(name.startswith(("dec_in_proj.", "decoder.")) for name in shapes)
    assert shapes["decoder.ups.0.0.weight_v"] == (1024, 512, 9)
    assert shapes["decoder.ups.6.0.weight_v"] == (16, 8, 4)
    assert shapes["decoder.conv_post.weight_v"] == (1, 8, 7)
    assert "decoder.conv_post.bias" not in shapes
    assert (BATCH, LATENT_FRAMES, SAMPLES) == (2, 207, 165600)


@pytest.mark.parametrize(
    "mean,std",
    [
        ([0.0] * 31, [1.0] * 32),
        ([[0.0] * 32], [1.0] * 32),
        ([np.nan] * 32, [1.0] * 32),
        ([0.0] * 32, [np.inf] * 32),
        ([0.0] * 32, [0.0] * 32),
        ([0.0] * 32, [-1.0] * 32),
    ],
)
def test_audio_decoder_rejects_bad_statistics_before_cuda(mean, std):
    with pytest.raises(ValueError, match="latent|32 latent"):
        build_audio_vae_decoder_engine({}, mean, std)


def test_audio_decoder_rejects_wrong_partition_and_dtype_before_cuda():
    with pytest.raises(ValueError, match="partition"):
        build_audio_vae_decoder_engine({}, [0.0] * 32, [1.0] * 32)
    # Broadcast views validate real released shapes without allocating trained
    # parameter storage or substituting these fixtures into a GPU request.
    weights = {
        name: np.broadcast_to(np.float32(1), shape) for name, shape in checkpoint_shapes().items()
    }
    validate_decoder_weights(weights)
    weights["dec_in_proj.weight"] = np.zeros((1,), np.float32)
    with pytest.raises(ValueError, match="FP32.*shape"):
        validate_decoder_weights(weights)
    weights["dec_in_proj.weight"] = np.broadcast_to(np.float16(1), (2048, 32, 1))
    with pytest.raises(ValueError, match="FP32.*shape"):
        validate_decoder_weights(weights)


def test_audio_weight_norm_preserves_original_output_axis():
    value = np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4)
    scale = np.array([2.0, 3.0], np.float32).reshape(2, 1, 1)
    actual = _weight_norm({"x.weight_v": value, "x.weight_g": scale}, "x")
    expected = value * (scale / np.sqrt((value * value).sum(axis=(1, 2), keepdims=True)))
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float32 and actual.flags.c_contiguous
    with pytest.raises(ValueError, match="weight norm"):
        _weight_norm({"x.weight_v": np.zeros_like(value), "x.weight_g": scale}, "x")


@pytest.mark.parametrize("consume", [False, True])
def test_audio_decoder_cleans_borrowed_buffers_on_graph_failure(monkeypatch, consume):
    from families.minimax_h3 import audio_vae_builder as audio

    network = object()
    weights = {"fixture": np.ones(1, np.float32)}

    class Config:
        def clear_flag(self, flag):
            assert flag == audio.trt.BuilderFlag.TF32

        def set_memory_pool_limit(self, _pool, value):
            self.limit = value

        def get_memory_pool_limit(self, _pool):
            return self.limit

    class Builder:
        def create_network(self, _flags):
            return network

        def create_builder_config(self):
            return Config()

    def fail(*_args):
        audio.op._WEIGHT_BUFFER_KEEPALIVE[id(network)] = [weights["fixture"]]
        raise RuntimeError("fixture graph failure")

    monkeypatch.setattr(audio.trt, "Builder", lambda _logger: Builder())
    monkeypatch.setattr(audio, "validate_decoder_weights", lambda _weights: None)
    monkeypatch.setattr(audio, "_network", fail)
    with pytest.raises(RuntimeError, match="fixture graph failure"):
        audio.build_audio_vae_decoder_engine(
            weights, [0.0] * 32, [1.0] * 32, consume_weights=consume, workspace_bytes=1 << 30
        )
    assert id(network) not in audio.op._WEIGHT_BUFFER_KEEPALIVE
    assert (not weights) == consume


@pytest.mark.gpu
@pytest.mark.trt
def test_trained_audio_decoder_matches_native_stereo_cold_warm(tmp_path, record_property):
    """Full trained 207-frame audio decoder, not generated-dialogue qualification."""
    import hashlib
    import importlib.metadata
    import json
    import os
    from pathlib import Path
    import subprocess

    from safetensors import safe_open
    import torch
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3_audio import (
        MiniMaxH3AudioBigVGANDecoder,
    )
    from tensorrt_model_connect.bundle_writer import BundleWriter

    from families.minimax_h3.audio_vae_builder import LATENT_DIM, DECODER_DIM

    assert importlib.metadata.version("diffusers") == "0.40.0"
    model_dir = Path(os.environ["TRTMC_MINIMAX_H3_MODEL_DIR"])
    runtime_root = Path(os.environ["TRTMC_RUNTIME_ROOT"])
    component = model_dir / "audio_vae"
    config = json.loads((component / "config.json").read_text())
    expected_config = dict(
        latent_channels=32,
        latent_dim=2048,
        decoder_dim=1024,
        decoder_rates=[5, 5, 2, 2, 2, 2, 2],
        decoder_kernel_sizes=[9, 9, 4, 4, 4, 4, 4],
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5]] * 3,
        sampling_rate=32000,
    )
    assert all(config[name] == value for name, value in expected_config.items())
    files = list(component.glob("*.safetensors"))
    assert len(files) == 1
    with safe_open(files[0], framework="pt", device="cpu") as reader:
        all_names = set(reader.keys())
        assert {name for name in all_names if name.startswith(("dec_in_proj.", "decoder."))} == set(
            checkpoint_keys()
        )
        state = {name: reader.get_tensor(name) for name in checkpoint_keys()}
    assert all(value.dtype == torch.float32 for value in state.values())
    weights = {name: value.numpy() for name, value in state.items()}
    validate_decoder_weights(weights)
    with files[0].open("rb") as file:
        checkpoint_sha = hashlib.file_digest(file, "sha256").hexdigest()
    assert checkpoint_sha == "52c59e67ba8de5477c81bfbced0327aabf500f1bfdeefd5ee754529241cb26cb"
    metadata = dict(
        hf_id="MiniMaxAI/MiniMax-H3",
        hf_revision="48d93ede732756e404a3b1b2f3b3a9b5a22f6cfc",
        checkpoint_sha256=checkpoint_sha,
        component_config=config,
        decoder_tensor_count=len(state),
        decoder_parameters=sum(value.numel() for value in state.values()),
        input_seed=47,
        latent_frames=207,
        samples_per_channel=165600,
        sample_rate=32000,
        rtol=1.3e-6,
        atol=1e-5,
        comparison_dtype="float32",
        TF32=False,
        full_H3_video_qualified=False,
        dialogue_quality_qualified=False,
    )
    (tmp_path / "case.json").write_text(json.dumps(metadata, indent=2))
    (tmp_path / "checkpoint-shapes.json").write_text(
        json.dumps({name: list(value.shape) for name, value in state.items()}, indent=2)
    )
    plan = build_audio_vae_decoder_engine(
        weights, config["latents_mean"], config["latents_std"], workspace_bytes=1 << 30
    )
    writer = BundleWriter(tmp_path / "audio.component.bundle")
    # Component fixture only; this is deliberately not passed to the public
    # H3 factory, which does not yet advertise image-reference audio/video.
    writer.set_header(family="minimax_h3", task="image_generation", backend="trt")
    writer.add_bytes("audio_vae.plan", plan)
    writer.finish()
    del plan, writer
    model = torch.nn.Module()
    with torch.device("meta"):
        model.dec_in_proj = torch.nn.Conv1d(32, LATENT_DIM, 1)
        model.decoder = MiniMaxH3AudioBigVGANDecoder(
            in_channels=LATENT_DIM,
            upsample_initial_channel=DECODER_DIM,
            upsample_rates=tuple(config["decoder_rates"]),
            upsample_kernel_sizes=tuple(config["decoder_kernel_sizes"]),
            resblock_kernel_sizes=tuple(config["resblock_kernel_sizes"]),
            resblock_dilation_sizes=tuple(tuple(x) for x in config["resblock_dilation_sizes"]),
        )
    assert set(model.state_dict()) == set(state)
    model.load_state_dict(state, assign=True)
    assert all(not x.is_meta for x in (*model.parameters(), *model.buffers()))
    model = model.cuda().eval()
    rows = np.random.default_rng(47).normal(0, 0.5, (2, 207, 32)).astype(np.float32)
    rows.tofile(tmp_path / "rows.f32.bin")
    latent = torch.from_numpy(rows).permute(0, 2, 1).contiguous().cuda()
    mean = torch.tensor(config["latents_mean"], device="cuda").view(1, 32, 1)
    std = torch.tensor(config["latents_std"], device="cuda").view(1, 32, 1)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.no_grad(), torch.backends.cudnn.flags(allow_tf32=False):
            expected = model.decoder(model.dec_in_proj(latent * std + mean))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    assert expected.shape == (2, 1, 165600) and expected.dtype == torch.float32
    expected = expected[:, 0].T.contiguous().cpu()
    expected.numpy().tofile(tmp_path / "upstream.interleaved.f32.bin")
    results, failures = {}, []
    for mode in ("staged", "resident"):
        prefix = tmp_path / mode
        run = subprocess.run(
            [
                str(runtime_root / "minimax_h3_audio_decoder_probe"),
                str(runtime_root),
                str(tmp_path / "audio.component.bundle"),
                str(tmp_path / "rows.f32.bin"),
                str(prefix),
                mode,
            ],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        (tmp_path / f"{mode}.probe.log").write_text(run.stdout + run.stderr)
        if run.returncode:
            failures.append(mode + ": probe failed: " + run.stdout + run.stderr)
            continue
        report = json.loads(run.stdout)
        assert report == dict(
            loads=2 if mode == "staged" else 1,
            sample_rate=32000,
            channels=2,
            samples_per_channel=165600,
        )
        cold = np.fromfile(str(prefix) + ".cold.bin", dtype=np.float32).reshape(165600, 2)
        assert cold.tobytes() == Path(str(prefix) + ".warm.bin").read_bytes()
        actual = torch.from_numpy(cold)
        assert torch.isfinite(actual).all() and actual.min() >= -1 and actual.max() <= 1
        error = actual - expected
        record_property(
            mode,
            json.dumps(
                dict(
                    **report,
                    max_abs_error=error.abs().max().item(),
                    mean_abs_error=error.abs().mean().item(),
                    sample_dtype="float32",
                    full_H3_video_qualified=False,
                )
            ),
        )
        try:
            # Frozen before the first GPU run. No half cast or threshold
            # relaxation: this audio checkpoint and its computation are FP32.
            torch.testing.assert_close(actual, expected, rtol=1.3e-6, atol=1e-5)
        except AssertionError as error:
            failures.append(mode + ": " + str(error))
        results[mode] = cold
    if set(results) == {"staged", "resident"}:
        assert results["staged"].tobytes() == results["resident"].tobytes()
    assert not failures, "\n".join(failures)
