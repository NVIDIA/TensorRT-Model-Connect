# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tiny-random parity: LTX-2.5 audio decoder engine (audio VAE decoder + vocoder with BWE) vs diffusers.

Real structure (causal pixel-norm audio decoder; BigVGAN-style anti-aliased SnakeBeta vocoder with
the real 160x stage-1 and 240x BWE upsampling, causal STFT + mel, Hann x3 resampler), narrow channels.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tensorrt")
if not torch.cuda.is_available():
    pytest.skip("CUDA is required for LTX-2.5 engine parity tests", allow_module_level=True)
safetensors_torch = pytest.importorskip("safetensors.torch")

from families.ltx2.tests.engine_runner import cosine, rel_l2, run_plan  # noqa: E402

TINY_AUDIO_VAE = {
    "attn_resolutions": None, "base_channels": 32, "causality_axis": "height", "ch_mult": [1, 2, 4],
    "double_z": True, "dropout": 0.0, "in_channels": 2, "is_causal": True, "latent_channels": 4, "mel_bins": 32,
    "mel_hop_length": 160, "mid_block_add_attention": False, "norm_type": "pixel", "num_res_blocks": 1,
    "output_channels": 2, "resolution": 256, "sample_rate": 16000,
}
TINY_VOCODER = {
    "act_fn": "snakebeta", "antialias": True, "antialias_kernel_size": 12, "antialias_ratio": 2,
    "bwe_act_fn": "snakebeta", "bwe_antialias": True, "bwe_antialias_kernel_size": 12, "bwe_antialias_ratio": 2,
    "bwe_final_act_fn": None, "bwe_final_bias": False, "bwe_hidden_channels": 32, "bwe_in_channels": 64,
    "bwe_leaky_relu_negative_slope": 0.1, "bwe_out_channels": 2,
    "bwe_resnet_dilations": [[1, 3, 5], [1, 3, 5], [1, 3, 5]], "bwe_resnet_kernel_sizes": [3, 7, 11],
    "bwe_upsample_factors": [6, 5, 2, 2, 2], "bwe_upsample_kernel_sizes": [12, 11, 4, 4, 4],
    "filter_length": 512, "final_act_fn": None, "final_bias": False, "hidden_channels": 64, "hop_length": 80,
    "in_channels": 64, "input_sampling_rate": 16000, "leaky_relu_negative_slope": 0.1, "num_mel_channels": 32,
    "out_channels": 2, "output_sampling_rate": 48000, "resnet_dilations": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "resnet_kernel_sizes": [3, 7, 11], "upsample_factors": [5, 2, 2, 2, 2, 2],
    "upsample_kernel_sizes": [11, 4, 4, 4, 4, 4], "window_length": 512,
}
SA = 6


def _save(module, folder, cfg):
    folder.mkdir()
    safetensors_torch.save_file({k: v.contiguous() for k, v in module.state_dict().items()},
                                str(folder / "diffusion_pytorch_model.safetensors"))
    (folder / "config.json").write_text(json.dumps(cfg), encoding="utf-8")


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return make_tiny(tmp_path_factory.mktemp("tiny_audio"))


def make_tiny(root):
    """Random tiny audio VAE + vocoder saved as diffusers folders under ``root``."""
    from diffusers import AutoencoderKLLTX2Audio
    from diffusers.pipelines.ltx2.vocoder import LTX2VocoderWithBWE

    gen = torch.Generator().manual_seed(4)
    vae = AutoencoderKLLTX2Audio(**TINY_AUDIO_VAE).eval()
    voc = LTX2VocoderWithBWE(**TINY_VOCODER).eval()
    with torch.no_grad():
        for name, p in list(vae.named_parameters()) + list(voc.named_parameters()):
            if name.endswith(("alpha", "beta")):
                p.copy_(0.2 * torch.randn(p.shape, generator=gen))
            elif p.ndim >= 2:
                p.copy_(torch.randn(p.shape, generator=gen) * (1.0 / p[0].numel()) ** 0.5)
            else:
                p.copy_(0.05 * torch.randn(p.shape, generator=gen))
        vae.latents_mean.copy_(0.2 * torch.randn(vae.latents_mean.shape, generator=gen))
        vae.latents_std.copy_(0.5 + torch.rand(vae.latents_std.shape, generator=gen))
        # A real windowed DFT basis and a positive mel filterbank keep the log-mel well conditioned.
        n_fft = 512
        nf = n_fft // 2 + 1
        t = torch.arange(n_fft, dtype=torch.float64)
        win = torch.hann_window(n_fft, periodic=True, dtype=torch.float64)
        f = torch.arange(nf, dtype=torch.float64)[:, None]
        cos = torch.cos(2 * torch.pi * f * t / n_fft) * win
        sin = -torch.sin(2 * torch.pi * f * t / n_fft) * win
        voc.mel_stft.stft_fn.forward_basis.copy_(torch.cat([cos, sin], 0).unsqueeze(1).float())
        voc.mel_stft.mel_basis.copy_(torch.rand(voc.mel_stft.mel_basis.shape, generator=gen) / 20)
    _save(vae, root / "audio_vae", TINY_AUDIO_VAE)
    _save(voc, root / "vocoder", TINY_VOCODER)
    return vae, voc, root


def test_audio_decoder_tiny_parity(tiny) -> None:
    from families.ltx2.audio_builder import build_audio_decoder_engine

    vae, voc, root = tiny
    lat_m = TINY_AUDIO_VAE["mel_bins"] // 4
    packed = torch.randn(1, SA, TINY_AUDIO_VAE["latent_channels"] * lat_m, generator=torch.Generator().manual_seed(1))
    plan = build_audio_decoder_engine(root, audio_frames=SA, debug_mel=True)
    got = run_plan(plan, {"audio_latents": packed})
    # fp64 CPU reference: the engine computes in fp32, and TensorRT(-RTX) may run fp32
    # convolutions at TF32-class internal precision, which a random-weight vocoder (~120
    # stacked convolutions without normalization) amplifies into a percent-level waveform
    # difference. The mel path (audio VAE) has no such stack and must match tightly; the
    # waveform must stay within that convolution-precision envelope.
    vae = vae.to("cpu", torch.float64)
    voc = voc.to("cpu", torch.float64)
    with torch.no_grad():
        z = packed.double() * vae.latents_std + vae.latents_mean
        z = z.unflatten(2, (-1, lat_m)).transpose(1, 2)
        mel = vae.decode(z, return_dict=False)[0]
        wave = voc(mel)
    got_mel = got["mel"].double().cpu()
    got_wave = got["waveform"].double().cpu()
    cm = cosine(got_mel, mel)
    cw = cosine(got_wave, wave)
    print(f"audio tiny: mel {tuple(got_mel.shape)} cos {cm:.6f} relL2 {rel_l2(got_mel, mel):.2e} | "
          f"wave {tuple(got_wave.shape)} cos {cw:.6f} relL2 {rel_l2(got_wave, wave):.2e}")
    assert tuple(got_mel.shape) == tuple(mel.shape)
    assert tuple(got_wave.shape) == tuple(wave.shape)
    assert torch.isfinite(got_wave).all()
    assert cm > 0.9999
    assert cw > 0.995
