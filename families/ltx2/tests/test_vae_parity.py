# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tiny-random parity: LTX-2.5 video VAE decoder engine vs diffusers ``AutoencoderKLLTX2Video``.

Same block structure as the real LTX-2.5 VAE (four up blocks: spatiotemporal, spatiotemporal,
temporal, spatial; upsample factors 2/1/2/2; non-causal zero-padded decoder; patch 4), narrow
channels, non-trivial latent statistics.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tensorrt")
if not torch.cuda.is_available():
    pytest.skip("CUDA is required for LTX-2.5 engine parity tests", allow_module_level=True)
safetensors_torch = pytest.importorskip("safetensors.torch")

from families.ltx2.tests.engine_runner import cosine, run_plan  # noqa: E402

TINY_VAE = {
    "block_out_channels": [8, 16, 32, 32],
    "decoder_block_out_channels": [16, 32, 32, 64],
    "decoder_causal": False,
    "decoder_inject_noise": [False, False, False, False, False],
    "decoder_layers_per_block": [1, 2, 1, 1, 1],
    "decoder_spatial_padding_mode": "zeros",
    "decoder_spatio_temporal_scaling": [True, True, True, True],
    "down_block_types": ["LTX2VideoDownBlock3D"] * 4,
    "downsample_type": ["spatial", "temporal", "spatiotemporal", "spatiotemporal"],
    "encoder_causal": True,
    "encoder_spatial_padding_mode": "zeros",
    "in_channels": 3,
    "latent_channels": 16,
    "layers_per_block": [1, 1, 1, 1, 1],
    "out_channels": 3,
    "patch_size": 4,
    "patch_size_t": 1,
    "resnet_norm_eps": 1e-6,
    "scaling_factor": 1.0,
    "spatio_temporal_scaling": [True, True, True, True],
    "timestep_conditioning": False,
    "upsample_factor": [2, 2, 1, 2],
    "upsample_residual": [False, False, False, False],
    "upsample_type": ["spatiotemporal", "spatiotemporal", "temporal", "spatial"],
}
F, H, W = 3, 2, 3


def test_vae_decoder_tiny_parity(tmp_path) -> None:
    from diffusers import AutoencoderKLLTX2Video

    from families.ltx2.vae_builder import build_vae_decoder_engine

    vae = AutoencoderKLLTX2Video(**TINY_VAE).eval()
    gen = torch.Generator().manual_seed(9)
    with torch.no_grad():
        for name, p in vae.named_parameters():
            if p.ndim >= 2:
                p.copy_(torch.randn(p.shape, generator=gen) * (1.0 / max(1, p[0].numel())) ** 0.5)
            else:
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=gen) if "norm" in name
                        else 0.05 * torch.randn(p.shape, generator=gen))
        vae.latents_mean.copy_(0.3 * torch.randn(16, generator=gen))
        vae.latents_std.copy_(0.5 + torch.rand(16, generator=gen))
    folder = tmp_path / "vae"
    folder.mkdir()
    state = {k: (v.to(torch.bfloat16) if k not in ("latents_mean", "latents_std") else v).contiguous()
             for k, v in vae.state_dict().items()}
    safetensors_torch.save_file(state, str(folder / "diffusion_pytorch_model.safetensors"))
    (folder / "config.json").write_text(json.dumps(TINY_VAE), encoding="utf-8")

    packed = torch.randn(1, F * H * W, 16, generator=gen)
    plan = build_vae_decoder_engine(folder, latent_frames=F, latent_height=H, latent_width=W)
    got = run_plan(plan, {"latents": packed})["frames"].float().cpu()  # [T, H, W, 3]
    for dtype in (torch.float32, torch.bfloat16):
        ref_vae = vae.to("cuda", dtype)
        z = packed.cuda().reshape(1, F, H, W, 16).permute(0, 4, 1, 2, 3)
        z = z * ref_vae.latents_std.view(1, -1, 1, 1, 1).float() + ref_vae.latents_mean.view(1, -1, 1, 1, 1).float()
        with torch.no_grad():
            video = ref_vae.decode(z.to(dtype), return_dict=False)[0].float()
        ref = (video / 2 + 0.5).clamp(0, 1)[0].permute(1, 2, 3, 0).cpu()
        assert tuple(got.shape) == tuple(ref.shape), (got.shape, ref.shape)
        c = cosine(got - 0.5, ref - 0.5)
        mae = float((got - ref).abs().mean())
        print(f"vae tiny vs {dtype}: cos(centered) {c:.6f} mean|diff| {mae:.2e} out {tuple(got.shape)}")
        assert c > 0.999
