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


TILE_GRID = (7, 6, 7)  # latent frames, rows, columns: 2 x 1 x 2 tiles of 5 x 6 x 5 latents
TILE_CONFIG = dict(tile_pixels=192, overlap_pixels=64, tile_frames=40, overlap_frames=8)


def tiny_vae(folder):
    """Tiny random VAE saved under ``folder``; returns it and its (advanced) generator."""
    from diffusers import AutoencoderKLLTX2Video

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
    folder.mkdir(parents=True, exist_ok=True)
    state = {k: (v.to(torch.bfloat16) if k not in ("latents_mean", "latents_std") else v).contiguous()
             for k, v in vae.state_dict().items()}
    safetensors_torch.save_file(state, str(folder / "diffusion_pytorch_model.safetensors"))
    (folder / "config.json").write_text(json.dumps(TINY_VAE), encoding="utf-8")
    return vae, gen


def diffusers_tile(vae, packed, grid, plan, tile, dtype=torch.float32):
    """diffusers decode of one tile's latents, ``(x + 1) / 2`` unclamped: ``[T, H, W, 3]``."""
    f, h, w = grid
    tf, th, tw = plan["tile_latent"]
    f0, h0, w0 = tile["latent_start"]
    z = packed.reshape(1, f, h, w, 16)[:, f0:f0 + tf, h0:h0 + th, w0:w0 + tw].permute(0, 4, 1, 2, 3).cuda()
    z = z * vae.latents_std.view(1, -1, 1, 1, 1).float() + vae.latents_mean.view(1, -1, 1, 1, 1).float()
    with torch.no_grad():
        video = vae.decode(z.to(dtype), return_dict=False)[0].float()
    return (video / 2 + 0.5)[0].permute(1, 2, 3, 0).cpu()


def tile_latents(packed, grid, plan, tile):
    """Packed ``[1, tf*th*tw, C]`` latents of one tile."""
    f, h, w = grid
    tf, th, tw = plan["tile_latent"]
    f0, h0, w0 = tile["latent_start"]
    part = packed.reshape(1, f, h, w, -1)[:, f0:f0 + tf, h0:h0 + th, w0:w0 + tw]
    return part.reshape(1, tf * th * tw, -1)


def test_vae_tile_engine_tiny_parity(tmp_path) -> None:
    """Every tile of a tile plan vs diffusers on the same latents, and the blended video."""
    import numpy as np

    from families.ltx2.tests.engine_runner import Engine
    from families.ltx2.vae_builder import build_vae_decoder_engine
    from families.ltx2.vae_tiling import TileConfig, blend_tiles, plan_tiles

    vae, _ = tiny_vae(tmp_path / "vae")
    ref_vae = vae.to("cuda", torch.float32)
    f, h, w = TILE_GRID
    plan = plan_tiles(f, h, w, TileConfig(**TILE_CONFIG))
    tf, th, tw = plan["tile_latent"]
    assert len(plan["tiles"]) == 4 and plan["tile_latent"] == [5, 6, 5]
    engine = Engine(build_vae_decoder_engine(tmp_path / "vae", latent_frames=tf, latent_height=th, latent_width=tw,
                                             clamp_output=False))
    packed = torch.randn(1, f * h * w, 16, generator=torch.Generator().manual_seed(4))
    got, ref = [], []
    for tile in plan["tiles"]:
        out = engine({"latents": tile_latents(packed, TILE_GRID, plan, tile)})["frames"].float().cpu()
        expected = diffusers_tile(ref_vae, packed, TILE_GRID, plan, tile)
        c = cosine(out - 0.5, expected - 0.5)
        print(f"tile {tile['latent_start']}: cos(centered) {c:.6f}")
        assert c > 0.999
        got.append(out.numpy().astype(np.float16))
        ref.append(expected.numpy())
    frames, height, width = (f - 1) * 8 + 1, h * 32, w * 32
    blended = blend_tiles(plan, got, frames, height, width)
    expected = blend_tiles(plan, ref, frames, height, width)
    c = cosine(torch.from_numpy(blended) - 0.5, torch.from_numpy(expected) - 0.5)
    print(f"blended tiles vs blended diffusers tiles: cos(centered) {c:.6f}")
    assert c > 0.999


def test_vae_decoder_tiny_parity(tmp_path) -> None:
    from families.ltx2.vae_builder import build_vae_decoder_engine

    vae, gen = tiny_vae(tmp_path / "vae")
    folder = tmp_path / "vae"

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
