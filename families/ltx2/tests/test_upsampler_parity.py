# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tiny-random parity: LTX-2.5 latent upsampler engine vs diffusers ``LTX2LatentUpsamplerModel``.

The engine takes the packed, normalized stage 1 latents and returns packed, normalized latents on
the 2x grid; the reference denormalizes, runs the diffusers model (``LTX2LatentUpsamplePipeline``
with ``latents_normalized=False`` semantics) and normalizes again, as stage 2's ``prepare_latents``
does. Both upsampler variants are covered (``upsampler.0`` and the 2x rational resampler).
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

C = 16
F, H, W = 3, 2, 3


def _tiny(tmp_path, rational: bool):
    from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel

    config = {"in_channels": C, "mid_channels": 64, "num_blocks_per_stage": 2, "dims": 3,
              "spatial_upsample": True, "temporal_upsample": False, "rational_spatial_scale": 2.0,
              "use_rational_resampler": rational}
    model = LTX2LatentUpsamplerModel(**config).eval()
    gen = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name:
                p.copy_((1.0 if name.endswith("weight") else 0.0) + 0.1 * torch.randn(p.shape, generator=gen))
            elif p.ndim >= 2:
                p.copy_(torch.randn(p.shape, generator=gen) * (1.0 / p[0].numel()) ** 0.5)
            else:
                p.copy_(0.05 * torch.randn(p.shape, generator=gen))
    folder = tmp_path / "latent_upsampler"
    folder.mkdir()
    safetensors_torch.save_file({k: v.contiguous() for k, v in model.state_dict().items()},
                                str(folder / "diffusion_pytorch_model.safetensors"))
    (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")
    vae = tmp_path / "vae"
    vae.mkdir()
    stats = {"latents_mean": 0.3 * torch.randn(C, generator=gen), "latents_std": 0.5 + torch.rand(C, generator=gen)}
    safetensors_torch.save_file(stats, str(vae / "diffusion_pytorch_model.safetensors"))
    (vae / "config.json").write_text(json.dumps({"scaling_factor": 1.0}), encoding="utf-8")
    return model, folder, vae, stats


@pytest.mark.parametrize("rational", [False, True])
def test_latent_upsampler_tiny_parity(tmp_path, rational: bool) -> None:
    from families.ltx2.upsampler_builder import build_latent_upsampler_engine

    model, folder, vae, stats = _tiny(tmp_path, rational)
    plan = build_latent_upsampler_engine(folder, vae, latent_frames=F, latent_height=H, latent_width=W)
    packed = torch.randn(1, F * H * W, C, generator=torch.Generator().manual_seed(11))
    got = run_plan(plan, {"latents": packed})["upsampled"].float().cpu()
    assert tuple(got.shape) == (1, F * 2 * H * 2 * W, C)
    mean, std = stats["latents_mean"].view(1, -1, 1, 1, 1), stats["latents_std"].view(1, -1, 1, 1, 1)
    z = packed.reshape(1, F, H, W, C).permute(0, 4, 1, 2, 3) * std + mean
    for dtype in (torch.float32, torch.bfloat16):
        ref_model = model.to("cuda", dtype)
        with torch.no_grad():
            up = ref_model(z.cuda().to(dtype)).float().cpu()
        ref = ((up - mean) / std).permute(0, 2, 3, 4, 1).reshape(1, -1, C)
        c = cosine(got, ref)
        print(f"upsampler tiny (rational={rational}) vs {dtype}: cos {c:.6f} relL2 {rel_l2(got, ref):.3e}")
        assert c > 0.999
