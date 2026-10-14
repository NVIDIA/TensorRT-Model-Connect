# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tiny-random parity: LTX-2.5 joint audio/video DiT engine vs diffusers ``LTX2VideoTransformer3DModel``.

The shrunk config keeps every LTX-2.5 feature (9-parameter AdaLN, prompt AdaLN, gated attention,
split RoPE with fps-scaled video coordinates, time-only cross-modal RoPE, a2v / v2a, STG on one
block, bias-free video FFN, cross timestep) and a non-square latent grid.
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

TINY_DIT = {
    "activation_fn": "gelu-approximate", "attention_bias": True, "attention_head_dim": 16,
    "attention_out_bias": True, "audio_attention_head_dim": 8, "audio_cross_attention_dim": 32,
    "audio_cross_attn_mod": True, "audio_ff_bias": True, "audio_gated_attn": True, "audio_hop_length": 160,
    "audio_in_channels": 16, "audio_num_attention_heads": 4, "audio_out_channels": 16, "audio_patch_size": 1,
    "audio_patch_size_t": 1, "audio_pos_embed_max_pos": 20, "audio_sampling_rate": 16000,
    "audio_scale_factor": 4, "base_height": 2048, "base_width": 2048, "caption_channels": 48,
    "causal_offset": 1, "cross_attention_dim": 64, "cross_attn_mod": True,
    "cross_attn_timestep_scale_multiplier": 1000, "ff_bias": False, "gated_attn": True, "in_channels": 16,
    "norm_elementwise_affine": False, "norm_eps": 1e-6, "num_attention_heads": 4, "num_layers": 3,
    "out_channels": 16, "patch_size": 1, "patch_size_t": 1, "perturbed_attn": True, "pos_embed_max_pos": 20,
    "qk_norm": "rms_norm_across_heads", "rope_double_precision": True, "rope_theta": 10000.0,
    "rope_type": "split", "timestep_scale_multiplier": 1000, "use_keyframes_abs_pos_embedding": True,
    "use_prompt_adaln_single": True, "use_prompt_embeddings": False, "vae_scale_factors": [8, 32, 32],
}
FRAMES, LH, LW = 3, 4, 6  # latent grid -> 17 pixel frames
FPS = 24.0
TEXT = 16
STG_BLOCK = 1


def _randomize(module, seed: int) -> None:
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in module.named_parameters():
            if "norm" in name and name.endswith("weight"):
                p.copy_(1.0 + 0.2 * torch.randn(p.shape, generator=gen))
            elif "scale_shift_table" in name:
                p.copy_(0.3 * torch.randn(p.shape, generator=gen))
            elif p.ndim >= 2:
                p.copy_(torch.randn(p.shape, generator=gen) / p.shape[-1] ** 0.5)
            else:
                p.copy_(0.1 * torch.randn(p.shape, generator=gen))


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from diffusers import LTX2VideoTransformer3DModel

    from families.ltx2.dit_builder import DiTShape, audio_latent_frames, build_dit_engine

    model = LTX2VideoTransformer3DModel(**TINY_DIT).eval()
    _randomize(model, 5)
    folder = tmp_path_factory.mktemp("tiny_dit") / "transformer"
    folder.mkdir()
    safetensors_torch.save_file({k: v.to(torch.bfloat16).contiguous() for k, v in model.state_dict().items()},
                                str(folder / "diffusion_pytorch_model.safetensors"))
    (folder / "config.json").write_text(json.dumps(TINY_DIT), encoding="utf-8")
    num_frames = (FRAMES - 1) * 8 + 1
    sa = audio_latent_frames(num_frames, FPS)
    shape = DiTShape(batch=2, latent_frames=FRAMES, latent_height=LH, latent_width=LW, audio_frames=sa,
                     text_len=TEXT, fps=FPS)
    plan = build_dit_engine(folder, shape, stg_blocks=(STG_BLOCK,))
    return model, plan, shape, folder


def _inputs(shape, seed=0):
    gen = torch.Generator().manual_seed(seed)
    b, s = shape.batch, shape.video_tokens
    return {
        "video_latent": torch.randn(b, s, 16, generator=gen),
        "audio_latent": torch.randn(b, shape.audio_frames, 16, generator=gen),
        "video_context": torch.randn(b, TEXT, 64, generator=gen).to(torch.bfloat16),
        "audio_context": torch.randn(b, TEXT, 32, generator=gen).to(torch.bfloat16),
        "timestep": torch.full((b,), 909.375),
    }


def _reference(model, shape, inp, dtype, *, stg_mask=None, isolate=False):
    model = model.to("cuda", dtype)
    b = shape.batch
    with torch.no_grad():
        v, a = model(
            hidden_states=inp["video_latent"].cuda().to(dtype),
            audio_hidden_states=inp["audio_latent"].cuda().to(dtype),
            encoder_hidden_states=inp["video_context"].cuda().to(dtype),
            audio_encoder_hidden_states=inp["audio_context"].cuda().to(dtype),
            timestep=inp["timestep"].cuda(), sigma=inp["timestep"].cuda(),
            encoder_attention_mask=torch.ones(b, TEXT, device="cuda"),
            audio_encoder_attention_mask=torch.ones(b, TEXT, device="cuda"),
            num_frames=shape.latent_frames, height=shape.latent_height, width=shape.latent_width, fps=FPS,
            audio_num_frames=shape.audio_frames, use_cross_timestep=True, isolate_modalities=isolate,
            spatio_temporal_guidance_blocks=[STG_BLOCK] if stg_mask is not None else None,
            perturbation_mask=None if stg_mask is None else stg_mask.cuda().to(dtype), return_dict=False)
    return v.float().cpu(), a.float().cpu()


@pytest.mark.parametrize("case", ["plain", "stg_mixed", "isolated"])
def test_dit_tiny_parity(tiny, case) -> None:
    model, plan, shape, _ = tiny
    inp = _inputs(shape)
    stg = torch.ones(shape.batch)
    av = torch.ones(shape.batch)
    stg_ref = None
    if case == "stg_mixed":
        stg = torch.tensor([1.0, 0.0])
        stg_ref = stg
    if case == "isolated":
        av = torch.zeros(shape.batch)
    got = run_plan(plan, {**inp, "stg_keep": stg, "av_keep": av})
    for dtype, floor in ((torch.float32, 0.999), (torch.bfloat16, 0.999)):
        rv, ra = _reference(model, shape, inp, dtype, stg_mask=stg_ref, isolate=(case == "isolated"))
        cv = cosine(got["video_velocity"].cpu(), rv)
        ca = cosine(got["audio_velocity"].cpu(), ra)
        print(f"dit tiny {case} vs {dtype}: video cos {cv:.6f} relL2 {rel_l2(got['video_velocity'].cpu(), rv):.3e}"
              f" | audio cos {ca:.6f} relL2 {rel_l2(got['audio_velocity'].cpu(), ra):.3e}")
        assert cv > floor
        assert ca > floor


def test_rope_grids_match_diffusers() -> None:
    from diffusers.models.transformers.transformer_ltx2 import LTX2AudioVideoRotaryPosEmbed

    from families.ltx2.dit_builder import DiTConfig, DiTShape, audio_latent_frames, rope_tables

    cfg = DiTConfig.from_dict(dict(TINY_DIT, num_layers=1))
    sa = audio_latent_frames((FRAMES - 1) * 8 + 1, FPS)
    shape = DiTShape(1, FRAMES, LH, LW, sa, TEXT, FPS)
    ours = rope_tables(cfg, shape)
    kw = dict(theta=10000.0, causal_offset=1, double_precision=True, rope_type="split")
    video = LTX2AudioVideoRotaryPosEmbed(dim=64, base_num_frames=20, scale_factors=(8, 32, 32), modality="video",
                                         num_attention_heads=4, **kw)
    audio = LTX2AudioVideoRotaryPosEmbed(dim=32, base_num_frames=20, scale_factors=[4], modality="audio",
                                         num_attention_heads=4, **kw)
    ca_v = LTX2AudioVideoRotaryPosEmbed(dim=32, base_num_frames=20, scale_factors=(8, 32, 32), modality="video",
                                        num_attention_heads=4, **kw)
    ca_a = LTX2AudioVideoRotaryPosEmbed(dim=32, base_num_frames=20, scale_factors=(8, 32, 32), modality="audio",
                                        num_attention_heads=4, **kw)
    vc = video.prepare_video_coords(1, FRAMES, LH, LW, "cpu", fps=FPS)
    ac = audio.prepare_audio_coords(1, sa, "cpu")
    refs = {"video": video(vc), "audio": audio(ac), "ca_video": ca_v(vc[:, 0:1]), "ca_audio": ca_a(ac[:, 0:1])}
    for key, (cos_ref, sin_ref) in refs.items():
        cos, sin = ours[key]
        # diffusers: [B, H, T, r]; ours: [T, H, r]
        assert torch.allclose(torch.from_numpy(cos).permute(1, 0, 2), cos_ref[0], atol=2e-6), key
        assert torch.allclose(torch.from_numpy(sin).permute(1, 0, 2), sin_ref[0], atol=2e-6), key


def half_grid(shape):
    """The two-stage stage 1 grid: half the latent rows and columns."""
    from dataclasses import replace

    return replace(shape, latent_height=shape.latent_height // 2, latent_width=shape.latent_width // 2)


def test_dit_two_grid_plan_tiny_parity(tiny) -> None:
    """One plan serving the full grid and the half-resolution stage 1 grid (run-time token count)."""
    from families.ltx2.dit_builder import build_dit_engine
    from families.ltx2.tests.engine_runner import Engine

    model, static_plan, shape, folder = tiny
    small = half_grid(shape)
    assert small.video_tokens != shape.video_tokens
    engine = Engine(build_dit_engine(folder, shape, stg_blocks=(STG_BLOCK,), extra_shapes=(small,)))
    keep = {"stg_keep": torch.ones(shape.batch), "av_keep": torch.ones(shape.batch)}
    static = run_plan(static_plan, {**_inputs(shape), **keep})
    for grid in (shape, small, shape):  # switch grids back and forth on one context
        inp = _inputs(grid, seed=grid.video_tokens)
        got = engine({**inp, **keep})
        assert tuple(got["video_velocity"].shape) == (grid.batch, grid.video_tokens, 16)
        rv, ra = _reference(model, grid, inp, torch.float32)
        cv = cosine(got["video_velocity"].cpu(), rv)
        ca = cosine(got["audio_velocity"].cpu(), ra)
        print(f"two-grid plan at {grid.video_tokens} tokens vs fp32: video cos {cv:.6f} | audio cos {ca:.6f}")
        assert cv > 0.999 and ca > 0.999
    got = engine({**_inputs(shape), **keep})
    c = cosine(got["video_velocity"].cpu(), static["video_velocity"].cpu())
    print(f"two-grid plan vs the static plan on the full grid: video cos {c:.6f}")
    assert c > 0.9999
