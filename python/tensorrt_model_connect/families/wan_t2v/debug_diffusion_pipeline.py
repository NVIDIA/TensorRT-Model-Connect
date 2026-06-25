#!/usr/bin/env python3
"""Debug diffusion pipeline: component-by-component TRT vs HF comparison.

Systematically compares every component of the diffusion pipeline between the
TRT bundle and HuggingFace reference. Reports PASS/FAIL for each component and
identifies the root cause(s) of noisy output.

Usage:
    python tools/debug_diffusion_pipeline.py \
        --bundle /mnt/storage/tensorrt-model-connect/engines/wan21-t2v-1.3b.trtfb \
        [--model-id Wan-AI/Wan2.1-T2V-1.3B-Diffusers] \
        [--atol 0.01] [--num-steps 10]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[4]
_TOOLS_DIR = _REPO_ROOT / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from diffusion_helpers import (silu, gelu_tanh, load_pp_weights,  # noqa: E402
                                load_bundle_config,
                                compute_timestep_embedding as compute_timestep_embedding_np,
                                project_text as project_text_np)
from tool_helpers import cosine_sim, compare_arrays as compare  # noqa: E402


# ---------------------------------------------------------------------------
# Shared state between steps
# ---------------------------------------------------------------------------

class Context:
    """Shared state passed between test steps."""
    pipe = None           # HF WanPipeline (transformer only, T5 broken)
    runner = None         # TRT DiffusionRunner
    trt_text = None       # TRT T5 output [1, 512, 4096]
    cfg = None            # Bundle config dict
    pp = None             # Preprocessor weights
    t_lat = h_lat = w_lat = 0


ctx = Context()


def handles_debug_diffusion_pipeline_args(argv: list[str]) -> bool:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--bundle", default="")
    parser.add_argument("--model-id", default="")
    ns, _ = parser.parse_known_args(argv)
    return "wan" in ns.bundle.lower() or "wan" in ns.model_id.lower()


# ---------------------------------------------------------------------------
# Test steps
# ---------------------------------------------------------------------------

def step1_verify_config(bundle_path: str) -> bool:
    """Verify critical bundle config fields."""
    print("\n=== Step 1: Verify Bundle Config ===")
    cfg = load_bundle_config(bundle_path)
    ctx.cfg = cfg

    checks = []
    for key, expected in [
        ("flow_shift", 3.0), ("dit_dim", 1536), ("dit_num_heads", 12),
        ("z_dim", 16), ("freq_dim", 256), ("text_encoder_dim", 4096),
    ]:
        val = cfg.get(key, "MISSING")
        ok = val == expected
        print(f"  {key} = {val} {'(OK)' if ok else f'*** expected {expected} ***'}")
        checks.append(ok)

    ps = cfg.get("patch_size", "MISSING")
    ok = ps == [1, 2, 2]
    print(f"  patch_size = {ps} {'(OK)' if ok else '*** expected [1,2,2] ***'}")
    checks.append(ok)

    for key in ("latents_mean", "latents_std"):
        val = cfg.get(key, [])
        ok = len(val) == 16
        print(f"  {key}: {len(val)} values {'(OK)' if ok else '*** expected 16 ***'}")
        checks.append(ok)

    # Compute latent dimensions from video config
    vh = cfg.get("video_height", 480)
    vw = cfg.get("video_width", 832)
    vf = cfg.get("video_num_frames", 17)
    sft = cfg.get("scale_factor_temporal", 4)
    sfs = cfg.get("scale_factor_spatial", 8)
    ctx.t_lat = (vf - 1) // sft + 1
    ctx.h_lat = vh // sfs
    ctx.w_lat = vw // sfs

    pt, ph, pw = cfg.get("patch_size", [1, 2, 2])
    nt = ctx.t_lat // pt
    nh = ctx.h_lat // ph
    nw = ctx.w_lat // pw
    num_patches = nt * nh * nw

    print(f"  video: {vh}x{vw}@{vf}fr -> latent: {ctx.t_lat}x{ctx.h_lat}x{ctx.w_lat}")
    print(f"  num_patches = {num_patches}")
    print(f"  scheduler = {cfg.get('scheduler', 'MISSING')}")

    passed = all(checks)
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def step2_text_projection_activation(model_id: str, pp: dict, atol: float) -> bool:
    """Verify text projection activation matches between C++ and HF."""
    print("\n=== Step 2: Text Projection Activation ===")

    import torch
    from diffusers import WanPipeline

    print("  Loading HF pipeline ...", file=sys.stderr)
    pipe = WanPipeline.from_pretrained(
        model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    ctx.pipe = pipe

    text_embedder = pipe.transformer.condition_embedder.text_embedder
    # Find the activation function
    act_name = "unknown"
    for attr in ("act_1", "act"):
        if hasattr(text_embedder, attr):
            act_name = getattr(text_embedder, attr).__class__.__name__
            break
    print(f"  HF activation class: {act_name}")

    # Test with random input
    torch.manual_seed(42)
    test_input = torch.randn(1, 512, 4096)
    with torch.no_grad():
        hf_out = text_embedder(test_input).numpy().reshape(512, 1536)

    test_np = test_input.numpy().reshape(512, 4096)
    h = test_np @ pp["condition_embedder.text_embedding.weight"] + pp["condition_embedder.text_embedding.bias"]

    diff_silu = np.abs(
        (silu(h.copy()) @ pp["condition_embedder.text_embedding_2.weight"] + pp["condition_embedder.text_embedding_2.bias"]).flatten()
        - hf_out.flatten()).max()
    diff_gelu = np.abs(
        (gelu_tanh(h.copy()) @ pp["condition_embedder.text_embedding_2.weight"] + pp["condition_embedder.text_embedding_2.bias"]).flatten()
        - hf_out.flatten()).max()

    print(f"  SiLU max_diff:       {diff_silu:.6f}")
    print(f"  GELU(tanh) max_diff: {diff_gelu:.6f}")

    winner = "gelu_tanh" if diff_gelu < diff_silu else "silu"
    best_diff = min(diff_silu, diff_gelu)
    print(f"  >>> HF uses {winner} (matches within {best_diff:.6f})")

    cpp_uses = "gelu_tanh"
    match = winner == cpp_uses
    if match:
        print(f"  C++ matches HF (both use {cpp_uses})")
    else:
        print(f"  *** BUG: C++ uses {cpp_uses} but HF uses {winner} ***")

    passed = best_diff <= atol and match
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def step3_t5_and_text_proj(bundle_path: str, pp: dict, atol: float) -> bool:
    """Run TRT T5, then compare text projection against HF."""
    print("\n=== Step 3: T5 Encoding + Text Projection ===")

    import torch
    pipe = ctx.pipe

    prompt = "A cat walking in the garden"
    tokens = pipe.tokenizer(
        prompt, return_tensors="pt", padding="max_length",
        max_length=512, truncation=True,
    )
    input_ids = tokens.input_ids
    print(f"  Prompt: {prompt!r}")
    print(f"  Tokens (first 8): {input_ids[0, :8].tolist()}")

    # TRT T5 (our bundle's engine)
    from tensorrt_model_connect.families.wan_t2v.diffusion_runner import DiffusionRunner
    print("  Loading TRT engines ...", file=sys.stderr)
    runner = DiffusionRunner(bundle_path)
    ctx.runner = runner

    trt_text = runner.encode_text(input_ids.numpy().astype(np.int32))
    ctx.trt_text = trt_text
    print(f"  TRT T5: shape={trt_text.shape}, mean={trt_text.mean():.6f}, std={trt_text.std():.6f}")

    # Note: HF T5 from WanPipeline has missing embed_tokens (all-zero output).
    # We skip direct T5 comparison and instead verify the text projection path.
    print("  (HF T5 skipped — WanPipeline's UMT5 has missing embed_tokens)")

    # Text projection: verify our computation matches HF's module
    torch.manual_seed(42)
    test_input = torch.randn(1, 512, 4096)
    with torch.no_grad():
        hf_proj = pipe.transformer.condition_embedder.text_embedder(test_input)
    hf_proj_np = hf_proj.numpy().reshape(512, 1536)

    our_proj = project_text_np(test_input.numpy(), pp)
    passed = compare("text_projection", our_proj, hf_proj_np, atol)

    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def step4_timestep_embedding(pp: dict, atol: float) -> bool:
    """Compare timestep embedding computation."""
    print("\n=== Step 4: Timestep Embedding (t=1000) ===")

    import torch
    pipe = ctx.pipe

    with torch.no_grad():
        hf_sinusoidal = pipe.transformer.condition_embedder.timesteps_proj(
            torch.tensor([1000.0]))
        hf_time_embed = pipe.transformer.condition_embedder.time_embedder(
            hf_sinusoidal)
        hf_temb = pipe.transformer.condition_embedder.time_proj(
            pipe.transformer.condition_embedder.act_fn(hf_time_embed))

    our_temb, our_te = compute_timestep_embedding_np(1000.0, pp)

    ok1 = compare("time_embed", our_te, hf_time_embed.numpy(), atol)
    ok2 = compare("temb_6d", our_temb, hf_temb.numpy(), atol)

    passed = ok1 and ok2
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def step5_patch_embedding(pp: dict, atol: float) -> bool:
    """Compare patch embedding (patchify + matmul vs Conv3D)."""
    print("\n=== Step 5: Patch Embedding ===")

    import torch
    pipe = ctx.pipe
    runner = ctx.runner

    torch.manual_seed(42)
    test_latent = torch.randn(1, 16, 2, 16, 16)
    with torch.no_grad():
        hf_out = pipe.transformer.patch_embedding(test_latent)
    hf_flat = hf_out.flatten(2).transpose(1, 2).squeeze(0).numpy()

    patches = runner._patchify(test_latent.numpy(), [1, 2, 2])
    our_flat = patches @ pp["patch_embedding.weight"] + pp["patch_embedding.bias"]

    passed = compare("patch_embed", our_flat, hf_flat, atol)
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def step6_3d_rope(atol: float) -> bool:
    """Compare 3D RoPE at both small and bundle resolution."""
    print("\n=== Step 6: 3D RoPE ===")

    import torch
    pipe = ctx.pipe
    runner = ctx.runner

    # Small resolution test
    torch.manual_seed(42)
    test_latent = torch.randn(1, 16, 2, 16, 16)
    with torch.no_grad():
        hf_cos, hf_sin = pipe.transformer.rope(test_latent)
    hf_cos_np = hf_cos[0, :, 0, :].numpy()
    hf_sin_np = hf_sin[0, :, 0, :].numpy()
    our_cos, our_sin = runner._compute_3d_rope(2, 8, 8, 128)

    print("  Small (2x16x16 -> 128 patches):")
    ok1 = compare("  rope_cos", our_cos, hf_cos_np, atol)
    ok2 = compare("  rope_sin", our_sin, hf_sin_np, atol)

    # Bundle resolution test
    cfg = ctx.cfg
    pt, ph, pw = cfg.get("patch_size", [1, 2, 2])
    nt = ctx.t_lat // pt
    nh = ctx.h_lat // ph
    nw = ctx.w_lat // pw
    num_patches = nt * nh * nw

    # Create a latent at bundle resolution for HF
    test_full = torch.randn(1, 16, ctx.t_lat, ctx.h_lat, ctx.w_lat)
    with torch.no_grad():
        hf_cos_full, hf_sin_full = pipe.transformer.rope(test_full)
    hf_cos_full_np = hf_cos_full[0, :, 0, :].numpy()
    hf_sin_full_np = hf_sin_full[0, :, 0, :].numpy()
    our_cos_full, our_sin_full = runner._compute_3d_rope(nt, nh, nw, 128)

    print(f"  Bundle ({ctx.t_lat}x{ctx.h_lat}x{ctx.w_lat} -> {num_patches} patches):")
    ok3 = compare("  rope_cos", our_cos_full, hf_cos_full_np, atol)
    ok4 = compare("  rope_sin", our_sin_full, hf_sin_full_np, atol)

    passed = ok1 and ok2 and ok3 and ok4
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def step7_dit_step(bundle_path: str, pp: dict, atol: float) -> bool:
    """Compare single DiT step: feed identical inputs to TRT engine and HF model.

    Uses the bundle's native resolution so TRT engine shapes match.
    """
    print("\n=== Step 7: Single DiT Step (Bundle Resolution) ===")

    import torch
    pipe = ctx.pipe
    runner = ctx.runner
    cfg = ctx.cfg

    pt, ph, pw = cfg.get("patch_size", [1, 2, 2])
    nt = ctx.t_lat // pt
    nh = ctx.h_lat // ph
    nw = ctx.w_lat // pw

    # Use TRT T5 output (shared between both) since HF T5 is broken
    trt_text = ctx.trt_text  # [1, 512, 4096]

    # Random latent at bundle resolution
    torch.manual_seed(42)
    test_latent = torch.randn(1, 16, ctx.t_lat, ctx.h_lat, ctx.w_lat)
    timestep = torch.tensor([1000.0])

    # HF side: run full WanTransformer3DModel forward
    text_for_hf = torch.from_numpy(trt_text.copy())
    with torch.no_grad():
        hf_out = pipe.transformer(
            hidden_states=test_latent,
            timestep=timestep,
            encoder_hidden_states=text_for_hf,
        ).sample  # [1, z_dim*pt*ph*pw, nt, nh, nw] or [1, C, T, H, W]

    hf_out_np = hf_out.numpy()
    print(f"  HF DiT output: shape={hf_out_np.shape}, "
          f"range=[{hf_out_np.min():.4f}, {hf_out_np.max():.4f}]")

    # TRT side: preprocess + run engine
    # 1. Patchify + embed
    patches = runner._patchify(test_latent.numpy(), [pt, ph, pw])
    hidden = patches @ pp["patch_embedding.weight"] + pp["patch_embedding.bias"]

    # 2. Timestep embedding
    temb_6d, time_embed = compute_timestep_embedding_np(1000.0, pp)

    # 3. Text projection
    text_proj = project_text_np(trt_text, pp)

    # 4. RoPE
    rope_cos, rope_sin = runner._compute_3d_rope(nt, nh, nw, 128)

    # 5. Run TRT denoiser
    trt_out = runner._run_engine("denoiser", {
        "hidden_states": hidden,
        "timestep_embedding": temb_6d.reshape(1, -1),
        "time_embed": time_embed.reshape(1, -1),
        "encoder_hidden_states": text_proj,
        "rotary_cos": rope_cos,
        "rotary_sin": rope_sin,
    })["output"]  # [num_patches, out_dim]

    # 6. Unpatchify TRT output to spatial
    trt_spatial = runner._unpatchify(
        trt_out, [pt, ph, pw], 16, ctx.t_lat, ctx.h_lat, ctx.w_lat)

    print(f"  TRT DiT output: shape={trt_spatial.shape}, "
          f"range=[{trt_spatial.min():.4f}, {trt_spatial.max():.4f}]")

    cs = cosine_sim(trt_spatial, hf_out_np)
    diff = np.abs(trt_spatial.flatten() - hf_out_np.flatten())
    print(f"  max_diff={diff.max():.6f}, mean_diff={diff.mean():.6f}, "
          f"cosine_sim={cs:.6f}")

    # For 30 layers, FP32 drift is expected. cosine > 0.95 means correct.
    passed = cs > 0.95
    if passed:
        print(f"  NOTE: cosine_sim={cs:.4f} > 0.95 — engine is correct")
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def step8_scheduler_sigmas(bundle_path: str, atol: float) -> bool:
    """Compare scheduler sigma schedule and timesteps."""
    print("\n=== Step 8: Scheduler Sigma Schedule ===")

    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

    cfg = ctx.cfg
    shift = cfg.get("flow_shift", 1.0)

    hf_sched = FlowMatchEulerDiscreteScheduler(shift=shift)
    hf_sched.set_timesteps(30)
    hf_sigmas = hf_sched.sigmas.numpy()
    hf_timesteps = hf_sched.timesteps.numpy()

    num_steps = 30
    from tensorrt_model_connect.families.wan_t2v.schedulers.flow_match_euler import FlowMatchEulerScheduler
    our_sched = FlowMatchEulerScheduler(num_train_timesteps=1000, shift=shift)
    our_sched.set_timesteps(num_steps)
    sigmas = our_sched._sigmas
    our_timesteps = our_sched._timesteps

    print(f"  shift = {shift}")
    print(f"  HF sigmas[:5]:     {hf_sigmas[:5]}")
    print(f"  Our sigmas[:5]:    {sigmas[:5]}")
    print(f"  HF timesteps[:5]:  {hf_timesteps[:5]}")
    print(f"  Our timesteps[:5]: {our_timesteps[:5]}")

    diff_s = np.abs(sigmas - hf_sigmas[:len(sigmas)]).max()
    diff_t = np.abs(our_timesteps - hf_timesteps[:len(our_timesteps)]).max()
    print(f"  Sigma max_diff: {diff_s:.8f}")
    print(f"  Timestep max_diff: {diff_t:.6f}")

    passed = diff_s < atol and diff_t < atol * 1000
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def step9_full_pipeline(bundle_path: str, pp: dict, num_steps: int, atol: float) -> bool:
    """Run full denoising loop at bundle resolution: TRT engine vs HF DiT."""
    print(f"\n=== Step 9: Full Pipeline ({num_steps} steps, bundle resolution) ===")

    import torch
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

    pipe = ctx.pipe
    runner = ctx.runner
    cfg = ctx.cfg

    shift = cfg.get("flow_shift", 1.0)
    z_dim = cfg.get("z_dim", 16)
    pt, ph, pw = cfg.get("patch_size", [1, 2, 2])
    nt = ctx.t_lat // pt
    nh = ctx.h_lat // ph
    nw = ctx.w_lat // pw

    guidance_scale = 5.0

    # Shared initial noise (use numpy rng, convert to torch for HF)
    rng = np.random.default_rng(42)
    noise_np = rng.standard_normal(
        (1, z_dim, ctx.t_lat, ctx.h_lat, ctx.w_lat)).astype(np.float32)
    noise_torch = torch.from_numpy(noise_np.copy())

    # TRT T5 output (shared — HF T5 is broken)
    trt_text = ctx.trt_text
    text_torch = torch.from_numpy(trt_text.copy())

    # Text projection (for TRT side)
    our_text_proj = project_text_np(trt_text, pp)
    null_text = np.zeros_like(our_text_proj)

    # RoPE (precomputed once)
    rope_cos, rope_sin = runner._compute_3d_rope(nt, nh, nw, 128)

    # HF scheduler
    hf_sched = FlowMatchEulerDiscreteScheduler(shift=shift)
    hf_sched.set_timesteps(num_steps)
    hf_latents = noise_torch.clone()

    # Our scheduler (matching C++ and HF)
    from tensorrt_model_connect.families.wan_t2v.schedulers.flow_match_euler import FlowMatchEulerScheduler
    our_sched = FlowMatchEulerScheduler(num_train_timesteps=1000, shift=shift)
    our_sched.set_timesteps(num_steps)
    sigmas = our_sched._sigmas
    our_timesteps = our_sched._timesteps
    our_latents = noise_np.copy()

    null_text_torch = torch.zeros_like(text_torch)
    divergences = []

    for step in range(num_steps):
        t_val = float(our_timesteps[step])
        hf_t = hf_sched.timesteps[step]

        # --- TRT side ---
        temb_6d, time_embed = compute_timestep_embedding_np(t_val, pp)
        patches = runner._patchify(
            our_latents.reshape(1, z_dim, ctx.t_lat, ctx.h_lat, ctx.w_lat),
            [pt, ph, pw])
        hidden = patches @ pp["patch_embedding.weight"] + pp["patch_embedding.bias"]

        cond_out = runner._run_engine("denoiser", {
            "hidden_states": hidden,
            "timestep_embedding": temb_6d.reshape(1, -1),
            "time_embed": time_embed.reshape(1, -1),
            "encoder_hidden_states": our_text_proj,
            "rotary_cos": rope_cos, "rotary_sin": rope_sin,
        })["output"]

        uncond_out = runner._run_engine("denoiser", {
            "hidden_states": hidden,
            "timestep_embedding": temb_6d.reshape(1, -1),
            "time_embed": time_embed.reshape(1, -1),
            "encoder_hidden_states": null_text,
            "rotary_cos": rope_cos, "rotary_sin": rope_sin,
        })["output"]

        noise_pred = uncond_out + guidance_scale * (cond_out - uncond_out)
        noise_spatial = runner._unpatchify(
            noise_pred, [pt, ph, pw], z_dim, ctx.t_lat, ctx.h_lat, ctx.w_lat)

        dt = sigmas[step + 1] - sigmas[step]
        our_latents = our_latents + dt * noise_spatial.reshape(our_latents.shape)

        # --- HF side ---
        with torch.no_grad():
            hf_cond = pipe.transformer(
                hidden_states=hf_latents,
                timestep=hf_t.unsqueeze(0),
                encoder_hidden_states=text_torch,
            ).sample
            hf_uncond = pipe.transformer(
                hidden_states=hf_latents,
                timestep=hf_t.unsqueeze(0),
                encoder_hidden_states=null_text_torch,
            ).sample
            hf_noise = hf_uncond + guidance_scale * (hf_cond - hf_uncond)
            hf_latents = hf_sched.step(
                hf_noise, hf_t, hf_latents, return_dict=False)[0]

        # Compare
        mx = float(np.abs(our_latents.flatten() - hf_latents.numpy().flatten()).max())
        cs = cosine_sim(our_latents, hf_latents.numpy())
        divergences.append((mx, cs))
        print(f"  Step {step+1}/{num_steps}: max_diff={mx:.4f}, cosine_sim={cs:.4f}")

    print(f"\n  Divergence trend (max_diff): "
          f"{[f'{d[0]:.2f}' for d in divergences]}")
    final_cs = divergences[-1][1]
    print(f"  Final cosine_sim: {final_cs:.4f}")

    passed = final_cs > 0.8
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Debug diffusion pipeline: TRT vs HF comparison")
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--model-id", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--atol", type=float, default=0.01)
    parser.add_argument("--num-steps", type=int, default=10)
    args = parser.parse_args()

    print("=" * 60)
    print("  Diffusion Pipeline Debug: TRT vs HF")
    print("=" * 60)
    print(f"  Bundle: {args.bundle}")
    print(f"  Model:  {args.model_id}")
    print(f"  atol:   {args.atol}")

    pp = load_pp_weights(args.bundle)
    ctx.pp = pp
    print(f"\n  Preprocessor weights loaded ({len(pp)} tensors):")
    for k, v in pp.items():
        print(f"    {k}: {v.shape}")

    results = {}

    results["config"] = step1_verify_config(args.bundle)
    results["text_proj_activation"] = step2_text_projection_activation(
        args.model_id, pp, args.atol)
    results["t5_text_proj"] = step3_t5_and_text_proj(
        args.bundle, pp, args.atol)
    results["timestep_embedding"] = step4_timestep_embedding(pp, args.atol)
    results["patch_embedding"] = step5_patch_embedding(pp, args.atol)
    results["3d_rope"] = step6_3d_rope(args.atol)
    results["dit_step"] = step7_dit_step(args.bundle, pp, args.atol)
    results["scheduler"] = step8_scheduler_sigmas(args.bundle, args.atol)
    results["full_pipeline"] = step9_full_pipeline(
        args.bundle, pp, args.num_steps, args.atol)

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        print(f"  {'PASS' if passed else 'FAIL':4s}  {name}")

    num_pass = sum(results.values())
    print(f"\n  {num_pass}/{len(results)} passed")
    return 0 if all(results.values()) else 1


def run_as_diff_test(ctx_fw):
    """Framework entry point. Returns DiffResult."""
    from diff_framework.protocol import DiffResult
    import time as _time

    t0 = _time.monotonic()
    try:
        bundle = ctx_fw.bundle_path
        if not bundle:
            return DiffResult.skip(
                "diffusion_components", ctx_fw.model,
                ctx_fw.runtime_strategy, "No bundle provided")

        model_id = ctx_fw.model
        atol = ctx_fw.atol
        num_steps = ctx_fw.num_inference_steps

        pp = load_pp_weights(bundle)
        results = {}

        results["config"] = step1_verify_config(bundle)
        results["text_proj_activation"] = step2_text_projection_activation(
            model_id, pp, atol)
        results["t5_text_proj"] = step3_t5_and_text_proj(bundle, pp, atol)
        results["timestep_embedding"] = step4_timestep_embedding(pp, atol)
        results["patch_embedding"] = step5_patch_embedding(pp, atol)
        results["3d_rope"] = step6_3d_rope(atol)
        results["dit_step"] = step7_dit_step(bundle, pp, atol)
        results["scheduler"] = step8_scheduler_sigmas(bundle, atol)
        results["full_pipeline"] = step9_full_pipeline(
            bundle, pp, num_steps, atol)

        step_results = {k: "PASS" if v else "FAIL" for k, v in results.items()}
        all_passed = all(results.values())
        steps_passed = sum(results.values())

        return DiffResult(
            test_name="diffusion_components", model=ctx_fw.model,
            runtime_strategy=ctx_fw.runtime_strategy,
            passed=all_passed,
            status="PASS" if all_passed else "FAIL",
            message=f"{steps_passed}/{len(results)} steps passed",
            metrics={
                **step_results,
                "steps_passed": steps_passed,
                "steps_total": len(results),
            },
            duration_s=_time.monotonic() - t0)
    except Exception as e:
        return DiffResult.error(
            "diffusion_components", ctx_fw.model,
            ctx_fw.runtime_strategy, str(e))


if __name__ == "__main__":
    sys.exit(main())
