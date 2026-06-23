#!/usr/bin/env python3
"""Validate DiT denoiser: TRT vs HuggingFace single-step comparison.

Runs the HF model forward pass, extracts intermediate representations
(patchified hidden, timestep embedding, projected text, RoPE) and feeds
those same tensors into the TRT engine for exact comparison of the
transformer block outputs.

Usage:
    python tools/validate_dit.py --model-dir <wan-diffusers-dir>
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[4]
_PYTHON_DIR = _REPO_ROOT / "python"
_TOOLS_DIR = _REPO_ROOT / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from diffusion_helpers import run_trt_engine as _run_trt_engine  # noqa: E402


def handles_validate_dit_args(argv: list[str]) -> bool:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model-dir", default="")
    ns, _ = parser.parse_known_args(argv)
    return "wan" in ns.model_dir.lower()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--atol", type=float, default=0.05)
    args = parser.parse_args()

    base = Path(args.model_dir)
    dit_dir = str(base / "transformer")

    DIM = 1536
    NUM_HEADS = 12
    NUM_LAYERS = 30
    FFN_DIM = 8960
    TEXT_SEQ = 16

    T_vid, H_vid, W_vid = 1, 12, 20
    pt, ph, pw = 1, 2, 2
    nt, nh, nw = T_vid // pt, H_vid // ph, W_vid // pw
    NUM_PATCHES = nt * nh * nw  # 60

    # --- HF reference: run full model, extract intermediates ---
    print(f"[validate-dit] Loading HF DiT (patches={NUM_PATCHES}) ...",
          file=sys.stderr)
    import torch
    from diffusers import WanTransformer3DModel
    hf_model = WanTransformer3DModel.from_pretrained(dit_dir, torch_dtype=torch.float32)
    hf_model.eval()

    torch.manual_seed(42)
    latent = torch.randn(1, 16, T_vid, H_vid, W_vid)
    timestep = torch.tensor([500.0])
    text_hidden = torch.randn(1, TEXT_SEQ, 4096)

    # Hook into the model to extract intermediates after patch + embed
    intermediates = {}

    def hook_after_patch(module, input, output):
        intermediates["after_blocks"] = output

    # Register hook on norm_out (captures hidden states right before final projection)
    hf_model.norm_out.register_forward_hook(hook_after_patch)

    with torch.no_grad():
        hf_model(
            hidden_states=latent,
            timestep=timestep,
            encoder_hidden_states=text_hidden,
        ).sample

    # Now extract the intermediate tensors by running manually
    with torch.no_grad():
        # Patch embedding
        hidden = hf_model.patch_embedding(latent)
        # hidden shape: [B, C_out, T/pt, H/ph, W/pw] -> flatten to [B, num_patches, dim]
        hidden = hidden.flatten(2).transpose(1, 2)  # [B, num_patches, dim]

        # condition_embedder returns tuple:
        # (time_embed [B,dim], block_temb [B,6*dim], text_proj [B,seq,dim], img_embed)
        cond = hf_model.condition_embedder(timestep, text_hidden)
        time_embed = cond[0]   # [B, dim] — for final scale/shift
        block_temb = cond[1]   # [B, 6*dim] — per-block modulation
        text_proj = cond[2]    # [B, text_seq, dim] — projected text

        # RoPE: returns (cos, sin) each [1, num_patches, 1, head_dim]
        rope_cos, rope_sin = hf_model.rope(latent)

    # These are the exact inputs to the DiT blocks
    hidden_np = hidden[0].numpy()      # [num_patches, dim]
    temb_np = block_temb.numpy()       # [1, 6*dim]
    time_embed_np = time_embed.numpy() # [1, dim]
    text_np = text_proj[0].numpy()     # [text_seq, dim]
    cos_np = rope_cos[0, :, 0, :].numpy()  # [num_patches, head_dim]
    sin_np = rope_sin[0, :, 0, :].numpy()  # [num_patches, head_dim]

    print("[validate-dit] Extracted intermediates:", file=sys.stderr)
    print(f"  hidden: {hidden_np.shape}", file=sys.stderr)
    print(f"  block_temb: {temb_np.shape}", file=sys.stderr)
    print(f"  time_embed: {time_embed_np.shape}", file=sys.stderr)
    print(f"  text: {text_np.shape}", file=sys.stderr)
    print(f"  cos: {cos_np.shape}, sin: {sin_np.shape}", file=sys.stderr)

    # Run the blocks manually in HF to get reference output
    with torch.no_grad():
        h = hidden.clone()  # [B, num_patches, dim]
        # Reshape temb from [B, 6*dim] to [B, 6, dim] as HF forward does
        block_temb_6d = block_temb.unflatten(1, (6, -1))  # [B, 6, dim]
        for block in hf_model.blocks:
            h = block(h, text_proj, block_temb_6d, (rope_cos, rope_sin))
        # Final: norm + scale/shift + proj
        # Use time_embed (not block_temb) for final modulation
        shift, scale = (hf_model.scale_shift_table +
                        time_embed.unsqueeze(1)).chunk(2, dim=1)
        h_normed = hf_model.norm_out(h.float()) * (1 + scale) + shift
        hf_block_out = hf_model.proj_out(h_normed)
        hf_block_out = hf_block_out.squeeze(0)  # [num_patches, out_dim]

    hf_final = hf_block_out.numpy()
    print(f"[validate-dit] HF block output: shape={hf_final.shape}, "
          f"range=[{hf_final.min():.4f}, {hf_final.max():.4f}]", file=sys.stderr)

    # --- TRT ---
    print("[validate-dit] Loading DiT weights & building TRT engine ...",
          file=sys.stderr)
    if str(_PYTHON_DIR) not in sys.path:
        sys.path.insert(0, str(_PYTHON_DIR))
    from tensorrt_model_connect.families.wan_t2v.standard_dit_builder import build_standard_dit_engine, load_dit_weights

    t0 = time.time()
    weights = load_dit_weights(dit_dir, dim=DIM, num_heads=NUM_HEADS,
                               num_layers=NUM_LAYERS, ffn_dim=FFN_DIM,
                               context_dim=4096)

    plan = build_standard_dit_engine(
        weights, dim=DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
        ffn_dim=FFN_DIM, context_dim=DIM, num_patches=NUM_PATCHES,
        text_seq_len=TEXT_SEQ, qk_norm=True, cross_attn_norm=True)
    print(f"[validate-dit] Engine built [{time.time()-t0:.1f}s, "
          f"{len(plan)/(1024*1024):.0f}MB]", file=sys.stderr)

    # Run TRT with extracted intermediates
    print("[validate-dit] Running TRT inference ...", file=sys.stderr)
    trt_results = _run_trt_engine(plan, {
        "hidden_states": hidden_np,
        "timestep_embedding": temb_np,
        "time_embed": time_embed_np,
        "encoder_hidden_states": text_np,
        "rotary_cos": cos_np,
        "rotary_sin": sin_np,
    }, {
        "output": (hf_final.shape, np.float32),
    })
    trt_final = trt_results["output"]

    print(f"[validate-dit] TRT output: shape={trt_final.shape}, "
          f"range=[{trt_final.min():.4f}, {trt_final.max():.4f}]", file=sys.stderr)

    # --- Compare ---
    max_diff = np.max(np.abs(hf_final - trt_final))
    mean_diff = np.mean(np.abs(hf_final - trt_final))
    cos_sim = np.sum(hf_final * trt_final) / (
        np.linalg.norm(hf_final) * np.linalg.norm(trt_final) + 1e-8)

    print("\n=== DiT Denoiser Validation ===")
    print(f"Patches: {NUM_PATCHES}, Layers: {NUM_LAYERS}")
    print(f"Max abs diff: {max_diff:.6f}")
    print(f"Mean abs diff: {mean_diff:.6f}")
    print(f"Cosine sim: {cos_sim:.6f}")

    if max_diff <= args.atol:
        print(f"PASS (max_diff={max_diff:.6f} <= atol={args.atol})")
        return 0
    else:
        print(f"FAIL (max_diff={max_diff:.6f} > atol={args.atol})")
        # Even if numerics don't perfectly match, cosine > 0.9 is acceptable
        # for FP32 TRT vs PyTorch (30 layers of accumulation)
        if cos_sim > 0.9:
            print(f"NOTE: cosine sim {cos_sim:.4f} > 0.9 suggests implementation is correct "
                  f"with expected FP32 drift")
        return 1


if __name__ == "__main__":
    sys.exit(main())
