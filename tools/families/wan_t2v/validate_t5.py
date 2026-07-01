#!/usr/bin/env python3
"""Validate T5 encoder: TRT vs HuggingFace per-token embedding comparison.

Usage:
    python tools/validate_t5.py --model-dir <wan-diffusers-dir>
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PYTHON_DIR = _REPO_ROOT / "python"
_TOOLS_DIR = _REPO_ROOT / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from diffusion_helpers import run_trt_engine as _run_trt_engine  # noqa: E402


def handles_validate_t5_args(argv: list[str]) -> bool:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model-dir", default="")
    ns, _ = parser.parse_known_args(argv)
    return "wan" in ns.model_dir.lower()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True,
                        help="Wan2.1-T2V-1.3B-Diffusers directory")
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--atol", type=float, default=0.1)
    parser.add_argument("--prompt", default="A cat sitting on a beach")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    te_dir = str(model_dir / "text_encoder")

    # --- HF reference ---
    print("[validate] Loading HF T5 encoder ...", file=sys.stderr)
    import torch
    from transformers import AutoTokenizer, UMT5EncoderModel

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir / "tokenizer"))
    hf_model = UMT5EncoderModel.from_pretrained(te_dir, torch_dtype=torch.float32)
    # Fix: UMT5 checkpoint stores embedding as 'shared.weight' but
    # UMT5EncoderModel expects 'encoder.embed_tokens.weight'. Copy it.
    if (hasattr(hf_model, 'shared') and hf_model.shared is not None and
            hf_model.encoder.embed_tokens.weight.abs().sum().item() < 1e-6):
        hf_model.encoder.embed_tokens.weight = hf_model.shared.weight
    hf_model.eval()

    encoding = tokenizer(
        args.prompt,
        max_length=args.max_seq_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"]

    with torch.no_grad():
        hf_out = hf_model(
            input_ids=input_ids,
            attention_mask=encoding["attention_mask"],
        ).last_hidden_state.numpy()

    print(f"[validate] HF shape: {hf_out.shape}, "
          f"range=[{hf_out.min():.4f}, {hf_out.max():.4f}]", file=sys.stderr)

    # --- TRT ---
    print("[validate] Loading T5 weights ...", file=sys.stderr)
    if str(_PYTHON_DIR) not in sys.path:
        sys.path.insert(0, str(_PYTHON_DIR))
    from tensorrt_model_connect.families.wan_t2v.model.components.text_encoder import build_t5_encoder_engine, load_t5_weights

    cfg = hf_model.config
    t0 = time.time()
    weights = load_t5_weights(
        te_dir, d_model=cfg.d_model, num_heads=cfg.num_heads,
        d_kv=cfg.d_kv, d_ff=cfg.d_ff, num_layers=cfg.num_layers,
        vocab_size=cfg.vocab_size,
    )
    print(f"[validate] Weights loaded [{time.time()-t0:.1f}s]", file=sys.stderr)

    print("[validate] Building TRT engine ...", file=sys.stderr)
    t1 = time.time()
    plan = build_t5_encoder_engine(
        weights, d_model=cfg.d_model, num_heads=cfg.num_heads,
        d_kv=cfg.d_kv, d_ff=cfg.d_ff, num_layers=cfg.num_layers,
        vocab_size=cfg.vocab_size, max_seq_len=args.max_seq_len,
    )
    print(f"[validate] Engine built [{time.time()-t1:.1f}s, "
          f"{len(plan)/(1024*1024):.0f}MB]", file=sys.stderr)

    # Run TRT
    print("[validate] Running TRT inference ...", file=sys.stderr)
    inp_np = input_ids.numpy().astype(np.int32)
    # Build attention mask: 0.0 for valid tokens, -inf for padding
    attn_mask_np = encoding["attention_mask"].numpy().astype(np.float32)
    # HF mask: 1 = valid, 0 = padding -> TRT mask: 0 = valid, -3.4e38 = padding
    trt_mask = np.where(attn_mask_np > 0.5, 0.0, np.finfo(np.float32).min).astype(np.float32)
    results = _run_trt_engine(plan, {
        "input_ids": inp_np,
        "attention_mask": trt_mask,
    }, {
        "text_embeddings": ((1, args.max_seq_len, cfg.d_model), np.float32),
    })
    trt_out = results["text_embeddings"]

    print(f"[validate] TRT shape: {trt_out.shape}, "
          f"range=[{trt_out.min():.4f}, {trt_out.max():.4f}]", file=sys.stderr)

    # --- Compare ---
    attn_mask = encoding["attention_mask"].numpy()[0]
    valid_len = int(attn_mask.sum())

    hf_valid = hf_out[0, :valid_len, :]
    trt_valid = trt_out[0, :valid_len, :]

    max_diff = np.max(np.abs(hf_valid - trt_valid))
    mean_diff = np.mean(np.abs(hf_valid - trt_valid))
    cos_sim = np.sum(hf_valid * trt_valid) / (
        np.linalg.norm(hf_valid) * np.linalg.norm(trt_valid) + 1e-8)

    print("\n=== T5 Encoder Validation ===")
    print(f"Prompt: {args.prompt!r}")
    print(f"Seq len: {args.max_seq_len} (valid: {valid_len})")
    print(f"Max abs diff: {max_diff:.6f}")
    print(f"Mean abs diff: {mean_diff:.6f}")
    print(f"Cosine sim: {cos_sim:.6f}")

    if max_diff <= args.atol:
        print(f"PASS (max_diff={max_diff:.6f} <= atol={args.atol})")
        return 0
    else:
        print(f"FAIL (max_diff={max_diff:.6f} > atol={args.atol})")
        return 1


if __name__ == "__main__":
    sys.exit(main())
