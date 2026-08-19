# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Diff tool for SegFormer segmentation: TRT vs HuggingFace comparison.

Compares per-pixel class predictions and logit values between the TRT
engine (via SegmentationTrtRunner) and HF transformers SegformerForSemanticSegmentation.

Usage:
    python -m tensorrt_model_connect.models.segformer.diff_segmentation \
      --model nvidia/segformer-b0-finetuned-ade-512-512 \
      --image tests/assets/test_image.jpg --atol 0.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="SegFormer TRT vs HF diff")
    parser.add_argument("--model", required=True, help="HF model ID or local path")
    parser.add_argument("--bundle", default=None, help="Path to .bundle artifact")
    parser.add_argument("--image", required=True, help="Test image path")
    parser.add_argument("--atol", type=float, default=0.5, help="Absolute tolerance for logits")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # HF reference
    print(f"[diff_seg] Loading HF model: {args.model}", file=sys.stderr)
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    from PIL import Image

    processor = SegformerImageProcessor.from_pretrained(args.model)
    model = SegformerForSemanticSegmentation.from_pretrained(args.model)
    model.eval()

    image = Image.open(args.image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    import torch
    with torch.no_grad():
        outputs = model(**inputs)
    hf_logits = outputs.logits.cpu().numpy()  # [1, num_classes, H, W]

    hf_preds = np.argmax(hf_logits[0], axis=0)  # [H, W]
    print(f"[diff_seg] HF logits shape: {hf_logits.shape}", file=sys.stderr)
    print(f"[diff_seg] HF predictions: {hf_preds.shape}, unique classes: {len(np.unique(hf_preds))}")

    if args.bundle:
        # TRT comparison via bundle
        sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
        from tensorrt_model_connect.models.segformer.debug_runner import (
            VisionTrtRunner,
            load_section_from_bundle,
        )

        engine_plan = load_section_from_bundle(args.bundle, "engine_plan")
        if engine_plan is None:
            print("[diff_seg] FAIL: no engine_plan in bundle", file=sys.stderr)
            sys.exit(1)

        runner = VisionTrtRunner(engine_plan)

        # Preprocess image for TRT
        pixel_values = inputs["pixel_values"].numpy()  # [1, 3, H, W]
        results = runner.encode(pixel_values=pixel_values[0])

        trt_logits = results.get("logits")
        if trt_logits is not None:
            trt_logits = trt_logits.reshape(hf_logits.shape)
            max_diff = np.max(np.abs(hf_logits - trt_logits))
            print(f"[diff_seg] Max logit diff: {max_diff:.6f} (atol={args.atol})")

            trt_preds = np.argmax(trt_logits[0], axis=0)
            pixel_match = np.mean(hf_preds == trt_preds)
            print(f"[diff_seg] Pixel agreement: {pixel_match:.4f}")

            if max_diff > args.atol:
                print(f"[diff_seg] FAIL: max_diff={max_diff} > atol={args.atol}")
                sys.exit(1)
            else:
                print("[diff_seg] PASS")
                sys.exit(0)

    print("[diff_seg] HF-only mode (no bundle comparison)")


if __name__ == "__main__":
    main()
