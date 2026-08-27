---
title: Image & Video Generation
description: Configure diffusion, classification, segmentation, and monocular-geometry task execution.
---

## Image and video diffusion

Build-time flags define the compiled shape and denoising profile. Runtime flags
select request values within the bundle's contract.

```bash
trtmc build MODEL_ID \
  --image-height 1024 \
  --image-width 1024 \
  --num-inference-steps 28 \
  -o diffusion.bundle

trtmc generate-video diffusion.bundle \
  --prompt "A sunrise over a mountain lake" \
  --output frames \
  --num-steps 28
```

Video models can additionally compile height, width, and frame count with
`--video-height`, `--video-width`, and `--video-num-frames`. A runtime request
must remain within the profiles packaged by the exact family build.

## Classification and segmentation

```bash
trtmc classify classifier.bundle --image input.jpg
trtmc segment segmenter.bundle --image input.jpg --output mask.png
trtmc segment-prompted prompted.bundle \
  --image input.jpg --output masks --point-x 0.5 --point-y 0.5
```

## Monocular geometry

MoGe consumes one RGB image and writes a directory containing row-major
`points.f32`, `depth.f32`, `mask.u8`, and normalized `intrinsics.json`:

```bash
trtmc geometry moge-2-vitl.bundle --image input.jpg --output geometry-output
```

The MoGe-2 profile uses FP32 and a fixed 1800-token build contract.

The public CLI also reserves `detect`, but an API command alone is not a
supported-model claim. Confirm an exact model-owned strategy and E2E manifest
in [Model Recipes](../models-recipes/model-recipes.md), organized by the exact
Hugging Face image or video task.

For the denoising-loop and task-result mental model, follow the
[Diffusion, Vision, and Time-Series Tutorial](../tutorials/intermediate/diffusion-and-time-series.md).

{/* Collaborative review anchor: batch 2. */}
