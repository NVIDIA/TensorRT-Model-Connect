---
title: Image & Video Generation
description: Configure diffusion, classification, and segmentation task execution.
---

## Image and video diffusion

Build-time flags define the compiled shape and denoising profile. Runtime flags
select request values within the bundle's contract.

```bash
trtmc build MODEL_ID \
  --image-height 1024 \
  --image-width 1024 \
  --num-inference-steps 28 \
  -o diffusion.trtfb

trtmc generate-video diffusion.trtfb \
  --prompt "A sunrise over a mountain lake" \
  --output frames \
  --num-steps 28
```

Video models can additionally compile height, width, and frame count with
`--video-height`, `--video-width`, and `--video-num-frames`. A runtime request
must remain within the profiles packaged by the exact family build.

## Classification and segmentation

```bash
trtmc classify classifier.trtfb --image input.jpg
trtmc segment segmenter.trtfb --image input.jpg --output mask.png
trtmc segment-prompted prompted.trtfb \
  --image input.jpg --output masks --point-x 0.5 --point-y 0.5
```

The public CLI also reserves `detect`, but an API command alone is not a
supported-model claim. Confirm an exact model-owned strategy and E2E manifest
in [Model Recipes](../models-recipes/model-recipes.md), organized by the exact
Hugging Face image or video task.

For the denoising-loop and task-result mental model, follow the
[Diffusion, Vision, and Time-Series Tutorial](../tutorials/intermediate/diffusion-and-time-series.md).
