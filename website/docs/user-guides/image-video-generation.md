---
title: Image & Video Generation
description: Build and execute image, video, classification, segmentation, and monocular-geometry families.
---

Set the installed native runtime directory once:

```bash
TRTMC_RUNTIME_ROOT=/opt/trtmc/lib
```

## Image and video generation

Build-time arguments define compiled dimensions and frame bounds. Request-time
arguments choose denoising and guidance values within that contract.

```bash
python -m tensorrt_model_connect build MODEL_ID \
  --task image_generation \
  --image-height 1024 \
  --image-width 1024 \
  --output image.bundle

trtmc generate-image image.bundle \
  --runtime-root "$TRTMC_RUNTIME_ROOT" \
  --prompt "A sunrise over a mountain lake" \
  --output generated.png \
  --height 1024 \
  --width 1024 \
  --num-steps 28
```

For a family that declares `video_generation`, build with its supported
`--video-num-frames` value and execute `generate-video`:

```bash
trtmc generate-video video.bundle \
  --runtime-root "$TRTMC_RUNTIME_ROOT" \
  --prompt "A sunrise over a mountain lake" \
  --output frames \
  --num-steps 28
```

The selected family owns schedulers, latent shapes, preprocessing,
postprocessing, and supported request ranges.

## Classification and segmentation

These commands emit JSON to standard output:

```bash
trtmc classify classifier.bundle \
  --runtime-root "$TRTMC_RUNTIME_ROOT" \
  --image input.jpg

trtmc segment segmenter.bundle \
  --runtime-root "$TRTMC_RUNTIME_ROOT" \
  --image input.jpg \
  > segmentation.json

trtmc segment-prompted prompted.bundle \
  --runtime-root "$TRTMC_RUNTIME_ROOT" \
  --image input.jpg \
  --point-x 0.5 \
  --point-y 0.5 \
  --foreground true
```

## Monocular geometry

MoGe consumes one RGB image and writes a directory containing row-major
`points.f32`, `depth.f32`, `mask.u8`, and normalized `intrinsics.json`:

```bash
trtmc geometry moge.bundle \
  --runtime-root "$TRTMC_RUNTIME_ROOT" \
  --image input.jpg \
  --output geometry-output
```

A CLI command is only a public API surface, not a model-support claim. Confirm
the exact checkpoint, task, build arguments, and evidence in
[Model Recipes](../models-recipes/model-recipes.md).
