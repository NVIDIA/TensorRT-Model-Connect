---
title: Image & Video
description: Build and run diffusion, classification, segmentation, and geometry tasks.
---

## Image and video generation

Build-time dimensions and frame count shape the TensorRT plans. Request-time
options must remain within the owning family's contract.

```bash
python -m tensorrt_model_connect build MODEL_ID \
  --image-height 1024 \
  --image-width 1024 \
  --video-num-frames 49 \
  --output media.bundle

trtmc generate-video media.bundle \
  --prompt "A sunrise over a mountain lake" \
  --output frames \
  --num-steps 28 \
  --seed 7
```

Use `generate-image` for one image and `generate-image-batch` with a prompt
file plus comma-separated seeds for a family that declares the batch Task.

## Perception

```bash
trtmc classify classifier.bundle \
  --image input.jpg

trtmc segment segmenter.bundle \
  --image input.jpg

trtmc segment-prompted prompted.bundle \
  --image input.jpg --point-x 0.5 --point-y 0.5 --foreground true

trtmc geometry moge.bundle \
  --image input.jpg --output geometry-output
```

Other current Task commands include `extract-features`, stereo `disparity`,
and `video-segment`. Exact input shape, fixed profile, output representation,
and supported checkpoint remain family-owned.

Continue with the
[Diffusion and Time-Series Tutorial](../tutorials/intermediate/diffusion-and-time-series.md).
