---
title: Image & Video Generation
description: Configure diffusion, classification, segmentation, and video-tracking task execution.
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

The public CLI also reserves `detect`, but an API command alone is not a
supported-model claim. Confirm an exact model-owned strategy and E2E manifest
in [Model Recipes](../models-recipes/model-recipes.md), organized by the exact
Hugging Face image or video task.

## HOI video tracking

The current SAM2-HOI recipe is a local source-package build, not a Hugging Face
checkpoint. It deliberately exposes no generic video CLI or shared
`IPipeline` extension. Load `libtrtmc_model_sam2_hoi.so` and invoke
`trtmc_sam2_hoi_video_run_jpeg_files_v1` through the model-owned public header
`trtmc/models/sam2_hoi_video.h`; see the
[SAM2-HOI video C ABI](../api/cpp-api.md#sam2-hoi-video-c-abi).

One request supplies exactly five nonempty JPEG paths in temporal order. Use
two nonempty output paths to materialize the ordered JSON plus uint8 NPY masks,
or two empty strings for a synchronous benchmark-discard call. The recipe's E2E
runner uses the same C ABI and compares all five frames against its frozen L3
snapshot, including exact object identities, labels, and interaction pairs plus
bounded boxes, scores, and binary-mask metrics.

For the denoising-loop and task-result mental model, follow the
[Diffusion, Vision, and Time-Series Tutorial](../tutorials/intermediate/diffusion-and-time-series.md).

{/* Collaborative review anchor: batch 2. */}
