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

For a bundle that declares the `hoi_video_tracking` task contract, pass an
ordered frame directory and separate metadata/mask destinations:

```bash
trtmc track-hoi sam2-hoi-tracking.bundle \
  --frames-dir frames \
  --output-json tracking.json \
  --output-masks-dir masks
```

For repeatable in-process timings, add `--benchmark N --warmup N` and write a
raw-sample receipt with `--benchmark-json`. The default `predecoded` scope
reuses decoded frame views; `--benchmark-scope loaded-request` instead includes
fresh directory enumeration and frame decode in every measured request. Both
discard JSON/NPY outputs during warmup and timing, then run one final untimed
request that writes the normal accuracy artifacts. Consult the receipt's
`timing_boundary` fields before comparing measurements from different scopes.
The top-level `frame_loading` object's `frame_decode_mode` and
`frame_decode_max_concurrency` record whether materialization used the serial
loader or the model's bounded batch loader.

`track-hoi` uses the generic `IVideoTrackingPipeline` capability. The current
SAM2-HOI recipe is a local source-package build, not a Hugging Face checkpoint;
its family recipe page records the exact source and nightly validation contract.
One request requires fixed-resolution frames, and the first frame with any HOI
detections must select exactly two tracked objects. Leading frames with no
detections are scanned but omitted; a nonempty frame with another selection
count fails instead of being skipped.

For the denoising-loop and task-result mental model, follow the
[Diffusion, Vision, and Time-Series Tutorial](../tutorials/intermediate/diffusion-and-time-series.md).

{/* Collaborative review anchor: batch 2. */}
