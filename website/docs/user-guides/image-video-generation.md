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
  --runtime-root /opt/trtmc/lib \
  --prompt "A sunrise over a mountain lake" \
  --output frames \
  --num-steps 28 \
  --seed 7
```

Use `generate-image` for one image and `generate-image-batch` with a prompt
file plus comma-separated seeds for a family that declares the batch Task.

To benchmark a built text-to-image bundle over HTTP, see
[Profile Image Generation with AIPerf](./profile-images-with-aiperf.md).
For one reference image and instruction, use
[Profile Image Editing with AIPerf](./profile-image-edits-with-aiperf.md).

For complete prompt-to-video clips with polling and download, see
[Profile Video Generation with AIPerf](./profile-videos-with-aiperf.md).

A bundle built for `initial_image_text_to_video` starts the video from an
image. Wan2.2-TI2V-5B builds it for 1280x704, the vertical 704x1280 or the
672x384 test profile and takes an image of exactly that size:

```bash
python -m tensorrt_model_connect build Wan-AI/Wan2.2-TI2V-5B \
  --task initial_image_text_to_video \
  --image-height 1280 --image-width 704 --video-num-frames 121 \
  --output wan22-portrait.bundle

trtmc generate-video wan22-portrait.bundle \
  --runtime-root /opt/trtmc/lib \
  --image first-frame.png \
  --prompt "The cat on the surfboard turns its head toward the waves" \
  --output frames --seed 42
```

`trtmc-server` serves such a bundle at POST `/v1/videos`. Send the image as a
multipart `input_reference` PNG or JPEG of the video size; the server rejects
other sizes. Requests without an image stay text-to-video. Upstream Wan
image-to-video scales an image to cover the frame and crops the center, which
Pillow does in one call:

```python
from PIL import Image, ImageOps

frame = ImageOps.fit(Image.open("photo.jpg").convert("RGB"), (704, 1280), Image.LANCZOS)
frame.save("first-frame.png")
```

## Perception

```bash
trtmc classify classifier.bundle \
  --runtime-root /opt/trtmc/lib --image input.jpg

trtmc segment segmenter.bundle \
  --runtime-root /opt/trtmc/lib --image input.jpg

trtmc segment-prompted prompted.bundle \
  --runtime-root /opt/trtmc/lib \
  --image input.jpg --point-x 0.5 --point-y 0.5 --foreground true

trtmc geometry moge.bundle \
  --runtime-root /opt/trtmc/lib \
  --image input.jpg --output geometry-output
```

Other current Task commands include `extract-features`, stereo `disparity`,
and `video-segment`. Exact input shape, fixed profile, output representation,
and supported checkpoint remain family-owned.

Continue with the
[Diffusion and Time-Series Tutorial](../tutorials/intermediate/diffusion-and-time-series.md).
