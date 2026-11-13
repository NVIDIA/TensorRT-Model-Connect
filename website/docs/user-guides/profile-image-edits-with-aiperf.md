---
title: Profile Image Editing with AIPerf
description: Send a reference image and instruction to an already-built native editing model.
---

First use [Build from Source](../getting-started/source-build.md) and
[Build a Bundle](./build-a-bundle.md). This example assumes the existing
[Qwen Image Edit 2511 profile](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/families/qwen_image/tests/manifests/qwen-image-edit-2511.json)
has been built as `qwen-image-edit-2511.bundle` and passed its family's
correctness checks. Build for the example reference image's **640×382 input**;
the requested output is **1024×1024, 20 steps, CFG 4, seed 42**.
The Qwen editing builder uses its image dimensions for the conditioning input.

Run both terminals in your prepared GPU environment, from the repository root.
`trtmc-server` must be on `PATH`.

## 1. Install AIPerf

```bash
python -m pip install -e '.[serve]' -C py-only=true
python -m venv .venv-aiperf
.venv-aiperf/bin/pip install aiperf==0.13.0
```

## 2. Start the built model (terminal 1)

```bash
trtmc-server ./qwen-image-edit-2511.bundle \
  --model-name qwen-image-edit-2511 \
  --startup-timeout 1800 --request-timeout 1800 \
  --records artifacts/aiperf-edits-server.jsonl
```

Wait for `Uvicorn running on http://127.0.0.1:8000`. Loading a large bundle
can take several minutes. Add `--runtime-root` when native libraries are in
a separate directory.

## 3. Benchmark it (terminal 2)

Prepare the existing example photo as a lossless PNG. Its pixels and dimensions
stay unchanged; this avoids a second lossy JPEG encoding in the client.

```bash
python - <<'PY'
import json
from pathlib import Path
from PIL import Image
Image.open("families/qwen_image/tests/data/test_img.jpeg").save("/tmp/edit-reference.png")
row = {"text": "Turn the image into a watercolor illustration while preserving the main subject.",
       "image": "/tmp/edit-reference.png", "extra": {"negative_prompt": " "}}
Path("/tmp/image-edits.jsonl").write_text(json.dumps(row) + "\n")
PY
.venv-aiperf/bin/aiperf profile \
  --url http://127.0.0.1:8000 --model qwen-image-edit-2511 \
  --endpoint-type image_edit --tokenizer builtin \
  --input-file /tmp/image-edits.jsonl --custom-dataset-type single_turn \
  --extra-inputs size:1024x1024 num_inference_steps:20 guidance_scale:4 cfg_scale:4 seed:42 \
  --concurrency 1 --warmup-request-count 1 --request-count 5 \
  --export-level raw --artifact-dir artifacts/aiperf-edits
```

AIPerf uploads the image and instruction using multipart POST `/v1/images/edits`.
Each response contains a complete edited PNG in `data[0].b64_json`, usable after
the server removes its temporary files. Raw exports retain the response.
Server records bind the uploaded reference hash and dimensions, instruction hash,
controls and output PNG hash to the same request ID.

Read client latency and images/sec in `profile_export_aiperf.json`. Server
`model_call_ms` measures `public_task_call_wall`, including the family's image
conditioning and generation. Uploads, image-container decoding, PNG encoding,
HTTP transfer, build, loading, startup and warmup are outside steady-state Task
timing. Keep client latency and Task time separate.

## Use another built editing model or reference

Select a bundle declaring `image_edit` or SDK `images_text_to_image_edit` and
consuming one reference image. Replace the bundle, API name, reference,
instruction and explicit controls with those supported by its existing full
profile. Follow the owning family's input/build geometry requirements; input
and output dimensions can differ. A plain image generator does not become an
editing model by receiving an image.

The server accepts one RGB PNG/JPEG upload, `size`, `seed`,
`num_inference_steps`, `guidance_scale`, `cfg_scale` and `negative_prompt`.
It rejects masks, multiple references, remote reference URLs, output batching,
streaming and unsupported options. For matched-noise checks,
`initial_latents_b64` uses the owning family's existing raw float32 layout;
increase `--max-body-bytes` when the replay upload exceeds the default limit.

For automated input/output hash checks, request/metric joins and saved edited
images, use [`image-edit-profile`](https://github.com/NVIDIA/TensorRT-Model-Connect/tree/main/apps/aiperf_qual#native-endpoint-group-coverage).
It reuses the image-generation qualifier and records build geometry separately
from actual input and requested output geometry. Owning-family correctness on
the served bundle remains a separate prerequisite.
