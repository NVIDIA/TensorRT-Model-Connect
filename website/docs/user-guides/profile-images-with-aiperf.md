---
title: Profile Image Generation with AIPerf
description: Benchmark an already-built text-to-image model and receive complete PNG outputs.
---

First follow [Build from Source](../getting-started/source-build.md) and
[Build a Bundle](./build-a-bundle.md), using an existing image-generation
profile. This example assumes the
[Stable Diffusion v1.5 profile](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/families/stable_diffusion/tests/manifests/stable-diffusion-v1-5.json)
has already been built as `stable-diffusion-v1-5.bundle` and passed its family's
correctness checks. Its workload is one **256×256 image, 10 steps, guidance 7.5**.

Run both terminals in your prepared GPU environment (the same container when
using Docker). `trtmc-server` must be on `PATH`.

## 1. Install AIPerf

```bash
python -m pip install -e '.[serve]' -C py-only=true
python -m venv .venv-aiperf
.venv-aiperf/bin/pip install aiperf==0.13.0
```

The client comes from PyPI; an AIPerf source checkout is unnecessary.

## 2. Start the built model (terminal 1)

```bash
trtmc-server ./stable-diffusion-v1-5.bundle \
  --model-name stable-diffusion-v1-5 \
  --startup-timeout 1800 \
  --records artifacts/aiperf-images-server.jsonl
```

Large image bundles can take several minutes to load.
Wait for `Uvicorn running on http://127.0.0.1:8000`. Add `--runtime-root`
if the native libraries are in a separate directory.

## 3. Benchmark it (terminal 2)

```bash
echo '{"text":"a photograph of an astronaut riding a horse"}' > /tmp/image-prompts.jsonl
.venv-aiperf/bin/aiperf profile \
  --url http://127.0.0.1:8000 --model stable-diffusion-v1-5 \
  --endpoint-type image_generation --tokenizer builtin \
  --input-file /tmp/image-prompts.jsonl --custom-dataset-type single_turn \
  --extra-inputs size:256x256 num_inference_steps:10 guidance_scale:7.5 cfg_scale:7.5 seed:0 \
  --concurrency 1 --warmup-request-count 1 --request-count 5 \
  --export-level raw --artifact-dir artifacts/aiperf-images
```

Each response contains one complete PNG in `data[0].b64_json`. It remains usable
after the server removes its temporary files. Raw AIPerf exports retain the
response; server records identify the same PNG by SHA-256 and include its actual
dimensions, count, generation controls and request ID.
Server records hash both positive and negative prompt text rather than storing it.

Read client request latency and throughput in `profile_export_aiperf.json`.
With one image per request, requests/sec is images/sec. Server `model_call_ms`
measures `public_task_call_wall`, including family preprocessing and generation.
PNG encoding, HTTP transfer, build, loading, startup and warmup are outside that
steady-state Task measurement. Keep client latency and Task time separate.

## Run another built image model

Replace the bundle, API name, prompt and explicit generation controls with those
of an existing full-model profile. The bundle must declare `image_generation`
or SDK `text_to_image` and produce one image. Video and image editing are
separate workloads. Dimensions must match the built profile. The server accepts
`seed`, `num_inference_steps`, `guidance_scale`, `cfg_scale`, `negative_prompt`
and `size`; it rejects batching, streaming, URL responses and unsupported options
such as `quality` rather than ignoring them.

For matched-noise validation, `initial_latents_b64` accepts the owning family's
existing raw float32 latent layout. A shared seed alone does not guarantee the
same random noise between native C++ and a Python reference. Increase
`--max-body-bytes` when replay noise needs a larger request (up to the worker's
16 MiB auxiliary-file limit). Model-specific latent creation and quality checks
stay in the owning family.

For automated request/metric joins, PNG identity checks and saved images, see
[`image-profile`](https://github.com/NVIDIA/TensorRT-Model-Connect/tree/main/apps/aiperf_qual#native-endpoint-group-coverage).
