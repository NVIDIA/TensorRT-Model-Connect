---
title: Profile Image Understanding with AIPerf
description: Benchmark an already-built vision-language model with one image and one question per request.
---

This example uses **Qwen3-VL-2B-Instruct**. First follow
[Build from Source](../getting-started/source-build.md) and
[Build a Bundle](./build-a-bundle.md), using the existing
[Qwen3-VL-2B profile](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/families/qwen_vl/tests/manifests/qwen3-vl-2b.json).
Include the `trtmc_model_qwen_vl` runtime target and run the family's correctness
checks. The commands below assume `qwen3-vl-2b.bundle` is already built and
`trtmc-server` is on `PATH`.

Build Qwen-VL bundles with the current builder: it preserves image aspect ratios
with smart resize, using the existing resize targets (nominally 196 image tokens
for Qwen3-VL, 256 for Qwen2.5-VL). Dynamic profiles also admit upstream rounding
at extreme aspect ratios. Older bundles
that force every image into a square should be rebuilt before qualification.

Run from the TRTMC checkout in your prepared GPU environment. In Docker, open
both terminals in the same container.

## 1. Install the client and prepare a question

```bash
python -m pip install -e '.[serve]' -C py-only=true
python -m venv .venv-aiperf
.venv-aiperf/bin/pip install aiperf==0.13.0
.venv-aiperf/bin/python apps/aiperf_qual/examples/prepare_vision_input.py \
  --image families/qwen_vl/tests/data/test_img.jpeg \
  --prompt 'What color is the vehicle in this image? Answer in one word.' \
  --output /tmp/aiperf-vision.jsonl
```

The input contains the original image bytes and the question. The client comes
from PyPI; no AIPerf source checkout is needed.

## 2. Start the model (terminal 1)

```bash
trtmc-server ./qwen3-vl-2b.bundle --model-name Qwen/Qwen3-VL-2B-Instruct \
  --records artifacts/aiperf-vision-server.jsonl
```

Wait for `Uvicorn running on http://127.0.0.1:8000`. Use the full binary path
and `--runtime-root` if your native libraries are in a separate directory.

## 3. Run AIPerf (terminal 2)

```bash
.venv-aiperf/bin/aiperf profile \
  --url http://127.0.0.1:8000 --model Qwen/Qwen3-VL-2B-Instruct \
  --tokenizer-revision 89644892e4d85e24eaac8bacfd4f463576704203 \
  --endpoint-type chat --endpoint /v1/chat/completions \
  --input-file /tmp/aiperf-vision.jsonl --custom-dataset-type single_turn \
  --extra-inputs max_tokens:10 temperature:1.0 top_k:1 \
  --concurrency 1 --warmup-request-count 5 --request-count 50 \
  --random-seed 0 --ui-type simple --export-level raw \
  --artifact-dir artifacts/aiperf-vision
```

Each request sends one image and one question, then receives a complete answer.
Read request latency and request throughput in `profile_export_aiperf.json`.
This is a nonstreaming workload: token TTFT and inter-token latency are not
qualified. Server records preserve request identity, image dimensions and bytes,
output-token counts, and `model_call_ms` under `public_task_call_wall`.
Task time includes family preprocessing and generation; transport image decoding
is outside it. Keep Task time separate from client latency. Build, load, startup,
and warmup are excluded from steady-state measurements.
Client token counts are estimates; image prompt-token counts are unavailable.
Server records retain the actual native output-token counts.

## Run another built model

Replace the bundle, API model name, tokenizer revision, image, question, and output-token limit.
Use that family's build guidance and correctness checks. The bundle must declare
`vision_language_generation` or SDK `images_text_to_text`; a text-only bundle from the same checkpoint does
not qualify. Image preprocessing and prompt templates remain owned by the family.

Rebuild older DeepSeek-OCR-2 bundles to use aspect-preserving padding with the
current family runtime before qualification. Rebuild older native InternVL3
bundles to include the checkpoint processor's image-token delimiters.

Native InternVL builds follow the checkpoint processor's image tiling (up to
12 local tiles plus a thumbnail). Use `--max-sequence-length 4096` to leave room
for those visual tokens and the answer. Rebuild older single-view bundles with
the current family runtime before qualification.

For a local tokenizer, omit `--tokenizer-revision` and add
`--tokenizer /path/to/tokenizer` to AIPerf. For Lance, run this once in its
prepared family environment:

```bash
python families/lance/tests/aiperf/prepare_tokenizer.py \
  --model-dir /path/to/Lance --output-dir /tmp/lance-tokenizer
```

Then use `--tokenizer /tmp/lance-tokenizer`. The automated `endpoint-profile`
runner accepts the same option and pins the default tokenizer to the manifest's
checkpoint revision.

For Phi4, use `families/phi4_multimodal/tests/aiperf/prepare_tokenizer.py` with
the same flags, then pass its output directory with `--tokenizer`.

The endpoint accepts one user message containing text and one embedded RGB
PNG/JPEG image. It rejects multi-image inputs, multi-turn conversations, system
prompts, streaming, stop sequences, and thinking controls. The server does not
fetch remote image URLs or open client-provided paths. Defaults allow 16 million
pixels and a 1 MiB request; raise `--max-body-bytes` when base64 image data exceeds
that limit.
