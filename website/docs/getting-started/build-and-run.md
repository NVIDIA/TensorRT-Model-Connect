---
title: Model Recipes
---

This page is a task index after the [Quick Start](quick-start.md). Exact
checkpoint, task, precision, topology, dependency, and validation support is
generated on [Models & Recipes](../models-recipes/overview.md) from the current
family-owned manifests.

## Text generation

```bash
python -m tensorrt_model_connect build Qwen/Qwen3-0.6B \
  --precision fp16 \
  --max-sequence-length 256 \
  --output /tmp/qwen.bundle

trtmc run /tmp/qwen.bundle \
  --prompt "What is the capital of France? Answer in one word." \
  --max-new-tokens 10 \
  --temperature 0 \
  --use-chat-template true \
  --enable-thinking false
```

Continue with [Text Generation](../tutorials/beginner/text-generation.md) for
sampling and request framing.

## Vision-language generation

```bash
python -m tensorrt_model_connect build Qwen/Qwen2.5-VL-3B-Instruct \
  --precision fp16 \
  --max-sequence-length 384 \
  --output /tmp/qwen25vl.bundle

trtmc run /tmp/qwen25vl.bundle \
  --prompt "Describe this image." \
  --image families/qwen_vl/tests/data/test_img.jpeg \
  --max-new-tokens 48
```

The Qwen-VL family owns image preprocessing, cache policy, Task implementation,
and validation. Similar Task shape does not imply shared runtime code.

## Speech and audio

```bash
python -m tensorrt_model_connect build openai/whisper-large-v3-turbo \
  --precision fp16 \
  --output /tmp/whisper.bundle

trtmc transcribe /tmp/whisper.bundle \
  --input families/whisper/tests/data/Recording.wav \
  --max-output-tokens 224
```

```bash
python -m tensorrt_model_connect build nvidia/magpie_tts_multilingual_357m \
  --precision fp16 \
  --output /tmp/magpie.bundle

trtmc generate-audio /tmp/magpie.bundle \
  --prompt "A clear short test sentence." \
  --output /tmp/magpie.wav
```

Use `transcribe-batch` for repeated `--input`, `transcribe-streaming` for the
streaming Task, and `speech-session` for full-duplex family contracts. Each is
a distinct public Task interface.

## Image, video, and perception

Use `generate-image`, `generate-image-batch`, or `generate-video` according to
the family task. Perception commands include `classify`, `extract-features`,
`disparity`, `geometry`, `segment`, `segment-prompted`, and `video-segment`.

```bash
python -m tensorrt_model_connect build nvidia/segformer-b0-finetuned-ade-512-512 \
  --precision fp16 \
  --output /tmp/segformer.bundle

trtmc segment /tmp/segformer.bundle \
  --image families/segformer/tests/data/test_img.jpeg
```

The command prints task JSON. Inspect the exact family manifest before
assuming a checkpoint, task, shape, or output format is supported.

## Time-series forecasting

Chronos-Bolt uses the `forecast` Task. Its input is a raw float32 file, not a
comma-separated CLI value:

```bash
python -m tensorrt_model_connect build amazon/chronos-bolt-tiny \
  --precision fp32 \
  --output /tmp/chronos.bundle

trtmc forecast /tmp/chronos.bundle \
  --input /path/to/history.f32
```

Neural operators use `solve --branch FILE --trunk FILE`. Do not interchange
the contracts. See [Time-Series](../user-guides/time-series.md).
