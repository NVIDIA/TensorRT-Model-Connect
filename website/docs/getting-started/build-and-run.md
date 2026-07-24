---
title: Build and Run
---

All examples use `$TRTMC` for the unified CLI. Set it to `trtmc` when using a
release wheel, or `./build/trtmc` when using a source build inside the dev
container.

```bash
TRTMC=trtmc
# Source build alternative:
# TRTMC=./build/trtmc
```

Run [Installation](installation.md) or
[Environment and First Repro](environment-and-repro.md) first if either command
fails.

## Text generation

```bash
$TRTMC build Qwen/Qwen3-0.6B -o /tmp/qwen3.trtfb --precision fp16

$TRTMC run /tmp/qwen3.trtfb \
  --prompt "Write one sentence about TensorRT." \
  --max-new-tokens 32 \
  --temperature 0.7 \
  --top-p 0.9
```

Useful runtime knobs include `--greedy`, `--top-k`, `--top-p`, `--min-p`, `--seed`, `--chat-template`, and `--no-thinking`. Use `--greedy` for deterministic smoke tests before trying sampling. Qwen text generation uses the native C++ BPE tokenizer from the bundle, so `--hf-python` is not needed for this path.

## Vision-language generation

```bash
$TRTMC build Qwen/Qwen2.5-VL-3B-Instruct \
  -o /tmp/qwen25vl.trtfb \
  --precision fp16 \
  --max-cache-length 384

$TRTMC run /tmp/qwen25vl.trtfb \
  --prompt "Describe this image." \
  --image tests/assets/test_image.jpg \
  --max-new-tokens 48
```

This Qwen-VL bundle routes through the model-owned
`runtime_strategy="qwen_vl_vision_language"`. Other vision-language families
use their own strategy keys and DSOs even when they implement the same public
`generate(prompt, image, ...)` task shape.

## Speech and audio

```bash
$TRTMC build openai/whisper-large-v3-turbo -o /tmp/whisper.trtfb --precision fp16

$TRTMC transcribe /tmp/whisper.trtfb \
  --audio tests/e2e/models/whisper/data/Recording.wav \
  --max-new-tokens 224
```

```bash
$TRTMC build nvidia/magpie_tts_multilingual_357m -o /tmp/magpie.trtfb --precision fp16

$TRTMC generate-audio /tmp/magpie.trtfb \
  --prompt "A clear short test sentence." \
  --output /tmp/magpie.wav
```

Streaming paths are exposed through `trtmc transcribe --stream` for cache-aware ASR and `trtmc serve-audio` for prompt-driven audio serving. Add `--hf-python /opt/venv/bin/python` only for runtime strategies that still need helper Python code.

## Diffusion and video

```bash
$TRTMC build black-forest-labs/FLUX.2-dev \
  -o /tmp/flux2.trtfb \
  --precision fp16 \
  --image-height 1024 \
  --image-width 1024 \
  --num-inference-steps 28

$TRTMC generate-video /tmp/flux2.trtfb \
  --prompt "A photo of a cat sitting on a windowsill at sunset" \
  --output /tmp/flux2-frames \
  --num-steps 28
```

Image diffusion and video diffusion use separate runtime strategies, but both use the same bundle and C++ runtime entrypoint family.

## Segmentation

```bash
$TRTMC segment /tmp/segformer.trtfb \
  --image tests/assets/test_scene.jpg \
  --output /tmp/mask.png
```

The public API and CLI reserve `IPipeline::detect()` and `trtmc detect`, but
the current model manifests and E2E catalog do not include an object-detection
owner. There is therefore no supported detector bundle to run in this guide.
Treat the command as an API contract for a future model implementation, not as
current support evidence.

## Neural-operator style solve

```bash
$TRTMC solve /tmp/operator.trtfb \
  --branch-input "0.05,0.15,0.30,0.50,0.65,0.80,0.95,1.10,1.18,1.24,1.28,1.31" \
  --trunk-input "2"
```
