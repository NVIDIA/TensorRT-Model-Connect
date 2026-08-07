---
title: Multimodal & Speech
description: Quick task and configuration lookup for vision-language, transcription, audio generation, and speech-to-speech bundles.
---

## Vision-language generation

```bash
trtmc run vision-language.trtfb \
  --image input.jpg \
  --prompt "Describe this image in one sentence." \
  --max-new-tokens 48
```

Image shape, dynamic resolution, preprocessing metadata, decoder layout, and
LoRA support are family-owned. Use registered Qwen-VL configuration fields
only with a Qwen-VL build that declares them.

## Speech recognition

```bash
trtmc transcribe speech-to-text.trtfb \
  --audio input.wav \
  --beam-size 1 \
  --source-language en \
  --target-language en \
  --task transcribe
```

Add `--stream` and the model-supported chunk/context controls for streaming
ASR. Beam size, language pairs, punctuation, timestamps, and maximum input
duration remain model-owned constraints.

## Audio generation and speech-to-speech

```bash
trtmc generate-audio text-to-audio.trtfb \
  --prompt "A clear short test sentence." \
  --output output.wav

trtmc speak speech-to-speech.trtfb \
  --audio-in input.wav \
  --audio-out response.wav
```

Use exact checkpoints from [Model Recipes](../models-recipes/model-recipes.md),
organized by the corresponding Hugging Face multimodal or audio task.
For concept-first labs covering preprocessing, engine components, streaming
state, and typed results, follow [Multimodal and Speech](../tutorials/intermediate/multimodal-and-speech.md)
and [Canary Decoding](../tutorials/intermediate/canary-decoding.md).
