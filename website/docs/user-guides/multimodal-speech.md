---
title: Multimodal & Speech
description: Quick task and configuration lookup for vision-language, transcription, audio generation, and speech-to-speech bundles.
---

## Vision-language generation

```bash
trtmc run vision-language.bundle \
  --image input.jpg \
  --prompt "Describe this image in one sentence." \
  --max-new-tokens 48
```

Image shape, dynamic resolution, preprocessing metadata, decoder layout, and
LoRA support are family-owned. Use registered Qwen-VL configuration fields
only with a Qwen-VL build that declares them.

## Speech recognition

```bash
trtmc transcribe speech-to-text.bundle \
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
trtmc generate-audio text-to-audio.bundle \
  --prompt "A clear short test sentence." \
  --output output.wav

trtmc speak speech-to-speech.bundle \
  --audio-in input.wav \
  --audio-out response.wav
```

## NVIDIA NemotronLabs VoiceChat native TensorRT

The `nemotron_voicechat` family builds the complete public checkpoint with the
TensorRT Native API. The resulting bundle contains cache-aware FastConformer
perception, RNN-T transcription, the hybrid Nemotron-H thinker, EAR-TTS, and
the RVQ codec. Runtime inference is C++/TensorRT only; it does not launch
Python, NeMo, PyTorch, ONNX, or a subprocess.

```bash
export VOICECHAT_CHECKPOINT=/path/to/NVIDIA-NemotronLabs-VoiceChat-11B
export VOICECHAT_SPEECH=/path/to/pinned/Speech

trtmc build "$VOICECHAT_CHECKPOINT" \
  --output nemotron-voicechat-11b.bundle \
  --precision fp32 \
  --max-cache-length 8192

trtmc speak nemotron-voicechat-11b.bundle \
  --audio-in "$VOICECHAT_SPEECH/examples/speechlm2/sample_audio/sample_general.wav" \
  --audio-out model_card_sample_general_output_native.wav \
  --seed 0
```

The validated native run emits 345,744 mono float32 samples at 22.05 kHz: 196
synchronized 80 ms codec frames. Its agent-text channel reproduces the
model-card response ending in “something called Rayleigh scattering.” Native
sampling is deterministic for a given seed, but its C++ random generator is
not bitwise identical to Torch CUDA Philox, so the native and reference WAV
files are not expected to have the same checksum.

For live use, cast the loaded pipeline to the optional C++
`ISpeechSessionProvider` declared in `trtmc/speech_session.h`, then create a
persistent `ISpeechSession`. Append arbitrary mono chunks and drain interleaved
agent-audio, agent-text, user-transcript, and lifecycle events. Sessions retain
model state across calls and support barge-in, finish, cancel, and reset.
Keeping the provider separate preserves the base
`IPipeline` ABI for existing optimized-runtime integrations.

For an opt-in real-engine lifecycle check, build the standalone model-owned
probe and run it against a prebuilt bundle and the pinned public sample:

```bash
c++ -std=c++17 -O2 -pthread -Iinclude -Isrc \
  tests/e2e/models/nemotron_voicechat/native_lifecycle_probe.cpp \
  -Lbuild -ltrtmc_core -Wl,-rpath,"$PWD/build" \
  -o /tmp/voicechat_native_lifecycle_probe

LD_LIBRARY_PATH="$PWD/build:${LD_LIBRARY_PATH:-}" \
/tmp/voicechat_native_lifecycle_probe \
  nemotron-voicechat-11b.bundle \
  "$VOICECHAT_SPEECH/examples/speechlm2/sample_audio/sample_general.wav" \
  build build/models response.wav lifecycle-receipt.json
```

The probe covers arbitrary chunk boundaries, output before input completion,
mid-response barge-in and same-session recovery, cancel, reset, and bounded
finish behavior. It is a large-memory-GPU local check rather than a normal
source-only unit test.

Use exact checkpoints from [Model Recipes](../models-recipes/model-recipes.md),
organized by the corresponding Hugging Face multimodal or audio task.
For concept-first labs covering preprocessing, engine components, streaming
state, and typed results, follow [Multimodal and Speech](../tutorials/intermediate/multimodal-and-speech.md)
and [Canary Decoding](../tutorials/intermediate/canary-decoding.md).

{/* Collaborative review anchor: batch 2. */}
