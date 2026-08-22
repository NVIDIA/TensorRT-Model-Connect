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
agent-audio, agent-text, user-transcript, function-call, and lifecycle events.
Sessions retain model state across calls and support model-confirmed barge-in,
multiple user turns, bounded backpressure, finish, cancel, and reset. Separate
optional interfaces provide function results, input commit/clear, response
creation/cancellation, and playback-aware response truncation. Finite WAV
inference uses `ISpeechBatchSessionProvider`, so live turn policy does not
change the model-card `trtmc speak` result. Keeping these capabilities separate
preserves the base `IPipeline` and existing speech-session ABI.

To expose the native session over a local Realtime WebSocket, install the
optional transport dependency and start the bundled worker through the Python
host:

```bash
python -m pip install 'tensorrt-model-connect[realtime]'
export TRTMC_REALTIME_TOKEN='replace-with-a-local-secret'

python -m tensorrt_model_connect.realtime \
  --bundle nemotron-voicechat-11b.bundle \
  --host 127.0.0.1 \
  --port 8765
```

Connect to `ws://127.0.0.1:8765/v1/realtime` with
`Authorization: Bearer $TRTMC_REALTIME_TOKEN`. Audio input and output use mono
24 kHz PCM16 carried as Base64 JSON deltas. The supported event subset includes
session configuration, streaming append, commit/clear, `response.create`,
response cancel/truncate, and function-call output. The host keeps audio,
command, and output queues bounded; it never executes tool calls itself.
The default native worker accepts at most five minutes of input audio per
connection, leaving headroom in VoiceChat's fixed recurrent and TTS caches for
agent output and function steps. Processed-input clear, cancel, and truncate
perform a real model-state rollback; they are synchronous and their replay
cost grows with conversation history.

For an opt-in real-engine lifecycle check, build the standalone model-owned
probe and run it against a prebuilt bundle and the pinned public sample:

```bash
cmake --build build --target test_nemotron_voicechat_native_lifecycle

LD_LIBRARY_PATH="$PWD/build:${LD_LIBRARY_PATH:-}" \
build/test_nemotron_voicechat_native_lifecycle \
  nemotron-voicechat-11b.bundle \
  "$VOICECHAT_SPEECH/examples/speechlm2/sample_audio/sample_general.wav" \
  build build/models response.wav lifecycle-receipt.json
```

The probe covers batch parity, arbitrary chunk boundaries, output before input
completion, model-confirmed mid-response barge-in and same-session recovery,
cancel/reset, exact bounded finish behavior, normal multi-turn conversation,
producer/consumer backpressure, and a real function-call/result continuation.
The model-owned E2E also drives the WebSocket host through non-silent function,
truncate, and cancel traces. These are large-memory-GPU checks rather than
normal source-only unit tests.

Use exact checkpoints from [Model Recipes](../models-recipes/model-recipes.md),
organized by the corresponding Hugging Face multimodal or audio task.
For concept-first labs covering preprocessing, engine components, streaming
state, and typed results, follow [Multimodal and Speech](../tutorials/intermediate/multimodal-and-speech.md)
and [Canary Decoding](../tutorials/intermediate/canary-decoding.md).

{/* Collaborative review anchor: batch 2. */}
