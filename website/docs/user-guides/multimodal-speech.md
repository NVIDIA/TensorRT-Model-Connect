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

For local streaming, cast the loaded pipeline to the optional C++
`ISpeechSessionProvider` declared in `trtmc/speech_session.h` and feed mono
float32 chunks directly:

```cpp
trtmc::SpeechSessionConfig config;
config.input_sample_rate = microphone_sample_rate;
auto* provider = dynamic_cast<trtmc::ISpeechSessionProvider*>(pipeline.get());
if (provider == nullptr)
    throw std::runtime_error("pipeline does not support speech streaming");
auto session = provider->create_speech_session(config);
for (const auto& chunk : microphone_chunks) {
    session->append_audio(chunk.data(), static_cast<int32_t>(chunk.size()));
    for (auto& event : session->take_events())
        consume(event);
}
session->finish_input();
bool finished = false;
while (!finished) {
    for (auto& event : session->wait_events(-1)) {
        finished |= event.kind == trtmc::SpeechSessionEventKind::kInputFinished;
        consume(event);
    }
}
```

`wait_events()` can drain agent audio/text and lifecycle events while the next
input chunk is being captured. The persistent session retains conversation
state and supports barge-in, multiple turns, bounded backpressure, cancel, and
reset. Cast the session to `ISpeechRealtimeControl` for explicit input
commit/clear and response create/cancel/truncate; tool-enabled sessions expose
`ISpeechToolSession`. Finite WAV inference uses `ISpeechBatchSessionProvider`,
so this live path does not change the model-card `trtmc speak` result.

### Full-duplex microphone example

The repository includes a local Linux ALSA
[microphone application](https://github.com/NVIDIA/TensorRT-Model-Connect/tree/main/examples/models/nemotron_voicechat/full_duplex).
Build its image once from the repository root; the example README documents
the pinned x86_64 override:

```bash
docker build --platform linux/arm64 \
  --file examples/models/nemotron_voicechat/full_duplex/Dockerfile \
  --tag trtmc-voicechat-full-duplex:local \
  .
```

After that build, start each conversation with one offline `docker run`. The
bundle remains a read-only host file and is not copied into the image:

```bash
docker run --rm --interactive --tty \
  --network none \
  --gpus 'device=0' \
  --device /dev/snd:/dev/snd \
  --mount "type=bind,src=$PWD/nemotron-voicechat-11b.bundle,dst=/models/model.bundle,readonly" \
  trtmc-voicechat-full-duplex:local
```

Use a headset or hardware acoustic echo cancellation so speaker output does
not look like a user interruption. See the example README for explicit ALSA
devices, x86_64 builds, and desktop audio boundaries.

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
These are large-memory-GPU checks rather than normal source-only unit tests.

Use exact checkpoints from [Model Recipes](../models-recipes/model-recipes.md),
organized by the corresponding Hugging Face multimodal or audio task.
For concept-first labs covering preprocessing, engine components, streaming
state, and typed results, follow [Multimodal and Speech](../tutorials/intermediate/multimodal-and-speech.md)
and [Canary Decoding](../tutorials/intermediate/canary-decoding.md).

{/* Collaborative review anchor: batch 2. */}
