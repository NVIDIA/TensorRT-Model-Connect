---
title: Multimodal & Speech
description: Quick task lookup for vision-language, transcription, audio generation, and speech-to-speech bundles.
---

Set the installed native runtime directory once:

```bash
TRTMC_RUNTIME_ROOT=/opt/trtmc/lib
```

## Vision-language generation

```bash
trtmc run vision-language.bundle \
  --runtime-root "$TRTMC_RUNTIME_ROOT" \
  --image input.jpg \
  --prompt "Describe this image in one sentence." \
  --max-new-tokens 48
```

Image shape, preprocessing, decoder layout, and output decoding belong to the
selected family.

## Speech recognition

```bash
trtmc transcribe speech-to-text.bundle \
  --runtime-root "$TRTMC_RUNTIME_ROOT" \
  --input input.wav \
  --beam-size 1 \
  --source-language en \
  --target-language en \
  --max-output-tokens 224
```

Use `transcribe-batch` or `transcribe-streaming` only when the bundle's family
implements the corresponding abstract interface. Language support, maximum
duration, punctuation, timestamps, and streaming context are family-owned.

## Audio generation and speech-to-speech

```bash
trtmc generate-audio text-to-audio.bundle \
  --runtime-root "$TRTMC_RUNTIME_ROOT" \
  --prompt "A clear short test sentence." \
  --output output.wav

trtmc speak speech-to-speech.bundle \
  --runtime-root "$TRTMC_RUNTIME_ROOT" \
  --input input.wav \
  --output response.wav
```

## Nemotron VoiceChat

The `nemotron_voicechat` family owns its checkpoint build, native runtime,
speech sessions, dependencies, and validation. Build and execute it through
the same public surfaces as every other family:

```bash
VOICECHAT_CHECKPOINT=/path/to/NVIDIA-NemotronLabs-VoiceChat-11B
VOICECHAT_SPEECH=/path/to/pinned/Speech

python -m pip install -r families/nemotron_voicechat/requirements.txt

python -m tensorrt_model_connect build "$VOICECHAT_CHECKPOINT" \
  --precision fp32 \
  --max-sequence-length 8192 \
  --output nemotron-voicechat-11b.bundle

trtmc speak nemotron-voicechat-11b.bundle \
  --runtime-root "$TRTMC_RUNTIME_ROOT" \
  --input "$VOICECHAT_SPEECH/examples/speechlm2/sample_audio/sample_general.wav" \
  --output response.wav \
  --seed 0
```

Runtime execution is native C++/TensorRT. It does not import the family's
Python build dependencies.

### Persistent speech sessions

Applications can load the bundle through the public loader and request the
optional session interface implemented by this family:

```cpp
auto task = trtmc::load_task("nemotron-voicechat-11b.bundle", "/opt/trtmc/lib");
auto* provider = dynamic_cast<trtmc::ISpeechSessionProvider*>(task.get());
if (provider == nullptr)
    throw std::runtime_error("family does not implement speech sessions");

trtmc::SpeechSessionConfig config;
config.input_sample_rate = microphone_sample_rate;
auto session = provider->create_speech_session(config);
for (const auto& chunk : microphone_chunks) {
    session->append_audio(chunk.data(), static_cast<int32_t>(chunk.size()));
    for (auto& event : session->take_events())
        consume(event);
}
session->finish_input();
```

The [full-duplex microphone example](https://github.com/NVIDIA/TensorRT-Model-Connect/tree/main/examples/models/nemotron_voicechat/full_duplex)
is a one-way application of these public APIs. The family and shared core do
not depend on the example.

Use exact checkpoints from [Model Recipes](../models-recipes/model-recipes.md).
