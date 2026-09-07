---
title: Multimodal & Speech
description: Vision-language, transcription, audio generation, and speech-session lookup.
---

## Vision-language generation

```bash
trtmc run vision-language.bundle \
  --runtime-root /opt/trtmc/lib \
  --image input.jpg \
  --prompt "Describe this image in one sentence." \
  --max-new-tokens 48
```

Image preprocessing, cache policy, tokenizer/template behavior, and supported
request values belong to the selected family.

## Transcription

```bash
trtmc transcribe speech-to-text.bundle \
  --runtime-root /opt/trtmc/lib \
  --input input.wav \
  --beam-size 1 \
  --source-language en \
  --target-language en \
  --translate false
```

Use `transcribe-batch` with repeated `--input` values. Use
`transcribe-streaming` with chunk samples and left/right attention contexts for
a family implementing the streaming interface. Offline punctuation,
timestamps, segmentation, language, and beam options remain Task/family
contracts.

## Audio and speech sessions

```bash
trtmc generate-audio text-to-audio.bundle \
  --runtime-root /opt/trtmc/lib \
  --prompt "A clear short test sentence." \
  --output output.wav

trtmc speak speech-to-speech.bundle \
  --runtime-root /opt/trtmc/lib \
  --input input.wav \
  --output response.wav
```

Nemotron VoiceChat declares `task=speech_session`. Use `speech-session` for a
finite WAV/session event run, or the repository's native full-duplex microphone
application for a persistent local session:

```bash
trtmc speech-session nemotron-voicechat.bundle \
  --runtime-root /opt/trtmc/lib \
  --input input.wav \
  --output response.wav \
  --timeout-ms 30000
```

The example under `examples/models/nemotron_voicechat/full_duplex/` continuously
captures and plays ALSA audio without a server, network port, Python runtime,
or checkpoint in the image. Its README documents the exact Docker, GPU,
bundle, and audio-device contract.

Use exact checkpoints from [Models & Recipes](../models-recipes/overview.md).
