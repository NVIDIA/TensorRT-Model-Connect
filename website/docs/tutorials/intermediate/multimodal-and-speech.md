---
title: Multimodal and Speech Tasks
---

# Multimodal and speech tasks

TRTMC exposes typed commands rather than one generic tensor interface. Each
family implements only the task contracts it owns.

## Vision-language generation

Build the family with the public builder, then run text generation with an
image input:

```bash
python -m tensorrt_model_connect build Qwen/Qwen2.5-VL-3B-Instruct \
  -o qwen-vl.bundle \
  --precision bf16 \
  --max-sequence-length 384

trtmc run qwen-vl.bundle \
  --runtime-root /opt/trtmc/lib \
  --image sample.png \
  --prompt "Describe the image." \
  --max-new-tokens 80
```

Resolution policy, preprocessing, prompt formatting, and the relationship
between vision and decoder engines are family-owned. The shared builder has no
`--set` configuration namespace.

## Offline and streaming transcription

```bash
trtmc transcribe whisper.bundle \
  --runtime-root /opt/trtmc/lib \
  --input recording.wav \
  --max-output-tokens 128

trtmc transcribe-streaming nemotron-asr.bundle \
  --runtime-root /opt/trtmc/lib \
  --input recording.wav \
  --chunk-samples 16000 \
  --max-new-tokens 128
```

Offline transcription uses `--input`, not the removed `--audio` spelling.
Streaming additionally exposes attention-context and language options when the
family supports them.

## Audio generation and speech sessions

```bash
trtmc generate-audio audio.bundle \
  --runtime-root /opt/trtmc/lib \
  --prompt "A calm spoken welcome" \
  --output welcome.wav \
  --seed 1234

trtmc speech-session voicechat.bundle \
  --runtime-root /opt/trtmc/lib \
  --input question.wav \
  --output answer.wav \
  --system-prompt "Answer concisely." \
  --timeout-ms 30000
```

The speech-session command drives the family session interface and records its
event stream. Applications needing incremental duplex control should use the
C++ task/session contracts in `core/runtime/include/trtmc/`; family code owns
the concrete conversation state and audio orchestration.

Inspect every bundle first and use its family manifest for exact inputs and
expected evidence.
