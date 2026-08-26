---
title: Serve Local Models
description: Keep text and speech bundles loaded behind local HTTP and Realtime APIs.
---

`trtmc serve` starts a Python API control plane and a fixed group of long-lived
native workers per registered bundle. Each worker loads its TensorRT pipeline
once and owns one execution lane for the process lifetime. The default group
has one replica.

## Install the control-plane dependencies

The native runtime remains C++, while the HTTP and WebSocket control plane
is an optional Python package extra:

```bash
python -m pip install "tensorrt-model-connect[serve]"
```

## Start a local speech and summary server

```bash
export TRTMC_SERVE_TOKEN="replace-with-a-random-local-token"

trtmc serve \
  --transcription-model asr=/models/nemotron-streaming.bundle \
  --chat-model chat=/models/qwen3-0.6b.bundle \
  --default-transcription-model asr \
  --default-chat-model chat \
  --host 127.0.0.1 \
  --port 8000
```

The initial server is loopback-only and rejects every non-loopback bind. Access
logs are disabled by default because browser WebSocket
clients authenticate through an `access_token` query parameter. Other server
logs continue on stderr.

Check readiness and the registered model roles:

```bash
curl http://127.0.0.1:8000/readyz \
  -H "Authorization: Bearer $TRTMC_SERVE_TOKEN"

curl http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer $TRTMC_SERVE_TOKEN"
```

`/healthz` is an unauthenticated, detail-free supervisor probe. It returns 200
while at least one native worker remains usable and 503 after every worker has
failed. `/readyz` remains the whole-registry readiness check and requires the
bearer token when authentication is configured.

## Generate a summary

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $TRTMC_SERVE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "chat",
    "messages": [
      {"role": "user", "content": "Summarize this transcript: ..."}
    ],
    "max_completion_tokens": 160,
    "stream": false
  }'
```

Text generation currently requires `"stream": false`. A request with
`"stream": true` fails explicitly until the native pipeline exposes token
callbacks and cooperative cancellation.

Request-affecting OpenAI options that are not implemented are also rejected
explicitly instead of being silently ignored. Their documented no-op values,
such as `n=1`, remain accepted for client compatibility. The `user` and
`metadata` fields are accepted as non-execution metadata; other unknown fields
with non-null values fail closed.

## Add bounded request lanes

The default one-replica group protects mutable pipeline state. Add an explicit
replica only after the model fits the available GPU memory and a target-hardware
load test shows useful throughput scaling:

```bash
trtmc serve \
  --chat-model chat=/models/qwen.bundle \
  --model-replicas chat=2
```

Each replica is an independent native process and may duplicate model and KV
memory. There is no server-side waiting queue: a request leases one idle
replica, and the server returns HTTP 429 immediately when all replicas are busy.

## Transcribe a WAV file

```bash
curl http://127.0.0.1:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer $TRTMC_SERVE_TOKEN" \
  -F model=asr \
  -F response_format=verbose_json \
  -F file=@meeting.wav
```

The initial file endpoint accepts PCM16 or float32 WAV input because those are
the formats decoded by the native runtime helper. `verbose_json` includes any
timestamp segments returned by the selected pipeline.

## Stream microphone audio

Connect to:

```text
ws://127.0.0.1:8000/v1/realtime?intent=transcription&access_token=TOKEN
```

The client sends OpenAI Realtime-style JSON events:

- `session.update` selects the model, PCM16 sample rate, and language.
- `input_audio_buffer.append` carries base64 little-endian mono PCM16.
- `input_audio_buffer.commit` finishes the current utterance.
- `input_audio_buffer.clear` discards and resets it.

The server returns `session.created`, `session.updated`, transcription
`delta`/`completed`/`failed` events, and structured `error` events. True partial
results require a bundle whose pipeline implements
`create_transcription_stream()`; other ASR bundles remain usable through the
offline endpoint.

## Current boundaries

- One native worker is one serial execution lane. Explicit replicas provide
  fixed parallel lanes; they are not continuous batching.
- Failed replicas are removed from scheduling but are not restarted. An
  authenticated `/readyz` response exposes `replicas`, `ready_replicas`, and
  `idle_replicas`; the model remains available while at least one lane is
  healthy.
- Text request cancellation, native token callbacks, tool calling, logprobs,
  and exact prompt-token counts are not exposed yet.
- A realtime ASR session holds one replica until commit,
  clear, failure cleanup, or disconnect.
- The Python control plane never receives TensorRT objects; bundle execution
  remains in the native worker process.

{/* Collaborative review anchor: batch 2. */}
