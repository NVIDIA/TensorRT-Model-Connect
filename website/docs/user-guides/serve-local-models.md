---
title: Serve Local Models
description: Keep text and speech bundles loaded behind local HTTP and Realtime APIs.
---

`trtmc serve` starts an optional Python HTTP/WebSocket control plane and a
fixed group of native worker processes. Each worker loads one bundle through
the public runtime loader and owns one serial execution lane for its lifetime.

## Install the optional control plane

```bash
python -m pip install "tensorrt-model-connect[serve]"
```

The native runtime remains C++. The Python package owns transport, validation,
authentication, and worker lifecycle only; TensorRT objects stay in the native
workers.

## Start a local server

```bash
export TRTMC_SERVE_TOKEN="replace-with-a-random-local-token"

trtmc serve \
  --runtime-root /opt/trtmc/lib \
  --chat-model chat=/models/qwen.bundle \
  --transcription-model asr=/models/whisper.bundle \
  --default-chat-model chat \
  --default-transcription-model asr \
  --host 127.0.0.1 \
  --port 8000
```

A non-empty token is required. Prefer `TRTMC_SERVE_TOKEN` to `--api-key` so the
credential does not appear in the process argument list.

The initial server accepts only loopback IP literals such as `127.0.0.1` and
`::1`. Hostnames such as `localhost` and every non-loopback bind are rejected.
Access logs are disabled by default because browser WebSocket clients may use
an `access_token` query parameter. Other diagnostics continue on stderr with
transport credentials redacted.

Check process health, whole-registry readiness, and registered models:

```bash
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/readyz \
  -H "Authorization: Bearer $TRTMC_SERVE_TOKEN"
curl http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer $TRTMC_SERVE_TOKEN"
```

`/healthz` is unauthenticated and detail-free. It remains healthy while at
least one native worker is usable. Authenticated `/readyz` reports registry and
replica state and fails when any configured model has no ready lane.

## Generate text

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $TRTMC_SERVE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "chat",
    "messages": [{"role": "user", "content": "Summarize this transcript."}],
    "max_completion_tokens": 160,
    "stream": false
  }'
```

The initial endpoint accepts exactly one text-only `user` message. Multi-turn
messages, stop sequences, text streaming, tool calling, log probabilities,
native token callbacks, and cooperative cancellation are not implemented.
Unsupported execution options fail explicitly rather than being silently
ignored.

## Transcribe a WAV file

```bash
curl http://127.0.0.1:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer $TRTMC_SERVE_TOKEN" \
  -F model=asr \
  -F response_format=verbose_json \
  -F file=@meeting.wav
```

The endpoint accepts PCM16 and IEEE float32 WAV data supported by the native
audio adapter. Oversized request bodies are rejected before multipart parsing
or temporary-file spooling.
`verbose_json` includes timestamp segments when the selected Task returns
them.

## Use Realtime transcription

Connect to:

```text
ws://127.0.0.1:8000/v1/realtime?intent=transcription&access_token=TOKEN
```

The client sends `session.update`, `input_audio_buffer.append`,
`input_audio_buffer.commit`, and `input_audio_buffer.clear`. The server returns
session, transcription, and structured error events. True partial results
require native streaming support. The `--require-streaming-transcription MODEL`
option requires every configured replica for that model to pass the native
streaming startup probe. Other speech bundles remain usable through the offline
endpoint without that requirement.

## Configure bounded concurrency

The default is one worker lane per model. Add replicas only after confirming
that the model copies fit in available GPU memory:

```bash
trtmc serve \
  --runtime-root /opt/trtmc/lib \
  --chat-model chat=/models/qwen.bundle \
  --model-replicas chat=2
```

Each replica is a separate native process and may duplicate model and KV-cache
memory. There is no server-side waiting queue: a request atomically leases one
idle lane, and the server returns HTTP 429 when every lane is busy.

Set a replica count above `1` only for a bundle that can be loaded independently
in each single-process worker. MPI/NCCL distributed bundles are not supported
by `trtmc serve`.

To use multiple GPUs, run one independent single-process server instance per
GPU and pin each process before startup, for example with
`CUDA_VISIBLE_DEVICES=0` and `CUDA_VISIBLE_DEVICES=1`. Each instance needs a
distinct port; any routing across instances remains external to this server.

## Scaling and failure boundary

This release intentionally implements one simple placement domain:

- replica counts and model assignments are fixed at process startup;
- one replica handles one request or Realtime session at a time;
- failed replicas are removed from scheduling and are not restarted;
- a model is degraded while some, but not all, configured replicas are ready;
- Realtime sessions hold one lane until commit, clear, failure cleanup, or
  disconnect;
- there is no continuous batching, dynamic model loading, cluster membership,
  cross-host routing, autoscaling, or rolling replacement.

Operational restart and multi-process placement belong to an external
supervisor. The loopback-only bind means this initial server is for local
applications; it is not a network-facing multi-node serving tier.
