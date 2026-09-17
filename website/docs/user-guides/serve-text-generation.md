---
title: Serve Text Generation
description: Expose text-generation bundles through the local hybrid server.
---

`trtmc-server` exposes a deliberately small subset of the
OpenAI Completions and Chat Completions protocols. It is a local evaluation
server, not a production or distributed serving system.

The public HTTP control plane uses FastAPI and Uvicorn. Each configured model
replica runs in a persistent C++ worker process that loads one bundle through
the public Model Connect Task API. This keeps model execution native and
isolates worker failures without making Model Connect depend on the server.

Install the optional control-plane dependencies:

```bash
pip install 'tensorrt-model-connect[serve]'
```

## Start one endpoint

Build `trtmc-server` alongside the runtime, backend, and selected family DSO,
then start a model with one command:

```bash
trtmc-server ./qwen3-0.6b.bundle \
  --model-name Qwen/Qwen3-0.6B \
  --host 127.0.0.1 \
  --port 8000
```

The worker normally finds family and backend DSOs beside the loaded
`libtrtmc_runtime`. Use `--runtime-root DIR` only for development,
multi-version testing, or custom deployment layouts.

Startup is synchronous. All configured workers must load and complete their
private readiness handshake before the HTTP readiness endpoint succeeds. A
partial startup failure closes workers that already started and exits nonzero.

Send a completion:

```bash
curl http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "prompt": "What is the capital of France? Answer in one word.",
    "max_tokens": 10,
    "temperature": 0
  }'
```

Or use the OpenAI Python client:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="EMPTY",  # Required by openai-python; ignored by this local server.
)
response = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[{"role": "user", "content": "Say hello in five words."}],
    max_completion_tokens=16,
    temperature=0,
)
print(response.choices[0].message.content)
```

The MVP has no built-in authentication and is intended for trusted local
evaluation or an isolated deployment. If remote access is needed later, place
it behind an authenticated TLS reverse proxy; do not expose the server directly.

## Serve several text models

Register multiple independently owned bundles with repeatable `--model`
arguments:

```bash
trtmc-server \
  --model Qwen/Qwen3-0.6B=qwen3.bundle \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0=tinyllama.bundle \
  --replicas 1
```

Each replica is a persistent process and a serialized execution lane. Different
models or replicas can execute concurrently. If every replica for a model is
busy, the server immediately returns `429 server_busy`; the MVP has no hidden
waiting queue. Add replicas only when the model and available device memory
support that placement.

## Supported protocol

| Route | Supported behavior |
| --- | --- |
| `POST /v1/completions` | One string `prompt`; `model` is required. |
| `POST /v1/chat/completions` | One text `user` message and optional preceding `system` message; content may be a string or text-only content blocks. |
| `GET /v1/models` | Configured text models. |
| `GET /health/live` | Control-plane process liveness. |
| `GET /health/ready` | Worker readiness and admission state. |
| `GET /metrics` | Prometheus text metrics for admission and inference. |

Generation accepts `temperature`, `top_p`, `min_p`, `top_k`, `seed`,
`enable_thinking`, `n`, `stream`, and `stream_options.include_usage`.
Completions accept `max_tokens`;
chat accepts either `max_tokens` or `max_completion_tokens`. The MVP
requires `n=1`. It rejects unknown fields, prompt arrays, non-text content
blocks, multi-turn chat, stop sequences, tools, and log probabilities instead
of silently ignoring them.

Streaming responses use the OpenAI-compatible server-sent event framing and
terminate with `data: [DONE]`. The native text task currently returns a complete
generation, so the MVP buffers inference before emitting the content chunk;
streaming provides client compatibility but does not yet reduce time to first
token. Incremental token delivery requires a future extension to the family and
runtime text-generation contract.

For example, a Pi custom provider can use the local endpoint without an
authentication header:

```json
{
  "providers": {
    "trtmc": {
      "baseUrl": "http://127.0.0.1:8000/v1",
      "api": "openai-completions",
      "apiKey": "EMPTY",
      "authHeader": false,
      "compat": {
        "supportsStore": false,
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": false,
        "supportsFinishReason": false,
        "maxTokensField": "max_tokens"
      },
      "models": [{"id": "Qwen/Qwen3-0.6B"}]
    }
  }
}
```

Pi sends text-only content blocks and consumes the buffered SSE response. Run
Pi with `--no-tools`; tool definitions and tool-result messages remain outside
the MVP protocol.

```bash
pi --provider trtmc --model Qwen/Qwen3-0.6B --no-tools
```

Model-specific chat templates, tokenization, sampling, stopping, engine
composition, and validation remain inside the family selected by the bundle.
The server only validates the transport envelope and asks the native worker to
call `ITextGeneration::generate`.

Responses report completion token counts supplied by the family result. Prompt
token counts remain zero because the generic Task result does not expose them.
The server leaves `finish_reason` null because the Task contract does not yet
distinguish stop causes.

## Capacity, failure, and shutdown

The MVP uses zero queued requests. Saturated replicas return `429` with a
`Retry-After` header. Worker request timeouts terminate that worker rather
than risking protocol desynchronization. A failed lane makes its model
degraded or unavailable; the MVP does not automatically restart workers.

Uvicorn stops new HTTP admission during shutdown and lets active requests
finish before application lifespan cleanup closes each worker. The worker first
receives a protocol shutdown request, followed by deterministic process
termination if it does not exit within the grace period.

Logs do not contain prompts or generated text. `/metrics` exposes readiness,
active and busy replicas, request totals, admission duration, inference
duration, and family-reported setup, prefill, and decode durations.

## Qualification boundary

Serving is family-agnostic, but every family owns its correctness claim. For an
initial cross-family check, start fresh workers for several declared text model
recipes and record startup, deterministic completion and chat responses,
overload behavior, metrics, and shutdown.

Compare generated text through each family's validation recipe. A successful
HTTP exchange proves transport integration only; it does not replace
family-owned numerical or semantic validation.
