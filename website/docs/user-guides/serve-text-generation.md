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
| `POST /v1/responses` | Stateless, nonstreaming text: one `input` string or one user message with string/`input_text` content; optional string `instructions`; `model` is required. |
| `POST /v1/chat/completions` | One text `user` message and optional preceding `system` message. Native `TextConversation` models also accept nonstreaming text history (up to 64 messages, final role `user`). Vision-language bundles accept one user message with text and one embedded image. |
| `GET /v1/models` | Configured models and their native capabilities. |
| `GET /health/live` | Control-plane process liveness. |
| `GET /health/ready` | Worker readiness and admission state. |
| `GET /metrics` | Prometheus text metrics for admission and inference. |

Completions and chat accept `temperature`, `top_p`, `min_p`, `top_k`, `seed`,
`enable_thinking`, `n`, `stream`, and `stream_options.include_usage`.
Completions accept `max_tokens`;
chat accepts either `max_tokens` or `max_completion_tokens`. The MVP
requires `n=1`. It rejects unknown fields, prompt arrays, unsupported media content,
unsupported history, stop sequences, tools, and log probabilities instead
of silently ignoring them.

For text history, the current Qwen family implements the existing
`TextConversation` Task for its supported ChatML templates. Send user/assistant
messages in their original order, optionally preceded by a system message,
and end with a user message. History requests are nonstreaming and must fit
the complete context limit; unsupported models return an explicit error.

Responses supports a [subset of the Responses wire protocol](https://developers.openai.com/api/reference/resources/responses/methods/create):
one text user input and optional string `instructions`, `max_output_tokens`
(at least 16), `temperature`, `top_p`, `min_p`, `top_k`, `seed`, and
`enable_thinking`. `store` defaults to
`false`. `store=true`, `previous_response_id`, conversations/history, tools,
media, reasoning-specific fields and `stream=true` are rejected. The endpoint
returns one typed assistant message containing native output text. Its
`completed` status indicates that the native Task call completed; native
EOS-versus-limit termination metadata is unavailable. `usage` is `null` because
native input-token counts are unavailable; output counts remain in the server's
request records. Streaming support described below applies to completions/chat.

For example, with the same built bundle and server:

```bash
curl http://127.0.0.1:8000/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","input":"Explain why the sky is blue.","max_output_tokens":32,"temperature":0,"enable_thinking":false,"store":false}'
```

For single-image chat, see [Profile Image Understanding with AIPerf](./profile-vision-with-aiperf.md).
Image requests require a vision-language bundle and accept one embedded RGB PNG/JPEG
plus text. System prompts, streaming, and thinking controls are rejected for this workload.

Streaming responses use OpenAI-compatible SSE and end with `data: [DONE]`.
Single-process Qwen bundles implement the existing `StreamingTextContinuation`
Task and deliver UTF-8 text deltas during decoding. A delta may cover several
tokens when a codepoint spans token boundaries. Families without that Task
retain buffered SSE. `/v1/models` declares `trtmc.streaming` as `incremental`
or `buffered`, and streamed responses carry `X-TRTMC-Streaming`.
Only incremental models support meaningful client TTFT/inter-token measurements.
See [Profile Text with AIPerf](./profile-text-with-aiperf.md).

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

Pi sends text-only content blocks and consumes the SSE response. Run
Pi with `--no-tools`; tool definitions and tool-result messages remain outside
the MVP protocol.

```bash
pi --provider trtmc --model Qwen/Qwen3-0.6B --no-tools
```

Model-specific chat templates, tokenization, sampling, stopping, engine
composition, and validation remain inside the family selected by the bundle.
For migrated bundles, the native worker calls the public header-only C++ Task
wrapper over the stable C ABI. It uses `TextContinuation`, or the primary
`ConditionalTextGeneration` / `TextTranslation` contract for those bundles.
History calls the existing `TextConversation` contract only when the loaded
model advertises it. The family formats and tokenizes the typed messages.
Translation uses the family's bundled language defaults; this HTTP protocol
does not add source/target-language parameters.

The worker reads the Task's declared Config fields and forwards only supplied
values, leaving omitted sampling defaults with the family. A positive declared
`max_new_tokens` default informs the existing HTTP token cap; otherwise the
server retains its 128-token safety default. Unknown fields, wrong types and
family-rejected settings return request errors without retiring the worker.
An HTTP parameter being recognized does not mean every family supports it:
chat templates, system prompts and sampling options must be supported by the
selected Task.

Legacy bundle modes first probe the public Model interface, so a family can
add semantic and streaming Tasks while retaining its existing bundle identity.
Only a family explicitly reporting that it has no SDK Model interface takes
the `ITextGeneration` compatibility path. That legacy case may load twice during
startup; loading is outside request timing. Other SDK load/invocation errors
never retry an older interface. The private protocol is version 2 for native
stream events; the Python frontend also accepts existing version-1 buffered
workers.

Use `--records /path/requests.jsonl` for optional terminal request timing records.
Valid incoming `X-Request-ID` values are preserved; omitted IDs are generated,
and invalid or repeated ID headers return 400. Records include success, error,
timeout, saturation and disconnect outcomes without prompt or generated text.
Nonstreaming native `model_call_ms` measures the public Task call separately
from HTTP and worker transport time. Streaming measures the public Task stream
including relay/backpressure. The optional
[AIPerf text-profile runner](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/apps/aiperf_qual/README.md#text-profiling-with-the-persistent-server)
joins these timings with client measurements.

Responses report completion token counts supplied by the family result. Prompt
token counts remain zero because the generic Task result does not expose them.
The server leaves `finish_reason` null because the Task contract does not yet
distinguish stop causes.

## Capacity, failure, and shutdown

The MVP uses zero queued requests. Saturated replicas return `429` with a
`Retry-After` header. Worker request timeouts terminate that worker rather
than risking protocol desynchronization. A failed lane makes its model
degraded or unavailable. When a client disconnects from an incremental stream,
the server confirms that the cancelled worker has exited and automatically loads
a replacement with the same settings. That replica remains unavailable during
model loading; other healthy replicas can continue serving. A failed replacement
load leaves the replica unavailable without repeatedly retrying startup.

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
