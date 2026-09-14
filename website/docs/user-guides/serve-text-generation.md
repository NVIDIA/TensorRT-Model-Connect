---
title: Serve Text Generation
description: Expose one text-generation bundle through the MVP HTTP server.
---

`trtmc-server` loads one bundle into one process and exposes a deliberately
small, non-streaming subset of the OpenAI Completions and Chat Completions
protocols. The server is an evaluation path, not a production serving system.
It does not depend on Triton or Dynamo.

## Start one endpoint

Build the native `trtmc-server` target alongside the runtime, backend, and
selected family DSO. Then start the endpoint with one command:

```bash
trtmc-server ./qwen3-0.6b.bundle \
  --model-name Qwen/Qwen3-0.6B \
  --host 127.0.0.1 \
  --port 8000
```

By default, the loader finds family and backend DSOs beside the already-loaded
`libtrtmc_runtime` shared library. This works for the standard build tree and
the relocatable installed layout. Use `--runtime-root DIR` only to override
that directory for development, multi-version testing, or a custom deployment.

Startup is synchronous: the bundle and its family/backend DSOs must load before
the server reports `server_ready`. A load or bind failure exits nonzero and the
readiness endpoint never reports ready.

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

Or use the OpenAI Python client for the initial chat shape:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
response = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[{"role": "user", "content": "Say hello in five words."}],
    max_completion_tokens=16,
    temperature=0,
)
print(response.choices[0].message.content)
```

The server does not authenticate requests. Keep the default loopback bind for
local evaluation; putting it on an untrusted network requires a separate
authentication and TLS proxy.

## Supported protocol

| Route | Supported behavior |
| --- | --- |
| `POST /v1/completions` | One string `prompt`; `model` is required. |
| `POST /v1/chat/completions` | One text `user` message and optional preceding `system` message. |
| `GET /v1/models` | The single configured model. |
| `GET /health/live` | Process liveness. |
| `GET /health/ready` | `200` while accepting work; `503` while draining. |
| `GET /metrics` | Prometheus text metrics for admission, queueing, and inference. |

Both generation routes accept `temperature`, `top_p`, `seed`, `n`, and
`stream`. Completions accept `max_tokens`; chat accepts either `max_tokens` or
`max_completion_tokens`. The MVP requires `n=1` and `stream=false`. It rejects
unknown fields, prompt arrays, multi-turn chat, structured message content,
tools, log probabilities, stop strings, and streaming instead of silently
ignoring them. Responses omit token usage because the generic Task result does
not report prompt token counts, and `finish_reason` is `null` because the Task
contract does not yet distinguish stop causes.

Sampling uses the Task contract's unlimited `top_k` mode so `temperature` and
`top_p` retain their expected effect. When `seed` is omitted, the server
supplies a non-negative per-request seed; provide an explicit seed for
repeatable output. Negative seeds are rejected because the Task contract uses
them as internal sentinel values.

Model-specific chat templates, tokenization, sampling, stopping, engine
composition, and validation remain inside the family selected by the bundle.
The server only validates and maps the public request into
`ITextGeneration::generate`.

## Capacity and shutdown

The server owns one Task and invokes it from one worker thread. The defaults
allow 16 waiting requests and 1 MiB of queued request bodies; the active request
does not count against the waiting-request limit. A full queue returns `429`
with error code `queue_full`. Requests arriving after shutdown begins return
`503` with `server_draining`.

Use these controls when exploring capacity:

```text
--queue-capacity COUNT
--max-queued-bytes BYTES
--max-body-bytes BYTES
--max-prompt-bytes BYTES
--max-new-tokens COUNT
--shutdown-grace-ms MS
```

`SIGINT` and `SIGTERM` first make readiness fail and stop new admission. Work
already waiting may start until the grace deadline; work not started by then
receives `503`. The currently executing family call is allowed to finish
because the Task API has no cancellation contract.

Errors use an OpenAI-style `error` object and successful inference responses
include an `X-Request-ID` header. Logs are JSON lines and do not include prompt
or generated text. `/metrics` exposes queue depth/bytes, active requests,
generation request totals by route/status, wall-clock queue/inference duration,
and family-reported setup/prefill/decode duration sums and counts.

## Qualify several text families

Serving is family-agnostic, but every family still owns its correctness claim.
For an initial cross-family check, build one bundle at a time from declared
recipes such as Qwen3 0.6B, Gemma 2 2B, Phi-3 Mini, and a Llama-family recipe
listed in [Supported Models](../models-recipes/overview.md). Start a fresh
server process for each bundle and record:

1. startup and readiness;
2. one deterministic completion request;
3. one deterministic single-turn chat request where the family supports a chat
   template;
4. an invalid request (`stream=true`) returning `400`;
5. an overload run returning `429` without concurrent Task calls;
6. queue and inference metrics after the run;
7. `SIGTERM` drain and exit behavior.

Compare generated text through that family's existing validation recipe. A
successful HTTP exchange proves transport integration only; it does not replace
family-owned numerical or semantic validation.
