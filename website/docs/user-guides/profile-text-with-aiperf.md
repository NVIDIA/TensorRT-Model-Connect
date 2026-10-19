---
title: Profile Text with AIPerf
description: Benchmark an already-built TRTMC text model with the pip-installed AIPerf client.
---

AIPerf sends requests to `trtmc-server` and reports latency and throughput.
The server loads your TRTMC bundle and runs the model on the GPU.

For sentence-to-vector workloads, see [Profile Embeddings with AIPerf](./profile-embeddings-with-aiperf.md).
For query/passage scoring, see [Profile Reranking with AIPerf](./profile-reranking-with-aiperf.md).

This example uses **Qwen3-0.6B**. Build the runtime and server using
[Build from Source](../getting-started/source-build.md), then build the model
using [Build a Bundle](./build-a-bundle.md). The commands below assume your
bundle is `qwen3-0.6b.bundle` and `trtmc-server` is on `PATH`.
Use your server binary's path instead if needed, for example `./build/trtmc-server`.

Run from the repository root in your prepared GPU environment. If you use
Docker, run both terminals in the same GPU container.

## 1. Install the client

Install the server dependencies in your TRTMC environment, and AIPerf in a
separate environment:

```bash
python -m pip install -e '.[serve]' -C py-only=true
python -m venv .venv-aiperf
.venv-aiperf/bin/pip install aiperf==0.13.0
```

This installs the released package from PyPI, as in the
[AIPerf quick start](https://pypi.org/project/aiperf/0.13.0/).
No AIPerf repository is needed. Plain `pip install aiperf` also installs from
PyPI; the version above is the release tested with this integration.

## 2. Start Qwen (terminal 1)

```bash
trtmc-server ./qwen3-0.6b.bundle --model-name Qwen/Qwen3-0.6B
```

Leave it running. Wait for `Uvicorn running on http://127.0.0.1:8000` before
starting the benchmark. The model loads once and stays loaded.

## 3. Run the benchmark (terminal 2)

```bash
.venv-aiperf/bin/aiperf profile \
  --url http://127.0.0.1:8000 \
  --model Qwen/Qwen3-0.6B \
  --endpoint-type completions \
  --tokenizer Qwen/Qwen3-0.6B \
  --synthetic-input-tokens-mean 32 --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 16 --output-tokens-stddev 0 \
  --concurrency 1 --warmup-request-count 10 --request-count 100 \
  --extra-inputs temperature:0 \
  --ui-type simple --artifact-dir artifacts/aiperf-qwen
```

This sends raw-text requests to `/v1/completions`: 10 warmup requests, then
100 measured requests, one at a time. Each prompt is approximately 32 tokens;
generation is capped at 16 tokens and can stop earlier at EOS. Keep prompt and
output lengths within your bundle's compiled context limit.

AIPerf prints a results table and saves
`artifacts/aiperf-qwen/profile_export_aiperf.json` and
`profile_export_aiperf.csv`. Check the successful request count first, then:

- **Request Latency (ms), p50/p95:** median and tail response time.
- **Request Throughput (requests/sec):** completed requests per second.

Use a new `--artifact-dir` for each run. Press **Ctrl+C in terminal 1** when
finished to stop the server and release GPU memory.

For Qwen chat, change `--endpoint-type completions` to `--endpoint-type chat`
and use `--extra-inputs temperature:0 enable_thinking:false`. Add `--streaming`
to measure time to first token and inter-token latency. See
[Serve Text Generation](./serve-text-generation.md#supported-protocol) for
supported requests and streaming capabilities. Keep concurrency at one with
one replica; excess simultaneous requests can receive `429`.

## Run another already-built model

Use the same three steps. Change only the bundle path, API model name, and
matching tokenizer. For example, with a built TinyLlama bundle:

```bash
# Terminal 1: stop the previous server first.
trtmc-server ./tinyllama.bundle --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

In terminal 2, use the same benchmark settings with TinyLlama's model name
and tokenizer:

```bash
.venv-aiperf/bin/aiperf profile \
  --url http://127.0.0.1:8000 \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --endpoint-type completions \
  --tokenizer TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --synthetic-input-tokens-mean 32 --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 16 --output-tokens-stddev 0 \
  --concurrency 1 --warmup-request-count 10 --request-count 100 \
  --extra-inputs temperature:0 \
  --ui-type simple --artifact-dir artifacts/aiperf-tinyllama
```

The server's `--model-name` must match AIPerf's `--model`. Use the tokenizer from the checkpoint
that produced the bundle; add `--tokenizer-revision MODEL_COMMIT` when you
built from a pinned Hugging Face revision. The runtime must include that
model's family and backend libraries. Start with nonstreaming `completions`;
chat and incremental streaming depend on the family's public Tasks.

Run the model's family correctness checks before interpreting performance.
For automatic correctness checks, server startup/shutdown, workload sweeps,
and joined client/native timing reports, use the optional
[text-profile runner](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/apps/aiperf_qual/README.md#text-profiling-with-the-persistent-server).

## Benchmark actual prompts

Put one original prompt in each JSONL record:

```json
{"text":"Explain why the sky is blue."}
{"text":"Write a short welcome message for a new teammate."}
```

Save this as `prompts.jsonl`. With the same server running, use
`--input-file prompts.jsonl --custom-dataset-type single_turn` in the benchmark
command above and remove both `--synthetic-input-tokens-*` options. AIPerf sends
the supplied text. Choose prompts that fit the built model's context limit
together with the requested output.

For a reproducible selection of real ShareGPT prompts, the optional
[text-profile runner](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/apps/aiperf_qual/README.md#pinned-real-prompts)
provides a pinned Qwen3-0.6B workload. It preserves selected strings, records
over-limit rejections, and verifies HTTP input/output against native dispatch
records. It uses the same published AIPerf package.

## Benchmark a conversation

With the current Qwen server and runtime, save this as `conversations.jsonl`:

```json
{"session_id":"example","turns":[{"text":"Remember the code word ORCHID. Reply OK."},{"text":"What code word did I give you?","delay":50}]}
```

Use the benchmark command above with `--endpoint-type chat`, replace the two
synthetic-input options with
`--input-file conversations.jsonl --custom-dataset-type multi_turn`, and use
`--extra-inputs temperature:0 enable_thinking:false`. Each turn is one request;
`--request-count 2` measures one complete conversation. The delay is 50
milliseconds after the first response. AIPerf includes the original user turn
and the actual model answer in the follow-up request.

Keep conversation requests nonstreaming. History requires the loaded model's
native `TextConversation` capability; unsupported models return an explicit
error. Qwen owns chat formatting and tokenization. Its existing bundles work
with the updated family library; follow the existing build guide for the
server/runtime installation. The complete history and requested output must
fit the bundle's context limit; history is never silently dropped.

For pinned ShareGPT conversations, per-turn timings, and automatic history
verification, use the optional
[conversation profile](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/apps/aiperf_qual/README.md#pinned-conversations).

## Replay request arrivals

With the same server running, save this as `trace.jsonl`:

```json
{"session_id":"first","timestamp":0,"input_length":32,"output_length":16}
{"session_id":"second","timestamp":1000,"input_length":48,"output_length":16}
```

Use the completions benchmark command with
`--input-file trace.jsonl --custom-dataset-type mooncake_trace --fixed-schedule`.
Remove the synthetic-input, output-token, concurrency and warmup options.
Set `--request-count 2`. Timestamps are milliseconds from the start of replay;
these requests are offered one second apart. AIPerf generates prompts from the
input lengths because this trace contains lengths rather than original text.
Keep each input and output target within the bundle's context limit.

Request arrivals continue while inference runs. One server replica has one
execution lane, so overlapping requests may return `429`. Include those
rejections when comparing offered load with completed throughput.

The optional [production-trace profile](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/apps/aiperf_qual/README.md#pinned-production-trace)
provides a pinned Azure example, verifies arrival timing and accounts for every
request. It uses the existing Qwen bundle and the published AIPerf package.

## Benchmark shared prompt prefixes

Use the same actual-prompt JSONL format with a common beginning and distinct
endings. For example:

```json
{"text":"A service has one execution lane. Requests arriving while it is busy can be rejected. Question: How many lanes are there?"}
{"text":"A service has one execution lane. Requests arriving while it is busy can be rejected. Question: When can a request be rejected?"}
```

Run these through AIPerf's `single_turn` loader as described above. Shared
text alone does not establish native cache reuse. The current Qwen pilot resets
generation state for each request and exposes no cross-request prefix-cache
hit metrics.

For an exact token-overlap report and a reproducible comparison with and
without process warmup, use the optional
[shared-prefix profile](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/apps/aiperf_qual/README.md#controlled-shared-prefixes).

## Benchmark stateless Responses

With the same Qwen server, change the completions benchmark command to
`--endpoint-type responses` and
`--extra-inputs temperature:0 enable_thinking:false store:false`. Keep
nonstreaming mode and an output limit of at least 16 tokens. AIPerf sends one
user text input to `/v1/responses` and reads the native answer from its typed
message output. The optional [Responses profile](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/apps/aiperf_qual/README.md#stateless-responses)
provides pinned prompts and automatic request/output verification.

This endpoint supports stateless single-turn text only. Storage, response-ID
continuation, tools, media, reasoning-specific fields and streaming are rejected.
Usage is unavailable because the native Task does not report input-token counts;
use AIPerf's default client tokenizer estimates.

## Benchmark SpecBench prompts

The optional [SpecBench profile](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/apps/aiperf_qual/README.md#specbench-ordinary-generation-baseline)
selects twelve fitting original prompts from a pinned source file and sends
them through AIPerf's existing `single_turn` loader. Use the same built Qwen
bundle and environment; the runner checks family correctness before profiling.

This is an ordinary-generation corpus baseline. It does not add or qualify a
draft model or speculative runtime, and it reports no acceptance rate or
speculative speedup. The corpus ledger, raw requests and native output hashes
show exactly what was measured.
