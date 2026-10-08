---
title: Profile Text with AIPerf
description: Benchmark an already-built TRTMC text model with the pip-installed AIPerf client.
---

AIPerf sends requests to `trtmc-server` and reports latency and throughput.
The server loads your TRTMC bundle and runs the model on the GPU.

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
