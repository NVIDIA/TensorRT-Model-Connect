---
title: Profile Embeddings with AIPerf
description: Benchmark single-sentence embeddings from an already-built TRTMC model.
---

This example uses **BAAI/bge-small-en-v1.5**. Prepare the runtime using
[Build from Source](../getting-started/source-build.md), including the `trtmc_model_bert`
and `trtmc_cli_bert` targets, and build the model using
[Build a Bundle](./build-a-bundle.md). Use the existing
[BGE FP16 profile](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/0bccb5e93af0e5ad346aaf44e404396f54b2e7de/families/bert/tests/benchmark/bge-small-en-v1.5.yaml)
and run its family correctness checks first. The commands below assume the bundle
is `bge-small-en-v1.5.bundle` and `trtmc-server` is on `PATH`.

Run from the repository root in your prepared GPU environment. In Docker, use
two terminals in the same container.

## 1. Install the client

```bash
python -m pip install -e '.[serve]' -C py-only=true
python -m venv .venv-aiperf
.venv-aiperf/bin/pip install aiperf==0.13.0
```

This is the published pip package; an AIPerf repository is unnecessary.

## 2. Start the model (terminal 1)

```bash
trtmc-server ./bge-small-en-v1.5.bundle \
  --model-name BAAI/bge-small-en-v1.5 \
  --records artifacts/aiperf-bge-server.jsonl
```

Wait for `Uvicorn running on http://127.0.0.1:8000`. The model stays loaded.
If needed, use your server binary's full path and `--runtime-root` pointing to
the directory containing its backend and family libraries.

## 3. Run the benchmark (terminal 2)

```bash
.venv-aiperf/bin/aiperf profile \
  --url http://127.0.0.1:8000 \
  --model BAAI/bge-small-en-v1.5 \
  --endpoint-type embeddings --endpoint /v1/embeddings \
  --tokenizer BAAI/bge-small-en-v1.5 \
  --tokenizer-revision 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a \
  --input-file apps/aiperf_qual/examples/embedding-inputs.jsonl \
  --custom-dataset-type single_turn \
  --concurrency 1 --warmup-request-count 5 --request-count 50 \
  --random-seed 42 --ui-type simple --export-level raw \
  --artifact-dir artifacts/aiperf-bge
```

This repeats the five fixed sentences for 50 measured requests after five
warmups. It is an integration example, not a statistical performance gate.
Use a new artifact directory for each run, then stop the server with **Ctrl+C**.

Check successful requests, **Request Latency** and **Request Throughput** in
`profile_export_aiperf.json` or `.csv`. Embeddings have no TTFT or inter-token
latency. Server records preserve request IDs and `model_call_ms` with
`timing_scope=public_task_call_wall`; that time excludes build/load/startup and
remains separate from AIPerf's HTTP latency. Token usage is unavailable from the
Task and is omitted from responses.

## Run another built embedding model

Use the same steps with its bundle, model name, matching tokenizer/revision,
and family runtime libraries. The server supports public semantic embedding
or pooled-feature Tasks and existing embedding/encoding bundle modes.
Validate the particular family before treating its model as qualified.
Use the [model-group inventory](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/apps/aiperf_qual/README.md#native-endpoint-group-coverage)
to find the current profiles and their family-owned manifests.

Each request accepts one non-empty string or a one-element string list and
returns the full float vector produced by the family. Pooling and normalization
remain family-owned. Token-ID inputs, multiple-input batches, `dimensions`,
non-float encodings, and streaming are rejected. Keep inputs within your bundle's
compiled sequence limit; output vectors remain complete regardless of log settings.
