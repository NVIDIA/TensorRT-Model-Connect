---
title: Profile Reranking with AIPerf
description: Benchmark text relevance scores from an already-built TRTMC reranker.
---

This example uses **nvidia/llama-nemotron-rerank-vl-1b-v2** with text passages.
Prepare the runtime using [Build from Source](../getting-started/source-build.md),
including the `trtmc_model_eagle_vlm` target, and build the model using
[Build a Bundle](./build-a-bundle.md). Use the existing
[FP16 profile](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/0bccb5e93af0e5ad346aaf44e404396f54b2e7de/families/eagle_vlm/tests/benchmark/nemotron-rerank-vl-1b-v2.yaml)
and run its family correctness checks first.

The commands assume `nemotron-rerank-vl-1b-v2.bundle` is already built and
`trtmc-server` is on `PATH`. Run from the repository root in your prepared GPU
environment; in Docker, use two terminals in the same container.

## 1. Install the client

```bash
python -m pip install -e '.[serve]' -C py-only=true
python -m venv .venv-aiperf
.venv-aiperf/bin/pip install aiperf==0.13.0
```

This uses the published pip package, without an AIPerf source checkout.

## 2. Start the model (terminal 1)

```bash
trtmc-server ./nemotron-rerank-vl-1b-v2.bundle \
  --model-name nvidia/llama-nemotron-rerank-vl-1b-v2 \
  --records artifacts/aiperf-rerank-server.jsonl
```

Wait for `Uvicorn running on http://127.0.0.1:8000`. Use the full binary path
and `--runtime-root` if your runtime libraries are in a separate directory.

## 3. Run the benchmark (terminal 2)

```bash
.venv-aiperf/bin/aiperf profile \
  --url http://127.0.0.1:8000 \
  --model nvidia/llama-nemotron-rerank-vl-1b-v2 \
  --endpoint-type nim_rankings --endpoint /v1/ranking \
  --tokenizer nvidia/llama-nemotron-rerank-vl-1b-v2 \
  --tokenizer-revision 9e95da054312436dfc703319dd2b793a3bee2465 \
  --input-file apps/aiperf_qual/examples/ranking-inputs.jsonl \
  --custom-dataset-type single_turn \
  --concurrency 1 --warmup-request-count 5 --request-count 50 \
  --random-seed 0 --ui-type simple --export-level raw \
  --artifact-dir artifacts/aiperf-rerank
```

This repeats three fixed queries with two text passages each: five warmups,
then 50 measured requests. It is an integration example, not a performance gate.
Use a new artifact directory for each run. Stop the server with **Ctrl+C**.

Check successful requests, **Request Latency** and **Request Throughput** in
`profile_export_aiperf.json` or `.csv`. For this fixed two-passage workload,
passages/sec is twice successful requests/sec. Server records retain passage
counts, text byte lengths, request IDs, and `model_call_ms` with
`timing_scope=public_task_call_wall`; native Task time remains separate from
HTTP latency. Reranking has no token-generation TTFT or inter-token latency.

## Run another built reranker

Find the current profiles in the [model-group inventory](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/apps/aiperf_qual/README.md#native-endpoint-group-coverage).

Use its bundle, model name, matching tokenizer/revision, family runtime libraries,
and family correctness criteria. A public document-relevance Task returning raw logits, or an existing
reranking bundle, can reuse the route; each model needs qualification.

`/v1/ranking` accepts `query.text` and a non-empty list of `passages[].text`.
It returns all real scores in descending order as `rankings[].logit`, retaining
each passage's original `index`. Ties retain input order. Keep each query/passage
pair within the bundle's compiled sequence limit. Only text inputs and
`truncate=NONE` are supported; truncation, images, streaming, and `top_n` are
rejected. HTTP passage lists do not establish native GPU batching.
