<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Performance serving (`trtmc-perf-serve`)

`trtmc-perf-serve` puts one operation of one backend behind HTTP so a load
generator such as [aiperf](https://github.com/ai-dynamo/aiperf) can measure it.
It covers every benchmark operation, not only text generation, and it is built
for measurement rather than production serving.

| Backend | Execution | Timed boundary (`model_call_ms`) |
| --- | --- | --- |
| `trtmc` | persistent `trtmc_benchmark_worker --serve` process | `public_task_call_wall`, the same dispatch and timing as a `trtmc-bench` sample |
| `reference` | Python adapter: HF eager or `torch.compile`, Diffusers | the model call after input preparation (`task-model-call-wall`) |

Both backends read the benchmark worker's operation-request schema. The base
request comes from a catalog profile and testcase, resolved in the same way as
`trtmc-bench`, so the candidate and the reference receive identical inputs.

## Serve

```bash
export PYTHONPATH=$PWD/apps/perf_serving:$PWD/apps/benchmark:$PWD/core/builder:$PWD

# TRTMC candidate (build with -DTRTMC_BUILD_EXAMPLES=ON for the worker)
python -m trtmc_perf_serving serve --backend trtmc --profile timm-vit-base-p16-224-augreg-in21k-ft-in1k \
  --bundle vit.bundle --runtime-root build --worker build/trtmc_benchmark_worker \
  --records runs/vit-trtmc/records.jsonl --scratch runs/vit-trtmc/scratch --port 8500

# Reference (eager or compile) for the same profile
python -m trtmc_perf_serving serve --backend reference --mode eager \
  --profile timm-vit-base-p16-224-augreg-in21k-ft-in1k \
  --records runs/vit-ref/records.jsonl --scratch runs/vit-ref/scratch --port 8501
```

`--set FIELD=VALUE` overrides base-request fields in the same way as
`trtmc-bench --set`. `--testcase` selects a testcase other than the profile's
first one.

## Routes

| Route | Operation | aiperf `--endpoint-type` |
| --- | --- | --- |
| `POST /v1/tasks/{operation}` with body `{"request": {...}}` | any | `raw` + `mooncake_trace` payloads |
| `POST /v1/completions`, `/v1/chat/completions` | `generate` | `completions`, `chat` |
| `POST /v1/embeddings` | `embed`, `encode` | `embeddings` |
| `POST /v1/ranking` (NIM shape) | `rerank` | `nim_rankings` |
| `POST /v1/audio/transcriptions` (multipart) | `transcribe` | `audio_transcription` |
| `POST /v1/images/generations` | `generate_image` | `image_generation` |
| `POST /v1/videos`, `GET /v1/videos/{id}` | `generate_image` (video) | `video_generation` (the reported latency includes the polling interval) |
| `GET /health/live`, `/health/ready`, `/v1/models`, `/v1/serving/info` | — | — |

A request body is merged over the profile's base request. Paths inside
`*_path`/`*_paths` fields are sent inline as
`{"$file": {"suffix": ".png", "b64": "..."}}` and written to per-request files
on the server. OpenAI fields that every backend cannot honor in the same way
(for example `ignore_eos`) are rejected with HTTP 400 instead of being
silently ignored.

`stream: true` returns buffered SSE for every backend: the complete output is
sent as one chunk. TTFT therefore equals request latency, and streaming
behavior stays comparable between the candidate and the reference.

## Measurement data

Every response carries `trtmc_timing` (`queue_ms`, `model_call_ms`,
`handler_ms`) and `trtmc_observation` (the operation's output summary, matching
the benchmark worker's observation fields). Each response also carries a
`Server-Timing` header. The server honors an incoming `X-Request-ID`, and
`--records` stores one JSONL record per request under that ID. aiperf sends
`X-Request-ID` and exports it with `--export-level raw`, so client-side
latency joins exactly with the server's model-call time. Use `model_call_ms`
for candidate/reference comparison; client latency adds HTTP transfer and file
handling.

The server has one execution lane. Up to `--max-queue` requests wait (the wait
is reported as `queue_ms`); beyond that the server answers 429 with
`Retry-After`. Use `--max-queue 0` to reproduce the no-queue admission of
`trtmc-server`.

## aiperf recipes

```bash
# Any operation: replay the catalog testcase request verbatim.
python -m trtmc_perf_serving payload --profile sam-vit-base --output sam.jsonl
aiperf profile --model trtmc --url http://127.0.0.1:8500/v1/tasks/segment_prompted --endpoint-type raw \
  --input-file sam.jsonl --custom-dataset-type mooncake_trace --tokenizer builtin \
  --concurrency 1 --warmup-request-count 5 --request-count 50 --export-level raw

# Text generation through the OpenAI surface.
aiperf profile --model trtmc --url http://127.0.0.1:8500 --endpoint-type completions --streaming \
  --tokenizer Qwen/Qwen3-0.6B --isl 128 --osl 64 --concurrency 1 --request-count 50 --export-level raw
```

Run the aiperf client without `HF_HUB_OFFLINE=1`. In offline mode, aiperf
0.13 loads tokenizers through `snapshot_download(local_files_only=True)`,
which rejects local tokenizer directories and incomplete cache snapshots.

## Reference coverage

Generic Python reference adapters exist for `generate`, `translate`, `encode`,
`embed`, `rerank`, `classify`, `detect`, `segment`, `segment_prompted`,
`extract_features`, `transcribe`, `generate_audio` (Bark), `generate_image`
(Diffusers), `solve`, and `regress` (Transformers PatchTST/PatchTSMixer). A model
whose native pipeline they cannot run (an upstream checkout such as PersonaPlex,
MoGe, LeRobot, Fast-FoundationStereo, SANA-WM; Ultralytics archives; a
speech-conditioned LM) is served by its family's own adapter:
`--reference-adapter families/<family>/reference/adapter.py`, a file defining
`Adapter(spec, host)` with `invoke(request, artifact_base)`. The family file imports nothing
from this package: `host` (`NativeHost`) carries the backend's model-agnostic mechanics (the
request's fields and pre-decoded input files, the timed call, output tensors and files written
after it, the result, and `host.Error` for a rejected request).

`reference-env --requirements FILE --root DIR [--prepare SCRIPT]` creates (once, cached
by digest) a virtual environment that layers `FILE` on this interpreter, runs `SCRIPT`
in it after the install (an upstream checkout), records its `pip freeze`, and prints its
interpreter; an environment whose packages changed since (or that has no freeze record) is
left in place and a fresh one is created next to it.
