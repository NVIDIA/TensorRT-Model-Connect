<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Performance matrix

The performance matrix is an application above Model Connect:

~~~text
performance matrix -> trtmc-bench -> public build and Task APIs
performance matrix -> reference runner
~~~

Neither the native core nor any family imports benchmark code. The candidate
worker loads a bundle with the public load_task(bundle, runtime_root) API and
calls the exact abstract Task interface implemented by that family.

## Task API benchmark

Use the packaged CLI for a single model or the checked-in multi-model example:

```bash
trtmc-bench list models
trtmc-bench run --model distilgpt2 --runtime-root /opt/trtmc/lib -o results/distilgpt2
trtmc-bench run apps/benchmark/example.yaml -o results/example
```

Missing bundles are built through the public build command and cached. Pass
`--no-build` when every selected bundle must already exist.

## Timing contract

Candidate measurements use one scope: public_task_call_wall.

The candidate runs each family's production Task implementation as loaded. The
benchmark does not override family runtime policy with tuning switches such as
CUDA Graph enablement.

The new architecture transfers control directly from the public Task interface
to the family implementation, so the old pipeline-call and model-call scopes are
the same boundary. Bundle loading, worker startup, warmup, telemetry, report
generation, and bundle building are excluded. Asset loading is excluded unless a
case explicitly sets asset_loading_included to true.

The release suite defaults to three warmups and ten measured iterations. A
reference within 5% of candidate p50 is equivalent. Candidate and reference
failures are operational failures, not red performance results.

## Commands

Point the environment at one installed runtime directory containing the runtime,
TensorRT backend, and selected family DSOs:

~~~bash
export TRTMC_PERF_WORKER=/opt/trtmc/bin/trtmc_benchmark_worker
export TRTMC_PERF_RUNTIME_ROOT=/opt/trtmc/lib
export TRTMC_PERF_BUNDLE_CACHE=/data/trtmc-bundles
export TRTMC_PERF_BUNDLE_ROOTS=/data/prebuilt-bundles
export TRTMC_ELF_REFERENCE_REPO=/opt/references/ELF
export TRTMC_LANCE_REFERENCE_REPO=/opt/references/Lance
export TRTMC_SANA_WM_REFERENCE_REPO=/opt/references/Sana
export TRTMC_SANA_WM_MODEL_DIR=/models/sana-wm
export PERSONAPLEX_OFFICIAL_REPO=/opt/references/personaplex
export TRTMC_FAST_FOUNDATION_STEREO_MODEL_DIR=/models/fast-foundation-stereo

python3 tools/perf_matrix.py check \
  apps/benchmark/performance/release.yaml \
  --environment apps/benchmark/performance/environments/gb300.yaml

python3 tools/perf_matrix.py run \
  apps/benchmark/performance/release.yaml \
  --environment apps/benchmark/performance/environments/gb300.yaml \
  --entry gpt2.generate
~~~

Bundle preparation is a separate, untimed step:

~~~bash
python3 tools/perf_matrix.py prepare \
  apps/benchmark/performance/release.yaml \
  --environment apps/benchmark/performance/environments/gb300.yaml \
  --entry gpt2.generate \
  --output artifacts/perf/bundle-preparation.json
~~~

Continue an interrupted run and regenerate its report:

~~~bash
python3 tools/perf_matrix.py resume artifacts/perf/<run-directory>
python3 tools/perf_matrix.py report artifacts/perf/<run-directory>
~~~

## Configuration

release.yaml owns only model, testcase, operation, measurement, reference, and
comparison semantics. Machine paths live in one environment YAML.

A release entry has one explicit testcase:

~~~yaml
- id: gpt2.generate
  family: gpt2
  operation: generate
  model: distilgpt2
  workload:
    testcase: distilgpt2
  baseline:
    runner: hf-transformers
    mode: torch-compile
~~~

The catalog reads families/*/tests/manifests/*.json directly. It has no second
registry and no family-specific benchmark plugin. A task that the worker does
not implement is reported as unsupported.

The entry, model, and model-selection options select work. resume continues
incomplete rows, and report regenerates JSON and HTML from stored measurements.

Reference runners are separate processes so their dependencies never enter the
candidate worker or core runtime. Optional Hugging Face revisions are model
inputs only; the benchmark does not calculate or validate repository, request,
bundle, source, or report fingerprints.

External reference checkouts must provide their official runtime dependencies.
In particular, PersonaPlex requires the real `sphn` package and Lance requires
the dependencies imported by its upstream `inference_lance.py`; the benchmark
does not install substitutes for either reference. Fast Foundation Stereo also
requires the image and OpenCV packages imported by its official source tree.

Install benchmark-only dependencies without adding them to a model family:

```bash
python -m pip install -r apps/benchmark/performance/requirements.txt
```
