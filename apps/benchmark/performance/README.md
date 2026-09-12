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
worker selects the bundle's runtime mode before execution: migrated bundles use
the public header-only Task SDK and C ABI, while existing bundles use
load_task(bundle, runtime_root). Both call the family's implementation directly;
a failed SDK call is never retried through an old interface.

## Task API benchmark

Use the packaged CLI for a single model or the checked-in multi-model example:

```bash
trtmc-bench list models
trtmc-bench run --model distilgpt2 --runtime-root /opt/trtmc/lib -o results/distilgpt2
trtmc-bench run apps/benchmark/example.yaml -o results/example
```

Missing bundles are built through the public build command and cached. Pass
`--no-build` when every selected bundle must already exist.

Managed cache reuse requires a matching benchmark receipt for the resolved
immutable Hugging Face snapshot, manifest, build arguments, builder sources,
and build environment. Changed or missing identities rebuild the bundle;
`--no-build` reports an error instead. Arbitrary local checkpoint directories
are mutable and are rebuilt rather than assumed unchanged. An explicitly
supplied bundle outside the managed cache remains the caller's provenance
responsibility. `--rebuild` forces a fresh managed build.

### Select another Task from the same bundle

A family-owned testcase can select a secondary Task without changing the
manifest's primary `task` or rebuilding the bundle:

```json
{
  "name": "token-features",
  "selected_task": "text_to_token_features",
  "inputs": {"token_ids": [7, 9]},
  "config": {}
}
```

The existing `--case token-features` option selects this workload. Alternatively,
`--task text_to_token_features` selects the interface for the chosen testcase;
its inputs must match that interface. The operation defaults from the selected
Task. An explicit incompatible operation is an error, not a request to try
another interface. `selected_task` is call selection, not family Config.

The worker checks that the loaded model actually binds the requested Task.
Bundle identity and managed-cache checks still use the manifest's primary Task.
Results, reproduction records and history comparison retain the selected Task,
so token features and pooled features cannot become the same performance series.
Performance references receive the selection separately from the unchanged
manifest. Their input and output support must match the chosen contract; the
text-only reference loaders reject token IDs instead of replacing them with
empty text. This selection path requires a family migrated to the Task SDK.

## DataFrame forecast formatting

Install `pandas` separately (`pip install pandas`) when using the optional table
helpers. They do not load a model or run Python inference:

```python
import pandas as pd
from trtmc_benchmark.dataframe import prepare_forecast_frame, format_forecast_frame

frame = pd.read_csv("history.csv", parse_dates=["ds"])
prepared = prepare_forecast_frame(frame, freq="D")
# prepared.request is the existing native worker's batch solve payload.
# Pass it as request= to an existing resolved batch forecast case.
```

Input columns default to `unique_id`, `ds`, and `values`. Series keep first-seen
order; time is sorted within each series. Timestamps must match the explicit
calendar frequency: gaps/duplicates are errors, not silently resampled data.
Use `value_columns=("x", "y")` for one two-channel series, not two batch items.
Missing values retain their masks; no shared imputation or normalization occurs.
`config_by_series={"sensor-a": {"frequency": 0}}` passes family Config unchanged;
calendar `freq` never selects a model's frequency category.

For an existing resolved `case` whose bundle declares one of
`batch_series_to_point_forecast`, `batch_series_to_quantile_forecast`, or
`batch_series_to_point_and_quantile_forecast`, reuse the normal worker:

```python
from pathlib import Path
from trtmc_benchmark.types import MeasurementSpec
from trtmc_benchmark.worker import find_worker, run_worker

case = case.with_values(request=prepared.request,
                        measurement=MeasurementSpec(warmup=0, iterations=1))
output_dir = Path("forecast-results")
output_dir.mkdir(parents=True, exist_ok=True)
receipt = run_worker(case, output_dir, find_worker())
forecast = format_forecast_frame(prepared, receipt["output_summary"])
```

The case retains its explicit runtime root. This example executes one complete
native batch; normal benchmark warmup/iteration settings repeat that batch.
Scalar-only routes reject the batch payload rather than retrying each series.
Real family native-batch support must be qualified separately; this helper does
not imply that the existing TimesFM batch-one bundle supports native batching.

The result keeps series/horizon/channel order, actual horizon offsets, output
channel labels/units, and all actual quantile levels. The `forecast` column is
the real point output, never a substituted median; quantile-only results have
no invented point column. Unknown output labels remain missing. Heterogeneous
quantile columns have missing cells where a series did not predict that level,
with actual per-series levels recorded in `forecast.attrs`.

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
candidate worker or core runtime. The benchmark-owned cache receipt protects
managed build reuse without changing the public bundle format. It does not
certify externally supplied bundle bytes.

HTML reports include p50, p95, task rates, measurement boundaries, reproduction
commands, and recorded evidence. Historical v1 results can be rendered without
relabelling their original timing scope; they cannot be resumed as v2 runs.
Historical latency deltas appear only when the recorded comparison identities
match. A performance result does not establish model correctness.

External reference checkouts must provide their official runtime dependencies.
In particular, PersonaPlex requires the real `sphn` package and Lance requires
the dependencies imported by its upstream `inference_lance.py`; the benchmark
does not install substitutes for either reference. Fast Foundation Stereo also
requires the image and OpenCV packages imported by its official source tree.

Install benchmark-only dependencies without adding them to a model family:

```bash
python -m pip install -r apps/benchmark/performance/requirements.txt
```
