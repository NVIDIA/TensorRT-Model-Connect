---
title: Performance Benchmarking
description: Measure public Task API calls without coupling benchmark policy to core or model families.
---

import Diagram from '@site/src/components/Diagram';

Benchmarking is an application above Model Connect:

```text
trtmc-bench -> public build/load/Task APIs
performance matrix -> trtmc-bench + an independent reference process
```

Neither core nor a family imports benchmark code. The benchmark catalog reads
`families/*/tests/manifests/*.json` directly, so there is no second model
registry to maintain.

## Run one model

List available manifest-backed cases and run one public Task operation:

```bash
trtmc-bench list models
trtmc-bench run \
  --model distilgpt2 \
  --runtime-root /opt/trtmc/lib \
  --output results/distilgpt2
```

If no matching bundle is supplied, the benchmark may build one through the
public build API and cache it. Use `--no-build` when a missing bundle must be an
error, or provide an exact path with `--bundle MODEL=/path/model.bundle`.

A checked-in YAML can select several models and cases:

```bash
trtmc-bench run apps/benchmark/example.yaml \
  --runtime-root /opt/trtmc/lib \
  --output results/example
```

Use `--dry-run` to inspect resolved work without execution and `--prepare-only`
to build missing bundles without timing them.

## Measurement contract

The supported timing scope is `public_task_call_wall`: the wall time of one
call through the concrete abstract Task interface returned by `load_task()`.
Bundle creation, process startup, DSO loading, warmup, telemetry, and report
generation are outside the timed call. Asset loading is outside by default and
must be declared when a case intentionally includes it.

<Diagram
  src="/img/diagrams/reference/benchmark-measurement-reporting.svg"
  alt="Benchmark stages separating setup and warmup from repeated public Task calls, metrics, telemetry, and reports"
  caption="Measure one named boundary. Keep setup and low-frequency telemetry outside the timed Task call."
/>

Override repetitions without changing the family:

```bash
trtmc-bench run \
  --model distilgpt2 \
  --runtime-root /opt/trtmc/lib \
  --warmup 3 \
  --iterations 10 \
  --telemetry auto \
  --output results/distilgpt2
```

The report records latency summaries and operation-specific rates or stage
times when the Task exposes them. `telemetry.json` is best-effort,
low-frequency `nvidia-smi` sampling over the worker process; it is not a
per-kernel measurement and is not sampled inside each timed call.

## Sweep request or measurement values

`--set` changes one value and `--sweep` expands a comma-separated set. Only
`request.*`, `measurement.*`, and `telemetry.*` namespaces are accepted.

```bash
trtmc-bench run \
  --model distilgpt2 \
  --runtime-root /opt/trtmc/lib \
  --set request.max_new_tokens=32 \
  --sweep request.temperature=0.0,0.7 \
  --output results/text-sweep
```

Change one dimension at a time when you need a causal conclusion. A faster
result is not useful if the request or output-quality contract changed.

## Release comparison matrix

The release matrix runs the same candidate Task boundary and a separately
owned reference process, then checks workload, timing, and output contracts.

<Diagram
  src="/img/diagrams/reference/benchmark-orchestration.svg"
  alt="Performance matrix resolving one family manifest, preparing a bundle, running candidate and reference processes, and comparing their declared contracts"
  caption="Candidate and reference dependencies stay isolated; the matrix compares only after both sides satisfy the same declared workload and timing contract."
/>

Validate a suite without measuring, prepare bundles, or run one entry:

```bash
python3 tools/perf_matrix.py check \
  apps/benchmark/performance/release.yaml \
  --environment apps/benchmark/performance/environments/gb300.yaml

python3 tools/perf_matrix.py prepare \
  apps/benchmark/performance/release.yaml \
  --environment apps/benchmark/performance/environments/gb300.yaml \
  --entry gpt2.generate \
  --output artifacts/perf/bundle-preparation.json

python3 tools/perf_matrix.py run \
  apps/benchmark/performance/release.yaml \
  --environment apps/benchmark/performance/environments/gb300.yaml \
  --entry gpt2.generate
```

Machine paths and reference checkouts belong in the selected environment file,
not in family manifests. Reference-only packages belong in
`apps/benchmark/performance/requirements.txt`; they must not become shared core
or unrelated family dependencies.

Continue or rerender a stored run:

```bash
python3 tools/perf_matrix.py resume artifacts/perf/RUN_DIRECTORY
python3 tools/perf_matrix.py report artifacts/perf/RUN_DIRECTORY
```

## Interpret results precisely

- A dry run or matrix `check` proves configuration only.
- A completed candidate measurement proves only its exact Task, request,
  bundle, runtime, and machine cohort.
- A comparison light requires both candidate and reference to complete their
  declared contracts.
- Operational failures and contract mismatches are not performance results.
- A documentation or CPU-CI pass is not GPU performance evidence.
- Publish the exact revision, model, bundle settings, hardware/software,
  warmups, iterations, raw samples, quality gate, and timing scope with any
  claim.

See `apps/benchmark/performance/README.md` for the maintained environment and
reference-runner contract.
