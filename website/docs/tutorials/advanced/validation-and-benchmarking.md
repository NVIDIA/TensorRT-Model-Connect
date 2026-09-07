---
title: Advanced Tutorial - Validation and Benchmarking
---

import Diagram from '@site/src/components/Diagram';

## Learning objectives

- run repository and family-owned checks at the correct boundary;
- separate source, build, E2E, target, and performance evidence;
- benchmark through public Task APIs without coupling core to benchmark code.

<Diagram
  src="/img/diagrams/tutorials/advanced/validation-contract.svg"
  alt="The same canonical input reaches a family Task implementation and its declared reference before a family-owned comparison"
  caption="A family owns its checkpoint, oracle, thresholds, and semantic evidence."
/>

## Repository consistency

Start with the thin shared checks:

```bash
PYTHONPATH=core/builder:apps/benchmark:. \
  python3 -m tools.model_ci validate
PYTHONPATH=core/builder:apps/benchmark:. \
  python3 tools/test_impact.py --validate
git diff --check
```

These checks prove physical ownership and repository consistency. They do not
build a TensorRT engine or run a model.

## Focused family validation

Run the tests in the family you changed:

```bash
PYTHONPATH=core/builder:. \
  python3 -m pytest families/qwen/tests -q
```

Real E2E requires a built CLI, runtime root, model access, and GPU. Select the
exact family testcase:

```bash
TRTMC_E2E=1 \
TRTMC_BINARY=/path/to/trtmc \
TRTMC_RUNTIME_ROOT=/path/to/runtime \
PYTHONPATH=core/builder:. \
python3 -m pytest families/qwen/tests/test_e2e.py \
  --e2e-testcase qwen3-0.6b-fp16 -v
```

The family test builds the bundle, invokes the native CLI through the public
Task API, and applies that family's declared oracle. It never falls through to
a generic comparator.

## Core and tool tests

Use these only when their owner changed:

```bash
PYTHONPATH=core/builder:apps/benchmark:. \
  python3 -m pytest core/builder/tests tools/tests \
    apps/benchmark/trtmc_benchmark/tests -q
```

C++ tests are built and run through the configured CMake build directory:

```bash
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

## Benchmark through public APIs

The benchmark application lives under `apps/benchmark/` and depends one way on
the public build, load, and Task APIs.

```bash
apps/benchmark/trtmc-bench list models
apps/benchmark/trtmc-bench run \
  --model distilgpt2 \
  --runtime-root /opt/trtmc/lib \
  --output results/distilgpt2
```

For the checked-in performance suite:

```bash
python3 tools/perf_matrix.py check \
  apps/benchmark/performance/release.yaml \
  --environment apps/benchmark/performance/environments/gb300.yaml
```

<Diagram
  src="/img/diagrams/tutorials/advanced/benchmark-timing-scopes.svg"
  alt="The current benchmark excludes setup and measures the wall time of one public Task call"
  caption="Name the measurement boundary before comparing results. Current candidate timing is public_task_call_wall."
/>

Current candidate measurements use `public_task_call_wall`. Bundle loading,
worker startup, warmup, telemetry, report generation, and bundle building are
outside that measurement. Do not compare it directly with an engine-only
enqueue time.

## Validation taxonomy

| Evidence | What it proves |
| --- | --- |
| Source/static | Ownership and declared contracts are internally consistent. |
| Unit | One isolated function or native component behaves as tested. |
| Build | TensorRT accepted one exact graph and environment. |
| Family E2E | One checkpoint, task, request, runtime, and oracle passed. |
| Target qualification | The E2E passed on the named hardware/software cohort. |
| Benchmark | The exact measured workload produced the recorded observations. |

A skipped or unrun tier is not a pass. Record the exact revision, command,
environment, hardware, and artifact for every claim.

## Self-check

1. What does `tools.model_ci validate` prove?
2. Which directory owns a normal model's E2E and thresholds?
3. Why is a TensorRT build not an accuracy result?
4. Which setup phases are excluded from `public_task_call_wall`?
