---
title: Benchmarking Reference
---

# Benchmarking reference

The benchmark application sits above TRTMC's public build and task APIs. It
does not register model families or import family implementation code.

## Single-model benchmark

Install the optional benchmark dependencies, then select an installed runtime
root containing the core runtime, TensorRT backend, and required family DSOs:

```bash
python -m pip install -r apps/benchmark/performance/requirements.txt

trtmc-bench list models
trtmc-bench run \
  --model distilgpt2 \
  --runtime-root /opt/trtmc/lib \
  -o results/distilgpt2
```

`trtmc-bench` reads family-owned manifests under
`families/*/tests/manifests/`. Missing bundles are built through the public
builder and cached; pass `--no-build` when every selected bundle must already
exist.

Run a checked-in multi-model specification with:

```bash
trtmc-bench run apps/benchmark/example.yaml -o results/example
```

Use `trtmc-bench --help` and subcommand help as the authoritative option list
for the installed version.

## Timing boundary

Candidate timing measures the public family task call. Bundle construction,
process startup, task loading, warmup, telemetry, and report generation are
outside that measurement. Asset loading is also excluded unless a case
explicitly includes it.

The release suite defaults to three warmups and ten measured iterations. A
reference within five percent of candidate p50 is considered equivalent.
Candidate or reference execution failures are operational failures, not slow
performance results.

## Release performance matrix

The matrix coordinates candidate and reference runs without introducing a
second model registry:

```bash
export TRTMC_PERF_WORKER=/opt/trtmc/bin/trtmc_benchmark_worker
export TRTMC_PERF_RUNTIME_ROOT=/opt/trtmc/lib
export TRTMC_PERF_BUNDLE_CACHE=/data/trtmc-bundles

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

Preparation is deliberately separate and untimed. Resume or regenerate a
report from stored observations with:

```bash
python3 tools/perf_matrix.py resume artifacts/perf/<run-directory>
python3 tools/perf_matrix.py report artifacts/perf/<run-directory>
```

The release YAML owns model, testcase, operation, measurement, reference, and
comparison semantics. Machine-specific paths belong in the environment YAML.
Reference implementations run in separate processes so their dependencies do
not enter the candidate worker or shared runtime.

## Adding coverage

- Add a weight or profile to the owning family's test manifest catalog.
- Add a matrix entry that names the family, model, operation, workload, and
  reference runner.
- Add benchmark code only when a genuinely new public task interface needs a
  task adapter.

Retain raw observations and reports together with the commit, model revision,
runtime root, target GPU, TensorRT version, warmup count, and iteration count.
Correctness validation should precede performance comparison.
