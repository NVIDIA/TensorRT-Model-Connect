---
title: Profiling Guide
---

# Profiling guide

TRTMC currently has no repository-supported `trtmc_profile.py`, layer-diff, or
Nsight conversion wrapper. Profile the installed public task command with the
standard NVIDIA tools available in your target environment.

## Establish a repeatable command

First prove that the same bundle and request complete normally:

```bash
trtmc run model.bundle \
  --runtime-root /opt/trtmc/lib \
  --prompt "The capital of France is" \
  --max-new-tokens 20 \
  --seed 1234
```

Hold these inputs constant across comparisons:

- repository commit and bundle bytes;
- model revision and precision;
- request inputs and generation controls;
- runtime root, TensorRT/CUDA/driver versions, and target GPU;
- warmup and measured iteration counts;
- device topology for tensor or context parallel families.

## Choose the measurement layer

- Use `trtmc-bench` for repeatable public task latency and throughput. See the
  [Benchmarking Reference](/reference/benchmarking).
- Use Nsight Systems around the same CLI or benchmark worker to study process,
  CPU, CUDA API, and kernel timelines.
- Use Nsight Compute only after narrowing the question to a specific kernel.
- Use family-owned tests and reference comparisons to establish correctness
  before interpreting a speedup.

Profile loading separately from steady-state task execution. The public CLI
requires `--runtime-root`; include its exact directory in the evidence so the
loaded shared objects are reproducible.

## Interpreting results

- Do not compare instrumented latency directly with an uninstrumented run.
- Separate first-request setup from warm steady-state measurements.
- Compare identical task outputs or accepted validation thresholds.
- Treat a failure to build, load, or execute as an operational failure, not a
  performance number.
- Keep profiler artifacts with the benchmark JSON and environment metadata.

Family implementations own their orchestration, so a family-specific hot path
should be investigated and fixed inside that family. Shared runtime changes
need evidence that the issue is model-agnostic.
