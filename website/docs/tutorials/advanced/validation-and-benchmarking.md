---
title: Validation and Benchmarking
---

# Validation and benchmarking

Validation asks whether a family implementation is correct. Benchmarking asks
how its public task call performs. Establish correctness first.

## Validate repository structure

```bash
PYTHONPATH=core/builder:. python3 -m tools.model_ci validate
PYTHONPATH=core/builder:. python3 tools/test_impact.py --validate
```

These checks enforce family ownership and test selection without executing a
model.

## Run a family E2E case

```bash
export TRTMC_BINARY="$PWD/build/apps/cli/trtmc"
export TRTMC_RUNTIME_ROOT="$PWD/build/install/lib"

PYTHONPATH=core/builder:. python3 -m pytest \
  families/qwen/tests/test_e2e.py \
  --e2e-testcase qwen3-0.6b-fp16 \
  -q
```

Use a manifest selector that exists in the checkout. The family owns its input,
reference, comparison logic, thresholds, topology, and produced evidence.

Run focused host and compiled checks as appropriate:

```bash
PYTHONPATH=core/builder:. python3 -m pytest families/qwen/tests -q
ctest --test-dir build --output-on-failure
```

## Benchmark the public task

```bash
trtmc-bench run \
  --model distilgpt2 \
  --runtime-root /opt/trtmc/lib \
  -o results/distilgpt2
```

For release comparisons, validate and run the checked-in matrix:

```bash
python3 tools/perf_matrix.py check \
  apps/benchmark/performance/release.yaml \
  --environment apps/benchmark/performance/environments/gb300.yaml

python3 tools/perf_matrix.py run \
  apps/benchmark/performance/release.yaml \
  --environment apps/benchmark/performance/environments/gb300.yaml \
  --entry gpt2.generate
```

Bundle preparation and warmup are outside the candidate measurement. Reference
runners execute in separate environments so their dependencies do not leak
into shared runtime code.

## Evidence checklist

- commit and model revision;
- family manifest and threshold file;
- bundle and build arguments;
- runtime root and exact request;
- GPU topology and TensorRT/CUDA/driver versions;
- validation output before performance results;
- raw observations, summary JSON, and report.

Never loosen a correctness threshold to obtain a green performance result.
