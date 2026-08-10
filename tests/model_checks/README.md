# Model checks

`tools/model_checks.py` maps selected models to exact Accuracy suites and Perf
entries.

## Setup

GB300 example:

```bash
export TRTMC_CHECK_STORAGE_ROOT=/runs
export TRTMC_CHECK_DATASET_ROOT=/mnt/data
export TRTMC_CHECK_BUILD_DIR=/runs/tmp/build-trt112
export TRTMC_CHECK_PYTHON=/opt/venv/bin/python3
```

`TRTMC_CHECK_BUILD_DIR` is the native CMake build output. The runner derives
the Accuracy binaries, model plugins, TensorRT backend, Perf worker, and Perf
bundle cache from this directory and `TRTMC_CHECK_STORAGE_ROOT`.

The controller Python must contain the repository's base dependencies. Missing
family build, reference, and scoring profiles are created under
`${TRTMC_CHECK_STORAGE_ROOT}/python-profiles` when first needed.

## Select

Show all configured Accuracy suites and Perf entries for one model:

```bash
$TRTMC_CHECK_PYTHON tools/model_checks.py check \
  --platform gb300 \
  --model distilgpt2
```

Select exact Accuracy suites per model:

```bash
$TRTMC_CHECK_PYTHON tools/model_checks.py check \
  --platform gb300 \
  --task accuracy \
  --model qwen25vl-3b \
  --model gpt2-125m \
  --accuracy-binding qwen25vl-3b=vlm_mmmu_pro_vision_mcq \
  --accuracy-binding gpt2-125m=mmlu_continuation_parity
```

Without an explicit binding, Accuracy runs every workload listed for the
model in `tests/validation/model_workloads.yaml`.

## Run

Run one model through Accuracy and then Perf:

```bash
$TRTMC_CHECK_PYTHON tools/model_checks.py run \
  --platform gb300 \
  --model distilgpt2 \
  --run-id gb300-distilgpt2-smoke
```

Run full Accuracy and Perf on separate GB300 machines:

```bash
$TRTMC_CHECK_PYTHON tools/model_checks.py run \
  --platform gb300 \
  --task accuracy \
  --all \
  --run-id gb300-accuracy-all

$TRTMC_CHECK_PYTHON tools/model_checks.py run \
  --platform gb300 \
  --task perf \
  --all \
  --run-id gb300-perf-all
```

Add `--dry-run` to print the resolved child commands, or `--verbose` to retain
detailed runner output.

Results are written below:

```text
${TRTMC_CHECK_STORAGE_ROOT}/results/<run-id>/
```

Resume with the same selection and run ID:

```bash
$TRTMC_CHECK_PYTHON tools/model_checks.py run \
  --platform gb300 \
  --task accuracy \
  --all \
  --run-id gb300-accuracy-all \
  --resume
```

## Platforms

Use `--platform gb300`, `--platform auto-thor`, or `--platform l4t-thor`.
Platform files under `tests/model_checks/platforms/` define task order and
hardware exclusions. Files under `tests/model_checks/environments/` define
paths and retention.

Checked-in environments delete Accuracy engines and Perf bundles after each
binding. GB300 and Auto Thor retain their shared Hugging Face cache. L4T Thor
deletes its isolated model cache and requires `TRTMC_CHECK_STORAGE_ROOT` on
`/dev/nvme0n1p1`.

See `benchmarks/performance/README.md` for the L4T TensorRT 11 bare-metal build.

## Add a model or dataset

- Define an Accuracy suite in `tests/validation/workloads.yaml`.
- Assign one or more suites in `tests/validation/model_workloads.yaml`.
- Add the Perf entry independently in `benchmarks/performance/release.yaml`.
- Add only verified hardware exclusions under `tests/model_checks/platforms/`.

Each Accuracy `MODEL=SUITE` binding has its own engine directory. Perf bundles
remain independent from Accuracy engines.
