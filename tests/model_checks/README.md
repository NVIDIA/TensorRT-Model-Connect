# Model checks

`tools/model_checks.py` maps selected models to exact Accuracy suites and Perf
entries.

Model Checks composes the public qualification interfaces: Validation model
records come from `tools.validation.catalog`, Performance entries come from
`tools.performance.catalog`, and each runner owns its `write_report()` logic.
The controller does not parse runner-private suites, ledgers, or report data.

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

The controller Python must contain the repository's base dependencies. Debug
runs create missing family build, reference, and scoring profiles under
`${TRTMC_CHECK_STORAGE_ROOT}/python-profiles` when first needed.

## Select

Show all configured Accuracy suites and Perf entries for one model:

```bash
$TRTMC_CHECK_PYTHON tools/model_checks.py check \
  --platform gb300 \
  --model distilgpt2
```

Before consuming GPU time, validate the selected target's exact native build
and Accuracy datasets. This is read-only and returns machine-readable blockers:

```bash
$TRTMC_CHECK_PYTHON tools/model_checks.py check \
  --platform gb300 \
  --environment gb300 \
  --model distilgpt2 \
  --revision <40-character-sha> \
  --target-preflight \
  --json
```

The JSON includes `resolved_revision` and `target_preflight`. A missing dataset,
native executable, TensorRT backend, model-plugin directory, worker metadata,
or matching embedded worker SHA makes the command fail before profile or bundle
preparation. When Perf is selected, preflight also loads the packaged TensorRT
backend through the declared Perf runner Python. The command fails if that
loader smoke test cannot resolve or load the backend and records the result in
`target_preflight.perf_backend_loader`.

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

By default, `run` is a formal qualification run. Each execution attempt freezes source identity,
prepares missing dependencies, runs preflight, and then switches measurement to
prebuilt-only mode. During development, add `--debug` to allow a dirty worktree
and on-demand dependency creation. Every run still resolves the active worktree
HEAD to a 40-character commit SHA and places the current repository Python paths
first in each child process.

Run one model through formal Accuracy and then Perf qualification:

```bash
$TRTMC_CHECK_PYTHON tools/model_checks.py run \
  --platform gb300 \
  --model distilgpt2 \
  --run-id gb300-distilgpt2-smoke
```

The single command owns both preparation and frozen measurement:

```bash
$TRTMC_CHECK_PYTHON tools/model_checks.py run \
  --platform gb300 \
  --model distilgpt2 \
  --run-id gb300-distilgpt2-qualification
```

For an editable debug run:

```bash
$TRTMC_CHECK_PYTHON tools/model_checks.py run \
  --platform gb300 \
  --model distilgpt2 \
  --debug \
  --run-id gb300-distilgpt2-debug
```

The controller first writes `native-build-identity.json`, then creates missing
Python profiles, pinned reference-source checkouts, and Perf bundles before
measurement. Measurement then runs with
dependency creation and Perf bundle builds disabled. Qualification also rejects
a dirty worktree, imports outside the active worktree, and a requested revision
different from HEAD. It rechecks that identity after preparation and before and
after every task, so evidence from an attempt cannot silently cross a mid-run
edit. Accuracy and Perf receipts record the exact SHA tested; native workers and
bundles must report that SHA. A campaign may be resumed on a later commit, but
all final Accuracy and Perf receipts for one model must agree on one SHA. Other
models may complete on different SHAs.

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

Seed an isolated L4T Accuracy cache from an existing `HF_HOME` tree on the same
filesystem. Files are hard-linked on demand; per-model cleanup never deletes the
seed:

```bash
$TRTMC_CHECK_PYTHON tools/model_checks.py run \
  --platform l4t-thor \
  --task accuracy \
  --model sam3 \
  --hf-cache-seed-dir /path/to/shared/hf-cache \
  --run-id l4t-sam3
```

Add `--dry-run` to print the resolved child commands, or `--verbose` to retain
detailed runner output.

Results are written below:

```text
${TRTMC_CHECK_STORAGE_ROOT}/results/<run-id>/
```

Resume with the same selection, intent, and run ID. Interrupted and retryable
cases run again; terminal cases remain intact even when HEAD has advanced:

```bash
$TRTMC_CHECK_PYTHON tools/model_checks.py run \
  --platform gb300 \
  --task accuracy \
  --all \
  --run-id gb300-accuracy-all \
  --resume
```

Resume is task-aware. Accuracy uses `--resume-existing` only when its own
`accuracy/run.json` exists; an Accuracy task that never initialized starts
normally in the existing campaign. Perf follows the same rule using its own run
metadata. A failed combined campaign retains `task_source_identity` for each
independently successful task so downstream supervision can reuse valid task
evidence without treating the whole campaign as passed.

After changing code that affects one model, invalidate that model as a unit so
all of its selected Accuracy and Perf evidence is regenerated on the current
HEAD:

```bash
$TRTMC_CHECK_PYTHON tools/model_checks.py run \
  --platform gb300 \
  --all \
  --run-id gb300-all \
  --resume \
  --invalidate-model distilgpt2
```

`--invalidate-model` is repeatable. It does not discard other models' evidence.
Managed stale engines and Perf bundles are rebuilt automatically when their
recorded source revision differs; Accuracy's lower-level `--force-build` remains
available for cache diagnosis, but forcing a build does not replace model-level
evidence invalidation.

## Shard a campaign

Sharding is an opt-in runner capability; no checked-in CI workflow enables it.
Run the same selection and run ID on independent hosts or GPUs that share the
campaign results directory. The index is zero-based:

```bash
$TRTMC_CHECK_PYTHON tools/model_checks.py run \
  --platform gb300 --all --run-id gb300-all --shard 0/2

$TRTMC_CHECK_PYTHON tools/model_checks.py run \
  --platform gb300 --all --run-id gb300-all --shard 1/2
```

Each worker writes only below `shards/<index>-of-<count>/`. Run exactly one
consolidator to publish the campaign-level Accuracy and Perf reports while the
workers are active:

```bash
$TRTMC_CHECK_PYTHON tools/model_checks.py consolidate \
  "$TRTMC_CHECK_STORAGE_ROOT/results/gb300-all" --watch
```

The immutable campaign inventory determines assignment by case order modulo
the shard count. Resume one failed shard with the same selection, `--shard`,
run ID, and `--resume`; use the same `--invalidate-model` on every shard that
owns cases for the changed model. A shard uses its own writable Perf bundle cache, so
independent workers never build into the same cache directory.

## Platforms

Use `--platform gb300`, `--platform auto-thor`, or `--platform l4t-thor`.
Platform files under `tests/model_checks/platforms/` define task order and
model exclusions. An excluded model runs no Accuracy suite or Perf entry and
does not appear in qualification reports. Files under
`tests/model_checks/environments/` define paths and retention.

Checked-in environments delete Accuracy engines and Perf bundles after each
binding. GB300 and Auto Thor retain their shared Hugging Face cache. L4T Thor
deletes its isolated model cache and requires `TRTMC_CHECK_STORAGE_ROOT` on
`/dev/nvme0n1p1`.

See `benchmarks/performance/README.md` for the L4T TensorRT 11 bare-metal build.

## Add a model or dataset

- Define an Accuracy suite in `tests/validation/workloads.yaml`.
- Assign one or more suites in `tests/validation/model_workloads.yaml`.
- Add the Perf entry independently in `benchmarks/performance/release.yaml`.
- Add platform-wide model exclusions under `tests/model_checks/platforms/`.

Each Accuracy `MODEL=SUITE` binding has its own engine directory. Perf bundles
remain independent from Accuracy engines.

<!-- Collaborative review anchor: batch 2. -->
