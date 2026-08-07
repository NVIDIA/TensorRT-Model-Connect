# Model checks

`tools/model_checks.py` is the small model-first entry point shared by local
runs and future CI callers. It resolves model profiles into independent
Accuracy and Perf bindings; the task runners and their native configuration
remain separate.

No NAS publishing or CI job is enabled by this tool.

## Selection

Use each model's default Accuracy suite and every matching Perf entry:

```bash
python tools/model_checks.py check \
  --platform gb300 \
  --model qwen25vl-3b
```

Run every Accuracy suite configured for one model:

```bash
python tools/model_checks.py check \
  --platform l4t-thor \
  --model qwen25vl-3b \
  --all-accuracy-suites
```

Select heterogeneous model/suite pairs without applying a shared suite list
to every model:

```bash
python tools/model_checks.py check \
  --platform gb300 \
  --model qwen25vl-3b \
  --model gpt2-125m \
  --accuracy-binding qwen25vl-3b=vlm_mmmu_pro_vision_mcq \
  --accuracy-binding gpt2-125m=mmlu_continuation_parity
```

`--accuracy-suite` is repeatable when the same selected suite set is valid for
every selected model. `--accuracy-binding` is the exact per-model form.

## Run

The checked-in execution environment uses machine paths supplied through
environment variables, so credentials and host-specific build locations do
not enter source control:

```bash
export TRTMC_CHECK_STORAGE_ROOT=/runs
export TRTMC_CHECK_DATASET_ROOT=/mnt/data
export TRTMC_CHECK_RUNTIME_ROOT=/runs/tmp/build-trt112
export TRTMC_CHECK_PYTHON=/opt/venv/bin/python

export TRTMC_PERF_WORKER=/runs/tmp/build-trt112/trtmc_benchmark_worker
export TRTMC_PERF_BUNDLE_CACHE=/runs/engines/perf
# ':' expands to no external roots; never recursively expose the managed cache.
export TRTMC_PERF_BUNDLE_ROOTS=:
export TRTMC_PERF_RUNTIME_DIRS=/runs/tmp/build-trt112

python tools/model_checks.py run \
  --platform gb300 \
  --model qwen25vl-3b \
  --all-accuracy-suites \
  --run-id qwen25vl-smoke \
  --dry-run
```

Remove `--dry-run` to execute. Tasks are launched in the platform profile's
order. Accuracy receives exact `MODEL=SUITE` bindings and Perf receives exact
entry IDs, so the two task configurations cannot silently broaden selection.

The checked-in platform environments keep Accuracy and Perf artifacts
separate. Accuracy isolates engines by exact `MODEL=SUITE` binding because a
suite's dataset can change static shapes, optimization profiles, or the
dataset-derived cache length. It may still share the HF cache per model. Perf
keeps entry-scoped work and its own bundle cache. The default GB300 and L4T
Accuracy policy deletes each passing binding's engine and retains the shared HF
cache. The L4T Perf policy does the same for a managed bundle; the GB300 Perf
default remains `retain` for compatibility with existing jobs.

Native runner policies are independent:

| Resource | Isolation | Retention values |
| --- | --- | --- |
| Accuracy engine | per model/suite binding | `retain`, `delete_on_pass`, `delete_always` |
| Accuracy HF cache | shared or per model | same; shared requires `retain` |
| Perf bundle | per entry/cache fingerprint | same |
| Perf HF cache | shared or per entry | same; shared requires `retain` |

This supports “delete only TRT engines”, “delete both engine and isolated HF
cache”, and “delete passing artifacts but preserve failed artifacts” without
sharing fingerprints between Accuracy and Perf.

On L4T Thor, `TRTMC_CHECK_STORAGE_ROOT` must resolve below
`/dev/nvme0n1p1`. The run is rejected before runner launch otherwise.

## Adding coverage

- Add or update the Accuracy suite and dataset recipe in
  `tests/validation/workloads.yaml`. Give dataset variants separate suite IDs;
  the suite ID is part of the Accuracy engine-isolation boundary.
- Add that suite to the model profile's `workloads` in
  `tests/validation/model_workloads.yaml`; set `default` only when the default
  should change.
- Add the model's task metadata under `tests/e2e/models/` when it is a new
  ready model profile.
- Add its independent Perf entry to
  `benchmarks/performance/release.yaml` when Perf applies.
- Add only evidence-backed hardware exclusions to the sparse platform profile
  under `tests/model_checks/platforms/`.

There is no additional model roster to synchronize. A model may have any
number of Accuracy suites, while Perf remains a separate list of concrete
entries.
