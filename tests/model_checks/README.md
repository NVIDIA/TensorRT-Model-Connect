# Model checks

`tools/model_checks.py` is the small model-first entry point shared by local
runs and future CI callers. It resolves model profiles into independent
Accuracy and Perf bindings; the task runners and their native configuration
remain separate.

No NAS publishing or CI job is enabled by this tool.

## Selection

Use every Accuracy benchmark configured for the model and every matching Perf
entry:

```bash
python tools/model_checks.py check \
  --platform gb300 \
  --model qwen25vl-3b
```

Auto Thor uses the same model-first interface with its own platform and
execution-environment profiles:

```bash
python tools/model_checks.py check \
  --platform auto-thor \
  --model qwen25vl-3b
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
every selected model. `--accuracy-binding` is the exact per-model form and can
select a suite that is globally configured but intentionally omitted from the
model's normal `workloads` list.

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
  --run-id qwen25vl-smoke \
  --dry-run
```

Remove `--dry-run` to execute. Tasks are launched in the platform profile's
order. Accuracy receives exact `MODEL=SUITE` bindings and Perf receives exact
entry IDs, so the two task configurations cannot silently broaden selection.
Normal execution prints a compact run header, task progress, errors, artifact
paths, and a final Accuracy/Perf status summary. Add `--verbose` when full
child, TRTMC, baseline, and reproduction commands are needed for debugging.

The unified runner does not depend on Python profiles baked into a container
image. It points both task runners at the shared
`${TRTMC_CHECK_STORAGE_ROOT}/python-profiles` cache and allows the existing
profile resolver to create missing environments. Before each Accuracy binding,
the resolver derives the required common, model-family, and suite-scoring
profiles from the selected model and workload. A matching ready environment is
reused; otherwise a fingerprinted virtual environment is created under the
shared cache, exact locked package versions are installed and verified, and
only then does model execution start. Perf uses the same shared cache for any
candidate-build or baseline profiles it declares. Creation is protected by a
file lock, so later runs safely reuse the environment.

`storage.python_profiles_root` may override the derived location in a custom
execution-environment YAML. The resolved directory must remain below that
environment's managed storage root. `request.json` records the resolved path.

Resume with the original selection, platform, environment, and run ID:

```bash
python tools/model_checks.py run \
  --platform l4t-thor \
  --all \
  --run-id model-check-l4t-thor-nightly \
  --resume
```

The stored request must match. Accuracy keeps exact terminal binding results;
Perf delegates to its native fingerprint- and revision-checked `resume` mode.

The checked-in platform environments keep Accuracy and Perf artifacts
separate. Accuracy isolates engines by exact `MODEL=SUITE` binding because a
suite's dataset can change static shapes, optimization profiles, or the
dataset-derived cache length. It may still share the HF cache per model. Perf
keeps entry-scoped work and its own bundle cache. The checked-in GB300, L4T,
and Auto Thor environments delete each managed Accuracy engine and Perf bundle
after its binding finishes, including failures, while retaining the shared HF
cache. They do not enforce a fixed free-space reserve; unattended jobs can add
one explicitly when the runner's disk capacity and workload peak are known.

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

On L4T Thor, `TRTMC_CHECK_STORAGE_ROOT` must be on the filesystem backed by
`/dev/nvme0n1p1`. The run compares device identities and is rejected before
runner launch when the storage root is on another filesystem.

## Adding coverage

- Add or update the Accuracy suite and dataset recipe in
  `tests/validation/workloads.yaml`. Give dataset variants separate suite IDs;
  the suite ID is part of the Accuracy engine-isolation boundary.
- Add that suite to the model profile's `workloads` in
  `tests/validation/model_workloads.yaml` when normal model and all-model runs
  should include it. Leave it out of the model list when it should remain an
  explicit-only experiment.
- Add the model's task metadata under `tests/e2e/models/` when it is a new
  ready model profile.
- Add its independent Perf entry to
  `benchmarks/performance/release.yaml` when Perf applies.
- Add only evidence-backed hardware exclusions to the sparse platform profile
  under `tests/model_checks/platforms/`.

There is no additional model roster to synchronize. A model may have any
number of Accuracy suites, while Perf remains a separate list of concrete
entries.

`--all` means every model's configured Accuracy workloads and every configured
Perf entry. A profile that exists only in Accuracy or only in Perf does not
become a missing-configuration blocker for the other task. Explicit `--model`
selection remains strict and reports a missing selected-task configuration as
a blocker.
