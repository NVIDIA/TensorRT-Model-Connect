<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Performance matrix

`tools/perf_matrix.py` compares TRTMC with the reference backend declared by each
entry in `release.yaml`. TRTMC measurements always use `trtmc-bench`; reference
frameworks run in separate Python processes and do not add dependencies to
`trtmc-bench`.

`tools/performance/catalog.py` owns suite loading, profile expansion, release
coverage, exclusions, and entry/model/family selection. `perf_matrix.py` owns
execution, evidence, classification, resume, and reporting. Other orchestration
tools consume the catalog and the public `perf_matrix.write_report()` interface
instead of importing runner internals.

The suite contains one row for every release-relevant single-process model
profile marked `ready` in the benchmark catalog. Profiles whose names contain an
`l0` segment are shorter PR-smoke duplicates and are deliberately excluded.
Other temporary omissions must be named under `excluded_profiles` with a reason.
The suite currently has 107 model-profile comparisons across 77 families and 78
`(family, operation)` contracts because some families expose multiple profiles
and `eagle_vlm` exposes both `embed` and `rerank`. Catalog profiles marked
`distributed` require their own multi-process launch and are not silently
included in this single-GPU matrix.

## Commands

These commands use the checked-in GB300 CI environment. Export the four
`TRTMC_PERF_*` paths listed under [Environment configuration](#environment-configuration)
before running them; `check` performs the same worker, storage, bundle, and
reference preflight as `run`.

Validate the complete matrix without running performance measurements:

```bash
python3 tools/perf_matrix.py check \
  benchmarks/performance/release.yaml \
  --environment benchmarks/performance/environments/gb300.yaml
```

Run the complete matrix:

```bash
python3 tools/perf_matrix.py run \
  benchmarks/performance/release.yaml \
  --environment benchmarks/performance/environments/gb300.yaml
```

Run one exact matrix entry:

```bash
python3 tools/perf_matrix.py run \
  benchmarks/performance/release.yaml \
  --environment benchmarks/performance/environments/gb300.yaml \
  --entry gpt2.generate
```

Run every release entry bound to one or more canonical model names:

```bash
python3 tools/perf_matrix.py run \
  benchmarks/performance/release.yaml \
  --environment benchmarks/performance/environments/gb300.yaml \
  --model distilgpt2 \
  --model qwen3-0.6b-fp8
```

`--model-selection FILE` accepts the owner/family JSON emitted by
`tools/model_ci.py` and expands every selected owner to its task-owned model
profiles and release entries. `--entry`, `--model`, and `--model-selection`
are mutually exclusive.

An additional profile under the same family-operation contract has a
profile-qualified entry ID:

```bash
python3 tools/perf_matrix.py run \
  benchmarks/performance/release.yaml \
  --environment benchmarks/performance/environments/gb300.yaml \
  --entry qwen.generate@qwen3-0.6b-fp8
```

Continue an interrupted run and automatically retry incomplete or failed entries:

```bash
python3 tools/perf_matrix.py resume artifacts/perf/<run-id>
```

If bundle preparation ran before the matrix campaign, attach its receipt and
regenerate the report:

```bash
python3 tools/perf_matrix.py report artifacts/perf/<run-id> \
  --preparation-receipt artifacts/perf/bundle-preparation.json
```

`--entry` selects an exact matrix entry ID. Base contract rows use
`family.operation`; additional profiles use `family.operation@model-profile`.
It is not a model input or testcase name. The default is the complete matrix.

## Suite configuration

The suite owns the comparison semantics. Every entry explicitly declares:

- the family, operation, and model profile;
- its workload source;
- warmup and measured iteration counts;
- its reference backend and mode;
- the measured timing boundary;
- the output-equivalence contract;
- the green/yellow/red equivalence margin.

The current workload source is an explicitly named model testcase:

```yaml
- id: gpt2.generate
  family: gpt2
  operation: generate
  model: distilgpt2
  workload:
    testcase: distilgpt2
    request:
      max_new_tokens: 12
```

`request` is optional and overrides the named testcase. The first testcase is
never selected implicitly. A future benchmark-dataset source can resolve several
samples inside one entry without adding report rows. Dataset support is not
implemented by the current script.

Output equivalence is task-specific. LocateAnything uses the `localization`
contract: both sides must emit valid `<ref>` plus homogeneous box or point
groups, with the same output type and count. Boxes are compared by minimum IoU,
points by maximum distance in the normalized 0..1000 coordinate space, and the
complete answers retain a bounded normalized text distance. Different hidden
EOS handling or token IDs therefore do not invalidate geometrically equivalent
structured outputs.

The first profile for a family-operation declares the complete reviewed
comparison contract. Further catalog profiles name that contract explicitly and
may override only profile-specific settings:

```yaml
additional_profiles:
  - model: qwen3-0.6b-fp8
    inherit: qwen.generate
```

The resolved row uses `qwen3-0.6b-fp8` as both its model profile and testcase,
while retaining the reviewed Qwen timing, reference, and output contracts. A
profile with different replay inputs or reference assets declares those
overrides in the same block.

A ready profile can be omitted only with an explicit reason:

```yaml
excluded_profiles:
  - model: model-profile-name
    reason: Excluded while its documented single-process blocker is unresolved.
```

The coverage check requires every non-L0 ready single-process catalog profile to
appear exactly once or in `excluded_profiles`. It rejects L0 entries, unknown
exclusions, and profiles that are both configured and excluded.

Suite-level `defaults.measurement` avoids repeating warmup and iteration counts.
The fully resolved workload and measurement values are recorded in `results.json`.
Reference runners must not supply hidden measurement defaults.

## Environment configuration

The environment file owns machine-specific execution settings:

- `trtmc-bench`, worker, and reference runner paths;
- bundle cache, bundle roots, and runtime directories;
- results and scratch roots;
- command timeout and local-files-only mode;
- minimum free disk space.

`storage.bundle_retention` accepts `retain`, `delete_on_pass`, or
`delete_always`. Deletion is limited to the resolved bundle/engine file below
the configured managed `bundle_cache`; sibling build evidence such as
`build.stdout.log`, `build.stderr.log`, and `build-timing.json` is retained for
diagnosis. Bundles found in external roots are preserved.
`execution.hf_cache_mode` is `shared` or `per_entry`, and
`execution.hf_cache_retention` uses the same retention values. A shared HF
cache can only be retained. With `per_entry`, failed entry work is preserved by
`delete_on_pass` for diagnosis. Shared-cache runs also preserve failed entry
scratch while removing successful entry scratch. Optional
`storage.storage_root` rejects results, scratch, and managed bundle paths
outside that filesystem before execution. The checked-in platform environments
use `delete_always` for managed bundles and omit the optional fixed free-space
reserve; a host-specific environment may opt into either policy differently.

An individual baseline can set `local_files_only: true` when that reference must
use an already-provisioned model snapshot even if the rest of the matrix may
access its configured model source.

The checked-in `environments/gb300.yaml` is the CI configuration. Repository
files use repository-relative paths. Private Internal CI supplies stable runner
paths to its performance workflow through these repository variables:

```text
TRTMC_PERF_WORKER
TRTMC_PERF_BUNDLE_CACHE
TRTMC_PERF_BUNDLE_ROOTS
TRTMC_PERF_RUNTIME_DIRS
```

Bundles below the managed cache are reusable only when their cache key matches
the current model manifest, build options and assets, TensorRT platform, and
generic plus family-owned builder sources. A stale managed bundle discovered
through a bundle root is preserved but ignored while the current bundle is
built. Bundles outside the managed cache are explicit prebuilt inputs and are
therefore trusted as supplied.

The script expands these values before preflight and records the resolved
environment, source path, and configuration SHA-256 in `results.json`. Missing
required values fail before any model runs. Another machine can use a separate
environment YAML without changing the suite.

Reference-specific upstream checkout paths remain process environment inputs:

```text
TRTMC_ELF_REFERENCE_REPO
TRTMC_LANCE_REFERENCE_REPO
TRTMC_SANA_WM_REFERENCE_REPO
PERSONAPLEX_OFFICIAL_REPO
```

CI should prebuild the selected Python profiles and set
`TRTMC_PYTHON_PROFILE_PREBUILT_ONLY=1`. Dependency installation is outside the
measured campaign.

### L4T Thor native build

Build the shared Accuracy/Perf binaries from the selected source revision and
TensorRT 11 installation. L4T does not always put `nvcc` on `PATH`, so pass its
absolute path; otherwise CMake omits model-owned CUDA sources and can produce an
incomplete runtime plugin.

```bash
cmake -S . -B "$TRTMC_CHECK_BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DTRTMC_BUILD_BACKEND_TRT=ON \
  -DTRTMC_BUILD_BENCHMARKS=ON \
  -DTRTMC_TRT_INCLUDE_DIR="$TRT_ROOT/include" \
  -DTRTMC_TRT_LIBRARY="$TRT_ROOT/lib/libnvinfer.so"
cmake --build "$TRTMC_CHECK_BUILD_DIR" --parallel
```

## Preflight and timing contract

`check` and `run` perform the same preflight:

1. validate suite coverage and configuration;
2. expand and validate the environment;
3. verify free disk space and required executables;
4. verify that the candidate worker is a Release build from the requested source
   revision;
5. resolve every selected `trtmc-bench` testcase;
6. validate candidate and reference timing contracts.

The reference implementation defines the measured boundary for each entry. The
suite records whether input preparation and external asset loading are timed.
TRTMC resolves one matching public timing scope. A scope, output, workload, or
compile-mode mismatch produces no performance light.

Green, yellow, and red are valid comparison results and return exit code zero.
Configuration errors, command failures, incomplete measurements, and contract
mismatches return a nonzero exit code. CI and manual execution use this same rule;
there is no CI-specific execution mode.

Suite, environment, storage, and candidate-worker provenance errors fail before
model execution. A candidate or reference preflight failure that belongs to one
matrix entry is instead recorded as a failed row, and `run` continues with the
remaining ready entries. `check` reports the ready and failed entry counts and
returns nonzero when any selected entry is not ready.

## Results and reproduction

Each new run creates a unique directory below `storage.results_root`. The final
directory contains:

```text
artifacts/
assets/
report.json
results.json
report.html
```

`results.json` remains the internal resume and execution record. `report.json`
is the public, queryable qualification snapshot and contains only selected
entries; catalog and platform exclusions are absent. `report.html` is a static
frontend that fetches `report.json`; it does not calculate counts, classify a
case, scan scratch directories, or reconstruct links. The schema and renderer
are published under `assets/`.

Green, yellow, and red mean the candidate and reference completed a valid
comparison. White means a selected terminal case did not form a valid
comparison. Pending and running entries have no light. The header therefore
shows `Comparable results` (green/yellow/red) separately from equally prominent
`Operational coverage` (comparable/selected plus white). Platform exclusions
contribute to neither denominator. A targeted run publishes only the targeted
entries even though the internal `results.json` retains the complete matrix for
resume.

Each executed candidate/reference command writes complete stdout and stderr to
`artifacts/<case>/logs/`; `report.json` exposes only relative links to those real
files. A preflight failure that launched no subprocess receives a published
diagnostic log containing the captured failure and replay command. Compute
precision identifies Reference and TRTMC independently, output validation is
shown as the prerequisite for latency comparison, and Reference/TRTMC p50
latencies are separate fields. Metrics, Logs, and Commands remain separate
report entries. The run environment snapshot is published at
`artifacts/run/environment.json`.

For valid comparisons, `report.json` publishes `measurement_stability`. It
checks the ten raw samples on each side: the first-five and last-five medians
must differ by at most 5%, and at least eight samples must lie within 5% of that
side's median. A settled first measurement is classified immediately. An
unsettled measurement starts both Reference and TRTMC once more in fresh
processes while reusing the prepared bundle. If the second measurement remains
unsettled, the case is white (`measurement_inconclusive`) and is excluded from
the comparable-result denominator. Metrics retains both measurements' raw
samples, and Logs and Commands retain the four leaf executions. Legacy results
without enforced evidence continue to render their stored shadow analysis.

The preparation receipt uses schema `trtmc.perf-bundle-preparation/v1`, scope
`test_task`, and the performance run's exact `git_commit`. Each entry under
`bundles` records `model`, the final `bundle` path used by the campaign,
`status`, `build_time_s`, and `included_in_performance_metrics: false`. The
report command rejects a revision mismatch, invalid build time, duplicate
record, or bundle that the campaign did not use. A matching task-level record
takes precedence over the candidate command's later cache lookup, so a bundle
rebuilt during preparation is reported as `Built`, not `Existing bundle`.

Each report row shows the exact leaf commands that were executed:

- the actual `trtmc-bench` candidate command;
- the actual HF eager, torch.compile, or task-reference command;
- the working directory.

These are recorded argv values, not a generated replay script or another
`perf_matrix.py` invocation. Temporary backend outputs live under the configured
scratch root and are removed after a normally completed run. Shared bundle,
runtime, Python-profile, and Hugging Face caches are reused but never deleted by
the script.

## CI

Controlled Internal CI can run one `entry` or the complete matrix without
adding another script mode. The unique run directory under `artifacts/perf/`
remains a private artifact; Source Actions and Pages do not publish the raw
performance report.

## Adding an entry

Adding a new family operation requires a complete base suite entry. Adding a
profile to an existing family operation requires an `additional_profiles` entry
that names the base contract. Choose an explicit testcase workload and review any
profile-specific replay, precision, and reference overrides. The suite coverage
check fails until every ready catalog profile is present exactly once.

Use the shared `hf-transformers` runner for supported encoder, causal-LM, and
seq2seq-LM execution. Use the existing `task-reference` runner for Diffusers, ASR,
TTS, VLM, embedding, reranking, vision, time-series, Qwen3-Omni, PersonaPlex, ELF,
Lance, or SANA-WM workloads. Do not silently substitute HF eager for a declared
torch.compile entry.

<!-- Collaborative review anchor: batch 2. -->
