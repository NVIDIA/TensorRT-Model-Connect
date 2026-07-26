<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Performance matrix

`tools/perf_matrix.py` compares TRTMC with the reference backend declared by each
entry in `release.yaml`. TRTMC measurements always use `trtmc-bench`; reference
frameworks run in separate Python processes and do not add dependencies to
`trtmc-bench`.

The suite contains one row for every release-relevant single-process model
profile marked `ready` in the benchmark catalog. Profiles whose names contain an
`l0` segment are shorter PR-smoke duplicates and are deliberately excluded. The
suite currently has 105 model-profile comparisons across 76 families and 77
`(family, operation)` contracts because some families expose multiple profiles
and `eagle_vlm` exposes both `embed` and `rerank`. Catalog profiles marked
`distributed` require their own multi-process launch and are not silently
included in this single-GPU matrix.

## Commands

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
overrides in the same block. The coverage check requires every non-L0 ready
single-process catalog profile exactly once and rejects L0 entries in the suite
as extras.

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

The checked-in `environments/gb300.yaml` is the CI configuration. Repository files
use repository-relative paths. Stable runner paths are supplied through the same
GitHub repository variables already exported by the performance workflow:

```text
TRTMC_PERF_WORKER
TRTMC_PERF_BUNDLE_CACHE
TRTMC_PERF_BUNDLE_ROOTS
TRTMC_PERF_RUNTIME_DIRS
```

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
directory contains only:

```text
results.json
report.html
```

`results.json` contains resolved configuration, raw samples, timing policies,
commands, bundle-preparation status and build time, and bounded diagnostic
output. `report.html` shows the family matrix, both infer-time p50 values,
TRTMC bundle preparation, measured scopes, and traffic lights. Bundle build time
is reported for run observability but is excluded from the infer-time comparison.

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

`.github/workflows/performance.yml` is manually dispatchable and callable from
another workflow. A manual dispatch may provide one `entry`; without it the job
runs the complete matrix. A nightly workflow can call the same job without adding
another script mode. The workflow uploads the unique run directory under
`artifacts/perf/` with 30-day retention.

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
