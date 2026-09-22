<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Performance qualification

This package is repository-only CI qualification. The installed user benchmark
remains `trtmc-bench` under `apps/benchmark`; it does not import this package.

## Ownership

- `matrix.py` selects and resumes multi-model campaigns.
- `runner.py` executes one resolved qualification case.
- `reference_protocol.py` defines the Accuracy/Performance reference command.
- `reference_harness.py` provides model-agnostic timing and result writing for
  family-owned references.
- `reporting.py` produces `report.json` and `report.html`.
- `families/<family>/tests/benchmark/*.yaml` owns model cases, datasets,
  metrics, gates, reference environment, and measurement settings.
- `families/<family>/tests/benchmark/reference.py` owns reference behavior that
  cannot be expressed by a framework-generic adapter.

Shared code must not select behavior from a family name. Adding a normal family
case changes only the family directory. Campaign configuration may select cases
or hardware, but does not redefine model behavior.

## Execution

The candidate runs through the installed `trtmc-bench` executable. The
reference runs in the Python environment prepared for its family, so conflicting
reference dependencies do not affect the candidate or another family.

Run a checked-in campaign entry with:

```bash
python3 -m qualification_tests.benchmark_qualification.performance check \
  qualification_tests/benchmark_qualification/performance/config/release.yaml \
  --environment qualification_tests/benchmark_qualification/performance/config/environments/gb300.yaml \
  --entry gpt2.generate

python3 -m qualification_tests.benchmark_qualification.performance run \
  qualification_tests/benchmark_qualification/performance/config/release.yaml \
  --environment qualification_tests/benchmark_qualification/performance/config/environments/gb300.yaml \
  --entry gpt2.generate
```

Resume a run or regenerate its report with:

```bash
python3 -m qualification_tests.benchmark_qualification.performance resume <run-directory>
python3 -m qualification_tests.benchmark_qualification.performance report <run-directory>
```

`tools/model_benchmark.py` is the thin per-model CI entry point. It discovers
family profiles and invokes the same single-case runner; it does not build a
temporary campaign or start a nested matrix process.

## Reference contract

A reference declares exactly one of:

```yaml
reference:
  adapter: hf-transformers-asr
```

or:

```yaml
reference:
  script: tests/benchmark/reference.py
```

Framework-generic adapters receive only the declared adapter and parameters.
Family scripts receive their own family identity but shared code never branches
on it. Both return `trtmc.perf-baseline/v1` with one finite positive timing
sample per measured iteration and the output summary required by the declared
contract.

Reference model loading occurs before warmup. The three timing fields
`timing_scope`, `input_preparation_included`, and `asset_loading_included` must
match the profile. Model load, compilation, and warmup are excluded from timed
samples.

## Results

Each run retains raw commands, logs, artifacts, and `results.json`, then writes:

- `report.json` for automation;
- `report.html` for inspection.

An output-contract mismatch is a model `failed` result. Reference/candidate
command failures, invalid reports, missing artifacts, and other execution
problems are `error`, not model gate failures.
