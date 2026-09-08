<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Qualification

`trtmc-qualify` discovers optional Accuracy and Performance configuration files
directly from model families. There is no central model-to-suite registry.

A model opts in by adding one of these files:

```text
families/<family>/tests/qualification/<model>.accuracy.yaml
families/<family>/tests/qualification/<model>.performance.yaml
```

Merely having an E2E manifest, including an L0 manifest, does not opt a model
into qualification. One model file may contain multiple suites, and every suite
may contain multiple cases.

Each family also owns `tests/qualification/executor.py`. The high-level runner
starts it in a fresh process for every case and understands only the result
envelope. Dataset interpretation, reference execution, metrics, and gates remain
inside the family.

An optional family `tests/qualification/data/` directory is only for small,
deterministic fixtures, sample indexes, or scorer metadata that can be committed
legally. Full benchmark datasets, model weights, and run results do not belong
there. The environment provides the external dataset root through
`storage.data_root`, and results are written only below the requested run output.

## Model file shape

A shared benchmark definition describes model-independent dataset and scoring
behavior once. A model selects it by name and keeps its reference, prompt,
workload, and gate settings in the family-local file:

```yaml
schema_version: trtmc.qualification/v1
model: gpt2-125m
kind: accuracy
suites:
  - benchmark: mmlu_continuation_parity
    gate_policy: blocking
    cases:
      - id: smoke
        sample_limit: 10
        reference:
          implementation: hf_transformers
          precision: fp32
        prompt:
          token_limit: 960
          truncation_side: left
        gate:
          min_pass_rate: 0.9
          min_allowed_failures: 1
        candidate:
          testcase: gpt2-125m
          request:
            max_new_tokens: 64
            top_k: 1
```

Shared definitions live in `apps/benchmark/qualification/suites/<name>.yaml`.
They do not list models or devices. If a benchmark's meaning changes, add a new
descriptive name instead of changing the existing definition. The high-level
runner passes the definition and model-owned case to the family executor without
interpreting their contents.

## Commands

Show the immutable plan without loading a model:

```bash
trtmc-qualify plan --kind accuracy --families-root families
```

Run every discovered Accuracy case:

```bash
trtmc-qualify run \
  --kind accuracy \
  --families-root families \
  --environment apps/benchmark/qualification/environments/example.yaml \
  --output artifacts/qualification/accuracy
```

The equivalent checked-in high-level run configuration is:

```bash
trtmc-qualify run \
  --run-config apps/benchmark/qualification/runs/all-accuracy.yaml \
  --families-root families \
  --output artifacts/qualification/accuracy
```

The run configuration owns the environment assignment and may contain exact
`models`, `suites`, or `cases` lists for a special device job. Omitting those
lists selects everything discovered, so an ordinary new model does not require
a high-level configuration change.

Select exact names when a smaller run is needed:

```bash
trtmc-qualify run \
  --kind accuracy \
  --families-root families \
  --model gpt2-125m \
  --suite mmlu_continuation_parity \
  --case smoke \
  --environment apps/benchmark/qualification/environments/example.yaml \
  --output artifacts/qualification/gpt2
```

The model, suite, and case options use exact names. They do not use tags or
pattern matching. Omit all three to run everything that was discovered.

Continue a run using its stored plan and environment:

```bash
trtmc-qualify resume artifacts/qualification/accuracy
```

Completed pass/fail results are terminal. Missing or malformed results are run
again; an `execution=error` result is archived under the case's `attempts/`
directory before that case is retried.

Regenerate the machine-readable and HTML reports without executing cases:

```bash
trtmc-qualify report artifacts/qualification/accuracy
```

`report.json` is authoritative. `report.html` renders the same data. Blocking
cases contribute pass/fail/error status. Observation-only cases are reported
separately and can never claim a pass or fail verdict. If no model has opted
into the selected assessment, the valid empty plan is reported as `empty` and
does not create a coverage failure.
