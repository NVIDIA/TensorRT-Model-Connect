<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Qualification

The current implementation covers GPT-2 continuation parity, Chronos-Bolt ETTh1
numeric parity, and compiled PyTorch-reference-vs-TensorRT Performance. It does
not restore all historical datasets or claim MMLU answer accuracy.

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

Execution requires POSIX process groups. The application owns each executor and
environment-preparation process group, stops its descendants on timeout or
SIGINT/SIGTERM, and does not advance while the parent is still running. Executors
must not detach workers into separate sessions.

## Python environments and preparation

The scheduler does not import family code or model dependencies. Executors use
`tools.python` from the machine environment, defaulting to the scheduler's Python
when omitted. A family may supply `tests/qualification/prepare_environment.py`;
no central registration or per-model environment setting is needed.

The optional script accepts `--request FILE --output FILE`. Its JSON request
contains `common_python`, `family_root`, selected `cases`, a writable
`environment_directory`, and `allow_create`. It returns `python` and optionally
`reference_python` (default: the same Python). These are actual interpreter
paths, including virtual-environment symlinks. The family owns installation and
compatibility checks. Reuse root `requirements.txt` for build dependencies.
Reference-only environments need not install the TRTMC builder package.

GPT-2 reuses a CUDA-capable common environment. If the common environment is not
compatible, its preparation script can create a private reference environment
with Torch 2.14.0 and Transformers 5.2.0 while retaining the common build Python.
Chronos-Bolt requires Transformers 4.57.6 and Chronos Forecasting 2.2.2 for both
conversion and the official reference. Its preparation script therefore returns
one family-private interpreter for both roles when the common environment does
not match those exact dependencies. Set `execution.allow_environment_creation:
true` explicitly during preparation to permit installation. Other families own
their own dependency choices.

The application prepares each selected family's environment once, records
actual package versions, and starts executors with their resolved Python. The
family executor invokes the benchmark/build CLI with that interpreter too.
Reference subprocesses do not inherit another environment's `PYTHONPATH` or
user-site packages. Family preparation must not modify the common environment.

All case preparation finishes before evaluation. Each implemented family resolves
an immutable HF snapshot and builds a fresh run-owned bundle through the public
benchmark CLI. Accuracy also validates the dataset and saves the selected samples
in the run's preparation directory before building. Evaluation and resume read
this saved selection, never a potentially changed external dataset. Chronos-Bolt
also verifies the ETTh1 checksum recorded by the historical benchmark contract.
New runs require `execution.allow_build: true`; prepared runs reuse their
recorded bundle without rebuilding during measurement. Preparation errors are
family/case-local and do not stop independent selected work.

Prepare without running Accuracy or Performance, then execute the prepared work:

```bash
trtmc-qualify prepare --kind performance --families-root families \
  --model gpt2-125m \
  --environment apps/benchmark/qualification/environments/example.yaml \
  --output artifacts/qualification/gpt2-performance
trtmc-qualify resume artifacts/qualification/gpt2-performance
```

`prepare` writes `preparation.json`, not an Accuracy/Performance report. `run`
performs both stages. The executor receives `phase: prepare`, `check`, or `run`;
preparation returns family-owned `details.prepared`. Its optional `input_files`
list identifies files whose presence, size, modification time, change time, and
filesystem identity the application checks before reuse. These are local file
state checks, not content authentication. Runs and prepared files must remain in
their original trusted storage. `check` verifies family-specific readiness without
rebuilding or measuring.

Accuracy is a conversion check, not a standalone Hugging Face score. The family
executor runs the Hugging Face model as the reference, runs the same inputs
through the converted TensorRT bundle, and compares the two outputs. A report is
invalid if it cannot identify the converted bundle used by the candidate.

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

Performance uses the same discovery path. Every Performance case measures both
the converted TensorRT bundle and its reference backend with the same workload.
GPT-2 compiles the official model's `forward` method with `torch.compile`.
Chronos-Bolt prefers the same compiled reference and falls back to eager official
PyTorch when compilation, output parity, or the two-attempt stability protocol
cannot produce a valid compiled comparison. Generated token IDs must match for
GPT-2; Chronos-Bolt requires the forecast tensors to satisfy its numeric-parity
gates with the selected reference. The report retains every attempted mode and
states whether fallback was used. A timing comparison is accepted only after
that conversion check. The two sides run sequentially on the GPU selected by the
high-level environment; model files never select or exclude a device.

Until a high-level device run owns a threshold, the timing comparison is
observation-only and does not claim a pass or fail verdict:

```yaml
schema_version: trtmc.qualification/v1
model: gpt2-125m
kind: performance
suites:
  - benchmark: text_generation_performance
    gate_policy: observation_only
    cases:
      - id: generate_64
        reference:
          implementation: hf_transformers
          mode: torch-compile
          compile_scope: model.forward
          precision: fp32
        candidate:
          testcase: gpt2-125m
          request:
            prompt: The capital of France is
            max_new_tokens: 64
          measurement:
            warmup: 5
            iterations: 10
```

The report records both raw latency samples, both metric sets, compile evidence,
the TensorRT bundle path, output-parity status, and
`reference_over_candidate_p50`. Model loading, conversion, compilation, and
warmup are excluded from timed samples on both sides.

Performance preserves the historical stability rule: ten samples per side,
first-half/last-half median drift at most 5%, and at least eight samples within
5% of the median. An unstable pair is repeated once in fresh processes using
the prepared bundle. Persistent instability is `measurement_inconclusive` and
produces no speed ratio. Both attempts remain in the report. GPT-2 requires
matching GPU identity and actual compiled graphs; compilation during measured
iterations is an execution error.

A device run may set an optional Performance target without changing family
configuration:

```yaml
performance:
  minimum_speedup: 0.95  # HF p50 / TensorRT p50
  blocking: true
```

The target belongs in `runs/<name>.yaml`. Its evaluation is separate from the
family's observation-only verdict; a blocking target failure makes the overall
run fail. Without a target, valid timing comparisons remain observations.

## Commands

Show the selected plan without loading a model:

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

Run every discovered Performance case with the corresponding high-level
configuration:

```bash
trtmc-qualify run \
  --run-config apps/benchmark/qualification/runs/all-performance.yaml \
  --families-root families \
  --output artifacts/qualification/performance
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

Plans and cases use generated run-local IDs, not content-derived identities.
Resume verifies the recorded repository revision and working-tree changes,
configuration/source file state, actual Python package versions, and prepared
file state. It does not hash source, dependencies, weights, or bundles.
Changed prepared files require a new run instead of silently reusing completed
results; changes to the original external dataset do not replace the saved samples.
Preparation receipts and executor logs are retained
under `preparations/` and `environments/` alongside case evidence in `items/`.

Earlier draft runs without input-file state records and saved Accuracy samples
must be prepared again; no legacy receipt conversion is supported.

Regenerate the machine-readable and HTML reports without executing cases:

```bash
trtmc-qualify report artifacts/qualification/accuracy
```

`report.json` is authoritative. `report.html` renders the same data. Blocking
cases contribute pass/fail/error status. Observation-only cases are reported
separately and can never claim a pass or fail verdict. If no model has opted
into the selected assessment, the valid empty plan is reported as `empty` and
does not create a coverage failure.

The report exposes completed, comparable, errored, inconclusive, and failed
counts separately. Detailed rows retain both backends, compilation/stability
evidence, optional run targets, and links to interpreter and preparation receipts.
