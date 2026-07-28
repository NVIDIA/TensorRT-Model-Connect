<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# TRTMC reference validation

`trtmc-validate` is an internal Dev/QA workflow for checking that a TRTMC model
still agrees with its original reference implementation.

Run a model's default workload:

```bash
python tools/trtmc_validate.py gpt2-125m
```

Run a different workload declared for that model:

```bash
python tools/trtmc_validate.py internvl3-8b vlm_mmmu_pro_vision_mcq
```

Run every validation-eligible ready single-device model:

```bash
python tools/trtmc_validate.py --all
```

Eligibility excludes manifests that require multiple devices, are marked
`skip`, or use `ci_tier: l0_only`; readiness alone does not select a model.

The all-model command supervises one isolated worker process per model. By
default it records a failed worker and continues with the remaining models.
Stop after the first failed model when that is preferable:

```bash
python tools/trtmc_validate.py --all --on-model-failure stop
```

Both policies return a nonzero exit status when any attempted model fails.
Process isolation also covers failures that happen before backend execution,
such as reference-environment setup or uncaught model-specific Python errors.

The same CLI is the CI case entry point. Generate a machine-readable matrix,
then run one exact model/workload binding in each CI node:

```bash
python tools/trtmc_validate.py --all --dry-run
python tools/trtmc_validate.py gpt2-125m mmlu_continuation_parity \
  --output validation-artifacts
```

For configured consistency workloads, the case result is always written to
`<output>/<model>/<workload>/comparison.json`; `report.json` and `report.html`
are written at the output root. Exit status `0` means reference consistency
passed, `1` means the case ran but validation failed, and `2` means CLI or
setup validation failed before the case could run. Requesting a model that is
explicitly marked not compared also writes
`<output>/<model>/not-compared/comparison.json` and returns `2`.

Dataset-backed workloads use the task-specific sample limits declared in
`model_workloads.yaml`. Fast encoder and classification workloads use larger
slices, while generation-heavy image, video, and audio workloads use smaller
slices. The selected limit is printed before execution and shown in the
`Samples` column of `report.html`.

Override the configured limit for one run, or request the complete dataset
explicitly:

```bash
python tools/trtmc_validate.py gpt2-125m --limit 100
python tools/trtmc_validate.py gpt2-125m --limit 0
```

The command creates a reference environment only when one does not already
exist, then prints the environment it used. Reference inference runs through
`tools/trtmc_reference.py`, outside the task-eval CLI. Its result is keyed by
the input slice and inference settings and reused from the shared reference
cache when the key already exists. TRTMC variants may declare the same
`reference_cache_identity` in `model_workloads.yaml` only when they use the
same reference model, prepared inputs, and inference contract. The explicit
identity lets those variants share one cached reference result without
weakening cache isolation for other models.

TRTMC bundles live in one shared validation engine directory. A required
rebuild removes the existing bundle and writes the replacement at the same
path; a failed replacement removes any partial bundle. Per-run result
directories therefore do not retain another copy of the bundle.

At completion the command prints the exact reference and TRTMC reproduction
commands, the per-model `comparison.json`, and the aggregate `report.html`.
Comparison runs through `tools/trtmc_compare.py`; task-eval
commands are not part of the validation result or its reproduction contract.
Every model/workload binding must resolve to an independent reference
runner selected by the prepared dataset kind. A catalog-wide test rejects new
bindings that would fall back to `task_eval.py` or E2E execution.

Every agreement or disagreement therefore means that both backends consumed
the aligned prepared inputs and produced outputs that were evaluated by the
declared, threshold-gated comparator. A task-eval suite whose comparison is
explicitly diagnostic-only may still be run directly for investigation, but
it is not a runnable reference-consistency binding. A model without the
complete threshold-gated contract stays in the catalog with
`not_compared_reason`. `--all` records it as a white **Not compared** row
without launching E2E, creating a reference environment, or building an
engine. Such rows make the aggregate report status `incomplete`; they are not
agreements, disagreements, execution errors, or attempted model failures.

`--all --dry-run` keeps these models visible with `workload: null`,
`status: not_compared`, and the reason. CI matrix generation can select only
entries that contain a workload.

The HTML artifact is named **TRTMC Reference Consistency Report** because it
covers task accuracy as well as token, embedding, and numerical agreement.
For large datasets it shows one dataset-run command and at most three
representative commands per backend. The first disagreement is preferred when
one exists.

When per-sample differences exist, the model row also shows up to 20 affected
samples. Each sample contains the exact input, both raw prediction records,
the comparison evidence, and native single-sample commands when the backends
provide them. The reference command invokes a standalone upstream-framework
entrypoint and the TRTMC command invokes the model executable directly; neither
command re-enters validation, comparison, or task-eval orchestration. The
complete set is written to `disagreements.jsonl`, while `comparison.json` and
`report.json` retain only bounded metadata.

Model-owned reference and TRTMC plugins record the subprocess command they
actually executed. These per-sample command logs stay in the model work
directory; the HTML includes only commands for disagreement samples. This
keeps a 1,000- or 10,000-sample report compact without reconstructing a command
after the failure.

For failed image, video, or audio samples, the report copies only the relevant
input/output media into that sample's `repro` directory. Image and video-frame
previews, playable video files up to the artifact size limit, and WAV/audio
controls are rendered next to the two result records. Passing samples do not
duplicate media.

The report keeps three statuses separate:

- `execution`: whether the programs completed or errored;
- `comparison`: whether TRTMC agrees or disagrees with the reference;
- `validation`: the final pass, fail, or skipped result.

An unimplemented consistency contract uses `execution: not_run`,
`comparison: not_run`, and `validation: not_compared`.

The HTML report renders each status as an independent colored signal and shows
the primary agreement metric next to it.

## Precision contract

Native Transformers text, embedding, VLM, and speech references use the
model manifest's FP16, BF16, or FP32 base precision. An explicit
`--hf-dtype` must match an unquantized TRTMC model's base precision; validation
rejects a conflicting override before inference.

Quantized candidates must declare their unquantized reference precision in the
model testcase's validation configuration:

```json
"precision": "bf16",
"quantization": {"format": "fp8"},
"task_eval": {
  "reference_precision": "bf16"
}
```

This means TRTMC FP8 with a BF16 base is compared with an unquantized HF BF16
reference. It is a quantization-quality comparison, not an assertion that HF
executed FP8 kernels. The same contract applies to FP4, NVFP4, MXFP, or future
quantization formats when those candidates are added. A quantized manifest
without `task_eval.reference_precision` fails before reference inference.

The resolved TRTMC base precision, quantization format, reference precision,
and comparison kind are stored in `comparison.json` and shown in the HTML
report. Reference cache keys include the effective reference dtype, so only
variants with the same reference computation can reuse an entry.

Report publication through `write_report` uses an exclusive advisory lock on
the output-root directory. The lock covers input reads, staging, commit or
rollback, recovery cleanup, and descriptor closure. The CLI completes its case
and run producers before it calls `write_report`; cooperating report
regenerators for the same output root must use that same path. A separate
process must not concurrently write case, run, or report artifacts in that
output root. Hidden `.report-stage-*` paths and transaction `.next`,
`.previous`, `.rollback`, and `.cleanup` names are private recovery state;
external writers must not create or mutate them. Writes that bypass the
advisory lock are outside the supported concurrency contract. Identity and
content checks reject many detected races, but they do not make unsupported
concurrent mutation safe. Publication requires Linux
`renameat2(RENAME_NOREPLACE)` and directory `flock` support.

## Add or extend a model

1. Reuse or add a dataset workload in
   `tests/task_eval/validation_suites.yaml`.
2. Add that workload under the model in `model_workloads.yaml`.
3. Add a workload sample limit if the workload is new.
4. Select one workload as the model default.

A model may list multiple workloads; callers select one by passing it after the
model name. If the aligned reference/TRTMC comparison is not implemented yet,
declare only:

```yaml
model-name:
  not_compared_reason: Aligned reference workload and output comparator are not implemented.
```

Do not use `e2e` as a validation workload.
