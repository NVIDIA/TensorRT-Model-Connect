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
python tools/trtmc_validate.py internvl3-2b vlm_mmmu_pro_vision_mcq
```

Run every single-device model whose catalog status is `ready`:

```bash
python tools/trtmc_validate.py --all
```

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
`tools/trtmc_reference.py`, outside the validation engine process. Its result is keyed by
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
Comparison runs through `tools/trtmc_compare.py`; validation-engine commands
are not part of the result or its reproduction contract.
Every model/workload binding must resolve to an independent reference
runner selected by the prepared dataset kind. A catalog-wide test rejects new
bindings that have no independent native reference runner.

Tasks whose outputs require a model-specific comparator use
`model_plugin_json`. The workload and prepared dataset remain task-owned; a
row selects the matching model manifest testcase only at execution time. The
reference, TRTMC runner, and comparator are invoked directly without calling
the E2E orchestrator. Array-valued outputs are persisted as artifacts so a
cached reference can be compared in later runs.

Prepare the fixed task datasets from public benchmark sources already staged
on the validation machine:

```bash
python tools/prepare_model_plugin_validation_datasets.py \
  --output-root /mnt/data \
  --flores-source /mnt/data/FLORES200_en_fr/flores200_en_fr_task_eval.json \
  --full-duplex-source /mnt/data/FullDuplexBench-v1.0-public \
  --mmlu-source /mnt/data/MMLU_Pro/mmlu_pro_dataset.json \
  --mmmu-source /mnt/data/MMMU_Pro_vision/mmmu_pro_vision_dataset.json \
  --seedtts-source /mnt/data/SeedTTS_en_meta/seedtts_en_meta.json
```

This writes the `mmmu-pro-vision`, `mmmu-pro-vision-square-448`,
`mmlu-generation-modes`, `flores200-en-fr`, `full-duplex-bench`, and
`seedtts-en-omni-audio` directories directly under `/mnt/data`. It also writes
`/mnt/data/trtmc_model_plugin_validation_manifest.json`, which lists only
those six managed directories and records each file's byte size and SHA256.
There is intentionally no aggregate `TRTMCValidation` directory. Wan workloads
read the existing root-level public `VBench` asset directly. Dev/QA machines
and NAS mirrors should copy the six directories without changing their
relative layouts and verify the manifest after transfer.

LocateAnything grounding accuracy uses the public `lscpku/RefCOCO_rec`
dataset pinned at revision
`566810e1ad62821ed3c6ab569ea33d80f5bdb874`. Stage that exact Hugging Face
snapshot, then convert its `testA` split:

```bash
python tools/prepare_refcoco_validation_dataset.py \
  --source-root /mnt/data/RefCOCO_rec/raw/lscpku/RefCOCO_rec \
  --output-dir /mnt/data/RefCOCO_rec/unified \
  --split testA
PYTHONPATH=python:. python tools/validation/engine.py eval \
  --suite refcoco_grounding \
  --model locateanything-3b \
  --dataset /mnt/data/RefCOCO_rec/unified/dataset.json \
  --limit 20
```

The source dataset card does not declare a license. The converter records this
as `source_license_status: not-declared-by-source-card`; operators must verify
their right to use the staged images and annotations. The generated manifest
records the source repository, immutable revision, normalized box format, and
official LocateAnything single-instance prompt.

Every agreement or disagreement therefore means that both backends consumed
the aligned prepared inputs and produced outputs that were evaluated by the
declared comparator. A model without that complete contract stays in the
catalog with `not_compared_reason`. `--all` records it as a white **Not
compared** row without launching E2E, creating a reference environment, or
building an engine. Such rows make the aggregate report status `incomplete`;
they are not agreements, disagreements, execution errors, or attempted model
failures.

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
command re-enters validation or comparison orchestration. The
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

Native Transformers text, embedding, VLM, speech, and model-owned plugin
references use the model manifest's FP16, BF16, or FP32 base precision. An explicit
`--hf-dtype` must match an unquantized TRTMC model's base precision; validation
rejects a conflicting override before inference.

Quantized candidates must declare their unquantized reference precision in the
model testcase's validation configuration:

The persisted manifest field remains named `task_eval` for compatibility with
existing reference-cache keys and result artifacts. It is metadata only; no
validation command imports or executes a task-eval module.

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

## Add or extend a model

1. Reuse or add a dataset workload in
   `tests/validation/workloads.yaml`.
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
