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

Dataset-backed workloads use the task-specific sample limits declared in
`model_workloads.yaml`. Fast encoder and classification workloads use larger
slices, while generation-heavy image, video, and audio workloads use smaller
slices. The selected limit is printed before execution and recorded next to the
actual prepared-input count in `comparison.json` and `report.html`.

Override the configured limit for one run, or request the complete dataset
explicitly:

```bash
python tools/trtmc_validate.py gpt2-125m --limit 100
python tools/trtmc_validate.py gpt2-125m --limit 0
```

The command creates a reference environment only when one does not already
exist, then prints the environment it used. Reference inference runs through
`tools/trtmc_reference.py`, outside the task-eval CLI. Its result is keyed by
the prepared inputs and inference settings and reused from the shared reference
cache when the key already exists.

TRTMC bundles live in one shared validation engine directory. A required
rebuild removes the existing bundle and writes the replacement at the same
path; a failed replacement removes any partial bundle. Per-run result
directories therefore do not retain another copy of the bundle.

At completion the command prints the exact reference and TRTMC reproduction
commands, the per-model `comparison.json`, and the aggregate `report.html`.
Dataset-backed comparison runs through `tools/trtmc_compare.py`; task-eval
commands are not part of the validation result or its reproduction contract.
Every non-`e2e` model/workload binding must resolve to an independent reference
runner selected by the prepared dataset kind. A catalog-wide test rejects new
bindings that would fall back to `task_eval.py`. Models whose validation
contract is still only `e2e` continue to use the E2E path and are not presented
as migrated task-eval workloads.

The HTML artifact is named **TRTMC Reference Consistency Report** because it
covers task accuracy as well as token, embedding, and numerical agreement.
For large datasets it shows one dataset-run command and at most three
representative commands per backend. The first disagreement is preferred when
one exists.

When per-sample differences exist, the model row also shows up to 20 affected
samples. Each sample contains the prepared input, both raw prediction records,
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

The HTML report renders each status as an independent colored signal and shows
the primary agreement metric next to it.

## Add or extend a model

1. Reuse or add a dataset workload in
   `tests/task_eval/validation_suites.yaml`.
2. Add that workload under the model in `model_workloads.yaml`.
3. Add a workload sample limit if the workload is new.
4. Select one workload as the model default.

Use `e2e` only when the model cannot use a dataset-backed workload yet. A model
may list multiple workloads; callers select one by passing it after the model
name.
