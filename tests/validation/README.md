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

## Add or extend a model

1. Reuse or add a dataset workload in
   `tests/task_eval/validation_suites.yaml`.
2. Add that workload under the model in `model_workloads.yaml`.
3. Select one workload as the model default.

Use `e2e` only when the model cannot use a dataset-backed workload yet. A model
may list multiple workloads; callers select one by passing it after the model
name.
