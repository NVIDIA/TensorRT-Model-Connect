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
exist, then prints the environment it used. At completion it prints the exact
HF and TRTMC reproduction commands, the per-model `comparison.json`, and the
aggregate `report.html`.

## Add or extend a model

1. Reuse or add a dataset workload in
   `tests/task_eval/validation_suites.yaml`.
2. Add that workload under the model in `model_workloads.yaml`.
3. Select one workload as the model default.

Use `e2e` only when the model cannot use a dataset-backed workload yet. A model
may list multiple workloads; callers select one by passing it after the model
name.
