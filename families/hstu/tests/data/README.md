<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# HSTU parity fixtures

| File | What it is |
| --- | --- |
| `../fixtures/request.json` | one ranking request at the measured geometry: 6 contextual ids, 4096 (video_id, action_weights) history events, 100 candidate video_ids; `families.hstu.tests.golden.make_record`, seed 1 |
| `../fixtures/request-small.json` | the same at the small geometry (16 history events, 8 candidates) |
| `golden.json` | the padded PyTorch reference's FP32 logits [100 candidates, 8 tasks] for `request.json` on the synthetic checkpoint `hstu-kuairand-1k-synthetic` (seed 0), plus the reference's own BF16 spread |
| `golden-small.json` | the same for `request-small.json` on `hstu-kuairand-1k-synthetic-small` (seed 0) |

The ids are uniform draws inside each table, not KuaiRand-1K interactions, and
the weights are seeded random: the fixtures test conversion and runtime
parity, never ranking quality.

Regenerate a golden (CPU, needs torch):

```bash
PYTHONPATH=core/builder:. python -m families.hstu.weights <checkpoint>
PYTHONPATH=core/builder:. python -m families.hstu.tests.golden \
  <checkpoint> families/hstu/tests/fixtures/request.json families/hstu/tests/data/golden.json
```

## How the tolerances were declared

Both were declared on 2026-11-06, before any TensorRT output was compared,
from the reference's own spread on that day's checkpoint. The generator has
since drawn every value from PCG64's raw stream (bytes independent of the
NumPy version); the numbers below are the current checkpoint's, and the
tolerances did not move.

**`hstu-kuairand-1k-rank` (BF16):** max |Δ| ≤ 2.5e-3 and task cosine ≥
0.9995. The reference's own BF16 run lands max |Δ| 6.78e-4 and min task
cosine 0.9999857 from its FP32 golden (golden max |logit| 0.160), so max |Δ|
has 3.7x that spread and the cosine budget (5e-4) 35x the spread's (1.43e-5).
TensorRT rounds at other op boundaries than the reference (GEMM outputs
before the FP32 epilogue, QKᵀ before SiLU). At 4096 history events this case
cannot see a structural defect: candidates seeing each other moves the
logits 2.1e-4, history attending to later history 2.1e-4, scaling_seqlen =
tokens 1.1e-4, history not interleaved 9.3e-5, an action before its item
4.4e-5, all under the BF16 spread.

**`hstu-kuairand-1k-small-rank` (FP32, TF32 off):** max |Δ| ≤ 1e-4 and task
cosine ≥ 0.99999, so only accumulation order separates TensorRT from the
reference. Each structural defect moves these logits past the tolerance
(`../test_reference.py`, which asserts more than 2x for each): an action
before its item 3.5x, history attending to later history 8.1x, history not
interleaved 9.1x, no position embedding 19.5x, scaling_seqlen = tokens 25.1x,
candidates seeing each other 43.7x, attention dropped 181x, u/v split swapped
196x (golden max |logit| 0.116).
