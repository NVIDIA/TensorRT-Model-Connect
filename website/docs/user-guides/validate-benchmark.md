---
title: Validate & Benchmark
description: Match the validation method to the claim and retain a reproducible evidence receipt.
---

Use the smallest test that can actually prove the claim:

| Claim | Minimum meaningful evidence |
| --- | --- |
| Parser/config behavior | Focused unit or contract test |
| Exact checkpoint builds | Successful build with exact model revision/config |
| Task correctness | Model-owned E2E input, oracle, comparator, and thresholds |
| Target compatibility | Full build/load/run on that target and software cohort |
| Performance | Fixed workload, warmup, repeated measurement, timing boundary, and quality gate |

Start repository validation with ownership checks:

```bash
python3 tools/model_ci.py validate
```

Then run the exact E2E manifest or focused test named by the model change. A
skipped GPU preflight is not a pass, and a documentation build is not model
parity evidence.

For performance, record model/revision, bundle checksum/config, execution
path, hardware/software cohort, complete command, input, warmup, measured
iterations, metric boundary, result, and task-quality result.

Use [Testing Reference](../reference/testing.md) and
[Benchmarking Reference](../reference/benchmarking.md) for command details.
The [Validation and Benchmarking Tutorial](../tutorials/advanced/validation-and-benchmarking.md)
is the course-style lab.

{/* Collaborative review anchor: batch 2. */}
