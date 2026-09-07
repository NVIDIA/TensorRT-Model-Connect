---
title: Validate & Benchmark
description: Match the validation method to the claim and retain a reproducible evidence receipt.
---

Use the smallest test that proves the claim:

| Claim | Minimum meaningful evidence |
| --- | --- |
| Resolver or bundle behavior | Focused core contract test |
| Exact checkpoint builds | Successful family build with recorded model revision and options |
| Task correctness | Family-owned input, reference, comparator, and thresholds |
| Target compatibility | Full build, inspect, load, and execution on that target |
| Performance | Fixed workload, warmup, repeated measurement, timing boundary, and quality gate |

Start with ownership and metadata validation:

```bash
PYTHONPATH=core/builder:apps/benchmark:. python3 -m tools.model_ci validate
```

Then run focused tests from the owning family. A normal model contribution
should not need to edit another family or shared test policy.

For a public Task API benchmark:

```bash
trtmc-bench list models
trtmc-bench run \
  --model distilgpt2 \
  --runtime-root /opt/trtmc/lib \
  --output results/distilgpt2
```

The benchmark is an application above ModelConnect. It calls the same public
build and abstract Task APIs as another consumer; core and families do not
import benchmark code.

Record model and revision, build options, bundle path, family/task/backend,
hardware and software cohort, complete command, input, warmup, measured
iterations, timing boundary, result, and task-quality result. Keep correctness,
compatibility, and performance as separate claims.

Use [Testing Reference](../reference/testing.md) and
[Benchmarking Reference](../reference/benchmarking.md) for detailed suites.
