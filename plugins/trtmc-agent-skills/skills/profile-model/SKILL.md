---
name: profile-model
description: >-
  Measure one TensorRT-Model-Connect model through its public Task API or run a
  checked-in performance-matrix entry with comparable evidence.
---

# Profile a Model

Choose the evidence level before running:

- For one model or testcase, use `trtmc-bench`.
- For a checked-in release comparison, use `tools/perf_matrix.py`.

The public benchmark boundary is `public_task_call_wall`. It excludes bundle
building, process startup, warmup, report generation, and bundle loading.
Current public tooling does not provide layer attribution; do not infer a
kernel or layer bottleneck from Task API wall time alone.

## Before timing

Run the owning family correctness testcase on the exact bundle. Record the
repository and checkpoint revisions, GPU and driver, runtime root, family DSO,
operation, input, seed, warmups, iterations, and synchronization boundary.

For a focused run:

```bash
trtmc-bench run --model <model> --case <case> \
  --bundle <bundle> --no-build \
  --warmup 3 --iterations 10 --runtime-root <runtime-root> \
  --output <result-dir>
```

Use the same bundle kind, hardware, input, seed, warmups, iterations, and
runtime policy for before/after comparisons. If any differ, report the
confounder instead of one causal speedup percentage.

For release evidence, first check and then run the exact entry:

```bash
python3 tools/perf_matrix.py check <suite> \
  --environment <environment> --entry <entry>
python3 tools/perf_matrix.py run <suite> \
  --environment <environment> --entry <entry>
```

Lead the report with correctness and evidence level. Include exact commands,
p50 and other suite-owned statistics, measurement scope, limitations, and any
target not run.
