---
name: profile-model
description: >-
  Measure one TensorRT-Model-Connect model through its public Task API or run a
  checked-in performance-matrix entry with comparable evidence.
---

# Profile a Model

Choose the evidence level before running:

- For one model or testcase, use `trtmc-bench`.
- For a checked-in release comparison, use `python3 -m qualification_tests.benchmark_qualification.performance`.

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
python3 -m qualification_tests.benchmark_qualification.performance check <suite> \
  --environment <environment> --entry <entry>
python3 -m qualification_tests.benchmark_qualification.performance run <suite> \
  --environment <environment> --entry <entry>
```

Lead the report with correctness and evidence level. Include exact commands,
p50 and other suite-owned statistics, measurement scope, limitations, and any
target not run.


## HTTP text load evidence

For text serving measurements, reuse the persistent `trtmc-server` through
`python3 -m trtmc_aiperf_qual text-profile --environment <environment> --config
<text-config> --out <result-dir>` with `PYTHONPATH=apps/aiperf_qual`. Consult
`website/docs/user-guides/profile-text-with-aiperf.md` and the checked-in
`apps/aiperf_qual/config/text/` examples. Keep the owning family's exact-bundle
correctness checks before timing; HTTP success does not replace them.

Require incremental capability before interpreting TTFT or inter-token
latency. Nonstreaming native records use `public_task_call_wall`; streaming
records include relay/backpressure and have a different scope. Preserve warmup
and failures when joining AIPerf exports to server records with `X-Request-ID`.
Native prompt counts are unavailable; client tokenizer counts are estimates.
AIPerf request latency ends at the last content response. Inspect the client
request lifecycle and server handler timing as well: terminal delivery or
cleanup can delay the next request without appearing in content latency.
The optional sequential reference comparison requires matching payloads,
outputs, token counts, precision and Task boundaries. It does not change the
qualification criteria or establish speedup from HTTP latency alone.
