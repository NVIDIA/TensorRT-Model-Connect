---
name: optimize-model-precision
description: >-
  Use when evaluating FP16, BF16, or supported quantization formats for a
  TensorRT-Model-Connect model. Establishes a model-owned correctness baseline,
  changes one effective build option at a time, detects ineffective precision,
  and retains comparable parity, memory, bundle, and performance evidence.
---

# Optimize Model Precision

## Objective

Find the lowest-cost configuration that satisfies the model's existing
correctness contract and improves a named resource or performance metric.
“Best” must name the objective: bundle size, device memory, setup time, prefill,
decode, throughput, or another model-owned metric.

Do not weaken an oracle, threshold, sample set, or pass criterion to make a
configuration qualify. If a test appears wrong, stop and escalate it to a
maintainer.

## Establish The Owned Baseline

Resolve the model through all relevant descriptors:

- Python family `MODEL.toml` and plugin;
- C++ model `MODEL.toml` and runtime strategy;
- E2E family `MODEL.toml`, manifest, testcases, thresholds, and
  `perf_validation.json` when present;
- model-first binding in `tests/validation/model_workloads.yaml`;
- optimized implementation/profile/qualification descriptors when selected.

List and dry-run the reference-consistency workload:

```bash
PYTHONPATH=python:. python3 tools/trtmc_validate.py --list
PYTHONPATH=python:. python3 tools/trtmc_validate.py \
  <model> <workload> \
  --dry-run \
  --output <baseline-plan-dir>
```

Build and validate the existing configuration before optimizing it. Record:
repository SHA, exact model revision, target/hardware, runtime path, effective
build options, bundle hash, workload and sample limit, seed/sampling, artifact
paths, correctness metrics, and the performance protocol.

## Build Matrix

Try only formats supported by the current CLI and owning family:

```bash
./build/trtmc build <model> -o <bundle>.bundle \
  --precision fp16 \
  --max-cache-length <N>

./build/trtmc build <model> -o <bundle>.bundle \
  --precision fp16 \
  --quantize <supported-format> \
  --quant-calibration-samples <N> \
  --max-cache-length <N>
```

The current quantization core resolves a `QuantPlan`; family hooks own
calibration data, adapters, exclusion patterns, and FP8 scales. Use the current
CLI help and `website/docs/features/quantization.md` for supported options.
Use `--quant-scales` for a reviewed generic scale artifact and the dedicated
FP8 scale flags only for their documented compatibility path. Do not bypass
the plan with ad hoc family Q/DQ code.

Precision and quantization are effective public build options. Changing one can
switch between native and optimized runtime implementations. Inspect every
bundle:

```bash
./build/trtmc inspect <bundle>.bundle
sha256sum <bundle>.bundle
```

Compare two configurations only when their implementation path, model revision,
inputs, and measurement protocol match. Otherwise report the runtime-path
change as a confounder. Treat `--rtx` as another backend variable and do not
combine a TensorRT-to-TensorRT-RTX switch with a precision conclusion.

## One Variable Per Attempt

Recommended order:

1. reproduce the current declared baseline;
2. try FP16 or the family's declared lower-precision default;
3. try BF16 when supported and motivated;
4. try one family-supported quantization format at a time;
5. vary calibration or exclusion policy only after isolating the failing
   boundary.

Use `$fp16-trt-network` when base precision is not threaded correctly. Use
`$debug-trt-mismatch` when an attempt executes but fails comparison.

## Detect Ineffective Precision

CLI acceptance and bundle size alone are insufficient. Compare:

- inspected runtime path and precision metadata;
- bundle sections and weight dtypes;
- device-memory measurements under the same workload;
- graph/debug evidence for work and state tensor dtypes;
- bundle size as a supporting signal.

A similar FP32/FP16 size suggests investigation, not a fixed 10-percent failure
rule. A smaller bundle does not prove that runtime tensors use the requested
dtype.

## Correctness Gate

Run the model-first binding with the candidate bundle:

```bash
PYTHONPATH=python:. python3 tools/trtmc_validate.py \
  <model> <workload> \
  --bundle <candidate.bundle> \
  --output <candidate-artifacts>
```

Also run the owning E2E case when it carries additional model/runtime
contracts. Keep execution, reference, comparison, and final validation status
separate. A passing smoke test is not parity; a dry run is not execution.

Use the existing manifest/testcase/threshold sidecars. Create a persistent new
variant only when the configuration is intended to become a supported profile,
and register it in the owning family `MODEL.toml`.

## Performance Gate

For a quick diagnosis, use `$profile-model`. For release or qualification
evidence, use the model-owned `perf_validation.json` and
`benchmarks/performance/release.yaml` through `tools/perf_matrix.py`.

If bundle preparation occurred outside the matrix campaign, retain its
`test_task` receipt and attach it with `tools/perf_matrix.py report
<run-directory> --preparation-receipt <receipt>`. The receipt revision and
bundle paths must match the run. Internal CI performance reports and raw
artifacts remain private; do not publish them through Source Actions or Pages.

Warmups, timed iterations, inputs, synchronization, target, power state, and
runtime path must match the baseline. Never treat missing/zero provider phase
timings as latency measurements.

## Attempt Record

Persist one entry after every attempt:

```json
{
  "repository_sha": "<sha>",
  "model_revision": "<revision>",
  "target": "<hardware>",
  "objective": "<metric>",
  "attempts": [
    {
      "precision": "fp16",
      "quantize": null,
      "effective_options": {},
      "runtime_path": "native-or-optimized-id",
      "bundle_sha256": "<sha256>",
      "correctness": {"status": "pass", "artifact": "<comparison.json>"},
      "performance": {"status": "pass", "artifact": "<result.json>"},
      "bundle_bytes": 0,
      "device_memory_bytes": 0,
      "code_changes": [],
      "notes": ""
    }
  ],
  "best_qualified_attempt": 0
}
```

An attempt is qualified only when the required correctness and objective
evidence both pass. Plain FP32 can remain the best result when no lower
precision configuration qualifies; report that honestly instead of redefining
completion.

## Final Report

Provide the full matrix, exact commands, artifact paths and hashes, objective
comparison, selected configuration, failures and first divergent boundaries,
code changes, and unrun target-hardware or broad-regression checks.

<!-- Collaborative review anchor: batch 2. -->
