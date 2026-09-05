---
name: optimize-model-precision
description: >-
  Evaluate supported precision, quantization, or selected FP32-layer choices
  for one TensorRT-Model-Connect family with matched correctness and timing.
---

# Optimize Model Precision

Name the objective first: bundle size, device memory, setup time, Task API
latency, throughput, or another measured family outcome. The best candidate is
the lowest-cost option that still passes the existing family contract.

## Establish a baseline

Use one family manifest and testcase. Record the repository and checkpoint
revisions, hardware, build request, seed, input, bundle path, correctness
result, warmups, iterations, and measured boundary.

Build the current declared configuration and run its E2E before exploring
alternatives. The public build interface exposes direct values for
`--precision`, `--quantization`, and repeated `--fp32-layer` selections.
Attempt only values the owning family implements.

```bash
python -m tensorrt_model_connect build <model-id-or-path> \
  --revision <revision> --precision <fp32-or-fp16-or-bf16> \
  --output <candidate.bundle>
trtmc inspect <candidate.bundle>
```

Change one effective value per attempt. If a candidate fails, localize the
first mismatch in the owner family before changing another value. Do not add
cross-family precision policy or weaken a threshold, dataset, seed, or oracle.

## Qualify a candidate

Run the exact family E2E testcase with the candidate configuration. Only after
it passes, compare Task API performance with matched inputs and protocol:

```bash
trtmc-bench run --model <model> --case <case> \
  --bundle <candidate.bundle> --no-build \
  --warmup 3 --iterations 10 --runtime-root <runtime-root> \
  --output <result-dir>
```

Use `tools/perf_matrix.py` when the claim targets a checked-in release entry.
Report every attempt in a compact table, including effective request values,
correctness, measured objective, confounders, and unrun targets. It is valid
for the original configuration to remain the best qualified result.
