---
title: Profiling
description: Move from reproducible Task-level timing to CUDA and kernel evidence without changing the workload.
---

Profile only after a repeatable benchmark identifies a real bottleneck. Keep
the exact model, bundle, request, runtime root, warmup, and device unchanged
between the benchmark and profiler run.

## Start with the public Task boundary

```bash
trtmc-bench run \
  --model distilgpt2 \
  --runtime-root /opt/trtmc/lib \
  --warmup 3 \
  --iterations 10 \
  --output results/distilgpt2
```

The benchmark reports `public_task_call_wall` latency and operation-specific
metrics. Optional GPU telemetry is sampled outside timed calls and is useful
for broad utilization, memory, or power context, not kernel attribution.

## Separate the phases

Measure these boundaries independently:

1. bundle preparation and TensorRT engine build;
2. process startup, DSO loading, and engine deserialization;
3. warmup;
4. repeated public Task calls; and
5. task-specific stages such as prefill/decode or denoising steps.

Do not compare a cold end-to-end command with a warm Task-only result.

## Collect a CUDA timeline

Use Nsight Systems when you need CPU/CUDA overlap, synchronization, memory
copies, or kernel sequencing. Profile the same CLI request used for the
reproduction:

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --output=profiles/gpt2 \
  trtmc run gpt2.bundle \
    --runtime-root /opt/trtmc/lib \
    --prompt "The capital of France is" \
    --max-new-tokens 20
```

Use Nsight Compute only after the timeline identifies a specific kernel. Scope
collection narrowly; full metric sets can perturb the workload substantially.

## Family ownership

Model-specific scheduling, cache behavior, preprocessing, postprocessing, and
custom kernels belong to `families/<family>/`. Shared core or backend code is a
candidate only when the profile demonstrates that the cost occurs inside that
shared contract for more than one owner.

For multi-device cases, retain rank/device mapping and inspect each rank rather
than collapsing traces into one unlabeled timeline.

## Reporting checklist

Record the exact Git revision, bundle header, family manifest/testcase,
hardware/software cohort, command, warmup, profiler version and options,
observed bottleneck, and any profiler overhead. A trace explains one measured
run; it does not by itself prove a general optimization or parity claim.
