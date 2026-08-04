---
name: profile-model
description: >-
  Use when diagnosing one model's runtime cost or producing comparable
  TensorRT-Model-Connect performance evidence. Routes quick investigation to
  the unified profiler and release or qualification claims to the checked-in
  performance matrix and model-owned performance contract.
---

# Profile Model

## Decide The Evidence Level

Choose one path before running:

| Question | Entry point | Claim boundary |
|---|---|---|
| Where is one model spending time? | `tools/trtmc_profile.py` | diagnostic |
| Does a code change improve one owned workload? | profiler plus matching model testcase | local comparison |
| Is a model release-ready against its reference? | `tools/perf_matrix.py` | release matrix |
| Does an optimized implementation qualify? | model-owned qualification producer | exact profile/target |

Profiler output is not automatically release or qualification evidence.

## Preconditions And Provenance

Use the supported GPU/TensorRT environment and record:

```bash
git rev-parse HEAD
nvidia-smi --query-gpu=name,uuid,driver_version,pstate,power.draw \
  --format=csv,noheader
python3 -c "import tensorrt as trt; print(trt.__version__)"
test -x ./build/trtmc
test -x ./build/trtmc-bench
```

Record the exact model revision, bundle SHA-256, native or optimized runtime
path, effective config, target, warmups, timed iterations, inputs, token/sample
counts, timing boundary, synchronization policy, and reference environment.
Without those, label results exploratory.

If a team container is required:

```bash
./scripts/bootstrap_workspace.sh --id <team-id> \
  --branch "$(git branch --show-current)" --detach
```

## Correctness Before Timing

Select the owning model-first workload or E2E testcase and prove it passes
before making performance claims:

```bash
PYTHONPATH=python:. python3 tools/trtmc_validate.py \
  <model> <workload> \
  --bundle <bundle.bundle> \
  --output <validation-artifacts>
```

Do not time a candidate with a failed, skipped, or unrun comparison. Preserve
the same model revision and workload when moving to profiling.

## Quick Single-Model Diagnosis

```bash
PYTHONPATH=python:. python3 tools/trtmc_profile.py \
  --model <model> \
  --bundle <bundle.bundle> \
  --prompt "<owned-testcase-prompt>" \
  --max-new-tokens <N> \
  --warmup 3 \
  --iterations 10 \
  --dtype float16 \
  --trtmc-binary ./build/trtmc \
  --hf-python <python> \
  --json \
  --output-dir <profile-dir>
```

Useful options:

- `--no-layer-profile` for a faster E2E-only run;
- `--cpu-profile` for host-phase attribution;
- `--nsight` for a kernel trace with binary and bundle;
- `--no-compile` when torch.compile is outside the comparison;
- `--compile-mode` only when the baseline contract names that mode.

Use `--trust-remote-code` only for a reviewed model that requires it.

The profiler is decoder-oriented. For audio, diffusion, vision, and other
multi-stage models, prefer their owned testcase/performance adapter rather than
forcing a text prompt through this tool.

## Runtime Configuration

GPU greedy selection and CUDA Graph control are runtime-config fields, not the
removed process knobs. Pass them through a supported config layer, for example:

```text
runtime.prefer_gpu_greedy=true
runtime.disable_cuda_graph=false
```

Do not use `TRTMC_GPU_ARGMAX` or `TRTMC_DISABLE_CUDA_GRAPH`. Change one runtime
setting at a time and record the resolved config. Treat TensorRT-RTX selection
as a separate backend experiment; do not attribute a `--rtx` result to a
runtime-config or precision change.

## Release Performance Matrix

Validate the selected row and environment before execution:

```bash
python3 tools/perf_matrix.py check \
  benchmarks/performance/release.yaml \
  --environment benchmarks/performance/environments/gb300.yaml \
  --entry <family.operation-or-profile-qualified-id>
```

Then run the exact row:

```bash
python3 tools/perf_matrix.py run \
  benchmarks/performance/release.yaml \
  --environment benchmarks/performance/environments/gb300.yaml \
  --entry <family.operation-or-profile-qualified-id>
```

The entry ID is not a model name. The checked-in suite owns the workload,
reference, measurement scope, warmups, iterations, and traffic-light margins.
The environment owns machine-specific executables, caches, and storage.

Use:

```bash
python3 tools/perf_matrix.py resume <run-directory>
```

only for an incomplete run produced by that matrix. Do not combine partial
results from different source revisions or targets.

When bundle preparation ran separately, attach its exact-revision receipt and
regenerate the report:

```bash
python3 tools/perf_matrix.py report <run-directory> \
  --preparation-receipt <bundle-preparation.json>
```

The receipt must use the `test_task` scope and match the campaign revision and
bundle paths. Controlled Internal CI may run the same matrix, but its raw
performance reports, runner details, and artifacts remain private. Source PRs
may report only authorized sanitized evidence.

## Interpretation

Attribute cost using recorded samples, not universal percentages:

- host-to-device, synchronization, or greedy selection suggests runtime
  overhead;
- binding/setup suggests launch or orchestration overhead;
- execution or a small set of layers suggests graph/kernel work;
- setup/build cost is separate from infer-time comparison;
- zero or missing provider phase fields mean unavailable unless the provider
  contract says otherwise.

Large speedups require extra baseline scrutiny: same inputs, warmups,
synchronization, dtype, compile mode, and measured boundary. Never describe a
speedup as meaningful when output equivalence failed.

## Before/After Rule

Use the same:

- repository base and model revision;
- bundle/runtime kind;
- target and runtime config;
- testcase/request and random seed;
- caches and preparation policy;
- warmup, iteration, and synchronization contract;
- reference backend and compile mode.

If any differ, report the confounder instead of a single causal percentage.

## Report

Lead with the evidence level and correctness result. Include exact commands,
SHAs/hashes, hardware, resolved runtime config, workload and measurement
contract, raw artifact paths, p50 and other suite-owned statistics, output
equivalence, observed bottleneck, comparison limitations, and the next
evidence-backed experiment.

<!-- Collaborative review anchor. -->
