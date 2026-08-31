---
title: MiniMax H3 Windows hot benchmark
description: Reproduce and audit same-process hot MiniMax H3 video-generation calls on Windows.
---

This benchmark measures the public MiniMax H3 video-generation call after one
same-process warmup. It is intended to make a locally authorized Windows
TensorRT-RTX result reproducible and auditable; it is not a portable latency
guarantee or a substitute for visual-quality qualification.

## Fixed workload

The launcher fixes the workload to:

- `MiniMaxAI/MiniMax-H3` using a locally authorized staged bundle;
- batch size 1, seed 0, and the checked-in public prompt;
- 1344x768 RGB output with 124 frames;
- 50 denoising grid points;
- one untimed pipeline warmup and two measured calls in one worker process;
- FirstBlockCache threshold `0.30`;
- retained denoiser head, tail, finish, and VAE engines;
- a 24 GiB retained tail weight budget, capped by the bundle budget;
- rotating denoiser/VAE execution contexts; and
- CUDA graphs disabled because the tail uses weight streaming.

The threshold and retained budget are a Windows 64 GiB cohort baseline. They
are not universally optimal. Keep them unchanged when reproducing that profile;
report different values as a separate experiment.

## Timing contract

For each measured sample, the timer starts immediately before the direct
`IPipeline::generate_image` call and stops as soon as it returns the host float
tensor. The measured call includes cache lookup, denoising, scheduler work,
VAE decoding, device-to-host transfer, and host output assembly. On the staged
64 GiB profile it also includes execution-context recreation.

The reported sample excludes:

- bundle hashing, validation, and pipeline construction;
- the first untimed pipeline warmup, plan deserialization, and JIT preparation;
- destruction of the previous host result;
- the post-return non-finite scan;
- GPU telemetry and video encoding; and
- writing benchmark evidence to disk.

Consequently, the complete command takes substantially longer than one reported
sample. With the default `1 + 2` request sequence, expect one untimed request
plus two hot requests. The launcher does not claim a storage-cold or system-cold
start because prior processes and bundle hashing can populate OS caches. Two
samples are enough to reproduce a specific run, but they are not a statistically
stable production p50.

## Build

Start in an x64 Visual Studio developer PowerShell and build from a clean Git
checkout. CUDA and TensorRT-RTX must be the compatible Windows SDK cohort used
to construct the supplied bundle.

The accepted baseline requires PowerShell 5.1 or newer, Git, Visual Studio 2022
C++ tools, `nvidia-smi`, a compatible driver, and the prepared project Python
environment kept active for authoritative bundle validation. It also requires a
compute-capability 12.x GPU with at least 60,000 MiB reported memory. Provision
enough pagefile/system commit and disk for the checkpoint, six plans, bundle,
build, and evidence.

```powershell
$CudaRoot = '<local-CUDA-12.9-Toolkit-root>'
$RtxRoot = '<local-TensorRT-RTX-SDK-root>'
$BuildDirectory = 'build-windows-h3'

& .\scripts\build_windows_h3.ps1 `
    -CudaRoot $CudaRoot `
    -TensorRtRtxRoot $RtxRoot `
    -BuildDirectory $BuildDirectory `
    -BuildTests `
    -BuildBenchmarks
```

The build records the clean source revision in the worker. The benchmark
launcher rejects a worker built from another revision. The build helper also
exports that revision to the Python builder so a newly built bundle carries the
same source provenance. Build the core, RTX backend, MiniMax H3 plugin, and
Release worker together; mixing DLLs from different revisions is unsupported.
The worker revision is verified directly; the other binaries are hashed. The
launcher also requires the CUDA and TensorRT-RTX DLLs beside the worker to match
the selected SDK files byte for byte.

## Run

Choose an output directory outside the source checkout and close competing GPU
or unified-memory workloads.

```powershell
$Bundle = '<authorized-local-MiniMax-H3-bundle>'
$EvidenceRoot = '<private-directory-outside-the-checkout>'

& .\scripts\run_windows_h3_hot_benchmark.ps1 `
    -Bundle $Bundle `
    -CudaRoot $CudaRoot `
    -TensorRtRtxRoot $RtxRoot `
    -BuildDirectory $BuildDirectory `
    -OutputDirectory $EvidenceRoot `
    -Warmup 1 `
    -Iterations 2 `
    -FirstBlockCacheThreshold 0.30 `
    -TailWeightBudgetGiB 24
```

The script computes the large bundle hash by default. Use `-SkipBundleHash`
only for an exploratory run and disclose that the resulting receipt has weaker
artifact identity. A run that skips it is never marked as the baseline profile.

By default the launcher invokes the checked-in authoritative Python validator
and requires the bundle's source, public checkpoint revision, current builder
source hash, declared checkpoint inventory, six-plan BF16 profile, CUDA 12.9,
and TensorRT-RTX provenance to agree with the checkout and selected SDK.
`-AllowUnverifiedBundleProvenance` is an explicit diagnostic escape hatch for
older local bundles; such a run is never marked as the baseline profile.

`-UseFastExit` is an opt-in diagnostic workaround for a worker that finishes
inference and persists its result but fails during third-party DLL teardown.
It bypasses global C++ destruction and is recorded in `summary.json`; a run that
uses it proves inference completion, not normal process teardown.
It produces `status=inference_completed` and `lifecycle_status=bypassed`, not
the normally accepted `status=completed` result.

## Evidence and acceptance

Preflight validates the clean checkout, Release worker, bundle, SDK cohort,
hardware, and artifact hashes before it creates an evidence directory. A
successful run writes a timestamped directory containing:

- `worker-request.json`, including the resolved paths and runtime configuration;
- `worker-metadata.json`, including the compiled source revision;
- `environment.json`, including toolchain versions, non-UUID GPU identity,
  bundle provenance, effective requested budgets, telemetry status, and
  artifact hashes;
- `worker.log`, `worker.stdout.log`, and optional `gpu-telemetry.csv`;
- `worker-result.json`, containing both raw measured samples; and
- `summary.json`, containing the median, exit mode, and output summary.

A worker-phase failure normally writes final `environment.json` and
`summary.json` records if the evidence directory remains writable.
`worker-result.json` exists only when the worker persisted a result. A preflight
failure writes no evidence directory.

An accepted baseline run has worker exit code 0, `status=completed`,
`lifecycle_status=normal_teardown`, `baseline_profile=true`, verified bundle
provenance, two finite positive observations, 124 frames, 383,975,424 RGB
elements, zero non-finite elements, the checked-in prompt, a complete bundle
hash, CUDA Toolkit 12.9, and matching side-by-side runtime DLLs. The launcher
also requires four retained engine-cache misses on the first request, cache hits
on later requests, and a runtime weight-budget record matching the
bundle-capped tail request. Text and AdaLN should hit their same-prompt caches
after warmup.

FirstBlockCache step positions can vary with the bundle, SDK, and numerical
trajectory. A matching count does not establish pixel, perceptual, or visual
equivalence. Keep playable-video inspection and quality metrics as a separate
qualification result.

## Capacity and portability

Retained engines trade memory for avoiding repeated plan reads and
deserialization. The retained cache is opt-in, scoped by CUDA device, and lives
for the backend process lifetime. The staged tail still streams weights and the
pipeline still rotates denoiser and VAE contexts.

Each new plan digest or requested budget can add another process-lifetime cache
entry, so this mode is intended for the isolated benchmark worker rather than
an unbounded multi-model service. On Windows, loaded backend DLLs are likewise
kept until process exit to avoid unsafe vendor-runtime destruction under the
loader lock; explicit backend unload/reload is not supported in this mode.

The repository does not redistribute checkpoints, SDK files, plans, bundles,
generated frames, telemetry, or benchmark receipts. Rebuild the bundle with
authorized inputs when the checkpoint, source, GPU, CUDA, or TensorRT-RTX cohort
changes.

Evidence is intentionally private by default: `worker-request.json` contains
resolved local paths, while receipts contain full artifact hashes and hardware
details. Keep the output directory outside the checkout. Redact or transform
those fields before sharing evidence beyond the authorized environment.
