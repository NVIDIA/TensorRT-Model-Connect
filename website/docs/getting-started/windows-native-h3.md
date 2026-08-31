---
title: Native Windows MiniMax H3
description: Build and run the fixed MiniMax H3 video path from authorized local inputs.
---

This is the source-build path for the native MiniMax H3 video runtime on
64-bit Windows. It uses TensorRT-RTX and builds the model's large plans in
separate processes so they are not all resident during the build.

This path is deliberately narrow:

- one Blackwell GPU with compute capability 12.x (the build emits SM 120 code);
- batch size 1 with tensor and context parallel size 1;
- BF16, TensorRT-RTX weight streaming, and the six-plan FirstBlockCache layout;
- 1344x768 output, 124 frames, and 50 denoising steps; and
- decoded video frames only.

These constraints describe the implemented profile. They are not benchmark or
qualification results.

:::caution Local assets and licenses

Supply a local MiniMax H3 checkpoint that matches the revision pinned by the
model family and a compatible TensorRT-RTX SDK obtained through channels you
are authorized to use. Review the
[MiniMax H3 license](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/48d93ede732756e404a3b1b2f3b3a9b5a22f6cfc/LICENSE)
and the licenses for the SDK and its Python bindings.

The repository does not download the checkpoint or a proprietary SDK. Keep
checkpoints, SDK files, plans, bundles, and generated media local unless their
licenses permit sharing them.

:::

## Prerequisites

Start in an x64 Visual Studio developer PowerShell with these tools available:

- PowerShell 5.1 or newer and Git;
- Visual Studio 2022 C++ build tools and a Windows SDK;
- CMake and Ninja;
- CUDA Toolkit 12.9 for Windows;
- a compatible NVIDIA display driver with `nvidia-smi` available;
- a CUDA 12.9-compatible
  [TensorRT for RTX](https://developer.nvidia.com/tensorrt-rtx) SDK for
  Windows; and
- a Python environment containing the project dependencies and the Python
  bindings supplied for the selected SDK.

The fixed hot profile requires about 64 GiB of CUDA-visible memory. Leave
enough system commit/pagefile and local disk space for the checkpoint, six
plans, staged bundle, build tree, and evidence. Close competing GPU and
unified-memory workloads before measuring.

The commands below use placeholders for the two SDK roots and the authorized
checkpoint. Do not commit those local paths.

## Build the native runtime

From the repository root:

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
if ($LASTEXITCODE -ne 0) { throw 'Native Windows build failed' }
```

The helper accepts only local SDK roots, selects the MiniMax H3 runtime model,
disables unrelated backends, and enables the distributable build mode.
It rejects a dirty checkout, records the exact Git revision in the worker, and
exports the same revision for the subsequent Python bundle build.
`-BuildTests` runs the portable DLL loader, backend loader, and CLI argument
tests. `-BuildBenchmarks` builds the isolated public-call benchmark worker. Its
outputs are:

```text
<build-directory>\trtmc.exe
<build-directory>\trtmc_core.dll
<build-directory>\trtmc_backend_trt_rtx.dll
<build-directory>\trtmc_benchmark_worker.exe
<build-directory>\models\minimax_h3\trtmc_model_minimax_h3.dll
```

## Build the bundle

Activate the prepared Python environment, install the source package, and make
the selected SDK runtime directories available to this PowerShell process:

```powershell
python -m pip install --no-deps -e . -C py-only=true

$RepositoryRoot = (Resolve-Path '.').Path
$BuildRoot = (Resolve-Path $BuildDirectory).Path
$env:TRTMC_MINIMAX_H3_SOURCE_REVISION = (& git rev-parse HEAD).Trim()
$env:PATH = @(
    (Join-Path $RtxRoot 'bin')
    (Join-Path $CudaRoot 'bin')
    $BuildRoot
    $env:PATH
) -join [IO.Path]::PathSeparator

$Checkpoint = '<authorized-local-MiniMax-H3-snapshot>'
$ArtifactRoot = Join-Path $RepositoryRoot '.local\minimax-h3'
$Bundle = Join-Path $ArtifactRoot 'model.bundle'
New-Item -ItemType Directory -Force -Path $ArtifactRoot | Out-Null

& (Join-Path $BuildRoot 'trtmc.exe') build $Checkpoint `
    --rtx `
    --precision bf16 `
    --output $Bundle
if ($LASTEXITCODE -ne 0) { throw 'MiniMax H3 bundle build failed' }
```

The MiniMax H3 family routes this command through its fixed six-plan staged
builder. `--precision bf16` makes the fixed precision explicit; other
precisions, arbitrary shapes, larger batches, and distributed builds are not
supported by this Windows path.

Plans are environment-bound. The local resume record binds them to sanitized
checkpoint-content, Python staged-builder, and SDK/backend-cohort identities
without storing local paths or hardware identifiers. If that identity changes,
use a fresh plans directory rather than reusing or editing staged artifacts.
Rebuild the native runtime before generating video after C++ source changes.
Keep `trtmc_core.dll`, the RTX backend, the MiniMax H3 plugin, and the worker
from the same build directory; mixing DLLs from different revisions is not a
supported ABI configuration.

## Generate video

`generate-video` accepts `--prompt`, not `--prompt-file`. Load the prompt field
from the checked-in public example, then run the native executable:

```powershell
$PromptSpec = Get-Content -Raw -LiteralPath `
    '.\tests\e2e\models\minimax_h3\prompts\t2va-example-1.json' |
    ConvertFrom-Json
$Frames = Join-Path $ArtifactRoot 'frames'

& (Join-Path $BuildRoot 'trtmc.exe') generate-video $Bundle `
    --prompt $PromptSpec.prompt `
    --output $Frames `
    --backend-dir $BuildRoot `
    --model-plugin-dir (Join-Path $BuildRoot 'models')
if ($LASTEXITCODE -ne 0) { throw 'MiniMax H3 video generation failed' }
```

The output directory contains the decoded video frames. This command does not
claim audio-output support.

## Reproduce the same-process hot benchmark

The optional hot benchmark keeps deserialized TensorRT-RTX engines across
same-process requests while rotating denoiser and VAE execution contexts. It
does not make the complete model resident, encode an MP4, or include cold
startup in the reported call time.

See the [MiniMax H3 Windows hot benchmark](../reference/minimax-h3-windows-hot-benchmark.md)
for the exact timing boundary, command, evidence files, and capacity caveats.

## Validation boundary

Static checks and Linux tests do not prove that MSVC compilation, Windows DLL
loading, plan construction, or inference works on a real Windows GPU. Treat
this as a local supported path only after the exact source revision has passed
the applicable Windows build and real-GPU checks. This page itself is not a
performance, memory-capacity, visual-parity, or qualification receipt.
