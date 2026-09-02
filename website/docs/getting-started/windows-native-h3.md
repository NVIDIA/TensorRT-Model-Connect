---
title: Native Windows MiniMax H3
description: Build, install, and run the public H3-Base workflows with the native ModelConnect TensorRT-RTX runtime.
---

This is the native Windows path for the public MiniMax H3-Base generation
workflows. Bundle construction is an offline release-engineering step. The
installed generation path is ModelConnect C++/CUDA plus TensorRT-RTX and
Windows system media APIs; it does not ship or invoke Python, PyTorch,
FastVideo, Triton, FFmpeg, or a subprocess.

## Supported H3-Base surface

One bundle can expose these workflows:

- T2VA: prompt to synchronized video and audio;
- FL2VA: first frame, last frame, or both, plus a prompt; and
- Ref2VA: an ordered mix of reference images, videos, and audio plus a prompt.

The target is 24 fps with 32 kHz stereo audio. Target frames follow the video
VAE's `17 * n + 5` alignment. A nominal five-second request uses 120 requested
frames and produces 124 frames (5.167 seconds); the longest local aligned
request is 345 frames (14.375 seconds). Canvases are multiples of 32, cover
aspect ratios from 1:4 through 4:1, use a 768-pixel short edge where the public
pixel budget permits it, and are capped at the 768x1344 pixel budget.

Ref2VA preserves argument order and enforces the released limits: at most 9
images, 3 videos, 3 explicit audio files, and 12 files total. Each video or
explicit audio reference is 2--15 seconds; total video duration and total
explicit-audio duration are each at most 15 seconds. An audio reference may be
the only reference modality. A video's soundtrack remains attached to that
video reference and does not consume the explicit-audio count or duration quota.

This integration intentionally does not implement H3-Context-IR or
H3-Regenerate-2K. They are separate services, not H3-Base checkpoint
capabilities. Bundle metadata records both exclusions as `false`, and no plan
section or CLI option implements or claims either service.

## Runtime boundary

The installable payload contains only:

- the runtime-only `trtmc.exe`, ModelConnect core, and the MiniMax H3 model
  plugin;
- the TensorRT-RTX backend and matching TensorRT-RTX runtime DLL;
- the locally built H3 bundle; and
- the native installer/uninstaller and license files.

CUDA runtime and the MSVC runtime are statically linked in this distribution.
The package step audits every PE import and rejects Python, Torch, CUDA runtime
DLL, FFmpeg, FastVideo, Triton, or dynamic MSVC runtime dependencies. Windows
Media Foundation provides native MP4 decode and H.264/AAC encode. The NVIDIA
display driver and Windows system DLLs remain platform prerequisites.

Python is used only on the build machine to translate authorized checkpoints
and adapters into TensorRT-RTX plans. It is not copied into the installer and
is not needed to install or generate media.

## Build the runtime

Start in an x64 Visual Studio 2022 developer PowerShell. Supply compatible
local CUDA 12.9 and TensorRT-RTX SDK roots. The helper requires a clean Git
checkout so the bundle and binaries can record one source revision.

```powershell
$CudaRoot = '<CUDA-12.9-root>'
$RtxRoot = '<TensorRT-RTX-root>'
$BuildRoot = 'D:\build\modelconnect-h3'

& .\scripts\build_windows_h3.ps1 `
    -CudaRoot $CudaRoot `
    -TensorRtRtxRoot $RtxRoot `
    -BuildDirectory $BuildRoot `
    -BuildTests
if ($LASTEXITCODE -ne 0) { throw 'Native runtime build failed' }
```

The helper builds a runtime-only CLI and the native setup executable with
static CUDA/MSVC runtimes. It does not build or download model weights.

## Build the complete bundle

Prepare a build-only Python environment with this project, the Python binding
from the selected TensorRT-RTX SDK, and the checkpoint-reading packages. The
authorized local checkpoint must contain the shared H3 components and
`transformer/`. Complete Ref2VA additionally requires the released, pinned
`transformer_ref/` partition. The accelerated T2VA/FL2VA profile requires the
strictly validated FastH3 adapter file.

```powershell
python -m pip install --no-deps -e . -C py-only=true

$Checkpoint = '<authorized-MiniMax-H3-snapshot>'
$FastH3Adapter = '<authorized-FastH3-adapter.safetensors>'
$TransformerRef = Join-Path $Checkpoint 'transformer_ref'
$Bundle = 'D:\artifacts\MiniMax-H3.bundle'
$env:TRTMC_MINIMAX_H3_SOURCE_REVISION = (& git rev-parse HEAD).Trim()
$env:PATH = @(
    (Join-Path $RtxRoot 'bin')
    (Join-Path $CudaRoot 'bin')
    $env:PATH
) -join [IO.Path]::PathSeparator

python -m tensorrt_model_connect build $Checkpoint `
    --rtx `
    --precision bf16 `
    --output $Bundle `
    --set "minimax_h3.fast_h3_adapter=$FastH3Adapter" `
    --set "minimax_h3.transformer_ref=$TransformerRef"
if ($LASTEXITCODE -ne 0) { throw 'MiniMax H3 bundle build failed' }
```

The builder validates exact checkpoint/adapter provenance and fails closed.
It never substitutes `transformer/` when `transformer_ref/` is absent. T2VA
and FL2VA use the authenticated native VSA 4-forward plan set. Ref2VA uses its
independent dense `transformer_ref` plans and released 50-point/49-forward
schedule with video/audio shifts 12 and 3.

The plans and bundle are very large. Put the checkpoint, plan-resume directory,
and output on a local NTFS volume with enough space for the checkpoint, plans,
and final bundle. Do not build and infer concurrently on a unified-memory
machine. The command creates a sibling `$Bundle.plans` directory and writes an
atomic receipt after every completed plan. Rerun the exact same command after
an interruption to resume; a checkpoint, adapter, source, TensorRT-RTX, or
workspace-profile mismatch fails closed instead of reusing incompatible
plans. The bundle and its `.effective_config.json` sidecar contain public
model identities and content digests, never the local checkpoint, adapter, or
`transformer_ref` paths.

## Create and install the native package

Run the packaging step in the same x64 developer PowerShell so `dumpbin.exe`
is available:

```powershell
$Package = 'D:\artifacts\MiniMax-H3-Windows'
& .\scripts\package_windows_h3.ps1 `
    -BuildDirectory $BuildRoot `
    -BundlePath $Bundle `
    -OutputDirectory $Package
if ($LASTEXITCODE -ne 0) { throw 'Native package failed' }
```

Double-click `MiniMaxH3Setup.exe` in that directory. The installer verifies
the SHA-256 manifest before a transactional per-user install, registers
`trtmc.exe`, and adds its `bin` directory to the user PATH unless `--no-path`
is selected. The default bundle path after installation is:

```text
%LOCALAPPDATA%\Programs\ModelConnect\MiniMax-H3\models\MiniMax-H3.bundle
```

## Generate video through ModelConnect

Open a new PowerShell after installation:

```powershell
$Bundle = Join-Path $env:LOCALAPPDATA `
    'Programs\ModelConnect\MiniMax-H3\models\MiniMax-H3.bundle'
```

T2VA, nominal five seconds, 16:9:

```powershell
trtmc generate-video $Bundle `
    --prompt 'A cinematic sunrise over a mountain lake with synchronized birds and wind.' `
    --num-frames 120 --height 768 --width 1344 --seed 0 `
    --output .\t2va-5s.mp4
```

T2VA, longest aligned local output:

```powershell
trtmc generate-video $Bundle `
    --prompt 'A continuous documentary shot with synchronized dialogue and ambience.' `
    --num-frames 345 --height 768 --width 1344 --seed 0 `
    --output .\t2va-14.375s.mp4
```

FL2VA accepts first-only, last-only, or both:

```powershell
trtmc generate-video $Bundle `
    --prompt 'Continue naturally between the supplied endpoints with synchronized ambience.' `
    --first-frame .\first.png --last-frame .\last.png `
    --num-frames 120 --seed 7 --output .\fl2va.mp4
```

Ref2VA preserves the order of the reference flags:

```powershell
trtmc generate-video $Bundle `
    --prompt 'Use <Picture 1> as the subject and <Audio 1> as the voice reference.' `
    --reference-image .\subject.png `
    --reference-audio .\voice.wav `
    --num-frames 120 --height 768 --width 1344 --seed 11 `
    --output .\ref2va.mp4
```

`--reference-video` accepts a native MP4 or a ModelConnect video directory.
MP4 output is written directly with H.264 video and, when generation succeeds,
one AAC stereo 32 kHz audio stream. The CLI prints separate pipeline-load,
input-decode, generation, media-write, and total timing fields to stderr.

See the [Windows H3 benchmark contract](../reference/minimax-h3-windows-hot-benchmark.md)
for the accepted timing boundary and artifact checks.
