---
title: Native Windows MiniMax H3
description: Build, install, and run the public H3-Base workflows with the native ModelConnect TensorRT-RTX runtime.
---

This is the native Windows path for the public MiniMax H3-Base generation
workflows. Bundle construction is an offline release-engineering step. The
installed generation path is ModelConnect C++/CUDA plus TensorRT-RTX and Windows
system media APIs; it does not ship or invoke Python, PyTorch, FastVideo,
Triton, FFmpeg, or a subprocess.

## Supported H3-Base surface

One bundle can expose these workflows:

- T2VA: prompt to synchronized video and audio;
- FL2VA: first frame, last frame, or both, plus a prompt; and
- Ref2VA: an ordered mix of reference images, videos, and audio plus a prompt.

The default bundle uses the official original H3 BF16 weights and dense
attention. It does not require a FastH3 adapter, LoRA, VSA engine, or an
external attention runtime. T2VA and FL2VA share one six-plan,
FirstBlockCache-capable visual implementation, while Ref2VA uses the official
`transformer_ref/` weights and its released dense conditioning schedule. The
shared denoiser head, tail, and finish engines contain two TensorRT optimization
profiles: profile 0 specializes the qualified 537-token, 124-frame, 1344x768
T2VA request, and profile 1 covers the public dynamic envelope. The runtime
selects the profile for each request; there is no separate five-second and
fifteen-second model or engine set.

The target is 24 fps with 32 kHz stereo audio. Target frames follow the video
VAE's `17 * n + 5` alignment. A nominal five-second request uses 120 requested
frames and produces 124 frames (5.167 seconds); the longest local aligned
request is 345 frames (14.375 seconds). Profile 0 is an exact-shape performance
specialization, not a fixed-prompt interface. Every other valid request,
including a different prompt length or a longer duration, routes to profile 1.
That finite native TensorRT dynamic profile covers every aligned duration from
124 through 345 frames. It accepts all 95 canvases emitted by the public
resolver (multiples of 32, trained aspect ratios from 1:4 through 4:1, a
768-pixel short edge where the `768x1344` pixel budget permits it), plus the
official explicit performance canvas `--height 544 --width 960` and its
transpose. Other explicit multiple-of-32 canvases accepted by the eager
Diffusers API are not silently generalized: the native runtime rejects them
before plan execution.

T2VA tokenizes each prompt supplied to the request; the prompt is not fixed or
padded to the qualification prompt. Profile 1 accepts 1--2641 tokens without
truncation. Prompts beyond that finite engine profile fail before generation
instead of being silently shortened.
The packaged single-video command accepts exactly one non-negative integer
`--seed`; CSV or negative values fail before loading the model.

Ref2VA preserves argument order and enforces the released limits: at most 9
images, 3 videos, 3 explicit audio files, and 12 files total. Each video or
explicit audio reference is 2--15 seconds; total video duration and total
explicit-audio duration are each at most 15 seconds. An audio reference may be
combined with image/video references but cannot be the only reference modality.
A video's soundtrack remains attached to that video reference and does not
consume the explicit-audio count or duration quota.
The Windows media reader scales video directly onto that aspect ratio's bounded
H3 resolver canvas and drops source rates above 24 fps before allocating float
frames; source-rate metadata above 240 fps fails closed. Reference presentation
timestamps remain strictly bounded to 15 seconds. Native MP3/AAC encoder
priming and tail padding are trimmed at that boundary without admitting a real
over-15-second presentation.

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
into TensorRT-RTX plans. It is not copied into the installer and is not needed
to install or generate media. Loading engines, executing every forward pass,
and writing the final MP4 all happen in the native C++ runtime.

The locked runtime never emits an effective-config file or another implicit
sidecar. `--runtime-cache PATH` is the one explicit exception: when the user
selects it, TensorRT-RTX may persist its JIT cache at exactly that path. No
cache file is created by default.

The hot T2VA visual path uses six TensorRT-RTX plans: `text_encoder.plan`,
`adaln_precompute.plan`, `denoiser_head.plan`, `denoiser_tail.plan`,
`denoiser_finish.plan`, and `vae_tile_decoder.plan`. The tail is one engine
covering transformer blocks 1 through 49; it is not split into 49 per-block
engines. T2VA additionally invokes `audio_vae_decoder.plan`. A complete FL2VA
bundle also contains `vision_encoder.plan` and
`fl2va_keyframe_vae_encoder.plan`, which are used when endpoint images are
supplied. The singular visual route preserves bounded phase residency while
avoiding per-block engine churn. Every supported request uses the same six
visual plan files, original weights, scheduler, and dense attention graph; only
the TensorRT optimization-profile index changes. The audio decoder is an
additional T2VA plan and is not part of the six-plan visual count.

The released AudioVAE treats the two stereo latents as independent inputs to
one shared mono decoder. The SM121 runtime invokes that same TensorRT-RTX plan
once per real channel, duplicates the selected latent only to satisfy the
plan's fixed batch-two profile, and retains batch item zero from each launch.
This preserves the original left and right latents without alternate weights
or a non-TensorRT audio path.

## Build the runtime

Start in an x64 Visual Studio 2022 developer PowerShell. Supply compatible
local CUDA 12.9 and TensorRT-RTX SDK roots. The locked distributable helper is
the SM121 Spark qualification build. With `-BuildTests`, GPU tests require a
live compute-capability 12.1 GPU. A developer build for another supported GPU
is outside this locked package and performance contract. The helper requires a
clean Git checkout so the bundle and binaries can record one source revision.

```powershell
$RepoRoot = (Resolve-Path '<ModelConnect-checkout>').Path
Set-Location -LiteralPath $RepoRoot
$CudaRoot = '<CUDA-12.9-root>'
$RtxRoot = '<TensorRT-RTX-root>'
$BuildRoot = 'D:\build\modelconnect-h3'

& (Join-Path $RepoRoot 'scripts\build_windows_h3.ps1') `
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
`transformer_ref/` partition. No adapter checkpoint is required.

Use one of two strictly verified pinned checkpoint layouts. A Hugging Face
cache view is the exact path printed by this pinned download:

```powershell
$H3Revision = '48d93ede732756e404a3b1b2f3b3a9b5a22f6cfc'
$Checkpoint = (& hf download MiniMaxAI/MiniMax-H3 --revision $H3Revision).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Pinned MiniMax-H3 download failed' }
```

An existing `hf download --local-dir` tree is also accepted. Keep its
`.cache\huggingface\download\*.metadata` files: the builder requires exactly
one metadata record for every base-checkpoint content file and verifies that
each record names `$H3Revision`. For both layouts, the builder performs one
complete content pass and requires every relative path, byte count, Hugging
Face blob ID, and SHA-256 to match the checked-in 55-file manifest for that
revision. A copied local-dir tree without its metadata is rejected. Downloader
`.incomplete` and `.lock` debris is not checkpoint content. `transformer_ref/`
is excluded from the base inventory and validated independently against its
fixed 14-shard SHA-256 manifest.

```powershell
$Checkpoint = '<path-to-MiniMax-H3-checkpoint>'
hf download MiniMaxAI/MiniMax-H3 `
    --revision $H3Revision `
    --local-dir $Checkpoint
if ($LASTEXITCODE -ne 0) { throw 'Pinned checkpoint download failed' }
```

Install the build-only package and prove the source/build revision boundary
before starting the bundle build:

```powershell
python -m pip install --no-deps -e $RepoRoot -C py-only=true
if ($LASTEXITCODE -ne 0) { throw 'ModelConnect build package install failed' }

$SourceRevision = (& git -C $RepoRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $SourceRevision -notmatch '^[0-9a-f]{40}$') {
    throw 'Unable to resolve the ModelConnect source revision'
}
if ((& git -C $RepoRoot status --porcelain | Out-String).Trim()) {
    throw 'MiniMax-H3 bundle construction requires a clean source checkout'
}
$RuntimeRevisionLine = (Select-String -LiteralPath `
    (Join-Path $BuildRoot 'CMakeCache.txt') `
    -Pattern '^TRTMC_SOURCE_REVISION:STRING=').Line
$RuntimeRevision = ($RuntimeRevisionLine -split '=', 2)[1]
if ($RuntimeRevision -ne $SourceRevision) {
    throw "Runtime/source revision mismatch: runtime=$RuntimeRevision source=$SourceRevision"
}

$TransformerRef = Join-Path $Checkpoint 'transformer_ref'
$Bundle = 'D:\artifacts\MiniMax-H3.bundle'
$env:TRTMC_MINIMAX_H3_SOURCE_REVISION = $SourceRevision
$env:PATH = @(
    (Join-Path $RtxRoot 'bin')
    (Join-Path $CudaRoot 'bin')
    $env:PATH
) -join [IO.Path]::PathSeparator

python -m tensorrt_model_connect build $Checkpoint `
    --rtx `
    --precision bf16 `
    --output $Bundle `
    --set "minimax_h3.transformer_ref=$TransformerRef"
if ($LASTEXITCODE -ne 0) { throw 'MiniMax H3 bundle build failed' }
```

Without an adapter setting, the builder selects the official original-weight
dense path. T2VA and FL2VA use the singular FirstBlockCache engine set described
above. Ref2VA uses its independent dense `transformer_ref` plans and released
50-point/49-forward schedule with video/audio shifts 12 and 3. The builder
never substitutes `transformer/` when `transformer_ref/` is absent.

The large dense-denoiser and AdaLN engines deliberately do not set an explicit
TensorRT workspace memory-pool limit. TensorRT-RTX therefore uses its default
maximum workspace policy and chooses tactics against the memory available on
the build machine. This is intentional for 128 GiB unified-memory Spark
systems; setting a smaller manual limit can make otherwise valid Myelin tactics
fail during engine construction. Smaller auxiliary engines retain their
recorded per-plan limits.

The plans and bundle are very large. Put the checkpoint, plan-resume directory,
and output on local storage with enough capacity, and do not build and infer
concurrently on a unified-memory machine. The command creates a sibling
`$Bundle.plans` directory and writes an atomic receipt after every completed
plan. Final assembly uses a stable same-directory bundle partial and journal.
After each plan range is copied, flushed, and SHA-256 verified, the journal is
atomically committed before that exact source plan is removed. Consequently,
the builder does not retain both a complete plans directory and a complete
second bundle copy: its assembly peak is approximately the checkpoint plus the
complete plan set plus the largest single plan. The base T2VA/FL2VA inventory
contains nine plans: the six-plan visual route, the audio decoder, and the two
FL2VA conditioning plans. Check free space on the selected volume before
starting; complete Ref2VA adds the dedicated `transformer_ref` plans.

Rerun the exact same command after an interruption to resume. Do not remove the
`.partial`, `.partial.json`, receipt, or surviving plan files while recovery is
in progress. Recovery rehashes every committed range, truncates an uncommitted
tail, and never rebuilds a plan already preserved in the committed bundle
prefix. A checkpoint, source, TensorRT-RTX, or workspace-profile
mismatch fails closed instead of reusing incompatible plans. The checked-in
exact base inventory is part of that resume identity; staged
receipts created by a builder that predates this inventory cannot be resumed.
Every resumed invocation fully validates its sources on entry, and an observed
post-build source-validation failure invalidates that plan set before returning.
Ordinary process interruption remains resumable; this practical consistency
boundary does not attempt to defend against a privileged actor that swaps and
exactly restores sources around both validation passes.

The bundle and the build-only `.effective_config.json` file contain public
model identities and content digests, never the local checkpoint or
`transformer_ref` paths. The package does not install that build record, and
generation does not re-create it.

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

By default, packaging makes an independent copy of the bundle. The package is
therefore unaffected if the build bundle is later modified, but the packaging
step needs one additional bundle-sized allocation. On a space-constrained
release machine, explicitly consume the build artifact with a same-volume
atomic rename:

```powershell
& .\scripts\package_windows_h3.ps1 `
    -BuildDirectory $BuildRoot `
    -BundlePath $Bundle `
    -OutputDirectory $Package `
    -ConsumeBundle
```

`-ConsumeBundle` never falls back to copy-and-delete and fails if the input and
package are on different volumes. Its use removes the original `$Bundle` path;
after the move, the only copy is
`$Package\payload\models\MiniMax-H3.bundle`. If a later package validation
fails, recover or reuse the bundle from that exact package path. The default
mode never consumes its input, and neither mode creates a hard link.

After creating `payload.manifest`, the package step stamps those exact bytes as
an RCDATA resource in the outer `MiniMaxH3Setup.exe`. Setup rejects an external
manifest that differs by even one byte before parsing it, so changing a payload
and recomputing the adjacent manifest is not sufficient. This anchor establishes
authenticity only when users trust the exact outer Setup: release engineering
must Authenticode-sign the stamped Setup and/or publish its official SHA-256.
An unsigned Setup downloaded without an independently authenticated SHA does
not protect against replacement of the entire Setup, manifest, and payload.

Double-click `MiniMaxH3Setup.exe` in that directory. The installer verifies
the SHA-256 manifest before and after an independent transactional copy,
registers `trtmc.exe`, and adds its `bin` directory to the user PATH unless
`--no-path` is selected. Replacement uses same-directory staging, backup, and
recovery renames. If a commit or final verification fails, the prior backup is
restored first; a locked tree is preserved at the exact recovery path reported
by Setup instead of being silently deleted. The installed bundle is never
hard-linked to the mutable package layout, so installation requires free space
at least equal to the packaged payload.

Install and uninstall mutations for the same current user and canonical
destination are serialized across Windows login and RDP sessions. A competing
Setup waits for at most 30 minutes, then exits with an exact timeout diagnostic;
an abandoned owner is acquired so the fixed recovery paths can be repaired
under the same lock. `--help` and `--verify-only` remain lock-free because they
do not mutate an installation.

The default bundle path after installation is:

```text
%LOCALAPPDATA%\Programs\ModelConnect\MiniMax-H3\models\MiniMax-H3.bundle
```

Setup exit code `0` means both the verified files and Windows registration
completed. Exit code `1` means the file transaction failed. Exit code `2` is an
explicit partial success: the verified runtime files are committed and usable
at the exact CLI path shown by Setup, but App Paths, uninstall metadata, or the
user PATH could not be fully registered. Registry writes are not claimed to be
atomic; rerun Setup to retry them.

## Generate video through ModelConnect

Open a new PowerShell after installation:

```powershell
$Bundle = Join-Path $env:LOCALAPPDATA `
    'Programs\ModelConnect\MiniMax-H3\models\MiniMax-H3.bundle'
```

`generate-video` is optimized for quick startup on this very large bundle. It
does not recompute SHA-256 for plan payloads. It validates the bundle header and
configuration, runtime ABI, section bounds, declared SHA-256 metadata format,
and the bundle file's identity and immutability across each engine-loading
window. This fast path trusts a bundle already verified by the native
installer. Complete payload-content authentication is performed only by the
following explicit command, whose syntax is also shown by `trtmc --help`:

```powershell
trtmc inspect $Bundle --validate-runtime
```

T2VA, nominal five seconds, 16:9:

```powershell
trtmc generate-video $Bundle `
    --prompt 'A cinematic sunrise over a mountain lake with synchronized birds and wind.' `
    --num-frames 120 --height 768 --width 1344 --seed 0 `
    --output .\t2va-5s.mp4
```

T2VA on the official smaller Diffusers performance canvas:

```powershell
trtmc generate-video $Bundle `
    --prompt 'A close tracking shot with synchronized footsteps and city ambience.' `
    --num-frames 120 --height 544 --width 960 --seed 0 `
    --output .\t2va-960x544.mp4
```

T2VA, longest aligned local output:

```powershell
trtmc generate-video $Bundle `
    --prompt 'A continuous documentary shot with synchronized dialogue and ambience.' `
    --num-frames 345 --height 768 --width 1344 --seed 0 `
    --output .\t2va-14.375s.mp4
```

For the opt-in qualified speed preset, keep the engines resident within one CLI
process, use a persistent TensorRT-RTX runtime cache, select its calibrated
`0.30` FirstBlockCache threshold, and include an unmeasured warmup before the
measured iteration:

```powershell
$RepoRoot = (Resolve-Path '<ModelConnect-checkout>').Path
$Prompt = (Get-Content -Raw `
    (Join-Path $RepoRoot `
        'tests\e2e\models\minimax_h3\prompts\t2va-example-1.json') |
    ConvertFrom-Json).prompt
$RuntimeCache = '.\minimax-h3-trt-rtx.cache'
trtmc generate-video $Bundle `
    --prompt $Prompt `
    --num-frames 120 --height 768 --width 1344 --seed 0 `
    --num-inference-steps 50 --guidance-scale 1 `
    --runtime-cache $RuntimeCache `
    --set "minimax_h3.retain_engines=true" `
    --set "minimax_h3.retained_tail_weight_budget_gib=24" `
    --set "minimax_h3.first_block_cache_threshold=0.30" `
    --warmup 1 --benchmark 1 `
    --output .\t2va-5s-benchmark.mp4
```

The `0.30` override above is the explicitly selected and measured dense
FirstBlockCache speed preset for this qualification machine, not the bundle or
runtime default. On the exact profile-0 fixture it executes six full tail
forwards and skips 43 of the 49 dense transformer forwards.
Thresholds are tuning inputs, not universal quality/performance constants:
lower values recompute more tail steps, while higher values may skip more work
but can change output quality. Calibrate non-default values on the target
machine using representative prompts and both short and long durations.

As a ModelConnect-specific quality guard, the first and final denoiser forwards
always execute the tail regardless of the cache metric; a non-finite metric also
forces a full tail evaluation. The threshold applies only to the 47 interior
opportunities in the 49-forward dense schedule, so at most 47 tails can be
skipped. The finish plan still executes on every forward.

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
`--reference-audio` accepts MP3, WAV, and other audio files supported by Windows
Media Foundation; decoding does not require FFmpeg or a Python runtime.
MP4 output is written directly with H.264 video and, when generation succeeds,
one AAC stereo 32 kHz audio stream. The CLI prints separate pipeline-load,
input-decode, generation, media-write, and total timing fields to stderr.

See the [Windows H3 benchmark contract](../reference/minimax-h3-windows-hot-benchmark.md)
for the accepted timing boundary and artifact checks.
