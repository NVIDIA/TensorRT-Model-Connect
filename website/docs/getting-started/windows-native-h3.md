---
title: Native Windows MiniMax H3
description: Build and run H3-Base with ModelConnect C++ and TensorRT-RTX.
---

This integration runs MiniMax H3-Base through the native ModelConnect C++
runtime and TensorRT-RTX. Python is used only to build the bundle; generation
does not invoke Python, PyTorch, FastVideo, Triton, FFmpeg, or a subprocess.
Windows Media Foundation reads reference media and writes H.264/AAC MP4 files.

One bundle supports:

- T2VA from a prompt;
- FL2VA from a prompt plus a first frame, last frame, or both; and
- Ref2VA from a prompt plus ordered image, video, and audio references.

All durations use the official BF16 weights, dense attention graph, and the
same six visual engines. TensorRT optimization profile 0 specializes the
qualified five-second request; profile 1 handles other supported prompt
lengths, canvases, and durations. There is no separate 15-second model.

H3 aligns frame counts to `17 * n + 5` at 24 fps. Consequently, a request for
120 frames produces 124 frames (5.167 seconds), while 345 frames produces
14.375 seconds. Prompts are tokenized per request and are not fixed to the
qualification prompt.

H3-Context-IR and H3-Regenerate-2K are separate services and are not included.

## Build the native runtime

Run the helper from a clean Git checkout in an x64 Visual Studio 2022 developer
PowerShell with Git, CUDA 12.9, Ninja, CMake, and TensorRT-RTX installed:

```powershell
$RepoRoot = (Resolve-Path '<ModelConnect-checkout>').Path
$CudaRoot = '<CUDA-12.9-root>'
$RtxRoot = '<TensorRT-RTX-root>'
$BuildRoot = '<build-directory>'

& (Join-Path $RepoRoot 'src\runtime\models\minimax_h3\build_windows.ps1') `
    -CudaRoot $CudaRoot `
    -TensorRtRtxRoot $RtxRoot `
    -BuildDirectory $BuildRoot
```

The helper builds only the runtime CLI, core library, TensorRT-RTX backend, and
MiniMax-H3 plugin. CUDA and MSVC runtimes are linked statically. The matching
TensorRT-RTX runtime DLL is copied beside `trtmc.exe`.
Pass `-BuildTests` when developing the integration to build and run its native
test set as well.

## Build the bundle

Install this repository's build-only Python package and the checkpoint-reading
dependencies. The builder can download the pinned checkpoint directly from
Hugging Face:

```powershell
$H3Revision = '48d93ede732756e404a3b1b2f3b3a9b5a22f6cfc'
$CheckpointRoot = '<checkpoint-directory>'
$Bundle = '<bundle-output-path>\MiniMax-H3.bundle'
$env:PATH = @((Join-Path $RtxRoot 'bin'), (Join-Path $RtxRoot 'lib'), `
    (Join-Path $CudaRoot 'bin'), $env:PATH) `
    -join [IO.Path]::PathSeparator

$PythonTag = & python -c `
    "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')"
$RtxWheels = @(Get-ChildItem -LiteralPath (Join-Path $RtxRoot 'python') `
    -Filter "tensorrt_rtx-*-$PythonTag-none-win_amd64.whl" -File)
if ($RtxWheels.Count -ne 1) {
    throw "Expected exactly one TensorRT-RTX wheel for $PythonTag"
}

python -m pip install `
    "torch>=2.0" "safetensors>=0.4" "numpy>=1.24" `
    "ml_dtypes>=0.4" "huggingface_hub>=0.23" `
    "tomli>=2.0; python_version < '3.11'" `
    $RtxWheels[0].FullName
python -m pip install --no-deps -e $RepoRoot -C py-only=true
$Checkpoint = (& python -c `
    "from huggingface_hub import snapshot_download; import sys; print(snapshot_download('MiniMaxAI/MiniMax-H3', revision='$H3Revision', local_dir=sys.argv[1], ignore_patterns=['transformer_ref/*']))" `
    $CheckpointRoot).Trim()
```

For a T2VA and FL2VA bundle, run:

```powershell
python -m tensorrt_model_connect build $Checkpoint `
    --rtx --precision bf16 `
    --output $Bundle
```

To include Ref2VA in a new bundle, add the released `transformer_ref` files to
the same checkpoint directory and run the following build command instead:

```powershell
$Checkpoint = (& python -c `
    "from huggingface_hub import snapshot_download; import sys; print(snapshot_download('MiniMaxAI/MiniMax-H3', revision='$H3Revision', local_dir=sys.argv[1], allow_patterns=['transformer_ref/*']))" `
    $CheckpointRoot).Trim()
$TransformerRef = Join-Path $Checkpoint 'transformer_ref'

python -m tensorrt_model_connect build $Checkpoint `
    --rtx --precision bf16 `
    --output $Bundle `
    --set "minimax_h3.transformer_ref=$TransformerRef"
```

Large plans are written to `$Bundle.plans` as they complete. Repeating the same
command resumes an interrupted build. The largest engines use TensorRT-RTX's
default workspace limit; no H3-specific workspace cap is imposed.

## Generate video

Use the runtime built above:

```powershell
$Trtmc = Join-Path $BuildRoot 'trtmc.exe'
```

T2VA, nominal five seconds:

```powershell
& $Trtmc generate-video $Bundle `
    --prompt 'A cinematic sunrise over a mountain lake with synchronized birds and wind.' `
    --num-frames 120 --height 768 --width 1344 --seed 0 `
    --output .\t2va-5s.mp4
```

T2VA, longest aligned output:

```powershell
& $Trtmc generate-video $Bundle `
    --prompt 'A continuous documentary shot with synchronized dialogue and ambience.' `
    --num-frames 345 --height 768 --width 1344 --seed 0 `
    --output .\t2va-14.375s.mp4
```

FL2VA:

```powershell
& $Trtmc generate-video $Bundle `
    --prompt 'Continue naturally between the supplied endpoints.' `
    --first-frame .\first.png --last-frame .\last.png `
    --num-frames 120 --seed 7 --output .\fl2va.mp4
```

Ref2VA preserves the order of reference flags:

```powershell
& $Trtmc generate-video $Bundle `
    --prompt 'Use <Picture 1> as the subject and <Audio 1> as the voice reference.' `
    --reference-image .\subject.png `
    --reference-audio .\voice.wav `
    --num-frames 120 --height 768 --width 1344 --seed 11 `
    --output .\ref2va.mp4
```

Reference videos and explicit audio references must be 2--15 seconds. Ref2VA
accepts at most 9 images, 3 videos, 3 explicit audio files, and 12 files total.

## Reproduce the five-second performance result

Use the checked-in qualification prompt, retain the five hot engines, and keep
a persistent TensorRT-RTX runtime cache:

```powershell
$Prompt = (Get-Content -Raw `
    (Join-Path $RepoRoot `
        'tests\e2e\models\minimax_h3\prompts\t2va-example-1.json') |
    ConvertFrom-Json).prompt
$RuntimeCache = '.\minimax-h3-dense-fbc.rtxcache'

& $Trtmc generate-video $Bundle `
    --prompt $Prompt `
    --num-frames 120 --height 768 --width 1344 --seed 0 `
    --num-inference-steps 50 --guidance-scale 1 `
    --runtime-cache $RuntimeCache `
    --set "minimax_h3.retain_engines=true" `
    --set "minimax_h3.retained_tail_weight_budget_gib=24" `
    --set "minimax_h3.first_block_cache_threshold=0.30" `
    --warmup 1 --benchmark 1 `
    --output .\minimax-h3-t2va-124f.mp4
```

On the qualified Spark system, the measured request completed in
542,663.046 ms (9:02.663). It used profile `0/2`, ran 49 transformer forwards,
and evaluated the tail 6 times while reusing it 43 times. The output contains
124 frames at 1344x768 and 24 fps plus stereo AAC audio at 32 kHz.

The `0.30` FirstBlockCache threshold is a measured preset, not a universal
default. Lower values recompute more tail steps; higher values can affect
quality. The first and final tail evaluations always run.

`generation_ms` measures the native `generate_video` call, including
conditioning, denoising, both VAEs, and device-to-host copies. It excludes
bundle loading and MP4 encoding, which the CLI reports separately. The
qualification ceiling for this exact five-second workload is 555,000 ms.
