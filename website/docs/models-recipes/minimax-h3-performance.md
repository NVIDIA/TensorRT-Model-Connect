---
title: MiniMax H3 performance and Quick Start
description: Six measured configurations, native CLI commands, and clean Windows setup.
---

September 10, 2026 · [PR #1241](https://github.com/NVIDIA/TensorRT-Model-Connect/pull/1241) · Branch: `codex/minimax-h3-performance`

## 1. Six configurations: performance and commands

**End-to-end minutes, including engine loading and MP4 writing.** Each result is one fresh-process run with a prepared RTX disk cache, not a repeated-prompt hot request. Both FBC thresholds are **0.3**.

| Configuration | 124 frames | 345 frames | Output resolution |
| --- | ---: | ---: | --- |
| Normal T2VA | 9.51 min | 46.99 min | 1344 x 768 |
| Normal FL2VA | 10.03 min | 50.71 min | 1344 x 768 |
| Normal REF2VA | 16.01 min | 82.33 min | 1344 x 768 |
| SR T2VA | 3.63 min | 12.37 min | 1296 x 720 |
| SR FL2VA | 3.66 min | 11.62 min | 1296 x 720 |
| SR REF2VA | 7.87 min | 29.18 min | 1296 x 720 |

All outputs include **32 kHz stereo audio**. At 24 fps, 124 / 345 frames are 5.167 / 14.375 seconds. SR explicitly generates at **864 x 480**, then upscales; normal never applies SR.

After Quick Start below, choose **one** command. Set `$Frames = 120` for the measured 124-frame output, or `$Frames = 345` for 345 frames. The argument arrays only shorten ordinary CLI commands; no wrapper script is required.

**Normal generation**

```powershell
# T2VA
& $Trtmc generate-video @Normal @T2VA @Run --num-frames $Frames
# FL2VA
& $Trtmc generate-video @Normal @FL2VA @Run --num-frames $Frames
# REF2VA
& $Trtmc generate-video @Normal @REF2VA @Run --num-frames $Frames
```

**Super resolution**

```powershell
# T2VA
& $Trtmc generate-video @SR @T2VA @Run --num-frames $Frames
# FL2VA
& $Trtmc generate-video @SR @FL2VA @Run --num-frames $Frames
# REF2VA
& $Trtmc generate-video @SR @REF2VA @Run --num-frames $Frames
```

Run sequentially, not concurrently. Prepare a fresh output/cache directory before each selected command; do not execute these blocks as a batch.

---

## 2. Quick Start

### A. Install ModelConnect

Install **VS 2022 C++ tools**, Git, CMake, Ninja, x64 Python 3.12+, **CUDA 12.9**, **TensorRT-RTX 1.6.1.120** and a compatible NVIDIA driver. Open x64 VS Developer PowerShell, then `pwsh -NoProfile` (**PowerShell 7.4+**). Use the matching RTX SDK DLL and Python wheel.

```powershell
git clone --branch codex/minimax-h3-performance --single-branch `
  https://github.com/yifeif-nv/TensorRT-Model-Connect-fork.git ModelConnect
$RepoRoot = (Resolve-Path './ModelConnect').Path
$CudaRoot = '<CUDA-root>'; $RtxRoot = '<TensorRT-RTX-root>'
$ArtifactRoot = (New-Item -ItemType Directory '<new-artifact-directory>').FullName
$BuildRoot = "$ArtifactRoot/build"; $InstallRoot = "$ArtifactRoot/install"
$env:PATH = "$RtxRoot/bin;$RtxRoot/lib;$CudaRoot/bin;$env:PATH"
$PSNativeCommandUseErrorActionPreference = $true; $ErrorActionPreference = 'Stop'
```

Install the CMake dependency, then build only the native CLI, RTX backend and H3 family. Adjust RTX library/DLL directories if your SDK uses a different layout.

```powershell
$JsonRoot = "$ArtifactRoot/json"; $JsonInstall = "$ArtifactRoot/dependencies"
git clone --depth 1 --branch v3.11.3 https://github.com/nlohmann/json.git $JsonRoot
cmake -S $JsonRoot -B "$JsonRoot/build" -DJSON_BuildTests=OFF `
  "-DCMAKE_INSTALL_PREFIX=$JsonInstall"
cmake --install "$JsonRoot/build"
$Cxx = (Get-Command cl.exe).Source -replace '\\', '/'
cmake -S $RepoRoot -B $BuildRoot -G Ninja -DCMAKE_BUILD_TYPE=Release `
  "-DCMAKE_CXX_COMPILER=$Cxx" "-DCMAKE_CUDA_HOST_COMPILER=$Cxx" `
  "-DCMAKE_CUDA_COMPILER=$CudaRoot/bin/nvcc.exe" "-DCUDAToolkit_ROOT=$CudaRoot" `
  "-DCMAKE_PREFIX_PATH=$JsonInstall" -DCMAKE_CUDA_ARCHITECTURES=native `
  -DCMAKE_CUDA_RUNTIME_LIBRARY=Static -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded `
  -DTRTMC_RUNTIME_MODELS=minimax_h3 -DTRTMC_BUILD_BACKEND_TRT=OFF `
  -DTRTMC_BUILD_BACKEND_RTX=ON "-DTRTMC_RTX_INCLUDE_DIR=$RtxRoot/include" `
  "-DTRTMC_RTX_LIBRARY_DIR=$RtxRoot/lib" "-DTRTMC_RTX_RUNTIME_DIR=$RtxRoot/bin" `
  -DTRTMC_ENABLE_BYOK=OFF -DTRTMC_BUILD_TESTS=OFF -DTRTMC_BUILD_EXAMPLES=OFF
cmake --build $BuildRoot --parallel --target `
  trtmc trtmc_core trtmc_backend_rtx trtmc_model_minimax_h3
cmake --install $BuildRoot --prefix $InstallRoot --config Release
```

Install build-time Python dependencies; generation itself uses native C++/CUDA, TRT-RTX and Windows Media Foundation, without Python.

```powershell
$PythonTag = python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')"
$Wheel = @(Get-ChildItem "$RtxRoot/python/tensorrt_rtx-*-$PythonTag-none-win_amd64.whl")
if ($Wheel.Count -ne 1) { throw 'Select the matching RTX Python wheel' }
python -m pip install $Wheel[0].FullName
python -m pip install -r "$RepoRoot/families/minimax_h3/requirements.txt"
python -m pip install 'torch>=2.6' 'safetensors>=0.4' 'numpy>=1.24' `
  'ml_dtypes>=0.4' 'onnx>=1.16' 'huggingface_hub>=0.23' 'sentencepiece>=0.1.99' `
  'cuda-python>=13.0.3,<14' 'apache-tvm-ffi==0.1.12' 'PyYAML>=6.0'
python -m pip install --no-deps -e $RepoRoot -C py-only=true
```

### B. Download weights and build bundles

Run the [checkpoint-download block](./minimax-h3.md#build-normal-or-super-resolution) to define `$Checkpoint` and download the original configuration/tokenizer/VAEs. **Then use these builds**, not the guide's default-FBC commands. Comfy INT8 denoisers and the NVFP4-AWQ text checkpoint download automatically; SR also downloads its Real-ESRGAN weights.

```powershell
$FBC = @('--set', 'minimax_h3.first_block_cache_threshold=0.3',
         '--set', 'minimax_h3.ref2va_first_block_cache_threshold=0.3')
python -m tensorrt_model_connect build $Checkpoint --backend trt_rtx `
  --precision bf16 --output "$ArtifactRoot/normal.bundle" @FBC
python -m tensorrt_model_connect build $Checkpoint --backend trt_rtx `
  --precision bf16 --output "$ArtifactRoot/sr.bundle" @FBC `
  --set minimax_h3.super_resolution=true
```

Build only the bundle(s) you need; each supports all three modes and occupies about **117.61 GiB**, plus checkpoint/staging space. Text weights are decoded to **BF16 matmul**, not native FP4 compute; denoiser linears use INT8. Vision/VAEs/SR remain floating-point.

---

### C. Set inputs and run

For the recorded benchmark, extract the separately supplied, authorized reproduction package into `$ArtifactRoot`. It supplies `resident-requests.json` and `inputs/first.png`, `last.png`, `audio.wav`. The branch alone does **not** include these media. Your own prompts/media work, but are a different benchmark.

```powershell
$Eval = Get-Content "$ArtifactRoot/resident-requests.json" -Raw | ConvertFrom-Json
$InputRoot = "$ArtifactRoot/inputs"
$T2VA = @('--prompt', $Eval.requests[0].prompt)
$FL2VA = @('--prompt', $Eval.requests[5].prompt,
  '--first-frame', "$InputRoot/first.png", '--last-frame', "$InputRoot/last.png")
$REF2VA = @('--prompt', $Eval.requests[6].prompt,
  '--reference-image', "$InputRoot/first.png", '--reference-audio', "$InputRoot/audio.wav")
$RuntimeRoot = "$InstallRoot/bin"; $Trtmc = "$RuntimeRoot/trtmc.exe"
$Normal = @("$ArtifactRoot/normal.bundle", '--height', '768', '--width', '1344')
$SR = @("$ArtifactRoot/sr.bundle", '--height', '480', '--width', '864')
```

**Before each run**, select the duration and prepare fresh paths:

```powershell
$Frames = 120 # 124 output frames; change to 345 for the long output
$RunRoot = New-Item -ItemType Directory (Join-Path $ArtifactRoot ([guid]::NewGuid().ToString('N')))
$Cache = Join-Path $RunRoot.FullName 'runtime.rtxcache'
$Run = @('--runtime-root', $RuntimeRoot, '--seed', '0',
  '--num-steps', '50', '--guidance-scale', '1', '--runtime-cache', $Cache,
  '--output', (Join-Path $RunRoot.FullName 'video.mp4'))
```

Now run **one command from section 1**. Find `video.mp4` in `$RunRoot.FullName`. For another mode or duration, repeat the fresh-path block first.

### D. Reproduce the timing conditions

Set the aliases below, then run the [three-mode seed procedure](https://github.com/yifeif-nv/TensorRT-Model-Connect-fork/blob/f030c5107bdd8e36298dde184681ca146c0de418/website/docs/models-recipes/minimax-h3-performance.md#optional-deterministic-request-order-for-creating-local-seeds) once to prepare a compatible cache per bundle. Preparation is not timed; historical seeds are not distributed.

```powershell
$NormalBundle = $Normal[0]; $SrBundle = $SR[0]
$T2vaPrompt = $T2VA[1]; $Fl2vaPrompt = $FL2VA[1]; $Ref2vaPrompt = $REF2VA[1]
$SeedRoot = "$ArtifactRoot/cache-seeds"
$OutputRoot = (New-Item -ItemType Directory -Force "$ArtifactRoot/outputs").FullName
```

Create fresh paths in C, then copy the matching seed **before** timing. Example for normal T2VA; change the seed and command together for other configurations:

```powershell
Copy-Item "$SeedRoot/normal.rtxcache" $Cache
$Timer = [Diagnostics.Stopwatch]::StartNew()
& $Trtmc generate-video @Normal @T2VA @Run --num-frames $Frames
$Timer.Stop()
if ($LASTEXITCODE) { throw 'Generation failed; do not report a successful timing' }
$Timer.Elapsed.TotalSeconds
```

Use a new process and fresh seed copy for each measurement. Do not add a full-generation warmup or require a repeated prompt. Keep seed 0, 50 schedule points, guidance 1, FBC 0.3, the exact prompts/media and recorded dimensions. Never time profiler runs as ordinary E2E.

Recorded runtime: `aa594f6010bd59204c2edc3061fc6e5042572c0d`; later commits are documentation-only. Each table cell is **n=1**. Build/download/cache preparation/QA are excluded; OS/driver caches were not cleared and clocks were not locked. Different hardware, rebuilt engines or local cache seeds can change latency.

**Scope:** dynamic prompts and the 124–345-frame grid remain supported; SR requires the fixed 864 x 480 base. All twelve outputs passed full media decode and sampled visual checks, not full-motion/listening qualification. FBC 0.3 is approximate (default: 0.08); detail/framing can differ, and SR is not pixel-equivalent to native output.

[C++ API example](./minimax-h3.md#c-api) · [Input limits](./minimax-h3.md#capabilities-and-inputs) · [Detailed measurement evidence](https://github.com/yifeif-nv/TensorRT-Model-Connect-fork/blob/f030c5107bdd8e36298dde184681ca146c0de418/website/docs/models-recipes/minimax-h3-performance.md)
