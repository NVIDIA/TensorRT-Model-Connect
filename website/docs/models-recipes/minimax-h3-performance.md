---
title: MiniMax H3 performance reproduction
description: Reproduce native T2VA, FL2VA, and REF2VA with dynamic inputs, explicit super resolution, and defined cache conditions.
---

Delivery snapshot: **12 of 12 primary configurations are complete** on the current request-memory candidate: all six short cases and all six long cases. The nine-request resident sequence and both short T2VA empty-cache checks are also complete. All primary outputs passed full media checks and scoped sampled scene inspection. No full-motion or listening pass is claimed. Earlier runtime variants are retained separately. This snapshot does not establish comprehensive performance or quality qualification.

## Delivery and comparison scope

- Baseline source: PR #1240, commit `19e37595033d802c7dedfed77f2a8f42b1d4a992`.
- Published PR state: `codex/minimax-h3-performance`, runtime commit `aa594f6010bd59204c2edc3061fc6e5042572c0d`; draft PR [#1241](https://github.com/NVIDIA/TensorRT-Model-Connect/pull/1241). Runtime-source publication was verified on September 10, 2026. Documentation-only follow-up commits do not change the measured runtime revision. The baseline PR is unchanged, and this optimization PR remains Draft pending review.
- Current measurement candidate: request-sized family-cache commit `aa594f6010bd59204c2edc3061fc6e5042572c0d`, built in a separate immutable measurement worktree. It has passed 14 CPU tests and completed all twelve primary GPU cases with full media decode; scoped visual findings are recorded below. The single-process nine-request resident sequence and both short T2VA empty-cache checks also completed with full media checks and sampled visual inspection. Completion of this selected matrix is not comprehensive qualification. Its results use the distinct `memory-` prefix; `bulk-` results remain measurements of the preceding `4dcfdc79` commit and are not relabelled.
- One normal bundle and one explicitly enabled SR bundle each contain T2VA, FL2VA, and REF2VA. There is no per-mode build, fixed-prompt selection, or user-selected optimization profile.
- Baseline and candidate comparisons use the same engine payloads, tokenizer, request, seed, canvas, and both FirstBlockCache thresholds set to **0.3**. Only the native runtime changes. Do not independently rebuild engines for one side of a runtime-only comparison.
- Candidate changes concern shared text/vision activation memory, actual-shape input/output allocation, activation-shape invalidation, Windows bulk plan-file reads, and request-sized family cache tensors. They do not change checkpoints, precision, attention mathematics, denoising schedule, dynamic profile ranges, weight-streaming budgets, or FBC thresholds.
- The twelve primary configurations are normal/SR × three modes × 124/345 output frames. All three normal modes use native **1344×768**, including FL2VA with its original 1344×768 endpoint fixtures. All three SR modes generate at **864×480 → 1296×720**. Every result includes audio. Video-track durations are 5.167 and 14.375 seconds at 24 fps, not strict 5 and 15 seconds; audio/container duration can include a small codec/sample-grid tail.
- Native 864×480 runs are optional, separately labelled SR-disabled controls. They are not substitutes for the normal 1344×768 FL2VA rows.

The branch's `website/docs/models-recipes/minimax-h3.md` is the public model guide. The commands below repeat the essential setup so the report can be used without any workstation-specific scripts.

### Runtime changes under evaluation

| Change | Intended mechanism | Evidence and remaining limits |
| --- | --- | --- |
| Live-shape text/vision activation storage | Extend the existing serial-context allocation policy to text/vision, sizing activation requirements from actual inputs. Coexisting serial contexts can share the arena; this does not promise persistence across destroyed modules. | Matched full-workflow timing and shape/mode transition checks. |
| Actual-shape input/output allocation | Allocate and initialize the requested tensor shapes while retaining the public dynamic ranges; invalidate cached activation requirements when shapes change. | Nine short resident requests with prompt/mode/reference transitions and all six one-shot long cases completed; scoped quality findings are below. |
| Request-sized family FirstBlockCache tensors (current candidate) | Allocate four BF16 hidden/cache tensors at validated live row counts; late-bind them while preserving profile-MAX ABI and initial-binding capacity validation. Auxiliary bindings and release policies remain unchanged. | Separate build and 14 CPU tests passed. All twelve primary cases and the nine-request resident sequence completed with automatic media checks; scoped visual findings and limitations are below. |
| Windows bulk plan reads | Replace the plan reader's MSVC `ifstream` path with checked `SetFilePointerEx` and `ReadFile`, reusing the existing read-only, write/delete-protected file handle. Host reads use chunks no larger than 64 MiB; device destinations keep the existing 4 MiB staging/upload path. Linux remains on `ifstream`. | All six short and six current-candidate long results are listed below; most matched original-PR timing pairs were not measured. |
| Load diagnostics | Record file-read bytes/wall time, upload wall time, deserialization, weight-budget setup, runtime cache/configuration, context creation, and module initialization. | Retained as generic runtime logging; nested durations are not additive. |

The inspected public MSVC 14.44 standard-library source, `__msvc_filebuf.hpp` (`xsgetn`), services this large `ifstream.read` path through approximately 4 KiB `fread` iterations, including binary reads. This identifies avoidable per-read overhead; the source observation alone does not establish a speedup or prove that physical storage is slow. Measured results are listed below. The bulk reader retains section bounds, seek/short-read failure handling, file identity, and the existing mutation guard; it does not alter engine bytes or model computation.

Pre-bulk runs, bulk-reader runs and profiler traces are retained as separately labelled preceding variants or diagnostics. Do not enter them as current request-memory candidate results. Deserialization includes reader callbacks; callbacks include file reads/uploads; runtime-configuration time includes runtime-cache time. These nested durations must not be added together. File-read wall time can include operating-system caching, page faults, and host copying; upload-call wall time is not DMA-only time.

The native 1344×768 long REF2VA request has 118,793 packed rows, whereas its broad dynamic profile allows 630,310. Source-level accounting shows that the four family cache tensors reserve 25.247 GiB at profile MAX but require 4.758 GiB for this request: 20.488 GiB of avoidable allocation. This is not a measured speedup. Read-only memory samples did not establish sustained hard-disk paging; high page-fault rates include soft faults. The old `bulk-normal-ref2va-345` diagnostic also overlapped CPU compilation of the separate candidate. Its observed duration will be preserved, but it is not a controlled formal comparison.

The twelve completed primary request-memory cases logged the following aggregate capacities for the four family cache tensors. These are request allocation sizes versus the preserved profile-MAX contract, not whole-process GPU-memory peaks or proof of a corresponding latency reduction.

| Current-candidate case | Requested rows | Profile-MAX rows | Allocated cache bytes | Profile-MAX cache bytes |
| --- | ---: | ---: | ---: | ---: |
| Normal T2VA, 124 frames | 37,804 | 112,367 | 1,625,874,432 | 4,832,679,936 |
| Normal FL2VA, 124 frames | 41,870 | 112,367 | 1,800,744,960 | 4,832,679,936 |
| Normal REF2VA, 124 frames | 52,535 | 86,662 | 2,259,425,280 | 3,727,159,296 |
| SR T2VA, 124 frames | 15,493 | 112,367 | 666,322,944 | 4,832,679,936 |
| SR FL2VA, 124 frames | 17,147 | 112,367 | 737,458,176 | 4,832,679,936 |
| SR REF2VA, 124 frames | 30,224 | 86,662 | 1,299,873,792 | 3,727,159,296 |
| Normal T2VA, 345 frames | 104,060 | 112,367 | 4,475,412,480 | 4,832,679,936 |
| Normal FL2VA, 345 frames | 108,126 | 112,367 | 4,650,283,008 | 4,832,679,936 |
| Normal REF2VA, 345 frames | 118,793 | 630,310 | 5,109,049,344 | 27,108,372,480 |
| SR T2VA, 345 frames | 42,554 | 112,367 | 1,830,162,432 | 4,832,679,936 |
| SR FL2VA, 345 frames | 44,208 | 112,367 | 1,901,297,664 | 4,832,679,936 |
| SR REF2VA, 345 frames | 57,287 | 630,310 | 2,463,799,296 | 27,108,372,480 |

## Software and runtime conditions

Use Windows with **PowerShell 7.2 or newer** and the x64 Visual Studio 2022 developer environment, Desktop development with C++, Git, CMake, Ninja, a supported NVIDIA driver, CUDA, and the matching TensorRT-RTX SDK. Do not run these commands in Windows PowerShell 5.1: its native stderr pipeline handling can stop a successful generation when ordinary diagnostic messages are logged. Runtime generation and media decoding/encoding are native C++ and Windows Media Foundation; Python, PyTorch, FFmpeg, and ComfyUI are not runtime dependencies.

Open the **x64 Visual Studio 2022 Developer PowerShell**, then run `pwsh -NoProfile` inside it and use that child shell for the commands below. The child inherits the compiler's `PATH`, `INCLUDE` and `LIB` environment. Opening an unrelated plain PowerShell 7 window does not initialize the Visual Studio toolchain.

| Component | Recorded validation environment |
| --- | --- |
| TensorRT-RTX runtime and Python wheel | [1.6.1.120](https://pypi.org/project/tensorrt-rtx/1.6.1.120/), matching `win_amd64` wheel |
| CUDA toolkit/compiler | CUDA 12.9; compiler 12.9.86 |
| C++ compiler | MSVC 19.44.35228.0; toolset directory 14.44.35207 |
| Native build | Release, Ninja, x64, static MSVC runtime (`/MT`), static CUDA runtime |
| CMake / Ninja | 3.31.6-msvc6 / 1.12.1 |
| Build-only Python | [CPython 3.13.5](https://www.python.org/downloads/release/python-3135/) |
| Observed build-only packages | [torch 2.12.0+cu130](https://download.pytorch.org/whl/cu130/torch/); [numpy 2.5.2](https://pypi.org/project/numpy/2.5.2/); [safetensors 0.8.0](https://pypi.org/project/safetensors/0.8.0/); [huggingface_hub 1.29.0](https://pypi.org/project/huggingface-hub/1.29.0/) |
| Diagnostic profiler | [Nsight Systems 2026.5.1](https://developer.nvidia.com/nsight-systems/get-started); diagnostic runs only |
| Host hardware and driver | Not publicly specified; unapproved host/driver details remain in the private environment record. |

The observed Python package versions are an environment record, not a claim that every version is mandatory. The public guide supports CPython 3.12 or newer with a matching RTX wheel. Allow substantial disk space for source checkpoints, staged plans, and final bundles. Do not delete source files or old artifacts automatically to make a build fit.

For scale, the measured normal and SR bundles are 126,279,130,165 and 126,283,045,708 bytes respectively: approximately 117.61 GiB each, or 235.22 GiB for both finished files alone. Checkpoints, staging, dependencies and outputs require additional space; 235.22 GiB is not a sufficient free-space requirement for building both from scratch. Bundle byte counts describe these artifacts, not binary equivalence after rebuilding on another system.

The commands reproduce the workflow and comparison protocol, not a promise of the recorded latency on arbitrary NVIDIA GPUs. Equivalent compute, usable memory, host execution, storage throughput, SDK/driver behavior and system load matter. The observed resource usage is not a minimum-hardware qualification. Rebuilding plans on another system can select different tactics; use one locally built bundle per delivery for both runtimes when evaluating a runtime-only change. Exact historical engine/cache equivalence cannot be claimed from matching checkpoint names alone.

### 1. Configure native dependencies and build the runtime

The tested runtime source is `aa594f6010bd59204c2edc3061fc6e5042572c0d`, published in PR #1241. This clean-checkout recipe deliberately refuses to proceed unless the tested commit is included in the fetched PR history. A later report/documentation commit is distinct from this measured runtime-source commit and must not silently replace it in the result provenance. Published source availability does not imply completion of the remaining performance or quality checks.

```powershell
$ErrorActionPreference = 'Stop'
if ($PSVersionTable.PSEdition -ne 'Core' -or $PSVersionTable.PSVersion -lt [version]'7.2') {
    throw 'Use PowerShell 7.2+ (pwsh), not Windows PowerShell 5.1, with the x64 VS developer environment'
}
$CheckoutRoot = '<ModelConnect-checkout>' # Choose a new, absent directory.
$TestedRuntimeCommit = 'aa594f6010bd59204c2edc3061fc6e5042572c0d'
if (Test-Path -LiteralPath $CheckoutRoot) { throw 'Choose a new directory for the clean checkout' }
git clone --no-checkout -o github https://github.com/NVIDIA/TensorRT-Model-Connect.git $CheckoutRoot
if ($LASTEXITCODE -ne 0) { throw 'Source clone failed' }
git -C $CheckoutRoot fetch github pull/1241/head:refs/remotes/github/pr-1241
if ($LASTEXITCODE -ne 0) { throw 'PR fetch failed' }
git -C $CheckoutRoot merge-base --is-ancestor $TestedRuntimeCommit refs/remotes/github/pr-1241
if ($LASTEXITCODE -ne 0) { throw 'The exact tested runtime commit is not yet available in this PR' }
git -C $CheckoutRoot switch --detach $TestedRuntimeCommit
if ($LASTEXITCODE -ne 0) { throw 'Could not check out the tested runtime commit' }
$RepoRoot = (Resolve-Path -LiteralPath $CheckoutRoot).Path
```

Continue in the same session, setting these dependency/artifact paths to directories on the target machine. These commands do not push or modify the upstream branches.

```powershell
$ErrorActionPreference = 'Stop'
$CudaRoot = '<CUDA-root>'
$RtxRoot = '<TensorRT-RTX-root>'
$ArtifactRoot = (New-Item -ItemType Directory -Force '<artifact-directory>').FullName
$BuildRoot = Join-Path $ArtifactRoot 'build'
$InstallRoot = Join-Path $ArtifactRoot 'install'
$OutputRoot = (New-Item -ItemType Directory -Force (Join-Path $ArtifactRoot 'outputs')).FullName
$InputRoot = Join-Path $ArtifactRoot 'inputs'
$SeedRoot = Join-Path $ArtifactRoot 'cache-seeds'
$env:PATH = @("$RtxRoot/bin", "$RtxRoot/lib", "$CudaRoot/bin", $env:PATH) -join [IO.Path]::PathSeparator

$JsonRoot = Join-Path $ArtifactRoot 'json'
$JsonInstall = Join-Path $ArtifactRoot 'dependencies'
git clone --depth 1 --branch v3.11.3 https://github.com/nlohmann/json.git $JsonRoot
cmake -S $JsonRoot -B "$JsonRoot/build" -DJSON_BuildTests=OFF "-DCMAKE_INSTALL_PREFIX=$JsonInstall"
cmake --install "$JsonRoot/build"

$Cxx = (Get-Command cl.exe).Source -replace '\\', '/'
cmake -S $RepoRoot -B $BuildRoot -G Ninja `
    -DCMAKE_BUILD_TYPE=Release "-DCMAKE_CXX_COMPILER=$Cxx" `
    "-DCMAKE_CUDA_COMPILER=$CudaRoot/bin/nvcc.exe" "-DCMAKE_CUDA_HOST_COMPILER=$Cxx" `
    "-DCUDAToolkit_ROOT=$CudaRoot" "-DCMAKE_PREFIX_PATH=$JsonInstall" `
    -DCMAKE_CUDA_ARCHITECTURES=native -DCMAKE_CUDA_RUNTIME_LIBRARY=Static `
    -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded -DTRTMC_RUNTIME_MODELS=minimax_h3 `
    -DTRTMC_BUILD_BACKEND_TRT=OFF -DTRTMC_BUILD_BACKEND_RTX=ON `
    "-DTRTMC_RTX_INCLUDE_DIR=$RtxRoot/include" "-DTRTMC_RTX_LIBRARY_DIR=$RtxRoot/lib" `
    "-DTRTMC_RTX_RUNTIME_DIR=$RtxRoot/bin" `
    -DTRTMC_ENABLE_BYOK=OFF -DTRTMC_BUILD_TESTS=OFF -DTRTMC_BUILD_EXAMPLES=OFF
cmake --build $BuildRoot --parallel --target trtmc trtmc_core trtmc_backend_rtx trtmc_model_minimax_h3
cmake --install $BuildRoot --prefix $InstallRoot --config Release
$RuntimeRoot = Join-Path $InstallRoot 'bin'
$Trtmc = Join-Path $RuntimeRoot 'trtmc.exe'
$env:PATH = @($RuntimeRoot, $env:PATH) -join [IO.Path]::PathSeparator
```

If the SDK places its import library in `lib/x64` or its runtime DLL in `lib`, adjust those directory arguments. Use separate build/install directories for baseline and candidate, keeping compiler flags and SDK versions identical.

### 2. Install build-only dependencies and fetch the original VAEs/configuration

Run these commands in a dedicated Python environment:

```powershell
python -m venv (Join-Path $ArtifactRoot 'builder-env')
& (Join-Path $ArtifactRoot 'builder-env/Scripts/Activate.ps1')
$PythonTag = python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')"
$RtxWheels = @(Get-ChildItem -LiteralPath (Join-Path $RtxRoot 'python') `
    -Filter "tensorrt_rtx-*-$PythonTag-none-win_amd64.whl" -File)
if ($RtxWheels.Count -ne 1) { throw "Expected exactly one matching TensorRT-RTX wheel" }
python -m pip install $RtxWheels[0].FullName
python -m pip install -r (Join-Path $RepoRoot 'families/minimax_h3/requirements.txt')
python -m pip install 'torch>=2.6' 'safetensors>=0.4' 'numpy>=1.24' 'ml_dtypes>=0.4' `
    'onnx>=1.16' 'huggingface_hub>=0.23' 'sentencepiece>=0.1.99' `
    'cuda-python>=13.0.3,<14' 'apache-tvm-ffi==0.1.12' 'PyYAML>=6.0'
python -m pip install --no-deps -e $RepoRoot -C py-only=true

$Checkpoint = Join-Path $ArtifactRoot 'checkpoint'
$Patterns = @('model_index.json', 'modular_model_index.json', 'processor/**', `
    'scheduler/**', 'audio_scheduler/**', 'tokenizer/**', `
    'transformer/config.json', 'vae/**', 'audio_vae/**')
python -c "from huggingface_hub import snapshot_download; import sys; snapshot_download('MiniMaxAI/MiniMax-H3', revision='48d93ede732756e404a3b1b2f3b3a9b5a22f6cfc', local_dir=sys.argv[1], allow_patterns=sys.argv[2:])" $Checkpoint $Patterns
```

Use credentials accepted by Hugging Face if the repository requires authentication. Original BF16 denoiser and text-encoder weights are not needed. Model weights remain subject to their respective licenses.

### 3. Build both delivery options with both FBC thresholds at 0.3

```powershell
$NormalBundle = Join-Path $ArtifactRoot 'MiniMax-H3-FBC03.bundle'
$SrBundle = Join-Path $ArtifactRoot 'MiniMax-H3-SR-FBC03.bundle'

python -m tensorrt_model_connect build $Checkpoint `
    --backend trt_rtx --precision bf16 --output $NormalBundle `
    --set minimax_h3.first_block_cache_threshold=0.3 `
    --set minimax_h3.ref2va_first_block_cache_threshold=0.3

python -m tensorrt_model_connect build $Checkpoint `
    --backend trt_rtx --precision bf16 --output $SrBundle `
    --set minimax_h3.super_resolution=true `
    --set minimax_h3.first_block_cache_threshold=0.3 `
    --set minimax_h3.ref2va_first_block_cache_threshold=0.3
```

These are build options, not runtime CLI flags. Product defaults remain 0.08. FBC is approximate; this report compares runtime implementations at the same 0.3 setting, not 0.3 against an uncached accuracy reference. Do not edit bundle metadata manually as a deployment workflow.

Missing full checkpoints are automatically downloaded from `Comfy-Org/MiniMax-H3`, pinned to `4cc1d817b6184899b41293954329f576cb5ae86b`:

| Purpose | Repository-relative checkpoint |
| --- | --- |
| T2VA / FL2VA denoiser | `diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors` |
| REF2VA denoiser | `diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors` |
| Shared text / vision encoder | `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` |

The pruned files are not used. INT8 denoiser linear layers use native TRT quantization operations. The NVFP4 AWQ file specifies full-precision matrix multiplication: its weights are decoded to BF16 at build time and its AWQ input scales are preserved. Thus `--precision bf16` does not select original BF16 checkpoint sources and does not promise native FP4 matrix multiplication. No ComfyUI runtime is involved.

SR additionally downloads the fixed Real-ESRGAN compact checkpoint pair and builds its native TRT plan. SR is enabled only by the explicit build option: base generation is 864×480 and final output is 1296×720, with audio and frame count preserved. Normal generation at 864×480 remains 864×480. A different generation canvas can change composition; native 768p and 480p+SR are not expected to be pixel-matched.

The SR source files are `realesr-general-x4v3.pth` and `realesr-general-wdn-x4v3.pth` from the Real-ESRGAN `v0.2.5.0` release. Preserve the branch's family-owned SR defaults; no runtime SR strength or alternate checkpoint is selected by the commands above.

For offline builds, the public guide documents local checkpoint-location overrides. Use a fresh staging/output path after changing weight sources or profiles. Finished bundles contain their runtime weights and need no external checkpoint directory.

## Exact evaluation inputs

The unchanged fixtures have been prepared in a separately supplied local reproduction package for authorized recipients, under its `inputs` directory. These fixtures are not approved for public redistribution, and no public fixture download is promised. Exact fixture reproduction requires the original files to be supplied separately with appropriate permission. Extract that authorized package into `$ArtifactRoot`, so `$InputRoot` resolves to the supplied files. Filenames alone are not sufficient for exact reproduction, and no replacement images or audio should be silently substituted.

| File | Recorded properties | Uses |
| --- | --- | --- |
| `first.png` | RGB, 1344×768, 1,330,627 bytes | FL2VA first frame; REF2VA `<Picture 1>` |
| `last.png` | RGB, 1344×768, 1,234,864 bytes | FL2VA last frame |
| `audio.wav` | Stereo PCM16, 32 kHz, 165888 samples/channel, 5.184 s, 663,630 bytes | REF2VA `<Audio 1>` |
| `reference-square.png` | RGB, 768×768, 476,255 bytes | Additional resident vision-state transition only; not a primary-matrix input |

Byte lengths and decoded properties describe the fixtures; they are not a substitute for distributing the original files. Normal and SR use identical prompts and reference files. Seed is 0 throughout. Preserve the line breaks in the following UTF-8 prompt strings:

```powershell
$T2vaPrompt = @'
integrated_multimodal_description: A single continuous cinematic shot of a small waterfall flowing over dark moss-covered rocks in a lush green forest. Clear water ripples in a shallow pool in the foreground. Ferns move gently in the breeze. The camera slowly pushes forward, with stable natural daylight and realistic water motion. No cuts or scene changes.
overall_soundscape: Continuous flowing water, gentle splashes, and distant birdsong.
non_diegetic_music: None.
'@
$Fl2vaPrompt = @'
integrated_multimodal_description: Continue this forest waterfall scene in one uninterrupted slow camera push. Keep the same moss-covered rock formations, waterfall placement, pool, foliage, and natural daylight throughout. Flowing water and small ripples move continuously while the ferns sway gently in the breeze. Preserve the composition of the supplied first and last frames. No cuts, no new objects, and no abrupt changes of lighting.
overall_soundscape: Soft steady waterfall, rippling water, a light breeze, and occasional distant birds.
non_diegetic_music: None.
'@
$Ref2vaPrompt = @'
integrated_multimodal_description: A continuous slow camera push toward the waterfall in <Picture 1>, retaining its rocks, ferns, and daylight. Water flows naturally, without cuts.
overall_soundscape: Flowing water and birdsong matching <Audio 1>.
non_diegetic_music: None.
'@
# Match the LF-separated receipt strings on Windows as well.
$T2vaPrompt = $T2vaPrompt.Replace("`r`n", "`n")
$Fl2vaPrompt = $Fl2vaPrompt.Replace("`r`n", "`n")
$Ref2vaPrompt = $Ref2vaPrompt.Replace("`r`n", "`n")
```

## Prepare the runtime-cache condition explicitly

These are independent forms of reuse; do not call all of them "warm":

| State | What is reused | How this report controls it |
| --- | --- | --- |
| Build/download caches | Checkpoint files and engine-build artifacts | Outside generation timing; engine payloads are held fixed within each runtime pair. |
| RTX disk runtime cache | Serialized runtime specialization data | A new absent path for `empty`, or an independent copy of one frozen seed for `seeded`. |
| In-process state | Loaded task, retained engines and any family conditioning caches | Only a resident sequence reuses a live task; actual cache hits must be read from logs. |
| FirstBlockCache | Approximate denoiser-tail reuse within generation | Both bundle thresholds stay at 0.3, independently of disk-cache condition. |
| OS/driver caches | File-system cached plan pages and driver-managed state | Not purged or assumed empty; execution order is recorded. |

`empty` below means **no RTX runtime-cache file exists at the specified path before launch**. The runtime may populate and save that path on successful task finalization. Reusing it for the next process changes the condition to seeded. Omitting `--runtime-cache` is not the empty-cache experiment: always pass the dedicated path. Neither a new filename nor a fresh process resets OS file caching or all driver-managed caches.

The current reproduction package contains fixtures/manifests, not the historical RTX seed files. Existing recorded seeded measurements used frozen copies of the original normal/SR delivery caches. Their full prior priming history is not reconstructed here. Until approved historical seeds and compatibility details are supplied, a new machine can reproduce a **controlled locally seeded comparison**, not the exact original cache contents or absolute timing.

### Optional deterministic request order for creating local seeds

If approved compatible seeds are provided, place them at `$SeedRoot/normal.rtxcache` and `$SeedRoot/sr.rtxcache` and skip this section. Otherwise the following explicit recipe, `three-short-requests-v1`, primes T2VA, FL2VA, then REF2VA once per delivery, using the exact prompts/fixtures above. Use one fixed runtime installation to create both seeds (the baseline runtime when comparing baseline/candidate), and record its source commit. These are **untimed preparation runs**, not primary results or proof that all long/prompt shapes are cached. Do not silently change the priming order or mix a partially completed priming run into the seed.

```powershell
$PrimeRuntimeRoot = $RuntimeRoot # Select and record ONE seed-creation runtime.
$PrimeTrtmc = Join-Path $PrimeRuntimeRoot 'trtmc.exe'
[void](New-Item -ItemType Directory -Force $SeedRoot)
foreach ($PrimeDelivery in @('normal', 'sr')) {
    $FrozenSeed = Join-Path $SeedRoot "$PrimeDelivery.rtxcache"
    if (Test-Path -LiteralPath $FrozenSeed) { throw 'Refusing to replace an existing seed' }
    $PrimeRoot = (New-Item -ItemType Directory (Join-Path $OutputRoot "prime-$PrimeDelivery")).FullName
    $PrimeCache = Join-Path $PrimeRoot 'working.rtxcache' # Initially absent.
    $PrimeBundle = if ($PrimeDelivery -eq 'sr') { $SrBundle } else { $NormalBundle }
    $PrimeHeight = if ($PrimeDelivery -eq 'sr') { 480 } else { 768 }
    $PrimeWidth = if ($PrimeDelivery -eq 'sr') { 864 } else { 1344 }
    foreach ($PrimeMode in @('t2va', 'fl2va', 'ref2va')) {
        $PrimePrompt = switch ($PrimeMode) {
            't2va' { $T2vaPrompt }; 'fl2va' { $Fl2vaPrompt }; 'ref2va' { $Ref2vaPrompt }
        }
        $PrimeOutput = Join-Path $PrimeRoot "$PrimeMode.mp4"
        $PrimeArgs = @('generate-video', $PrimeBundle, '--runtime-root', $PrimeRuntimeRoot,
            '--prompt', $PrimePrompt, '--num-frames', '120', '--seed', '0',
            '--num-steps', '50', '--guidance-scale', '1',
            '--height', "$PrimeHeight", '--width', "$PrimeWidth",
            '--runtime-cache', $PrimeCache, '--output', $PrimeOutput)
        if ($PrimeMode -eq 'fl2va') {
            $PrimeArgs += @('--first-frame', (Join-Path $InputRoot 'first.png'),
                            '--last-frame', (Join-Path $InputRoot 'last.png'))
        }
        if ($PrimeMode -eq 'ref2va') {
            $PrimeArgs += @('--reference-image', (Join-Path $InputRoot 'first.png'),
                            '--reference-audio', (Join-Path $InputRoot 'audio.wav'))
        }
        & $PrimeTrtmc @PrimeArgs 2>&1 | Tee-Object -FilePath (Join-Path $PrimeRoot "$PrimeMode.log")
        if ($LASTEXITCODE -ne 0 -or !(Test-Path -LiteralPath $PrimeOutput)) {
            throw "Seed preparation failed for $PrimeDelivery/$PrimeMode"
        }
    }
    if (!(Test-Path -LiteralPath $PrimeCache) -or (Get-Item -LiteralPath $PrimeCache).Length -eq 0) {
        throw 'No completed runtime-cache seed was produced'
    }
    Copy-Item -LiteralPath $PrimeCache -Destination $FrozenSeed
}
```

Never pass `$FrozenSeed` itself to a measured generation: each process receives its own copy and may update that copy. A seed must be compatible with the target plans and TRT-RTX environment; do not assume portability across rebuilds or machines. If a seed is rejected, retain that failure and create a separately labelled local seed; do not silently substitute an empty cache.

## Native CLI: run any of the twelve cases

Define this helper once in the same PowerShell session as the setup variables and prompts above, then run the explicit case commands below. It only assembles arguments for the native `trtmc generate-video` command; no additional runtime script, Python dependency, or product workflow is required. The short measurement requests `120`, which produces 124 output frames; the long measurement requests `345`.

```powershell
$RuntimeLabel = 'candidate' # Must describe the installation selected by $RuntimeRoot/$Trtmc.
function Invoke-H3Case {
    param(
        [Parameter(Mandatory)][ValidateSet('normal', 'sr')][string]$Delivery,
        [Parameter(Mandatory)][ValidateSet('t2va', 'fl2va', 'ref2va')][string]$Mode,
        [Parameter(Mandatory)][ValidateSet(120, 345)][int]$Frames,
        [ValidateSet('seeded', 'empty')][string]$CacheCondition = 'seeded',
        [switch]$Profile,
        [switch]$Native480pControl # Optional extra control, outside the twelve-case matrix.
    )
    if ($Native480pControl -and $Delivery -ne 'normal') {
        throw 'The native 480p control requires the normal (SR-disabled) bundle'
    }
    if ($Profile) { [void](Get-Command nsys -CommandType Application -ErrorAction Stop) }
    $CanvasLabel = if ($Native480pControl) { 'normal-480p-control' } else { $Delivery }
    $RunName = "$RuntimeLabel-$CanvasLabel-$Mode-$Frames-$CacheCondition"
    if ($Profile) { $RunName += '-profiled' }
    $RunRoot = Join-Path $OutputRoot $RunName
    if (Test-Path -LiteralPath $RunRoot) { throw 'Use a new result name; existing outputs must not be overwritten' }
    [void](New-Item -ItemType Directory $RunRoot -ErrorAction Stop)
    $Bundle = if ($Delivery -eq 'sr') { $SrBundle } else { $NormalBundle }
    $Prompt = switch ($Mode) { 't2va' { $T2vaPrompt }; 'fl2va' { $Fl2vaPrompt }; 'ref2va' { $Ref2vaPrompt } }
    $RuntimeCache = Join-Path $RunRoot 'runtime.rtxcache'
    $VideoOutput = Join-Path $RunRoot 'video.mp4'

    if ($CacheCondition -eq 'seeded') {
        $SeedCache = Join-Path $SeedRoot "$Delivery.rtxcache"
        if (!(Test-Path -LiteralPath $SeedCache -PathType Leaf) -or (Get-Item -LiteralPath $SeedCache).Length -eq 0) {
            throw 'A nonempty compatible frozen seed is required; prepare or obtain it first'
        }
        Copy-Item -LiteralPath $SeedCache -Destination $RuntimeCache -ErrorAction Stop
    } elseif (Test-Path -LiteralPath $RuntimeCache) {
        throw 'The empty-cache experiment requires a new absent cache path'
    }

    $Height = if ($Delivery -eq 'sr' -or $Native480pControl) { 480 } else { 768 }
    $Width = if ($Delivery -eq 'sr' -or $Native480pControl) { 864 } else { 1344 }
    $CliArgs = @('generate-video', $Bundle, '--runtime-root', $RuntimeRoot,
        '--prompt', $Prompt, '--num-frames', "$Frames", '--seed', '0',
        '--num-steps', '50', '--guidance-scale', '1',
        '--runtime-cache', $RuntimeCache, '--output', $VideoOutput,
        '--height', "$Height", '--width', "$Width")
    if ($Mode -eq 'fl2va') {
        $CliArgs += @('--first-frame', (Join-Path $InputRoot 'first.png'),
                     '--last-frame', (Join-Path $InputRoot 'last.png'))
    }
    if ($Mode -eq 'ref2va') {
        $CliArgs += @('--reference-image', (Join-Path $InputRoot 'first.png'),
                     '--reference-audio', (Join-Path $InputRoot 'audio.wav'))
    }
    $LogPath = Join-Path $RunRoot 'generation.log'
    if ($Profile) {
        $TracePrefix = Join-Path $RunRoot 'trace'
        $Timer = [Diagnostics.Stopwatch]::StartNew()
        & nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none `
            --discard-environment=true --cuda-memory-usage=true -o $TracePrefix $Trtmc @CliArgs `
            2>&1 | Tee-Object -FilePath $LogPath
    } else {
        $Timer = [Diagnostics.Stopwatch]::StartNew()
        & $Trtmc @CliArgs 2>&1 | Tee-Object -FilePath $LogPath
    }
    $RunExitCode = $LASTEXITCODE
    $Timer.Stop()
    if ($RunExitCode -ne 0 -or !(Test-Path -LiteralPath $VideoOutput -PathType Leaf)) {
        throw "Generation failed or output is missing: exit $RunExitCode"
    }
    if ($Profile) {
        "Diagnostic enclosing profile wall seconds (NOT a formal result): $($Timer.Elapsed.TotalSeconds)"
        & nsys stats --report cuda_api_sum,cuda_gpu_mem_size_sum,cuda_gpu_mem_time_sum,cuda_gpu_kern_sum `
            --format csv --output (Join-Path $RunRoot 'summary') "$TracePrefix.nsys-rep"
        if ($LASTEXITCODE -ne 0) { throw 'Generation completed, but Nsight statistics export failed' }
    } else {
        "Observed launch-to-exit wall seconds: $($Timer.Elapsed.TotalSeconds)"
    }
}
```

Run these twelve commands **sequentially**, not concurrently. All six normal commands generate at 1344×768. All six SR commands explicitly select the SR bundle and its required 864×480 base, producing 1296×720. Every command defaults to `-CacheCondition seeded`; append `-CacheCondition empty` for a separately labelled empty-cache measurement.

The list is grouped by capability, not the historical execution order. The results CSV/JSON records UTC start/end times; use that chronology and note intervening activity when comparing OS-cache-sensitive loading times. Running the commands in a different order is a new local measurement protocol, not a reconstruction of identical OS/driver cache state.

```powershell
Invoke-H3Case -Delivery normal -Mode t2va   -Frames 120
Invoke-H3Case -Delivery normal -Mode t2va   -Frames 345
Invoke-H3Case -Delivery normal -Mode fl2va  -Frames 120
Invoke-H3Case -Delivery normal -Mode fl2va  -Frames 345
Invoke-H3Case -Delivery normal -Mode ref2va -Frames 120
Invoke-H3Case -Delivery normal -Mode ref2va -Frames 345
Invoke-H3Case -Delivery sr     -Mode t2va   -Frames 120
Invoke-H3Case -Delivery sr     -Mode t2va   -Frames 345
Invoke-H3Case -Delivery sr     -Mode fl2va  -Frames 120
Invoke-H3Case -Delivery sr     -Mode fl2va  -Frames 345
Invoke-H3Case -Delivery sr     -Mode ref2va -Frames 120
Invoke-H3Case -Delivery sr     -Mode ref2va -Frames 345
```

Keep native runtime directory and executable together. `--runtime-root` selects that runtime installation; it is not a model-precision or workflow switch. Neither a benchmark flag nor a runtime threshold override is required or supported by the native command. Cache preparation happens before timing; an unprofiled measurement includes native invocation through process exit, ordinary logging, cache finalization and teardown, but excludes later quality checks.

To switch the measured implementation, set **both** `$RuntimeRoot` to that implementation's installed `bin` directory and `$Trtmc = Join-Path $RuntimeRoot 'trtmc.exe'`; `$RuntimeLabel` only names the result and does not select binaries. Build/install baseline and candidate separately, then reuse the same `$NormalBundle`, `$SrBundle` and frozen per-delivery seeds. Each repeat needs a new result name, for example `$RuntimeLabel = 'candidate-repeat2'`; the helper refuses existing output directories. Record source commit, bundle/build provenance, exact arguments, cache condition/seed recipe, run order and exit status alongside the wall result. Do not reuse a measured run's updated cache as the next side's starting seed.

## Completed measurements and comparison coverage

Dimensions below are width×height. Every normal primary row uses native 1344×768; the FL2VA first/last fixtures are also 1344×768. Every SR primary row uses a 864×480 base and 1296×720 output.

Every candidate cell in this primary matrix refers to the single current runtime revision **`aa594f60`**. The PR #1240 baseline column retains completed earlier measurements; it does not imply that every row has a newly completed baseline pair. Each PSNR/SSIM entry identifies whether its reference is the preceding **`4dcfdc79` bulk output** or a **historical native2 same-controls output**. Historical native2 references are for quality only, not PR #1240 timing baselines. An additional direct normal-T2VA comparison with the original PR baseline is reported separately below. No original PR #1240 baseline timing has been completed for the other five short configurations.

| Bundle | Mode | Output frames | Base → output canvas | PR #1240 baseline wall s | Current candidate wall s | Current candidate generation s | Quality / status |
| --- | --- | ---: | --- | --- | --- | --- | --- |
| Normal | T2VA | 124 | 1344×768 → same | 693.563 | 570.797 | 562.484 | Full decode; sampled continuity preserved; versus bulk: PSNR 33.130097 dB / SSIM 0.956300 |
| Normal | FL2VA | 124 | 1344×768 → same | Not measured | 601.877 | 596.361 | Full decode; sampled continuity preserved; no paired 1344×768 baseline |
| Normal | REF2VA | 124 | 1344×768 → same | Not measured | 960.365 | 954.824 | Full decode; sampled scene retained; versus historical native2: PSNR 26.922813 dB / SSIM 0.834018 |
| SR | T2VA | 124 | 864×480 → 1296×720 | Not measured | 218.031 | 212.987 | Full decode; sampled continuity preserved; versus historical native2: PSNR 32.334520 dB / SSIM 0.939048 |
| SR | FL2VA | 124 | 864×480 → 1296×720 | Not measured | 219.617 | 214.338 | Full decode; sampled continuity preserved; versus historical native2: PSNR 42.010766 dB / SSIM 0.983258 |
| SR | REF2VA | 124 | 864×480 → 1296×720 | Not measured | 472.491 | 467.342 | Full decode; sampled continuity preserved; versus bulk: PSNR 29.575921 dB / SSIM 0.888277 |
| Normal | T2VA | 345 | 1344×768 → same | Not measured | 2819.268 | 2809.620 | Full decode; sampled scene continuity preserved; no paired 345-frame baseline |
| Normal | FL2VA | 345 | 1344×768 → same | Not measured | 3042.447 | 3032.232 | Full decode; sampled scene consistent with endpoints; no paired 345-frame baseline |
| Normal | REF2VA | 345 | 1344×768 → same | Not measured | 4940.005 | 4929.568 | Full decode; sampled scene retained, precise framing differs; versus intermediate bulk: PSNR 24.978547 dB / SSIM 0.823267 |
| SR | T2VA | 345 | 864×480 → 1296×720 | Not measured | 742.129 | 732.879 | Full decode; sampled scene continuity preserved; no paired 345-frame baseline |
| SR | FL2VA | 345 | 864×480 → 1296×720 | Not measured | 697.108 | 687.752 | Full decode; sampled scene consistent with endpoints; no paired 345-frame baseline |
| SR | REF2VA | 345 | 864×480 → 1296×720 | Not measured | 1750.920 | 1741.642 | Full decode; sampled scene retained with substantial forward camera move; no paired 345-frame baseline |

All twelve primary configurations are complete: six short cases and six long cases, with full media checks and scoped sampled visual review. The nine-request resident sequence and both short T2VA empty-cache cases are also complete. Each primary timing is one observation (`n=1`); other original-PR baseline comparisons are not measured. Neither these completed runs nor their sampled reviews establish comprehensive qualification.

Only calculate a speedup for a matched, completed pair. If a baseline long case is not run, mark it **not measured**, not estimated from a short result. Do not mix original BF16 bundle timings into this table. Earlier short INT8 delivery runs are historical evidence until the matching executable, cache seed, and measurement boundary are established.

### First completed current-candidate long case: normal T2VA, 345 frames

Case `memory-normal-t2va-345` completed with exit 0 on `aa594f60`, using the normal bundle, the exact T2VA waterfall prompt, seed 0, 345 requested/output frames, native 1344×768, 50 schedule points, guidance 1 and FBC 0.3. It was an unprofiled fresh-process run with an independent copy of the normal RTX disk-cache seed. Receipt wall-clock markers are September 10, 2026, 04:03:23.8508262–04:50:23.2413334 (UTC−07:00). The process stopwatch measured **2819.2677199 s (46.988 min)**; the family pipeline measured **2809.619646 s**. There is no completed matched 345-frame baseline, so this is a standalone latency result, not a runtime speedup or paired quality claim.

| Logged stage | Stage wall s, including load | Engine-execution s within stage | Engine launches |
| --- | ---: | ---: | --- |
| Text encoder | 39.588287 | 0.707296 | 1 |
| AdaLN precomputation | 9.680437 | 3.116124 | 49 |
| Denoising | 2581.388642 | 2566.792687 | 49 head / 8 tail / 49 finish |
| Video VAE | 170.000519 | 166.393501 | 20 |
| Super-resolution, disabled | 0.000000 | Not applicable | 0 |
| Audio VAE | 8.748268 | 6.673917 | 2 |

Engine execution is already included in each stage wall duration; do not add the two columns. Across 178 engine launches, logged execution totaled 2743.683524 s. The denoising engine times are 285.659491 s for heads, 2275.953156 s for tails and 5.180039 s for finish passes. FBC executed **8 full / 41 skipped** tails, with full tails at steps 1, 5, 12, 24, 36, 43, 47 and 49. The video decoder reported 28 spatial tiles. Text, AdaLN, denoiser-resident and VAE-resident cache-hit flags were all zero; the RTX disk cache loaded 25,438,048 bytes and saved 28,166,788 bytes. Those flags and byte counts describe different caches and do not establish coverage of every long-request specialization.

The live request used **104,060 packed rows**, against a 112,367-row profile MAX. Its four family cache tensors allocated **4,475,412,480 bytes**, rather than the **4,832,679,936-byte** profile-MAX capacity. The denoising stage consumed **91.88%** of the family pipeline, and its logged engine execution accounted for about **99.43%** of that stage. This long case remains execution-dominated despite the loading changes; these engine timers are not a fresh Nsight trace and do not isolate attention's share. Its observed wall time is 4.939 times the current 124-frame normal T2VA result, while the frame-count ratio is 2.782. Both recorded 8 full / 41 skipped tails; matching counts do not prove identical per-step decisions. This duration comparison is not a runtime speedup measurement or a linear extrapolation rule.

Full video/audio decode passed: **345 frames at 1344×768 and 24 fps**, with a **14.375 s** video track. Stereo 32 kHz AAC decodes to **460800 samples/channel (14.400 s)** with no NaN/Inf samples. Audio RMS is −26.632387 / −27.747976 dBFS (left/right), −27.154459 dBFS overall; peak is −13.003982 dBFS. The all-frame scene-change diagnostic scored 345 frames, flagged zero events and had maximum score 0.999 at threshold 10; it is an inspection aid, not a temporal-quality pass criterion.

Review of twelve contact-sheet frames (0, 31, …, 341) retained the waterfall, dark moss-covered rocks, pool, foliage and progressive camera approach, without obvious sampled garbling, replacement scene or abrupt composition jump. The final three decoded frames are not sampled by that contact sheet. There is no paired 345-frame quality reference, and no full-motion playback or listening pass is claimed. The private case evidence retains the receipt, generation log, media/decode/audio/temporal checks, contact sheet and scoped quality record; raw local paths and logs are not public report artifacts.

### Second completed current-candidate long case: normal FL2VA, 345 frames

Case `memory-normal-fl2va-345` completed with exit 0 on `aa594f60`, using the normal bundle, the exact FL2VA prompt and original 1344×768 first/last endpoint fixtures, seed 0, 345 requested/output frames, native 1344×768, 50 schedule points, guidance 1 and FBC 0.3. The request was an unprofiled fresh-process run with an independent copy of the normal RTX disk-cache seed. Receipt wall-clock markers are September 10, 2026, 04:50:27.3286195–05:41:09.8990357 (UTC−07:00). The process stopwatch measured **3042.4471722 s (50.707 min)**; the family pipeline measured **3032.231756 s**. No matched 345-frame FL2VA baseline is established, so this is a standalone latency result, not a speedup or paired accuracy claim.

| Logged stage | Stage wall s, including load | Engine-execution s within stage | Engine launches |
| --- | ---: | ---: | --- |
| Text and endpoint conditioning | 46.165081 | 5.018982 | 2 endpoint VAE / 2 vision / 1 text |
| AdaLN precomputation | 9.824457 | 3.129322 | 49 |
| Denoising | 2793.256562 | 2778.094562 | 49 head / 8 tail / 49 finish |
| Video VAE | 173.551156 | 169.679905 | 20 |
| Super-resolution, disabled | 0.000001 | Not applicable | 0 |
| Audio VAE | 9.217117 | 7.108043 | 2 |

The text stage includes endpoint VAE encoding, vision encoding and text encoding; do not add those components again. Engine execution is already part of each stage wall time. Across 182 engine launches, logged execution totaled 2963.030814 s. Denoising engine times are 312.341725 s for heads, 2460.444344 s for tails and 5.308493 s for finish passes. FBC executed **8 full / 41 skipped** tails, with full tails at steps 1, 6, 14, 27, 38, 45, 48 and 49. Video decoding reported 28 spatial tiles. The tiny disabled-SR entry is call overhead, not super-resolution work.

The request packed **108,126 rows**, including 2016 condition-video rows, within a 112,367-row profile MAX. Its four family cache tensors allocated **4,650,283,008 bytes**, against **4,832,679,936 bytes** at profile MAX. Text, AdaLN, denoiser-resident and VAE-resident cache-hit flags were all zero. The RTX disk cache loaded 25,438,048 bytes and saved 29,015,533 bytes; cache growth does not prove coverage of every shape. Denoising consumed **92.12%** of the family pipeline, with logged engine execution accounting for about **99.46%** of that stage. These timers support an execution-dominated long request but are not a new Nsight trace or an attention-specific breakdown.

Full video/audio decode passed: **345 frames at 1344×768 and 24 fps**, with a **14.375 s** video track. Stereo 32 kHz AAC decodes to **460800 samples/channel (14.400 s)** with no NaN/Inf samples. Audio RMS is −26.742686 / −27.320009 dBFS (left/right), −27.021761 dBFS overall; peak is −12.466553 dBFS. The all-frame scene-change diagnostic scored 345 frames, flagged zero events and had maximum score 0.784 at threshold 10; zero events is an inspection aid, not a quality guarantee.

Review of the original first/last images and twelve output contact-sheet frames (0, 31, …, 341) retained the same moss-covered rock formations, three-part waterfall, pool and foliage. The framing gradually approaches the waterfall consistently with the supplied endpoint composition, without obvious sampled garbling, scene substitution or abrupt composition jumps; local water and foliage detail evolves. This is not exact endpoint-pixel verification. The final three decoded frames are not sampled, no matched 345-frame baseline is available, and neither full-motion playback nor listening was performed. Raw receipts, logs and scoped QA remain private evidence.

### Completed SR long cases: T2VA and FL2VA, 345 frames

Cases `memory-sr-t2va-345` and `memory-sr-fl2va-345` completed with exit 0 on `aa594f60`. Both use the same explicitly enabled SR bundle, **864×480 base → 1296×720 output**, their exact mode-specific prompts, seed 0, 345 requested/output frames, 50 schedule points, guidance 1 and FBC 0.3. FL2VA additionally uses the original 1344×768 first/last endpoint fixtures, normalized to the smaller generation canvas. Each is an unprofiled fresh-process run with its own copy of the same SR disk-cache seed. Neither has a matched 345-frame baseline; these side-by-side results do not establish runtime speedup, paired accuracy or equivalence to native 768p generation.

Receipt wall-clock markers on September 10, 2026 (UTC−07:00) are 05:41:13.9164059–05:53:36.171114 for T2VA and 05:53:40.2571644–06:05:17.4934787 for FL2VA. The measured walls are **12.369 min** and **11.618 min** respectively for the complete video/audio generation command; super-resolution itself takes only **14.987 s** and **14.527 s** within those totals.

| Boundary / logged stage | SR T2VA, 345 frames, s | SR FL2VA, 345 frames, s |
| --- | ---: | ---: |
| Launch through process exit | 742.1290649 | 697.1082302 |
| Family pipeline | 732.879020 | 687.751773 |
| Text / text-and-endpoint conditioning, including load | 39.561882 | 42.325520 |
| AdaLN precomputation, including load | 9.791664 | 9.960739 |
| Denoising, including load | 551.489135 | 503.673539 |
| Video VAE, including load | 107.420814 | 107.865074 |
| Super-resolution | 14.987011 | 14.526615 |
| Audio VAE, including load | 9.535262 | 9.306513 |
| Logged engine execution, already within stages | 666.671476 | 620.234192 |

T2VA executed 222 engine launches: 1 text, 49 AdaLN, 49 denoiser heads, **8 full tails / 41 skipped**, 49 finishes, 20 video VAE, 44 SR and 2 audio VAE. FL2VA executed 225 launches: 2 endpoint VAE, 2 vision, 1 text, 49 AdaLN, 49 heads, **7 full tails / 42 skipped**, 49 finishes, 20 video VAE, 44 SR and 2 audio VAE. Both report 15 spatial video-decoder tiles. FL2VA's text stage includes all endpoint/vision/text encoding; engine execution is not an extra column to add to stage wall. T2VA's full tails were at steps 1, 5, 13, 25, 37, 44, 48 and 49; FL2VA's were at 1, 6, 16, 30, 41, 46 and 49. Different prompts, conditioning and full-tail counts prevent attributing their wall difference to a single optimization or concluding that FL2VA is generally faster.

T2VA packed **42,554 rows** and allocated **1,830,162,432 bytes** for the four family cache tensors; FL2VA packed **44,208 rows**, including 810 condition-video rows, and allocated **1,901,297,664 bytes**. Both preserve the 112,367-row / 4,832,679,936-byte profile-MAX contract. Text, AdaLN, denoiser-resident and VAE-resident cache-hit flags were zero in both fresh processes. Each loaded 25,940,792 bytes of RTX disk cache; T2VA saved 28,676,300 bytes and FL2VA saved 28,687,016 bytes. These observations distinguish live allocation, resident state and disk caching; they do not prove full specialization coverage.

Both outputs passed full video/audio decode: **345 frames, 1296×720, 24 fps, 14.375 s video**, and **stereo 32 kHz AAC with 460800 decoded samples/channel (14.400 s)** and no NaN/Inf samples. T2VA audio RMS is −26.053277 / −28.115843 dBFS (left/right), −26.963248 overall, with peak −12.592615 dBFS. FL2VA RMS is −25.251731 / −24.074271 dBFS, −24.623218 overall, with peak −10.675439 dBFS. Each scene-change diagnostic scored 345 frames and flagged zero events at threshold 10; maximum scores were 1.312 and 0.726 respectively. Zero events and finite audio are diagnostic checks, not motion or listening qualification.

Twelve T2VA samples (0, 31, …, 341) retain the small waterfall, large dark moss-covered rocks, green foliage and shallow pool with a gradual camera approach, without obvious sampled garbling, replacement scene or abrupt composition jump. FL2VA review of both original endpoints and the corresponding twelve output samples retains the waterfall streams, central rocks, pool and foliage with a gradual move toward the closer endpoint composition; no obvious sampled garbling, scene substitution, new dominant objects or abrupt composition jump was observed. Fine water/foliage detail evolves. The final three decoded frames are not sampled, and FL2VA endpoint-pixel equality is not asserted across the different input/base canvases. Neither case has paired PSNR/SSIM, a matched long-video quality reference, full-motion playback or a listening pass. Scoped QA and raw receipts/logs remain private evidence.

### Completed SR REF2VA long case, 345 frames

Case `memory-sr-ref2va-345` completed with exit 0 on `aa594f60`, using the SR bundle, the exact REF2VA prompt, the original ordered image/audio references, seed 0, 345 requested/output frames, 50 schedule points, guidance 1 and FBC 0.3. It generated at **864×480 → 1296×720** in an unprofiled fresh process with an independent copy of the SR disk-cache seed. Receipt wall-clock markers are September 10, 2026, 06:05:21.5158413–06:34:32.5537408 (UTC−07:00). The complete command took **1750.9196092 s (29.182 min)** and the family pipeline took **1741.641654 s**. SR itself accounted for **14.214224 s** within that total. No matched 345-frame SR REF2VA baseline is available, so these are standalone timings, not a speedup or paired accuracy claim.

| Logged stage | Stage wall s, including load | Engine-execution s within stage | Engine launches |
| --- | ---: | ---: | --- |
| Shared text and vision encoding | 57.305147 | 16.427222 | 1 vision / 1 text |
| Reference image/audio encoding | 4.680483 | 3.403622 | 7 image VAE / 1 audio VAE |
| AdaLN precomputation | 9.560837 | 3.099296 | 49 |
| Denoising | 1538.692978 | 1517.187050 | 49 head / 9 tail / 49 finish |
| Video VAE decoding | 108.049297 | 101.733646 | 20 |
| Super-resolution | 14.214224 | 13.274069 | 44 |
| Audio VAE decoding | 9.025728 | 6.826353 | 2 |

Logged engine execution totaled **1661.951258 s across 232 launches**, already included in the stage wall durations. Denoising engine times were 206.112789 s for heads, 1305.436359 s for tails and 5.637902 s for finish passes. FBC executed **9 full / 40 skipped** tails, with full tails at steps 1, 4, 10, 20, 32, 40, 45, 48 and 49. Denoising occupied **88.35%** of the family pipeline, and engine execution accounted for about **98.60%** of that stage. These engine timers are not a new attention-specific Nsight profile; different conditioning and full-tail counts also prevent a single-cause comparison with the T2VA/FL2VA results.

The runtime selected denoiser profile **1 of 2 (zero-based)** and packed **57,287 live rows**, within its 630,310-row MAX. The four family cache tensors allocated **2,463,799,296 bytes**, compared with the **27,108,372,480-byte** profile-MAX contract. The RTX disk cache loaded 25,940,792 bytes and saved 30,955,057 bytes. These are separate allocation/cache observations, not proof that every specialization was already cached; REF2VA does not log the four T2VA/FL2VA conditioning/resident-hit flags, so they are not inferred here.

Full video/audio decode passed: **345 H.264 frames at 1296×720 and 24 fps (14.375 s)**, with **stereo 32 kHz AAC and 460800 decoded samples/channel (14.400 s)**. Audio RMS is −29.787400 / −29.747068 dBFS (left/right), −29.767187 overall; peak is −14.142183 dBFS, with no NaN/Inf samples. The scene-change diagnostic scored 345 frames, flagged zero events and had maximum score 0.861 at threshold 10; this remains an inspection aid rather than a continuity guarantee.

Reference/contact inspection at frames 0, 31, …, 341 retained the supplied waterfall, rocks, pool and foliage, ending in a closer view of the same rocks and water. The forward camera move is substantial: the framing is not unchanged. No obvious sampled garbling, replacement scene or abrupt composition jump was observed. The final three decoded frames are not sampled. No matched long-output reference, pixel parity, full-motion playback or listening pass is claimed; scoped QA and raw receipts/logs remain private evidence.

### Completed normal REF2VA long case, 345 frames

Case `memory-normal-ref2va-345` completed with exit 0 on `aa594f60`, using the normal bundle, the exact REF2VA prompt and original ordered image/audio references, seed 0, 345 requested/output frames, native **1344×768**, 50 schedule points, guidance 1 and FBC 0.3. It was an unprofiled fresh-process run with an independent copy of the normal RTX disk-cache seed. Receipt wall-clock markers are September 10, 2026, 06:34:36.4534772–07:56:56.6015318 (UTC−07:00). The complete command took **4940.0052086 s (82.333 min)** and the family pipeline took **4929.567827 s**. No original PR #1240 matched long REF2VA timing baseline was measured.

| Logged stage | Stage wall s, including load | Engine-execution s within stage | Engine launches |
| --- | ---: | ---: | --- |
| Shared text and vision encoding | 58.314781 | 16.355333 | 1 vision / 1 text |
| Reference image/audio encoding | 4.602564 | 3.293916 | 7 image VAE / 1 audio VAE |
| AdaLN precomputation | 9.670724 | 3.154041 | 49 |
| Denoising | 4672.632863 | 4650.855335 | 49 head / 9 tail / 49 finish |
| Video VAE decoding | 174.825516 | 163.679110 | 20 |
| Super-resolution, disabled | 0.000001 | Not applicable | 0 |
| Audio VAE decoding | 9.274648 | 7.152364 | 2 |

Across **188 engine launches**, logged execution totaled **4844.490100 s**, already included in stage wall times. Denoising engine times were 569.553868 s for heads, 4069.134500 s for tails and 12.166966 s for finishes. FBC executed **9 full / 40 skipped** tails. Denoising occupied **94.79%** of the family pipeline; its logged engine execution occupied about **99.53%** of that stage. This is an execution-dominated long request, not a fresh attention-specific profile. The disabled-SR entry is call overhead, not super-resolution work.

The request packed **118,793 live rows** within a 630,310-row profile MAX. Its four family cache tensors allocated **5,109,049,344 bytes**, against the **27,108,372,480-byte** profile-MAX contract. The RTX disk cache loaded 25,438,048 bytes and saved 30,435,779 bytes. These allocation/cache observations do not prove full specialization coverage or establish a latency improvement by themselves.

Full video/audio decode passed: **345 H.264 frames at 1344×768 and 24 fps (14.375 s)**, with **stereo 32 kHz AAC and 460800 decoded samples/channel (14.400 s)**. Audio RMS is −29.390001 / −29.276883 dBFS (left/right), −29.333074 overall; peak is −13.779859 dBFS, with no NaN/Inf samples. The all-frame scene-change diagnostic scored 345 frames, flagged zero events and reached maximum score 1.432 at threshold 10; this is an inspection aid, not a motion-quality guarantee.

The direct all-frame comparison against the **intermediate `4dcfdc79` bulk output** produced PSNR **24.978547 dB** (per-frame minimum 20.483076, maximum 34.630736) and SSIM **0.823267**. All 49 full/skip FBC decisions matched. Aligned decoded-audio subtraction had RMS −44.750563 dBFS and peak −21.639181 dBFS, with finite samples. These are quality-only comparisons, not an original-PR baseline. The older bulk run's 5473.4675169-second wall overlapped CPU compilation, so no controlled speedup is calculated from it.

Inspection of the original image and both matching twelve-frame contact sheets (0, 31, …, 341) retained the same waterfall, moss-covered rocks, pool and foliage with a substantial forward camera move. Fine water/foliage detail and precise framing/progression differ, particularly in later close-ups. No obvious sampled garbling or replacement scene was observed; this is not near-pixel reproduction or an assertion of unchanged framing. The final three frames are not sampled. No full-motion playback or listening pass is claimed. Raw receipts, comparison logs and the scoped QA record remain private evidence.

### Completed short T2VA empty-cache checks

These additional fresh-process, unprofiled runs use the same `aa594f60` runtime, corresponding bundle, prompt, seed 0, 120 requested / 124 output frames, 50 schedule points, guidance 1 and FBC 0.3 as their seeded primary references. Reproduce the empty-path condition explicitly with:

```powershell
Invoke-H3Case -Delivery normal -Mode t2va -Frames 120 -CacheCondition empty
Invoke-H3Case -Delivery sr     -Mode t2va -Frames 120 -CacheCondition empty
```

The dedicated cache path was absent in each newly created output directory; the runtime saved it on successful completion. An absent RTX cache does **not** mean an OS/driver-cold machine. These are one ordered sample per condition, without resetting OS/driver caches, not a repeatability estimate or an isolated causal measurement of runtime-cache savings.

| Delivery / condition | Process wall s | Family pipeline s | Initial RTX cache bytes | Final RTX cache bytes | Full / skipped tails |
| --- | ---: | ---: | ---: | ---: | --- |
| Normal, seeded primary | 570.7969257 | 562.483866 | 25,438,048 | 25,438,048 | 8 / 41 |
| Normal, empty path | 580.0891387 | 574.847324 | Absent | 15,501,460 | 8 / 41 |
| SR, seeded primary | 218.0305484 | 212.987288 | 25,940,792 | 25,940,792 | 7 / 42 |
| SR, empty path | 242.4612920 | 237.500313 | Absent | 16,041,001 | 7 / 42 |

Every one of the 49 FBC tail decisions matches within each empty/seeded pair. Normal executes full tails at steps 1, 5, 13, 25, 37, 44, 48 and 49; SR at 1, 7, 19, 33, 42, 47 and 49. Text, AdaLN and resident cache-hit flags are zero in all four runs. The different final disk-cache sizes describe their different contents/history, not proof of missing required specialization or coverage of other shapes.

The empty-path stage durations are below, in seconds. Engine execution is already part of the stages; neither its total nor the family total is added to the stage durations.

| Empty-path delivery | Text | AdaLN | Denoising | Video VAE | SR | Audio VAE | Engine execution total / launches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Normal | 47.502475 | 10.174940 | 439.407215 | 65.620852 | 0.000000 | 12.062294 | 489.80438232 s / 165 |
| SR | 46.569459 | 10.658987 | 118.509443 | 44.113635 | 6.119438 | 11.495466 | 152.78300626 s / 180 |

Both outputs passed full decode: 124 frames at 24 fps, normal 1344×768 or SR 1296×720, and finite 32 kHz stereo audio with 165888 decoded samples/channel. Direct comparisons against their respective same-source seeded outputs cover all 124 decoded frames:

| Empty versus seeded | PSNR dB (minimum–maximum) | SSIM | Float audio-difference RMS / peak dBFS | Temporal diagnostic maximum |
| --- | --- | ---: | --- | ---: |
| Normal T2VA | 26.675576 (25.349386–27.231228) | 0.873510 | −42.535977 / −19.933908 | 1.113 |
| SR T2VA | 30.241110 (29.475977–30.922989) | 0.924053 | −39.637379 / −19.430063 | 1.438 |

Each pair's twelve-frame contact sheets retain the waterfall, rocks, pool, foliage and progressive framing, with local water/ripple/foliage differences but no obvious sampled garbling, replacement scene or abrupt composition jump. Neither pair is pixel- or audio-identical despite identical FBC decisions; these observations do not isolate the source of the numerical differences. Both temporal diagnostics scored all 124 frames with zero flagged events at threshold 10, an inspection aid only. No full-motion or listening review was performed. Append-only paired evidence is under each `memory-*-t2va-124-empty/qa-seeded` directory; the original automatic-check files remain intact.

### Why REF2VA remains slower with caching enabled

FirstBlockCache is **enabled at 0.3 in all four current short cases below**; the observed full/skipped counts confirm tail reuse. REF2VA packs additional reference conditioning into the denoising sequence. Both builders use dense native TensorRT attention, so longer sequences still require more attention arithmetic on executed blocks. FBC skips tails, not the 49 head and 49 finish passes.

| Current 124-frame case | Actual packed rows | Full / skipped tails | Denoising stage s, including load |
| --- | ---: | ---: | ---: |
| Normal T2VA | 37,804 | 8 / 41 | 439.067409 |
| Normal REF2VA | 52,535 | 8 / 41 | 808.384805 |
| SR T2VA | 15,493 | 7 / 42 | 112.782122 |
| SR REF2VA | 30,224 | 8 / 41 | 346.347515 |

At otherwise equal attention dimensions, the sequence-length arithmetic ratios are `(52535 / 37804)^2 ≈ 1.931` for normal generation and `(30224 / 15493)^2 ≈ 3.806` for SR-base generation. These describe **attention's quadratic arithmetic only**, not measured E2E multipliers or proof of the cause of an elapsed-time difference. SR REF2VA also executes one more full tail than SR T2VA; conditioning, loading and other stages have separate costs.

The earlier PR #1240 T2VA Nsight trace attributed 270.831 of 429.808 denoiser GPU-kernel seconds to attention (about 63%), supporting attention as an important optimization target. That is not a profile of the current REF2VA runs and does not establish their exact bottleneck. The observed REF slowdown is therefore not evidence that FBC was omitted.

### Current request-memory candidate: two completed short pilots

Cases `memory-normal-t2va-124` and `memory-sr-ref2va-124` completed on `aa594f60`, with exact launch-to-exit times of 570.7969257 s and 472.4911404 s respectively. Both were unprofiled fresh-process requests using the same engine payloads, request/seed and independent copies of the same per-delivery RTX cache seed as their bulk-runtime references. Both used FBC 0.3 and executed 8 full / 41 skipped denoiser steps. The earlier bulk runs preceded these pilots; no OS-cache purge or clock lock was applied. These are one observation per case/runtime, not repeated-run timing distributions or an attribution of every timing difference to cache allocation.

| Boundary / stage | Normal T2VA, 124 frames, s | SR REF2VA, 124 frames, s |
| --- | ---: | ---: |
| Launch through process exit | 570.797 | 472.491 |
| Family generation pipeline | 562.484 | 467.342 |
| Text / text-and-vision encoding, including load | 40.458 | 57.178 |
| Reference conditioning encoding | Not applicable | 4.445 |
| AdaLN precomputation, including load | 10.834 | 9.417 |
| Denoising, including load | 439.067 | 346.348 |
| Video VAE, including load | 67.064 | 41.081 |
| Super-resolution | Not enabled | 5.123 |
| Audio VAE, including load | 4.971 | 3.697 |

Stage durations include their engine execution and must not be added to a separate engine-execution column. Pipeline time and process-wall time are different boundaries.

For normal T2VA, all 124 frames and stereo audio decoded. Against `bulk-normal-t2va-124`, decoded-MP4 PSNR is **33.130097 dB** (minimum 32.601053 dB) and SSIM **0.956300**. Inspection of both twelve-frame contact sheets retained the same waterfall, moss-covered rock/pool layout and progressive camera move. Visible differences concern local water and foliage detail, with no obvious garbling, scene replacement or abrupt framing jump in the samples.

A separate **direct** comparison of this same current-candidate T2VA output with `baseline-normal-t2va-124` from PR #1240 measured PSNR **32.478840 dB** (minimum 31.758680 dB) and SSIM **0.939337** across all 124 aligned decoded frames. Both twelve-frame contact sheets retain the same waterfall/rocks/pool/foliage and progressive framing, with local water and foliage differences and no sampled garbling or scene substitution. The aligned stereo audio subtraction has overall RMS −41.401679 dBFS and peak −19.672071 dBFS, with no NaN/Inf samples. These are direct comparisons, not metrics inferred through the bulk intermediate; they do not establish pixel identity, full-motion quality or a listening pass. The primary row keeps its explicitly labelled bulk-reference metrics.

For SR REF2VA, full decode passed independently for all 124 frames at 1296×720. Against `bulk-sr-ref2va-124`, decoded-MP4 PSNR is **29.575921 dB** and SSIM **0.888277**. Both twelve-frame contact sheets retain the same waterfall, rocks, foliage and coherent camera progression; local water/ripple details differ. Logged full/skip decisions agree for all 49 steps, although cache metric values and media are not numerically identical. There is still no completed original PR #1240 REF2VA baseline pair.

Both pilots have 32 kHz stereo audio with 165888 decoded samples per channel and no NaN/Inf samples. SR REF2VA's aligned audio difference from its bulk reference has overall RMS −46.755 dBFS and peak −22.844 dBFS, so audio identity is not claimed. Neither pilot has full-motion playback or listening qualification. The scene-change diagnostics flagged no events; this and contact-sheet inspection are limited checks, not universal perceptual-equivalence guarantees.

### Current request-memory candidate: completed normal FL2VA, 124 frames

Case `memory-normal-fl2va-124` completed on `aa594f60` in **601.8772073 s** from launch through process exit, with a **596.360619 s** family pipeline. It used an unprofiled fresh process, a seeded RTX disk cache, seed 0, the original first/last endpoint fixtures and FL2VA prompt, and native 1344×768 generation. FBC 0.3 executed **7 full / 42 skipped** denoiser steps. There is no matched native 1344×768 FL2VA baseline, so this is a standalone result, not a measured speedup or paired accuracy claim.

| Logged stage | Seconds |
| --- | ---: |
| Text and endpoint conditioning, including load | 47.441187 |
| AdaLN precomputation, including load | 9.804005 |
| Denoising, including load | 465.457702 |
| Video VAE, including load | 69.658701 |
| Super-resolution, disabled | 0.000000 |
| Audio VAE, including load | 3.914040 |

The logged text stage includes endpoint VAE encoding, vision encoding and text encoding; those components must not be added again. Engine execution is included within these stage durations. Denoising launched 49 heads, 7 tails and 49 finish passes; video VAE decoding launched 7 times, with 28 spatial tiles reported.

Full media decode passed: 124 frames at 1344×768 and 24 fps, with 32 kHz stereo audio containing 165888 decoded samples per channel and finite samples. Inspection of twelve contact-sheet frames (0, 11, …, 121) retained the same waterfall, moss-covered rocks, pool and foliage with gradual camera progression; no obvious garbling, scene substitution or abrupt framing jump was visible in those samples. The scene-change diagnostic scored all 124 frames, flagged zero events and had maximum score 1.156. This is an inspection aid, not a temporal-quality guarantee. No full-motion playback or listening review is claimed.

### Current request-memory candidate: remaining three short configurations

The following three cases completed on `aa594f60` with unprofiled fresh processes, seeded RTX disk caches, seed 0, the exact mode-specific prompts/media above and FBC 0.3. These are measured current-candidate results. The historical references named below support quality comparison only; their times are not used as PR #1240 baseline cells or controlled speedup denominators.

| Boundary / logged stage | Normal REF2VA, 124 frames, s | SR T2VA, 124 frames, s | SR FL2VA, 124 frames, s |
| --- | ---: | ---: | ---: |
| Launch through process exit | 960.3654436 | 218.0305484 | 219.6166786 |
| Family generation pipeline | 954.823610 | 212.987288 | 214.337592 |
| Text / text-and-vision / endpoint conditioning, including load | 60.350940 | 39.556451 | 43.307906 |
| Reference image/audio encoding | 4.581689 | Not applicable | Not applicable |
| AdaLN precomputation, including load | 9.956151 | 9.843114 | 10.086893 |
| Denoising, including load | 808.384805 | 112.782122 | 112.017454 |
| Video VAE, including load | 67.798490 | 41.920477 | 39.986431 |
| Super-resolution | 0.000001 (disabled) | 5.111415 | 5.081926 |
| Audio VAE, including load | 3.648204 | 3.739915 | 3.820298 |

For normal REF2VA the text row covers text/vision, with reference image/audio encoding separately timed. For SR FL2VA it includes endpoint VAE, vision and text encoding. Engine execution is part of each stage, not an extra duration to add. The tiny disabled-SR entry is the logged call overhead, not super-resolution work. Normal REF2VA executed 8 full / 41 skipped steps, SR T2VA 7 / 42, and SR FL2VA 6 / 43; all executed 49 heads and 49 finish passes. All three decoded video through seven VAE launches, and each SR case executed 16 SR launches.

**Normal REF2VA quality reference:** `normal-ref2va-fbc0p3`, a historical native2 same-controls output. Across all 124 decoded YUV420p frames at 1344×768, PSNR is **26.922813 dB** (minimum 24.223287 dB, maximum 32.571514 dB) and SSIM **0.834018**. Both twelve-frame contact sheets preserve the waterfall, boulders, foliage and coherent forward camera movement. Local water details and precise camera progression/framing differ; no obvious garbled blocks, replacement scene or abrupt framing jump is visible in the samples. Although both thresholds are 0.3, the candidate used **8 full / 41 skipped** steps versus the historical reference's **9 / 40**, with nine per-step tail decisions differing. Therefore any historical wall-time difference cannot be attributed solely to allocation or loading changes. The scene-change diagnostic scored all 124 frames, flagged zero events and had maximum score 2.773.

**SR T2VA quality reference:** `sr-t2va-fbc0p3`, a historical native2 same-controls output. Across all 124 decoded YUV420p frames at 1296×720, PSNR is **32.334520 dB** (minimum 31.356799 dB, maximum 32.972277 dB) and SSIM **0.939048**. Both matched twelve-frame contact sheets preserve the waterfall, central rocks, foreground pool, foliage and gradual forward camera movement. Water/ripple and foliage details differ, without obvious garbling or scene substitution in the inspected samples. Both runs used 7 full / 42 skipped steps. The scene-change diagnostic flagged zero events across 124 frames, with maximum score 1.510.

**SR FL2VA quality reference:** `sr-fl2va-fbc0p3`, a historical native2 same-controls output. Across all 124 decoded YUV420p frames at 1296×720, PSNR is **42.010766 dB** (minimum 41.463300 dB, maximum 43.991973 dB) and SSIM **0.983258**. Both matched twelve-frame contact sheets retain the same waterfall, rock formations, pool, foliage and consistent framing progression. Slight local water/ripple and foliage texture differences remain, with no obvious garbling, scene substitution, abrupt composition jump or material perceptual regression visible in the samples. Both runs used 6 full / 43 skipped steps. The scene-change diagnostic flagged zero events across 124 frames, with maximum score 0.748.

All three checks independently decoded video/audio and verified 32 kHz stereo audio with 165888 samples per channel and no NaN/Inf samples. The following aligned audio-difference measurements use decoded floating-point samples; audio identity or inaudibility is not asserted.

| Current case | Audio RMS left / right, dBFS | Difference RMS overall, dBFS | Difference peak, dBFS |
| --- | --- | ---: | ---: |
| Normal REF2VA | −28.582193 / −28.076620 | −46.455471 | −21.667750 |
| SR T2VA | −29.076779 / −30.555302 | −41.582394 | −19.137849 |
| SR FL2VA | −26.826067 / −26.516305 | −37.794226 | −17.952525 |

Historical comparison artifacts are retained in each case's append-only `qa-historical-native2` directory; raw logs remain private. SR FL2VA uses the final `audio-subtraction-float.log` measurement, not the preliminary diagnostics excluded by its QA notes. No full-motion playback, listening pass or universal perceptual-equivalence claim follows from these sampled inspections and diagnostics.

### Earlier bulk-reader short measurements, retained separately

These completed results belong to **`4dcfdc79`**, not the current candidate. They are the media/timing references for the two request-memory pilots above.

| Bulk-reader case | Base → output canvas | Output frames | Launch-to-exit s | Pipeline s | Full / skipped steps |
| --- | --- | ---: | ---: | ---: | ---: |
| `bulk-normal-t2va-124` | 1344×768 → same | 124 | 596.982 | 591.795 | 8 / 41 |
| `bulk-sr-ref2va-124` | 864×480 → 1296×720 | 124 | 498.019 | 492.655 | 8 / 41 |

The completed bulk SR REF2VA 124-frame run used FBC 0.3 with 8 full / 41 skipped steps. Against the allocation-and-diagnostics **pre-bulk** output, all 124 decoded video frames matched exactly (YUV420p PSNR ∞, SSIM 1.000000; an RGB-converted comparison also had PSNR ∞). The aligned beginning/middle/end stills retain the same coherent waterfall scene and camera progression. Decoded stereo audio is not identical: subtraction RMS was −56.95 dBFS and peak −38.16 dBFS, with no nonfinite samples. Listening/full-motion review is not claimed. The automatic scene-change diagnostic flagged no events, which is an inspection aid rather than a temporal-quality guarantee. This comparison isolates the intermediate runtime variants; it does not describe the current request-memory output, and the pre-bulk timing is not substituted for the still-pending original PR baseline.

### Completed long-REF diagnostic, preceding the request-sized family caches

Case `bulk-normal-ref2va-345` used commit `4dcfdc79`, the normal bundle, 345 output frames at 1344×768, the original image/audio/prompt fixtures, and FBC 0.3. Its observed CLI wall time was **5473.468 s (91.22 min)**; family pipeline time was 5463.473 s. This is not entered as a controlled formal result: CPU compilation of the separate candidate overlapped part of the run, and there is no completed matched long baseline.

| Diagnostic stage | Wall seconds | Engine-execution seconds |
| --- | ---: | ---: |
| Text and vision | 58.352 | 16.571 |
| Reference image/audio encoding | 5.511 | 4.361 |
| AdaLN | 9.496 | 3.114 |
| Denoising | 5203.173 | 5180.842 |
| Video VAE | 177.432 | 165.810 |
| Audio VAE | 9.247 | 6.510 |

Engine time is included within the corresponding stage and is not added to it. Denoising executed 49 heads, 9 tails and 49 finish passes (40 skipped tails); the video VAE executed 20 times. Full media decode passed with 345 frames at 24 fps and 32 kHz stereo audio. Twelve sampled frames retain the same waterfall/rocks/foliage and a progressive camera move, without visible garbling or scene replacement. The 345-frame scene-change diagnostic flagged no events (maximum score 1.508 at threshold 10), which is an inspection aid, not a motion-quality guarantee. Audio statistics contained no NaN/Inf samples. Full-motion playback, listening review and paired long-video accuracy are not claimed.

### Earlier controlled pair: PR #1240 versus bulk-reader normal T2VA, 124 frames

Both are unprofiled fresh-process requests, with the same engine bundle, original
prompt (94 tokens), seed 0, 1344×768 canvas, and independent copies of the same
RTX seed cache. Both report text/AdaLN/resident cache hits as zero and FBC
8 full / 41 skipped denoiser steps. Bulk runtime `4dcfdc79` ran first, baseline second; no
operating-system cache purge or clock lock was applied.

| Boundary / stage | PR #1240 baseline s | Bulk runtime `4dcfdc79` s |
| --- | ---: | ---: |
| Launch through process exit | 693.563 | 596.982 |
| Family generation pipeline | 688.107 | 591.795 |
| Text encoding, including load | 117.170 | 55.207 |
| AdaLN precomputation, including load | 28.049 | 14.271 |
| Denoising, including load | 463.209 | 446.826 |
| Video VAE, including load | 75.189 | 70.821 |
| Audio VAE, including load | 4.413 | 4.558 |

The observed process-wall reduction is **96.581 seconds (13.9%)**, or 1.162×
throughput for this one paired request. This is not a multi-run statistical
estimate or a promise for another GPU. The stages include their engine execution
and must not be added to a separate engine-execution column. Minor compute-time
variation remains; do not attribute every sub-stage difference to file reading.

All 124 frames and both audio channels decoded. Sampled review of the 12
frames showed the same waterfall/rock layout and continuous camera progression,
with local water and foliage differences rather than scene replacement. Decoded
MP4 PSNR is 31.830 dB (minimum 31.181 dB), SSIM 0.935705. These outputs are not
pixel-identical; sampled-frame review is not a complete motion or listening test.

### Completed intermediate result: SR REF2VA, allocation + diagnostics, pre-bulk

This completed run **does not contain the Windows bulk-reader change** and does not fill the final candidate matrix above. Case ID: `candidate-sr-ref2va-124`. It used a fresh process, an existing seeded disk runtime cache, FBC 0.3, seed 0, the recorded REF2VA prompt/image/audio fixtures, and no profiler.

| Variant | Base → output | Frames | Launch-to-exit s | Pipeline s | Full / skipped steps |
| --- | --- | ---: | ---: | ---: | ---: |
| Allocation + diagnostics, pre-bulk | 864×480 → 1296×720 | 124 | **562.718** | 556.998 | 8 / 41 |
| Historical same-request FBC 0.3 output | 864×480 → 1296×720 | 124 | 608.977 | 603.891 | 8 / 41 |

The historical row supplies a visual reference and timing context, **not a controlled speedup claim**: identical starting cache histories and a paired baseline/candidate execution order have not been established. The later bulk-reader and request-memory runtimes have their own separately labelled measurements above.

| Pre-bulk pipeline stage | Seconds |
| --- | ---: |
| Text/vision encoder | 101.672 |
| Reference conditioning encoder | 7.271 |
| AdaLN | 23.593 |
| Denoiser | 367.304 |
| Video VAE | 47.065 |
| Super-resolution | 5.534 |
| Audio VAE | 4.501 |

CPU-only media checks decoded the entire video/audio successfully: 124 H.264 frames at 1296×720 and 24 fps, with 32 kHz stereo AAC audio containing 165888 samples/channel (5.184 s). Audio RMS was −28.61 / −28.11 dBFS (left/right), peak −14.06 dBFS, with no NaNs or infinities. An aligned decoded-float audio subtraction from the historical output had overall RMS −47.62 dBFS; the audio is not asserted to be identical. Listening review was not performed in this check.

The full-clip decoded-MP4 comparison measured **PSNR 27.163 dB and SSIM 0.838065** in the common YUV420p format. Both contact sheets, aligned frames 0/60/123, and a 1:1 final-frame waterfall crop were inspected. The main waterfall/boulder arrangement, foliage and progressive camera push remain consistent; local water/ripple and foliage details differ. No scene substitution, garbled image or abrupt framing jump was visible in the inspected samples. This is not pixel-equivalent reproduction, a full-motion playback review, or a universal quality guarantee; the metrics alone do not establish perceptual degradation.

The private evidence set retains the original receipt/generation log, `media.json`, full-decode log, PSNR/SSIM logs, audio statistics/difference log, `quality.json`, contact sheets and aligned comparison stills. Only approved media and sanitized summaries should accompany a public report; raw local traces and machine paths remain private.

### Optional additional native 480p controls

These SR-disabled controls are outside the twelve primary configurations. Keep any completed compact FL2VA timings here, with their actual executable/cache provenance, rather than putting them in the native 768p rows. Add separate T2VA or REF2VA control rows only if those additional requests are run.

| Bundle / control | Mode | Output frames | Base → output canvas | Baseline wall s | Candidate wall s | Status |
| --- | --- | ---: | --- | --- | --- | --- |
| Normal, SR disabled | FL2VA | 124 | 864×480 → same | Not entered | Not entered | Optional; matching provenance pending |
| Normal, SR disabled | FL2VA | 345 | 864×480 → same | Not entered | Not entered | Optional; not measured |

To run such a control, use `Invoke-H3Case -Delivery normal -Mode fl2va -Frames 120 -Native480pControl` (or `-Frames 345`). The helper uses 864×480 and a separately named `normal-480p-control` case directory. It keeps the prompt, seed, original fixtures, and selected cache condition unchanged. This test-only switch selects a smaller canvas on the same normal bundle; it does not enable SR. Its final MP4 remains 864×480. This is a separate canvas comparison, not evidence of a native 768p speedup.

## Optional native C++ resident consumer

The candidate's `families/minimax_h3/tools/benchmark_video.cpp` uses the public `load_task` / `IVideoGeneration::generate_video` API, loads one task, runs ordered requests, and writes audio/video through the existing native Windows media implementation. This avoids process/task recreation but does not imply every engine or conditioning result is retained.

```powershell
$ConsumerBuild = Join-Path $ArtifactRoot 'resident-consumer'
cmake -S (Join-Path $RepoRoot 'families/minimax_h3/tools') -B $ConsumerBuild -G Ninja `
    -DCMAKE_BUILD_TYPE=Release `
    "-DTRTMC_RUNTIME_LIBRARY=$InstallRoot/lib/trtmc_runtime.lib" `
    "-DCMAKE_PREFIX_PATH=$JsonInstall"
cmake --build $ConsumerBuild
$Consumer = Join-Path $ConsumerBuild 'h3_benchmark_video.exe'
```

Use the reproduction package's **exact** `resident-requests.json`, `resident-token-counts.json` and `RESIDENT-REPRODUCTION.md`, extracted into `$ArtifactRoot`. The manifest selects the SR bundle and requests A, A, B, long-A, A, FL2VA, landscape REF2VA, square REF2VA, then restored landscape REF2VA. All nine requests ask for 120 frames and produce 124 frames; "long-A" means a 1662-token prompt, not a 345-frame video. This sequence must not be replaced by an illustrative duration/mode loop when comparing recorded resident results.

The supplied manifest resolves its paths relative to `$ArtifactRoot`, including `install/bin`, `MiniMax-H3-SR-FBC03.bundle`, `inputs/`, and `outputs/resident/`. Select the initial cache condition explicitly before running it:

```powershell
$ResidentCacheCondition = 'seeded' # Or 'empty'; applies before the FIRST request only.
if ($ResidentCacheCondition -notin @('seeded', 'empty')) { throw 'Unknown runtime-cache condition' }
$Manifest = Join-Path $ArtifactRoot 'resident-requests.json'
$SequenceRoot = (New-Item -ItemType Directory (Join-Path $OutputRoot 'resident')).FullName
$ResidentCache = Join-Path $SequenceRoot 'runtime.rtxcache' # Must initially be absent.
if ($ResidentCacheCondition -eq 'seeded') {
    $ResidentSeed = Join-Path $SeedRoot 'sr.rtxcache'
    if (!(Test-Path -LiteralPath $ResidentSeed) -or (Get-Item -LiteralPath $ResidentSeed).Length -eq 0) {
        throw 'Prepare or obtain a compatible frozen SR seed first'
    }
    Copy-Item -LiteralPath $ResidentSeed -Destination $ResidentCache
}
& $Consumer $Manifest --validate-only
if ($LASTEXITCODE -ne 0) { throw 'Invalid resident request manifest' }
$SequenceTimer = [Diagnostics.Stopwatch]::StartNew()
& $Consumer $Manifest 2>&1 | Tee-Object -FilePath (Join-Path $SequenceRoot 'sequence.log')
$SequenceExitCode = $LASTEXITCODE
$SequenceTimer.Stop()
if ($SequenceExitCode -ne 0) { throw 'Resident sequence failed; preserve partial receipts' }
"Whole resident-process wall seconds: $($SequenceTimer.Elapsed.TotalSeconds)"
```

Use a fresh extraction/output directory for every repeat, and ensure the manifest's `runtime_root` names the matching tested runtime installation. The final three requests change prompt and reference content/count as well as vision shape; they are a conditioning-state transition check, not a controlled image-shape-only timing comparison. Separate normal-bundle or 124→345→124 resident experiments may use the source tool's documented schema, but must keep their own exact manifests and results. Do not overwrite an old sequence, share its writable cache with an active process, or resume in a new process and label that continuation resident. `--validate-only` checks inputs and schema without loading engines or generating media; it is not a model/quality pass.

The public model page also includes a minimal standalone C++ consumer returning `VideoResult` in memory. Applications own decoded input media and output-container writing, and should keep the task alive across repeated requests.

### Completed resident sequence: all nine requests

The unprofiled nine-request SR sequence completed with exit 0 on `aa594f60`, using one native process and the same task instance for all requests. Its process stopwatch measured **2371.9650594 s**, including startup, task setup, all requests and finalization. Receipt wall-clock markers are September 10, 2026, 02:56:06.9569415–03:35:39.1104044 (UTC−07:00). Initial task loading was **0.6728343 s**, counted once; it is not added again for each receipt and can exclude lazy engine loading inside later generation calls. The process began with the seeded SR disk cache, used FBC 0.3 throughout, and did not reset or copy the cache between requests.

The request wall below covers preparation through MP4 writing, not fresh-process E2E. Prompt tokens and Qwen presentation rows are separate counts; endpoint/reference features explain their difference. A dash in the cache-hit column means not logged, not a claimed hit or miss.

| Request | Prompt tokens / presentation rows | Prepare s | Generate API s | MP4 write s | Request wall s | Full / skipped steps | Text / AdaLN cache hit |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1. A, initial waterfall | 94 / 94 | 0.001648 | 207.605254 | 2.884220 | 210.491576 | 7 / 42 | No / No |
| 2. A, immediate identical repeat | 94 / 94 | 0.000013 | 143.051405 | 2.243493 | 145.294933 | 7 / 42 | Yes / Yes |
| 3. B, distinct rowboat scene | 143 / 143 | 0.000010 | 178.027006 | 2.510664 | 180.540533 | 6 / 43 | No / Yes |
| 4. Long-A, detailed waterfall | 1662 / 1662 | 0.000027 | 159.605237 | 2.456659 | 162.065810 | 4 / 45 | No / Yes |
| 5. Original A restored | 94 / 94 | 0.000013 | 184.542096 | 2.493185 | 187.035336 | 7 / 42 | No / Yes |
| 6. FL2VA, original endpoints | 112 / 938 | 0.067861 | 188.119774 | 2.504224 | 190.695336 | 6 / 43 | No / Yes |
| 7. REF2VA, landscape image + audio | 62 / 7243 | 0.165078 | 470.203771 | 2.458956 | 472.832176 | 8 / 41 | — |
| 8. REF2VA, square image | 152 / 4256 | 0.017407 | 340.933282 | 2.569101 | 343.524426 | 8 / 41 | — |
| 9. Original landscape REF2VA restored | 62 / 7243 | 0.108665 | 473.323847 | 2.496047 | 475.931179 | 8 / 41 | — |

The corresponding logged family-stage durations are below, in seconds. Text includes endpoint conditioning for request 6 and text/vision for requests 7–9; the separate reference-encoding column covers their image/audio encoders. Stage durations include engine execution, and the logged family total is not interchangeable with the enclosing Generate API or request-wall boundaries.

| Request | Text / conditioning | Reference encode | AdaLN | Denoising | Video VAE | SR | Audio VAE | Family total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 38.204 | — | 9.823 | 111.256 | 39.454 | 5.083 | 3.751 | 207.605 |
| 2 | 0.000 | — | 0.000 | 99.001 | 36.021 | 5.361 | 2.628 | 143.048 |
| 3 | 43.559 | — | 0.000 | 90.625 | 36.090 | 5.107 | 2.615 | 178.026 |
| 4 | 39.614 | — | 0.000 | 76.442 | 36.013 | 4.913 | 2.585 | 159.605 |
| 5 | 40.130 | — | 0.000 | 100.915 | 35.925 | 4.927 | 2.607 | 184.538 |
| 6 | 43.424 | — | 0.000 | 101.075 | 35.905 | 4.997 | 2.626 | 188.070 |
| 7 | 58.328 | 4.354 | 10.067 | 351.458 | 38.207 | 4.958 | 2.657 | 470.083 |
| 8 | 50.413 | 2.180 | 10.297 | 231.965 | 38.116 | 5.224 | 2.597 | 340.867 |
| 9 | 67.934 | 3.752 | 10.223 | 345.410 | 38.213 | 4.931 | 2.686 | 473.206 |

All nine outputs passed full decode and metadata/finite-audio checks: 124 frames at 1296×720 and 24 fps, with 32 kHz stereo audio. Every request's scene-change diagnostic scored all 124 frames and flagged zero events; this is an inspection aid, not a visual or motion-quality pass. Sampled review of the first six confirmed that B changes to the requested rowboat/lake scene, the 1662-token prompt produces a coherent waterfall scene, restoring A returns to the original waterfall/rocks/pool composition, and FL2VA produces a coherent scene matching its endpoint-conditioned one-shot reference. The sequence exercises dynamic prompt sizes and T2VA→FL2VA→REF2VA/reference changes in one task; it does not qualify every supported prompt, reference combination or duration.

The immediate repeat's decoded video matches request 1 exactly (PSNR ∞, SSIM 1); decoded audio is not identical (float subtraction RMS −57.267037 dBFS). After the intervening different/long prompts, restored A has PSNR 28.071317 dB and SSIM 0.897538, with the original sampled scene preserved but local water/foliage differences; its audio subtraction RMS is −43.354522 dBFS. No full-motion playback or listening pass is claimed. This is not a deterministic-replay guarantee.

The FL2VA request has PSNR 41.970213 dB and SSIM 0.983255 against `memory-sr-fl2va-124`, with matching prompt, endpoints, seed, frame count and 6 full / 43 skipped steps. The sampled scene is preserved without obvious garbling or stale rowboat content. This is a cross-CLI quality comparison, not a timing pair: process lifetime and cache history differ. Listening and full-motion playback remain unperformed.

Request 7's landscape REF2VA output was directly compared with the same-source one-shot `memory-sr-ref2va-124`: all 124 decoded frames give PSNR **29.482559 dB** (minimum 26.722012 dB, maximum 35.925391 dB) and SSIM **0.886139**. Both twelve-frame contact sheets retain the waterfall, moss-covered rocks, foliage, pool and progressive camera push, with fine water/ripple differences but no obvious sampled garbling, scene replacement or abrupt layout jump. Prompt, image/audio, seed, frames, guidance and 8 full / 41 skipped steps match; process lifetime and cache history differ, so this is a quality comparison, not a fresh-process timing pair. Its finite stereo audio has 165888 decoded samples/channel; the aligned float difference from the one-shot has RMS −47.818782 dBFS and peak −23.048062 dBFS. The temporal diagnostic maximum is 0.944, with no flagged events. No listening or full-motion review is claimed.

Request 8's twelve-frame contact sheet and square input reference show the requested field scene: pink clothing, a dark lamb, grass and sheep remain consistent while the framing moves closer. No obvious sampled garbling, waterfall carry-over, replacement scene or abrupt composition jump was observed. The square input is reframed into the unchanged 1296×720 landscape output; pixel equality to that input is neither expected nor measured. Its audio is finite, but no listening was performed: requested sound content and absence of stale waterfall sound are not verified. This deliberate prompt/reference change is not an accuracy comparison against the waterfall output.

Request 9's matching twelve-frame contact sheets restore the waterfall, rocks, foliage, pool and progressive camera push seen in request 7, without sampled person/lamb/field leftovers, garbling or scene substitution. The direct comparison across all 124 decoded frames gives PSNR **33.448951 dB** (minimum 29.890545 dB, maximum 36.940942 dB) and SSIM **0.941632**. Prompt, image/audio, seed, frame count, 62 tokens / 7243 presentation rows and 8 full / 41 skipped steps match; the preceding resident state differs. Aligned float audio-difference RMS is **−48.240559 dBFS**, with peak −22.191081 dBFS and no NaN/Inf samples; the audio is not identical and has not been listened to. Fine visual details differ, and sampled stills do not establish full-motion continuity.

Prompt changes can alter both cache hits and FBC full/skip decisions. In particular, the long prompt ran only four full denoiser steps, so its shorter measured time is not evidence that longer prompts are inherently faster. The completed sequence validates execution, full media decoding and sampled scene checks for these nine exact requests; full-motion/listening qualification and the long-duration matrix must not be inferred from that completion.

## Measurement and quality rules

- **Fresh process / one-shot:** each CLI case launches a new executable process, loads one task and performs one request. Formal wall timing covers invocation through exit, including lazy engine loading, generation, audio/video encoding, ordinary logging, cache finalization and teardown. It excludes checkpoint download, engine building, preparation/copying of the per-case cache and post-run QA. The family's logged `total_ms` is the pipeline/generation column, not whole-process wall.
- **Seeded disk cache:** copy one immutable normal or SR seed into a new cache file for every matched case. Starting contents and seed recipe must match on both sides. A nonempty file or "cache loaded" log proves neither coverage of every requested shape nor an engine/conditioning-cache hit. Local `three-short-requests-v1` seeds are a defined new protocol, not a reconstruction of the undistributed historical seeds.
- **Empty RTX disk cache:** use a new absent cache path and label it separately. This does not mean disk caching is disabled, nor that OS file pages, driver state or the machine are cold. No file-cache purge or reboot is part of these experiments. Record order and relevant prior activity; do not manufacture a correction for its effect.
- **Resident:** one process and one `load_task` instance handle the full ordered sequence. Count `task_load_ms_once` once, even though receipts repeat it. It can exclude lazy engine creation, which remains in the relevant `generate_video_ms`. `prepare_ms` includes input decoding; `generate_video_ms` measures the public API; `write_mp4_ms` measures encoding; `request_wall_ms` spans preparation through encoding but excludes initial task loading, receipt writing and final process teardown. Measure whole-sequence process wall separately. Do not reset/cache-copy between requests, count the first request as a warmed repeat, or compare per-request wall directly with one-shot launch-to-exit wall.
- **Profiles:** full dynamic ranges remain available. Engine/profile capacity in a log is not necessarily the actual packed token count. Record actual shapes, prompt/media inputs, full/skip FBC counts, and stage timings when attributing a change.
- **Nsight:** traces are diagnostic. Profiled application wall times and trace collection/export time are excluded from formal timing tables. Do not publish raw traces, local paths, environment dumps, device identities, or raw kernel identifiers.
- **Concurrency:** run one GPU workload at a time; use the same runtime/SDK/driver and comparable system load for a pair. Record any deviation rather than applying an invented correction.
- **Media gate:** require exit 0 and a finished MP4, decode every frame, verify expected dimensions/count/fps and stereo audio metadata, inspect same-time frames across baseline/candidate, and listen to beginning/middle/end audio. Check motion/scene continuity and correspondence to references; a scalar similarity score alone is not a quality guarantee. Record any nonfinite samples, decoding failures, flashes, cuts, or changed content.
- **Code checks by revision:** the preceding bulk runtime compiled; 14 focused CPU tests and the dynamic input/output GPU regression passed. Its real RTX shared-arena test also passed with two contexts, nonzero activation requirements, and 8→512→8 shape transitions with verified softmax results. These generic GPU checks are evidence for that revision, not newly rerun tests of `aa594f60`. The current request-memory revision separately built and passed 14 CPU tests, including live-capacity/profile-bound cases; all twelve primary cases, all nine resident requests and both short T2VA empty-cache cases completed with full media checks. Scoped visual findings and remaining manual-review limits are recorded above. Two resident manifests (seven/nine requests, longest prompt 1662 tokens) previously passed native input validation; the nine-request manifest has now also executed in one task. On the current published revision, `python tools/legal_headers.py --check` found zero issues and `git diff --check 19e37595...HEAD` passed. No current-head remote premerge pass is claimed. None of these checks replaces comprehensive input coverage, full-motion review or listening qualification.

## Full-run Nsight findings

The exact PR #1240 runtime was profiled for normal T2VA, 124 output frames,
1344×768, seed 0, and FBC 0.3 with an existing RTX disk-cache seed. Its
689.358-second family pipeline is **diagnostic**, not an unprofiled benchmark.
Trace collection/export brought the enclosing command to 701.604 seconds.

| Baseline trace observation | Measured value | Interpretation |
| --- | ---: | --- |
| Union of GPU activity intervals | 506.397 s | Actual active timeline coverage, not a sum of overlapping APIs. |
| Denoiser GPU kernel time | 429.808 s | The denoiser kernel window is 430.131 s: little GPU idle time within this window. |
| Denoiser attention kernels | 270.831 s | The largest remaining compute component; runtime loading fixes do not remove this work. |
| Denoiser matrix / pointwise kernels | 103.923 / 55.054 s | Remaining denoiser kernel categories. |
| Video VAE kernels | 67.046 s | A material independent decode cost. |
| Startup / pre-AdaLN / pre-denoiser GPU-idle gaps | 81.243 / 21.267 / 31.457 s | Later load instrumentation isolated large plan-file reads as a major contributor. |
| Captured runtime-cache loads / stores | 108 / 36 | 0.750 / 0.156 s in the captured events; no JIT compile events were captured. |

Summed `cuKernelGetFunction` API durations were 408.314 seconds across concurrent
threads, but their interval union was only 91.979 seconds, of which 75.311 seconds
overlapped GPU work. It would be incorrect to describe this as 408 seconds of
serial JIT compilation or to conclude that the runner cache was disabled.

The loading changes therefore target observed GPU-idle portions without changing
attention mathematics. Dynamic sequence length still increases attention work;
long-video latency must be measured and cannot be extrapolated linearly from
the short-video result. Raw traces remain private because they contain local
paths and device identifiers; only these aggregated findings are distributable.

To reproduce diagnostic collection, use the same helper and cache condition
with `-Profile`. It creates a separate `-profiled` case directory, invokes the
native CLI under Nsight, and exports statistics after stopping the profile timer:

```powershell
Invoke-H3Case -Delivery normal -Mode t2va -Frames 120 -CacheCondition seeded -Profile
```

Profiled wall time includes profiler overhead and trace collection/export; it is
diagnostic and must not be entered as formal generation timing. The later statistics
export is outside that timer. Use the Nsight CLI matching the host installation. If stdout is buffered during
profiling, inspect the completed trace and process-stream records before deciding
that the target is stalled. Do not publish the generated raw reports unchanged.

### Long REF2VA: what the source does and does not establish

The normal and REF denoiser builders call the same transformer-block/rotary-position helpers. Both use fused QKV and BF16 native, non-decomposable TensorRT attention, without an attention mask; TF32 is disabled and the existing default-max workspace policy is unchanged. There is no source evidence of a separate decomposed or FP32 attention fallback in REF2VA.

For the recorded requests, 118,793 long-REF packed rows versus 104,060 long-T2VA rows imply about 1.30 times the attention's quadratic work. Comparing long REF with the 37,804-row short T2VA instead implies about 9.87 times that work. These ratios describe attention arithmetic, not measured E2E multipliers. They do not establish which factor dominates the unprofiled long REF run.

Two additional native-builder experiments remain unmeasured: raising the current builder optimization level from 1, and relaxing the REF fallback profile's `extra_memory_target=0.0`. Both require engine rebuilding and matched quality/timing checks; the latter can also increase weight/engine memory. They are not changes in the tested runtime candidate, and no speedup is attributed to them.

## Supported range versus this report's tested subset

The twelve-case matrix tests two output lengths, two canvas/delivery combinations (native 1344×768 and 864×480→1296×720 SR), and one prompt/media request per mode. Optional native 480p controls are additional cases, not part of the twelve. This does not validate every permitted prompt length, reference combination, aspect ratio, or duration.

- Request frame count is adjustable; output rounds up to `17*n + 5` within 124–345. Valid output counts are `124, 141, 158, 175, 192, 209, 226, 243, 260, 277, 294, 311, 328, 345`. Do not request 360 as a strict 15-second output.
- T2VA supports up to 2641 tokenizer tokens. FL2VA shares 2641 presentation rows between text and endpoint features/tags. REF2VA shares 262144 presentation rows between text and references. Prompt strings are tokenized per request; the reported short fixture is not a fixed prompt length.
- A normal bundle supports its finite native canvas resolver (aspect ratios 1:4 through 4:1 and 32-pixel rounding), including common 768×768, 1344×768, 768×1344, 960×544, 544×960, and 864×480 canvases. Arbitrary multiples of 32 are not automatically valid. Pass both dimensions or neither.
- SR fixes the base at 864×480 and output at 1296×720 for all modes; incompatible explicit dimensions are rejected. It is never automatically selected from resolution.
- FL2VA accepts first-only, last-only, or both endpoints. REF2VA accepts 1–12 ordered inputs: at most 9 images, 3 videos, and 3 explicit audio files, including audio-only requests. REF2VA image/video aspect ratios must be within 1:4–4:1. Video/audio inputs must be 2–15 seconds; each aggregate category of video, video soundtrack, and explicit audio duration is limited to 15 seconds. Keep reference tags aligned with ordered inputs. Endpoint inputs cannot be mixed with references.
- The fixed sampling configuration is 50 schedule points, guidance scale 1, and 24 fps; seed is adjustable. Negative prompts and supplied initial latents are unsupported. H3-Context-IR and H3-Regenerate-2K are not included.

## Delivery status and remaining limits

- The measured runtime is `aa594f6010bd59204c2edc3061fc6e5042572c0d`; documentation-only follow-up commits are not new runtime measurements. PR #1241 remains Draft pending review. All 12 primary configurations, nine resident requests and two short empty-cache checks completed; scoped quality-review status is recorded above.
- Each primary result is a single observation (`n=1`), not a latency distribution. Most original-PR baseline pairs were not measured; historical quality references and the CPU-build-confounded long bulk run do not supply those missing timing baselines. No unmeasured speedup is implied.
- Full media decode, metadata, finite-audio checks and sampled visual review are the completed media evidence. Full-motion playback and listening remain outside this report's completed review. The tested inputs do not exhaust supported prompts, references, durations or canvases, and similarity metrics are not a universal quality guarantee.
- Exact reference fixtures, resident manifests and source/bundle/seed provenance are retained in the authorized local evidence. Do not publicly upload unapproved fixtures, videos, raw receipts or traces. Historical RTX cache seeds are not distributed; the report documents an explicit seeding recipe rather than promising identical seed bytes.
- Undisclosed host and driver details remain private. The setup describes how to reproduce the workflow and cache condition, not a guarantee of identical absolute latency on arbitrary hardware or I/O systems.
- Formal one-shot timings are unprofiled launch-to-exit observations with the stated request, bundle and FBC 0.3 settings. Resident request boundaries and diagnostic trace/export timings are separate. Keep these boundaries distinct when quoting results or attaching approved media and concise QA findings.
