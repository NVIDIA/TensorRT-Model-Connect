---
title: MiniMax H3
description: Native text, endpoint, and reference-conditioned video with audio and optional super resolution.
---

MiniMax H3 uses the ModelConnect C++ runtime and TensorRT-RTX. Python and
PyTorch are build-time dependencies. On Windows, the CLI decodes reference
media and writes H.264/AAC MP4 files through Media Foundation.

## Capabilities and inputs

| Capability | Required inputs | Bundle option |
| --- | --- | --- |
| T2VA | Nonempty UTF-8 prompt | Included |
| FL2VA | Prompt and a first frame, last frame, or both | Included |
| Ref2VA | Prompt and ordered image, video, or audio references | Included |
| Super resolution (all three modes) | Fixed `480x864` base, 124--345 frames | `super_resolution=true` |

Every mode generates synchronized video and audio. First/last frames cannot
be combined with reference inputs. H3-Context-IR and H3-Regenerate-2K are not
included.

- **Duration:** `--num-frames` is adjustable for every request. At 24 fps, H3
  rounds up to `17 * n + 5`. Supported output counts are `124, 141, 158, 175,
  192, 209, 226, 243, 260, 277, 294, 311, 328, 345`. A request for 120 frames
  produces 124 frames (5.167 seconds); use 345 for the longest output
  (14.375 seconds), not 360. The default is 124 frames, including the compact `480x864` canvas.
- **Canvas:** provide both `--height` and `--width`, or neither. T2VA and
  Ref2VA default to `768x1344`; FL2VA derives a canvas from the first supplied
  endpoint. Common canvases are `768x768`, `768x1344`, `1344x768`, `544x960`,
  and `960x544`. The finite native canvas set follows the public aspect-ratio
  resolver (1:4 through 4:1, rounded to 32-pixel axes), plus `544x960`,
  `960x544`, and the compact `480x864` canvas. Arbitrary multiples of 32 are
  not necessarily supported.
- **Prompt:** tokenized per request, with no fixed prompt length. T2VA accepts
  up to 2641 tokenizer tokens. FL2VA shares 2641 presentation rows between
  prompt tokens and endpoint image tags/features. Ref2VA shares 262144
  presentation rows between prompt and reference media.
- **References:** Ref2VA accepts 1--12 ordered files: at most 9 images,
  3 videos, and 3 explicit audio files. Audio-only input is supported. Each
  video/audio input must be 2--15 seconds; aggregate durations of reference
  videos, video soundtracks, and explicit audio are each limited to 15 seconds.
  Visual references must have an aspect ratio between 1:4 and 4:1.
- **Sampling:** 50 schedule points, guidance scale 1, and 24 fps are fixed.
  The seed is adjustable. Negative prompts and supplied initial latents are
  not supported.
- **Super resolution:** explicitly enabled at build time for all three modes.
  An SR bundle always generates at `480x864` and outputs `720x1296`; a normal
  bundle never applies SR. Select the bundle, not a different runtime workflow.

## Windows setup

Use an x64 Visual Studio 2022 developer PowerShell with Desktop development
with C++, Git, CMake, Ninja, CPython 3.12 or newer, CUDA 12.9, and
TensorRT-RTX 1.6.1.120. Use the runtime DLL and matching `win_amd64` Python
wheel from the same TensorRT-RTX SDK. Allow substantial disk space for the
checkpoint and staged plans.

Set the following paths to your checkout, SDKs, and artifact directory:

```powershell
$RepoRoot = (Resolve-Path '<ModelConnect-checkout>').Path
$CudaRoot = '<CUDA-root>'
$RtxRoot = '<TensorRT-RTX-root>'
$ArtifactRoot = (New-Item -ItemType Directory -Force '<artifact-directory>').FullName
$BuildRoot = Join-Path $ArtifactRoot 'build'
$InstallRoot = Join-Path $ArtifactRoot 'install'
$OutputRoot = (New-Item -ItemType Directory -Force (Join-Path $ArtifactRoot 'outputs')).FullName
$env:PATH = @("$RtxRoot/bin", "$RtxRoot/lib", "$CudaRoot/bin", $env:PATH) -join [IO.Path]::PathSeparator
```

Install the required header-only CMake dependency:

```powershell
$JsonRoot = Join-Path $ArtifactRoot 'json'
$JsonInstall = Join-Path $ArtifactRoot 'dependencies'
git clone --depth 1 --branch v3.11.3 https://github.com/nlohmann/json.git $JsonRoot
cmake -S $JsonRoot -B "$JsonRoot/build" -DJSON_BuildTests=OFF "-DCMAKE_INSTALL_PREFIX=$JsonInstall"
cmake --install "$JsonRoot/build"
```

Build and install only the CLI, RTX backend, and H3 family. If your SDK uses
`lib/x64` or keeps its DLL in `lib`, adjust the library/runtime directory
arguments to match:

```powershell
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
```

Install build-only Python dependencies. `--no-deps` keeps the selected RTX
wheel from being replaced by the standard TensorRT package:

```powershell
$PythonTag = python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')"
$RtxWheels = @(Get-ChildItem -LiteralPath (Join-Path $RtxRoot 'python') `
    -Filter "tensorrt_rtx-*-$PythonTag-none-win_amd64.whl" -File)
if ($RtxWheels.Count -ne 1) { throw "Expected exactly one TensorRT-RTX wheel for $PythonTag" }
python -m pip install $RtxWheels[0].FullName
python -m pip install -r (Join-Path $RepoRoot 'families/minimax_h3/requirements.txt')
python -m pip install 'torch>=2.6' 'safetensors>=0.4' 'numpy>=1.24' 'ml_dtypes>=0.4' `
    'onnx>=1.16' 'huggingface_hub>=0.23' 'sentencepiece>=0.1.99' `
    'cuda-python>=13.0.3,<14' 'apache-tvm-ffi==0.1.12' 'PyYAML>=6.0'
python -m pip install --no-deps -e $RepoRoot -C py-only=true
```

## Build: normal or super resolution

Both builds include **T2VA, FL2VA, and REF2VA**, dynamic prompt lengths, and
the same full Comfy INT8 denoisers plus NVFP4 AWQ text-encoder checkpoint.
There is no BF16-versus-quantized workflow selection and no separate REF2VA
build. ModelConnect selects the correct denoiser from the request's inputs.

First download the original model's small configuration/tokenizer files and
its video/audio VAEs. Original BF16 denoiser and text-encoder weights are not
needed:

```powershell
$Checkpoint = Join-Path $ArtifactRoot 'checkpoint'
$Patterns = @('model_index.json', 'modular_model_index.json', 'processor/**', `
    'scheduler/**', 'audio_scheduler/**', 'tokenizer/**', `
    'transformer/config.json', 'vae/**', 'audio_vae/**')
python -c "from huggingface_hub import snapshot_download; import sys; snapshot_download('MiniMaxAI/MiniMax-H3', revision='48d93ede732756e404a3b1b2f3b3a9b5a22f6cfc', local_dir=sys.argv[1], allow_patterns=sys.argv[2:])" $Checkpoint $Patterns
```

**Normal generation (default):**

```powershell
$Bundle = Join-Path $ArtifactRoot 'MiniMax-H3.bundle'
python -m tensorrt_model_connect build $Checkpoint `
    --backend trt_rtx --precision bf16 --output $Bundle
```

**Super resolution:** use the same command with one additional option and a
different output path:

```powershell
$Bundle = Join-Path $ArtifactRoot 'MiniMax-H3-SR.bundle'
python -m tensorrt_model_connect build $Checkpoint `
    --backend trt_rtx --precision bf16 --output $Bundle `
    --set minimax_h3.super_resolution=true
```

The SR option fixes generation to **width 864 × height 480**, then produces
**width 1296 × height 720** for all three input modes. Omitted dimensions use
that base; any other explicit generation dimensions are rejected. Original
reference inputs keep their normal conditioning preprocessing. A normal
bundle never enables SR, even when asked to generate at 864×480.

Changing the generation canvas can change composition even with the same
prompt and seed. SR postprocesses the generated 480p frames; it does not
promise a pixel-matched version of a native 768p generation.

At build time, missing quantized files are downloaded from
[Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/4cc1d817b6184899b41293954329f576cb5ae86b),
pinned to revision `4cc1d817b6184899b41293954329f576cb5ae86b`:

- `diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors` (T2VA/FL2VA).
- `diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors` (REF2VA).
- `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` (shared text/vision encoder).

These are the full checkpoints, not the pruned versions. The NVFP4 file
explicitly specifies full-precision matrix multiplication: the builder
decodes its weights to BF16 and preserves its AWQ input scales, matching
that checkpoint's semantics. `--precision bf16` describes the residual/text
compute; it does not select original BF16 source checkpoints or promise
FP4 matrix multiplication. INT8 denoiser linear layers use native TRT
quantization operations. No ComfyUI installation is required.

The SR build also downloads the fixed
[Real-ESRGAN compact checkpoint pair](https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.2.5.0)
and builds its native TRT plan. Audio and frame count are unchanged by SR.

For offline builds, pre-download the three pinned HF files into
`$Checkpoint` with their listed paths and HF download metadata. Advanced
file-location overrides are `minimax_h3.quantized_transformer`,
`minimax_h3.quantized_ref_transformer`, and
`minimax_h3.quantized_text_encoder`; they select locations, not alternate
model recipes. SR file overrides require `super_resolution=true`.

FirstBlockCache defaults to threshold `0.08` for both base and REF denoisers.
The build options `minimax_h3.first_block_cache_threshold` and
`minimax_h3.ref2va_first_block_cache_threshold` remain available for tuning.
Caching is approximate and may change the result. No timing measured on an
older BF16 bundle is a performance guarantee for this delivery.

To reproduce the `0.3` validation setting for all three modes, explicitly
set both thresholds while building a new bundle:

```powershell
$Bundle = Join-Path $ArtifactRoot 'MiniMax-H3-FBC03.bundle'
python -m tensorrt_model_connect build $Checkpoint `
    --backend trt_rtx --precision bf16 --output $Bundle `
    --set minimax_h3.first_block_cache_threshold=0.3 `
    --set minimax_h3.ref2va_first_block_cache_threshold=0.3
```

For SR validation, add `--set minimax_h3.super_resolution=true` and use a
different output path, such as `MiniMax-H3-SR-FBC03.bundle`. These are build
options, not runtime CLI flags; the product default remains `0.08`. Use the
same prompt, seed, input media, frame count, and canvas when comparing cache
settings. A higher threshold can skip more work and alter the generated
video or audio.

Plans are staged on disk and compatible completed stages can be reused after
an interrupted build. Changing source weights or profiles requires a fresh
staging directory/output path. This delivery broadens the dynamic profiles
to include 124-frame 480p generation; old plans must be rebuilt. Old
resolution-triggered SR bundles are rejected with a rebuild message.

The final bundle needs no external checkpoints, Python, PyTorch, or ComfyUI
at runtime. The weights remain subject to the
[MiniMax-H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE).

## CLI

Use the installed native CLI and runtime directory:

```powershell
$RuntimeRoot = Join-Path $InstallRoot 'bin'
$Trtmc = Join-Path $RuntimeRoot 'trtmc.exe'
$RuntimeCache = Join-Path $ArtifactRoot 'minimax-h3.rtxcache'
```

T2VA, nominal five seconds; change `--num-frames` to 345 for the longest output:

```powershell
& $Trtmc generate-video $Bundle --runtime-root $RuntimeRoot `
    --prompt 'A cinematic sunrise over a mountain lake with synchronized ambience.' `
    --num-frames 120 --seed 0 `
    --runtime-cache $RuntimeCache --output (Join-Path $OutputRoot 't2va.mp4')
```

FL2VA; omit either endpoint flag for first-only or last-only generation:

```powershell
& $Trtmc generate-video $Bundle --runtime-root $RuntimeRoot `
    --prompt 'Continue naturally between the supplied endpoints.' `
    --first-frame .\first.png --last-frame .\last.png --num-frames 120 --seed 7 `
    --runtime-cache $RuntimeCache --output (Join-Path $OutputRoot 'fl2va.mp4')
```

Ref2VA; repeat `--reference-image`, `--reference-video`, and `--reference-audio`
in the desired order. Each modality uses its own numbered prompt tags. A video
with sound also contributes an `<Audio N>` tag:

```powershell
& $Trtmc generate-video $Bundle --runtime-root $RuntimeRoot `
    --prompt 'Use <Picture 1> as the subject and <Audio 1> as the voice reference.' `
    --reference-image .\subject.png --reference-audio .\voice.wav `
    --num-frames 120 --seed 11 `
    --runtime-cache $RuntimeCache --output (Join-Path $OutputRoot 'ref2va.mp4')
```

For SR, point `$Bundle` at `MiniMax-H3-SR.bundle` and use the same commands
above. Omit dimensions, or specify `--height 480 --width 864`; output is
`720x1296` for T2VA, FL2VA, and REF2VA. Frame count and prompt length remain
dynamic. Normal generation at 480p stays 480p and does not imply SR.

## C++ API

The installed package exposes the same task interface used by the CLI. Save
this as `CMakeLists.txt` alongside `main.cpp`:

```cmake
cmake_minimum_required(VERSION 3.24)
project(h3_consumer LANGUAGES CXX)
set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreaded)
find_package(trtmc CONFIG REQUIRED)
add_executable(h3_consumer main.cpp)
target_link_libraries(h3_consumer PRIVATE trtmc::trtmc_runtime)
target_compile_features(h3_consumer PRIVATE cxx_std_17)
```

This complete `main.cpp` generates a T2VA result in memory:

```cpp
#include <trtmc/runtime/family_loader.h>
#include <trtmc/task.h>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: h3_consumer <bundle> <runtime-directory> <runtime-cache>\n";
        return 1;
    }
    auto task = trtmc::load_task(argv[1], argv[2], 0, argv[3], false);
    auto* video = dynamic_cast<trtmc::IVideoGeneration*>(task.get());
    if (!video) throw std::runtime_error("bundle does not provide video generation");
    trtmc::VideoGenerationRequest request;
    request.mode = trtmc::VideoGenerationMode::kTextToVideoAudio;
    request.prompt = "A sunrise over a mountain lake with synchronized ambience.";
    request.config.video_num_frames = 120;
    // Leave dimensions unset: normal defaults or the SR bundle's fixed base.
    request.config.num_steps = 50;
    request.config.guidance_scale = 1.0F;
    request.config.seed = 0;
    const auto result = video->generate_video(request);
    std::cout << result.frames.num_frames << " frames at " << result.fps
              << " fps; " << result.audio.channels << " audio channels\n";
}
```

Configure and run from the consumer directory:

```powershell
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release `
    "-DCMAKE_PREFIX_PATH=$InstallRoot" "-DCUDAToolkit_ROOT=$CudaRoot"
cmake --build build
$env:PATH = @($RuntimeRoot, $env:PATH) -join [IO.Path]::PathSeparator
& .\build\h3_consumer.exe $Bundle $RuntimeRoot $RuntimeCache
```

For conditioning, populate decoded host-resident inputs before calling
`generate_video(request)`:

| Mode | Request fields |
| --- | --- |
| `kFirstLastFrameToVideoAudio` | `first_frame`, `last_frame`, or both: `VideoImageInput` with HWC float pixels in `[0, 1]` and matching height/width/channels |
| `kReferenceToVideoAudio` | Ordered `references`: `VideoReferenceInput` with `kind` and its matching `image`, `video`, or `audio` field |
| SR (any of the three modes) | Load the SR bundle; omit dimensions or set height 480 and width 864 |

`VideoClipInput` carries THWC pixels, frame count, rational frame rate, and an
optional soundtrack. `AudioResult` carries interleaved samples, sample rate,
channel count, and total interleaved sample count. Keep reference order consistent with
prompt tags. Applications own decoding and output-container writing;
`VideoResult` owns the generated THWC RGB frames and interleaved audio samples.
Keep the loaded task alive to reuse it for subsequent requests.
