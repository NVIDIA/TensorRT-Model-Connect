---
title: Native Windows MiniMax H3
description: Build and run T2VA, FL2VA, Ref2VA, and optional super resolution.
---

MiniMax H3 runs through the ModelConnect C++ runtime and TensorRT-RTX. Python
and PyTorch are build-time tools only; video generation does not start Python,
PyTorch, FFmpeg, or another subprocess. On Windows, the CLI uses Media
Foundation for reference media and H.264/AAC MP4 output.

One bundle can contain all of the following:

- T2VA from a prompt;
- FL2VA from a prompt plus a first frame, last frame, or both;
- Ref2VA from a prompt plus ordered image, video, and audio references; and
- optional Real-ESRGAN compact super resolution for T2VA generated at
  `480x864`, producing `720x1296` output.

The same dynamic H3 implementation supports the public 5--15 second duration
range. H3 aligns frame counts to `17 * n + 5` at 24 fps, so a 120-frame request
produces 124 frames (5.167 seconds), and a 345-frame request produces 345
frames (14.375 seconds). Prompts are tokenized for every request.

H3-Context-IR and H3-Regenerate-2K are separate services and are not included.

## Prerequisites

Use an x64 Visual Studio 2022 developer PowerShell with:

- Visual Studio 2022 Desktop development with C++;
- Git, CMake, and Ninja;
- 64-bit CPython 3.12 or newer;
- CUDA 12.9; and
- TensorRT-RTX 1.6.1.120, including the `win_amd64` Python wheel that matches
  the selected CPython version.

Keep the TensorRT-RTX Python wheel and runtime DLL from the same SDK package.
The MiniMax checkpoint is large, and building all optional plans requires
substantial free disk space.

Set these paths once:

```powershell
$RepoRoot = (Resolve-Path '<ModelConnect-checkout>').Path
$CudaRoot = '<CUDA-root>'
$RtxRoot = '<TensorRT-RTX-root>'
$ArtifactRoot = (New-Item -ItemType Directory -Force `
    '<artifact-directory>').FullName
$BuildRoot = Join-Path $ArtifactRoot 'build'
$InstallRoot = Join-Path $ArtifactRoot 'install'
$OutputRoot = (New-Item -ItemType Directory -Force `
    (Join-Path $ArtifactRoot 'outputs')).FullName
```

## Build and install the native runtime

The family-owned helper builds the runtime, TensorRT-RTX backend, CLI, and
MiniMax H3 plugin:

```powershell
& (Join-Path $RepoRoot 'families\minimax_h3\runtime\build_windows.ps1') `
    -CudaRoot $CudaRoot `
    -TensorRtRtxRoot $RtxRoot `
    -BuildDirectory $BuildRoot

cmake --install $BuildRoot --prefix $InstallRoot --config Release
```

Pass `-BuildTests` to the helper when developing the integration.

## Install the build-only Python environment

```powershell
$env:PATH = @(
    (Join-Path $RtxRoot 'bin'),
    (Join-Path $RtxRoot 'lib'),
    (Join-Path $CudaRoot 'bin'),
    $env:PATH
) -join [IO.Path]::PathSeparator

$PythonTag = python -c `
    "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')"
$RtxWheels = @(Get-ChildItem -LiteralPath (Join-Path $RtxRoot 'python') `
    -Filter "tensorrt_rtx-*-$PythonTag-none-win_amd64.whl" -File)
if ($RtxWheels.Count -ne 1) {
    throw "Expected exactly one TensorRT-RTX wheel for $PythonTag"
}

python -m pip install $RtxWheels[0].FullName
python -m pip install -r `
    (Join-Path $RepoRoot 'families\minimax_h3\requirements.txt')
python -m pip install `
    'torch>=2.6' 'safetensors>=0.4' 'numpy>=1.24' 'ml_dtypes>=0.4' `
    'onnx>=1.16' 'huggingface_hub>=0.23' 'sentencepiece>=0.1.99' `
    'cuda-python>=13.0.3,<14' 'apache-tvm-ffi==0.1.12' 'PyYAML>=6.0'
python -m pip install --no-deps -e $RepoRoot -C py-only=true
```

`--no-deps` deliberately avoids installing the standard TensorRT Python
package over the TensorRT-RTX wheel selected above.

## Download the checkpoints

Authenticate with Hugging Face if the repository asks for it, then download
the pinned public H3 snapshot. The allowlist avoids duplicate source layouts
that are not consumed by this builder.

```powershell
$H3Revision = '48d93ede732756e404a3b1b2f3b3a9b5a22f6cfc'
$CheckpointRoot = Join-Path $ArtifactRoot 'checkpoint'
$RootPatterns = @(
    'model_index.json', 'modular_model_index.json',
    'processor/**', 'scheduler/**', 'audio_scheduler/**',
    'text_encoder/**', 'tokenizer/**', 'transformer/**',
    'transformer_ref/**', 'vae/**', 'audio_vae/**'
)
$Checkpoint = (python -c `
    "from huggingface_hub import snapshot_download; import sys; print(snapshot_download('MiniMaxAI/MiniMax-H3', revision=sys.argv[2], local_dir=sys.argv[1], allow_patterns=sys.argv[3:]))" `
    $CheckpointRoot $H3Revision $RootPatterns).Trim()
$TransformerRef = Join-Path $Checkpoint 'transformer_ref'
```

The optional public ConvRot INT8 transformer reduces the transformer footprint
without changing the runtime API:

```powershell
$QuantRevision = '4cc1d817b6184899b41293954329f576cb5ae86b'
$QuantRoot = Join-Path $ArtifactRoot 'quantized-checkpoint'
$Quant = (python -c `
    "from huggingface_hub import hf_hub_download; import sys; print(hf_hub_download('Comfy-Org/MiniMax-H3', filename='diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors', revision=sys.argv[2], local_dir=sys.argv[1]))" `
    $QuantRoot $QuantRevision).Trim()
```

To include the optional `480x864` to `720x1296` super-resolution plan,
download the two public Real-ESRGAN compact checkpoints:

```powershell
$SuperResolutionRoot = New-Item -ItemType Directory -Force `
    (Join-Path $ArtifactRoot 'super-resolution')
$SuperResolutionModel = Join-Path $SuperResolutionRoot 'realesr-general-x4v3.pth'
$SuperResolutionWeakModel = `
    Join-Path $SuperResolutionRoot 'realesr-general-wdn-x4v3.pth'

Invoke-WebRequest `
    -Uri 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth' `
    -OutFile $SuperResolutionModel
Invoke-WebRequest `
    -Uri 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth' `
    -OutFile $SuperResolutionWeakModel
```

## Build one bundle with every workflow

```powershell
$Bundle = Join-Path $ArtifactRoot 'MiniMax-H3.bundle'

python -m tensorrt_model_connect build $Checkpoint `
    --backend trt_rtx --precision bf16 `
    --output $Bundle `
    --set "minimax_h3.transformer_ref=$TransformerRef" `
    --set "minimax_h3.quantized_transformer=$Quant" `
    --set "minimax_h3.super_resolution_model=$SuperResolutionModel" `
    --set "minimax_h3.super_resolution_weak_model=$SuperResolutionWeakModel"
```

Plans are staged on disk as they complete. Repeating an interrupted build
reuses completed compatible stages. The final bundle contains the native plans
and runtime metadata; checkpoint files, Python, and PyTorch are not runtime
dependencies.

To use BF16 transformer weights, omit `quantized_transformer`. To exclude
Ref2VA or super resolution, omit the corresponding option and checkpoint files.
T2VA and FL2VA remain available from the same bundle.

The checkpoint is covered by the
[MiniMax-H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)
in addition to this repository's license.

## Generate MP4 video from the CLI

Use the installed runtime directory explicitly:

```powershell
$RuntimeRoot = Join-Path $InstallRoot 'bin'
$Trtmc = Join-Path $RuntimeRoot 'trtmc.exe'
$RuntimeCache = Join-Path $ArtifactRoot 'minimax-h3.rtxcache'
```

T2VA, nominal five seconds:

```powershell
& $Trtmc generate-video $Bundle `
    --runtime-root $RuntimeRoot `
    --prompt 'A cinematic sunrise over a mountain lake with synchronized ambience.' `
    --num-frames 120 --height 768 --width 1344 `
    --num-steps 50 --guidance-scale 1 --seed 0 `
    --runtime-cache $RuntimeCache `
    --output (Join-Path $OutputRoot 't2va-5s.mp4')
```

T2VA with the optional super-resolution plan:

```powershell
& $Trtmc generate-video $Bundle `
    --runtime-root $RuntimeRoot `
    --prompt 'A continuous documentary shot with synchronized dialogue and ambience.' `
    --num-frames 345 --height 480 --width 864 `
    --num-steps 50 --guidance-scale 1 --seed 0 `
    --runtime-cache $RuntimeCache `
    --output (Join-Path $OutputRoot 't2va-720p.mp4')
```

When the bundle contains the super-resolution plan, a T2VA request at exactly
`480x864` is upscaled automatically to `720x1296`. The generated audio track is
kept unchanged.

FL2VA:

```powershell
& $Trtmc generate-video $Bundle `
    --runtime-root $RuntimeRoot `
    --prompt 'Continue naturally between the supplied endpoints.' `
    --first-frame .\first.png --last-frame .\last.png `
    --num-frames 120 --seed 7 `
    --runtime-cache $RuntimeCache `
    --output (Join-Path $OutputRoot 'fl2va.mp4')
```

Ref2VA preserves the order of reference flags:

```powershell
& $Trtmc generate-video $Bundle `
    --runtime-root $RuntimeRoot `
    --prompt 'Use <Picture 1> as the subject and <Audio 1> as the voice reference.' `
    --reference-image .\subject.png `
    --reference-audio .\voice.wav `
    --num-frames 120 --height 768 --width 1344 --seed 11 `
    --runtime-cache $RuntimeCache `
    --output (Join-Path $OutputRoot 'ref2va.mp4')
```

Reference videos and explicit audio references must be 2--15 seconds. Ref2VA
accepts at most 9 images, 3 videos, 3 explicit audio files, and 12 files total.
Audio may be the only reference. Prompt and dialogue text is UTF-8.

## Consume the bundle from C++

Link the installed runtime package:

```cmake
cmake_minimum_required(VERSION 3.24)
project(h3_consumer LANGUAGES CXX)

find_package(trtmc CONFIG REQUIRED)
add_executable(h3_consumer main.cpp)
target_link_libraries(h3_consumer PRIVATE trtmc::trtmc_runtime)
target_compile_features(h3_consumer PRIVATE cxx_std_17)
```

Configure the consumer with `-DCMAKE_PREFIX_PATH=<install-root>` and keep the
installed runtime directory available while the application runs.

```cpp
#include <trtmc/runtime/family_loader.h>
#include <trtmc/task.h>

#include <stdexcept>
#include <utility>

trtmc::VideoImageInput decode_image(const char* path);
trtmc::AudioResult decode_audio(const char* path);

int main() {
    auto task = trtmc::load_task(
        "MiniMax-H3.bundle", "<install-root>\\bin", 0,
        "minimax-h3.rtxcache", false);
    auto* video = dynamic_cast<trtmc::IVideoGeneration*>(task.get());
    if (video == nullptr)
        throw std::runtime_error("bundle does not provide video generation");

    trtmc::VideoGenerationRequest request;
    request.mode = trtmc::VideoGenerationMode::kFirstLastFrameToVideoAudio;
    request.prompt = "Continue naturally from the supplied first frame.";
    request.first_frame = decode_image("first.png");
    request.config.video_num_frames = 120;
    request.config.height = 768;
    request.config.width = 1344;
    request.config.num_steps = 50;
    request.config.guidance_scale = 1.0F;
    request.config.seed = 7;
    trtmc::VideoResult result = video->generate_video(request);

    trtmc::VideoGenerationRequest reference_request;
    reference_request.mode = trtmc::VideoGenerationMode::kReferenceToVideoAudio;
    reference_request.prompt =
        "Use <Picture 1> as the subject and <Audio 1> as the voice reference.";
    reference_request.config = request.config;

    trtmc::VideoReferenceInput image;
    image.kind = trtmc::VideoReferenceKind::kImage;
    image.image = decode_image("subject.png");
    reference_request.references.push_back(std::move(image));

    trtmc::VideoReferenceInput audio;
    audio.kind = trtmc::VideoReferenceKind::kAudio;
    audio.audio = decode_audio("voice.wav");
    reference_request.references.push_back(std::move(audio));
    trtmc::VideoResult reference_result =
        video->generate_video(reference_request);
}
```

Keep `references` in prompt order. `VideoResult` owns contiguous THWC RGB frames
and interleaved audio samples; applications may write them with their preferred
container implementation. The CLI uses the same C++ task interface and adds
Windows-native media decoding and MP4 muxing.
