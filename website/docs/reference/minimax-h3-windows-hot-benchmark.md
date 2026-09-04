---
title: MiniMax H3 Windows native benchmark
description: Measure original-weight dense H3 video-and-audio generation in the native ModelConnect TensorRT-RTX runtime.
---

This contract measures the public `generate_video()` call inside the installed
ModelConnect C++ runtime. The generation process does not load or invoke Python,
PyTorch, FastVideo, Triton, FFmpeg, a sidecar, or a subprocess. `trtmc.exe`
keeps one pipeline alive for the warmup and measured request and writes the
measured result as an MP4 through Windows Media Foundation.

## Timing boundary

`--warmup 1 --benchmark 1` performs one unmeasured request followed by one
measured request in the same process. The timer begins immediately before
`IPipeline::generate_video(request)` and ends when it returns synchronized host
video and audio.

The reported `generation_ms` includes native conditioning, denoising, video VAE
decode, audio VAE decode, device-to-host copies, and host result assembly. It
excludes bundle and pipeline loading, the warmup, MP4 encoding, and file output.
The CLI reports those outer costs separately as `load_ms`, `media_write_ms`, and
`total_ms`. Do not report `total_ms` as the hot single-generation latency.

The qualified path uses a persistent TensorRT-RTX runtime cache and retains the
hot engines inside the CLI process. The warmup populates the process-local
text/AdaLN caches and retained-engine map. Each request still creates fresh
execution contexts, and the staged runtime destroys the denoiser contexts
before VAE decode to bound memory use.

## Original-weight dense engine boundary

The qualified bundle contains the official original H3 BF16 weights and uses
TensorRT-RTX dense attention. It contains no FastH3 adapter, LoRA, VSA engine,
VSA cubin/PTX, or external attention runtime.

The hot visual path uses the same six plan files for short and long requests:

- `text_encoder.plan`;
- `adaln_precompute.plan`;
- `denoiser_head.plan`;
- `denoiser_tail.plan`;
- `denoiser_finish.plan`; and
- `vae_tile_decoder.plan`.

T2VA additionally invokes `audio_vae_decoder.plan`. The denoiser head, tail,
and finish engines each contain two TensorRT optimization profiles:

- profile 0 exactly specializes the 537-token, 124-output-frame, 1344x768
  qualification tuple; and
- profile 1 covers the public dynamic envelope of 1--2641 prompt tokens,
  124--345 output frames, and the supported canvases.

The runtime selects the profile for each request. This is one implementation
and one engine set, not separate five-second and fifteen-second models. An exact
qualification request must print:

```text
[minimax-h3] denoiser optimization_profile=0/2 packed_rows=38247
```

The dense scheduler executes 49 transformer forwards. Head and finish execute
on every forward. FirstBlockCache may reuse the tail residual on interior
forwards; the first and final tail evaluations are unconditional, and a
non-finite cache metric also forces a full tail evaluation. Head, tail, and
finish execute serially on one CUDA stream and share one TensorRT-RTX
user-managed activation arena sized for the selected profile.

## Five-second T2VA workload

Use the exact bundle, runtime revision, checkpoint, TensorRT-RTX SDK, driver,
prompt, geometry, seed, and runtime settings in every reproduction. Close other
GPU and unified-memory workloads before starting.

```powershell
$InstallRoot = Join-Path $env:LOCALAPPDATA 'Programs\ModelConnect\MiniMax-H3'
$Trtmc = Join-Path $InstallRoot 'bin\trtmc.exe'
$Bundle = Join-Path $InstallRoot 'models\MiniMax-H3.bundle'
$RepoRoot = (Resolve-Path '<ModelConnect-checkout>').Path
$Prompt = (Get-Content -Raw `
    (Join-Path $RepoRoot `
        'tests\e2e\models\minimax_h3\prompts\t2va-example-1.json') |
    ConvertFrom-Json).prompt
$RuntimeCache = Join-Path (Get-Location) 'minimax-h3-dense-fbc.rtxcache'
$FiveSecondLog = Join-Path (Get-Location) 'minimax-h3-t2va-124f.log'

& $Trtmc generate-video $Bundle `
    --prompt $Prompt `
    --num-frames 120 --height 768 --width 1344 --seed 0 `
    --num-inference-steps 50 --guidance-scale 1 `
    --runtime-cache $RuntimeCache `
    --set minimax_h3.retain_engines=true `
    --set minimax_h3.retained_tail_weight_budget_gib=24 `
    --set minimax_h3.first_block_cache_threshold=0.30 `
    --warmup 1 --benchmark 1 `
    --output .\minimax-h3-t2va-124f.mp4 `
    2>&1 | Tee-Object -FilePath $FiveSecondLog
if ($LASTEXITCODE -ne 0) { throw "MiniMax-H3 124-frame benchmark failed" }
```

The fixture is exactly 537 tokenizer tokens, so the request routes to profile
0. The nominal 120 frames align to 124 output frames, or 5.167 seconds at 24
fps. The acceptance ceiling is 555,000 ms (9:15) for the hot measured sample on
the qualified Spark hardware/software cohort. This is a qualification threshold,
not a portable latency guarantee or a cold CLI-startup measurement.

The `0.30` FirstBlockCache threshold is a measured delivery preset, not the
bundle default or a universal quality/performance constant. On this exact
fixture it produces six full tail evaluations and 43 cached tail reuses. A
different prompt or profile may produce a different cache-decision sequence.

## Current original-weight dense result

The qualification candidate uses the exact source tree published by signed
revision `6e5285d81e596d5fe7064fafd125e10475a2e512`. The run used an NVIDIA RTX
Spark N1X with a 63,424 MiB CUDA aperture, driver 616.67, CUDA 12.9.1, and
TensorRT-RTX 1.6.1.

| Request | Profile | Independent measured samples | Dense schedule | Output |
| --- | --- | ---: | --- | --- |
| 537 tokens, 120 nominal frames, 1344x768 | `0/2` | 542,101.711 ms (9:02.102); 546,701.314 ms (9:06.701) | 49 forwards, 6 full / 43 reused tails | 124 frames / 5.167 s, synchronized audio |

Both independent `1 + 1` runs passed the native finite-value validation for RGB
and audio and remained below the 555-second acceptance ceiling. The written MP4
contains 1344x768 H.264 video at 24 fps and AAC stereo audio at
32 kHz. The container reports a 5.166625-second video track and a 5.184-second
audio track; the small difference is normal AAC framing. Representative frames
were inspected and showed no checkerboard corruption or the severe accuracy
failure seen in the discarded approximate-weight experiment.

In the faster sample, the measured component totals were 481,797.980 ms for
denoising, 58,021.071 ms for video VAE decode, and 2,199.365 ms for audio VAE
decode. The measured request reused the text and AdaLN results from warmup.
Raising the retained-tail
weight budget above 24 GiB is not part of this contract: on this 63,424 MiB
CUDA aperture it increased memory pressure and did not improve the measured
latency.

## Verify the qualified hot path

The `1 + 1` command must record five retained-engine fills during warmup and
five retained-engine hits during the measured request: denoiser head, tail,
finish, video VAE, and audio VAE. Text and AdaLN reuse are reported separately.

```powershell
function Assert-MiniMaxH3DenseHotPath(
    [string] $LogPath,
    [double] $CeilingMs = 555000
) {
    $Misses = @(Select-String -LiteralPath $LogPath -SimpleMatch `
        '[trtmc.rtx_engine_cache] hit=0 retained=1').Count
    $Hits = @(Select-String -LiteralPath $LogPath -SimpleMatch `
        '[trtmc.rtx_engine_cache] hit=1').Count
    if ($Misses -ne 5 -or $Hits -ne 5) {
        throw "Expected 5 warmup fills and 5 measured hits; got $Misses and $Hits"
    }

    $Profiles = @(Select-String -LiteralPath $LogPath -SimpleMatch `
        'denoiser optimization_profile=0/2 packed_rows=38247').Count
    if ($Profiles -ne 2) { throw "Expected two profile-0 selections; got $Profiles" }

    $Validations = @(Select-String -LiteralPath $LogPath -Pattern `
        '^\[trtmc\.video_validation\].*status=passed$').Count
    if ($Validations -ne 2) { throw "Expected two successful validations; got $Validations" }

    $Measured = @(Select-String -LiteralPath $LogPath -Pattern `
        '^\[minimax-h3\.perf\].*text_cache_hit=1 adaln_cache_hit=1.*attention_mode=dense.*transformer_forwards=49.*full_denoiser_steps=6 skipped_denoiser_steps=43$')
    if ($Measured.Count -ne 1) { throw 'Measured dense/cache contract did not match' }

    $Sample = @(Select-String -LiteralPath $LogPath -Pattern `
        '^\[trtmc\.video_benchmark_sample\].*generation_ms=([0-9]+(?:\.[0-9]+)?)$')
    if ($Sample.Count -ne 1) { throw "Expected one measured sample; got $($Sample.Count)" }
    $Milliseconds = [double]::Parse(
        $Sample[0].Matches[0].Groups[1].Value,
        [Globalization.CultureInfo]::InvariantCulture)
    if (-not [double]::IsFinite($Milliseconds) -or
        $Milliseconds -le 0 -or $Milliseconds -gt $CeilingMs) {
        throw "Invalid or over-ceiling sample: $Milliseconds ms"
    }
}

Assert-MiniMaxH3DenseHotPath $FiveSecondLog
```

`denoiser_resident_hit=0` and `vae_resident_hit=0` are expected: those fields
describe execution contexts, not retained TensorRT-RTX engine objects. Use a
new runtime-cache filename whenever the bundle, TensorRT-RTX SDK, or driver
changes.

## Long aligned T2VA workload

The same bundle and six-plan visual implementation handle the longest released
local alignment below 15 seconds. This request routes to profile 1:

```powershell
& $Trtmc generate-video $Bundle `
    --prompt $Prompt `
    --num-frames 345 --height 768 --width 1344 --seed 0 `
    --num-inference-steps 50 --guidance-scale 1 `
    --runtime-cache $RuntimeCache `
    --set minimax_h3.retain_engines=true `
    --set minimax_h3.retained_tail_weight_budget_gib=24 `
    --set minimax_h3.first_block_cache_threshold=0.30 `
    --warmup 1 --benchmark 1 `
    --output .\minimax-h3-t2va-345f.mp4
```

It produces 345 frames, or 14.375 seconds at 24 fps. This document does not
assign a latency ceiling to the original-weight dense long request until that
profile has a separate qualification run. Historical FastH3/VSA timings do not
qualify this implementation.

## Acceptance

A five-second result is accepted only when:

- the native dependency audit passed during packaging;
- runtime binaries, plugin, bundle, and TensorRT-RTX DLL come from one package;
- the bundle uses the original official BF16 weights and contains no adapter,
  LoRA, FastH3, VSA, or external attention runtime;
- the request selects profile `0/2` in the shared six-plan visual route;
- the command uses an explicit runtime cache, retained engines, a 24 GiB tail
  budget, and the calibrated `0.30` threshold;
- the measured sample is finite, positive, and no greater than 555,000 ms;
- the log reports dense attention, 49 forwards, 6 full and 43 reused tails,
  and successful RGB/audio validation;
- the output contains 124 frames at 1344x768 and 24 fps, plus AAC stereo audio
  at 32 kHz; and
- visual and audible playback are separately inspected for corruption.

FL2VA and Ref2VA use the same CLI timing mechanism, but they are separate
workloads and are outside this exact five-second dense T2VA qualification.
