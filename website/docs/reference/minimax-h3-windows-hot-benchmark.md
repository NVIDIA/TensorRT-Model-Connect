---
title: MiniMax H3 Windows native benchmark
description: Measure same-process ModelConnect H3 video-and-audio generation without a Python or third-party runtime.
---

This contract measures the public `generate_video()` call inside the installed
ModelConnect C++ runtime. It does not use a Python adapter, FastVideo, Triton,
FFmpeg, a sidecar, or a subprocess. `trtmc.exe` keeps one pipeline alive for
the warmup and measured iterations and writes only the last measured result as
an MP4 through Windows Media Foundation.

## Timing boundary

`--warmup 1 --benchmark 2` performs one unmeasured request followed by two
measured requests in the same process with the same prompt and request. Before
each timer starts, the CLI destroys the prior host result. Each sample begins
immediately before `IPipeline::generate_video(request)` and ends when it
returns synchronized host video and audio.

The qualified hot path also supplies an explicit TensorRT-RTX runtime-cache
file and enables retained engines. The warmup populates the process-local
retained-engine map and JIT runtime cache. Each measured request creates fresh
execution contexts from those engines; the H3 staged-memory policy still
destroys all 51 denoiser contexts before VAE decode, so this does not make the
denoiser and VAE contexts resident at the same time.

The 49 segmented VSA transition engines retain their deserialized engine
objects, but request a zero resident-weight budget from TensorRT-RTX whether
retention is enabled or disabled. Their weights remain fully streamable so the
simultaneously loaded transition set does not consume roughly 39 GiB merely by
falling back to the bundle-wide per-engine budget. The legacy FirstBlockCache
`denoiser_tail_plan` keeps its separate retained-tail budget behavior.

The entry, 49 transition, and finish execution contexts run serially on the
pipeline's explicit CUDA stream and share one TensorRT-RTX user-managed
activation arena sized to the largest context. This replaces 51 simultaneous
maximum-shape activation allocations with one allocation without changing the
execution order. Before each SM121 attention launch, the runtime also performs
a device-wide synchronization so TensorRT-RTX auxiliary and weight streams
have completed before the next context can reuse that arena. CUDA graph capture
is rejected for this explicitly serialized path because captured contexts
require stable, private activation addresses.

Each `generation_ms` sample includes native conditioning, denoising, video VAE
decode, audio VAE decode, device-to-host copies, and host result assembly. It
excludes bundle/pipeline loading, the warmup, destruction of the prior host
result, MP4 encoding, and file output.

The CLI also reports:

- `load_ms`: bundle validation, backend/plugin discovery, and pipeline load;
- `input_decode_ms`: native image/video/audio reference decode and request
  assembly;
- `media_write_ms`: native H.264/AAC MP4 encoding and file output;
- `total_ms`: the complete process-side operation after argument parsing,
  including warmups and all measured calls; and
- a sample summary with median, mean, minimum, and maximum generation time.

Do not compare `total_ms` with a same-process hot target. For an even number of
samples the reported median is the mean of the two middle observations.

## Native VSA backend boundary

The qualified SM121 path assembles its sanitized VSA PTX into an `sm_121a`
cubin at build time and embeds that cubin in the MiniMax H3 model plugin. The
installed payload has no PTX sidecar, performs no driver PTX JIT, and neither
loads nor invokes Python, Triton, FastVideo, or another attention runtime. On
an SM121 device the plugin must print this line once per generation:

```text
[minimax-h3] VSA attention backend=sm121_embedded_cubin
```

Loading, configuring, or launching that specialization is fail-closed on
SM121. Every packed attention output is checked before it can reach the next
transformer block. A non-finite output causes the complete attention branch to
be rebuilt and replayed once with the in-tree portable CUDA implementation;
a failed replay terminates generation. The denoiser finish outputs and updated
scheduler latents are also checked at each step. Other supported NVIDIA
architectures use the portable backend directly. Do not set a model-plugin
directory, backend directory, or external VSA-plugin option for this package.
The locked CLI discovers its sibling ModelConnect DLLs, and the MiniMax H3
plugin contains both paths.

## Five-second T2VA workload

Use the exact bundle, bundle-builder revision, runtime revision, checkpoint,
adapter, TensorRT-RTX SDK, driver, prompt, geometry, and seed in every
reproduction. Close other GPU and unified-memory workloads before starting.

```powershell
$InstallRoot = Join-Path $env:LOCALAPPDATA 'Programs\ModelConnect\MiniMax-H3'
$Trtmc = Join-Path $InstallRoot 'bin\trtmc.exe'
$Bundle = Join-Path $InstallRoot 'models\MiniMax-H3.bundle'
$RepoRoot = (Resolve-Path '<ModelConnect-checkout>').Path
$Prompt = (Get-Content -Raw `
    (Join-Path $RepoRoot `
        'tests\e2e\models\minimax_h3\prompts\t2va-example-1.json') |
    ConvertFrom-Json).prompt
$RuntimeCache = Join-Path (Get-Location) 'minimax-h3-fast-h3.rtxcache'
$FiveSecondLog = Join-Path (Get-Location) 'minimax-h3-t2va-124f.log'

& $Trtmc generate-video $Bundle `
    --prompt $Prompt `
    --num-frames 120 --height 768 --width 1344 --seed 0 `
    --runtime-cache $RuntimeCache `
    --set minimax_h3.retain_engines=true `
    --warmup 1 --benchmark 2 `
    --output .\minimax-h3-t2va-124f.mp4 `
    2>&1 | Tee-Object -FilePath $FiveSecondLog
if ($LASTEXITCODE -ne 0) { throw "MiniMax-H3 124-frame benchmark failed" }
```

The request aligns 120 nominal frames to 124 output frames, or 5.167 seconds
at 24 fps. The acceptance ceiling is 555,000 ms (9:15) for the same-process
hot `generation_ms` median on the qualified Spark hardware/software cohort.
It is a qualification threshold, not a portable latency guarantee.

## Long aligned T2VA workload

The longest released local alignment that remains within 15 seconds is 345
frames, or 14.375 seconds:

```powershell
$LongLog = Join-Path (Get-Location) 'minimax-h3-t2va-345f.log'

& $Trtmc generate-video $Bundle `
    --prompt $Prompt `
    --num-frames 345 --height 768 --width 1344 --seed 0 `
    --runtime-cache $RuntimeCache `
    --set minimax_h3.retain_engines=true `
    --warmup 1 --benchmark 2 `
    --output .\minimax-h3-t2va-345f.mp4 `
    2>&1 | Tee-Object -FilePath $LongLog
if ($LASTEXITCODE -ne 0) { throw "MiniMax-H3 345-frame benchmark failed" }
```

The acceptance ceiling is 1,200,000 ms (20:00) for the hot `generation_ms`
median. A full `1 + 2` command runs three generations, so its wall time is
expected to be much longer than one reported sample.

## Qualified SM121 result

The retained bundle used for this result was constructed at ModelConnect
builder revision `45bff91397da2875f93c0af9b847eb7308fce60d`; its embedded metadata
and build receipt record that revision. The qualified native SM121 runtime was
implementation revision `f029eeeb595b41ef6decf120aa9512fd59e6c4c0`. The
intervening commits change native runtime/build/package code, not the Python
bundle builder or model-plan inputs. Keep these two provenance fields separate:
a bundle rebuilt from a later clean revision records that later revision and is
not expected to have the retained bundle's byte hash.

The run used an NVIDIA RTX Spark N1X (driver 616.67), CUDA 12.9.1, and
TensorRT-RTX 1.6. Both commands used the same prompt fixture (file SHA-256
`44DE7939AAABA9EAFCE0600653417900EADCC5362EC32C2BD5FE6FA70192E787`; extracted
prompt UTF-8 SHA-256
`98F36B879692095E099AE824C18D9E93E7006A490E082FD474A5F531769DCF06`), seed,
bundle, cache policy, and `1 + 2` same-process timing contract shown above. The
194,569,514,211-byte bundle SHA-256 was
`18B69E84EF919399489A0D538117E84938F5768C365433C9C1D125772263F7E3`.

| Request | Output | Measured samples | Hot median |
| --- | --- | --- | --- |
| 120 nominal frames | 124 frames / 5.167 s | 286,417.720 ms; 285,284.762 ms | 285,851.241 ms (4:45.851) |
| 345 frames | 345 frames / 14.375 s | 964,505.670 ms; 959,489.510 ms | 961,997.590 ms (16:01.998) |

These historical `f029eeeb` runs predate build-time cubin assembly and recorded
the legacy `sm121_embedded_ptx` label: 53 warmup engine-cache fills, 106
measured cache hits, three selections, zero `portable_cuda` selections, and
three successful finite-output validations. The 124-frame MP4 SHA-256 is
`18D6C5395D9B56FD35CC87A4419D37B6548EA30358AEE1B92575012A9E9FE38D`; the
345-frame MP4 SHA-256 is
`E69AAAD4764C8E1DB0F193B383ED0F391055A8848AC48B5C3BC0E3D9C81FD37F`.
Both files contain 1344x768 H.264 video at 24 fps and AAC-LC stereo audio at
32 kHz. Normal AAC tail padding makes the encoded audio/container duration a
few milliseconds longer than the raw generated audio.

Pipeline loading and the warmup are intentionally outside each
`generation_ms` sample. On this qualification run, the complete process took
45:23.639 for the 124-frame command and 1:18:40.582 for the 345-frame command;
those wall times cover load, warmup, both measured generations, validation,
and MP4 output and must not be reported as single-generation latency.

## Verify the hot engine cache

For either exact `1 + 2` command above, the warmup must retain 53 engines: the
FastH3 entry, 49 transitions, finish, video VAE, and audio VAE. The two measured
requests must then produce 106 retained-engine hits. Validate the complete log
before accepting its timing:

```powershell
function Assert-MiniMaxH3HotCache(
    [string] $LogPath,
    [double] $MedianCeilingMs
) {
    $MissRecords = @(Select-String -Path $LogPath -SimpleMatch `
        '[trtmc.rtx_engine_cache] hit=0 retained=1')
    $HitRecords = @(Select-String -Path $LogPath -SimpleMatch `
        '[trtmc.rtx_engine_cache] hit=1')
    if ($MissRecords.Count -ne 53 -or $HitRecords.Count -ne 106) {
        throw "Expected 53 warmup cache fills and 106 measured hits; got $($MissRecords.Count) and $($HitRecords.Count)"
    }

    $WarmupValidation = @(Select-String -Path $LogPath -Pattern `
        '^\[trtmc\.video_validation\] phase=warmup .*status=passed$')
    if ($WarmupValidation.Count -ne 1 -or
        @($MissRecords | Where-Object LineNumber -gt $WarmupValidation[0].LineNumber).Count -ne 0 -or
        @($HitRecords | Where-Object LineNumber -lt $WarmupValidation[0].LineNumber).Count -ne 0) {
        throw 'Engine-cache misses must be confined to warmup and all hits to measured requests'
    }

    $SampleRecords = @(Select-String -Path $LogPath -Pattern `
        '^\[trtmc\.video_benchmark_sample\].*generation_ms=([0-9]+(?:\.[0-9]+)?)$')
    if ($SampleRecords.Count -ne 2) {
        throw "Expected exactly two measured samples; got $($SampleRecords.Count)"
    }
    $Samples = @($SampleRecords | ForEach-Object {
        [double]::Parse($_.Matches[0].Groups[1].Value,
            [Globalization.CultureInfo]::InvariantCulture)
    })
    if (@($Samples | Where-Object {
        [double]::IsNaN($_) -or [double]::IsInfinity($_) -or $_ -le 0
    }).Count) {
        throw 'Generation samples must be finite and positive'
    }
    $ComputedMedian = ($Samples[0] + $Samples[1]) / 2.0
    $SummaryRecord = @(Select-String -Path $LogPath -Pattern `
        '^\[trtmc\.video_benchmark_summary\].*median_ms=([0-9]+(?:\.[0-9]+)?).*$')
    if ($SummaryRecord.Count -ne 1) { throw 'Expected exactly one benchmark summary' }
    $ReportedMedian = [double]::Parse(
        $SummaryRecord[0].Matches[0].Groups[1].Value,
        [Globalization.CultureInfo]::InvariantCulture)
    if ([math]::Abs($ComputedMedian - $ReportedMedian) -gt 0.002 -or
        $ReportedMedian -gt $MedianCeilingMs) {
        throw "Invalid or over-ceiling median: computed=$ComputedMedian reported=$ReportedMedian ceiling=$MedianCeilingMs"
    }

    $Sm121 = @(Select-String -Path $LogPath -SimpleMatch `
        'VSA attention backend=sm121_embedded_cubin').Count
    $PortableBackend = @(Select-String -Path $LogPath -SimpleMatch `
        'VSA attention backend=portable_cuda').Count
    $Recovery = @(Select-String -Path $LogPath -Pattern `
        'replaying .*portable_cuda|replaying finish').Count
    if ($Sm121 -ne 3 -or $PortableBackend -ne 0 -or $Recovery -ne 0) {
        throw "Expected three clean SM121 cubin selections; got SM121=$Sm121 portable_backend=$PortableBackend recovery=$Recovery"
    }

    $Validations = @(Select-String -Path $LogPath -Pattern `
        '^\[trtmc\.video_validation\].*status=passed$').Count
    if ($Validations -ne 3) {
        throw "Expected three successful output validations; got $Validations"
    }
}

Assert-MiniMaxH3HotCache $FiveSecondLog 555000
# After running the long workload instead, call:
# Assert-MiniMaxH3HotCache $LongLog 1200000
```

`denoiser_resident_hit=0` and `vae_resident_hit=0` in `[minimax-h3.perf]` are
expected for this staged path: those fields describe execution contexts, not
the retained TensorRT-RTX engines. A measured request with an engine-cache miss
is not a qualified hot sample. Use a new runtime-cache filename whenever the
bundle, TensorRT-RTX SDK, or driver changes.

## Acceptance

Keep the complete stderr/stdout log and record the separate runtime and bundle
builder Git revisions, bundle hash, bundle inspection output, TensorRT-RTX
version, driver version, GPU, request, and MP4 hash. A timing result is accepted
only when:

- the native dependency audit passed during packaging;
- the runtime-only CLI, core, RTX backend, H3 plugin, bundle, and
  TensorRT-RTX DLL are from one package;
- the bundle declares the authenticated FastH3 adapter, 51 native segmented
  VSA plans, four transformer forwards, the public dynamic canvas/frame
  profiles, and no external VSA plugin;
- the command uses an explicit `--runtime-cache` and
  `--set minimax_h3.retain_engines=true`, and its cache validation reports 53
  warmup fills followed by 106 measured hits;
- both measured samples are finite and positive and the command exits zero;
- the output has the requested aligned geometry at 24 fps;
- generation returns audio and the MP4 contains H.264 video plus one AAC
  stereo 32 kHz audio stream; and
- visual and audible playback are separately inspected for corruption.

FL2VA and Ref2VA use the same CLI timing mechanism, but their results are
separate workloads. Ref2VA uses the independent `transformer_ref` partition
and its released 50-point/49-forward schedule; it must never be reported as a
four-forward FastH3 result.
