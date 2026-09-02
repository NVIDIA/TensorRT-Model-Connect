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

## Five-second T2VA workload

Use the exact bundle, source revision, checkpoint, adapter, TensorRT-RTX SDK,
driver, prompt, geometry, and seed in every reproduction. Close other GPU and
unified-memory workloads before starting.

```powershell
$Bundle = Join-Path $env:LOCALAPPDATA `
    'Programs\ModelConnect\MiniMax-H3\models\MiniMax-H3.bundle'
$Prompt = (Get-Content -Raw `
    .\tests\e2e\models\minimax_h3\prompts\t2va-example-1.json |
    ConvertFrom-Json).prompt

trtmc generate-video $Bundle `
    --prompt $Prompt `
    --num-frames 120 --height 768 --width 1344 --seed 0 `
    --warmup 1 --benchmark 2 `
    --output .\minimax-h3-t2va-124f.mp4
```

The request aligns 120 nominal frames to 124 output frames, or 5.167 seconds
at 24 fps. The current performance qualification target is approximately
8--9 minutes for the same-process hot `generation_ms` median on the qualified
Spark hardware/software cohort. It is a target, not a portable latency
guarantee.

## Long aligned T2VA workload

The longest released local alignment that remains within 15 seconds is 345
frames, or 14.375 seconds:

```powershell
trtmc generate-video $Bundle `
    --prompt $Prompt `
    --num-frames 345 --height 768 --width 1344 --seed 0 `
    --warmup 1 --benchmark 2 `
    --output .\minimax-h3-t2va-345f.mp4
```

The current qualification target is approximately 20 minutes for the hot
`generation_ms` result. A full `1 + 2` command runs three generations, so its
wall time is expected to be much longer than one reported sample.

## Acceptance

Keep the complete stderr/stdout log and record the Git revision, bundle hash,
bundle inspection output, TensorRT-RTX version, driver version, GPU, request,
and MP4 hash. A timing result is accepted only when:

- the native dependency audit passed during packaging;
- the runtime-only CLI, core, RTX backend, H3 plugin, bundle, and
  TensorRT-RTX DLL are from one package;
- the bundle declares the authenticated FastH3 adapter, 51 native segmented
  VSA plans, four transformer forwards, the public dynamic canvas/frame
  profiles, and no external VSA plugin;
- both measured samples are finite and positive and the command exits zero;
- the output has the requested aligned geometry at 24 fps;
- generation returns audio and the MP4 contains H.264 video plus one AAC
  stereo 32 kHz audio stream; and
- visual and audible playback are separately inspected for corruption.

FL2VA and Ref2VA use the same CLI timing mechanism, but their results are
separate workloads. Ref2VA uses the independent `transformer_ref` partition
and its released 50-point/49-forward schedule; it must never be reported as a
four-forward FastH3 result.
