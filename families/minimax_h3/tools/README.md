# Native resident video benchmark

This Windows-only consumer loads one ModelConnect task and executes an ordered
sequence of full video-and-audio requests. It reuses the CLI's existing native
image/media readers and Media Foundation MP4 writer. Generation and media I/O
need no Python, FFmpeg, or ComfyUI runtime.

Build separately from the repository's top-level build, using its matching
runtime import library and the existing JSON development package:

```powershell
cmake -S families/minimax_h3/tools -B build/h3-video-benchmark -G Ninja `
    -DCMAKE_BUILD_TYPE=Release `
    "-DTRTMC_RUNTIME_LIBRARY=$InstallRoot/lib/trtmc_runtime.lib" `
    "-DCMAKE_PREFIX_PATH=$JsonInstall"
cmake --build build/h3-video-benchmark
$env:PATH = @($RuntimeRoot, $env:PATH) -join [IO.Path]::PathSeparator
& ./build/h3-video-benchmark/h3_benchmark_video.exe requests.json --validate-only
& ./build/h3-video-benchmark/h3_benchmark_video.exe requests.json 2>&1 |
    Tee-Object -FilePath sequence.log
```

Use a configured C++ build shell. Keep the runtime's normal TensorRT-RTX/CUDA
dependency directories on `PATH`. All paths inside the UTF-8 JSON manifest are
resolved relative to the manifest's directory, not the process working directory.
Each output MP4 and its adjacent `.receipt.json` must not already exist.
`$InstallRoot`, `$RuntimeRoot` (the installed `bin` directory), and `$JsonInstall`
are the paths from the model's Windows setup instructions. If using an
uninstalled build tree, point `TRTMC_RUNTIME_LIBRARY` at that tree's
`trtmc_runtime.lib` instead.

```json
{
  "schema_version": 1,
  "bundle": "artifacts/video.bundle",
  "runtime_root": "runtime",
  "runtime_cache": "cache/sequence.rtxcache",
  "cuda_graphs": false,
  "requests": [
    {
      "mode": "t2va",
      "prompt": "A continuous shot of a forest waterfall with flowing-water ambience.",
      "num_frames": 124,
      "seed": 0,
      "output": "results/01.mp4"
    },
    {
      "mode": "fl2va",
      "prompt": "Continue between these waterfall endpoints without cuts.",
      "first_frame": "inputs/first.png",
      "last_frame": "inputs/last.png",
      "num_frames": 124,
      "seed": 0,
      "output": "results/02.mp4"
    },
    {
      "mode": "ref2va",
      "prompt": "Retain the waterfall in <Picture 1>, with ambience matching <Audio 1>.",
      "references": [
        {"kind": "image", "path": "inputs/reference.png"},
        {"kind": "audio", "path": "inputs/reference.wav"}
      ],
      "num_frames": 345,
      "seed": 0,
      "output": "results/03.mp4"
    }
  ]
}
```

References remain ordered; supported kinds are `image`, `video`, and `audio`.
Video/audio decoding uses the loaded family's reference-media policy. Supply
one or both endpoints only for FL2VA, or references only for REF2VA. Every item
constructs a new request, so inputs are not carried between modes. Optional
`height` and `width` must appear together; omission uses the bundle's defaults.
Other optional request fields are `num_steps` (default 50), `guidance_scale`,
and `cfg_scale` (both default to the family-selected value). Prompt, mode, and
output are required; frames default to 124 and seed to 0. SR is selected by
loading an SR bundle, not by this tool changing a normal bundle's behavior.

For prompt-reuse checks, use separate items in the order A, A, B, long-A, A,
holding shape, seed, and media inputs fixed. For dynamic-duration checks, use
124, 345, 124 with the same prompt. This is a true resident task sequence, but
does not imply that the family retains every engine or every conditioning result.

## Timing and evidence

- `task_load_ms_once` is the single initial `load_task` duration, repeated as
  context in receipts; do not sum it per request. Lazy engine loading remains
  inside the corresponding generation call.
- `prepare_ms` includes construction and media decoding for that request.
- `generate_video_ms` measures the complete public `generate_video` call,
  excluding media preparation and MP4 writing. Family stage timings remain in
  the captured runtime log between `request_begin` and `request_end` markers.
- `write_mp4_ms` measures native output encoding. `request_wall_ms` covers
  preparation through encoding, excluding receipt writing and output cleanup.
  None of these fields is process-launch-to-exit timing.
- Receipts identify the process, ordered request, output shape/audio metadata,
  input manifest, and whether the disk cache existed before task loading.
  An existing cache file does not establish a hit, and a fresh process is not
  a resident repeat. Use a new cache filename for a disk-cache-cold sequence.
- Cache thresholds are properties of the loaded bundle. This consumer does
  not raise or override them; retain the bundle metadata and logged thresholds
  when comparing baseline and candidate runs. Profiling runs must be reported
  separately from ordinary wall-time measurements.

`--validate-only` checks the manifest, mode/input consistency, input file
existence, and output conflicts without loading any task. It does not validate
engine shapes, decode reference media, or generate video. Execution fails on
the first request error and preserves completed receipts and any partial output.
Do not relabel an unrun request as successful. Generated receipts/logs contain
user-provided prompts and local paths: retain raw evidence privately and sanitize
any public report.
