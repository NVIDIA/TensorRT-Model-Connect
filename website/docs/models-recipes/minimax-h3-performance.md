---
title: MiniMax H3 performance reproduction
description: Two-page performance summary and deployment entry points for native MiniMax H3.
---

September 10, 2026 | [PR #1241](https://github.com/NVIDIA/TensorRT-Model-Connect/pull/1241) | Branch: `codex/minimax-h3-performance`

## 1. Results

**12/12 configurations completed**, plus two empty-cache checks and a nine-request resident sequence. One runtime supports T2VA, FL2VA and REF2VA, with dynamic prompts and duration. Choose a normal bundle or an explicitly enabled SR bundle.

**E2E minutes, including engine loading and MP4 writing:**

| Mode | Normal, 124 frames | Normal, 345 frames | SR, 124 frames | SR, 345 frames |
| --- | ---: | ---: | ---: | ---: |
| T2VA | 9.51 | 46.99 | 3.63 | 12.37 |
| FL2VA | 10.03 | 50.71 | 3.66 | 11.62 |
| REF2VA | 16.01 | 82.33 | 7.87 | 29.18 |

Normal: **1344 x 768**. SR: **864 x 480 -> 1296 x 720**. Outputs are **5.167 / 14.375 seconds**, 24 fps, with 32 kHz stereo audio.

**Measurement conditions:** fresh process, copied seeded RTX disk cache, seed 0, 50 schedule points, guidance 1, and **both FBC thresholds 0.3**. Each cell is one unprofiled observation; download, build, cache preparation and QA are excluded. OS/driver caches were not cleared and clocks were not locked. Empty-cache short T2VA took 9.67 min normal / 4.04 min SR; these are not fully cold-machine measurements.

**Optimization:** bulk plan reads and request-sized I/O, activation and cache storage. Normal short T2VA decreased from 693.56 to 570.80 seconds versus PR #1240 in the single matched observation (17.7% less wall time). Most other baseline pairs were not measured. The baseline Nsight trace put 63% of denoiser GPU time in attention; long native REF2VA still spends 94.8% of generation in denoising. Model weights and sampling mathematics were unchanged.

## 2. Deployment

**Tested runtime:** `aa594f6010bd59204c2edc3061fc6e5042572c0d`. Documentation commits do not change that measured revision. Follow [Windows setup](./minimax-h3.md#windows-setup), then the checkpoint-download block in [Build](./minimax-h3.md#build-normal-or-super-resolution). Use the 0.3 build commands below instead of the guide's default builds.

- Windows; **PowerShell 7.2+**: run `pwsh -NoProfile` from x64 VS 2022 Developer PowerShell.
- CUDA **12.9**, TensorRT-RTX **1.6.1.120**, matching SDK DLL/wheel and a compatible NVIDIA driver.
- Recorded build tools: MSVC 19.44, CMake 3.31.6, Ninja 1.12.1, Python 3.13.5.
- Finished bundles occupy about **117.61 GiB each**; checkpoints and build staging need additional space. Hardware/driver identities are not disclosed, so identical latency on another machine is not guaranteed.

**Weights/runtime:** full Comfy INT8 denoisers; NVFP4-AWQ text weights decoded to **BF16 matmul**, not native FP4 GEMM. Vision, VAEs and Real-ESRGAN remain floating-point. Runtime is ModelConnect **C++/CUDA + TRT-RTX**, including CUDA scheduler/VAE helpers and Windows Media Foundation media I/O. No Python, PyTorch, ComfyUI or FFmpeg is needed for generation; Python is build-time.

After setup defines `$Checkpoint` and `$ArtifactRoot`, build either or both bundles into fresh output paths:
```powershell
$FBC = @('--set', 'minimax_h3.first_block_cache_threshold=0.3',
         '--set', 'minimax_h3.ref2va_first_block_cache_threshold=0.3')
python -m tensorrt_model_connect build $Checkpoint --backend trt_rtx `
  --precision bf16 --output "$ArtifactRoot/normal.bundle" @FBC
if ($LASTEXITCODE) { throw 'Normal build failed' }
python -m tensorrt_model_connect build $Checkpoint --backend trt_rtx `
  --precision bf16 --output "$ArtifactRoot/sr.bundle" @FBC `
  --set minimax_h3.super_resolution=true
if ($LASTEXITCODE) { throw 'SR build failed' }
```

---

## 3. Run and reproduce

Use the [recorded prompts, input files and cache protocol](https://github.com/yifeif-nv/TensorRT-Model-Connect-fork/blob/f030c5107bdd8e36298dde184681ca146c0de418/website/docs/models-recipes/minimax-h3-performance.md#exact-evaluation-inputs) for benchmark reproduction. The authorized local package contains the original images/audio and the nine-request C++ manifest. Historical cache seeds are not distributed; the protocol explains how to create a separately labelled local seed. Rebuilt engines/seeds do not guarantee identical historical timings.

The common native command is below. Set `$Prompt` for the chosen mode, `$Cache` to a fresh copy of the matching bundle's seed, and `$Output` to an unused MP4 path. Cache copying is outside timing.
```powershell
$RuntimeRoot = Join-Path $InstallRoot 'bin'
$Trtmc = Join-Path $RuntimeRoot 'trtmc.exe'
& $Trtmc generate-video "$ArtifactRoot/normal.bundle" `
  --runtime-root $RuntimeRoot --prompt $Prompt `
  --height 768 --width 1344 --num-frames 120 --seed 0 `
  --num-steps 50 --guidance-scale 1 `
  --runtime-cache $Cache --output $Output
if ($LASTEXITCODE) { throw 'Generation failed' }
```

Before executing, append the selected mode's input flags to the generation command (before the exit-code check):

| Mode | Input flags |
| --- | --- |
| T2VA | None beyond `--prompt` |
| FL2VA | `--first-frame "$ArtifactRoot/inputs/first.png" --last-frame "$ArtifactRoot/inputs/last.png"` (either endpoint may be omitted) |
| REF2VA | `--reference-image "$ArtifactRoot/inputs/first.png" --reference-audio "$ArtifactRoot/inputs/audio.wav"`; preserve reference order and prompt tags |

**Select the other configurations:** change `--num-frames 120` to `345` for long output. For SR, select `sr.bundle` and change dimensions to `--height 480 --width 864`. These two lengths x two bundles x three modes cover the twelve results. SR is never selected automatically from resolution. For empty-cache testing, use a new absent cache path and report it separately.

**Inputs remain dynamic:**

- Output rounds up to `17*n + 5` within **124-345 frames**; 120 requested frames produce 124.
- T2VA: up to **2641 text tokens**. FL2VA: **2641 combined text/endpoint rows**. REF2VA: **262144 combined text/reference rows**, with **1-12 ordered references**.
- Normal uses the supported finite canvas set, not arbitrary dimensions. SR requires the fixed **864 x 480** base. See the [model input limits](./minimax-h3.md#capabilities-and-inputs) for reference counts, durations and aspect ratios.

**C++ API:** use `trtmc::load_task` and `IVideoGeneration::generate_video(request)`; keep the task alive for repeated requests. The [complete C++ example](./minimax-h3.md#c-api) includes compilation and conditioning fields. It returns frames/audio in memory; applications own media decoding/container writing. The [native resident consumer](https://github.com/yifeif-nv/TensorRT-Model-Connect-fork/blob/f030c5107bdd8e36298dde184681ca146c0de418/families/minimax_h3/tools/README.md) also handles MP4 I/O and ordered requests. Resident request timings are not fresh-process E2E timings.

## 4. Quality and limits

All twelve outputs passed full audio/video decode, finite-audio checks and sampled visual inspection. No obvious sampled corruption or replacement scene was found. **Detail and framing can differ**, especially long REF2VA; outputs are not pixel-identical. Full-motion playback and listening were not completed.

FBC is approximate; **0.3 is the measured build setting, not the product default (0.08)**. SR is not pixel-equivalent to native-resolution generation. The tested subset is not exhaustive, and these timings are not universal hardware or quality guarantees. PR #1241 remains Draft pending review.

**More detail only when needed:** [all twelve exact commands, source pins, stage timings and validation evidence](https://github.com/yifeif-nv/TensorRT-Model-Connect-fork/blob/f030c5107bdd8e36298dde184681ca146c0de418/website/docs/models-recipes/minimax-h3-performance.md). Raw logs/traces and unapproved fixtures stay private.
