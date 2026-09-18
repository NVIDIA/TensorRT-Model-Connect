---
title: Diffusion and Time-Series Tasks
---

# Diffusion and time-series tasks

These tasks use the same bundle and runtime-root mechanics as text generation,
but expose different typed request contracts.

## Image generation

```bash
python -m tensorrt_model_connect build black-forest-labs/FLUX.1-schnell \
  -o flux.bundle \
  --precision bf16 \
  --image-height 1024 \
  --image-width 1024

trtmc generate-image flux.bundle \
  --prompt "A brass robot reading beside a window" \
  --output robot.png \
  --height 1024 \
  --width 1024 \
  --num-steps 4 \
  --seed 1234
```

The bundle dimensions are build inputs; sampling steps, guidance, seed, and
output path are request inputs. Use `generate-image-batch` with newline-delimited
prompts and a matching seed list when the family implements batch generation.

## Video generation

```bash
python -m tensorrt_model_connect build Wan-AI/Wan2.1-T2V-1.3B \
  -o wan.bundle \
  --precision bf16 \
  --image-height 480 \
  --image-width 832 \
  --video-num-frames 81

trtmc generate-video wan.bundle \
  --prompt "Ocean waves under moonlight" \
  --output waves.mp4 \
  --height 480 \
  --width 832 \
  --num-steps 20 \
  --seed 1234
```

Family-specific acceleration policies are encoded in family code rather than
an open-ended `--set` registry.

## Time-series forecasting

The forecast CLI consumes raw float32 values:

```bash
python -m tensorrt_model_connect build amazon/chronos-bolt-tiny \
  -o chronos.bundle \
  --precision fp32

trtmc forecast chronos.bundle \
  --input history.f32 \
  --frequency H
```

Optional `--mask` input uses the family contract. Neural-operator families use
`solve --branch FILE` and optional `--trunk FILE` instead of comma-separated
values on the command line.

For every task, confirm shape, dtype, file format, supported dimensions, and
numerical thresholds in the owning family's tests and manifest.
