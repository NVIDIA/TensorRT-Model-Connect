---
title: CLI Reference
description: Current build, inspect, and execution commands.
---

Building and executing are intentionally separate programs:

- `python -m tensorrt_model_connect build` is the Python build CLI.
- `trtmc` is the native inspect and execution CLI.

## Build a bundle

```text
python -m tensorrt_model_connect build MODEL --output PATH [OPTIONS]
```

| Option | Meaning |
| --- | --- |
| `MODEL` | Hugging Face model ID or local snapshot directory |
| `-o, --output PATH` | Required output bundle |
| `--revision REV` | Hugging Face revision used for download |
| `--task TASK` | Override the owning family's default with another declared task |
| `--precision {fp16,bf16,fp32}` | TensorRT build precision |
| `--backend {trt,trt_rtx}` | Engine backend recorded by the bundle |
| `--max-sequence-length N` | Family-validated sequence bound |
| `--image-height N`, `--image-width N` | Family-validated image bounds |
| `--video-num-frames N` | Family-validated video frame bound |
| `--max-batch-size N` | Build batch bound |
| `--tensor-parallel-size N` | Requested tensor-parallel size |
| `--context-parallel-size N` | Requested context-parallel size |
| `--quantization NAME` | Family-owned quantization selection |
| `--fp32-layer N` | Repeatable family-owned FP32 layer selection |
| `--dynamic-kv-cache` | Opt in where the family implements runtime-sized KV |
| `--verbose` | Enable verbose build output |

A family must reject unsupported values. The CLI has no `--config`, `--set`,
provider profile, fallback, or migration mode.

## Inspect a bundle

```bash
trtmc inspect model.bundle
```

Inspect prints JSON containing the header and named section ranges. It does not
load TensorRT, a backend, or a family DSO.

## Execute a bundle

```text
trtmc COMMAND BUNDLE --runtime-root DIR [OPTIONS]
```

Every execution command requires `--runtime-root`. The loader does not search
implicit locations.

| Command | Abstract Task API |
| --- | --- |
| `run` | text or vision-language generation |
| `encode`, `embed`, `rerank` | encoding, embedding, ranking |
| `classify`, `extract-features` | image classification or features |
| `disparity`, `geometry` | stereo or monocular geometry |
| `segment`, `segment-prompted`, `video-segment` | segmentation |
| `generate-audio`, `speak`, `speech-session` | audio generation and speech |
| `transcribe`, `transcribe-batch`, `transcribe-streaming` | speech recognition |
| `generate-image`, `generate-image-batch`, `generate-video` | media generation |
| `solve`, `forecast`, `control`, `generate-world` | operators, time series, robotics, world models |

The selected command must match the bundle's `task`; a mismatch is an error.

## Text generation

```bash
trtmc run qwen.bundle \
  --runtime-root /opt/trtmc/lib \
  --prompt "What is TensorRT?" \
  --max-new-tokens 64 \
  --temperature 0.8 \
  --top-p 0.95 \
  --use-chat-template true \
  --enable-thinking false
```

Other supported text options include `--top-k`, `--min-p`, `--seed`,
`--repetition-penalty`, language token IDs, LoRA adapter IDs, and the explicit
text-diffusion replay inputs accepted by the selected family.

## Direct optional paths

Runtime-sized KV is explicit:

```bash
trtmc run llama.bundle --runtime-root /opt/trtmc/lib \
  --kv-cache-size 1GiB --prompt "Hello"
```

TensorRT-RTX uses the same Task API:

```bash
trtmc run model.bundle --runtime-root /opt/trtmc/lib \
  --runtime-cache kernels.cache --cuda-graphs --prompt "Hello"
```

BYOK runtime loading requires all three fields together:

```text
--byok-library DSO --byok-function FUNCTION --byok-name KERNEL
```

Run `trtmc --help` and
`python -m tensorrt_model_connect --help` against the exact revision being
used for the final parser contract.
