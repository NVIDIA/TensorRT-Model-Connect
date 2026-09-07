---
title: CLI Reference
---

Building and native execution use separate entry points. The Python module
builds bundles; the `trtmc` executable inspects and runs them.

## Build a bundle

```bash
python -m tensorrt_model_connect build MODEL -o OUTPUT.bundle [OPTIONS]
```

`MODEL` is a Hugging Face model ID or a local snapshot directory. `--output`
is required. A remote model is downloaded at `--revision` when supplied. The
resolver reads root model metadata and requires exactly one dependency-free
`families/*/support.py` to claim it. It then imports only the selected
family's `model.py` and calls `build(request, writer)` once.

### Build options

| Option | Contract |
| --- | --- |
| `-o`, `--output PATH` | Required output bundle path. |
| `--task TASK` | Override the family-owned default with another task declared by that family. |
| `--revision REVISION` | Hugging Face revision used for snapshot download. |
| `--precision fp16\|bf16\|fp32` | Requested build precision; default is `fp32`. The family must support or reject it. |
| `--backend trt\|trt_rtx` | Bundle backend identity; default is `trt`. |
| `--max-sequence-length N` | Optional family-consumed sequence bound. |
| `--image-height N`, `--image-width N` | Optional image build dimensions. |
| `--video-num-frames N` | Optional video frame count. |
| `--max-batch-size N` | Maximum build batch; default is `1`. |
| `--tensor-parallel-size N` | Requested TP size; default is `1`. |
| `--context-parallel-size N` | Requested CP size; default is `1`. |
| `--quantization NAME` | Family-owned quantization selection. |
| `--fp32-layer INDEX` | Keep one family-local layer in FP32; repeatable. |
| `--dynamic-kv-cache` | Request the direct dynamic-KV build path from a compatible family. |
| `--verbose` | Enable verbose family/TensorRT build output. |

There are no compatibility aliases, configuration registries, graph-selection
subcommands, automatic fallbacks, or default output-name derivation. An
unsupported request fails in its owning family.

## Inspect a bundle

```bash
trtmc inspect MODEL.bundle
trtmc version
```

`inspect` accepts exactly one path and prints the format, family, task,
backend, and section bounds. Family-owned section payloads are not decoded.

## Run a bundle

Every execution command has this shape:

```bash
trtmc COMMAND MODEL.bundle [--runtime-root DIR] [OPTIONS]
```

When `--runtime-root` is omitted, the CLI searches the active runtime and CLI
installation, followed by colon-separated `TRTMC_RUNTIME_PATH` entries. It
does not search the current directory unless `.` is explicitly present in that
variable. One selected root must contain the requested backend and family DSOs;
their descriptors must declare the active product-build identity, expected
kind, and bundle ID before either factory is called. Common load options are:

| Option | Contract |
| --- | --- |
| `--runtime-root DIR` | Select one exact plugin root without search fallback; exact-build validation still applies. |
| `--kv-cache-size BYTES\|GB\|GiB` | Runtime-sized KV capacity for a compatible bundle. |
| `--runtime-cache PATH` | TensorRT-RTX cache path; rejected by the standard TensorRT backend. |
| `--cuda-graphs` | Enable TensorRT-RTX CUDA graphs; rejected by the standard backend. |
| `--byok-library DSO`, `--byok-function NAME`, `--byok-name NAME` | Load one TVM-FFI BYOK binding. All three are required together. |

### Task commands

| Task | Command and primary inputs |
| --- | --- |
| Text or vision-language generation | `run` with `--prompt`, and optional `--image` |
| Encoding and embedding | `encode --text TEXT`, `embed --text TEXT` |
| Reranking | `rerank --query TEXT --document TEXT` |
| Classification and features | `classify --image PATH`, `extract-features --image PATH` |
| Stereo and geometry | `disparity --left PATH --right PATH`, `geometry --image PATH --output DIR` |
| Segmentation | `segment --image PATH`, `segment-prompted --image PATH` with text or point prompt, `video-segment --frame PATH --prompt TEXT` |
| Audio | `generate-audio --prompt TEXT --output PATH`, `speak --input WAV --output WAV`, `speech-session --input WAV` |
| Transcription | `transcribe --input WAV`, repeatable-input `transcribe-batch`, or `transcribe-streaming --input WAV` |
| Image and video generation | `generate-image`, `generate-image-batch`, or `generate-video`, each with the command-specific prompt/output options |
| Numeric tasks | `solve --branch F32 [--trunk F32]`, `forecast --input F32 [--mask F32]` |
| Robotics and world models | `control --image PATH --state F32`, `generate-world --prompt TEXT --image PATH --output DIR` |

`F32` inputs are raw binary float32 files. Media inputs are decoded by private
CLI code under `apps/cli/`; these file formats are not part of the C++ Task API.

### Text generation options

`run` accepts `--max-new-tokens`, `--temperature`, `--top-k`, `--top-p`,
`--min-p`, `--seed`, `--repetition-penalty`, `--use-chat-template true|false`,
`--enable-thinking true|false`, source/forced language token IDs, family-owned
text-diffusion replay inputs, and a paired `--lora-adapter` /
`--lora-adapter-id`. The loaded family decides which values it supports.

```bash
trtmc run qwen.bundle \
  --prompt "Hello" \
  --max-new-tokens 32 \
  --temperature 0 \
  --use-chat-template true \
  --enable-thinking false
```

### Transcription and generation options

Offline transcription accepts language/translation fields, beam size, length
penalty, punctuation, timestamps, input-duration limits, and segmented decode
controls. Streaming transcription instead accepts chunk size, attention
contexts, language, and a new-token bound.

Image/video generation accepts negative prompt, height, width, steps, seed,
guidance/CFG scale, and optional initial float32 latents. Audio generation and
speech commands expose only the options listed by their Task contracts in
`apps/cli/cli.cpp`.

Unknown commands, unknown command-specific options, duplicate options, task
interface mismatches, invalid values, and missing DSOs fail with a nonzero exit
status. Run `trtmc help` for the compiled executable's concise synopsis and
`python -m tensorrt_model_connect build --help` for the exact build parser.
