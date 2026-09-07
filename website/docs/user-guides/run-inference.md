---
title: Run Inference
description: Select the native command matching the bundle's declared Task.
---

Inspect the bundle, then call the Task named by its family manifest/header.
Every execution command requires `--runtime-root DIR`.

| Task | Command | Primary result |
| --- | --- | --- |
| Text / vision-language generation | `run` | Generated text and token IDs |
| Encoding / embedding / reranking | `encode`, `embed`, `rerank` | Vector or score |
| Classification / features | `classify`, `extract-features` | Class result or feature vector |
| Stereo / geometry / segmentation | `disparity`, `geometry`, `segment*`, `video-segment` | Task JSON and optional files |
| Speech recognition | `transcribe`, `transcribe-batch`, `transcribe-streaming` | Transcript or stream chunks |
| Audio and speech | `generate-audio`, `speak`, `speech-session` | WAV output or session events |
| Image/video generation | `generate-image`, `generate-image-batch`, `generate-video` | PNG or frame output |
| Numeric/robotics/world | `solve`, `forecast`, `control`, `generate-world` | Typed numeric or media result |

```bash
trtmc run qwen3-0.6b.bundle \
  --runtime-root /opt/trtmc/lib \
  --prompt "What is the capital of France? Answer in one word." \
  --max-new-tokens 10 \
  --temperature 0 \
  --top-k 1
```

The CLI loads exactly one family DSO and backend DSO from the runtime root. A
wrong Task command fails instead of attempting another family or interface.
See the [CLI Reference](../api/cli-reference.md) for command-specific inputs.
