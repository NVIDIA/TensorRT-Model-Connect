---
title: Run Inference
description: Select the native command that matches a bundle's abstract Task contract.
---

Inspect the bundle, then call the command matching its `task` field. Inspection
does not need a runtime directory; every execution command does.

| Task | Command | Typical result |
| --- | --- | --- |
| Text or vision-language generation | `trtmc run` | Generated text and token IDs |
| Encoder features | `trtmc encode` | Feature values |
| Embedding or reranking | `trtmc embed`, `trtmc rerank` | Vector or score |
| Speech recognition | `trtmc transcribe` | Transcript and token IDs |
| Audio or speech-to-speech | `trtmc generate-audio`, `trtmc speak` | Audio file and metadata |
| Image or video diffusion | `trtmc generate-image`, `trtmc generate-video` | Image or frame paths |
| Segmentation or classification | `trtmc segment`, `trtmc segment-prompted`, `trtmc classify` | JSON mask or scores |
| Monocular geometry | `trtmc geometry` | Points, depth, mask, and intrinsics |
| Forecasting or neural operators | `trtmc forecast`, `trtmc solve` | Numeric values |

Example text request:

```bash
trtmc inspect qwen3-0.6b.bundle

trtmc run qwen3-0.6b.bundle \
  --runtime-root /opt/trtmc/lib \
  --prompt "What is the capital of France? Answer in one word." \
  --max-new-tokens 10 \
  --temperature 0 \
  --top-k 1
```

The loader opens exactly the family and backend named in the bundle. The
family implements the requested abstract interface and drives engines through
the Engine API. It does not call another family.

Use `trtmc help` for the complete command list and then read the task-specific
guide before changing request controls.
