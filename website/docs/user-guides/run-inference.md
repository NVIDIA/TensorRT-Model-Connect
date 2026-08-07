---
title: Run Inference
description: Select the public task command that matches a bundle's declared task contract.
---

Inspect the bundle, then call the task surface declared by its model manifest.
The CLI is task-oriented; it is not one generic tensor runner.

| Task | Command | Result |
| --- | --- | --- |
| Text or vision-language generation | `trtmc run` | Generated text |
| Encoder features | `trtmc encode` | Feature values |
| Embedding / reranking | `trtmc embed`, `trtmc rerank` | Vector or score |
| Speech recognition | `trtmc transcribe` | Transcript and optional timestamps |
| Text-to-audio / speech-to-speech | `trtmc generate-audio`, `trtmc speak` | Audio file or stream |
| Image/video diffusion | `trtmc generate-video` | One or more frames |
| Segmentation / classification | `trtmc segment`, `segment-prompted`, `classify` | Mask, prompted result, or class scores |
| Time-series / neural operator | `trtmc solve` | Numeric output vector |

Example deterministic text request:

```bash
trtmc run qwen3-0.6b.trtfb \
  --prompt "What is the capital of France? Answer in one word." \
  --max-new-tokens 10 \
  --greedy
```

Shared loading controls such as `--backend-dir`, `--model-plugin-dir`,
`--runtime-cache`, `--config`, and `--set` have execution-path-specific
meaning. Read [Configure Runtime Behavior](configure-runtime.md) before using
one as a generic fix.
