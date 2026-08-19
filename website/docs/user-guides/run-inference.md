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
| SAM2-HOI video tracking | Model-owned C ABI | Ordered detections, interaction pairs, IDs, and binary masks |
| Time-series / neural operator | `trtmc solve` | Numeric output vector |

Example deterministic text request:

```bash
trtmc run qwen3-0.6b.bundle \
  --prompt "What is the capital of France? Answer in one word." \
  --max-new-tokens 10 \
  --greedy
```

Shared loading controls such as `--backend-dir`, `--model-plugin-dir`,
`--runtime-cache`, `--config`, and `--set` have execution-path-specific
meaning. Read [Configure Runtime Behavior](configure-runtime.md) before using
one as a generic fix.

SAM2-HOI is an explicit exception to the task-oriented CLI. Its family DSO
exports `trtmc_sam2_hoi_video_run_jpeg_files_v1`, which accepts exactly five
JPEG paths and separate JSON/mask destinations. Use the public
`trtmc/models/sam2_hoi_video.h` contract described under
[SAM2-HOI video C ABI](../api/cpp-api.md#sam2-hoi-video-c-abi); do not infer this
entrypoint from another family or from the task name.

{/* Collaborative review anchor: batch 2. */}
