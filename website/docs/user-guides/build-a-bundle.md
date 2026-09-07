---
title: Build a Bundle
description: Build one family-owned TensorRT bundle from an exact checkpoint.
---

Start from an exact supported model ID or compatible local snapshot:

```bash
python -m tensorrt_model_connect build Qwen/Qwen3-0.6B \
  --revision MODEL_COMMIT \
  --precision fp16 \
  --max-sequence-length 256 \
  --output qwen3-0.6b.bundle
```

`--output` is required. Pin an immutable revision for reproducible evidence.
The resolver requires exactly one `families/*/support.py` match and imports
only that family's `model.py`.

## Choose explicit typed inputs

| Need | Option |
| --- | --- |
| Non-default family task | `--task TASK` |
| Precision/backend | `--precision`, `--backend` |
| Text/image/video build bounds | `--max-sequence-length`, `--image-height`, `--image-width`, `--video-num-frames` |
| Batch/topology | `--max-batch-size`, `--tensor-parallel-size`, `--context-parallel-size` |
| Family-owned quantization | `--quantization NAME`, repeatable `--fp32-layer INDEX` |
| Compatible dynamic KV build | `--dynamic-kv-cache` |

The selected family implements or explicitly rejects each non-default value.
Do not copy an option combination to a different checkpoint and infer support.
There is no generic config file/options registry or build fallback.

## Retain the build receipt

Record the exact source revision, model ID/revision, family requirements,
complete command, output checksum, family/task/backend from inspection,
precision/topology, target GPU, and TensorRT/CUDA cohort. A successful build
proves artifact construction only; it does not prove Task correctness or
parity.

Next, [inspect the bundle](inspect-a-bundle.md), then run its declared Task.
