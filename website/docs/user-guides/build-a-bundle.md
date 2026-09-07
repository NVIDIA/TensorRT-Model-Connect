---
title: Build a Bundle
description: Build one TensorRT bundle from an exact checkpoint and retain a reproducible receipt.
---

Start from an exact Hugging Face ID in
[Supported Models](../models-recipes/overview.md) or a compatible local
checkpoint directory:

```bash
python -m tensorrt_model_connect build Qwen/Qwen3-0.6B \
  --revision MODEL_COMMIT \
  --precision fp16 \
  --max-sequence-length 16384 \
  --output qwen3-0.6b.bundle
```

Omit `--revision` only when reproducibility is not required. A local directory
is used as-is and does not accept a remote revision.

## Select only supported build inputs

The build CLI intentionally has one small common surface:

| Need | Argument |
| --- | --- |
| Choose another family-supported task | `--task` |
| Engine precision | `--precision fp16|bf16|fp32` |
| Engine backend | `--backend trt|trt_rtx` |
| Sequence, image, video, or batch bounds | `--max-sequence-length`, `--image-height`, `--image-width`, `--video-num-frames`, `--max-batch-size` |
| Model-owned parallel build | `--tensor-parallel-size`, `--context-parallel-size` |
| Family-supported quantization | `--quantization` |
| Runtime-sized KV support | `--dynamic-kv-cache` |

Do not copy a combination to an arbitrary family. The selected family's
`support.py` declares tasks, and its plain `model.py` decides which build
values are legal and writes all model-owned sections. The function implements
`build(request, writer)` directly; it does not inherit from shared builder
code.

If the family has extra build or reference packages, install only its own file:

```bash
python -m pip install -r families/FAMILY/requirements.txt
```

## Retain the build receipt

Record:

- exact model ID and revision, or the local snapshot identity;
- complete build command and output bundle path;
- selected family, task, backend, precision, shapes, and topology;
- build environment, TensorRT/CUDA versions, GPU, and SM architecture; and
- the first build error if the command fails.

Inspect the artifact before execution:

```bash
trtmc inspect qwen3-0.6b.bundle
```

Inspection reports shared routing fields and section offsets. It does not prove
task correctness; that evidence belongs to the selected family's tests.
