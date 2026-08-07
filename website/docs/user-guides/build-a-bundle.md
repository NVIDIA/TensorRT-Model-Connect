---
title: Build a Bundle
description: Build one native TensorRT bundle from an exact checkpoint and retain the configuration receipt.
---

Start from an exact Hugging Face ID in [Supported Models](../models-recipes/overview.md)
or a compatible local checkpoint directory:

```bash
trtmc build Qwen/Qwen3-0.6B \
  --model-revision MODEL_COMMIT \
  -o qwen3-0.6b.bundle
```

Omit `--model-revision` for exploration only. A reproducible result pins an
immutable revision and records the complete command.

## Choose one configuration source

| Need | Surface | Example |
| --- | --- | --- |
| Common CLI option | Dedicated flag | `--precision fp16` |
| Registered feature schema | Config file | `--config profile.json` |
| One schema override | Repeatable key/value | `--set namespace.field=value` |
| Model-owned parallel build | Topology flag | `--tensor-parallel-size 4` |

```bash
trtmc build MODEL_ID \
  --model-revision MODEL_COMMIT \
  --precision fp16 \
  --config build-profile.json \
  --set qwen_vl_vision.dynamic_resolution=true \
  -o model.bundle
```

Do not copy that combination to an arbitrary family. The selected family owns
which schemas, precision modes, quantization formats, graph shapes, and
topologies it supports.

## Retain the build receipt

Record at least:

- exact model ID and immutable revision;
- output bundle name and checksum;
- complete build command and config file;
- build environment, TensorRT/CUDA cohort, and SM architecture;
- family, runtime strategy, precision, quantization, and topology; and
- whether the resulting bundle is native or platform-specialized.

Run [Inspect a Bundle](inspect-a-bundle.md) before inference. The
[CLI Reference](../api/cli-reference.md#trtmc-build) is the source for the
complete option inventory.
