---
title: Build Pipeline
description: How one checkpoint is resolved to one family-owned TensorRT bundle.
---

import Diagram from '@site/src/components/Diagram';

The build pipeline performs one control transfer from shared code to the owning
family. After that transfer, the family owns the graph, weights, sections, and
errors.

<Diagram
  src="/img/diagrams/architecture/build-route-selection.svg"
  alt="A model ID or local checkpoint resolves to exactly one support module before the selected plain model build function writes a bundle"
  caption="Zero or multiple family matches fail. A selected family build is called once and never falls back to another implementation."
/>

## Build from the CLI

```bash
python -m tensorrt_model_connect build Qwen/Qwen3-0.6B \
  --revision MODEL_COMMIT \
  --precision fp16 \
  --max-sequence-length 4096 \
  --output qwen3.bundle
```

The positional model may also be a prepared local directory.

## Resolution and build steps

1. If the input is a Hugging Face ID, the CLI downloads the requested snapshot.
2. Core reads `config.json` or `model_index.json`.
3. Each lightweight `families/*/support.py` describes the metadata it owns and
   the tasks it implements.
4. Exactly one family must match. The family default task is used unless
   `--task` selects another task declared by that same family.
5. Core imports only `families.<family>.model`.
6. Core calls the module's plain `build(request, writer)` function once.
7. The family constructs TensorRT networks, maps weights, and streams named
   sections through `BundleWriter`.
8. The writer publishes the completed output path only after a successful
   build.

There is no builder inheritance, central family registry, provider probe,
profile digest, retry, or fallback.

## BuildRequest

The shared request contains only fields with a current cross-family meaning,
including model directory, output path, family, task, precision, backend,
shape bounds, parallel sizes, quantization, optional FP32 layer selections,
dynamic KV opt-in, and the concrete graph-transform callback used by BYOK.

A family must either implement a requested value or reject it. Do not add an
options bag or a configuration registry for a family-specific idea.

## Family dependencies

A family may declare build/reference/test packages in
`families/<family>/requirements.txt`. The common environment uses one pinned
base image. Adding a family dependency does not create another image digest or
central dependency catalog.

## Failure behavior

The pipeline fails closed for an unknown model, ambiguous ownership,
unsupported task or option, family import error, TensorRT build error, or
bundle write error. It never tries a sibling family and never silently changes
the requested backend or precision.
