---
title: TriAttention
---

import Diagram from '@site/src/components/Diagram';

TriAttention is a cache-compaction design that scores older KV rows, protects
selected prompt and recent rows, and compacts the remainder before the logical
budget is exceeded.

:::caution Historical design, not current support

The current repository does not contain a TriAttention builder, runtime, CLI
option, or family manifest. The previous shared config and runtime path was
removed by the family-isolation refactor. Do not pass the old TriAttention
flags or treat this page as a supported runbook.

:::

## Design idea

The retained diagram explains the model-level algorithm, not the current source
layout:

<Diagram
  src="/img/diagrams/features/triattention-runtime-sequence.svg"
  alt="Conceptual TriAttention sequence from candidate scoring through row selection and cache compaction"
  caption="Historical algorithm sketch. No current family declares or tests this feature."
  sequence
/>

The important limits remain distinct:

- the TensorRT engine's legal shape capacity;
- the memory allocated by the runtime;
- the logical number of KV rows retained by a compaction policy.

A logical policy cannot increase the engine's compiled shape limit or prove that
a long prompt fits in the physical allocation.

## Requirements for a future family-owned implementation

If a model family needs this behavior, the smallest acceptable implementation
would stay under that one family:

```text
families/<family>/
├── model.py                 # writes any family-owned policy sections
├── runtime/                 # scoring, selection, compaction, and bindings
├── requirements.txt        # only if calibration needs extra packages
└── tests/                   # exact dense and compacted comparisons
```

The family runtime would implement the existing abstract Task and Engine
interfaces. It must not add a central config schema, shared model cache policy,
base class, sibling-family dependency, or content digest.

## Evidence required before claiming support

1. An exact checkpoint and family-owned calibration input, if calibration is
   required.
2. A dense control and compacted bundle built from the same revision.
3. Identical prompts, sampling controls, token limits, and hardware.
4. Runtime evidence that compaction actually ran.
5. Family-owned parity or task-quality thresholds.
6. Separate synchronized measurements for any performance claim.

Until one family owns that complete path and its manifest passes, TriAttention
remains design context only.
