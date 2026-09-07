---
title: Units and Ownership
description: What core, each model family, applications, and artifacts own.
---

import Diagram from '@site/src/components/Diagram';

The unit of model contribution is one directory under `families/`. A normal
model PR should change that directory only.

<Diagram
  src="/img/diagrams/architecture/build-ownership.svg"
  alt="Build dependencies point from the CLI and selected family toward narrow core contracts"
  caption="Core discovers a family at runtime, while the selected family source depends on narrow build and bundle contracts."
/>

<Diagram
  src="/img/diagrams/architecture/runtime-ownership.svg"
  alt="Runtime source dependencies point from applications, families, and backends toward abstract core interfaces"
  caption="A family implements an abstract Task interface and drives an abstract Engine interface; families never depend on each other."
/>

## Ownership table

| Unit | Owns | Does not own |
| --- | --- | --- |
| `core/builder` | metadata loading, unique family resolution, `BuildRequest`, `BundleWriter`, one optional graph-transform callback | model graphs, weights, family defaults, retry or fallback |
| `core/runtime` | bounded bundle access, public Task and Engine interfaces, explicit DSO loading, TensorRT engine primitives | model pipelines, tokenization, preprocessing, request policy |
| `families/<family>` | checkpoint identity, tasks/default, graph, weights, sections, runtime pipeline, bindings, pre/postprocessing, tests, optional dependencies | sibling code, central source lists, shared model policy |
| `apps/` | CLI and benchmark applications over public APIs | behavior required by core or a family |
| `examples/` | complete example applications over public APIs | hidden runtime hooks or core dependencies |
| `.bundle` | small dispatch header and family-defined byte sections | hashes, family schemas, backward-compatibility metadata |

## Family boundary

A family normally contains:

```text
families/<family>/
├── support.py
├── model.py
├── requirements.txt   # optional
├── runtime/
└── tests/
```

`model.py` is a plain module with a `build(request, writer)` function. It must
not inherit from a builder base class. Similar family code may be copied; code
similarity alone is not a reason to create a shared dependency.

The family runtime DSO implements one or more abstract interfaces from
`trtmc/task.h`. It consumes bundle sections through `BundleReader` and
creates engines through `IBackend`. It does not link the runtime loader or
another family.

## When shared code is justified

Change core only when an existing family cannot express a genuinely
model-independent contract. The contract must already have at least two real
consumers and a stable meaning. Convenience, deduplication, or a possible
future model is not enough.

For the precise arrow semantics and control transfers, read
[AI-Native Horizontal Scaling Architecture](ai-native-horizontal-scaling.md).
