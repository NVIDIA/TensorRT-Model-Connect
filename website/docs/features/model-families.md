---
title: Model Families
---

A model family is one complete, self-owned vertical slice:

```text
families/<family>/
├── support.py
├── model.py
├── requirements.txt       # optional
├── runtime/
└── tests/
```

## What a family owns

- `support.py` matches standard checkpoint metadata and returns the supported
  Task identities and one meaningful default.
- `model.py` exposes one plain `build(request, writer)` function. Inheritance is
  forbidden.
- `requirements.txt`, when present, contains only that family's extra build,
  reference, or test dependencies.
- `runtime/` builds `libtrtmc_model_<family>.so` and implements one or more
  abstract interfaces from `trtmc/task.h`.
- `tests/` owns exact checkpoint manifests, thresholds, reference logic, Python
  checks, and optional C++ checks.

The family may duplicate another family's implementation, but it must not
import, include, link, or otherwise depend on that sibling.

## Discovery and build

The resolver reads `config.json`, `model_index.json`, and snapshot filenames,
then asks every dependency-light `support.py`. Zero matches and multiple
matches are errors. Once one family is selected, the core imports only that
family's `model.py` and calls it once.

There is no `MODEL.toml`, family plugin base class, priority list, registry, or
compatibility scan.

```bash
PYTHONPATH=core/builder:apps/benchmark:. \
  python3 -m tools.model_ci validate
```

The validator checks the physical ownership contract. The live model inventory
is generated from `families/*/tests/manifests/*.json`.

## Native runtime

At load time the bundle names one family, task, and backend. The explicit loader
opens only that family DSO and backend DSO. The family factory receives a
read-only bundle reader and the abstract Engine backend, then returns an
abstract Task interface.

```text
ITextGeneration <|.. QwenTextPipeline
ITextGeneration <|.. LlamaTextPipeline
ITranscription  <|.. WhisperPipeline
```

The hollow-triangle relation means each concrete pipeline implements the
abstract interface. The interface never depends on a model.

## Add or change a model

A normal contribution modifies only `families/<family>/**`. It should not
require a core registry edit, shared source list, sibling-family edit, or a
cross-family helper. See [Add a Model Family](../extend/add-model-family.md).

Support is exact. A nearby fine-tune, precision, topology, or task is not
supported merely because another manifest in the same family passes.
