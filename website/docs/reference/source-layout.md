---
title: Source Layout
description: Repository ownership and the allowed dependency direction between core, families, applications, and tools.
---

```text
core/
  builder/                  Python build control plane and its tests
  runtime/                  C++ bundle, Task, Engine, loader, and backend code

families/<family>/          complete model-owned vertical slice
  support.py                exact checkpoint identity and supported tasks
  model.py                  plain build(request, writer) entrypoint
  requirements.txt          optional family-only dependencies
  runtime/                  one libtrtmc_model_<family>.so
  tests/                    manifests, oracles, thresholds, fixtures, units

apps/
  cli/                      native CLI and private file/media adaptation
  benchmark/                benchmark worker, reports, and performance policy
  devtoolkit/               developer environment helper

examples/                   public-API consumers, including BYOK and hardware apps
requirements/base.txt       thin build-tool delta over the pinned GPU base
tools/                      repository validation and CI orchestration
website/                    documentation generated from current family ownership
```

## Shared build code

`core/builder/tensorrt_model_connect/` contains only the minimum build
contracts and mechanics:

- model metadata loading and exact family resolution;
- `BuildRequest` and one selected family `build()` call;
- streaming `BundleWriter`;
- the explicit BYOK graph-transform callback; and
- the Python command-line entrypoint.

It does not contain family config, weight mapping, graph helpers, tokenizers,
model registries, or builder inheritance.

## Shared runtime code

`core/runtime/include/trtmc/` is the public native surface. The implementation
is split into bounded bundle I/O, stable device/engine primitives, the exact
family/backend loader, and TensorRT backend DSOs.

Family DSOs depend on the factory, `BundleReader`, abstract Task interfaces,
and abstract Engine API. They do not link the loader implementation, concrete
backend implementation, or sibling families.

## Dependency direction

```text
examples/apps -> public APIs
family build  -> build contract, BundleWriter, TensorRT build API
family DSO    -> family factory, BundleReader, Task API, Engine API
backend DSO   -> Engine API
```

There is no reverse dependency from core or a family to `examples/` or
`apps/benchmark/`. A normal model contribution changes only
`families/<family>/**`.

No production source lives under the retired `python/`, `src/`, `include/`,
`tests/`, `benchmarks/`, or `scripts/` roots.
