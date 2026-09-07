---
title: Runtime Lifecycle
---

The native runtime performs one exact load and returns an abstract Task:

```cpp
auto task = trtmc::load_task("model.bundle", "/opt/trtmc/lib");
```

## Load sequence

1. `BundleReader` validates the format-1 header and all section bounds.
2. The loader validates the `family` and `backend` names as safe DSO tokens.
3. It loads `libtrtmc_backend_<backend>.so` from the explicit runtime root.
4. It loads `libtrtmc_model_<family>.so` from that same root.
5. It resolves the single `trtmc_create_family` factory and passes a
   `FamilyContext` containing the read-only bundle reader and abstract backend.
6. It verifies that the returned `ITask::task()` matches the bundle header.

There is no current-directory search, environment fallback, registry lookup,
strategy switch, sibling-family probe, or load retry.

## Ownership after transfer

The selected family implements one or more interfaces in
`core/runtime/include/trtmc/task.h`. It owns preprocessing, postprocessing,
request orchestration, tokenizer/sampler state, family sections, and engine
binding. It creates engines only through the abstract `IBackend`/`IEngine`
contract. The backend owns TensorRT runtime objects, not model policy.

`FamilyContext.reader` is read-only. A factory may consume its sections before
returning or copy the lightweight `BundleReader` into the pipeline for deferred
reads; it must not retain a reference to the temporary factory context.

## Optional load settings

Runtime-sized KV capacity is passed directly to compatible families.
TensorRT-RTX runtime cache and CUDA graph settings are accepted only when the
bundle selects `trt_rtx`; the standard backend rejects them. TVM-FFI BYOK is an
explicit extension DSO and three-part binding, not a general plugin registry.

## Teardown

Applications destroy Task objects before the loaded family and backend
libraries leave scope. Families release their streams, buffers, communicators,
engines, and family-local state; the loader owns the dynamic-library handles.
