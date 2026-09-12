---
title: C and C++ Task SDK
---

The new SDK is experimental until its first stable release is announced.
It applies to families migrated to the internal semantic Task contracts.
Adding a shared contract does not advertise it on an unmigrated model.

## One binary boundary

Applications use the C ABI directly, or compile the header-only C++17 wrapper.
The C implementation calls an internal abstract Task implemented by the loaded
family. There is no intermediate adapter framework and no public C++ vtable ABI.

```text
application
    | depends on
    v
trtmc.hpp: header-only C++ convenience API
    | depends on
    v
trtmc.h + libtrtmc_c.so.1: public C ABI
    | depends on and invokes
    v
internal abstract Task
    ^ implements and depends on
    |
model family
```

Runtime, backend and family DSOs use release-coupled internal C++ contracts
and must be upgraded together. Applications do not include family headers.

## C++ application

For a model that advertises `TextContinuation`:

```cpp
#include <trtmc/trtmc.hpp>
#include <iostream>

int main() {
    auto model = trtmc::Model::load("model.bundle");
    auto text = model.task<trtmc::TextContinuation>();
    auto result = text.run({"Hello"});
    std::cout << result.text();
}
```

Build using the installed SDK, not internal runtime headers:

```cmake
cmake_minimum_required(VERSION 3.20)
project(my_app LANGUAGES CXX)
find_package(trtmc CONFIG REQUIRED)
add_executable(my_app main.cpp)
target_compile_features(my_app PRIVATE cxx_std_17)
target_link_libraries(my_app PRIVATE trtmc::c)
```

User compilation does not require CUDA or TensorRT development headers. The
deployed runtime still needs its native dependencies and selected family/backend
DSOs. With no explicit runtime root, the SDK uses its own library directory;
`LoadOptions::runtime_root` selects another installation.

`model.tasks()` lists the loaded model's supported contracts. Use
`model.supports<Task>()` before selecting an optional Task. A family can implement
several Tasks, but only advertises those supported by this particular bundle.
Single request, native batch and streaming are separate execution contracts;
required images, frames, state and actions remain typed operands.

Selecting another Task does not change the bundle's primary Task or rebuild it.
The native CLI uses `--task TASK_ID` for explicit selection. Benchmark testcases
can declare `selected_task` in their family-owned manifest, or use
`trtmc-bench run --task TASK_ID` with the usual model and input options.
Selection stays outside Config and must name an interface bound by the loaded
model; unsupported selections fail without retrying another Task.

## C calling convention

`trtmc_get_api(1, 0, &core)` is the only exported SDK symbol. It returns a
versioned table of ordinary C function pointers. C11 callers then use:

1. `core->model_load(...)` to obtain an opaque model handle.
2. `core->model_get_task_api(model, task_id, 1, 0, ...)` to obtain the exact
   typed Task table. Check its version and `byte_size` before accessing fields.
3. A typed Task method, such as `text->run(...)`, followed by `result_view(...)`.
4. The matching release functions for owned results, sessions, models and errors.

The installed `trtmc/trtmc.h` defines every core operation; group headers define
the complete Task tables and layouts. `trtmc/types.h` defines status codes,
fixed-width integer fields and the seven Config value kinds. Public signatures
use only C data, pointers and opaque handles: no STL containers, exceptions,
references or compiler-sized C enums cross the boundary. `TRTMC_CALL` selects
the platform's native C convention (`__cdecl` on Windows); the current runtime
build and loader target ELF platforms. This is not a serialized wire protocol
or a promise of binary interoperability across different machine architectures.

Always check status, even when the returned error handle is null: an allocation
failure can prevent error-detail allocation. C++ wrappers turn failures into
`trtmc::Error`, retaining the C status, and release handles through RAII.

## Config and ownership

`text.config_fields()` returns the family-declared fields for that loaded Task.
The caller may pass an owned `trtmc::Config`; the Core Runtime's C entry point
transports typed key/value views without imposing sampling defaults. It checks
names, duplicates and types against the family-declared field table before
execution. The Family Runtime supplies defaults and validates ranges and
combinations. Missing is different from explicit zero, false or empty.
There is no model-global configuration merger or silently ignored option.

An embedding result may have an empty `embedding_space` when a local checkpoint
has no known space identifier. Its vectors, pooling and normalization are still
available; two empty identifiers do not imply cross-model compatibility. Families
must not fabricate identifiers or require a hash just to compute an embedding.

`SeriesToRegressionValues` consumes one `SeriesHistory` and returns finite
predictions indexed by target, with optional target names and units. It has no
forecast horizon or probability-distribution parameters. It uses the existing
`forecast --input` CLI and `regress` benchmark operation; independent-series
batching is not implied by the scalar Task.

Family Runtime returns `IModel::task_bindings()` records built with
`bind<SharedTaskInterface>(*this, fields)`. Core Runtime retains the correctly
adjusted interface address and immutable metadata snapshot, rather than testing
each Task with repeated RTTI casts. This does not change the release-coupled
internal C++ boundary or expose a public C++ vtable.

| Object | Lifetime |
| --- | --- |
| Synchronous inputs and Config views | Borrowed until that call returns. |
| C model metadata | Borrowed until model release; C++ metadata queries copy it. |
| Result views | Borrowed from their owned result; release the result after reading. |
| Stream/session | Retains the model and its DSO; family copies retained inputs before returning from creation. |
| Device mask view | Producer-complete at return; invalidated by the next segment attempt or session release, as its typed contract states. |
| Crop-provider buffers | Kept by the callback lease while consumed; not retained beyond the enclosing pose call. |

Stateful contracts define their own `next`, `append`, `finish`, `cancel`, reset
or clone operations. Do not replace their lifecycle with a generic `run` loop.
A live session excludes conflicting execution on its model with `TRTMC_BUSY`;
this does not add a shared scheduler or promise multi-tenant execution.

## Versions and extension

Core and Task table versions are negotiated independently. Only v1.0 is
implemented now. A successful query reports the exact requested table version;
an unknown version returns `TRTMC_VERSION_MISMATCH`, not an older fallback.

- A family can add an optional Config key using an existing value kind without
  changing shared layouts. It must declare, validate, consume and test that key.
- A family can implement another existing Task without changing core bindings.
  Its loaded-model declaration and implementation must both exist.
- A new required input/output or lifecycle needs a separate reviewed Task
  contract, not an arbitrary JSON/Config operand or a claimed Cartesian product
  of existing modalities.
- After stable publication, existing layouts and array-element strides remain
  fixed. Additive table operations need a new negotiated table revision while
  retaining the published revision; breaking layouts or meanings need a new
  major contract. Do not append fields to array elements in place.

Header-only ranking and quantile-summary helpers operate on existing results;
they do not add another model call. A median derived from quantiles is not the
model's independent point estimate. Class scores require interpretable labels
or vocabulary identity, rather than anonymous ordinals.

## Migration boundary

The older `load_task()` and `trtmc/task.h` interfaces remain release-coupled
implementation paths only while existing families are migrated. Applications
select the old or semantic path from the bundle's primary Task before execution;
an SDK failure never retries an older method. This is not a compatibility ABI
promise. Existing CLI commands, examples and benchmark workloads remain usable
for their existing families during that conversion.

The SDK has more Task contracts than the CLI and benchmark frontends have
commands. Do not infer frontend support from the presence of a C table. A
migration preserves its existing workload and tests; a newly introduced
application workload is a separate frontend change. See
[Add a Model Family](../extend/add-model-family.md) for the family-only checklist.

### Existing object-detection bundles

Bundles whose header task is `object_detection` expose the existing
`IObjectDetection` runtime interface. The family owns its score calibration and
returns only the boxes it keeps. Until that family migrates to the SDK, use its
release-coupled runtime headers:

```cpp
#include <trtmc/runtime/family_loader.h>
#include <trtmc/task.h>

auto task = trtmc::load_task("detr-resnet-50.bundle", "/opt/trtmc/lib");
auto* detector = dynamic_cast<trtmc::IObjectDetection*>(task.get());
if (detector == nullptr) throw std::runtime_error("not an object-detection bundle");
auto result = detector->detect(pixels.data(), height, width);
```

The loader verifies that `ITask::task()` exactly matches the bundle header.
The command-line equivalent is:

```text
trtmc detect detr-resnet-50.bundle --runtime-root /opt/trtmc/lib --image input.jpg
```

`load_task()` lives in `libtrtmc_runtime.so`; bundle reading and engine
primitives live in `libtrtmc_core.so`. A family factory receives a
`FamilyContext` containing a `const BundleReader&` and an abstract `IBackend&`.
The reader exposes immutable metadata and on-demand section reads only. The
factory must copy the lightweight reader into its pipeline if it will read a
section after the factory returns; it must never retain the context reference.
