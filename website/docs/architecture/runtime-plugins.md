---
title: Family Runtime DSOs
---

The pre-#1093 runtime-plugin registry no longer exists. The physical load unit is
one family DSO named `libtrtmc_model_<family>.so`.

The bundle header names exactly one family and backend. The loader opens that
family DSO from the selected runtime root, resolves `trtmc_create_family`, and
receives an implementation of an abstract Task interface. A family owns its
factory, pipeline, preprocessing, postprocessing, dispatch, bindings, state,
samplers, and any genuinely model-specific TensorRT plugin.

There is no registrar macro, runtime-strategy map, central manifest, sibling
fallback, or hot plugin marketplace. To extend an existing family, edit only
its `families/<family>/runtime/` implementation and family-owned tests. To add
a new family, follow [Add a Model Family](../extend/add-model-family.md).

## Three layers, one public binary boundary

The new Task SDK is experimental until its stable release is explicitly
announced. It separates the user ABI from release-coupled implementation code:

```text
Application
    | depends on
    v
Header-only C++ SDK: trtmc.hpp
    | depends on
    v
C ABI: trtmc.h + libtrtmc_c.so.1
    | depends on
    v
Internal abstract Task interfaces
    ^ implements and depends on
    |
Model family
    | depends on
    v
Engine / backend contracts
```

Nodes follow control flow from top to bottom; arrows represent dependencies.
The C implementation directly invokes an internal virtual method implemented
by the selected family. It does not include that family's headers and does not
introduce another adapter layer. A family may implement several independent
Task interfaces through multiple inheritance.

Only the C boundary is intended to become a stable binary interface. Runtime,
backend and family DSOs use internal C++ contracts and ship together. Users
compile the header-only convenience API with their own C++17 compiler; no STL
object, exception or C++ class layout crosses the public binary boundary.

The only exported SDK symbol is `trtmc_get_api`. It returns a versioned core
function table; the loaded model supplies access to versioned Task tables.
Tables contain ordinary C function pointers. Every call checks the actual
model's support, not just whether a caller previously obtained a table.
Core and Task versions are negotiated separately. Published array-element
layouts remain fixed; a breaking change requires a new contract version.

## Calling a migrated family

For a bundle whose family has implemented `TextContinuation`:

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

Use `find_package(trtmc CONFIG REQUIRED)` and link `trtmc::c`. The SDK-only
consumer does not need CUDA or TensorRT development headers. An omitted runtime
root uses the installed C library directory; an explicit `LoadOptions` root
selects another installation. The installation must still contain the selected
family and backend DSOs.

Request data and synchronous Config views are borrowed only until the call
returns. Results own their storage; views borrow their result owner. A stream
or state handle retains its model and family DSO. Streaming families must copy
or parse all retained inputs before returning from `start`. C callers release
owned result/error/session handles explicitly; C++ callers use RAII. Errors
become status codes and optional owned details, including allocation failures.

## Family-owned configuration and execution

Single, native batch and streaming are separate Tasks. Shared code never
implements a batch by looping over single-request inference. Required images,
audio, frames, camera/action sequences and state remain typed Task operands.

Each loaded family declares accepted Config keys for each Task. The carrier
supports int64, float64, bool, string and homogeneous lists of integers, floats
or strings. Defaults, semantic validation and execution belong to the family.
Missing is distinct from explicit zero, false, empty string or empty list.
Unknown keys, duplicates, wrong types and unsupported combinations fail;
shared code does not silently filter supplied controls.

An optional key using an existing value kind can be added entirely inside the
family: declare it, parse it, consume it and test it. No shared config registry
or new Task table is needed. Changes to required input/output or lifecycle are
different contracts, not configuration keys.

During family-by-family conversion, applications select the existing or new
path from the bundle's primary Task **before** execution. An SDK error never
retries the old method. Existing paths are temporary migration work, not an
additional public compatibility promise, and disappear with the last family
using them. A family migration must qualify its real inference path and retain
its existing examples, benchmark behavior and validation coverage; CPU protocol
fixtures do not establish model quality or GPU support.
