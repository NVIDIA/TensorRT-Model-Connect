---
title: Add a Model Family
---

Create one directory:

```text
families/my_family/
├── __init__.py
├── support.py
├── model.py
├── requirements.txt       # optional; only this family's extra dependencies
├── runtime/
│   ├── CMakeLists.txt
│   └── plugin.cpp
└── tests/
    ├── test_support.py       # dependency-free identity/default-task policy
    ├── test_e2e.py
    ├── manifests/<case>.json
    └── thresholds/<testcase>.json  # optional numeric override
```

`support.py` owns checkpoint identity and task capabilities without importing
the family implementation or its dependencies:

```python
from tensorrt_model_connect.model_support import family_support

describe = family_support(
    model_types=("my_model",),
    architectures=("MyModelForGeneration",),
    tasks=("text_continuation",),
    default_task="text_continuation",
)
```

Support uses exact normalized identity only. Do not add broad prefixes,
priority, scoring, or fallback. Zero matching families means unsupported;
multiple matching families are an ownership error. `support.py` may import only
the shared support contract. When an upstream repository has no standard model
identity field, write a small `describe(metadata)` function that checks its
exact family-owned root JSON shape or declare a minimal set of exact sentinel
files; do not match the repository name.

`model.py` must expose exactly one plain function. Any helper it calls is
implemented in this file or elsewhere in the same family directory:

```python
def _build_my_family_engine(request):
    # Family-owned TensorRT graph, weight mapping, and serialization.
    ...


def build(request, writer):
    if request.context_parallel_size != 1:
        raise ValueError("my_family does not support context parallelism")
    if request.task != "text_continuation":
        raise ValueError("my_family supports only text_continuation")
    if request.backend != "trt":
        raise ValueError("my_family supports only the trt backend")
    writer.set_header(family="my_family", task=request.task, backend=request.backend)
    writer.add_json("runtime.json", {"tensor_parallel_size": 1})
    writer.add_bytes("engine.plan", _build_my_family_engine(request))
```

The builder must not inherit from a base class. It may import the shared
`BuildRequest` and `BundleWriter` contracts, but it must not import another
family or shared model implementation.

The snippet is a structural example, not a model builder. A real implementation
must consume or explicitly reject every build option it receives; it must not
silently discard dimensions, precision, parallelism or backend choices.

If build, official reference, or E2E code needs a package outside the pinned
base environment, put the ordinary pip requirement directly in the optional
root `requirements.txt`. Do not add a project extra, shared family lock,
profile, inheritance, hash, or include of another family's file. Families with
no extra dependency omit the file.

The runtime CMake file creates `trtmc_model_my_family`. Its factory exports
`trtmc_create_family`, reads only sections owned by this family, and returns a
concrete `trtmc::internal::IModel` implementing the selected semantic Task
interfaces from `trtmc/internal/`. For example, these are the declarations a
text family implements; model execution and config parsing stay in its own files:

```cpp
#include <trtmc/internal/model.h>
#include <trtmc/internal/text.h>

class MyModel final : public trtmc::internal::IModel,
                      public trtmc::internal::ITextContinuation {
public:
    const char* task() const noexcept override {
        return trtmc::internal::ITextContinuation::kTask.data();
    }
    std::vector<trtmc::internal::TaskInfo> task_info() const override;
    std::vector<trtmc::internal::ConfigField>
    config_fields(std::string_view task_id) const override;
    trtmc::internal::TextResult run(
        const trtmc::internal::TextContinuationRequest&,
        trtmc::internal::ConfigView) override;
};
```

The internal interfaces never include or link a concrete family. The shared C
entry point directly invokes the implemented virtual method. Do not add a
family-specific C export, shared adapter, or user-facing family header. Multiple
inheritance is allowed for distinct runtime Tasks; builder inheritance is not.

`support.py` declares accepted **build/primary Task modes**. `IModel::task_info()`
declares all semantic Tasks available on this particular **loaded bundle**;
these sets need not be identical. The primary `task()` must match the bundle
header, and every advertised semantic Task must actually be implemented.

If the family graph contains distributed collectives, that same family owns
its communicator setup and NCCL loading. A replicated plan that only selects a
rank-specific section must not load NCCL.

`FamilyContext.reader` is read-only. A factory normally consumes its sections
before returning. If a pipeline needs deferred section loading, copy the
`BundleReader` value into that pipeline; do not retain a reference to the
factory context.

Adding the directory must not require editing a core registry, source list, or
strategy map. Run:

```bash
python -m pip install -r families/my_family/requirements.txt  # only if present
python tools/test_impact.py --validate
```

`test_e2e.py` directly builds the bundle and invokes the native CLI through
`TRTMC_BINARY` and `TRTMC_RUNTIME_ROOT`. It runs exactly the family-owned oracle
declared by the case: an official reference when required, otherwise explicit
runtime or task invariants. It never falls through to a generic comparator. A
threshold sidecar exists only when the case overrides a numeric default, and
contains only values the family test reads. It must not import a central runner,
comparator, or sibling-family fixture.

The expected diff boundary for a normal family contribution is
`families/my_family/**`. Examples, benchmarks, and BYOK are optional consumers
of public APIs; neither a family nor core may import their implementation.

## Migrate an existing family to the Task SDK

Keep the migration in that family's directory:

1. Select the existing internal contracts that match the actual required
   operands, outputs and lifecycle. Batch is not a loop over the scalar API;
   prompt images, prior masks, frames and state are not Config values.
2. Implement `IModel` and those Tasks on the family pipeline. Declare support
   for the loaded checkpoint/bundle, not every interface present in the DSO.
   Keep defaults, configuration validation and model execution family-owned.
3. Update `support.py`, the builder's bundle primary Task and the affected
   family manifests together. Retaining an old primary ID while changing only
   the DSO selects the old application path; do not rely on a failed call to
   discover the new API. Existing unused primary modes must not remain advertised.
4. Return complete typed outputs: preserve raw tensors, token/feature axes,
   clocks, masks and actual scores. Initial detector boxes are not newly tracked
   boxes; an independent point forecast is not an interpolated median. Do not
   invent unavailable values to fill a result.
5. Keep checkpoint identity/default assertions in `tests/test_support.py`;
   the existing CPU pre-merge invocation discovers these files without a central
   family list. Update the family's runtime tests, manifests, fixtures and
   oracles without weakening their passing criteria.
6. Prove one minimal build/load/run before expanding. Run existing examples and
   benchmark workloads through the unchanged shared applications, plus direct
   C/C++ calls for the newly implemented Tasks. Then qualify the actual model's
   numerical, device and performance behavior in its family tests.

The shared SDK contains more Tasks than the CLI or benchmark frontends expose.
Preserve each family's existing application workloads during migration; do not
claim an unrelated new CLI command exists just because its SDK Task is defined.
A new Task contract or new frontend workload is a separate shared change, not
an excuse to add a cross-family implementation dependency.

Runtime, backend and family DSOs ship together. The public binary boundary is
the [C SDK](../api/cpp-api.md), with the convenience C++ header compiled by the
application. The old internal interfaces can be removed after their last family
has migrated; they are not a promised backward-compatible user ABI.

An E2E manifest may declare an exact `hf_id` (and, when available,
`hf_revision`) or omit `hf_id` for a prepared local checkpoint supplied through
the family-specific model-directory environment variable. Do not invent an HF
ID for a checkpoint that is not published there.
