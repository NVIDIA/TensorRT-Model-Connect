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
    tasks=("text_generation", "embedding"),
    default_task="text_generation",
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
    if request.task != "text_generation":
        raise ValueError("my_family supports only text_generation")
    writer.set_header(family="my_family", task=request.task, backend="trt")
    writer.add_json("runtime.json", {"tensor_parallel_size": 1})
    writer.add_bytes("engine.plan", _build_my_family_engine(request))
```

The builder must not inherit from a base class. It may import the shared
`BuildRequest` and `BundleWriter` contracts, but it must not import another
family or shared model implementation.

If build, official reference, or E2E code needs a package outside the pinned
base environment, put the ordinary pip requirement directly in the optional
root `requirements.txt`. Do not add a project extra, shared family lock,
profile, inheritance, hash, or include of another family's file. Families with
no extra dependency omit the file.

The runtime CMake file creates `trtmc_model_my_family`. Its factory exports
`trtmc_create_family`, reads only sections owned by this family, and returns a
concrete implementation of an abstract interface in `trtmc/task.h`.
The family pipeline depends on and implements `trtmc/task.h`; `trtmc/task.h`
never includes or links a family.

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

An E2E manifest may declare an exact `hf_id` (and, when available,
`hf_revision`) or omit `hf_id` for a prepared local checkpoint supplied through
the family-specific model-directory environment variable. Do not invent an HF
ID for a checkpoint that is not published there.
