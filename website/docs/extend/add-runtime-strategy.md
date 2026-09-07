---
title: Extend a Family Runtime
---

The runtime-strategy registry described by earlier versions of this page was
removed by PR #1093. A bundle now names one family and task; the loader opens
that family's DSO and calls its single factory.

Use this guide to add native behavior to an existing family. For a new model
owner, follow [Add a Model Family](add-model-family.md).

## 1. Choose the existing Task contract

Start from the user-visible operation in
`core/runtime/include/trtmc/task.h`. Implement an existing interface whenever
it expresses the requested behavior. Add a shared Task interface only for a
genuinely new application contract, not to expose family internals.

Record the family, task string, bundle sections, engine bindings, request and
result types, preprocessing/postprocessing, and validation oracle before
editing.

## 2. Implement inside the family

Keep every model-specific source below:

```text
families/<family>/runtime/
```

The family factory exported as `trtmc_create_family` receives a read-only
`BundleReader` and abstract `IBackend`. It constructs a concrete Task
implementation, creates engines through the Engine API, and owns all model
state and orchestration. It must not import, include, or link another family,
the loader implementation, or application code.

Add sources, private dependencies, and native tests in that same family's
`runtime/CMakeLists.txt`. The root CMake glob discovers the directory; do not
add a central source or strategy entry.

## 3. Keep the bundle contract family-local

Update the same family's `model.py` to write any new sections. Shared bundle
code treats names and payloads as opaque. The factory validates the expected
task and required section shapes and fails explicitly on a mismatch.

## 4. Validate

```bash
python3 -m tools.model_ci validate
python3 tools/test_impact.py --validate
python3 -m pytest families/<family>/tests -m 'not gpu and not trt'

cmake -S . -B build -DTRTMC_BUILD_TESTS=ON
cmake --build build --target trtmc_model_<family>
ctest --test-dir build --output-on-failure
```

Then run the family-owned `test_e2e.py` with the exact checkpoint, compiled
CLI, runtime root, and selected testcase. The final evidence must prove bundle
construction, exact DSO loading, the public Task method, and the declared
model-owned oracle.
