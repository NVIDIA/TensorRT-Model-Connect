---
title: Add a Model Family
---

Every model is one independently owned source tree:

```text
python/tensorrt_model_connect/models/<owner>/
  MODEL.toml
  __init__.py
  model.py
  <Python build helpers>
  runtime/
    plugin.cpp
    pipeline.h
    pipeline.cpp
    <model state, samplers, and CUDA sources>
  tests/
    test_<owner>_e2e.py
    manifests/<case-name>.json
    cpp/<focused native tests>
    <Python tests, assets, thresholds, and local E2E helpers>
  tools/                         # optional
```

Adding a normal model must not require a second owner directory, a central
model list, a compatibility alias, or imports/includes from a sibling model.
Copy the closest existing owner and keep only the files the new implementation
needs.

## 1. Create the owner and descriptor

Use the same ID for the directory, Python builder, native target, and E2E
ownership. A minimal descriptor combines all discovery surfaces:

```toml
id = "example"
aliases = ["example", "ExampleModel"]
prefixes = ["example"]

runtime_plugins = ["plugin.cpp|register_example_plugin"]
runtime_strategies = ["example_decoder_kv_cache"]

test_manifests = [
  "tests/manifests/example-small-fp16.json",
]

[e2e_defaults.text_generation_causal]
reference_backend = "hf_transformers"
oracle_level = "L1_external_reference"
stages = [
  { name = "full_generation", required = true },
]
```

Do not declare `runtime_library`: CMake derives
`libtrtmc_model_<owner>.so` from the canonical owner ID. Do not add legacy
runtime-strategy aliases. Unsupported legacy inputs should fail rather than
route through another owner name.

Add `architecture_patterns`, `diffusion_pipeline_classes`, config adapters,
Python profiles, runtime config schemas, or model-owned runtime tests only when
the implementation needs them. The directory is the registry; optional files
are discovered relative to this descriptor.

## 2. Implement the Python build entry

`model.py` provides two direct module functions:

```python
def matches(config) -> bool:
    ...


def build(model_dir, output_path, **options):
    # config -> weights -> engines/components -> complete bundle
    ...
```

The shared builder resolves one descriptor, imports this module, and calls
`build()` exactly once. The owner decides precision defaults, checkpoint
mapping, graph construction, parallel layout, quantization policy, component
assembly, and bundle sections. It must either consume a requested option or
reject it explicitly; it must not silently fall back to another builder.

Keep helpers local to the owner. Similar code in another model is not a reason
to create a shared framework. Promote code only when it is a stable,
model-independent format or runtime primitive whose change should revalidate
all models.

## 3. Implement the native runtime

Place the native implementation under the owner's `runtime/` directory. The
registrar must claim a strategy listed in the same `MODEL.toml`:

```cpp
REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(
    register_example_plugin,
    ExamplePlugin,
    "example_decoder_kv_cache");
```

Model-local headers use the owner runtime include directory; never include a
sibling model's runtime source. Shared registry, loader, backend, bundle, and
device interfaces remain under `include/trtmc/` and `src/runtime/`.

Declare the runtime target and focused C++ tests in `runtime/CMakeLists.txt`;
store test sources under `tests/cpp/`:

```cmake
add_library(trtmc_model_example SHARED
  pipeline.cpp
  plugin.cpp
  ${TRTMC_MODEL_example_REGISTRATION_SOURCE}
)

if(TRTMC_BUILD_TESTS)
  trtmc_add_test(test_example_pipeline
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_example_pipeline.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED REQUIRES_TRT
  )
  target_link_libraries(test_example_pipeline PRIVATE trtmc_model_example)
endif()
```

## 4. Add the exact E2E contract

Store each manifest under
`python/tensorrt_model_connect/models/<owner>/tests/manifests/` and list its
owner-root-relative path in `test_manifests`. A buildable native manifest needs a unique `name`,
concrete `hf_id`, logical `family`, exact `runtime_strategy`, generic
`task_strategy`, build settings, and at least one real testcase and oracle.

The manifest's `task_strategy` selects the shared runner contract, CLI command
set, and default performance mode. Put `performance_mode` in the root
descriptor only when this owner genuinely differs from that task default. If
the owner supports optional diff-framework diagnostics, declare their concrete
class names locally with `diff_framework_check_classes`; owners without those
diagnostics need no central exemption entry.

Copy the small `test_<owner>_e2e.py` entry point from the closest owner. Keep
inputs, local runners, references, thresholds, and assets in the same
`tests/` tree. The shared E2E harness supplies generic orchestration; it does
not own model-specific behavior.

## 5. Validate the owner

From the repository root:

```bash
PYTHONPATH=python:. python3 -m pytest \
  tests/builder/test_unified_model_layout.py \
  tests/builder/test_model_owned_architecture.py -q

python3 tools/model_ci.py validate
python3 tools/check_runtime_strategy_matrix.py

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target trtmc trtmc_model_example -j
```

Build and run the exact declared E2E case:

```bash
E2E_MODEL=example-small-fp16
ENGINE_DIR=/tmp/trtmc-example-engines

PYTHONPATH=python:. python3 -m pytest \
  "python/tensorrt_model_connect/models/example/tests/test_example_e2e.py::test_model_e2e[${E2E_MODEL}]" \
  --e2e-model "$E2E_MODEL" \
  --engine-dir "$ENGINE_DIR" \
  --trtmc-binary ./build/trtmc \
  --model-plugin-dir ./build/models \
  --rebuild-engines -v
```

A successful compile proves construction only. Model support requires the
owner's real build, native DSO load, task execution, and declared oracle to
pass on the required environment.

## Optional exact optimized implementation

An exact-qualified optimized implementation also stays under this owner, for
example:

```text
python/tensorrt_model_connect/models/<owner>/<implementation>/
  IMPLEMENTATION.toml
  profiles/*.toml
```

It is an additional artifact contract, not a second model owner or a fallback
route. Keep its adapter and qualification tests under the same owner tree.

{/* Collaborative review anchor: batch 2. */}
