---
title: Add a Runtime Strategy
---

This page adds a **native** runtime strategy. A native runtime strategy is a
model-owned dispatch key. At configure time, each
`src/runtime/models/<owner>/MODEL.toml` claims one or more unique strategies
and maps them to one model DSO. At load time, the bundle's strategy selects
that DSO before `PipelineRegistry` looks up the registered plugin.

Use this guide to add another strategy to an existing runtime owner. For a new
model, follow [Add a Model Family](add-model-family.md); every new supported
model needs the Python family, model-owned runtime DSO, and E2E descriptor.

Do not create a synthetic native strategy for a delegated optimized runtime.
That path uses a family-owned implementation manifest/profile, an embedded
`libtrtmc_impl_*.so`, and optimized-runtime qualification evidence.

## 1. Choose owner and contracts

Before editing, record:

- runtime owner, such as `qwen`;
- unique strategy key, such as `qwen_decoder_kv_cache`;
- existing `IPipeline` method that exposes the task;
- bundle sections and config fields the plugin consumes;
- E2E `task_strategy`, which groups the user-visible task separately from
  runtime dispatch.

Do not reuse another model's strategy key. Similar implementations may share
source patterns, but a strategy can have only one manifest owner.

## 2. Implement inside the owner

Keep the plugin, pipeline, state, sampler, and model-specific helpers under:

```text
src/runtime/models/<owner>/
```

The `IPipelinePlugin::create()` implementation should:

1. Read strategy-specific fields and sections from `PipelineContext`.
2. Create backend modules through `IBackend`.
3. Construct model-owned tokenizers, state, samplers, schedulers, or
   preprocessors.
4. Return a concrete `IPipeline` that overrides only the supported task
   methods.

Register every strategy claimed by the plugin:

```cpp
REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(
    register_example_plugin,
    ExamplePlugin,
    "example_runtime_strategy");
```

## 3. Declare the model manifest

Add the plugin/registrar pair and strategy to the owner's manifest:

```toml
id = "example"
runtime_library = "libtrtmc_model_example.so"
runtime_plugins = ["plugin.cpp|register_example_plugin"]
runtime_strategies = ["example_runtime_strategy"]
```

`cmake/trtmc_pipeline_plugins.cmake` discovers model manifests automatically.
Do not add a source entry to a central CMake list. Configuration rejects a
missing source, malformed registrar pair, duplicate strategy, or strategy
without an owner. It generates the strategy-to-DSO index and one registrar
translation unit for the model DSO.

If the owner has focused C++ tests or model-owned config schemas, declare them
in the same manifest with `runtime_tests` or `runtime_config_schemas`.

## 4. Emit the exact key

The Python family plugin must emit the exact model-owned strategy in bundle
`config.json`. Keep the corresponding Python descriptor in
`python/tensorrt_model_connect/families/<family>/MODEL.toml`.

The E2E manifest then records both axes:

```json
{
  "family": "example",
  "runtime_strategy": "example_runtime_strategy",
  "task_strategy": "text_generation_causal"
}
```

`runtime_strategy` selects the implementation; `task_strategy` selects the
harness/oracle contract. They are not interchangeable.

## 5. Validate

Run repository descriptor validation first:

```bash
python3 tools/model_ci.py validate
```

Configure and build the owner target after replacing `example` with the real
owner:

```bash
cmake -S . -B build -DTRTMC_BUILD_TESTS=ON
cmake --build build --target trtmc_model_example
ctest --test-dir build --output-on-failure --no-tests=error -R 'example'
```

Finally run the exact E2E model manifest with the newly built model plugin:

```bash
ENGINE_DIR=/tmp/trtmc-engines
mkdir -p "${ENGINE_DIR}"
pytest 'tests/test_e2e.py::test_e2e[example-model]' -v \
  --engine-dir "${ENGINE_DIR}" \
  --trtmc-binary ./build/trtmc \
  --model-plugin-dir ./build/models
```

The final evidence should prove descriptor consistency, model-DSO loading,
plugin construction, the public task method, and comparison against the
manifest's declared oracle.
