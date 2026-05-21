---
title: Add a Runtime Strategy
---

Add a runtime strategy when an existing `IPipeline` contract cannot represent the model's execution behavior.

## 1. Add or reuse a pipeline

If the task contract is new, add a model runtime folder under `src/runtime/models/`. Override only the `IPipeline` methods the pipeline supports.

## 2. Add a plugin

Create `src/runtime/plugins/<strategy>_plugin.cpp` implementing `IPipelinePlugin`.

The plugin should:

1. Parse strategy-specific config from `ctx.config_json`.
2. Extract required sections from `ctx.bundle`.
3. Create tokenizers, engines, caches, schedulers, and domain plans.
4. Return a concrete `IPipeline`.

## 3. Register through the manifest

Add one line to `cmake/trtmc_pipeline_plugins.cmake`:

```cmake
"my_strategy_plugin.cpp|register_my_strategy_plugin"
```

In the plugin source, register strategies:

```cpp
REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(
    register_my_strategy_plugin,
    MyStrategyPlugin,
    "my_runtime_strategy");
```

## 4. Emit the strategy from the builder

Set `runtime_strategy` in the Python family plugin so bundle `config.json` selects the new runtime path.

## 5. Test

Add focused C++ tests for the plugin and pipeline, then add an E2E manifest that proves the full build/run contract.
