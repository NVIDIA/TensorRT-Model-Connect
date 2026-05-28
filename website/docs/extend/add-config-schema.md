---
title: Add a Config Schema
---

Use config schemas for feature knobs that need build-time or runtime configuration.

## Why use schemas

The CLIs already expose:

```bash
--config FILE
--set NS.FIELD=VALUE
```

A schema lets new features use this generic surface instead of adding another bespoke flag.

## Where schemas live

| Layer | Path |
| --- | --- |
| C++ public generated headers | `include/trtmc/config/schemas/` |
| C++ schema implementation | `src/runtime/config/schemas/` |
| Python runtime config | `python/tensorrt_model_connect/runtime_config/schemas/` |
| CMake manifest | `cmake/trtmc_config_schemas.cmake` |

## Implementation checklist

1. Add the schema definition in the owning namespace.
2. Add parsing, defaults, validation, and type conversion.
3. Register the schema through the manifest.
4. Consume resolved values in the owning builder, plugin, or pipeline.
5. Add cross-language schema tests.
6. Add CLI tests for `--config` and `--set`.

Unknown namespaces or fields should fail fast.
