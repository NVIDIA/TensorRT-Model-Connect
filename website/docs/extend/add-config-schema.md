---
title: Add a Config Schema
---

Use config schemas for feature knobs that need build-time or runtime
configuration. The public surface stays:

## Why use schemas

```bash
./build/trtmc run model.trtfb \
  --config profile.json \
  --set runtime.prefer_gpu_greedy=true
```

A schema lets new features use this generic surface instead of adding another
bespoke flag. `--set` is repeatable, and it overrides the same field from
`--config` for that invocation.

## Choose the owner

| Kind | Python source | C++ source | Registration |
| --- | --- | --- | --- |
| Shared platform/runtime feature | `python/tensorrt_model_connect/runtime_config/schemas/<namespace>.py` | `include/trtmc/config/schemas/<namespace>.h` and `src/runtime/config/schemas/<namespace>.cpp` | `cmake/trtmc_config_schemas.cmake` |
| One model family | `python/tensorrt_model_connect/families/<family>/runtime_config_schema.py` | `src/runtime/models/<owner>/config_schema.h` and `.cpp` | `runtime_config_schemas` in `src/runtime/models/<owner>/MODEL.toml` |

Do not put a single-model namespace in the shared schema directories. The
Python loader discovers family sidecars without importing every family plugin;
CMake builds a model-owned C++ schema into the same DSO as its consumer.

## Define the same contract in both languages

Both definitions must use the same:

- namespace and field names;
- type tags (`bool`, `int64`, `float`, `string`, and other supported registry
  types);
- default values;
- allowed layers;
- value validation.

Use `audio_bark` as a complete model-owned example:

- `python/tensorrt_model_connect/families/bark/runtime_config_schema.py`
- `src/runtime/models/bark/config_schema.h`
- `src/runtime/models/bark/config_schema.cpp`
- `src/runtime/models/bark/MODEL.toml`

Its runtime manifest declares:

```toml
runtime_config_schemas = [
  "config_schema.cpp|register_audio_bark_schema",
]
```

The C++ source uses
`REGISTER_CONFIG_SCHEMA_FACTORY_WITH_MANIFEST(...)`; CMake generates the call
inside the Bark model DSO. The Python sidecar registers its `SCHEMA` at module
load.

## Consume resolved values

Build-time code receives values from the Python `ConfigBundle`. A migrated C++
model plugin reads typed values from `ctx.runtime_config`:

```cpp
if (ctx.runtime_config != nullptr) {
    const bool greedy =
        ctx.runtime_config->get<bool>("audio_bark", "greedy");
    // Pass greedy into the model-owned pipeline configuration.
}
```

Handle the nullable pointer. It is non-null after successful runtime config
resolution.

## Error behavior

- Python `trtmc build` rejects malformed `--set`, unknown namespaces/fields,
  invalid types, and invalid values, and exits nonzero.
- The C++ CLI resolves explicit `--config`/`--set` input before dispatch and
  also exits nonzero when that validation fails.
- Direct `PipelineFactory` callers currently get best-effort behavior: a
  runtime-config resolution exception prints
  `[trtmc.config] Failed to resolve runtime config`, returns a null config to
  the plugin, and continues with that plugin's local fallback behavior.

Successful resolution writes `<bundle>.effective_config.json` beside the
bundle. Failed factory resolution does not write a new file. Check stderr and
the freshly written effective-config artifact when proving an override took
effect.

## Validation checklist

1. Add matching Python and C++ schema tests for defaults, layer allowlists,
   type coercion, validators, and unknown fields.
2. Add CLI tests for `--config` and repeated `--set`.
3. Test that the owning builder/plugin consumes the resolved value; registration
   alone is not feature coverage.
4. For a model-owned C++ schema, add the source/registrar entry to the runtime
   `MODEL.toml` and build that model DSO.
5. Verify the effective-config artifact and the user-visible behavior.
