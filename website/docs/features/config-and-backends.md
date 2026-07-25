---
title: Config and Backends
---

## Schema-driven config

Both build and runtime CLIs expose a generic config surface:

```bash
--config profile.json
--set namespace.field=value
```

The goal is to add native build/runtime feature knobs through registered
schemas instead of growing custom CLI flags for every feature. Optimized
implementations receive the public option tuple through their family-owned
adapter contract; the generic router does not reinterpret those options.

Schema sources live under:

- `src/runtime/config/schemas/`
- `include/trtmc/config/schemas/`
- `python/tensorrt_model_connect/runtime_config/schemas/`
- `cmake/trtmc_config_schemas.cmake`

Model-owned schemas instead live beside their owners:

- Python: `python/tensorrt_model_connect/families/<family>/runtime_config_schema.py`
- C++: `src/runtime/models/<owner>/config_schema.cpp`
- Registration: the owner's `runtime_config_schemas` entry in
  `src/runtime/models/<owner>/MODEL.toml`

Build routing happens before the native schema-driven builder: `trtmc build`
first offers the model, revision, target, and public option tuple to an exact
qualified optimized profile. If one claims it, that family-owned adapter owns
the option semantics. If none claims it, the native builder resolves registered
schemas and rejects unknown namespaces, fields, or invalid values.

At runtime, the two paths differ:

| Bundle path | Config behavior |
| --- | --- |
| Native | `PipelineFactory` merges `SessionRequest > PlatformProfile > BundleDefault > BuildTime > SchemaDefault` and attaches the result as `PipelineContext::runtime_config`. |
| Optimized | `optimized_runtime.json` claims the bundle before native config/plugin/backend dispatch. The embedded implementation receives `LoadOptions` through its private factory request and decides which options it supports; the current Qwen Edge-LLM implementation rejects runtime `--config`/`--set`. |

The C++ CLI pre-validates explicit `--config`/`--set` values against registered
schemas and exits nonzero on invalid input. That validation does not turn the
result into a native `ConfigBundle` for an optimized implementation.

Direct `PipelineFactory` callers on the native path have a different current
error behavior: the factory catches a resolution error, prints
`[trtmc.config] Failed to resolve runtime config`, and continues with
`runtime_config == nullptr`; the owning native plugin then chooses its local
fallback behavior. Successful resolution writes
`<bundle>.effective_config.json`; a failed factory resolution does not write a
new effective-config file. Callers using the factory API must treat that
warning as an error if silent fallback is unacceptable.

## Native backend DSOs

The native runtime path loads TensorRT backends dynamically:

- Standard TensorRT backend: `libtrtmc_backend_trt.so`
- ABI-suffixed standard backend alias when available: `libtrtmc_backend_trt_<major>_<minor>.so`
- TensorRT-RTX backend: `libtrtmc_backend_trt_rtx.so`

Use `--backend-dir` to add explicit backend search directories:

```bash
./build/trtmc run /tmp/model.trtfb \
  --prompt "Hello" \
  --backend-dir /opt/trtmc/backends
```

An optimized bundle bypasses this selection. The host materializes the
integrity-bound artifact tree, loads its exact embedded
`libtrtmc_impl_*.so`, and lets that implementation own downstream runtime
dependencies. `--backend-dir` is not a generic optimized-runtime DSO search
path.

## Runtime cache and CUDA graphs

The same public options have path-specific ownership:

| Option | Native path | Optimized path |
| --- | --- | --- |
| `--runtime-cache` | TRT-RTX JIT cache path passed to the native plugin/backend. | Root directory where the generic host materializes the integrity-bound optimized artifact tree. |
| `--cuda-graphs` | Passed to the native plugin/backend; use only when supported. | Forwarded in `LoadOptions`; the embedded implementation decides whether it supports or rejects the option. |

For a native TRT-RTX bundle:

```bash
--runtime-cache /tmp/trtmc-rtx.cache
--cuda-graphs
```

For an optimized bundle, use a directory as the cache root rather than an RTX
cache-file name:

```bash
./build/trtmc run /tmp/optimized.trtfb \
  --prompt "Hello" \
  --runtime-cache /tmp/trtmc-optimized-cache
```

Check the selected implementation's qualification contract before passing
provider-specific options such as CUDA graphs.
