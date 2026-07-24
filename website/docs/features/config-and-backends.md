---
title: Config and Backends
---

## Schema-driven config

Both build and runtime CLIs expose a generic config surface:

```bash
--config profile.json
--set namespace.field=value
```

The goal is to add feature knobs through registered schemas instead of growing custom CLI flags for every feature.

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

Python build-time config resolution rejects unknown namespaces, fields, and
invalid values. The C++ CLI also resolves explicit `--config`/`--set` input
before dispatch and exits nonzero on an invalid value. Direct
`PipelineFactory` callers have a different current behavior: the factory
catches a resolution error, prints
`[trtmc.config] Failed to resolve runtime config`, and continues with
`runtime_config == nullptr`; the owning plugin then chooses its local fallback
behavior. Successful resolution writes
`<bundle>.effective_config.json`; a failed factory resolution does not write a
new effective-config file. Callers using the factory API must treat that
warning as an error if silent fallback is unacceptable.

## Backend DSOs

The runtime loads TensorRT backends dynamically:

- Standard TensorRT backend: `libtrtmc_backend_trt.so`
- ABI-suffixed standard backend alias when available: `libtrtmc_backend_trt_<major>_<minor>.so`
- TensorRT-RTX backend: `libtrtmc_backend_trt_rtx.so`

Use `--backend-dir` to add explicit backend search directories:

```bash
./build/trtmc run /tmp/model.trtfb \
  --prompt "Hello" \
  --backend-dir /opt/trtmc/backends
```

## Runtime cache and CUDA graphs

TRT-RTX paths can use:

```bash
--runtime-cache /tmp/trtmc-rtx.cache
--cuda-graphs
```

Use these only when the selected backend supports them.
