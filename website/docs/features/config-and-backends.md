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
