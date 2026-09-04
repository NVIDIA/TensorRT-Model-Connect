---
title: Source Layout
---

```text
families/<family>/             complete model-owned vertical slice; optional requirements.txt
requirements/base.txt         thin build-tool delta over the pinned GPU base
core/builder/                  Python-only build control plane and tests
core/runtime/include/trtmc/    public native Task, Engine, bundle, and loader APIs
core/runtime/bundle/           reader-only bundle container implementation
core/runtime/primitives/       libtrtmc_core.so device and engine primitives
core/runtime/loader/           libtrtmc_runtime.so exact DSO loader
core/runtime/tensorrt/         libtrtmc_backend_trt.so implementation
apps/cli/                      native CLI and private image/audio file I/O
apps/benchmark/                benchmark application, workers, and performance policy
tools/model_ci.py              family inventory and impact
website/                       documentation generated from family ownership
```

`core/builder/` contains only Python. `core/runtime/` contains only C++ headers
and sources. No production source lives under the retired `python/`, `src/`,
`include/`, or `tests/` roots.
