# C-Linkage C++ Runtime Edge

This directory contains the thin C-linkage C++ subset for the plugin-composed
runtime. It is not a C-compatible header or a complete, stable C ABI: the
declarations use C++ types, pipeline creation returns `trtmc::IPipeline*`, and
there is no exported pipeline-destroy function. The only behavior-bearing
source below this directory is `api/trtmc_c.cpp`.

Current ownership:

- `api/trtmc_c.cpp`: exported C-linkage validation, error mapping, pipeline
  creation, and batch-result conversion.
- `src/bundle/`: `.bundle` format and bundle-view implementations.
- `src/runtime/config/`: layered runtime configuration and schema handling.
- `src/runtime/registry/pipeline_factory.cpp`: bundle-to-pipeline selection.
  It recognizes an embedded optimized-runtime implementation before using the
  native strategy/plugin registry.

`bundle/README.md` is an ownership note only; no C-linkage bundle-helper
implementation remains in that subdirectory.

<!-- Collaborative review anchor. -->
