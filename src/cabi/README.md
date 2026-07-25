# C ABI Runtime Edge

This directory contains the thin C ABI edge for the plugin-composed runtime.
The only behavior-bearing source below this directory is
`api/trtmc_c.cpp`.

Current ownership:

- `api/trtmc_c.cpp`: exported C API validation, error mapping, pipeline
  creation, and batch-result conversion.
- `src/bundle/`: `.trtfb` format and bundle-view implementations.
- `src/runtime/config/`: layered runtime configuration and schema handling.
- `src/runtime/registry/pipeline_factory.cpp`: bundle-to-pipeline selection.
  It recognizes an embedded optimized-runtime implementation before using the
  native strategy/plugin registry.

`bundle/README.md` is an ownership note only; no C ABI bundle-helper
implementation remains in that subdirectory.
