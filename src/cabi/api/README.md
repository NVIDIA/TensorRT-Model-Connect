# C API Entrypoint

`trtmc_c.cpp` implements the exported C entrypoints declared by
`include/trtmc/pipeline.h` and wires them to the C++ runtime.

Focus areas:

- Pipeline option and handle validation with thread-local error reporting.
- `.trtfb` validation followed by registry-based composition through
  `PipelineFactory::from_bundle`.
- C wrappers for pipeline creation and batch image generation.
- Conversion and ownership rules for C ABI image-result buffers.

Bundle parsing itself lives under `src/bundle/`; runtime strategy resolution
and model-plugin loading live under `src/runtime/registry/`.
