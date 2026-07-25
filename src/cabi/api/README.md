# C API Entrypoint

`trtmc_c.cpp` implements the exported C entrypoints declared by
`include/trtmc/pipeline.h` and wires them to the C++ runtime.

Focus areas:

- Pipeline option and handle validation with thread-local error reporting.
- `.trtfb` validation followed by bundle-kind selection through
  `PipelineFactory::from_bundle`: embedded optimized-runtime factory or native
  strategy/plugin composition.
- C wrappers for pipeline creation and batch image generation.
- Conversion and ownership rules for C ABI image-result buffers.

Bundle parsing itself lives under `src/bundle/`. Optimized-runtime hosting
lives under `src/runtime/providers/`; native runtime-strategy resolution and
model-plugin loading live under `src/runtime/registry/`.
