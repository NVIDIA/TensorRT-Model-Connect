# C-Linkage C++ Entrypoint

`trtmc_c.cpp` implements the exported C-linkage entrypoints declared by
`include/trtmc/pipeline.h` and wires them to the C++ runtime. This is a C++
subset with C linkage, not a C-compatible public header or complete stable C
ABI: it exposes `trtmc::IPipeline*` and `std::uint64_t`, and it has no exported
pipeline-destroy function.

Focus areas:

- Pipeline option and handle validation with thread-local error reporting.
- `.trtfb` validation followed by bundle-kind selection through
  `PipelineFactory::from_bundle`: embedded optimized-runtime factory or native
  strategy/plugin composition.
- C-linkage wrappers for pipeline creation and batch image generation.
- Conversion and ownership rules for image-result buffers.

Bundle parsing itself lives under `src/bundle/`. Optimized-runtime hosting
lives under `src/runtime/providers/`; native runtime-strategy resolution and
model-plugin loading live under `src/runtime/registry/`.
