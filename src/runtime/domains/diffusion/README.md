# Diffusion Domain Helpers

This directory contains the small model-neutral subset shared by diffusion
implementations. Model pipelines, schedulers, configuration types, batch/chunk
planning, preprocessing, and weight-key policy remain under
`src/runtime/models/<family>/`.

Key files:

- `diffusion_math.h`: header-only CPU matrix, activation, sinusoidal embedding,
  and timestep-MLP helpers.
- `kernels/dit_rms_norm_rope.cu`: optional TVM-FFI CUDA module, built only when
  `TRTMC_BUILD_DIFFUSION_KERNELS` is enabled.

How to understand:

1. Start with the owning model's `MODEL.toml`, `plugin.cpp`, and pipeline under
   `src/runtime/models/<family>/`.
2. Follow includes into this directory only for the shared math or optional
   kernel behavior above.

There is no shared `diffusion_types.h`; each model that needs such a contract
owns a family-prefixed type header.

<!-- Collaborative review anchor. -->
