# Diffusion Domain Helpers

Shared diffusion math and batching utilities.
Model-specific runtime pipelines live under `src/runtime/models/<model>/`.
Scheduler policy lives in the owning model folder.
Preprocessor section parsing and model weight-key loading live in the owning
model folder.

Key files:
- `diffusion_math.h`: numeric helpers used by diffusion runtimes.
- `diffusion_types.h`: shared diffusion runtime value types.

Batch/chunk planning and seed fallback policy are model-owned under
`src/runtime/models/<diffusion-family>/`.

How to understand:
1. Start at the owning model pipeline in `src/runtime/models/<model>/`.
2. Follow calls into this folder for shared helper behavior only.
