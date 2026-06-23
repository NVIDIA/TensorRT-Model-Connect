# Diffusion Domain Helpers

Shared diffusion math, scheduler, batching, and preprocessing utilities.
Model-specific runtime pipelines live under `src/runtime/models/<model>/`.

Key files:
- `batch_utils.*`: tensor and batch-shape helper functions.
- `diffusion_math.h`: numeric helpers used by diffusion runtimes.
- `diffusion_preprocessor.*`: generic preprocessor weight loading.
- `diffusion_scheduler_helpers.h`: scheduler configuration helpers.
- `diffusion_types.h`: shared diffusion runtime value types.

How to understand:
1. Start at the owning model pipeline in `src/runtime/models/<model>/`.
2. Follow calls into this folder for shared helper behavior only.
