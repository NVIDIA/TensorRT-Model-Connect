# Perception Domain Helpers

Shared domain helpers for generic perception preprocessing and postprocessing.
Concrete model runtimes own their prompted segmentation, detection, and neural
operator implementations under `src/runtime/models/<model>/`.

Key files:
- `segmentation_preprocess_seam.h`: generic segmentation image preprocessing.
- `segmentation_postprocess_seam.h`: generic segmentation class-map postprocessing.
- `perception_types.h`: generic perception value types.

How to understand:
1. Start from the owning runtime model pipeline.
2. Use this folder only for behavior shared by multiple perception model families.
3. Keep single-family helpers beside their owning runtime model.
