# Multimodal Runtime Domain

Cross-modal runtime helpers that are generic enough to be shared.

Key files:
- No behavior-bearing helpers remain in the shared multimodal domain.

How to understand:
1. Inspect model-owned `src/runtime/models/<family>/image_preprocessor.*`
   files for VL preprocessing strategies and config parsing.
2. Inspect model-owned `src/runtime/models/<family>/pipeline.*` files for
   end-to-end vision-language decode behavior.
