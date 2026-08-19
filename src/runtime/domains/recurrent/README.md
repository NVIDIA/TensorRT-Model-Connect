# Recurrent Domain

Shared recurrent helper implementations are retired. Recurrent validation,
output initialization, and runtime behavior live in each owning model family
under `python/tensorrt_model_connect/models/<family>/runtime/`.

Current recurrent contract owners:
- `python/tensorrt_model_connect/models/mamba/runtime/mamba_recurrent_step_contracts.h`
- `python/tensorrt_model_connect/models/rwkv/runtime/rwkv_recurrent_step_contracts.h`
- `python/tensorrt_model_connect/models/nemotron_h/runtime/nemotron_h_recurrent_step_contracts.h`
- `python/tensorrt_model_connect/models/qwen3_5/runtime/qwen3_5_recurrent_step_contracts.h`

<!-- Collaborative review anchor: batch 2. -->
