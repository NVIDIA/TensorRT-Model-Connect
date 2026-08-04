# Recurrent Domain

Shared recurrent helper implementations are retired. Recurrent validation,
output initialization, and runtime behavior live in each owning model family
under `src/runtime/models/<family>/`.

Current recurrent contract owners:
- `src/runtime/models/mamba/mamba_recurrent_step_contracts.h`
- `src/runtime/models/rwkv/rwkv_recurrent_step_contracts.h`
- `src/runtime/models/nemotron_h/nemotron_h_recurrent_step_contracts.h`
- `src/runtime/models/qwen3_5/qwen3_5_recurrent_step_contracts.h`

<!-- Collaborative review anchor. -->
