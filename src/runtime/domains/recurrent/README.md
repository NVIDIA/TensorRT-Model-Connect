# Recurrent Backends

Recurrent/stateful decode implementations.

Key files:
- `two_state_*`: two-tensor recurrent state types, single-step runtime, and generate loop.
- `multi_state_*`: multi-tensor recurrent state types, single-step runtime, and generate loop.
- `hybrid_backend.*`: hybrid recurrent+attention runtime path.

How to understand:
1. Start with `*_backend.cpp` (`generate`).
2. Follow into `*_decode_runtime.*` for one-step TensorRT execution.
3. Inspect `*_step_state.*` for persistent state layout/updates.
