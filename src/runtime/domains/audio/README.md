# Audio Domain Helpers

Shared audio-domain primitives used by model-owned runtime plugins.

Key files:
- `audio_types.*`: common audio result/value types.
- `mel_spectrogram.*`: mel feature extraction helpers.

How to understand:
1. Start in `src/runtime/models/<model>` for model-specific runtime behavior.
2. Use this directory only for reusable audio primitives that are not owned by a
   single model family.
3. Use `core/device_kv_cache.*` and `core/trt_decode_runtime.*` for shared
   decode behavior.
