# Kolibri1

This family implements greedy FP16 text continuation from the released
block-scaled FP8 checkpoint. The builder dequantizes each weight matrix using
its checkpoint scales and emits a fixed KV profile.

The runtime keeps the first half of the decoder engines resident. It caches
the remaining serialized plans in host memory and loads one transient engine
at a time for those layers. Explicit KV tensors preserve state across tokens;
the plans are read once rather than reread for every generated token.

Direct tests compare two native continuations with the official dequantized
FP16 model, including exact token IDs and decoded text. The official reference
runs in its own process so its CUDA resources are released before native execution.
