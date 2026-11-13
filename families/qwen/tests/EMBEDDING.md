# Qwen3-Embedding-0.6B

The Qwen family owns both generation and the checkpoint-specific embedding task.
Embedding discovery uses the sentence-transformers sidecar; the builder then
requires the pinned 0.6B dimensions, last-token pooling, and Normalize module.
Other embedding sizes and pooling contracts are rejected.

```python
from pathlib import Path
from tensorrt_model_connect import BuildRequest, build

build(BuildRequest(
    model_dir=Path("/path/to/Qwen3-Embedding-0.6B"),
    output_path=Path("qwen3-embedding.bundle"),
    family="qwen",
    task="embedding",
    precision="bf16",
    max_sequence_length=256,
))
```

Only FP16/BF16, one text per request, single-device, unquantized builds are
implemented. The cacheless causal graph produces final hidden rows; the runtime
appends EOS when missing, selects the final token, and L2-normalizes the vector.
Inputs exceeding the engine profile fail instead of being silently truncated.

Use the public SDK `model.task<trtmc::TextToEmbedding>()`. `Query` applies the
standard web-retrieval instruction; `Document` passes text unchanged. `Default`
also passes text unchanged, allowing an explicitly formatted custom instruction.
The result identifies the checkpoint embedding space, `last_token` pooling and
`l2` normalization. The family-owned `qwen_embedding_consumer` executable provides
a public-SDK E2E consumer: pass bundle, runtime root, and already formatted text.

The manifest pins `Qwen/Qwen3-Embedding-0.6B` at
`97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`. Select its E2E with
`--e2e-model qwen3-embedding-0.6b`; the case is also marked `premerge` for
Community GPU selection. It compares native output with Transformers
using cosine >= 0.99, L2 distance <= 0.1, and unit-norm error <= 0.001. Build the
`qwen_embedding_consumer` CMake target through `TRTMC_NATIVE_BUILD_DIR`, a
configured native CMake build tree; the E2E builds and locates that executable
there because the isolated runtime directory contains only runtime libraries.
Provide the normal Qwen E2E runtime and CUDA environment. CPU contract tests and protocol fixtures do not qualify
TensorRT engine construction, tokenizer parity, numerical accuracy, or speed.
Target-GPU BF16 E2E and performance qualification remain outstanding.
