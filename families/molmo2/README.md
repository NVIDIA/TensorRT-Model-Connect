# Molmo2

This family builds `allenai/Molmo2-8B` into a native TensorRT bundle for
single-image, greedy vision-language generation. The default build uses BF16,
one GPU, and a 2,048-token total sequence limit. The checkpoint revision is
`e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b`.

The family owns the SigLIP vision tower, the ordered `[-3, -9]` feature
concatenation, masked 2×2 attention pooling, SwiGLU image projection, and the
36-layer Qwen3-derived text decoder. Image features are added to their token
embeddings. Prefill allows bidirectional attention between image tokens while
retaining causal attention for text. Native preprocessing retains both the
overview and up to eight overlapping crops, including padding in pooling
indices and the checkpoint's image framing and BOS token.

Prepare the source-build environment described in
[Build from Source](../../website/docs/getting-started/source-build.md).
No additional family packages are needed to build or run the model. Build the
native targets `trtmc`, `trtmc_backend_trt`, and `trtmc_model_molmo2`.

```bash
PYTHONPATH=core/builder:. python -m tensorrt_model_connect build \
  allenai/Molmo2-8B \
  --revision e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b \
  --precision bf16 --max-sequence-length 2048 \
  -o molmo2-8b-bf16.bundle

build/trtmc run molmo2-8b-bf16.bundle --runtime-root build \
  --image families/molmo2/tests/data/test_img.jpeg \
  --prompt 'What is the main object in this image? Answer with one word.' \
  --max-new-tokens 16 --top-k 1
```

The bundle contains `vision.plan`, `engine.plan`, the FP32 host embedding
table, tokenizer, and the family runtime contract. A masked sentinel cache row
lets prefill and decode share one dynamic decoder engine. Request state is
reset before every generation; TensorRT contexts and weights remain loaded.
The native image API accepts RGB HWC floats in `[0,1]` and applies the
checkpoint's uint8 resize/normalization semantics.

The unmodified upstream reference needs a separate environment using
[tests/requirements.txt](tests/requirements.txt), because its Transformers
4.57.1 API differs from the builder's 5.2.0 pin. The test manifest pins both
the selected checkpoint revision as metadata; it does not authenticate supplied
local files. Native Unicode tokenization requires ICU's development package.
The native tokenizer applies the checkpoint's NFC normalizer and uses ICU's
Unicode letter, number, and whitespace properties for Qwen3 pre-tokenization.

To compare token IDs without downloading the model weights, build
`test_molmo2_tokenizer`, set `TRTMC_MOLMO2_TOKENIZER_DIR` to the selected
tokenizer snapshot and `TRTMC_RUNTIME_ROOT` to the native build directory, and run
`python -m pytest families/molmo2/tests/test_tokenizer.py -q`.

For direct E2E:

```bash
export TRTMC_MOLMO2_MODEL_DIR=/path/to/pinned/checkpoint
export TRTMC_MOLMO2_REFERENCE_PYTHON=/path/to/reference/python
export TRTMC_RUNTIME_ROOT=/path/to/native/build
export TRTMC_BINARY="$TRTMC_RUNTIME_ROOT/trtmc"

PYTHONPATH=core/builder:apps/benchmark:. python -m pytest \
  families/molmo2/tests --e2e-model molmo2 -q
```

Build `test_molmo2_preprocess` to include the native qualification executable.
By default E2E builds a fresh bundle and runs the independent reference. For
an explicitly prepared bundle/reference run, the test accepts
`TRTMC_MOLMO2_BUNDLE` and `TRTMC_MOLMO2_REFERENCE_RESULTS`.

Video, multiple images, batching, parallel execution, quantization, FP16,
sampling, system prompts, and LoRA are outside this initial path. FP32 graph
construction is implemented but has not been qualified on the full checkpoint.
