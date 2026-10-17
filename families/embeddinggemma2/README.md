# EmbeddingGemma2 text embeddings

This family builds the independent text backbone of
[`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2/tree/914f7f89142e33e77833254d9c9b90c3cef7303b)
into an `embedding` bundle. It exposes the native `TextToEmbedding` Task with
768-dimensional mean-pooled, L2-normalized vectors. Query input receives
`task: search result | query: `; document input receives `title: none | text: `;
default input receives no prefix. BOS/EOS come from `tokenizer.json`, including
for empty text. Inputs exceeding engine capacity fail without truncation.

The scoped implementation supports unpadded single inputs, TP1/CP1, FP32 and
BF16, and a capacity of 2 to 8192 tokens including prefixes and special tokens. FP16 is
rejected, as documented by the publisher. It does not implement image, audio,
video, batching, quantization, or retrieval indexing. Modality placeholder
tokens are rejected by the text runtime. No compatible multimodal Task is
advertised. Engine output must be finite FP32 token features with the configured
embedding dimension; invalid or zero-norm pooled vectors fail explicitly.

The bidirectional graph owns local/global attention, layer-specific head sizes,
Q/K/V normalization, projection-only per-layer inputs and saved layer scalars.
Only `language_model.*` tensors are loaded from single or sharded safetensors;
vision and audio tensors remain outside the engine. TensorRT performs all GPU
lowering. The family-local BPE implementation originates from the repository's
Gemma tokenizer at upstream `d79aaae87fd0b28af55e3e27f877ed4c40f28619`, with empty
text framing corrected here.

## Validation

`python -m pytest families/embeddinggemma2/tests -q` runs checkpoint shape,
option, weight-loader and graph tests. Native CTests exercise the real pipeline
and tokenizer without an engine. The graph test double evaluates actual network
operations on NumPy arrays; BF16 dtype metadata is tested, but BF16 rounding or
TensorRT execution is not simulated. The optional tiny FP32 reference test uses
the unmodified official Transformers text model.

The project pins Transformers 5.2.0, which predates this model. The family declares
its pinned official reference in `requirements.txt`; Community GPU CI installs it
in the selected family's container before unit and E2E tests. For a manual run,
install it in a separate environment:

```sh
python -m venv --system-site-packages /tmp/embeddinggemma2-reference
/tmp/embeddinggemma2-reference/bin/python -m pip install \
  -r families/embeddinggemma2/requirements.txt
export TRTMC_EMBEDDINGGEMMA2_REFERENCE_PYTHON=/tmp/embeddinggemma2-reference/bin/python
```

For GPU qualification, cache the checkpoint at the revision above and configure
`TRTMC_RUNTIME_ROOT` and `TRTMC_NATIVE_BUILD_DIR` for the built runtime. An
explicit `TRTMC_EMBEDDINGGEMMA2_MODEL_DIR` can select a local checkpoint, whose
revision the operator must record. Then run:

```sh
python -m pytest families/embeddinggemma2/tests/test_e2e.py \
  --e2e-model embeddinggemma2-text-bf16 -q
```

The three role cases build the bundle, execute the native C++ SDK consumer, and
compare its output against the official model with unused towers disabled.
`embeddinggemma2-text-fp32` provides the corresponding explicit FP32 cases.
Selected cases fail if CUDA, cached weights, native tools, or the reference
environment are missing. The reference interpreter defaults to the current Python;
the optional variable above selects a separate interpreter. Both require
CUDA-enabled PyTorch and the pinned official source, checked before reference
inference. GPU engine/checkpoint correctness
is qualified only for the role, precision and capacity of cases that pass;
performance requires separate measurements. Selected cases retain native and
reference vectors, thresholds and source revision under
`$TRTMC_E2E_ARTIFACT_DIR/evidence/embeddinggemma2/<case>/evidence.json`.

The shape fixture records only text configuration and safetensors header shapes
from the pinned publisher checkpoint. It contains no model weights or tokenizer
assets. Publisher weights are Apache-2.0; downloaded assets are not redistributed
by this family.
