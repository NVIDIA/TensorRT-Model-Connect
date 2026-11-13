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

## FP8 quantization study (not shipped)

FP8 self-quantization of the attention and MLP projections was built and
measured on TensorRT 11.4 for the checkpoint above. It is not shipped: FP8 is
0.89x to 1.00x of BF16 on B200 and B300 for 8 to 2048 tokens (best projection
subset 0.97x to 1.02x), the plan is only 21% smaller, and FP8 does not build on
TensorRT 11.1 on B200. Quality is fine (SciFact nDCG@10 unchanged, mean cosine
0.9987, no NaN). This section records the recipe, the method and the numbers,
so that the study does not have to be repeated. The FP8 code was not merged;
this family has no FP8 build.

### Summary and decision

| Item | fp32 | BF16 | FP8 | FP8 vs BF16 |
|---|---|---|---|---|
| Time, 8 tokens, B200 / B300 | 4.398 / 4.603 ms | 1.437 / 1.402 ms | 1.494 / 1.429 ms | 0.96x / 0.98x |
| Time, 512 tokens, B200 / B300 | 7.853 / 7.799 ms | 2.557 / 2.436 ms | 2.853 / 2.745 ms | 0.90x / 0.89x |
| Time, 2048 tokens, B200 / B300 | 35.045 / 34.368 ms | 5.626 / 5.350 ms | 5.746 / 5.324 ms | 0.98x / 1.00x |
| Plan size (capacity 256, B200) | 1040.5 MiB | 524.2 MiB | 414.8 MiB | -21% |
| SciFact nDCG@10 (Transformers fp32 0.8409) | 0.8409 | 0.8425 | 0.8415 | unchanged |
| Cosine vs Transformers fp32, mean / min | 1.00000 / 1.0000 | 0.99997 / 0.9997 | 0.99870 / 0.9908 | small |
| Non-finite outputs (5683 inputs) | 0 | 0 | 0 | none |

Decision: do not ship FP8 for EmbeddingGemma2.

- Speed: no gain on B200 or B300 at any length (0.89x to 1.02x of BF16 over
  all measured projection subsets). The plan is launch and attention bound:
  639 layers (BF16) and 615 (FP8, B200) to 755 (FP8, B300) per call, about
  2 us per kernel at 8 tokens. The projection GEMMs are small (K = 512 or
  2048, N at most 2048): at 2048 tokens all 168 projections are about
  0.46 TFLOP per call, about 0.3 ms at BF16 peak, so FP8 could shorten at most
  about 6% of the 5.4 to 5.6 ms. The rest is the full-attention layers (head
  size 512, fp32 scores), norms, RoPE and elementwise layers. FP8 adds
  Quantize work around each GEMM, which cancels the GEMM gain.
- Size: the BF16 embedding table (262144 x 512, 256 MiB) is half of the plan
  and is not a GEMM, so the size gain stops at about 21%.
- FP8 vs fp32 is 1.6x to 6.5x faster, but BF16 alone gives all of that.

### Model

| Item | Value |
|---|---|
| Checkpoint | `google/embeddinggemma-2`, revision `914f7f89142e33e77833254d9c9b90c3cef7303b` (manifest revision), text backbone only |
| Model config | 24 layers, hidden 512, MLP 2048, 4 heads, head size 256 / 512, vocab 262144, 768-d output |
| Family precision | BF16 (fp32 is the second manifest; fp16 is rejected) |
| Batch | 1 (the family rejects `max_batch_size` above 1); the token count is the GEMM M size |

No ModelOpt FP8 or NVFP4 checkpoint exists. The public quantized repositories
are GGUF, MLX, LiteRT, ONNX and a W4A16 export, none usable here, so the
weights were self-quantized from the BF16 checkpoint.

### Recipe that was measured

No official FP8 config exists for this model. The recipe follows the ModelOpt
guidelines for small text encoders: every attention and MLP linear in FP8
W8A8. Every quantized projection is built as Quantize(FP8) -> Dequantize(BF16)
-> MatMul, with an FP8 weight constant.

| Choice | Decision |
|---|---|
| Quantized | q, k, v, o, gate, up, down of all 24 layers (168 projections). The attention is an explicit matmul with fp32 softmax, so `o_proj` builds in FP8 on TensorRT 11.4. |
| BF16 | embedding table, per-layer-input projections, norms, `embedding_projection` |
| Weight scale | per tensor, amax / 448, at build time |
| Activation scale | static, per tensor, ModelOpt 0.46.1 `FP8_DEFAULT_CFG` (max calibration) on the official model in BF16, scale = amax / 448 |
| Calibration data | 512 texts of `sentence-transformers/natural-questions` rows 0 to 255: 256 questions (query prefix) and 256 Wikipedia passages (document prefix) |
| Outlier rule | a scale above 20x the median of its kind is stored as `null` and stays BF16; none found (largest ratio: `down_proj` 4.4x) |
| Dequantize type | BF16 (the engine type; no overflow, no NaN) |
| Scale table key | model config plus a weight fingerprint; a missing scale is a build error |

Variants measured: q, k, v, o and MLP (the recipe, built twice); q, k, v and
MLP; MLP only; the recipe plus the per-layer-input projections.

### Method

- GPUs: B200 and B300, one job per GPU type, all variants built and timed
  together with the family builder settings. TensorRT 11.4.0.106 (target),
  then 11.1.0.106.
- Speed: CUDA graph replay, median of 9 interleaved rounds of 100 runs.
  Capacity 256 for 8 to 256 tokens, capacity 2048 for 512 to 2048 tokens.
- Held-out quality data (not used for calibration): SciFact (BEIR) 300 test
  queries and 5183 corpus documents (title and text, truncated to 256 tokens),
  plus 200 natural-questions passages (rows 50000 to 50199, default role):
  5683 inputs.
- Reference: the official Transformers model at the revision above (fp32 and
  BF16), mean pooling and L2 normalization as the runtime does.

### Speed (TensorRT 11.4, ms)

| GPU | Tokens | fp32 | BF16 | FP8 (recipe) | FP8 vs BF16 | Best FP8 variant vs BF16 |
|---|---|---|---|---|---|---|
| B200 | 8 | 4.398 | 1.437 | 1.494 | 0.96x | 0.97x (q, k, v, MLP) |
| B200 | 32 | 4.465 | 1.569 | 1.623 | 0.97x | 0.98x |
| B200 | 128 | 4.478 | 2.214 | 2.250 | 0.98x | 0.99x |
| B200 | 256 | 5.235 | 3.244 | 3.250 | 1.00x | 1.00x |
| B200 | 512 | 7.853 | 2.557 | 2.853 | 0.90x | 1.00x (q, k, v, MLP) |
| B200 | 1024 | 14.294 | 3.789 | 3.979 | 0.95x | 0.98x (MLP) |
| B200 | 2048 | 35.045 | 5.626 | 5.746 | 0.98x | 0.99x (MLP) |
| B300 | 8 | 4.603 | 1.402 | 1.429 | 0.98x | 0.99x (recipe, repeat build) |
| B300 | 32 | 4.637 | 1.528 | 1.553 | 0.98x | 0.99x |
| B300 | 128 | 4.549 | 2.128 | 2.150 | 0.99x | 0.99x |
| B300 | 256 | 5.152 | 3.117 | 3.138 | 0.99x | 1.00x (MLP) |
| B300 | 512 | 7.799 | 2.436 | 2.745 | 0.89x | 0.99x (q, k, v, MLP) |
| B300 | 1024 | 14.354 | 3.633 | 3.757 | 0.97x | 1.01x (q, k, v, MLP) |
| B300 | 2048 | 34.368 | 5.350 | 5.324 | 1.00x | 1.02x (MLP) |

Plan size (capacity 256, B200; B300 within 0.3%): fp32 1040.5 MiB, BF16
524.2 MiB, FP8 recipe 414.8 MiB (-21%), recipe plus per-layer-input
projections 403.0 MiB (-23%).

### Quality (TensorRT 11.4, B200, 5683 held-out inputs; B300 within 0.01 in every column)

| Build | Cosine vs Transformers fp32, mean / 1st percentile / min | SciFact nDCG@10 | Top-10 overlap | Top-1 agreement | Non-finite inputs |
|---|---|---|---|---|---|
| Transformers fp32 | - | 0.8409 | - | - | 0 |
| Transformers BF16 | - | 0.8409 | - | - | 0 |
| fp32 engine | 1.00000 / 1.0000 / 1.0000 | 0.8409 | 100.0% | 100.0% | 0 |
| BF16 engine | 0.99997 / 0.9999 / 0.9997 | 0.8425 | 97.8% | 99.3% | 0 |
| FP8 recipe | 0.99870 / 0.9963 / 0.9908 | 0.8415 | 92.4% | 97.3% | 0 |
| FP8 q, k, v, MLP | 0.99885 / 0.9967 / 0.9903 | 0.8443 | 92.5% | 97.3% | 0 |
| FP8 MLP only | 0.99913 / 0.9974 / 0.9946 | 0.8433 | 94.0% | 97.3% | 0 |
| FP8 recipe plus per-layer-input projections | 0.99851 / 0.9957 / 0.9875 | 0.8431 | 92.2% | 97.0% | 0 |

Top-10 overlap and top-1 agreement are against Transformers fp32 retrieval.
Retrieval quality (nDCG@10) is unchanged; the neighbor lists change more than
with the BF16 engine (92% vs 98% top-10 overlap). Non-finite scan: 0 in all
5683 inputs for every engine on both GPUs (11.4) and in 1500 inputs on 11.1
(B300).

The FP8 code did not change the unquantized builds: BF16 and fp32 plans have
the same layer counts as main (639 / 708), plan sizes within 0.002%, the same
latency and the same embeddings.

### TensorRT 11.1

- B300: FP8 builds; mean cosine 0.99856, no non-finite output; speed as on
  11.4 (0.89x to 1.00x of BF16).
- B200: every FP8 build fails (`Error lowering to NVVM` on an `arith.divf`,
  then `autotuner ... Must have costs`) for both capacities. BF16 and fp32
  build. This is a TensorRT 11.1 FP8 problem on B200, not a problem on main.

### NVFP4 (not tried)

NVFP4 was not built. The projections are not bound by weight reads (K = 512 or
2048, 125M parameters in GEMMs; the plan is launch and attention bound), FP8
GEMMs already give no speed, and NVFP4 adds a dynamic activation-quantize
kernel per projection. The plan size would still be dominated by the BF16
embedding table.

### Found on the way

No problem on main was found.

### What was not tried

| Item | Note |
|---|---|
| Batch above 1 | the family supports batch 1 only |
| Percentile calibration, per-channel weight scales | not measured; quality is already fine |
| H100 | not measured |
| Native runtime with an FP8 bundle | not run; the engines were called from Python |

### How to reproduce

1. Calibrate per-tensor activation scales with ModelOpt `FP8_DEFAULT_CFG` on
   the official model in BF16 with the calibration texts above.
2. Build the BF16 and FP8 bundles with the recipe above.
3. Time the engines with CUDA graph replay and compare the embeddings and the
   SciFact retrieval with the official Transformers model.

### When to revisit

- A TensorRT release that lowers the per-layer launch cost or fuses the
  Quantize into the preceding norm, so the FP8 GEMM gain shows.
- Batch support above 1, where the projection GEMMs grow.
- A need for the 21% smaller plan.
