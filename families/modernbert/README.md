# ModernBERT runtime Tasks

Rebuild existing ModernBERT bundles after this migration. Retired build names
`encoding` and `embedding` are replaced by:

| Build Task | Runtime result |
| --- | --- |
| `text_to_pooled_features` (default) | First-token CLS hidden features, without normalization |
| `text_to_embedding` | Mean of all input-token hidden features, L2 normalized |

Each bundle advertises only its selected Task. Legacy `reranking`,
`text_pair_to_relevance` and `text_query_documents_to_relevance` bundles are
rejected before engine loading. The encoder returns hidden states and has no
trained scoring head. Its first hidden feature is not a relevance score; future
reranking support needs a scoring head and checkpoint-specific pair tokenization.

Runtime Config is empty. Default/query/document embedding roles use
the same input processing; no checkpoint-specific prefixes are inferred.
The pooled-feature Task accepts UTF-8 text or token IDs. Inputs must be nonempty
after tokenization, within the built sequence profile and vocabulary. The TP1
dynamic encoder receives exactly the actual input length. Fixed TP plans receive
zero-padded IDs and a fresh validity mask; pooling excludes padding rows.

The existing FP32 TP1 and TP4 E2E cases retain their official CLS reference and
cosine thresholds. They execute the CLI and public C/C++ consumers against the
same bundle. Native CPU contracts cover bindings, masks, pooling, unsupported relevance rejection,
invalid inputs and engine failures; CPU results do not establish GPU model parity.
Set `TRTMC_NATIVE_BUILD_DIR` to the CMake build root containing both SDK consumers,
including when `TRTMC_BINARY` points to a CLI installed from a wheel.

## FP8 (`quantization=fp8`)

FP8 self-quantizes the original checkpoint at build time (TP1, `precision=fp16`
only; TP plans stay unquantized). Weights get one scale per checkpoint module
(`max(abs(weight)) / 448`); activations use static per-tensor scales from
`fp8_activation_scales.json`, calibrated once with `calibrate_fp8_scales.py`
(ModelOpt `FP8_DEFAULT_CFG`, WikiText-2 training text). Quantized: `attn.Wqkv`,
`mlp.Wi` and `mlp.Wo` of every layer, except `layers.0.mlp.Wi` and modules whose
input scale is above 20x the median of their kind (stored as `null`). The
attention output projection `attn.Wo` stays fp16: TensorRT 11.4 cannot compile an
FP8 Quantize after the fused attention. Calibrated model configs:
ModernBERT-base, ModernBERT-large and granite-embedding-small-english-r2; any other
config raises until it is calibrated.
