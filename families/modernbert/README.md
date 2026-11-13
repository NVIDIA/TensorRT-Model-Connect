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

### NVFP4

NVFP4 was measured on top of this FP8 recipe and is not shipped. The study
built real TensorRT engines (no NVFP4 code is in the family): NVFP4 weights with
an FP8 scale per block of 16, FP4 activations quantized at run time with the
FP8 activation scales. Variants: (a) `mlp.Wi` and `mlp.Wo` in NVFP4,
`attn.Wqkv` FP8; (b) all three kinds in NVFP4. B300, TensorRT 11.4, same job as
the FP8 measurements (FP8 there is 1.01x to 1.37x of fp16 at batch 1).

| Model | Variant | Plan vs FP8 | Batch 1 speed vs FP8 (128 to 8192 tokens) | Batch 8 / 32 vs FP8 | Masked-LM KL vs PyTorch (x FP8) | STS-B |
|---|---|---|---|---|---|---|
| Granite small r2 | (a) MLP NVFP4 | -11% | 0.89x to 1.00x | 1.00x / 0.98x | - | 0.7617 (FP8 0.7672) |
| Granite small r2 | (b) all NVFP4 | -14% | 0.86x to 1.01x | 0.98x / 0.95x | - | 0.7597 |
| ModernBERT-base | (a) MLP NVFP4 | -11% | 0.89x to 1.00x | 1.00x / 0.99x | 0.121 (5.3x) | - |
| ModernBERT-base | (b) all NVFP4 | -20% | 0.86x to 1.00x | 1.01x / 0.99x | 0.187 (8.2x) | - |
| ModernBERT-large | (a) MLP NVFP4 | -19% | 0.89x to 1.00x | 1.04x / 1.05x | 0.077 (6.2x) | - |
| ModernBERT-large | (b) all NVFP4 | -27% | 0.85x to 1.00x | 1.06x / 1.07x | 0.119 (9.6x) | - |

No variant gave a non-finite output.

Reasons:

- Quality: KL is 5.3x to 9.6x the FP8 KL; masked-LM accuracy drops by 2.2 to
  3.1 points (base) and 1.1 to 1.8 points (large) against FP8.
- Speed: at batch 1 NVFP4 is 0.85x to 1.01x of FP8. The only gain is
  ModernBERT-large at batch 8 and 32 on B300 (1.04x to 1.07x), and the family
  builds batch 1 only. At base, batch 32, the FP4 GEMMs save 0.31 ms against
  FP8, but the run-time activation quantize kernels add 0.87 to 1.43 ms.
- TensorRT problems: on B200 the TensorRT 11.4 builder crashed (segmentation
  fault) on the NVFP4 engines of Granite and base with this recipe.
