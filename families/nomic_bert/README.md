# Native tokenizer normalization

This family is implemented by the existing semantic `text_to_embedding` Task.
Its graph, runtime role prefixes, truncation and pooling remain defined by the
family builder and pipeline. This change only corrects native WordPiece
normalization and character classification.

The family-local `runtime/unicode_data.h` reproduces the independently versioned
components of the fixed Tokenizers 0.22.2 oracle:

| Data | UnicodeData version | SHA256 |
| --- | --- | --- |
| Nonspacing-mark categories | [8.0.0](https://www.unicode.org/Public/8.0.0/ucd/UnicodeData.txt) | `38b17e1118206489a7e0ab5d29d7932212d38838df7d3ec025ecb58e8798ec20` |
| Canonical decomposition and combining classes | [9.0.0](https://www.unicode.org/Public/9.0.0/ucd/UnicodeData.txt) | `68dfc414d28257b9b5d6ddbb8b466c768c00ebdf6cbf7784364a9b6cad55ee8f` |
| Simple lowercase mappings | [17.0.0](https://www.unicode.org/Public/17.0.0/ucd/UnicodeData.txt) | `2e1efc1dcb59c575eedf5ccae60f95229f706ee6d031835247d843c11d96470c` |

The reference's `unicode-normalization-alignments` 0.1.12 crate declares Unicode
9.0.0. Canonical decompositions are identical in the cited versions 8 and 9.
Hangul decomposition is algorithmic; U+0130 lowercases to `i` plus U+0307. Stable
combining-class ordering precedes nonspacing-mark removal and lowercase, while
`strip_accents: false` preserves the original order. The Unicode notice is in
`UNICODE_LICENSE.txt`; no extra runtime dependency is introduced.

Native installation also carries the notice under `share/trtmc/families/nomic_bert`.

The existing native tokenizer test retains prefix/framing assertions and adds
distinct-vocabulary normalization cases, independent normalizer flags and
punctuation boundaries. Matching unknown tokens cannot stand in for the expected
IDs in those cases. A source-only probe of the production normalizer matches all
1,112,064 Unicode scalars in three lowercase/accent modes and 3,630 constructed
mark sequences per mode against Tokenizers 0.22.2, with cleaning and Chinese
spacing disabled to isolate those stages. This does not cover every
Unicode string, model inference parity or performance. Canonical ordering needs
a temporary codepoint buffer and stable sorting within each combining sequence.

Rebuild the family runtime to obtain corrected token IDs. No bundle schema or
public Task API change is required.

## FP8 quantization study (not shipped)

FP8 self-quantization on top of the fp16 graph was built and measured on
TensorRT 11.4 (and 11.1) for `nomic-ai/nomic-embed-text-v1.5`. It is not
shipped: FP8 is 1.05x to 1.20x faster than fp16 and the plan is 40% smaller,
but 8% of the SciFact queries change their top-1 document. This section records
the recipe and the numbers, so that the study does not have to be repeated. The
FP8 code was not merged; this family has no FP8 build.

### Recipe that was measured

No FP8 or NVFP4 checkpoint of nomic-embed-text exists on the Hub (only GGUF,
ONNX int8 and MLX ports), and no official quantization config exists for it.
Each quantized projection is built as
`input -> Quantize(FP8) -> Dequantize(fp16) -> MatMul`, with the weight
`FP8 constant -> Dequantize(fp16)`.

| Choice | Decision |
|---|---|
| Quantized (60 projections) | all five projections of every layer: `Wqkv`, `out_proj`, `fc11`, `fc12`, `fc2`; they run as 48 FP8 GEMM kernels (TensorRT fuses q/k/v into one GEMM and `fc11`/`fc12` into one) |
| Not quantized | everything else stays as in the fp16 build (attention is explicit matmul + softmax, so the Quantize in front of `out_proj` builds on TensorRT 11.1 and 11.4) |
| Weight scale | per tensor, amax / 448, at build time |
| Activation scale | static, per tensor; ModelOpt `FP8_DEFAULT_CFG` (max calibration) on 544 fiqa, nfcorpus and LongEmbed qmsum texts, not the evaluation sets; no outlier scale |
| Dequantize type | fp16 (an fp32 Dequantize gives the same quality and is slower) |

### Results

TensorRT 11.4.0.106, batch 1 (the family accepts batch 1 only), ms per call,
fp16 to FP8. Quality against the Hugging Face fp32 model.

| GPU | 128 tokens | 512 tokens | 2048 tokens | Plan (512) |
|---|---|---|---|---|
| B200 | 0.437 to 0.363 ms (1.20x) | 0.652 to 0.562 ms (1.16x) | 1.238 to 1.028 ms (1.20x) | 263.6 to 157.0 MiB |
| B300 | 0.370 to 0.330 ms (1.12x) | 0.542 to 0.518 ms (1.05x) | 1.131 to 1.015 ms (1.11x) | 263.0 to 156.5 MiB |

| Set | Engine | Cosine mean (worst) | nDCG@10 (HF 0.7032 / 0.6363) | Top-1 agreement with HF |
|---|---|---|---|---|
| SciFact, 512 tokens | fp16 | 0.999998 (0.999991 or higher, both sets) | 0.7027 to 0.7032 | 99.7% to 100% |
| SciFact, 512 tokens | FP8 | 0.99779 (0.9944) | 0.6973 | 92.0% |
| SciFact, 512 tokens | FP8, MLP only (`fc11`, `fc12`, `fc2`) | 0.99933 | - | 96.7% |
| 2WikiMQA, 2048 tokens | fp16 | see SciFact row | 0.6356 to 0.6364 | 99.7% to 100% |
| 2WikiMQA, 2048 tokens | FP8 | 0.99632 (0.9933) | 0.6416 | 92.0% |

No output is non-finite on the held-out and stress inputs. ModelOpt-style fake
quantization of the same recipe on the Hugging Face model gives the same loss
(cosine 0.99779, top-1 92.7%), so the loss comes from the recipe, not from
TensorRT. `Wqkv` is the most sensitive projection (alone: cosine 0.99867,
top-1 93.7%). MLP-only FP8 is 1.03x to 1.18x faster than fp16.

### Why it is not shipped

- Quality loss: 8% of the SciFact queries change their top-1 document, and the
  SciFact nDCG@10 drops from 0.7032 to 0.6973. fp16 keeps both.
- The gain over fp16 is 1.05x to 1.20x; fp16 alone is 5.9x to 6.6x faster than
  the fp32 default at 512 tokens.

### NVFP4 (not tried)

The 137M parameter encoder is not bound by weight reads: 8-bit weights are about
110 MB, read in about 15 us, against 0.54 to 0.63 ms per fp16 call at 512
tokens, where most of the time is GEMM compute and attention. 4-bit weights
cannot pay here.

### When to revisit

- A recipe that keeps top-1 retrieval close to fp16, for example MLP-only FP8
  (top-1 96.7%) or a better calibration of `Wqkv`, measured on more retrieval
  sets.
- A use case that accepts the top-1 change for 1.05x to 1.20x and a 40%
  smaller plan.
