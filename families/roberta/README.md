# RoBERTa encoder tasks

The family supports `encoding` (the first token's hidden state) and `embedding`
(mean pooling followed by L2 normalization). The graph emits encoder hidden
states; it does not load or execute a sequence-classification scoring head.

`reranking` is therefore unsupported. Builds reject that task before checkpoint
loading, and the native factory rejects older bundles labeled `reranking`
before engine deserialization. A hidden-state coordinate is not a reranking
logit. Existing encoding and embedding bundles keep their current task IDs.

Implementing a reranker requires its checkpoint's scoring head, query/document
tokenization and official-reference validation. Loading an encoder backbone
from a sequence-classification checkpoint does not provide those behaviors.

## Runtime input capacity

A fixed-sequence bundle accepts at most the number of token IDs declared by its
`input_ids` engine binding. The limit includes tokenizer-added special tokens.

Native `embed`, `encode`, and token-ID encoding reject inputs above
that capacity with an invalid-argument error before submitting work to the
engine. Shorten or split longer text before invoking the bundle. Dynamic-input
engines retain their existing profile validation.

## Quantization

`--precision fp16 --quantization fp8` quantizes the original checkpoint at build
time (#2028). Query, key, value, `intermediate.dense` and `output.dense` of every
layer get per-tensor FP8 weights (amax / 448) and static activation scales from
`fp8_activation_scales.json` (ModelOpt `FP8_DEFAULT_CFG`, WikiText). The
attention output projection, embeddings, layer norms, attention and GELU stay
16-bit. Scales ship for `FacebookAI/roberta-base` and `FacebookAI/roberta-large`;
XLM-RoBERTa, CamemBERT and fine-tunes need their own calibration run. TensorRT
11.4, batch 1, seq 512: FP8 is 1.03x to 1.06x faster than fp16, the plan is 31%
(base) and 39% (large) smaller, and task accuracy stays within 0.5 point of
PyTorch (SST-2, MNLI).

### NVFP4

NVFP4 was not tried; no NVFP4 engine was built. RoBERTa at batch 1 is not bound
by weight reads (numbers from #2028, B200 and B300, TensorRT 11.4, seq 512):

| Item | base | large |
|---|---|---|
| FP8 projection weights | 78 MB | 277 MB |
| Time to read them at about 8 TB/s | about 0.01 ms | about 0.035 ms |
| FP8 engine call, batch 1 | 0.50 to 0.56 ms | 1.09 to 1.22 ms |
| Weight reads as share of the call | 2% to 3% | 2% to 3% |
| FP8 vs fp16, batch 1 | 1.03x to 1.06x | 1.04x to 1.06x |
| Five quantized projections, fp16 to FP8 (B300, profiler) | 0.299 to 0.261 ms | 0.741 to 0.638 ms |
| Extra Quantize and element-wise work in the FP8 engine | +0.030 ms | +0.071 ms |

4-bit weights could save at most the weight-read time. The GEMMs at 512 rows are
compute bound, and NVFP4 adds one dynamic quantize kernel per projection.
