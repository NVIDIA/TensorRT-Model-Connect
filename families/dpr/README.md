# DPR

This family builds DPR context and question encoders (BERT-base backbone) as one
TensorRT engine for the `text_to_pooled_features` task; the output is the CLS
row. The manifests build `facebook/dpr-ctx_encoder-single-nq-base` in fp16 with
a 256-token window, and fp32 with tensor parallelism (4 GPUs). Batch is 1 and
the window is static.

## Quantization

`--precision fp16 --quantization fp8` quantizes the original checkpoint at build
time (#2073). Query, key, value, `intermediate.dense` and `output.dense` of
every layer get per-tensor FP8 weights (amax / 448) and static activation scales
from `fp8_activation_scales.json` (ModelOpt `FP8_DEFAULT_CFG`, Natural Questions
questions or passages). The attention output projection, embeddings, layer
norms, attention and GELU stay fp16. Scales ship for
`facebook/dpr-ctx_encoder-single-nq-base` and
`facebook/dpr-question_encoder-single-nq-base`. TensorRT 11.4, batch 1: FP8 is
1.12x to 1.14x faster than fp16 for the question encoder at 64 tokens and 1.02x
to 1.07x for the context encoder at 256 and 512 tokens, the plan is 35% smaller,
and retrieval stays within 0.3 point of PyTorch.

### NVFP4

NVFP4 was not tried; no NVFP4 engine was built. DPR at batch 1 is not bound by
weight reads (numbers from #2073, B200 and B300, TensorRT 11.4):

| Item | Value |
|---|---|
| FP8 projection weights | 78 MB |
| Time to read them at about 8 TB/s | about 0.01 ms |
| FP8 engine call, batch 1, 64 to 512 tokens | 0.33 to 0.54 ms |
| Weight reads as share of the call | 2% to 3% |
| GEMM kernel time at 64 to 256 rows | about 0.2 ms |
| FP8 vs fp16, question encoder, 64 tokens | 1.12x to 1.14x |
| FP8 vs fp16, context encoder, 256 and 512 tokens | 1.02x to 1.07x |

4-bit weights could save at most the weight-read time. The GEMM kernels are
bound by kernel latency and math, not by weight reads, and NVFP4 adds one
dynamic quantize kernel per projection.
