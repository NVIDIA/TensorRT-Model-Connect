# DeBERTa

This family builds DeBERTa v1 encoders (`microsoft/deberta-base` in the
manifests; `deberta-large` and the MNLI fine-tunes build from the same code
path) as one TensorRT engine for the `text_to_pooled_features` and
`text_to_embedding` tasks. DeBERTa uses disentangled attention
(content-to-position and position-to-content terms over a shared relative
position table) and post-norm layers. Batch is 1. The family default is fp32;
the manifests build fp16, and fp32 with tensor parallelism (4 GPUs).

## Quantization

`--precision fp16 --quantization fp8` quantizes the original checkpoint at build
time (#2032). The fused `in_proj` (query, key, value), the attention output
projection and the two feed-forward projections of every layer get per-tensor
FP8 weights (amax / 448) and static activation scales from
`fp8_activation_scales.json` (ModelOpt `FP8_DEFAULT_CFG`). Embeddings, the
relative position table and its projections, layer norms, GELU and the
attention matmuls stay fp16. B200 and B300, TensorRT 11.4, batch 1: FP8 is
1.03x to 1.10x faster than fp16, the plan is 30% (base) to 37% (large) smaller,
and masked-LM and MNLI accuracy stay at the fp16 level.

### NVFP4

NVFP4 was not tried; no NVFP4 engine was built. DeBERTa at batch 1 is not bound
by weight reads (numbers from #2032, TensorRT 11.4, batch 1):

| Item | base | large |
|---|---|---|
| FP8 encoder weights | 85 MB | 302 MB |
| Time to read them at about 8 TB/s | about 0.01 ms | about 0.04 ms |
| fp16 engine call, seq 128 to 512 (B200, B300) | 0.50 to 1.25 ms | 1.30 to 2.86 ms |
| Weight reads as share of the call | 1% to 2% | 1% to 2% |
| FP8 vs fp16 | 1.04x to 1.06x | 1.03x to 1.10x |

4-bit weights could save at most the weight-read time, and NVFP4 adds dynamic
activation quantize kernels.
