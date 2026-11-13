# ELECTRA

This family builds ELECTRA discriminators as one TensorRT engine for the
`text_to_pooled_features` and `text_to_embedding` tasks
(`google/electra-base-discriminator` in the manifests). Batch is 1 and the
window is static (512 tokens by default, 256 in the manifests). The manifests
build fp16, bf16 and fp32 with tensor parallelism (4 GPUs).

## Quantization

`--precision fp16 --quantization fp8` quantizes the original checkpoint at build
time (#2078). Query, key, value, `intermediate.dense` and `output.dense` of
every layer get per-tensor FP8 weights (amax / 448) and static activation scales
from `fp8_activation_scales.json` (ModelOpt `FP8_DEFAULT_CFG`, WikiText). The
attention output projection, embeddings, layer norms, attention and GELU stay
fp16. Scales ship for `electra-base-discriminator` and
`electra-large-discriminator`. TensorRT 11.4, batch 1, seq 512: FP8 is 1.03x to
1.07x faster than fp16, the plan is 35% (base) and 41% (large) smaller, and
replaced-token-detection accuracy stays within 0.08 point of PyTorch.

### NVFP4

NVFP4 was not tried; no NVFP4 engine was built. ELECTRA at batch 1 is not bound
by weight reads (numbers from #2078, B200 and B300, TensorRT 11.4):

| Item | base | large |
|---|---|---|
| FP8 projection weights | 78 MB | 277 MB |
| Time to read them at about 8 TB/s | about 0.01 ms | about 0.035 ms |
| FP8 engine call, batch 1, seq 512 | 0.48 to 0.53 ms | 1.04 to 1.17 ms |
| Weight reads as share of the call | 2% to 3% | 2% to 3% |
| FP8 vs fp16, batch 1, seq 512 | 1.04x to 1.07x | 1.03x to 1.06x |

4-bit weights could save at most the weight-read time. The GEMMs at 512 rows
are compute bound, and NVFP4 adds one dynamic quantize kernel per projection.
