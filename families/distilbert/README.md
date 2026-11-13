# DistilBERT

This family builds DistilBERT encoders as one TensorRT engine for the
`text_to_pooled_features` and `text_to_embedding` tasks
(`distilbert/distilbert-base-uncased` in the manifests). Batch is 1 and the
window is static (512 tokens by default). The manifests build fp16, bf16 and
fp32 with tensor parallelism (4 GPUs).

## Quantization

`--precision fp16 --quantization fp8` quantizes the original checkpoint at build
time (#2074). `q_lin`, `k_lin`, `v_lin`, `ffn.lin1` and `ffn.lin2` of every
layer get per-tensor FP8 weights (amax / 448) and static activation scales from
`fp8_activation_scales.json` (ModelOpt `FP8_DEFAULT_CFG`, WikiText). The
attention output projection, embeddings, layer norms, attention and GELU stay
fp16. Scales ship for `distilbert-base-uncased`. TensorRT 11.4, batch 1, seq 512:
FP8 is 1.08x (B200) and 1.11x (B300) faster than fp16, the plan is 29% smaller,
and SST-2 accuracy stays within 0.12 point of PyTorch.

### NVFP4

NVFP4 was not tried; no NVFP4 engine was built. DistilBERT at batch 1 is not
bound by weight reads (numbers from #2074, B200 and B300, TensorRT 11.4):

| Item | distilbert-base-uncased |
|---|---|
| FP8 projection weights | 39 MB |
| Time to read them at about 8 TB/s | about 0.005 ms |
| FP8 engine call, batch 1, seq 512 | 0.23 to 0.26 ms |
| Weight reads as share of the call | about 2% |
| FP8 vs fp16, batch 1, seq 512 / seq 128 | 1.08x to 1.11x / 1.05x to 1.09x |

4-bit weights could save at most the weight-read time. The GEMMs at 512 rows
are compute bound, and NVFP4 adds one dynamic quantize kernel per projection.
