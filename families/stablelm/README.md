# StableLM

This family builds StableLM decoders (StableLM-2 and StableLM-3B-4E1T:
LayerNorm, SwiGLU MLP, partial rotary embeddings, QKV biases) for the
`text_generation` task, with a decode engine and a prefill engine
(`stabilityai/stablelm-2-1_6b` in the manifests). Batch is 1. The manifests
build fp16 with layer 23 in fp32 (`fp32_layers=[23]`) and fp32 with tensor
parallelism (4 GPUs).

## Quantization

`--precision fp16 --quantization fp8` quantizes the original checkpoint at build
time (#2001). The q, k, v, gate, up and down projections of every layer get
per-tensor FP8 weights (amax / 448) and static activation scales from
`fp8_activation_scales.json` (ModelOpt `FP8_DEFAULT_CFG`, Wikipedia and Nemotron
samples). The attention output projection, `lm_head`, embeddings, norms, a few
outlier down projections and the layers in `fp32_layers` stay unquantized.
Dequantize and the matmul run in fp32 (a fp16 Dequantize gives non-finite
decode logits). Scales ship for `stablelm-2-1_6b` and `stablelm-3b-4e1t`.
TensorRT 11.4, batch 1: the decode plan is 33% (1.6B) and 41% (3B) smaller,
prefill is 1.09x to 1.33x faster and decode 1.045x to 1.21x faster than fp16.
KL against the PyTorch model is 13 to 14 times the fp16 floor.

### NVFP4

NVFP4 was not tried; no NVFP4 engine was built. At hidden sizes 2048 and 2560,
decode is bound by kernel count, not by weight reads: the 1.6B decode plan runs
412 layers in 1.27 ms on B300 (#2001). FP8, which halves the weight bytes of
fp16, gives only a small decode gain (numbers from #2001, TensorRT 11.4, batch
1, 256-token build, decode at position 128):

| Model | GPU | fp16 decode | FP8 decode | FP8 vs fp16 |
|---|---|---|---|---|
| stablelm-2-1_6b | B300 | 1.2707 ms | 1.2160 ms | 1.045x |
| stablelm-2-1_6b | B200 | 1.4061 ms | 1.1974 ms | 1.17x |
| stablelm-3b-4e1t | B300 | 2.0434 ms | 1.8251 ms | 1.12x |
| stablelm-3b-4e1t | B200 | 2.2650 ms | 1.8778 ms | 1.21x |

4-bit weights could save at most part of the remaining weight-read time, and
NVFP4 adds one dynamic quantize kernel per projection. On the earlier dense
decoder families (gpt2, opt, xglm, bloom, codegen), NVFP4 gave no decode gain
and 3 to 10 times the KL of FP8 (#2001).
