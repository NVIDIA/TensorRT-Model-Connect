# SmolLM3

This family builds the dense SmolLM3 decoder (`HuggingFaceTB/SmolLM3-3B` in the
manifests) for `text_generation` at batch 1. The default build keeps the whole
65536-token context in a TensorRT native KV cache, in bf16, as two engines in one
bundle: a prefill engine (chunks of up to 64 tokens) and a decode engine (one
token per step). A build with `max_sequence_length` below the model context uses
the standard decoder graph instead.

## Quantization

`quantization=fp8` (precision fp16 or bf16) quantizes the original BF16
checkpoint at build time. The q, k, v, gate, up and down projections of every
layer run as FP8 GEMMs, except the `down_proj` of layers 1 and 33 (massive
activations): FP8 weights with a per-tensor scale (amax / 448), static
per-tensor activation scales from `fp8_activation_scales.json` (ModelOpt
`FP8_DEFAULT_CFG`). The attention output projection, the embedding, the tied
`lm_head` and the norms stay bf16.

### NVFP4

NVFP4 was built and measured and is not shipped. No ModelOpt NVFP4 checkpoint
of SmolLM3-3B exists, so the study used self-quantization of the BF16
checkpoint with the FP8 recipe's layer list: the same 214 projections as NVFP4
W4A4 (16-value blocks with e4m3 block scales and one `weight_scale_2` per
tensor, computed at build time; activations through a dynamic block quantize
with a static per-tensor input scale from the same ModelOpt max calibration as
FP8). The fused gate/up dual-GEMM kernel stays on. The plan inspector shows
FP4 layers in every engine (180 per engine on B200, 248 on B300).

Default build (native KV cache, 65536 rows), bf16, batch 1. Speed is GPU time
per engine call, median of 7 interleaved rounds; decode at position 100,
prefill is one 64-token chunk at position 0. Quality is teacher forced over
5766 held-out WikiText-2 test tokens against the PyTorch bf16 model (TensorRT
11.4).

| Item | bf16 | FP8 | NVFP4 | NVFP4 vs FP8 |
|---|---|---|---|---|
| Decode ms, B200 / B300 (TensorRT 11.4) | 3.34 / 3.22 | 2.72 / 2.71 | 2.66 / 2.82 | 1.02x / 0.96x |
| Decode ms at position 4000, B200 / B300 | 4.48 / 4.37 | 3.91 / 3.86 | 3.79 / 3.94 | 1.03x / 0.98x |
| Prefill ms (64 tokens), B200 / B300 | 3.45 / 3.38 | 2.95 / 3.00 | 3.11 / 3.20 | 0.95x / 0.94x |
| Decode / prefill ms, B200 (TensorRT 11.1) | 3.07 / 4.90 | 2.58 / 4.22 | 2.58 / 4.34 | 1.00x / 0.97x |
| Bundle size | 12.46 GiB | 7.60 GiB | 5.46 GiB | -28% |
| KL to PyTorch, B200 / B300 | 0.0012 / 0.0012 | 0.0110 / 0.0105 | 0.101 / 0.102 | 9.1x / 9.7x |
| Top-1 agreement with PyTorch | 98.3% | 94.5% | 84.4% | -10.1 points |
| Perplexity (PyTorch 9.98) | 9.98 | 10.06 | 10.67 | |
| Non-finite logits or KV cache | none | none | none | |

Short-context build (`max_sequence_length` 256), decode / 128-token prefill:
NVFP4 is 0.95x / 1.00x of FP8 on B200 and 0.87x / 0.85x on B300 (TensorRT 11.4).

Reasons:

- Speed: NVFP4 is not faster than FP8 on B300 and only 2% to 3% faster at
  decode on B200; prefill is slower on both. A decode step has about 930
  engine layers and takes about 2.7 ms, while reading the FP8 weights takes
  well under 0.5 ms at full memory bandwidth, so 4-bit weights can save only a
  small share. The dynamic activation quantize adds work in every projection.
- Quality: the NVFP4 KL is about 9x the FP8 KL (the limit is about 2x), and
  top-1 agreement drops by 10 points.
- The only gain is bundle size (2.1 GiB less than FP8).

The NaN scan covered the held-out windows, prefill chunks of 1 to 64 tokens,
and 16k-token random, repeated-token and WikiText prompts followed by 32
decode steps, on B200 and B300.
