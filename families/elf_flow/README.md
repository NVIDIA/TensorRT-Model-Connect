# ELF Flow

This family builds the ELF-B embedded language flow models
(`embedded-language-flows/ELF-B-xsum`, `ELF-B-de-en`, `ELF-B-owt`) from the
GitHub ELF source as a TensorRT denoiser and decoder engine (12 transformer
blocks, hidden size 768, 104.6M parameters) plus a T5-small text encoder engine.
The manifests use fp16 with blocks 5 to 11 in fp32 for xsum (1088 tokens) and
de-en (128 tokens), and fp32 for owt (1024 tokens). The family has no quantized
build.

## FP8 quantization study (not shipped)

FP8 self-quantization of the block projections of ELF-B was built and measured
on TensorRT 11.4 and 11.1 (B200, B300). It is not shipped: with the precision of
the manifests, FP8 in the fp16 blocks gives 1.00x to 1.01x, and FP8 in all
blocks gives 1.01x to 1.04x over an all-fp16 engine. The only large speedup
(1.7x to 1.9x) comes from replacing the fp32 blocks with FP8, which removes the
fp32 blocks that the manifests keep for exact token agreement with the upstream
model, and which does not build on TensorRT 11.4. This section records the
recipe, the method and the numbers, so that the study does not have to be
repeated. The FP8 code was not merged; this family has no FP8 build.

### Summary and decision

Denoiser call time, TensorRT 11.4, median ms (x is baseline time over FP8 time):

| Configuration | Baseline B200 / B300 | FP8 B200 / B300 | x B200 / B300 |
|---|---|---|---|
| xsum manifest (1088 tokens, fp16, blocks 5 to 11 fp32), FP8 in the 5 fp16 blocks | 6.402 / 5.618 | 6.390 / 5.601 | 1.002 / 1.003 |
| de-en manifest (128 tokens, fp16, blocks 5 to 11 fp32), FP8 in the 5 fp16 blocks | 1.365 / 1.256 | 1.356 / 1.248 | 1.007 / 1.006 |
| all-fp16, 1088 tokens, FP8 in all 12 blocks | 2.635 / 2.542 | 2.585 / 2.484 | 1.019 / 1.023 |
| all-fp16, 128 tokens, FP8 in all 12 blocks | 0.635 / 0.597 | 0.613 / 0.574 | 1.034 / 1.042 |
| manifests with FP8 in the fp32 blocks too (fp32 Dequantize there), owt fp32 with FP8 | - | build fails | - |

Decision: do not ship FP8 for ELF Flow.

- The model is small (12 blocks of 7M parameters). One call is 0.6 ms to 2.6 ms
  in fp16 and is bound by launches, attention, norms and RoPE; the projection
  GEMMs leave little time to save.
- The manifests keep blocks 5 to 11 in fp32 (xsum, de-en) or the whole model in
  fp32 (owt). Those fp32 blocks, not the GEMM precision of the fp16 blocks, set
  the run time: an all-fp16 engine is already 2.1x to 2.4x faster than the xsum
  and de-en manifest engines. FP8 in the fp16 blocks only gives 1.00x to 1.01x.
- Replacing the fp32 blocks with FP8 is 1.7x to 1.9x faster on TensorRT 11.1,
  but it removes the fp32 blocks that the xsum and de-en cases need for exact
  token agreement (threshold `contract_min_upstream_token_agreement_rate` 1.0),
  and on TensorRT 11.4 the engine does not build.
- The only gain at the manifest precision is plan size (328 MiB to 294 MiB).

### Recipe that was measured

No FP8 or NVFP4 checkpoint of ELF exists on the Hub (only the original JAX and
torch weights). The recipe is the dense-layer W8A8 recipe of the other FP8
builds, applied to the four projections of each block.

| Choice | Decision |
|---|---|
| Quantized (48 matmuls) | `attn.qkv`, `attn.proj`, `mlp.w12`, `mlp.w3` of each of the 12 blocks; in the fp16-block variant only the blocks that run fp16 (20 matmuls) |
| Kept | input projections, timestep and self-conditioning embedders, attention matmuls and softmax, final norm and linear, decoder head and 32128-word unembedding, T5 encoder |
| Weight | `FP8 constant -> Dequantize`, per-tensor scale amax / 448 at build time |
| Activation | `Quantize -> Dequantize` with a static per-tensor scale (speed only, see Method) |
| Dequantize type | fp16 in fp16 blocks; fp32 in fp32 blocks (variant "FP8 in the fp32 blocks too") |

### Method

- GPUs: B200 and B300, one GPU per job; TensorRT 11.4.0.106 (target) and
  11.1.0.106.
- Engine: the family's `build_elf_flow_engine` with the block `_dense` calls
  replaced by the FP8 form above; synthetic weights with the real ELF-B shapes
  (speed does not depend on weight values) and the manifest precision,
  sequence length and `fp32_layers`.
- Speed: one denoiser call, CUDA graph replay, 200 replays per round, median of
  9 interleaved rounds (5 on TensorRT 11.1); all variants of one configuration
  built and timed in the same process.
- Non-finite scan: every output of every built engine (all finite).
- Quality: FP8 quality on the real weights was not measured, because no FP8
  variant that keeps the manifest precision is faster. The fp16 baseline was
  checked on the real weights (one denoiser step, 4 timesteps x 2 decoder modes,
  fp16 against an fp32 engine, B200, TensorRT 11.1): finite everywhere,
  denoised relative L2 0.002 to 0.003 for xsum and owt; for owt at t = 0.95
  with the decoder on, relative L2 0.024 (denoised) and 0.027 (logits), top-1
  0.923. So owt is already lossy in fp16 at high t, and FP8 would add error on
  top.

### Results

Manifest precision, FP8 only in the fp16 blocks (blocks 0 to 4), ms:

| GPU | TensorRT | Tokens | Manifest engine | FP8 fp16 blocks | x | plan MiB |
|---|---|---|---|---|---|---|
| B200 | 11.4 | 1088 (xsum) | 6.402 | 6.390 | 1.002 | 328.5 to 294.8 |
| B300 | 11.4 | 1088 (xsum) | 5.618 | 5.601 | 1.003 | 328.6 to 294.9 |
| B200 | 11.4 | 128 (de-en) | 1.365 | 1.356 | 1.007 | 328.1 to 294.4 |
| B300 | 11.4 | 128 (de-en) | 1.256 | 1.248 | 1.006 | 328.1 to 294.4 |
| B200 | 11.1 | 1088 | 6.303 | 6.297 | 1.001 | 328.6 to 294.9 |
| B300 | 11.1 | 1088 | 5.768 | 5.744 | 1.004 | 328.7 to 295.0 |
| B200 | 11.1 | 128 | 1.351 | 1.338 | 1.009 | 328.2 to 294.5 |
| B300 | 11.1 | 128 | 1.252 | 1.245 | 1.006 | 328.2 to 294.5 |

All-fp16 engine (no fp32 blocks), FP8 in all 12 blocks, ms:

| GPU | TensorRT | 128 tokens fp16 / FP8 (x) | 1024 tokens fp16 / FP8 (x) | 1088 tokens fp16 / FP8 (x) | plan MiB |
|---|---|---|---|---|---|
| B200 | 11.4 | 0.635 / 0.613 (1.034) | 2.477 / 2.412 (1.027) | 2.635 / 2.585 (1.019) | 200.6 to 119.6 |
| B300 | 11.4 | 0.597 / 0.574 (1.042) | 2.370 / 2.319 (1.022) | 2.542 / 2.484 (1.023) | 200.6 to 119.7 |
| B200 | 11.1 | 0.635 / 0.611 (1.039) | 2.521 / 2.479 (1.017) | 2.682 / 2.667 (1.006) | 200.6 to 119.7 |
| B300 | 11.1 | 0.598 / 0.581 (1.030) | 2.417 / 2.375 (1.018) | 2.592 / 2.545 (1.018) | 200.6 to 119.7 |

FP8 in the fp32 blocks too (TensorRT 11.1; on 11.4 these engines do not build),
ms:

| GPU | Configuration | Manifest engine | FP8 | x | plan MiB |
|---|---|---|---|---|---|
| B200 | xsum, 1088 tokens | 6.303 | 3.517 | 1.79 | 328.6 to 153.1 |
| B300 | xsum, 1088 tokens | 5.768 | 3.362 | 1.72 | 328.7 to 153.2 |
| B200 | de-en, 128 tokens | 1.351 | 0.727 | 1.86 | 328.2 to 152.7 |
| B300 | de-en, 128 tokens | 1.252 | 0.699 | 1.79 | 328.2 to 152.7 |
| B200 | owt, 1024 tokens, fp32 | 10.217 | 3.809 | 2.68 | 400.2 to 157.2 |
| B300 | owt, 1024 tokens, fp32 | 9.846 | 3.677 | 2.68 | 400.2 to 157.2 |

These FP8 engines are still slower than the all-fp16 engine of the same length
(for example owt 3.809 ms FP8 against 2.521 ms fp16 on B200), so their gain is
the removal of fp32 compute, not FP8.

TensorRT 11.4 build failure: with FP8 Quantize and Dequantize in an fp32 block
(fp32 Dequantize, fused `IAttention` path of the fp32 blocks), the 11.4 builder
reports that it cannot find an implementation, after it fails to type an FP8
softmax inside the fused attention. The same network builds on 11.1. This only
affects the study code; the manifests on main have no FP8 layer.

### NVFP4 (not tried)

The FP8 GEMMs already give at most 1.04x; NVFP4 GEMMs on these small matrices
(M = 140 to 1100, K = 768 or 2048) would not change that, and 4-bit weights
would move the outputs further from the exact token agreement the manifests
check.

### What was not tried

| Item | Note |
|---|---|
| FP8 quality on the real weights, calibration of activation scales | not measured (no faster variant at the manifest precision) |
| fp16 for blocks 5 to 11 of xsum and de-en | outside this study; the manifests need these blocks in fp32 for exact token agreement |
| FP8 attention (BMM1, BMM2) | not measured |
| H100, the native runtime with an FP8 bundle | not measured |

### When to revisit

- If the xsum and de-en cases move to all-fp16 (no fp32 blocks), FP8 is worth
  at most 1.01x to 1.04x on top, which still does not pay.
- If the fp32 blocks could run FP8 within the token-agreement contract and
  TensorRT 11.4 or newer builds FP8 Quantize and Dequantize in front of the fused
  fp32 attention, FP8 in all blocks would give 1.7x to 1.9x over the manifest
  engine; this needs a real-weight quality check first.
