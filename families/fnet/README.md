# FNet

This family builds FNet encoders (`google/fnet-base`, `google/fnet-large`) for
`text_to_pooled_features`: batch 1, a fixed padded window (512 tokens when
`max_sequence_length` is not given, 256 in the manifests), output = the CLS row.
Token mixing is the real part of a 2D DFT, built as fixed cosine / sine
matrices: the sequence axis in fp32, the hidden axis in the work precision. The
manifests use fp16 (TP1) and fp32 (TP4). The family has no quantized build.

## Known issue: fp16 NaN at the 512-token window

fp16 builds at the default 512-token window return NaN for nearly every real
input: 293 / 300 held-out WikiText-2 passages for fnet-base and 300 / 300 for
fnet-large (B200 and B300, TensorRT 11.4; fnet-base on TensorRT 11.1: 98 / 100).
The 2D Fourier output exceeds the fp16 range (largest value in the fp32
PyTorch model: 69,042 for base and 85,324 for large at 512, above 65,504): the
hidden-axis fp16 GEMM overflows to inf and the next layer norm turns it into
NaN. Windows of 256 and 128 have no NaN (large at 256: 55,233, 16% below the
limit), and the premerge case uses 256, so CI does not see it.

A probe that scales the input of the hidden-axis DFT by 1/64 and the fp32
result back by 64 removes the NaN at the same speed (fnet-base, 512, B200:
0.752 ms vs 0.756 ms) with masked-LM KL 5.9e-06 against PyTorch fp32. Running
the hidden-axis DFT in fp32 also works but is 1.41x to 1.47x slower. Neither is
applied here; this is a known issue.

## FP8 quantization study (not shipped)

FP8 self-quantization of the feed-forward projections was built and measured
on TensorRT 11.4 (and 11.1) for fnet-base and fnet-large. It is not shipped: FP8
is only 1.03x to 1.11x faster than fp16 at batch 1 and loses much more quality
than on attention encoders (masked-LM top-1 agreement with PyTorch 84% to 96%,
fp16: 99.6% to 99.9%). This section records the recipe, the method and the
numbers, so that the study does not have to be repeated. The FP8 code was not
merged; this family has no FP8 build.

### Summary and decision

| Item | fp16 | FP8 (fp16 engine) | FP8 vs fp16 |
|---|---|---|---|
| Latency, batch 1, B200 (base 128 to 512, large 256 and 512, TensorRT 11.4) | - | - | 1.03x to 1.11x |
| Latency, batch 8 x 512, B200 / B300 | - | - | 1.05x to 1.08x |
| Plan size, base / large, 512 | 161.9 / 457.2 MiB | 108.0 / 265.4 MiB | -33% / -42% |
| Masked-LM KL vs PyTorch fp32, mean | 5e-06 to 3e-05 | 0.007 to 0.05 | - |
| Masked-LM top-1 agreement with PyTorch fp32 | 99.6% to 99.9% | 84% to 96% | - |
| STS-B Spearman, base 512 / 256 (PyTorch 0.2199 / 0.2032) | 0.2201 / 0.2049 | 0.2066 / 0.1731 | -0.014 / -0.032 |
| Non-finite outputs | see "Known issue" | 0 | none |

Decision: do not ship FP8 for FNet.

- Speed: the two feed-forward GEMMs are only 18% to 30% of the fp16 time, and
  FP8 shortens them by 15% to 26%, a saving of 4% to 7% of the call. The largest
  kernel is the fp32 sequence-axis DFT, which FP8 does not touch.
- Quality: FNet mixes every token and hidden unit in every layer with a fixed
  Fourier transform and no attention normalization, so the per-projection FP8
  error spreads and builds up over 12 / 24 layers. For comparison, electra base
  FP8 has task KL 0.0008.
- The variant that helps quality (`output.dense` left fp16) halves the KL but
  removes the speed gain (0.92x to 0.98x of fp16).
- The quality bar of issue #1724 (CLS relative L2 <= 0.04 on the first 400
  STS-B sentences) fails: base FP8 0 / 400 at 256 and 512 (fp16 400 / 400),
  large FP8 379 / 400 at 512 and 48 / 400 at 256 (B200).

### Recipe that was measured

No FP8 or NVFP4 checkpoint of FNet exists on the Hub (only the google fp32
checkpoints and community fine-tunes), and no official quantization config
exists for it. Each quantized projection is built as
`activation -> Quantize -> Dequantize(fp16) -> MatMul -> + bias`, with the
weight `FP8 constant -> Dequantize(fp16)`.

| Choice | Decision |
|---|---|
| Quantized | per layer `intermediate.dense` and `output.dense`, the only projections in an FNet layer: 24 (base) / 48 (large) FP8 GEMMs |
| fp16 / fp32 | embeddings, embedding projection, the Fourier transform (fixed cosine / sine matrices), layer norms, GELU |
| Weight scale | per tensor, amax / 448 of the fp32 weights, at build time |
| Activation scale | static, per tensor; ModelOpt 0.46.1 `FP8_DEFAULT_CFG` (max calibration), keyed by model config and two weight sums |
| Calibration data | 544 WikiText-2 train passages (16 to 510 tokens), framed `[CLS] ... [SEP]` and padded to the window (512 and 256) as the runtime pads |
| Outlier rule | a scale above 20x the median of its kind stays fp16; measured: no outlier (largest ratio 7.4x base, 4.1x large) |
| Dequantize type | fp16; no non-finite value, fp32 Dequantize was not needed |

At 512 tokens FP8 was measured on top of the 1/64 scaling probe of "Known
issue" (`fp16+hs` below), because fp16 at 512 returns NaN; at 128 and 256 FP8 is
built on the unchanged graph.

### Method

- GPUs: B200 and B300, one GPU per job; TensorRT 11.4.0.106 (target) and
  11.1.0.106 (B200, fnet-base); torch 2.14.0, Transformers 5.17.0.
- Reference: Transformers `FNetForPreTraining` in fp32, TF32 off.
- Quality: 300 held-out WikiText-2 test passages of 16 to 510 tokens, padded to
  the window. Masked LM: 15% of the body tokens replaced by `[MASK]` (seeded),
  the PyTorch MLM head applied to the engine features; KL(PyTorch || engine) per
  masked token. STS: CLS cosine Spearman on the STS-B test split (1,379 pairs).
- Non-finite scan: the held-out passages, 2,758 STS-B sentences, 22 stress
  inputs (random ids, repeated token, all `[MASK]`, `[CLS][SEP]`) and one batch of 8.
- Speed: CUDA graph replay, median of 9 interleaved rounds of 50 calls. Batch 8
  is a scratch build (the family builds batch 1 only).

### Results

Speed, TensorRT 11.4 (ms; x is fp16 time over FP8 time; at 512 fp16 is
`fp16+hs`, the fp16 build of the family at 512 has the same speed but NaN):

| Model, window | GPU | fp32 | fp16 | FP8 | x | FP8 without `output.dense` |
|---|---|---|---|---|---|---|
| base, 512 | B200 | 2.838 | 0.752 | 0.715 | 1.05 | 0.775 |
| base, 512 | B300 | 2.743 | 0.690 | 0.652 | 1.06 | 0.714 |
| base, 256 | B200 | 1.726 | 0.507 | 0.466 | 1.09 | 0.525 |
| base, 256 | B300 | 1.704 | 0.462 | 0.429 | 1.08 | 0.491 |
| base, 128 | B200 | 1.214 | 0.413 | 0.371 | 1.11 | 0.447 |
| base, 128 | B300 | 1.163 | 0.391 | 0.357 | 1.10 | 0.403 |
| large, 512 | B200 | 9.663 | 1.598 | 1.547 | 1.03 | 1.649 |
| large, 512 | B300 | 8.066 | 1.455 | 1.397 | 1.04 | 1.498 |
| large, 256 | B200 | 5.347 | 1.058 | 1.019 | 1.04 | 1.110 |
| large, 256 | B300 | 5.070 | 0.969 | 0.925 | 1.05 | 1.030 |
| base, batch 8 x 512 | B200 / B300 | 17.25 / 15.19 | 2.612 / 2.492 | 2.464 / 2.374 | 1.06 / 1.05 | 2.685 / 2.604 |
| large, batch 8 x 512 | B200 / B300 | 61.13 / 45.64 | 6.840 / 6.275 | 6.177 / 5.898 | 1.08 / 1.06 | 6.844 / 6.472 |

TensorRT 11.1 (B200, base): 0.751 ms fp16, 0.714 ms FP8 (1.05x) at 512; 0.501
and 0.468 ms (1.07x) at 256. Plan size (B200, 512): base fp32 321.1, fp16 161.9,
FP8 108.0 MiB; large 911.0, 457.2, 265.4 MiB.

Quality, B200, TensorRT 11.4 (B300 is in the same class: base 512 FP8 KL 0.0097,
agreement 94.14%; TensorRT 11.1, base 512, 100 passages: KL 0.0097, 93.30%):

| Model, window | Engine | MLM KL mean (p99) | MLM top-1 agree | CLS cosine mean (min) | CLS relative error | STS Spearman |
|---|---|---|---|---|---|---|
| base, 512 | fp16+hs | 5.9e-06 (2.9e-05) | 99.83% | 0.99998 (0.99997) | 0.0023 | 0.2201 |
| base, 512 | FP8 | 0.0095 (0.064) | 93.99% | 0.99738 (0.99295) | 0.082 | 0.2066 |
| base, 512 | FP8 without `output.dense` | 0.0057 (0.042) | 95.38% | 0.99898 (0.99580) | 0.059 | 0.2090 |
| large, 512 | fp16+hs | 4.6e-06 (2.7e-05) | 99.91% | 0.99998 | 0.0025 | 0.2305 |
| large, 512 | FP8 | 0.0066 (0.054) | 95.55% | 0.99929 (0.99724) | 0.073 | 0.2289 |
| base, 256 | fp16 | 1.2e-05 (6.7e-05) | 99.81% | 0.99998 | 0.0032 | 0.2049 |
| base, 256 | FP8 | 0.021 (0.113) | 89.48% | 0.99558 (0.97339) | 0.117 | 0.1731 |
| large, 256 | fp16 | 2.9e-05 (1.0e-04) | 99.61% | 0.99997 | 0.0038 | 0.1590 |
| large, 256 | FP8 | 0.050 (0.247) | 83.52% | 0.99858 (0.99691) | 0.118 | 0.1690 |

PyTorch fp32 STS Spearman: base 0.2199 (512) / 0.2032 (256), large 0.2302 /
0.1585. The premerge prompt (`The capital of France is Paris.`, 256) gives CLS
cosine 0.97455 for FP8 (fp16 0.99981), above the case threshold of 0.8.

Kernel time per call, fnet-base, B200, TensorRT 11.4 (layer profiler, ms):

| Window | Build | fp32 sequence-axis DFT | other GEMMs | FP8 feed-forward GEMMs | other kernels |
|---|---|---|---|---|---|
| 512 | fp16 | 0.337 | 0.335 | - | 0.173 |
| 512 | FP8 | 0.340 | 0.136 | 0.170 | 0.171 |
| 256 | fp16 | 0.144 | 0.306 | - | 0.140 |
| 256 | FP8 | 0.145 | 0.129 | 0.146 | 0.134 |
| batch 8 x 512 | fp16 | 1.698 | 0.711 | - | 0.328 |
| batch 8 x 512 | FP8 | 1.702 | 0.222 | 0.361 | 0.329 |

"Other GEMMs" in fp16 holds the hidden-axis DFT (0.136 ms at 512, the same in
both builds), the feed-forward GEMMs and the embedding projection. The fp16
feed-forward GEMMs are about 0.20 ms (512), 0.18 ms (256) and 0.49 ms (batch 8);
FP8 brings them to 0.17, 0.15 and 0.36 ms. The fp32 sequence-axis DFT (an FFMA
kernel, TF32 off) is 40% of the batch-1 time at 512 and 62% at batch 8.

### NVFP4 (not tried)

FNet at batch 1 is not bound by weight reads (the FP8 feed-forward weights, 57
MB base / 201 MB large, are read in about 0.01 to 0.03 ms of a 0.7 to 1.6 ms
call), and FP8 is already too lossy; 4-bit would lose more.

### What was not tried

| Item | Note |
|---|---|
| Per-channel weight scales, percentile or entropy calibration, SmoothQuant | not measured |
| FP8 for the hidden-axis DFT GEMM | not measured; cosine / sine values in [-1, 1] lose precision in e4m3 |
| H100, the native runtime with an FP8 bundle | not measured |

### When to revisit

- A faster sequence-axis DFT (TF32 or an FFT kernel): the feed-forward GEMMs
  would then be a larger share of the time and the FP8 gain would grow.
- A recipe that keeps masked-LM top-1 agreement near the fp16 level (above 99%)
  without leaving `output.dense` in fp16.

Batch 8 gave about the same ratio as batch 1 (1.05x to 1.08x), so larger batches
alone do not change the result.
