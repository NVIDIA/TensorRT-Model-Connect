# OpenFold3

This family builds OpenFold3 v0.5.0 (`of3-ob-2025-06-30-174k` checkpoint) for
protein structure prediction as a bundle of TensorRT plans that the native C++
runtime runs on one CUDA stream: input embedding, the 48 Pairformer blocks and
the trunk cycle (3 recycles), diffusion conditioning, the atom transformers, the
24 diffusion token blocks with the 200-step sampler, and the confidence heads.
Bundles are exact-request: one protein chain, query-only MSA, one diffusion
sample, batch-1 plans. The default precision is mixed fp16 (projections fp16;
norms, attention, triangle contractions and the diffusion state fp32); mixed
bf16 is the alternative. See `examples/models/openfold3/README.md` for the
build and the qualified ubiquitin profile. The family has no quantized build.

## FP8 quantization study (not shipped)

FP8 self-quantization of the Pairformer pair stack and the diffusion token
blocks was built and measured on TensorRT 11.4 (and 11.1) for the qualified
ubiquitin request (76 tokens, 601 atoms). It is not shipped: the GPU time per
prediction is only 1.08x to 1.09x faster, the process wall time is not faster,
and the predicted structure moves well beyond the fp16 rebuild floor (lDDT 0.79
to 0.96 against the fp16 engine). Two FP8 builds of the same recipe gave
different structures. This section records the recipe, the method and the
numbers, so that the study does not have to be repeated. The FP8 code was not
merged; this family has no FP8 build.

### Summary and decision

| Item | fp16 (default) | FP8 | FP8 vs fp16 |
|---|---|---|---|
| GPU time per prediction, B200 / B300 (TensorRT 11.4) | 575 / 528 ms | 525 to 531 / 484 to 489 ms | 1.08x to 1.09x |
| Process wall time, B200 / B300 | 2.6 to 3.4 / 2.5 to 2.7 s | 2.7 to 3.6 / 2.5 to 3.0 s | not faster (run-to-run spread 0.2 to 1.2 s) |
| Bundle plan size | 907 to 908 MiB | 724 to 741 MiB | -18% to -20% |
| lDDT vs fp16 engine (fp16 rebuild 1.000, bf16 0.998 to 1.000) | 1.000 | 0.785 to 0.962 | far below the floor |
| pTM | 0.852 to 0.853 | 0.798 to 0.837 | -0.015 to -0.055 |
| Non-finite outputs | none | none | none |

Decision: do not ship FP8 for OpenFold3.

- Speed: the diffusion token blocks are about 64% of the GPU time and their FP8
  plans are 1.13x to 1.15x faster, which gives 8% to 9% of GPU time. The
  process wall time (2.5 to 3.5 s) is mostly bundle load and setup, so the gain
  is not visible there.
- FP8 in the Pairformer pair stack is slower than fp16 (K = 128 GEMMs at 76
  tokens).
- Quality: FP8 moves the structure far beyond the floor, and two FP8 builds of
  the same recipe on B300 landed in different structures (lDDT vs fp16 0.954
  and 0.785), while fp16 rebuilds stay at lDDT 1.000.
- bf16 (already supported) is 1.02x to 1.04x faster than fp16 at floor quality.

### Recipe that was measured

No FP8, NVFP4 or ModelOpt checkpoint of OpenFold3 exists on the Hub, and no
official quantization config exists for it. The recipe is FP8 W8A8 with static
per-tensor scales. Each quantized projection is built as
`input -> Quantize -> Dequantize(fp16) -> MatMul`, with the weight
`FP8 constant -> Dequantize(fp16)`.

| Choice | Decision |
|---|---|
| Quantized, Pairformer (48 blocks) | triangle multiplication `linear_a_p`, `linear_b_p`, `linear_a_g`, `linear_b_g`, `linear_g`, `linear_z`; triangle attention `mha.linear_q/k/v/g/o`; pair transition `swiglu.linear_a/b`, `linear_out` |
| Quantized, diffusion token blocks (24) | attention `mha.linear_q/k/v/g/o`; adaLN `linear_g`, `linear_s`, `linear_ada_out`; conditioned transition `swiglu.linear_a/b`, `linear_out`, `linear_g` and its adaLN |
| Kept fp16 / fp32 | triangle-attention bias `linear_z` (4 heads), token pair-bias `linear_z`, sequence attention, single transition, norms, attention cores, triangle contractions, atom transformers, confidence heads |
| Weight scale | per tensor, amax / 448, at build time |
| Activation scale | static, per tensor, max calibration (max abs input / 448) with forward pre-hooks on the eager OpenFold3 v0.5.0 model in BF16 autocast, 4 single-sequence proteins (thioredoxin, CI2, engrailed homeodomain, calmodulin N-lobe), disjoint from ubiquitin |
| Outlier rule | a scale above 20x the median of its kind stays unquantized; measured: no outlier |
| Dequantize type | fp16; no non-finite value |

Two variants were measured: "pair + token" (all of the above, 1536
projections) and "token only" (diffusion token blocks only).

### Method

- GPUs: B200 and B300, one GPU per job; TensorRT 11.4.0.106 (target) and
  11.1.0.106 (B200 only); torch 2.14.0 (CUDA 13).
- Request: the qualified ubiquitin request (3 recycles, 200 diffusion steps,
  1 sample, seed 42), the same pinned random samples in every engine.
- GPU time per prediction: median time of each plan times its call count,
  summed. Wall time: median of 3 or 5 interleaved runs of the native
  qualification binary (whole process, including bundle load).
- Quality: lDDT, RMSD and pLDDT error against the fp16 engine (floor: an fp16
  rebuild and the bf16 engine), and lDDT against the BF16 eager OpenFold3
  reference from `examples/models/openfold3/generate_reference.py`. The fp16
  engine itself reaches lDDT 0.783 (CA RMSD 2.58 A) against that reference.
- The token-only rows come from a second job with a fresh fp16 build, so they
  are compared to that build.

### Results

GPU time split per prediction, fp16, TensorRT 11.4:

| Part | Calls | B200 ms | B300 ms |
|---|---|---|---|
| Diffusion token transformer (4 plans x 6 blocks) | 200 x 4 | 367 (64%) | 330 (63%) |
| Diffusion score input and output | 200 x 2 | 150 | 140 |
| Pairformer (8 plans x 6 blocks) and trunk cycle | 4 x 9 | 56 | 55 |
| Embedder, conditioning, confidence | - | 2 | 2 |
| Total | - | 575 | 528 |

Speed and size, TensorRT 11.4 (x is fp16 GPU time over variant GPU time):

| GPU | Variant | Pairformer plan ms | Token plan ms | GPU time ms | x | Wall s | Plans MiB |
|---|---|---|---|---|---|---|---|
| B200 | fp16 (default) | 1.568 | 0.459 | 575 | 1.00 | 3.43 | 908 |
| B200 | bf16 | 1.393 | 0.436 | 553 | 1.04 | 2.76 | 911 |
| B200 | FP8 pair + token | 1.760 | 0.400 | 531 | 1.08 | 3.61 | 740 |
| B200 | fp16 rebuild (second job) | 1.544 | 0.458 | 574 | 1.00 | 2.63 | - |
| B200 | FP8 token only (second job) | 1.536 | 0.399 | 525 | 1.09 | 2.67 | 726 |
| B300 | fp16 (default) | 1.569 | 0.413 | 528 | 1.00 | 2.65 | 907 |
| B300 | bf16 | 1.345 | 0.409 | 519 | 1.02 | 2.59 | 911 |
| B300 | FP8 pair + token | 1.635 | 0.361 | 484 | 1.09 | 2.98 | 741 |
| B300 | fp16 rebuild (second job) | 1.605 | 0.410 | 526 | 1.00 | 2.55 | - |
| B300 | FP8 token only (second job) | 1.607 | 0.363 | 489 | 1.08 | 2.47 | 724 |

Plan times are for one plan of 6 blocks. TensorRT 11.1 (B200): fp16 and FP8
pair + token bundles build (913 and 749 MiB) and run; the Pairformer plan goes
from 1.502 to 1.697 ms with FP8 and the token plan stays at 0.422 ms, so there
is no gain on 11.1.

Quality, ubiquitin:

| GPU | Variant | lDDT vs fp16 | RMSD vs fp16 A | pLDDT MAE vs fp16 | pTM | lDDT vs eager ref |
|---|---|---|---|---|---|---|
| B200 | fp16 rebuild | 1.000 | 0.03 | 0.21 | 0.853 | 0.783 |
| B200 | bf16 | 1.000 | 0.04 | 0.21 | 0.854 | 0.783 |
| B200 | FP8 pair + token | 0.944 | 0.80 | 2.84 | 0.837 | 0.777 |
| B200 | FP8 token only | 0.961 | 0.68 | 2.32 | 0.825 | 0.791 |
| B300 | fp16 rebuild | 1.000 | 0.04 | 0.23 | 0.852 | 0.783 |
| B300 | bf16 | 0.998 | 0.20 | 0.75 | 0.852 | 0.782 |
| B300 | FP8 pair + token, build 1 | 0.954 | 0.70 | 6.08 | 0.806 | 0.780 |
| B300 | FP8 pair + token, build 2 | 0.785 | 2.41 | 6.44 | 0.804 | 0.880 |
| B300 | FP8 token only | 0.962 | 0.65 | 1.74 | 0.832 | 0.787 |
| B200, TensorRT 11.1 | FP8 pair + token | 0.913 | 1.66 | 6.65 | 0.798 | 0.781 |

All outputs are finite for every variant. With FP8, mean pLDDT drops by 1.7 to
6.3 points. A scan over many inputs was not run: bundles are exact-request
(one 15 to 25 min build per input).

### Why there is no gain

- Only the diffusion token blocks gain from FP8 (1.13x to 1.15x per plan, fp16
  projections at 76 rows). They are about 64% of the GPU time, so the best case
  is 8% to 9% of GPU time.
- The pair-stack GEMMs are small (K = 128 at 76 tokens) and do not gain from
  FP8: the Pairformer plans are 4% to 13% slower and slightly larger (59.3 vs
  57.3 MiB).
- The process wall time is dominated by bundle load and setup, and its spread
  between runs is larger than any FP8 effect.
- The diffusion rollout is sensitive: small changes in the token blocks move the
  sample to a different nearby structure, so FP8 error shows up as large lDDT
  changes, and two FP8 builds of the same recipe can disagree.

### NVFP4 (not tried)

No part is bound by weight reads (76-row GEMMs, small K in the pair stack), so
NVFP4 has no speed to gain, and FP8 already fails the quality bar.

### What was not tried

| Item | Note |
|---|---|
| FP8 attention cores, triangle contractions, atom transformers | not measured |
| Percentile or entropy calibration, per-channel weight scales | not measured |
| Longer sequences (up to 128 tokens), other proteins, several diffusion samples | not measured; one 15 to 25 min build per request |
| H100 | not measured |

### When to revisit

- Longer sequences or batched diffusion samples, where the token and pair GEMMs
  have more rows and FP8 GEMMs can pay for the Quantize cost; the family builds
  batch-1 exact-request bundles today.
- A recipe that keeps the diffusion rollout stable (lDDT vs fp16 near the fp16
  rebuild floor across several FP8 builds).
