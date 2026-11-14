# Boltz-2

This family builds `boltz-community/boltz-2` for `structure_prediction` as one
multi-plan TensorRT bundle with one static profile (117 tokens, 928 padded
atoms, MSA depth 8), run by the native runtime (`trtmc predict-structure`).
The trunk (input embedder, MSA module, 64 Pairformer blocks) is bf16-mixed. The
diffusion score model keeps an fp32 state, norms and attention, as upstream
Boltz-2 does; under the bf16 bundle precision the projection matmuls of its
token transformer run in bf16 (`--diffusion-token-precision fp32` builds the
upstream fp32 projections). The family has no quantized build.

## FP8 quantization study (not shipped)

FP8 (W8A8, static per-tensor scales) was built and measured on TensorRT 11.4
(B200, B300) and 11.1 (B200) for the Pairformer pair stack and the diffusion
token transformer. It is not shipped: FP8 makes one diffusion sample 1.16x to
1.17x faster, but it moves the predicted structures, and a bf16 cast of the same
token projections (the default build) is faster and keeps the quality. The FP8
code was not merged.

### Summary and decision

Speed and plan size against the fp32-projection build (TensorRT 11.4,
B200 / B300); lDDT and aligned RMSD against the seeded eager PyTorch reference
on the four stable inputs below:

| Variant | Per extra diffusion sample | Plans (MiB) | lDDT, stable inputs | RMSD (A) |
|---|---|---|---|---|
| fp32 token projections (upstream policy) | 1.00x | 1919 | 1.000 | 0.01 to 0.02 |
| FP8 pair stack + token transformer | 1.16x / 1.17x | 1165 (-39%) | 0.87 to 0.98 | 0.41 to 1.65 |
| FP8 token transformer only | 1.17x / 1.20x | 1162 | 0.90 to 0.98 | - |
| FP8 pair stack only | 0.98x / 1.00x | 1923 | 0.99 to 1.00 | - |
| bf16 token projections (default) | 1.34x / 1.27x | 1407 | 0.988 to 1.000 | 0.02 to 0.34 |

Decision: do not ship FP8 for Boltz-2.

- Pair stack: its GEMMs have K = 128 at no more than 117 tokens; FP8 plans are
  3% slower and add 0.3 to 0.45 s of plan load per run.
- Token transformer: FP8 is 1.30x to 1.33x faster than fp32 per plan, but
  1.19x slower than the bf16 cast (0.492 against 0.413 ms per 6-layer plan,
  B200): at 117 rows the GEMMs are small and Quantize adds work; the gain comes
  from leaving fp32 math, not from 8-bit weights.
- Quality: FP8 moves stable predictions by 0.4 to 1.7 A RMSD where bf16 stays at
  the floor. On BPTI FP8 drops below the premerge lDDT bar (0.49 to 0.51; fp32
  0.69 to 0.83).

### Recipe that was measured

- Quantized: the pair-stack projections of all 64 Pairformer blocks (triangle
  multiplication and attention projections, `transition_z`; 1472 projections)
  and the 336 projections of the 24 diffusion token layers (one outlier,
  `token_transformer.layers.23.transition.b_to_a`, kept unquantized).
- Kept: Pairformer sequence attention and single transition, triangle
  contractions, norms, attention cores, MSA, template, atom encoder and decoder,
  confidence, affinity, embeddings.
- Weights: per-tensor FP8 e4m3, scale amax / 448, at build time. Activations:
  static per-tensor max calibration on the eager reference over 7
  single-sequence proteins, disjoint from the evaluation inputs. Dequantize to
  bf16; no non-finite output.

### Method

- Speed: all variants built and timed in one job per GPU type; per-plan medians
  and interleaved native wall-time medians (1 and 5 diffusion samples).
- Quality: native prediction against the seeded eager reference (same noise).
  Inputs: the packaged 117-token monomer and 6 single-sequence requests;
  "stable" inputs (ubiquitin, GB1, c-Src SH3, insulin A+B) are those where the
  fp32-projection engine reproduces the reference (lDDT 1.000).
- Non-finite scan: 200 held-out fragments (24 to 117 residues), all finite for
  FP8 and bf16.

### NVFP4 (not tried)

The time is not bound by weight reads: the 6-layer token plans are 0.41 to
0.64 ms of small-M GEMMs and elementwise work, the pair-stack weights are tiny,
and FP8 already loses accuracy on the diffusion path.

### When to revisit

- A larger token profile: at 256 tokens an FP8 pair transition is 5% to 7%
  faster than bf16 (synthetic probe), outside the current 117-token envelope.
