# FoundationPose

This family builds the FoundationPose refiner and scorer from the NGC
`nvidia/isaac/foundationpose` 1.0.1 ONNX weights as two TensorRT engines for
the `pose_hypothesis_refinement` task, in fp32 (family default) or fp16 (the
manifests use fp16). Both networks are a convolutional encoder on 160x160
RGB-XYZ crops followed by a small transformer. The refiner engine takes up to 42
pose hypotheses per call and the scorer up to 252; the native runtime refines
the hypotheses and then scores them.

## Quantization

`quantization=fp8` (fp16 only) self-quantizes the NGC 1.0.1 ONNX weights at
build time; FP8 weights are never stored. In each engine the 9 3x3 convolutions
of the paired-feature trunk (`encodeAB` / `encoderAB`) run as FP8 convolutions
with one weight scale per tensor and static per-tensor activation scales from
`fp8_activation_scales.json`. The image encoders, the transformer heads and the
output projections stay fp16. On B200 and B300 with TensorRT 11.4, FP8 is 1.13x
to 1.23x faster than fp16 at the registration batch sizes (1.04x to 1.09x at
batch 1) and the plans are 37% smaller. Other weights are refused until
`calibrate_fp8_scales.py` is run for them.

### NVFP4

NVFP4 is not used. TensorRT has no FP4 convolution (block-scaled FP4 needs a
2-D GEMM input), and the GEMMs that could take NVFP4 (the transformer linears)
are a small part of the time:

| FP8 engine, batch (B200, TensorRT 11.4, layer profile) | GEMM time | Total |
|---|---|---|
| scorer, 252 | 0.05 ms | 5.27 ms |
| refiner, 42 | 0.08 ms | 1.60 ms |

FP8 on these linears was measured and made the refiner slower (extra Quantize
kernels around 400-token GEMMs), so they stay fp16.
