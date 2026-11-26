# LeRobot ACT

This family builds the LeRobot Action Chunking Transformer policy
(`lerobot/act_aloha_sim_transfer_cube_human` in the manifests: ResNet-18 image
backbone, four encoder layers, one decoder layer, 100 x 14 action chunk per
observation, 52M parameters) as one TensorRT engine: batch 1, fp32 (default) or
fp16. The family has no quantized build.

## FP8 quantization study (not shipped)

FP8 self-quantization on top of the fp16 graph was built and measured on
TensorRT 11.4 (and 11.1). It is not shipped: FP8 is only 1.02x to 1.03x faster
than fp16 on TensorRT 11.4, and its action error is 17x to 34x the fp16 error,
so it fails the fp16 action accuracy check of the E2E case. This section records
the recipe and the numbers, so that the study does not have to be repeated. The
FP8 code was not merged; this family has no FP8 build.

### Recipe that was measured

No FP8 or NVFP4 checkpoint of any LeRobot ACT policy exists on the Hub, and no
official quantization config exists for it. Each quantized linear is built as
`input -> Quantize(FP8) -> Dequantize(fp16) -> MatMul`, with the weight
`FP8 constant -> Dequantize(fp16)`.

| Choice | Decision |
|---|---|
| Quantized (30 projections) | q, k, v, out and both feed-forward linears of the four encoder layers, of the decoder cross-attention and of the decoder feed-forward |
| Not quantized | ResNet-18 convolutions (FP8 is slower there, see Results), decoder self-attention, latent and state projections, action head, norms, attention cores |
| Weight scale | per tensor, amax / 448, at build time |
| Activation scale | static, per tensor, max calibration over 200 frames of episodes 0 to 9 of `lerobot/aloha_sim_transfer_cube_human` |
| Dequantize type | fp16; no non-finite output |

### Results

Engine only, batch 1 (the family accepts batch 1 only), ms per action chunk,
median of interleaved rounds. Quality on 501 recorded frames against the
LeRobot float32 policy; "task" is the mean absolute error against the recorded
demonstration actions.

| Item | fp16 | FP8 | FP8 vs fp16 |
|---|---|---|---|
| B200, TensorRT 11.4 | 0.447 ms | 0.433 to 0.439 ms | 1.02x to 1.03x |
| B300, TensorRT 11.4 | 0.419 to 0.423 ms | 0.411 to 0.412 ms | 1.02x to 1.03x |
| B200, TensorRT 11.1 | 0.461 to 0.474 ms | 0.430 to 0.434 ms | 1.06x to 1.07x |
| B300, TensorRT 11.1 | 0.427 to 0.432 ms | 0.410 ms | 1.05x |
| Plan size | 70.5 MB | 49.0 MB | -30% |
| Action mean abs error | 6.6e-05 to 1.36e-04 | 2.3e-03 | 17x to 34x |
| Action max abs error | 2.9e-03 to 4.1e-03 | 0.10 | - |
| Task error (LeRobot float32 0.009827) | 0.009829 to 0.009830 | 0.01004 | - |
| fp16 E2E action check | passes | fails | - |
| Non-finite chunks (501 frames) | 0 | 0 | - |

FP8 on the 21 convolutions alone is slower than fp16 (B200 0.453 ms against
0.435 ms). With the convolutions added to the projections the gain stays at 1%
to 2% and the error grows (mean abs 3.3e-03).

### Why it is not shipped

- 2% to 3% faster than fp16 on TensorRT 11.4, the target version. The fp16
  build alone is 4.7x to 5.1x faster than the fp32 default.
- 17x to 34x the fp16 action error; it fails the fp16 accuracy check of the E2E
  case.

### NVFP4 (not tried)

The model has 52M parameters. An fp16 chunk reads 70 MB of weights in 0.42 ms
(about 170 GB/s, far from the memory bound), and FP8 already gains only 2% to
3%, so 4-bit weights cannot pay here.

### When to revisit

- A policy with a much larger transformer, where the GEMMs and weight reads
  take a larger share of the time.
- A calibration or recipe that keeps the action error near the fp16 level.
