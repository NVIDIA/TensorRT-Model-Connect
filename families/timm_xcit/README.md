# timm XCiT Task SDK

This family implements `IModel` and `IImageToClassScores`, with its own
`bind` declaration. Build/support/manifests use `image_to_class_scores`;
rebuild old `classification` bundles. There is no legacy alias or fallback.

The family preserves its graph, weights and image preprocessing. It returns
all host-owned logits in checkpoint class order, not probabilities or top-k.
Only explicit checkpoint `label_names` and `vocabulary_id` are published;
unknown metadata stays empty. No optional runtime Config fields are declared.

## Validate

```sh
cmake --build build --target test_timm_xcit_task_contract test_timm_xcit_image_preprocess
ctest --test-dir build --output-on-failure -R '^timm_xcit_(task_contract|image_preprocess)$'
python -m pytest families/timm_xcit/tests/test_e2e.py --e2e-testcase xcit-tiny-12-p16-224-fb-in1k -q
```

Use the existing E2E environment: native CLI, runtime libraries, cached checkpoint,
and `TRTMC_NATIVE_BUILD_DIR` pointing to the build directory. The Task contract
target also builds the C11 and C++17 SDK consumers. The original CLI/reference
oracle is retained; the two SDK consumers use the same decoded RGB pixels and
compare every score after releasing the model. Different JPEG decoders are not
claimed bitwise equivalent.

## Benchmark scope

The existing family-owned reference protocol retains the original entry ID,
workload, precision, 3 warmups, 10 measurements, 5% margin and top-class oracle.
Inference and complete host scores are timed; argmax, finite scans and JSON
reporting are not. The existing reference policy excludes preprocessing while
the native Task includes family preprocessing; this scope difference is not
claimed resolved and no performance improvement is claimed.

## FP8 quantization study (not shipped)

FP8 self-quantization of the attention and MLP linears was built and measured on
TensorRT 11.4 (and 11.1) for the two manifest checkpoints and five larger XCiT
configs. It is not shipped: at batch 1 (the only batch the family builds) the
FP8 engine is 0.94x to 0.95x of the fp16 engine for xcit-nano-12-p16 and
xcit-tiny-12-p16 at the family's builder optimization level 1, and 1.01x to
1.03x at level 3. The plan is 19% to 31% smaller. This section records the
recipe, the method and the numbers, so that the study does not have to be
repeated. The FP8 code was not merged; this family has no FP8 build.

### Summary and decision

Manifest checkpoints, batch 1, TensorRT 11.4 (x is fp16 time over FP8 time):

| Item | fp16 (manifest) | FP8 (fp16 engine) | FP8 vs fp16 |
|---|---|---|---|
| nano latency, level 1, B200 / B300 | 0.580 / 0.534 ms | 0.613 / 0.570 ms | 0.95x / 0.94x |
| tiny latency, level 1, B200 / B300 | 0.614 / 0.575 ms | 0.647 / 0.611 ms | 0.95x / 0.94x |
| nano latency, level 3, B200 / B300 | 0.498 / 0.453 ms | 0.493 / 0.447 ms | 1.01x / 1.01x |
| tiny latency, level 3, B200 / B300 | 0.557 / 0.514 ms | 0.539 / 0.498 ms | 1.03x / 1.03x |
| Plan size, level 1, B200, nano / tiny | 7.4 / 14.5 MiB | 5.6 / 9.9 MiB | -24% / -31% |
| Top-1 agreement with PyTorch fp32, nano / tiny, B200 | 99.8% / 99.6% | 96.4% / 98.4% | -3.4 / -1.2 points |
| Minimum logit cosine vs PyTorch fp32, nano / tiny, B200 | 0.9997 / 0.9991 | 0.985 / 0.974 | lower |
| Non-finite outputs | 0 | 0 | none |

Decision: do not ship FP8 for timm XCiT.

- Speed: at batch 1 the engine is bound by kernel launches, not by math. The
  token count is 196 and the linears are 128 to 768 wide, so each fp16 or FP8
  GEMM takes about 10 microseconds in the layer profiler, at either precision.
  At level 1 the 12 extra Quantize layers (264 instead of 252 layers for nano)
  are not fused and make FP8 slower. At level 3 they are fused (252 layers in
  both plans) and FP8 is 1% to 3% faster than the level-3 fp16 engine, which is
  itself 1.10x to 1.18x faster than level 1 without any quantization.
- Larger configs (not in the manifests) gain at most 1.07x at batch 1 and 1.15x
  at batch 32 (scratch build; the family builds batch 1 only).
- Quality: top-1 stays within noise of the fp16 engine, but FP8 changes the top
  class of 2% to 5% of the images on nano and tiny (near ties), and the minimum
  logit cosine against PyTorch drops from above 0.999 to 0.96 to 0.99.
- The only clear gain is plan size, which is small for the manifest checkpoints
  (2 to 5 MiB).

### Recipe that was measured

No FP8 or NVFP4 checkpoint of any timm XCiT classifier exists on the Hub, and no
official quantization config exists for one. The recipe follows the guidelines
for small vision encoders (the `timm_vit` flow): all attention and MLP linears,
nothing else. Each quantized linear is built as
`input -> Quantize -> Dequantize(fp16) -> MatMul`, with the weight
`FP8 constant -> Dequantize(fp16)`, then the bias add.

| Choice | Decision |
|---|---|
| Quantized (per block) | `attn.qkv` (q, k and v share one input scale, each keeps its own weight scale), `attn.proj`, `mlp.fc1`, `mlp.fc2`; per class-attention block the key and value (`attn.kv`, shared input) |
| fp16 | stem and depthwise convolutions, position encoding, layer norms, the channel-attention matmuls and softmax, class-attention query, output and MLP projections (one-token GEMVs), classifier head |
| Weight scale | per tensor, amax / 448, at build time; FP8 weights are never stored |
| Activation scale | static, per tensor; ModelOpt 0.46.1 `FP8_DEFAULT_CFG` (max calibration) on the timm PyTorch model, 500 ImageNet validation images |
| Outlier rule | a scale above 20x the median of its kind stays fp16; measured: no outlier |
| Dequantize type | fp16; no non-finite value, fp32 Dequantize was not needed |

### Method

- GPUs: B200 and B300, one GPU per job; TensorRT 11.4.0.106 (target) and
  11.1.0.106 (B200 sanity check). torch 2.14, timm 1.0.28, ModelOpt 0.46.1.
- Builds: the family's `build()` with `quantization=fp8` on an fp16 build. The
  level-3 rows replace only the builder optimization level (1 on main, 3 in the
  pending level change); everything else is the family code.
- Data: ImageNet validation images from `clip-benchmark/wds_imagenet1k`.
  Calibration: 500 images (every 25th image, the first 500 of a 2000-image set).
  Study 1 (all configs): the other 1500 images of that set. Study 2 (manifest
  checkpoints, levels 1 and 3): 500 images of a separate 4000-image set (every
  12th image, shuffled with seed 0, indices 1100 to 1599; 6 of them are also
  calibration images).
- Quality: top-1 against the labels, top-1 agreement and logit cosine against
  the timm PyTorch fp32 model, batch 1. The fp16 engine is the floor.
- Non-finite scan: the evaluation images and 17 to 26 stress inputs (noise with
  standard deviation 1, 3 and 10, constants), all variants, B200 and B300.
- Speed: CUDA graph replay at batch 1. Study 2: all four variants of a model
  built in one job, median of 21 interleaved rounds of 100 replays (spread at
  most 2.5%). Study 1: variants built one after another in one job, median of
  5 runs of 200 replays, level 1 only.

### Results

Study 2, manifest checkpoints, batch 1 (x is fp16 time over FP8 time at the
same level; plan MiB fp16 to FP8):

| Model | GPU | TensorRT | fp16 L1 ms | FP8 L1 ms | x L1 | fp16 L3 ms | FP8 L3 ms | x L3 | plan MiB L1 | plan MiB L3 |
|---|---|---|---|---|---|---|---|---|---|---|
| nano | B200 | 11.4 | 0.580 | 0.613 | 0.95 | 0.498 | 0.493 | 1.01 | 7.4 to 5.6 | 9.6 to 7.7 |
| nano | B300 | 11.4 | 0.534 | 0.570 | 0.94 | 0.453 | 0.447 | 1.01 | 6.9 to 5.2 | 10.7 to 7.9 |
| tiny | B200 | 11.4 | 0.614 | 0.647 | 0.95 | 0.557 | 0.539 | 1.03 | 14.5 to 9.9 | 16.7 to 12.1 |
| tiny | B300 | 11.4 | 0.575 | 0.611 | 0.94 | 0.514 | 0.498 | 1.03 | 14.1 to 9.6 | 17.8 to 12.2 |
| nano | B200 | 11.1 | 0.555 | 0.602 | 0.92 | 0.485 | 0.484 | 1.00 | 7.4 to 5.6 | 9.5 to 7.7 |
| tiny | B200 | 11.1 | 0.596 | 0.648 | 0.92 | 0.538 | 0.535 | 1.01 | 14.5 to 9.9 | 16.6 to 11.9 |

Study 2 quality, 500 images, TensorRT 11.4, B200 (B300 in brackets where it
differs; PyTorch fp32 top-1: nano 72.0%, tiny 78.8%):

| Model | Engine | Top-1 | Agreement with PyTorch top-1 | Mean / min logit cosine | Non-finite rows |
|---|---|---|---|---|---|
| nano | fp16 L1 | 72.2% | 99.8% | 0.99995 / 0.9997 | 0 |
| nano | FP8 L1 | 73.0% (72.4%) | 96.4% (96.2%) | 0.9964 / 0.985 (0.981) | 0 |
| nano | fp16 L3 | 72.4% | 99.6% | 0.99995 / 0.9997 | 0 |
| nano | FP8 L3 | 72.4% | 97.0% (96.4%) | 0.9964 / 0.988 (0.987) | 0 |
| tiny | fp16 L1 | 79.2% | 99.6% | 0.99996 / 0.9991 | 0 |
| tiny | FP8 L1 | 79.4% (78.4%) | 98.4% (98.2%) | 0.9966 / 0.974 (0.969) | 0 |
| tiny | fp16 L3 | 79.0% | 99.8% | 0.99995 / 0.9975 | 0 |
| tiny | FP8 L3 | 78.8% | 98.4% | 0.9966 / 0.965 | 0 |

TensorRT 11.1 gives the same picture (FP8 agreement 96.0% to 98.8%, no
non-finite output).

Study 1, all configs, level 1, batch 1, TensorRT 11.4, 1500 held-out images
(x is fp16 time over FP8 time; top-1 on B200):

| Config | x B200 | x B300 | plan MiB fp16 to FP8 (B200) | Top-1 PyTorch / fp16 / FP8 | FP8 agreement with PyTorch | FP8 mean logit cosine |
|---|---|---|---|---|---|---|
| nano_12_p16 | 0.93 | 0.95 | 7.4 to 5.6 | 67.87 / 68.00 / 67.67 | 95.0% | 0.9962 |
| tiny_12_p16 | 0.95 | 0.94 | 14.5 to 9.9 | 74.87 / 74.93 / 75.40 | 96.5% | 0.9965 |
| small_12_p8 | 1.02 | 1.04 | 52.3 to 32.2 | 81.40 / 81.53 / 81.60 | 98.9% | 0.9980 |
| small_24_p16 | 1.07 | 1.05 | 93.0 to 53.6 | 81.20 / 81.20 / 81.27 | 98.9% | 0.9984 |
| medium_24_p16 | 1.01 | 0.97 | 163.0 to 91.7 | 80.33 / 80.40 / 80.20 | 99.1% | 0.9986 |
| large_24_p16 | 1.00 | 1.01 | 362.8 to 200.0 | 82.47 / 82.47 / 82.40 | 99.3% | 0.9990 |
| large_24_p8 | 1.05 | 1.04 | 363.5 to 200.7 | 84.40 / 84.40 / 84.47 | 99.7% | 0.9989 |

Batch 8 and 32 (scratch builds, level 1, TensorRT 11.4): FP8 is 1.00x to
1.04x for nano and tiny and 1.01x to 1.15x for the larger configs (best:
large_24_p8 at batch 32, B200, 23.1 ms fp16 and 20.1 ms FP8). Study 1 found no
non-finite output on 1500 images and 26 stress inputs.

### NVFP4 (not tried)

FP8 already gives no speedup at batch 1, the manifest plans are 7 to 15 MiB, so
weight reads are not the limit, and FP8 already moves the top class of 2% to 5%
of the images on the small configs. NVFP4 would lose more accuracy for no
expected speed gain.

### What was not tried

| Item | Note |
|---|---|
| FP8 for the depthwise convolutions, the one-token class-attention GEMVs or the channel-attention matmuls | not measured; small or launch-bound work |
| Layer-norm output Q/DQ, fixed softmax-output scale, percentile calibration | not measured |
| 384 px configs, H100, native runtime with an FP8 bundle | not measured |

### When to revisit

- The family builds batch sizes above 1: FP8 is 1.08x to 1.15x at batch 32 for
  the larger configs.
- Larger configs (small_24 and up) become manifest checkpoints and the family
  builds at level 3, if FP8 then gains clearly more than the 1% to 3% measured
  for nano and tiny.
- A TensorRT release that cuts the per-kernel overhead of small GEMMs at batch 1.
