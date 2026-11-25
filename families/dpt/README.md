# DPT-Hybrid (MiDaS)

This family builds the Transformers DPT-Hybrid depth model
(`DPTForDepthEstimation` with `is_hybrid: true`, as in
[`Intel/dpt-hybrid-midas`](https://huggingface.co/Intel/dpt-hybrid-midas)) as one
TensorRT engine for the `monocular_depth` task. A BiT ResNet stem and three
bottleneck stages feed a ViT-B/16 encoder; the DPT neck fuses BiT stages 1-2
with two ViT layers, and the head predicts relative inverse depth (MiDaS units,
larger is nearer) for every pixel of the input image.

```sh
trtmc build Intel/dpt-hybrid-midas -o dpt-hybrid-midas.bundle
trtmc depth dpt-hybrid-midas.bundle --image photo.jpg
```

The default precision is fp32, the checkpoint's own; `--precision fp16` builds
an fp16 engine. Batch is 1. The checkpoint ships only `pytorch_model.bin`, so the
build needs torch 2.10 or later (earlier versions can run code from a crafted file
even with `weights_only=True`); it is read with `weights_only=True`.

## Image pipeline

The family reproduces the checkpoint's `DPTImageProcessor` and
`post_process_depth_estimation` in C++:

- The input (RGB float32 in `[0, 1]`) is taken back to its 8-bit samples and
  resized to 384x384 with Pillow 13's bilinear filter (fixed-point, two passes,
  the vertical pass first when the height shrinks more than twice as much as the
  width), then rescaled by 1/255 and normalized with mean and std 0.5. On the
  test image, in landscape and portrait orientation, the engine input is
  bit-identical to the processor's `pixel_values`.
- The 384x384 depth map is resized back to the input size bicubically
  (`align_corners=False`, A = -0.75), as `post_process_depth_estimation` does.

Both are adapted from Pillow and PyTorch; see `NOTICE`.

## Accuracy

`tests/test_e2e.py` compares the CLI's depth map with the official pipeline in
float32 on the CPU (`DPTImageProcessor`, `DPTForDepthEstimation`,
`post_process_depth_estimation`) on the repository's test photograph in
landscape and in portrait orientation. Errors are relative to the largest
reference depth. 1x NVIDIA B300, CUDA 13, TensorRT 11.4, six builds of the fp32
engine and four of the fp16 one:

| Engine | Largest error | Mean error | Correlation | Engine call, batch 1 |
|---|---|---|---|---|
| fp32 | 2.8e-6 - 7.8e-6 | 2.0e-7 - 2.1e-7 | > 0.999999999999 | 8.7 ms |
| fp16 | 6.2e-3 - 3.0e-2 | 2.5e-4 - 5.7e-4 | >= 0.9999977 | 1.7 ms |

The fp32 builds are identical; the fp16 builds pick different kernels, and their
largest error on the landscape input moves between 1.3e-2 and 3.0e-2. For scale, the official model's
own float32 run on the GPU differs from its CPU run by up to 5.5e-6, and its
float16 run by up to 6.2e-3 (mean 2.8e-4 - 5.3e-4, compared on the 384x384
maps). On those maps the fp16 engine matches the official float16 run up to the
99th percentile (1.7e-3 against 1.5e-3); its largest errors, on about 0.1 % of
the pixels, are two to five times larger. They build up across the ViT encoder
and the neck: running one stage or the head in fp32 removes at most half of
them, so the engine stays fp16 throughout.

The bounds were also checked against known-wrong outputs: twelve plausible
porting mistakes injected into the float32 reference (resize filter,
normalization, rescale factor, channel order, BiT stem padding and
weight-standardization epsilon, ViT taps, GELU form, LayerNorm epsilon, fusion
and output interpolation), each on both inputs. The fp32 bounds (largest error
1e-4, mean 1e-5, correlation 0.999999) reject all 24 outputs, none of which
errs by less than 6.3e-4 (largest) or 4.9e-5 (mean). The fp16 bounds (0.05,
0.002, 0.99999) leave the fp16 engine at least 1.7 times its measured errors
and reject 16; the four mistakes they pass (rescale 1/256,
weight-standardization epsilon, tanh GELU, LayerNorm epsilon) err less than
fp16 rounding does, and the fp32 case, which runs in premerge, rejects them.

## Not supported

- DPT checkpoints other than DPT-Hybrid: the ViT-only `dpt-large`, and BEiT,
  DINOv2 or Swin backbones. The build refuses them by name.
- `DPTForSemanticSegmentation`, `add_projection`, batch norm in the fusion units,
  other image-processor settings (aspect-ratio-preserving resize, other
  resampling filters, padding).
- Other input resolutions: images of any size are resized to the checkpoint's
  384x384, and the position embeddings are not interpolated.
- Batch sizes above 1, quantization, tensor parallelism.

## Tests

```sh
cmake --build build --target test_dpt_image_ops test_dpt_pipeline
ctest --test-dir build --output-on-failure -R '^dpt_'
PYTHONPATH=core/builder:apps/benchmark:. python3 -m pytest families/dpt/tests --e2e-model dpt -q
```

`test_dpt_image_ops` checks the resize against Pillow byte for byte, the whole
preprocessing against `DPTImageProcessor` bit for bit, and the bicubic resize
against torch, with goldens from `tests/cpp/generate_image_ops_goldens.py`.
`test_dpt_pipeline` checks, with a stand-in engine, that the runtime refuses a
malformed `runtime.json`, an engine whose inputs or outputs differ from it, and
a non-finite depth map. `tests/test_support.py` covers identity, the default
task, the resolved geometry, every refused configuration value and the E2E case
selection.
