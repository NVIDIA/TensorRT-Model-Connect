# timm Swin Task SDK

This family implements `IModel` and `IImageToClassScores`, with its own
`bind` declaration. Build/support/manifests use `image_to_class_scores`;
rebuild old `classification` bundles. There is no legacy alias or fallback.

The family preserves its graph, weights and image preprocessing. It returns
all host-owned logits in checkpoint class order, not probabilities or top-k.
Only explicit checkpoint `label_names` and `vocabulary_id` are published;
unknown metadata stays empty. No optional runtime Config fields are declared.

## Validate

```sh
cmake --build build --target test_timm_swin_task_contract test_timm_swin_image_preprocess
ctest --test-dir build --output-on-failure -R '^timm_swin_(task_contract|image_preprocess)$'
python -m pytest families/timm_swin/tests/test_e2e.py --e2e-testcase swin-tiny-patch4-window7-224-ms-in1k -q
```

Use the existing E2E environment: native CLI, runtime libraries, cached checkpoint,
and `TRTMC_NATIVE_BUILD_DIR` pointing to the build directory. The Task contract
target also builds the C11 and C++17 SDK consumers. The original CLI/reference
oracle is retained; the two SDK consumers use the same decoded RGB pixels and
compare every score after releasing the model. Different JPEG decoders are not
claimed bitwise equivalent.

## FP8 quantization

`quantization=fp8` self-quantizes the original checkpoint at build time (fp16 precision
only). Weight scale is `amax / 448` per projection, activation scales are static and were
calibrated once with ModelOpt (`calibrate_fp8_scales.py`), stored in
`fp8_activation_scales.json` per model config (swin tiny, small and base, patch 4,
window 7, 224). Quantized: `attn.qkv`, `mlp.fc1`, `mlp.fc2` of every block and the
patch-merging reduction. Left fp16: patch embedding, norms, window attention, `attn.proj`
(TensorRT 11.4 cannot compile an FP8 Quantize after the fused attention) and the head.
An unknown model config or a projection without a scale raises and names the calibration
tool. No FP8 manifest or premerge case exists.

### NVFP4

NVFP4 was measured on top of this FP8 recipe for Swin tiny, small and base and
is not shipped. The study built real TensorRT engines (no NVFP4 code is in the
family): NVFP4 weights with an FP8 scale per block of 16, FP4 activations
quantized at run time with the FP8 activation scales, everything else as in the
FP8 build. B200 and B300, TensorRT 11.4, batch 1 (the family build) and batch
8 and 32 (a batched copy of the graph). Quality: B200, batch 1, 1500 held-out
ImageNet images; KL is against the fp16 engine.

| Variant | Speed vs FP8 (all GPUs, sizes, batches) | Plan vs FP8 | KL, x FP8 (tiny / small / base) | Top-1 (tiny / small / base) |
|---|---|---|---|---|
| fp16 | FP8 is 1.02x to 1.07x faster | FP8 is 40% to 42% smaller | - | 82.47% / 83.40% / 84.13% |
| FP8 (shipped) | 1.00x | 0% | 1.0x | 82.33% / 83.87% / 84.20% |
| NVFP4 `fc1` + `fc2`, rest FP8 | 0.87x to 0.92x | -19% to -22% | 9.5x / 7.2x / 8.1x | 81.53% / 83.27% / 83.73% |
| NVFP4 `qkv`, `fc1`, `fc2`, merging | 0.86x to 0.92x | -28% to -31% | 14.3x / 12.0x / 11.6x | 80.80% / 83.00% / 84.13% |
| NVFP4 `fc1` only, rest FP8 | 0.93x to 0.97x | -9% to -11% | 5.7x / 5.2x / 5.2x | 81.40% / 83.40% / 83.73% |

Every NVFP4 variant is slower than FP8 in all 54 measured cells and is 0.88x to
1.01x of fp16. No variant gave a non-finite output (144 engines, real and
stress inputs).

Reasons:

- Speed: the FP4 GEMMs save little. At batch 1 on B300 the 75 quantized GEMMs
  take 0.445 ms in FP8 and 0.366 ms in FP4, because most are small (49 tokens
  per window). NVFP4 adds a run-time activation quantize kernel per FP4
  projection (0.06 to 0.18 ms at batch 1), and at batch 1 the GELU no longer
  fuses into the `fc1` GEMM (0.08 ms).
- Quality: KL is 5.2x to 14.3x the FP8 KL; top-1 agreement with PyTorch drops
  from 98.7% to 98.9% (FP8) to 94.5% to 97.7%.

## Benchmark scope

The existing family-owned reference protocol retains the original entry ID,
workload, precision, 3 warmups, 10 measurements, 5% margin and top-class oracle.
Inference and complete host scores are timed; argmax, finite scans and JSON
reporting are not. The existing reference policy excludes preprocessing while
the native Task includes family preprocessing; this scope difference is not
claimed resolved and no performance improvement is claimed.
