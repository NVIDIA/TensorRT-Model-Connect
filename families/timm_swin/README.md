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

## Benchmark scope

The existing family-owned reference protocol retains the original entry ID,
workload, precision, 3 warmups, 10 measurements, 5% margin and top-class oracle.
Inference and complete host scores are timed; argmax, finite scans and JSON
reporting are not. The existing reference policy excludes preprocessing while
the native Task includes family preprocessing; this scope difference is not
claimed resolved and no performance improvement is claimed.
