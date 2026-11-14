# timm ViT Task SDK

This family implements `IModel` and `IImageToClassScores`, with its own
`bind` declaration. Build/support/manifests use `image_to_class_scores`;
rebuild old `classification` bundles. There is no legacy alias or fallback.

The family preserves its graph, weights and image preprocessing. It returns
all host-owned logits in checkpoint class order, not probabilities or top-k.
Only explicit checkpoint `label_names` and `vocabulary_id` are published;
unknown metadata stays empty. No optional runtime Config fields are declared.

## Validate

```sh
cmake --build build --target test_timm_vit_task_contract test_timm_vit_image_preprocess
ctest --test-dir build --output-on-failure -R '^timm_vit_(task_contract|image_preprocess)$'
python -m pytest families/timm_vit/tests/test_e2e.py --e2e-testcase timm-vit-base-p16-224-augreg-in21k-ft-in1k -q
```

Use the existing E2E environment: native CLI, runtime libraries, cached checkpoint,
and `TRTMC_NATIVE_BUILD_DIR` pointing to the build directory. The Task contract
target also builds the C11 and C++17 SDK consumers. The original CLI/reference
oracle is retained; the two SDK consumers use the same decoded RGB pixels and
compare every score after releasing the model. Different JPEG decoders are not
claimed bitwise equivalent. The existing TP4 implementation and manifest remain; the added direct SDK E2E checks select only TP1.

## Quantization

`quantization=fp8` (fp16 precision, tensor parallel size 1) quantizes
`timm/vit_base_patch16_224.augreg_in21k_ft_in1k` at build time: `qkv`,
`attn.proj`, `mlp.fc1` and `mlp.fc2` of every block run as FP8 GEMMs
(per-tensor weight scale amax / 448, static activation scales from
`fp8_activation_scales.json`, calibrated with ModelOpt `FP8_DEFAULT_CFG` on 500
ImageNet images). Patch embedding, norms, attention matmuls and the head stay
fp16. At batch 1 FP8 is 1.03x to 1.11x faster than fp16 (B200 and B300,
TensorRT 11.1 and 11.4), the plan is 49% smaller and top-1 stays at the fp16
level. Another model config raises and names the calibration tool.

### NVFP4

NVFP4 was measured in a separate experiment and is not shipped (no NVFP4 code
is in the family). The experiment built engines with `mlp.fc1` and `mlp.fc2`
in NVFP4 (self-quantized) and `qkv` and `attn.proj` in FP8. B200, TensorRT 11.1
and 11.4; quality on the 1500 held-out ImageNet images.

| Build | Plan MiB | Batch 1 ms | Batch 32 ms | Top-1 | KL to PyTorch (x FP8) |
|---|---|---|---|---|---|
| fp16 | 166.7 | 0.419 | 2.13 | 85.33% | 0.000006 |
| FP8 (shipped) | 85.8 | 0.397 | 1.78 | 85.47% | 0.0032 (1.0x) |
| NVFP4 `fc1` + `fc2`, rest FP8 | 62.5 | 0.454 | 2.18 | 84.53% | 0.0199 (6.2x) |

Non-finite outputs were not recorded for the NVFP4 engine.

Reasons:

- Speed: NVFP4 is slower than FP8 and than fp16 at batch 1, and gives no gain
  at batch 32. ViT-B at batch 1 is launch bound, not weight-bandwidth bound:
  the weights are read in about 0.02 ms of a 0.4 ms call.
- Quality: KL is 6.2x the FP8 KL and top-1 drops by 0.8 point against fp16.

## Benchmark scope

The existing family-owned reference protocol retains the original entry ID,
workload, precision, 3 warmups, 10 measurements, 5% margin and top-class oracle.
Inference and complete host scores are timed; argmax, finite scans and JSON
reporting are not. The existing reference policy excludes preprocessing while
the native Task includes family preprocessing; this scope difference is not
claimed resolved and no performance improvement is claimed.
