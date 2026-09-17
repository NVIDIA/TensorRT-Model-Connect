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

## Benchmark scope

The existing family-owned reference protocol retains the original entry ID,
workload, precision, 3 warmups, 10 measurements, 5% margin and top-class oracle.
Inference and complete host scores are timed; argmax, finite scans and JSON
reporting are not. The existing reference policy excludes preprocessing while
the native Task includes family preprocessing; this scope difference is not
claimed resolved and no performance improvement is claimed.
