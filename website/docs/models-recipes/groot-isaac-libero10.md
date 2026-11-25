---
title: GR00T N1.7 on Isaac LIBERO-10
---

Use the `lerobot_groot` family command to build the FP16 Isaac policy. It selects
the `libero_10` subdirectory of the [release checkpoint](https://huggingface.co/nvidia/GR00T-N1.7-LIBERO/tree/658540327860f93dd2cb9c1ed3e3b10fb792a0eb/libero_10),
not the sibling policies. Build on the machine where the bundle will run:

```sh
trtmc lerobot_groot build nvidia/GR00T-N1.7-LIBERO \
  --revision 658540327860f93dd2cb9c1ed3e3b10fb792a0eb \
  --precision fp16 -o isaac-libero10.bundle
```

This command builds the native FP16 policy even with a complete Edge SDK
installed. To opt in to the experimental Edge route, add
`--edge-builder-python /absolute/path/to/python`. The Edge route requires **both** a Model Connect runtime
built with the matching Edge 0.12.0 GR00T C++ archive and headers **and** the
matching Edge Python builder. The base 0.12.0 SDK omits that archive and those
headers; installing the Python wheel alone does not enable offload. For a
[Model Connect source build](../getting-started/source-build.md), configure
`-DTRTMC_ENABLE_EDGELLM=ON -DTRTMC_EDGELLM_RELEASE=0.12.0 -DTRTMC_EDGELLM_POLICY_SDK=ON`.
This provisions the pinned policy components or checks that the external SDK
you selected already contains them.
If Edge preparation fails, the builder warns and retries the same FP16 policy
through native TensorRT.

The C++ task is `ImagesStateTextToActionChunk`: pass two named 256 × 256 RGB
cameras (`image` and `wrist_image`), an eight-value robot state, and an
instruction. The result is 16 actions with seven values each.

**Validation boundary:** The native FP16 policy passed the independent FP32 action
comparison on A30 for episode 1272 frame 0 (mean absolute error 0.002540; limit 0.0026).
Before the conditioner precision correction, the same observation missed on A30 (0.003133);
separate GB300 engine builds passed (0.001853) and missed (0.002731). The A30 result
does not establish accuracy on GB300 or across builds. Edge completed inference but missed
the maximum error limit (0.2722 vs 0.03); the released runner returned the same actions
on the same engines. Neither route has been qualified here for closed-loop task success
or other embodiments.
