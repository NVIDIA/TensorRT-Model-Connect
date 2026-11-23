---
title: GR00T N1.7 on Isaac LIBERO-10
---

Use the `lerobot_groot` family command to build the FP16 Isaac policy. It selects
the `libero_10` subdirectory of the [release checkpoint](https://huggingface.co/nvidia/GR00T-N1.7-LIBERO/tree/658540327860f93dd2cb9c1ed3e3b10fb792a0eb/libero_10),
not the sibling policies. Build on the machine where the bundle will run:

```sh
trtmc lerobot_groot build nvidia/GR00T-N1.7-LIBERO \
  --precision fp16 -o isaac-libero10.bundle
```

With the standard installation, the family can build and run the native
TensorRT policy. Optional Edge offload requires **both** a Model Connect runtime
built with the matching Edge 0.12.0 GR00T C++ archive and headers **and** the
matching Edge Python builder. The base 0.12.0 SDK omits that archive and those
headers; installing the Python wheel alone does not enable offload. For a
[Model Connect source build](../getting-started/source-build.md), configure
`-DTRTMC_ENABLE_EDGELLM=ON -DTRTMC_EDGELLM_RELEASE=0.12.0 -DTRTMC_EDGELLM_POLICY_SDK=ON`.
This provisions the pinned policy components or checks that the external SDK
you selected already contains them. Use
`--edge-builder-python /absolute/path/to/python` for an isolated builder.
If Edge preparation fails, the builder warns and retries the same FP16 policy
through native TensorRT.

The C++ task is `ImagesStateTextToActionChunk`: pass two named 256 × 256 RGB
cameras (`image` and `wrist_image`), an eight-value robot state, and an
instruction. The result is 16 actions with seven values each.

**Validation boundary:** On one recorded observation with fixed initial noise,
native passed the independent FP32 comparison (max absolute action error
0.0159; limit 0.03). Edge completed inference but missed the same gate (0.2722).
The released Edge runner returned the same actions on the same engines. Edge
remains executable, not accuracy-qualified. This is not a closed-loop task
success result or a claim about other embodiments.
