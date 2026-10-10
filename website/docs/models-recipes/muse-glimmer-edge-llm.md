---
title: Muse-Glimmer with Edge-LLM
---

# Muse-Glimmer with Edge-LLM

The `muse_glimmer` family owns the optional Edge 0.11 text route for
`RadixArk/Muse-Glimmer-NVFP4`. Its original mixed NVFP4/MXFP8 weights are
preserved. Selection, ONNX commands, paired execution, and validation remain
inside the family.

Use a native SM120 Edge SDK with ONNX tools and complete kernels. A compatible
Python wheel can provide the exporter; the native SDK remains required.
Build on the target device, not by cross-compiling.

The owning `trtmc muse_glimmer build` command accepts
`--execution-variant autoregressive` or `--execution-variant dflash`.
The latter requires `--companion` pointing to the matched official assistant
or DFlash2 checkpoint. The draft architecture selects the version, using the
same public block16 workflow. No shared CLI extension is required.

The text-only NVFP4 route does not establish FP16 image/video support.
The original FP16 checkpoint and FP16 pairs remain blocked by available
single-device capacity. The existing final-answer quality checks are unchanged.

See the [family-owned recipe](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/families/muse_glimmer/edge_llm/README.md)
for commands, dependency requirements, precision/profile boundaries and
qualification limitations.
