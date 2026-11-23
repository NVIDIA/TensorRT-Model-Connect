---
title: Muse-Glimmer with Edge-LLM
---

# Muse-Glimmer with Edge-LLM

The `muse_glimmer` family owns Edge selection, ONNX commands, paired execution
and native TensorRT fallback. Build on the target device. Both paths preserve
the prepared checkpoint's NVFP4/MXFP8 recipe and the requested draft.

Edge 0.11 is the default and its text offload requires a compatible SM120 SDK.
To opt in to 0.12, configure Model Connect with `TRTMC_EDGELLM_RELEASE=0.12.0`,
Edge support and ONNX tools enabled. An installed Model Connect runner uses its own SDK. For a standalone
build, point `CMAKE_PREFIX_PATH` at the matched
native SDK. The family option `--edge-builder-python /absolute/path/to/python`
selects a matching wheel or source-installed exporter; it does not replace the
C++ SDK. The 0.12 path uses the installed SDK's GPU architectures.

For ordinary generation, leave `--execution-variant` at its default.
For a block-16 chain, use `--execution-variant dflash --companion /models/matched-draft`.
For a greedy DFlash2 tree, use `--execution-variant dflash_tree` with the matched
`incoai/Muse-Glimmer-30B-DFlash2` companion. The tree has up to four children
per node and 32 verification nodes including the anchor. Model Connect does
not substitute the chain or ordinary decoding for an explicit tree request.

All six documented 0.12 combinations built and ran through both Edge and native
TensorRT: the original FP16 and prepared NVFP4 checkpoints, each in ordinary,
DFlash V1 and DFlash V2 chain modes. **They are not all accuracy-qualified.**

- **Original FP16:** Both routes distinguished opposed image colors and both
  video orders in all three modes. On both routes, each draft matched its own
  target-only engine on all six greedy text and media requests. The
  disabled-thinking scientific check still expects a channel header in the
  generated output that the template puts in the prompt.
  The DFlash2 tree passed all four media checks and the thinking-enabled
  science check on both routes. Native matched target-only IDs on all six
  requests; Edge matched on five, with a comma instead of a newline between
  the two correct video colors. Both tree routes still missed the
  disabled-thinking header.
- **Prepared NVFP4:** Text only. Both native pairs matched their respective
  targets; both released Edge pairs showed a same-engine greedy discrepancy.
  The DFlash2 tree also ran on both backends. Its native output matched its own
  target with thinking off and on; the Edge tree differed from target-only
  execution. Neither tree route passed the disabled-thinking header gate,
  and the Edge route also missed a required term with thinking enabled.
  A result on this checkpoint does not qualify media execution.

See the [family-owned recipe](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/families/muse_glimmer/edge_llm/README.md)
for commands, dependency requirements, precision/profile boundaries and
qualification limitations.
