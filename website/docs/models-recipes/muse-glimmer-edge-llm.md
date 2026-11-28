---
title: Muse-Glimmer with Edge-LLM
---

# Muse-Glimmer with Edge-LLM

The `muse_glimmer` family owns Edge selection, ONNX commands, paired execution
and native TensorRT fallback. Build on the target device. Both paths preserve
the prepared checkpoint's NVFP4/MXFP8 recipe and the requested draft.

Edge-enabled builds default to 0.12. For the earlier text offload, select
`-DTRTMC_EDGELLM_RELEASE=0.11.0` and use a compatible SM120 SDK.
For 0.12, also set
`-DTRTMC_EDGELLM_ONNX=ON -DTRTMC_EDGELLM_ALL_KERNELS=ON`: ONNX tools alone
do not satisfy Muse's kernel requirement. Provision the SDK from pinned Edge
source on the target GPU. A packaged runner uses its own SDK; for a standalone
runner, set `CMAKE_PREFIX_PATH` to the corresponding Model Connect SDK installation.
The family option `--edge-builder-python /absolute/path/to/python`
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
  scientific check scores direct answers when thinking is disabled, because
  the template has already opened the public channel.
  The DFlash2 tree passed all four media checks and the thinking-enabled
  science check on both routes. Native matched target-only IDs on all six
  requests; Edge matched on five, with a comma instead of a newline between
  the two correct video colors. The earlier test expected a generated
  header even with thinking disabled; neither tree route generated one.
- **Prepared NVFP4:** Text only. Both native pairs matched their respective
  targets; both released Edge pairs showed a same-engine greedy discrepancy.
  The DFlash2 tree also ran on both backends. Its native output matched its own
  target with thinking off and on; the Edge tree differed from target-only
  execution. The earlier test counted the absent header as a failure with thinking
  disabled. Separately, the Edge route missed a required term with thinking
  enabled.
  A result on this checkpoint does not qualify media execution.

For native image and video requests, the processor's resized patch grid must
fit the bundle's visual profile. If the processor's rounded fallback exceeds
that profile, native execution reports the limit instead of silently reducing
the image to a single token.

See the [family-owned recipe](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/families/muse_glimmer/edge_llm/README.md)
for commands, dependency requirements, precision/profile boundaries and
qualification limitations.
