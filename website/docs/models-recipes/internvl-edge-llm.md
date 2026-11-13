---
title: InternVL with Edge-LLM
---

InternVL owns optional Edge-LLM 0.11.0 execution for its admitted InternVL3 and
InternVL3.5 configurations. The family selects the direct or ONNX builder and
maps persistent C++ inference; no shared backend selector is added.

```sh
trtmc internvl build /path/to/InternVL3-2B-hf --precision fp16 \
  --max-sequence-length 384 -o internvl.bundle
```

Provision the native Edge SDK on the execution GPU. InternVL3.5 and AWQ require
ONNX tools; AWQ additionally requires all native kernels. A compatible installed
Python wheel can supply builder tools, but does not replace that native SDK.
The family-only `--int4-gemm-plugin-version` option preserves the recorded INT4
backend controls without changing quality gates.

Nine exact installed single-image profiles passed the local rollout. AWQ 1B,
2B and 14B remain executable with documented accuracy failures. The recorded
InternVL3.5-14B FP16/context 384 build cannot fit the tested 40/48 GiB devices and
is rejected by the Edge route there before repeating a known failure; the unchanged
request then attempts the family-owned native TensorRT builder. Larger GPUs are not thereby
qualified or blacklisted. See the
[owning recipe](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/families/internvl/edge_llm/README.md)
for the precise scope, existing image-health helper and failure boundaries.

## Native fallback

Without the optional SDK, or after an unavailable/failed Edge build, ordinary
requests use the family's TensorRT Python graph and Model Connect C++ runtime.
TP1 prefill and decode share one decoder engine's weights through two execution
contexts. Current image tiling and prompt framing are preserved. Existing split
bundles remain readable; rebuilding selects the shared-engine layout.

Prepared group128 GEMM AWQ checkpoints retain their quantized values, reconstructed
at build time for native matmuls. This uses more storage than packed INT4 and does
not claim Edge kernel performance. Explicit `--int4-gemm-plugin-version` requests
remain Edge-specific and are not silently reinterpreted as native controls.

Retained native checks cover the exact InternVL3-2B-hf, InternVL3-2B-AWQ and
InternVL3_5-1B-HF image workloads described in the owning recipe. They are not
qualification of every size, context, image, platform, or current PR revision.
