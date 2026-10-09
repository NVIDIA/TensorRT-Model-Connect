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

Nine exact installed single-image profiles passed the local rollout. AWQ1B,
2B and14B remain executable with documented accuracy failures. The recorded
InternVL3.5-14B FP16/context384 build cannot fit the tested40/48GiB devices and
is rejected there before repeating a known failure. Larger GPUs are not thereby
qualified or blacklisted. See the
[owning recipe](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/families/internvl/edge_llm/README.md)
for the precise scope, existing image-health helper and failure boundaries.
