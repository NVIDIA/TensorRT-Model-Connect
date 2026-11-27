---
title: InternVL with Edge-LLM
---

InternVL selects its own Edge builder and C++ runner. To use the default
Edge 0.12.0 SDK, enable the native dependency with `TRTMC_ENABLE_EDGELLM=ON`. Build on the execution GPU. A Python wheel
or source installation supplies the builder tools, not the native C++ SDK.

An installed Model Connect wheel uses its own native Edge SDK, not a
different SDK on `CMAKE_PREFIX_PATH`. To switch releases, use a matching
wheel or a standalone Model Connect builder and runner together.

### Edge 0.12.0

For InternVL3 original checkpoints the family selects the direct builder.
InternVL3.5 uses the official Edge ONNX exporter to preserve the nested
decoder's query/key normalization. For the latter, configure
`TRTMC_EDGELLM_ONNX=ON` and use the release's ONNX Python extras:

```sh
trtmc internvl build /path/to/InternVL3_5-2B-HF --precision fp16 \
  --max-sequence-length 4096 --edge-builder-python /path/to/edge-env/bin/python \
  -o internvl.bundle
trtmc run internvl.bundle --image vehicle.jpg \
  --prompt "What color is the vehicle in this image? Answer in one word." --max-new-tokens 10
```

The original InternVL3 1B, 2B, 8B and 14B and InternVL3.5 1B, 2B, 4B and 8B checkpoints
matched the independent FP32 reference on the vehicle-image case at batch 1,
TP1 and context 4096 through both Edge and native TensorRT. This does not
qualify other sizes, weight formats, images or GPU targets. The native path
uses TensorRT directly, not ONNX.

For the InternVL3.5-14B original at the same profile, ONNX export succeeded
on an x86 A100 40 GiB with CUDA 13.4 and TensorRT 11.4.0.106. The Edge
LLM builder then requested a single 45,718,743,296-byte allocation. The
Edge visual builder and decode did not run there. The family warns and
retries the unchanged native request for this measured profile when the
allocation exceeds total GPU memory. The native fallback matched the
independent FP32 answer and passed the image-feature health check.

On a native Linux ARM64 GB300, the same checkpoint and build profile passed
the official ONNX exporter, LLM builder and visual builder with the 0.12
SDK's CTM decode kernels enabled. The Model Connect C++ runner matched the
independent FP32 token IDs on the same vehicle-image question; the visual
encoder produced all 1792 image tokens. This does not change the A100 result
or qualify other prompts and GPU targets.

For the prepared InternVL3 1B, 2B, 8B and 14B AWQ checkpoints, configure
`TRTMC_EDGELLM_ALL_KERNELS=ON` and use an Edge builder Python with PyTorch
installed to read the checkpoint files. Leave `--quantization` unset.
The direct Edge route with `--int4-gemm-plugin-version 2` and the native
fallback without that Edge-specific option both matched the FP32 reference
for the same prepared weights on the vehicle-image case above.

The 14B AWQ Edge build and decode passed on a 40 GiB A100. The native
TensorRT builder could not allocate 29,535,502,976 bytes on that GPU; the
same native request passed on a 48 GiB RTX A6000. This is an observed
build-memory limit, not a minimum-memory guarantee or an SM restriction.

### Explicit Edge 0.11.0 build

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
