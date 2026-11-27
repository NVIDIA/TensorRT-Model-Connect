---
title: Configure Runtime Behavior
description: Put a setting at build, load, or request time without crossing ownership boundaries.
---

First identify when the setting takes effect:

| Lifecycle | Examples | Rebuild? |
| --- | --- | --- |
| Build | Precision, engine bounds, topology, quantization, dynamic-KV graph | Yes |
| Family bundle state | Tokenizer, preprocessing weights, engine layout, private runtime JSON | Yes |
| Load | Runtime root, runtime-sized KV capacity, TensorRT-RTX cache/CUDA graphs, BYOK binding | No |
| Request | Prompt, sampling, language, media, denoising steps, seed | No |

The current API has no generic `--config` / `--set` registry. Build values are
typed `BuildRequest` fields; runtime values are loader inputs or typed Task
configs. Family-only state remains in family sections and implementation code.

```bash
trtmc run model.bundle \
  --runtime-root /opt/trtmc/lib \
  --kv-cache-size 4GiB \
  --prompt "Hello" \
  --temperature 0
```

Only compatible family bundles accept runtime-sized KV cache. Likewise,
`--runtime-cache` and `--cuda-graphs` are valid only for a `trt_rtx` bundle.
Unsupported values fail; they are not silently ignored.

See [Configuration and Backends](../features/config-and-backends.md),
[Quantization](../features/quantization.md), and
[Multi-Device Execution](../features/multi-device.md).

## Optional native Edge-LLM SDK

Edge offload is disabled by default. Configure Model Connect with
`-DTRTMC_ENABLE_EDGELLM=ON` to use the default Edge-LLM 0.12.0 SDK.
To build against 0.11.0 instead, also set `-DTRTMC_EDGELLM_RELEASE=0.11.0`.
An existing build directory retains its cached release, even if 0.11.0 was
the old default. To upgrade, configure a fresh Edge-enabled directory and rebuild.
Build on the inference GPU host with the matching CUDA and TensorRT stack;
cross-compilation is not supported. When reusing an installed SDK, Model Connect
checks the actual JSON headers for C++ ABI compatibility, not just their version.

- `CMAKE_PREFIX_PATH` selects an existing matching native SDK. Alternatively,
  provision the pinned source during the Model Connect build. An installed
  Model Connect wheel uses the native SDK it was built with.
- `TRTMC_EDGELLM_ALL_KERNELS=ON` requests all operator groups supported by
  the local GPU when provisioning from source.
- `TRTMC_EDGELLM_ONNX=ON` requests the native ONNX builder executables.
  Install matching Python tools separately in the builder interpreter, from
  PyPI or source. A Python installation does not replace the C++ SDK, and this
  option does not select a family's build flow.
- `TRTMC_EDGELLM_POLICY_SDK=ON` requests the optional 0.12 GR00T and Pi0.5
  policy libraries and headers. It does not enable a model by itself.

Follow the repository's [pinned SDK installation instructions](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/cmake/edge_llm/README.md)
for dependencies, native architecture selection and offline provisioning.
Each family decides whether and how to use the package. Installing the SDK
is not evidence that a model, precision, input modality or execution variant
has passed validation. Ordinary model builds never install missing SDK tools.
