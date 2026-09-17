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

Provision Edge-LLM explicitly when building Model Connect, not during model
builds or inference. `TRTMC_ENABLE_EDGELLM=ON` selects the public Edge-LLM
0.10.1 snapshot at `e8b29522938901f6df19ebeedd4b69bc8edbcd97`. The default is
`OFF`. Configure and build on the inference GPU host; cross compilation is
rejected. The package must match the native CPU/GPU and CUDA/TensorRT stack.
Runtime compilation must also use the exact pinned JSON dependency headers;
the SDK rejects same-version development headers with an incompatible C++ ABI.

- `TRTMC_EDGELLM_ALL_KERNELS=ON` requests all upstream operator groups supported
  by the local GPU.
- `TRTMC_EDGELLM_ONNX=ON` also installs the original Python exporter and native
  C++ ONNX engine builder. It does not select a model's build flow.
- `CMAKE_PREFIX_PATH` points builders and runtime compilation to the installed
  SDK. Reuse fails explicitly if a requested capability is absent.

Follow the repository's [pinned SDK installation instructions](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/cmake/edgellm/README.md)
for dependencies, native architecture selection and offline provisioning.
Each family decides whether and how to use the package. Installing the SDK
is not evidence that a model, precision, input modality or execution variant
has passed validation. Ordinary model builds never install missing SDK tools.
