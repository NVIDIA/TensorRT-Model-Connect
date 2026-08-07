# TensorRT-Model-Connect

> Reference implementations for deploying diverse model families on TensorRT.
> Build from a supported checkpoint, run through task-oriented C++ APIs, and
> use the family-owned implementation as a blueprint for your own changes.

[Documentation](https://nvidia.github.io/TensorRT-Model-Connect/) |
[Quick Start](https://nvidia.github.io/TensorRT-Model-Connect/getting-started/quick-start) |
[Model Support](https://nvidia.github.io/TensorRT-Model-Connect/models-recipes/overview) |
[API Reference](https://nvidia.github.io/TensorRT-Model-Connect/api/overview)

Build a deployment bundle from its canonical Hugging Face model ID, then run
native inference in two commands:

```bash
trtmc build Qwen/Qwen3-0.6B --output qwen3-0.6b.bundle
trtmc run qwen3-0.6b.bundle \
  --prompt "What is the capital of France? Answer in one word." \
  --max-new-tokens 10 \
  --greedy
```

If `trtmc` is not installed, start with
[System Requirements](https://nvidia.github.io/TensorRT-Model-Connect/getting-started/environment-and-repro)
and [Installation](https://nvidia.github.io/TensorRT-Model-Connect/getting-started/installation).
Developers compiling the native CLI, backends, or model DSOs should use the
[Build from Source](https://nvidia.github.io/TensorRT-Model-Connect/getting-started/source-build)
guide.

## AI-native quick start

Give an AI coding agent with terminal, Docker, and NVIDIA GPU access this
prompt:

```text
/goal Clone https://github.com/NVIDIA/TensorRT-Model-Connect.git into a new TensorRT-Model-Connect directory in the current workspace. Detect the current GPU compute capability, modify the repository development Docker image, build and start the container, install TensorRT-Model-Connect, compile the CLI, TensorRT backend, and all native model DSOs only for that SM, then build and run an end-to-end Qwen/Qwen3-0.6B smoke test. Do not commit or push changes. Report the result of the test, show exact command, input and output of the inference run.
```

![TensorRT-Model-Connect build and runtime map](website/static/img/diagrams/trtmc-system-map.svg)

## Why TensorRT-Model-Connect?

- Start from a supported Hugging Face or local checkpoint and build TensorRT
  engines without an intermediate ONNX export step.
- Hand a versioned `.bundle` artifact from the Python-first build environment
  to native C++ task APIs such as text generation, transcription, image and
  video generation, segmentation, embedding, and forecasting.
- Use model-family-owned builders, runtime pipelines, helper kernels, and
  validation contracts as concrete blueprints for modification and
  customization.
- Keep native TensorRT execution and exactly qualified optimized-runtime
  dispatch behind the same task-oriented application boundary.

Read the [Project Overview](https://nvidia.github.io/TensorRT-Model-Connect/getting-started/project-overview)
for the architecture boundary, intended users, and comparison with other
TensorRT integration paths.

TensorRT-Model-Connect is a reference implementation. Users are responsible
for trusting the checkpoints, bundles, native libraries, and local environment
they provide when building or running models.

## Explore the documentation

| Goal | Start here |
| --- | --- |
| Complete the first Qwen inference | [Quick Start](https://nvidia.github.io/TensorRT-Model-Connect/getting-started/quick-start) |
| Select and install an environment | [Get Started](https://nvidia.github.io/TensorRT-Model-Connect/getting-started/overview) |
| Compile the CLI, backends, and model DSOs | [Build from Source](https://nvidia.github.io/TensorRT-Model-Connect/getting-started/source-build) |
| Find an exact checkpoint or model recipe | [Models & Recipes](https://nvidia.github.io/TensorRT-Model-Connect/models-recipes/overview) |
| Look up task and feature workflows | [User Guides](https://nvidia.github.io/TensorRT-Model-Connect/user-guides/overview) |
| Learn through progressive labs and self-checks | [Tutorials](https://nvidia.github.io/TensorRT-Model-Connect/learning-path) |
| Look up CLI, Python, C++, bundle, and config contracts | [Reference](https://nvidia.github.io/TensorRT-Model-Connect/api/overview) |
| Understand architecture or extend the repository | [Developer Guide](https://nvidia.github.io/TensorRT-Model-Connect/developer-guide/overview) |
| Review compatibility, limitations, and lifecycle policy | [Release & Support](https://nvidia.github.io/TensorRT-Model-Connect/release-support/overview) |
| Give a coding agent repository-specific guidance | [AI & Agent Guide](https://nvidia.github.io/TensorRT-Model-Connect/agent-guide) |

## Supported models

The [Supported Models](https://nvidia.github.io/TensorRT-Model-Connect/models-recipes/overview)
page is the single source of truth for exact checkpoints, Hugging Face
architectures, TRTMC profiles, precision, quantization, optimized-runtime
dispatch, configuration, and qualification evidence.

## Contributing and support

- Read [CONTRIBUTING.md](CONTRIBUTING.md) before proposing source or model
  integration changes.
- Report reproducible defects through
  [GitHub Issues](https://github.com/NVIDIA/TensorRT-Model-Connect/issues).
- Report security concerns according to [SECURITY.md](SECURITY.md).
- TensorRT-Model-Connect is licensed under the terms in [LICENSE](LICENSE).

<!-- Collaborative review anchor. -->
