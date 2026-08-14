<div align="center">

<h1>TensorRT-Model-Connect</h1>

<p><strong>A collection of C++ reference implementations for diverse AI models on NVIDIA TensorRT, continuously expanded through an agentic workflow.</strong></p>

[Documentation](https://nvidia.github.io/TensorRT-Model-Connect/)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Quick Start](https://nvidia.github.io/TensorRT-Model-Connect/getting-started/quick-start)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Model Support](https://nvidia.github.io/TensorRT-Model-Connect/models-recipes/overview)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[API Reference](https://nvidia.github.io/TensorRT-Model-Connect/api/overview)

</div>

## 🤖 AI Native QuickStart

Give an AI coding agent with terminal, Docker, and NVIDIA GPU access this
prompt:

```text
/goal Use the current TensorRT-Model-Connect checkout, or clone
https://github.com/NVIDIA/TensorRT-Model-Connect.git if none is provided. Read
AGENTS.md, then follow website/docs/getting-started/source-build.md and
website/docs/getting-started/quick-start.md exactly. Do not modify source,
tests, Dockerfiles, git history, or remote state. Report the selected GPU,
exact commands, bundle path, inference output, and any deviation from the
documentation.
```

## 📦 Build a Deployment Bundle

Once your environment is ready, use this workflow to build a deployment bundle
from a supported Hugging Face model and run native inference.

If `trtmc` is not installed, start with
[System Requirements](https://nvidia.github.io/TensorRT-Model-Connect/getting-started/environment-and-repro)
and [Installation](https://nvidia.github.io/TensorRT-Model-Connect/getting-started/installation).
Developers compiling the native CLI, backends, or model DSOs should use the
[Build from Source](https://nvidia.github.io/TensorRT-Model-Connect/getting-started/source-build)
guide.

```bash
trtmc build Qwen/Qwen3-0.6B \
  --precision bf16 \
  --max-cache-length 16384 \
  --output qwen3-0.6b.bundle
trtmc run ./qwen3-0.6b.bundle \
  --prompt "What is the capital of France? Answer in one word." \
  --chat-template \
  --no-thinking \
  --max-new-tokens 64 \
  --temperature 0.7 \
  --top-k 20 \
  --top-p 0.8 \
  --seed 42
```

## What is TensorRT-Model-Connect?
**TensorRT Model Connect is an extensive collection of AI Model reference implementations in C++, on top of NVIDIA TensorRT**. Model Connect is powered by an agentic workflow that continuously adds support for upcoming models, drastically reducing integration effort on user side and time until new models become compatible.
<img width="1318" height="1088" alt="MC-what-it-is" src="https://github.com/user-attachments/assets/85850b96-5a30-4531-bcec-98e00883dedb" />

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

## 📚 Explore the documentation

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
