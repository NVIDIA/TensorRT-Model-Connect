<div align="center">

<h1>TensorRT-Model-Connect</h1>

<p><strong>Deploy supported Hugging Face models for end-to-end TensorRT inference in just two commands.</strong></p>

[Documentation](https://nvidia.github.io/TensorRT-Model-Connect/)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Quick Start](https://nvidia.github.io/TensorRT-Model-Connect/getting-started/quick-start)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Model Support](https://nvidia.github.io/TensorRT-Model-Connect/models-recipes/overview)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[API Reference](https://nvidia.github.io/TensorRT-Model-Connect/api/overview)

</div>

<a id="example-code"></a>

## 💻 Example Code

```bash
trtmc build Qwen/Qwen3-0.6B -o qwen3-0.6b.bundle
trtmc run ./qwen3-0.6b.bundle --prompt "What is the capital of France? Answer in one word." --chat-template --no-thinking
# Generated text: Paris
```

The same bundle works from
[C++](https://nvidia.github.io/TensorRT-Model-Connect/api/cpp-api):

```cpp
auto pipeline = trtmc::load("./qwen3-0.6b.bundle");
std::cout << pipeline->generate("What is the capital of France? Answer in one word.").text << '\n';
```

<a id="what-is-tensorrt-model-connect"></a>

## 🔎 What is TensorRT-Model-Connect?
**TensorRT Model Connect is an extensive collection of AI Model reference implementations in C++, on top of NVIDIA TensorRT**. Model Connect is powered by an agentic workflow that continuously adds support for upcoming models, drastically reducing integration effort on user side and time until new models become compatible.
<img width="1318" height="1088" alt="MC-what-it-is" src="website/static/img/readme/model-connect-overview.png" />

<a id="choose-the-right-tensorrt-path"></a>

## 🧭 Choose the right abstraction layer

- Use TensorRT-Model-Connect to explore models quickly and evaluate broad
  model coverage.
- For production LLM/VLM deployment on NVIDIA edge platforms where performance
  is the priority, start directly with
  [TensorRT Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM).

<img width="1606" height="979" alt="TensorRT abstraction layers from Model Connect through Edge-LLM to TensorRT" src="website/static/img/readme/tensorrt-stack.png" />

<a id="why-tensorrt-model-connect"></a>

## 💡 Why TensorRT-Model-Connect?

- Start from a supported Hugging Face or local checkpoint and build TensorRT
  engines without an intermediate ONNX export step.
- Hand a versioned `.bundle` artifact from the Python-first build environment
  to native C++ task APIs such as text generation, transcription, image and
  video generation, segmentation, embedding, and forecasting.
- Use each family's self-contained `model.py` build recipe, runtime pipeline,
  helper kernels, and validation contracts as concrete blueprints for
  modification and customization. The shared Python entry point only resolves
  the owning family and calls `model.build()` once.
- Keep native TensorRT execution and exactly qualified optimized-runtime
  dispatch behind the same task-oriented application boundary.

Read the [Project Overview](https://nvidia.github.io/TensorRT-Model-Connect/getting-started/project-overview)
for the architecture boundary, intended users, and comparison with other
TensorRT integration paths.

TensorRT-Model-Connect is a reference implementation. Users are responsible
for trusting the checkpoints, bundles, native libraries, and local environment
they provide when building or running models.

<a id="getting-started"></a>

## 🚀 Getting Started

**Recommended Quick Start: AI-Native**

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

Want to know more? See the
[Quick Start documentation](https://nvidia.github.io/TensorRT-Model-Connect/getting-started/quick-start).

<a id="explore-the-documentation"></a>

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

<a id="supported-models"></a>

## 🧩 Supported models

The [Supported Models](https://nvidia.github.io/TensorRT-Model-Connect/models-recipes/overview)
page is the single source of truth for exact checkpoints, Hugging Face
architectures, TRTMC profiles, precision, quantization, optimized-runtime
dispatch, configuration, and qualification evidence.

<a id="get-help-and-file-an-issue"></a>

## 🛟 Get help and file an issue

Start with [Get Help and File an Issue](https://nvidia.github.io/TensorRT-Model-Connect/release-support/get-help)
to choose the right support route and collect the model, environment, command,
and log details maintainers need. Use the
[issue chooser](https://github.com/NVIDIA/TensorRT-Model-Connect/issues/new/choose)
for usage questions, reproducible bugs, feature or model requests, and
documentation corrections.

Do not disclose suspected security vulnerabilities in a public issue. Follow
[SECURITY.md](SECURITY.md) to report them privately to NVIDIA PSIRT.

<a id="contributing"></a>

## 🤝 Contributing

- Read [CONTRIBUTING.md](CONTRIBUTING.md) before proposing source or model
  integration changes.
- TensorRT-Model-Connect is licensed under the terms in [LICENSE](LICENSE).

<!-- Collaborative review anchor: batch 2. -->
