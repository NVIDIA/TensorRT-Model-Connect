<div align="center">

<h1>TensorRT-Model-Connect</h1>

<p><strong>Deploy supported Hugging Face models for end-to-end TensorRT inference in just two commands.</strong></p>

[Documentation](https://nvidia.github.io/TensorRT-Model-Connect/)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Quick Start](https://nvidia.github.io/TensorRT-Model-Connect/getting-started/quick-start)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Model Support](https://nvidia.github.io/TensorRT-Model-Connect/models-recipes/overview)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[API Reference](https://nvidia.github.io/TensorRT-Model-Connect/api/python-builder)

</div>

> **Public Preview** — TensorRT-Model-Connect is evolving rapidly and is
> intended for evaluation and feedback. APIs, scope, and direction may change.
> See [Project status](#project-status).

<a id="news-and-updates"></a>

## 📰 News and Updates

<!-- Newest first. Keep no more than the three most recent entries. -->

- **2026-08-23 · Blog** — [AI-Native by Design: What We Learned Building
  TensorRT-Model-Connect](https://nvidia.github.io/TensorRT-Model-Connect/blog/ai-native-by-design)
- **2026-08-18 · Announcement** — [TensorRT-Model-Connect enters public
  preview](https://x.com/NVIDIAAI/status/2089750360869233059)

<a id="example-code"></a>

## 💻 Example Code

Try [Nemotron Voice Lab for Windows](examples/windows_voicechat/README.md) for a
local voice application with TensorRT-RTX, live transcripts, audio-reactive visuals,
and stream mode. The example contains source and setup scripts; dependencies and
model weights are downloaded separately on the user's PC.

```bash
python -m tensorrt_model_connect build Qwen/Qwen3-0.6B \
  --max-sequence-length 16384 \
  --output qwen3-0.6b.bundle
trtmc run ./qwen3-0.6b.bundle \
  --runtime-root /opt/trtmc/lib \
  --prompt "What is the capital of France? Answer in one word." \
  --use-chat-template true \
  --enable-thinking false
# Generated text: Paris
```

The same bundle works from
[C++](https://nvidia.github.io/TensorRT-Model-Connect/api/cpp-api):

```cpp
#include <iostream>
#include <stdexcept>

#include <trtmc/runtime/family_loader.h>
#include <trtmc/task.h>

auto task = trtmc::load_task("./qwen3-0.6b.bundle", "/opt/trtmc/lib");
auto* text = dynamic_cast<trtmc::ITextGeneration*>(task.get());
if (text == nullptr) throw std::runtime_error("unexpected task");
std::cout << text->generate("What is the capital of France? Answer in one word.").text << '\n';
```

<a id="get-started-and-stay-tuned"></a>

## 🌟 Get Started & Stay Tuned

<div align="center">

<img width="960" height="540" alt="Animated TensorRT-Model-Connect project hero" src="TRTMCHERO-small.gif" />

</div>

Ready to try TensorRT-Model-Connect? Follow the
[Quick Start](https://nvidia.github.io/TensorRT-Model-Connect/getting-started/quick-start)
to build and run your first supported model.

We're glad you're here. [Star the repository](https://github.com/NVIDIA/TensorRT-Model-Connect)
to keep TensorRT-Model-Connect on your radar as we share new model integrations,
releases, examples, and community updates. Ideas and feedback are always
welcome through the [issue chooser](https://github.com/NVIDIA/TensorRT-Model-Connect/issues/new/choose).

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
- Use model-family-owned builders, runtime pipelines, helper kernels, and
  validation contracts as concrete blueprints for modification and
  customization.
- Keep native TensorRT and optional TensorRT-RTX execution behind the same
  task-oriented application boundary.

Read the [Architecture Overview](https://nvidia.github.io/TensorRT-Model-Connect/architecture/ai-native-horizontal-scaling)
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
| Select and install an environment | [Quick Start](https://nvidia.github.io/TensorRT-Model-Connect/getting-started/quick-start) |
| Compile the CLI, backends, and model DSOs | [Build from Source](https://nvidia.github.io/TensorRT-Model-Connect/getting-started/source-build) |
| Find an exact checkpoint or model recipe | [Models & Recipes](https://nvidia.github.io/TensorRT-Model-Connect/models-recipes/overview) |
| Look up task and runtime contracts | [C++ Task API](https://nvidia.github.io/TensorRT-Model-Connect/api/cpp-api) |
| Understand the bundle contract | [Bundle Format](https://nvidia.github.io/TensorRT-Model-Connect/architecture/bundle-format) |
| Understand the source layout | [Source Layout](https://nvidia.github.io/TensorRT-Model-Connect/reference/source-layout) |
| Add an isolated model family | [Add a Model Family](https://nvidia.github.io/TensorRT-Model-Connect/extend/add-model-family) |
| Understand the dependency boundaries | [Architecture](https://nvidia.github.io/TensorRT-Model-Connect/architecture/ai-native-horizontal-scaling) |
| Contribute to the project | [Contributing](https://nvidia.github.io/TensorRT-Model-Connect/extend/contributing) |

<a id="supported-models"></a>

## 🧩 Supported models

The [Supported Models](https://nvidia.github.io/TensorRT-Model-Connect/models-recipes/overview)
page is the single source of truth for exact checkpoints, Hugging Face
architectures, family-owned tasks, precision, quantization, topology, and
validation evidence.

<a id="get-help-and-file-an-issue"></a>

## 🛟 Get help and file an issue

Collect the model, environment, command, and log details maintainers need, then
use the [issue chooser](https://github.com/NVIDIA/TensorRT-Model-Connect/issues/new/choose)
for usage questions, reproducible bugs, feature or model requests, and
documentation corrections.

Do not disclose suspected security vulnerabilities in a public issue. Follow
[SECURITY.md](SECURITY.md) to report them privately to NVIDIA PSIRT.

<a id="project-status"></a>

## 🧪 Project status

TensorRT-Model-Connect is an experimental project. Its APIs, scope, and
direction may evolve as we learn from users. This preview is intended to inform
future decisions and does not establish a specific product roadmap or release
timeline.

<a id="contributing"></a>

## 🤝 Contributing

- Read [CONTRIBUTING.md](CONTRIBUTING.md) before proposing source or model
  integration changes.
- TensorRT-Model-Connect is licensed under the terms in [LICENSE](LICENSE).

<!-- Collaborative review anchor: batch 2. -->
