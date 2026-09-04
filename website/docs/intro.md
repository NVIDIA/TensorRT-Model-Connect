---
title: TensorRT-Model-Connect
slug: /
---

TensorRT-Model-Connect turns one model checkpoint into a native TensorRT
bundle. Every supported model is a self-contained vertical slice under
`families/<family>/`: Python builder, runtime DSO, bundle sections, and tests.

The shared core resolves one family, writes or reads the bundle container, and
transfers control. It does not contain model logic, a model registry, or a
runtime-strategy switch.

- [Quick start](getting-started/quick-start.md)
- [Supported models](models-recipes/overview.md)
- [AI-native horizontal scaling architecture](architecture/ai-native-horizontal-scaling.md)
- [Add a family](extend/add-model-family.md)
