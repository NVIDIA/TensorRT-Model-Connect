---
title: Documentation Design Notes
---

The documentation uses established public patterns:

| Source | Pattern used here |
| --- | --- |
| [Docusaurus](https://docusaurus.io/docs/docs-introduction) | A docs-first site with hierarchy and generated routes. |
| [Viz.js](https://viz-js.com/) and [Graphviz](https://graphviz.org/documentation/) | Reviewable diagram sources and checked-in accessible SVG output. |
| [Diataxis](https://diataxis.fr/) | Tutorials, task guides, reference, and explanation have different jobs. |
| [Google documentation guidance](https://google.github.io/styleguide/docguide/best_practices.html) | Direct language, runnable examples, and docs maintained with code. |
| [Universal Design for Learning](https://www.cast.org/resources/about-universal-design-for-learning/) | Concepts appear as prose, diagrams, and hands-on checks. |
| [TensorRT-LLM docs](https://nvidia.github.io/TensorRT-LLM/) | Separate getting-started, model, API, feature, and developer routes. |
| [Kubernetes tasks](https://kubernetes.io/docs/tasks/) | Goal-oriented pages focus on one outcome. |

Applied rules:

- The first path is a runnable checkpoint-to-Task workflow.
- The generated model inventory comes from current family-owned manifests.
- Tutorials teach concepts; guides complete tasks; references define exact
  surfaces; architecture explains ownership and dependencies.
- Every diagram has reviewable source, accessible metadata, and checked-in SVG
  output.
- Source paths and commands are executable claims and are updated with the code.
- Model support, GPU execution, parity, performance, and release status are
  distinct evidence levels.
- Historical context is explicitly marked and never presented as current
  architecture.
