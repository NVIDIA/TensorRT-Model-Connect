---
title: Documentation Research Notes
---

The site structure follows patterns from current public documentation:

| Project | Pattern used here |
| --- | --- |
| [Docusaurus](https://docusaurus.io/docs/docs-introduction) | Docs-only mode with hierarchical pages and sidebar-driven navigation. |
| [Docusaurus Mermaid diagrams](https://docusaurus.io/docs/3.8.1/markdown-features/diagrams) | Versioned architecture diagrams as Mermaid code blocks rendered by `@docusaurus/theme-mermaid`. |
| [Diataxis](https://diataxis.fr/) | Separate learning-oriented tutorials, goal-oriented guides, information-oriented reference, and understanding-oriented explanation. |
| [Google documentation best practices](https://google.github.io/styleguide/docguide/best_practices.html) | Keep docs fresh with code, prefer simple direct language, and put the simplest use case first. |
| [Universal Design for Learning](https://www.cast.org/resources/about-universal-design-for-learning/) | Present concepts in multiple forms: motivation, visual representation, and hands-on proof points. |
| [ysyx course handouts](https://ysyx.oscc.cc/docs/en/2407/f/1.html) | Teach through staged handouts with information-box categories, required tasks, reflective questions, learning logs, and independent problem-solving habits. |
| Historical local TensorRT documentation snapshot (not part of this repository) | Use restrained NVIDIA technical-doc styling: dense side navigation, right-page table of contents, neutral cards, admonitions, tabbed examples, dropdowns, and reference-heavy tables. |
| [TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/) | Separate getting started, model support, CLI reference, API reference, features, and developer guide sections. |
| [NVIDIA TensorRT getting started](https://developer.nvidia.com/tensorrt-getting-started) | Route learners by beginner, intermediate, and expert levels across videos, notebooks, samples, and guides. |
| [Kubernetes](https://kubernetes.io/docs/tasks/) | Task pages focus on one outcome and provide a short sequence of steps. |
| [vLLM](https://docs.vllm.ai/en/latest/) | Route users by intent: quickstart for users, user guide/tutorials for operators, developer guide/API reference for contributors. |

Applied rules:

- The first screen is documentation, not marketing.
- Quick start is short and runnable.
- The home page now acts as a course entry point: it explains the promise, shows a visual map, and routes by learner intent.
- Tutorials are grouped by beginner, intermediate, and advanced workflows.
- The learning path includes outcomes and proof points, not only reading links.
- Learning pages now use course-handout units: required reading, required tasks, progress checks, common traps, learning-log prompts, further reading, and optional exercises.
- Reference pages are separated from conceptual architecture pages.
- Extension docs are task-oriented and name exact files to edit.
- Architecture and unit design pages use diagrams first, then source-level tables.
- Core diagrams have static SVG versions so they render even if client-side Mermaid fails.
- Beginner material teaches inference vocabulary before asking users to debug TensorRT runtime behavior.
- The visual style favors technical documentation over marketing: square corners, light borders, neutral surfaces, concise tables, and NVIDIA-green accents.
- Source-derived counts and strategy names are taken from the current checkout, not copied from older wiki text.
