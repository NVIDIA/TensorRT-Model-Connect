/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

const sidebars = {
  docs: [
    'intro',
    'learning-path',
    {
      type: 'category',
      label: 'Getting Started',
      items: [
        'getting-started/glossary',
        'getting-started/environment-and-repro',
        'getting-started/inference-fundamentals',
        'getting-started/quick-start',
        'getting-started/installation',
        'getting-started/build-and-run',
        'getting-started/model-support'
      ]
    },
    {
      type: 'category',
      label: 'Tutorials',
      items: [
        'tutorials/beginner/inspect-bundles',
        'tutorials/beginner/text-generation',
        'tutorials/beginner/bring-your-own-kernel',
        'tutorials/intermediate/multimodal-and-speech',
        'tutorials/intermediate/canary-decoding',
        'tutorials/intermediate/diffusion-and-time-series',
        'tutorials/advanced/quantization-and-runtime-knobs',
        'tutorials/advanced/validation-and-benchmarking'
      ]
    },
    {
      type: 'category',
      label: 'API Manual',
      items: [
        'api/overview',
        'api/python-builder',
        'api/cli-reference',
        'api/cpp-api'
      ]
    },
    {
      type: 'category',
      label: 'Architecture',
      items: [
        'architecture/overview',
        'architecture/bundle-format',
        'architecture/runtime-plugins',
        'architecture/build-system'
      ]
    },
    {
      type: 'category',
      label: 'Unit Design',
      items: [
        'unit-design/overview',
        'unit-design/building-blocks',
        'unit-design/python-builder',
        'unit-design/cpp-runtime',
        'unit-design/testing'
      ]
    },
    {
      type: 'category',
      label: 'Features',
      items: [
        'features/model-families',
        'features/runtime-strategies',
        'features/quantization',
        'features/sampling',
        'features/config-and-backends'
      ]
    },
    {
      type: 'category',
      label: 'Extend',
      items: [
        'extend/overview',
        'extend/add-model-family',
        'extend/add-runtime-strategy',
        'extend/add-config-schema'
      ]
    },
    {
      type: 'category',
      label: 'Reference',
      items: [
        'reference/source-layout',
        'reference/testing',
        'reference/benchmarking',
        'reference/profiling',
        'reference/e2e-l0-replacements',
        'reference/documentation-research'
      ]
    },
    {
      type: 'category',
      label: 'Feature Context',
      collapsed: true,
      items: [
        'context/triattention-native-cpp-worklog',
        'context/config-registry-status',
        'context/model-plugin-encapsulation-plan',
        'context/optimized-runtime-family-adapter-plan',
        'context/adr/README'
      ]
    },
    {
      type: 'category',
      label: 'Operations',
      collapsed: true,
      items: [
        'operations/ai-agent-system',
        'operations/ai-local-pipeline',
        'operations/ai-staging',
        'operations/model-e2e-task-prompt'
      ]
    },
    {
      type: 'category',
      label: 'Wiki Archive',
      collapsed: true,
      items: [
        'wiki/Home',
        'wiki/Architecture-Overview',
        'wiki/Static-Design',
        'wiki/Dynamic-Design',
        'wiki/Pipeline-Deep-Dive',
        'wiki/Source-Layout',
        'wiki/Runtime-Target-Architecture',
        'wiki/Testing-and-Validation',
        'wiki/Traceability-Matrix',
        'wiki/ISO-26262-Compliance',
        'wiki/Adding-a-Model-Family',
        'wiki/Architecture-Extensibility-Assessment',
        'wiki/HF-vs-TRT-Comparison',
        'wiki/TRT-Internals',
        'wiki/FP8-Quantization-Guide',
        'wiki/Agentic-Quantization-Core-Minimal-Plan'
      ]
    }
  ]
};

module.exports = sidebars;
