/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

const sidebars = {
  docs: [
    'intro',
    {
      type: 'category',
      label: 'Getting Started',
      link: {
        type: 'doc',
        id: 'getting-started/overview'
      },
      items: [
        'getting-started/environment-and-repro',
        'getting-started/installation',
        'getting-started/glossary',
        'getting-started/quick-start'
      ]
    },
    {
      type: 'category',
      label: 'Learn & Tutorials',
      link: {
        type: 'doc',
        id: 'learning-path'
      },
      items: [
        {
          type: 'category',
          label: 'Beginner: Understand One Inference',
          items: [
            'getting-started/inference-fundamentals',
            'tutorials/beginner/inspect-bundles',
            'tutorials/beginner/text-generation'
          ]
        },
        {
          type: 'category',
          label: 'Intermediate: Model Recipes',
          items: [
            'getting-started/build-and-run',
            'tutorials/intermediate/multimodal-and-speech',
            'tutorials/intermediate/canary-decoding',
            'tutorials/intermediate/diffusion-and-time-series'
          ]
        },
        {
          type: 'category',
          label: 'Advanced: Optimize, Extend, and Validate',
          items: [
            'tutorials/advanced/quantization-and-runtime-knobs',
            'tutorials/advanced/multi-device-inference',
            'tutorials/advanced/bring-your-own-kernel',
            'tutorials/advanced/validation-and-benchmarking'
          ]
        }
      ]
    },
    {
      type: 'category',
      label: 'API Reference',
      link: {
        type: 'doc',
        id: 'api/overview'
      },
      items: [
        'api/python-builder',
        'api/cli-reference',
        'api/cpp-api'
      ]
    },
    {
      type: 'category',
      label: 'Architecture & Design',
      link: {
        type: 'doc',
        id: 'architecture/overview'
      },
      items: [
        {
          type: 'category',
          label: 'System Architecture',
          items: [
            'architecture/bundle-format',
            'architecture/runtime-lifecycle',
            'architecture/build-system'
          ]
        },
        {
          type: 'category',
          label: 'Component Design',
          items: [
            'architecture/units-and-ownership',
            'architecture/build-pipeline',
            'architecture/validation-design'
          ]
        },
        'reference/source-layout'
      ]
    },
    {
      type: 'category',
      label: 'Contribute & Extend',
      link: {
        type: 'doc',
        id: 'extend/overview'
      },
      items: [
        'extend/contributing',
        'extend/add-model-family',
        'extend/add-optimized-runtime',
        'extend/add-runtime-strategy',
        'extend/add-config-schema',
        'extend/model-validation'
      ]
    },
    {
      type: 'category',
      label: 'Feature Reference & Context',
      collapsed: true,
      link: {
        type: 'doc',
        id: 'features/overview'
      },
      items: [
        {
          type: 'category',
          label: 'Model & Runtime Integration',
          items: [
            'getting-started/model-support',
            'features/model-families',
            'features/runtime-strategies',
            'context/optimized-runtime-family-adapter-plan',
            'context/model-plugin-encapsulation-plan'
          ]
        },
        {
          type: 'category',
          label: 'Inference Behavior & Optimizations',
          items: [
            'features/multi-device',
            'features/tvm-ffi',
            'features/sampling',
            'features/triattention',
            'context/triattention-native-cpp-worklog'
          ]
        },
        {
          type: 'category',
          label: 'Build, Quantization & Configuration',
          items: [
            'features/quantization',
            'features/config-and-backends',
            'context/config-registry-status'
          ]
        },
        {
          type: 'category',
          label: 'Validation, CI & Performance',
          items: [
            'reference/testing',
            'reference/benchmarking',
            'reference/profiling',
            'reference/e2e-l0-replacements',
            'context/traceability-and-safety'
          ]
        },
        {
          type: 'category',
          label: 'Design & Project History',
          items: [
            'context/adr/README',
            'reference/documentation-research'
          ]
        }
      ]
    }
  ]
};

module.exports = sidebars;
