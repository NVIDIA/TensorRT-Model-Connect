/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

const path = require('path');
const {
  collectModelSupportInventory,
} = require('./plugins/model-support-inventory');

const modelInventory = collectModelSupportInventory(path.resolve(__dirname, '..'));
const modelRecipeTaskItems = modelInventory.taskRecipes.map((task) => ({
  type: 'category',
  label: task.label,
  collapsed: true,
  items: [
    {
      type: 'link',
      label: 'Task overview',
      href: `/models-recipes/model-recipes/tasks/${task.slug}`,
      autoAddBaseUrl: true,
    },
    ...task.families.map((family) => ({
      type: 'link',
      label: family.family,
      href: `/models-recipes/model-recipes/families/${family.slug}`,
      autoAddBaseUrl: true,
    })),
  ],
}));

const sidebars = {
  docs: [
    'intro',
    {
      type: 'category',
      label: 'Get Started',
      link: {type: 'doc', id: 'getting-started/overview'},
      items: [
        'getting-started/project-overview',
        'getting-started/environment-and-repro',
        'getting-started/installation',
        'getting-started/source-build',
        'getting-started/quick-start',
        'getting-started/troubleshooting'
      ]
    },
    {
      type: 'category',
      label: 'Models & Recipes',
      link: {type: 'doc', id: 'models-recipes/overview'},
      items: [
        {
          type: 'category',
          label: 'Model Recipes',
          link: {type: 'doc', id: 'models-recipes/model-recipes'},
          items: modelRecipeTaskItems,
        },
      ]
    },
    {
      type: 'category',
      label: 'User Guides',
      link: {type: 'doc', id: 'user-guides/overview'},
      items: [
        'user-guides/build-a-bundle',
        'user-guides/inspect-a-bundle',
        'user-guides/run-inference',
        {
          type: 'category',
          label: 'Task Guides',
          items: [
            'user-guides/text-generation',
            'user-guides/multimodal-speech',
            'user-guides/image-video-generation',
            'user-guides/time-series'
          ]
        },
        'user-guides/configure-runtime',
        'features/quantization',
        'features/multi-device',
        'user-guides/validate-benchmark'
      ]
    },
    {
      type: 'category',
      label: 'Tutorials',
      link: {type: 'doc', id: 'learning-path'},
      items: [
        {
          type: 'category',
          label: 'Foundations',
          items: [
            'getting-started/inference-fundamentals',
            'tutorials/beginner/inspect-bundles',
            'tutorials/beginner/text-generation'
          ]
        },
        {
          type: 'category',
          label: 'Task Labs',
          items: [
            'tutorials/intermediate/multimodal-and-speech',
            'tutorials/intermediate/canary-decoding',
            'tutorials/intermediate/diffusion-and-time-series'
          ]
        },
        {
          type: 'category',
          label: 'Advanced Labs',
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
      label: 'Reference',
      link: {type: 'doc', id: 'api/overview'},
      items: [
        'api/cli-reference',
        'api/python-builder',
        'api/cpp-api',
        'architecture/bundle-format',
        'features/config-and-backends',
        'features/sampling',
        'reference/testing',
        'reference/benchmarking',
        'reference/profiling',
        'getting-started/glossary'
      ]
    },
    {
      type: 'category',
      label: 'Developer Guide',
      link: {type: 'doc', id: 'developer-guide/overview'},
      items: [
        {
          type: 'category',
          label: 'Architecture',
          items: [
            'architecture/overview',
            'architecture/units-and-ownership',
            'architecture/build-pipeline',
            'architecture/runtime-lifecycle',
            'architecture/build-system',
            'architecture/validation-design',
            'features/model-families',
            'features/runtime-strategies'
          ]
        },
        {
          type: 'category',
          label: 'Contribute & Extend',
          items: [
            'extend/overview',
            'extend/contributing',
            'extend/add-model-family',
            'extend/add-runtime-strategy',
            'extend/add-optimized-runtime',
            'extend/add-config-schema',
            'extend/model-validation'
          ]
        },
        'features/tvm-ffi',
        'features/triattention',
        'reference/source-layout'
      ]
    },
    {
      type: 'category',
      label: 'Release & Support',
      link: {type: 'doc', id: 'release-support/overview'},
      collapsed: true,
      items: [
        'release-support/compatibility',
        'release-support/known-issues',
        'release-support/troubleshooting',
        'release-support/release-notes',
        'release-support/migration-guide',
        'release-support/deprecation-policy'
      ]
    },
    {
      type: 'category',
      label: 'AI & Agent Guide',
      link: {type: 'doc', id: 'agent-guide'},
      collapsed: true,
      items: []
    }
  ]
};

module.exports = sidebars;
