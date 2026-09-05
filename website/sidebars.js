/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

const path = require('path');
const {collectModelSupportInventory} = require('./plugins/model-support-inventory');

const inventory = collectModelSupportInventory(path.resolve(__dirname, '..'));
const recipes = inventory.taskRecipes.map((task) => ({
  type: 'category',
  label: task.label,
  items: [
    {
      type: 'link',
      label: 'Task overview',
      href: `/models-recipes/model-recipes/tasks/${task.slug}`,
    },
    ...task.families.map((family) => ({
      type: 'link',
      label: family.family,
      href: `/models-recipes/model-recipes/families/${family.slug}`,
    })),
  ],
}));

module.exports = {
  docs: [
    'intro',
    'getting-started/quick-start',
    {
      type: 'category',
      label: 'Models & Recipes',
      items: [
        'models-recipes/overview',
        'models-recipes/model-recipes',
        ...recipes,
      ],
    },
    {
      type: 'category',
      label: 'Architecture',
      items: [
        'architecture/ai-native-horizontal-scaling',
        'architecture/bundle-format',
        'reference/source-layout',
      ],
    },
    {
      type: 'category',
      label: 'API',
      items: ['api/python-builder', 'api/cpp-api'],
    },
    {
      type: 'category',
      label: 'Contribute',
      items: ['extend/add-model-family', 'extend/contributing'],
    },
  ],
};
