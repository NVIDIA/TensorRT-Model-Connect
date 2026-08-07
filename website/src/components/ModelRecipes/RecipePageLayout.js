/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import {DocsSidebarProvider} from '@docusaurus/plugin-content-docs/client';
import {usePluginData} from '@docusaurus/useGlobalData';
import Layout from '@theme/Layout';
import DocRootLayout from '@theme/DocRoot/Layout';

function recipeSidebarItems(taskRecipes) {
  return [
    {
      type: 'category',
      label: 'Models & Recipes',
      collapsed: false,
      collapsible: true,
      items: [
        {
          type: 'link',
          label: 'Supported Models',
          href: '/models-recipes/overview',
          autoAddBaseUrl: true,
        },
        {
          type: 'category',
          label: 'Model Recipes',
          collapsed: false,
          collapsible: true,
          items: [
            {
              type: 'link',
              label: 'Recipe index',
              href: '/models-recipes/model-recipes',
              autoAddBaseUrl: true,
            },
            ...taskRecipes.map((task) => ({
              type: 'category',
              label: task.label,
              collapsed: true,
              collapsible: true,
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
            })),
          ],
        },
      ],
    },
  ];
}

export default function RecipePageLayout({title, description, children}) {
  const {taskRecipes} = usePluginData('model-support-inventory');
  return (
    <Layout title={title} description={description}>
      <DocsSidebarProvider name="docs" items={recipeSidebarItems(taskRecipes)}>
        <DocRootLayout>{children}</DocRootLayout>
      </DocsSidebarProvider>
    </Layout>
  );
}
