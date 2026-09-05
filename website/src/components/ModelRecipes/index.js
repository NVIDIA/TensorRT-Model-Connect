/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import Link from '@docusaurus/Link';
import {usePluginData} from '@docusaurus/useGlobalData';

const CATEGORY_ORDER = [
  'Multimodal',
  'Natural Language Processing',
  'Computer Vision',
  'Audio',
  'Time Series',
  'Biology',
];

export default function ModelRecipeTaskIndex() {
  const {taskRecipes} = usePluginData('model-support-inventory');
  const categories = new Map();
  for (const task of taskRecipes) {
    if (!categories.has(task.category)) categories.set(task.category, []);
    categories.get(task.category).push(task);
  }

  return (
    <>
      {CATEGORY_ORDER.filter((category) => categories.has(category)).map((category) => (
        <section key={category}>
          <h2>{category}</h2>
          <table>
            <thead>
              <tr>
                <th>Hugging Face task</th>
                <th>Model families</th>
                <th>Declared recipes</th>
                <th>What it covers</th>
              </tr>
            </thead>
            <tbody>
              {categories.get(category).map((task) => (
                <tr key={task.slug}>
                  <td>
                    <Link to={`/models-recipes/model-recipes/tasks/${task.slug}`}>
                      {task.label}
                    </Link>
                  </td>
                  <td>{task.families.length}</td>
                  <td>{task.recipeCount}</td>
                  <td>{task.description}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      ))}
    </>
  );
}
