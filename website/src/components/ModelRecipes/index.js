/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import Link from '@docusaurus/Link';
import {usePluginData} from '@docusaurus/useGlobalData';
import styles from './styles.module.css';

const CATEGORY_ORDER = [
  'Multimodal',
  'Natural Language Processing',
  'Computer Vision',
  'Audio',
  'Time Series',
  'Biology',
  'Molecular Modeling',
  'Robotics',
];

export default function ModelRecipeTaskIndex() {
  const {taskRecipes, familyRecipes} = usePluginData('model-support-inventory');
  const categories = new Map();
  for (const task of taskRecipes) {
    if (!categories.has(task.category)) categories.set(task.category, []);
    categories.get(task.category).push(task);
  }

  return (
    <>
      <section>
        <h2>All model families</h2>
        <p>Open a family to view its checkpoint recipes, configuration schemas, and command parameters.</p>
        <ul className={styles.familyList}>{familyRecipes.map((family) => <li key={family.family}>
          <Link to={`/models-recipes/model-recipes/families/${family.slug}`}><code>{family.family}</code></Link>{' '}({family.profiles.length} recipes)
        </li>)}</ul>
      </section>
      {[...CATEGORY_ORDER, ...[...categories.keys()].filter((category) => !CATEGORY_ORDER.includes(category)).sort()].filter((category) => categories.has(category)).map((category) => (
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
