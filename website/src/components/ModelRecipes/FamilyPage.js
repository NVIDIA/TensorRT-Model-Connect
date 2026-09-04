/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import Link from '@docusaurus/Link';
import Layout from '@theme/Layout';
import {usePluginData} from '@docusaurus/useGlobalData';
import RecipePageLayout from './RecipePageLayout';

function parallelLabel(profile) {
  if (profile.parallelMode === 'single_device') return 'single device';
  return `${profile.parallelMode === 'tensor_parallel' ? 'TP' : 'CP'}${profile.parallelSize}`;
}

export default function ModelFamilyRecipePage({familySlug}) {
  const {familyRecipes, taskRecipes} = usePluginData('model-support-inventory');
  const family = familyRecipes.find((candidate) => candidate.slug === familySlug);
  if (!family) {
    return (
      <Layout title="Model family not found">
        <main className="container margin-vert--lg"><h1>Model family not found</h1></main>
      </Layout>
    );
  }
  const taskBySlug = new Map(taskRecipes.map((task) => [task.slug, task]));
  return (
    <RecipePageLayout
      title={`${family.family} model recipes`}
      description={`Family-owned recipes for ${family.family}.`}
    >
      <article>
        <p><Link to="/models-recipes/model-recipes">← All model recipe tasks</Link></p>
        <h1><code>{family.family}</code></h1>
        <p>
          This directory owns its Python builder, native runtime DSO, bundle section semantics,
          and tests. It has no source dependency on another family.
        </p>
        <h2>Task interfaces</h2>
        <ul>
          {family.taskSlugs.map((slug) => (
            <li key={slug}>
              <Link to={`/models-recipes/model-recipes/tasks/${slug}`}>
                {taskBySlug.get(slug)?.label || slug}
              </Link>
            </li>
          ))}
        </ul>
        <h2>Owned recipes</h2>
        <table>
          <thead>
            <tr>
              <th>Recipe</th>
              <th>Checkpoint</th>
              <th>Abstract task</th>
              <th>Build</th>
              <th>Manifest</th>
            </tr>
          </thead>
          <tbody>
            {family.profiles.map((profile) => (
              <tr key={profile.sourcePath}>
                <td><code>{profile.profile}</code></td>
                <td><code>{profile.hfId}</code></td>
                <td><code>{profile.task}</code></td>
                <td>{profile.precision}; {parallelLabel(profile)}</td>
                <td><code>{profile.sourcePath}</code></td>
              </tr>
            ))}
          </tbody>
        </table>
      </article>
    </RecipePageLayout>
  );
}
