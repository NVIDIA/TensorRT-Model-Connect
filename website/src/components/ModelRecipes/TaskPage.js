/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import Link from '@docusaurus/Link';
import Layout from '@theme/Layout';
import {usePluginData} from '@docusaurus/useGlobalData';
import RecipePageLayout from './RecipePageLayout';

export default function ModelTaskRecipePage({taskSlug}) {
  const {taskRecipes} = usePluginData('model-support-inventory');
  const task = taskRecipes.find((candidate) => candidate.slug === taskSlug);

  if (!task) {
    return <Layout title="Model task not found"><main className="container margin-vert--lg"><h1>Model task not found</h1></main></Layout>;
  }

  return (
    <RecipePageLayout title={`${task.label} model recipes`} description={task.description}>
      <article>
        <p><Link to="/models-recipes/model-recipes">← All model recipe tasks</Link></p>
        <h1>{task.label}</h1>
        <p>{task.description}</p>
        <p>
          Classification follows the Hugging Face task taxonomy. See the{' '}
          <a href={task.hfUrl}>Hugging Face {task.label} task page</a> for the
          ecosystem-level task definition.
        </p>

        <h2>Model families</h2>
        <table>
          <thead>
            <tr>
              <th>Model family</th>
              <th>Declared recipes</th>
              <th>Exact checkpoint examples</th>
              <th>Runtime CLI</th>
            </tr>
          </thead>
          <tbody>
            {task.families.map((family) => (
              <tr key={family.family}>
                <td>
                  <Link to={`/models-recipes/model-recipes/families/${family.slug}`}>
                    <code>{family.family}</code>
                  </Link>
                </td>
                <td>{family.recipeCount}</td>
                <td>{family.hfIds.map((hfId) => <div key={hfId}><code>{hfId}</code></div>)}</td>
                <td>{family.cliCommands.map((command) => <div key={command}><code>trtmc {command}</code></div>)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </article>
    </RecipePageLayout>
  );
}
