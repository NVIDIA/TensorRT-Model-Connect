/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import Link from '@docusaurus/Link';
import Layout from '@theme/Layout';
import {usePluginData} from '@docusaurus/useGlobalData';
import RecipePageLayout from './RecipePageLayout';
import styles from './styles.module.css';

export default function ModelTaskRecipePage({taskSlug}) {
  const {taskRecipes, familyRecipes} = usePluginData('model-support-inventory');
  const task = taskRecipes.find((candidate) => candidate.slug === taskSlug);

  if (!task) {
    return <Layout title="Model task not found"><main className="container margin-vert--lg"><h1>Model task not found</h1></main></Layout>;
  }

  return (
    <RecipePageLayout title={`${task.label} model recipes`} description={task.description}>
      <article className={styles.page}>
        <p><Link to="/models-recipes/model-recipes">← All model recipe tasks</Link></p>
        <h1>{task.label}</h1>
        <p>{task.description}</p>
        {task.hfUrl && (
          <p>
            See the <a href={task.hfUrl}>related Hugging Face task page</a> for the
            ecosystem-level task definition.
          </p>
        )}

        <h2>Model families</h2>
        <div className={styles.tableScroll}><table>
          <thead>
            <tr>
              <th>Model family</th>
              <th>Declared recipes</th>
              <th>Exact checkpoint examples</th>
              <th>Family native CLI</th>
            </tr>
          </thead>
          <tbody>
            {task.families.map((family) => {
              const owner = familyRecipes.find((candidate) => candidate.family === family.family);
              const commands = owner?.declaredCliCommands || [];
              return <tr key={family.family}>
                <td>
                  <Link to={`/models-recipes/model-recipes/families/${family.slug}`}>
                    <code>{family.family}</code>
                  </Link>
                </td>
                <td>{family.recipeCount}</td>
                <td>{(owner?.profiles.filter((profile) => profile.tasks.includes(task.task)) || []).map((profile) => (
                  <div key={profile.sourcePath}><Link to={profile.url}><code>{profile.hfId}</code></Link><small className={styles.note}>{profile.profile}</small></div>
                ))}</td>
                <td>{commands.length ? commands.map((command) => <div key={command}><code>trtmc {family.family} {command}</code></div>) : 'No native family CLI declared'}</td>
              </tr>;
            })}
          </tbody>
        </table></div>
      </article>
    </RecipePageLayout>
  );
}
