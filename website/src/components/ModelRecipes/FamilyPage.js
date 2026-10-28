/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import Link from '@docusaurus/Link';
import Layout from '@theme/Layout';
import {usePluginData} from '@docusaurus/useGlobalData';
import RecipePageLayout from './RecipePageLayout';
import {FamilyParameters} from './ParameterDetails';
import styles from './styles.module.css';

function parallelLabel(profile) {
  if (profile.parallelMode === 'single_device') return 'single device';
  return `${profile.parallelMode === 'tensor_parallel' ? 'TP' : 'CP'}${profile.parallelSize}`;
}

export default function ModelFamilyRecipePage({familySlug, details}) {
  const {familyRecipes, taskRecipes} = usePluginData('model-support-inventory');
  const entry = familyRecipes.find((candidate) => candidate.slug === familySlug);
  if (!entry) {
    return (
      <Layout title="Model family not found">
        <main className="container margin-vert--lg"><h1>Model family not found</h1></main>
      </Layout>
    );
  }
  const family = {...entry, details: details || entry.details};
  const taskBySlug = new Map(taskRecipes.map((task) => [task.slug, task]));
  return (
    <RecipePageLayout
      title={`${family.family} model recipes`}
      description={`Family-owned recipes for ${family.family}.`}
    >
      <article className={styles.page}>
        <p><Link to="/models-recipes/overview">← Supported models</Link></p>
        <h1><code>{family.family}</code></h1>
        <p>
          Explore this family's checkpoint recipes, build parameters, configuration schemas,
          and runtime commands. The information below comes from the current family source.
        </p>
        <nav aria-label="Family page sections" className={styles.sectionLinks}>
          <a href="#checkpoint-recipes">Checkpoint recipes</a>
          <a href="#cli-parameters">CLI parameters</a>
          <a href="#build-parameters">Build request</a>
          <a href="#configuration-schema">Configuration schemas</a>
          <a href="#configuration-constraints">Restrictions</a>
        </nav>
        <h2>Task interfaces</h2>
        {!family.taskSlugs.length && <p>No task interfaces are declared by an E2E manifest in this family.</p>}
        <ul>
          {family.taskSlugs.map((slug) => (
            <li key={slug}>
              <Link to={`/models-recipes/model-recipes/tasks/${slug}`}>
                {taskBySlug.get(slug)?.label || slug}
              </Link>
            </li>
          ))}
        </ul>
        <h2 id="checkpoint-recipes">Checkpoint recipes</h2>
        <p>Select a recipe to inspect its exact checkpoint revision, build settings, test cases, and configuration check results.
          {' '}A declared recipe is not evidence of a successful GPU run.</p>
        {!family.profiles.length && <p>This family has no E2E checkpoint manifest in the current checkout.</p>}
        {family.profiles.length > 0 && <div className={styles.tableScroll}>
        <table>
          <thead>
            <tr>
              <th>Recipe</th>
              <th>Checkpoint</th>
              <th>Task interfaces (primary first)</th>
              <th>Build</th>
              <th>Build input checks</th>
            </tr>
          </thead>
          <tbody>
            {family.profiles.map((profile) => (
              <tr key={profile.sourcePath}>
                <td><Link to={profile.url}><code>{profile.profile}</code></Link></td>
                <td><Link to={profile.url}><code>{profile.hfId}</code></Link></td>
                <td>{profile.tasks.map((task) => <div key={task}><code>{task}</code></div>)}</td>
                <td>{profile.precision}; {parallelLabel(profile)}</td>
                <td>{family.details?.profileChecks?.[profile.sourcePath]?.status || 'not-checked'}</td>
              </tr>
            ))}
          </tbody>
        </table>
        </div>}
        <FamilyParameters family={family} />
      </article>
    </RecipePageLayout>
  );
}
