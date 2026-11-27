/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import Link from '@docusaurus/Link';
import Layout from '@theme/Layout';
import {usePluginData} from '@docusaurus/useGlobalData';
import RecipePageLayout from './RecipePageLayout';
import {FamilyParameters, SettingsTable, SourceVerification} from './ParameterDetails';
import styles from './styles.module.css';

export default function ModelCheckpointRecipePage({familySlug, profileSlug, details}) {
  const {familyRecipes, taskRecipes, sourceRevision} = usePluginData('model-support-inventory');
  const entry = familyRecipes.find((candidate) => candidate.slug === familySlug);
  const family = entry && {...entry, details: details || entry.details};
  const profile = family?.profiles.find((candidate) => candidate.slug === profileSlug);
  if (!profile) {
    return <Layout title="Checkpoint recipe not found"><main className="container margin-vert--lg"><h1>Checkpoint recipe not found</h1></main></Layout>;
  }
  const manifest = profile.manifest;
  const settings = Object.fromEntries(Object.entries(manifest).filter(([key]) => key !== 'testcases'));
  const taskBySlug = new Map(taskRecipes.map((task) => [task.slug, task]));
  const verification = family.details?.profileChecks?.[profile.sourcePath];
  const siblings = manifest.hf_id ? family.profiles.filter((candidate) => candidate.manifest.hf_id === manifest.hf_id && candidate.sourcePath !== profile.sourcePath) : [];
  return (
    <RecipePageLayout title={`${profile.profile} checkpoint recipe`} description={`Checkpoint configuration and family parameters for ${profile.hfId}.`}>
      <article className={styles.page}>
        <p><Link to="/models-recipes/overview">Supported models</Link>{' / '}<Link to={profile.familyUrl}>{family.family}</Link></p>
        <h1>{profile.profile}</h1>
        <p><strong>Checkpoint:</strong> <code>{profile.hfId}</code></p>
        <p><strong>Family:</strong> <Link to={profile.familyUrl}><code>{family.family}</code></Link>{' · '}
          <strong>Manifest:</strong> <a href={profile.sourceUrl}><code>{profile.sourcePath}</code></a></p>
        {sourceRevision && <p className={styles.muted}>Documentation source: <code>{sourceRevision}</code></p>}
        <nav aria-label="Checkpoint page sections" className={styles.sectionLinks}>
          <a href="#recipe-settings">Recipe settings</a>
          <a href="#verification">Configuration checks</a>
          <a href="#test-coverage">Test cases</a>
          <a href="#cli-parameters">CLI parameters</a>
          <a href="#build-parameters">Build request</a>
          <a href="#configuration-schema">Configuration schemas</a>
        </nav>
        <section id="recipe-settings">
          <h2>Declared recipe settings</h2>
          <p>Every top-level manifest setting is shown below exactly as declared. Omitted settings are not assigned invented defaults.</p>
          <SettingsTable settings={settings} />
        </section>
        <SourceVerification verification={verification} />
        <section id="test-coverage">
          <h2>Task interfaces and test cases</h2>
          <ul>{profile.taskSlugs.map((slug) => <li key={slug}>
            <Link to={`/models-recipes/model-recipes/tasks/${slug}`}>{taskBySlug.get(slug)?.label || slug}</Link>
          </li>)}</ul>
          <p>These are the test inputs and validation settings declared in the manifest.
            {' '}Request mapping checks use the CPU benchmark adapter without loading a model.
            {' '}A mapped request does not prove that the native runtime accepts its configuration.
            {' '}Missing adapters or fixture inputs leave the check incomplete; they do not establish that the model is unsupported.</p>
          {(manifest.testcases || []).map((testcase, index) => (
            <details key={`${testcase.name}-${index}`} className={styles.details}>
              <summary>{testcase.name || `Test case ${index + 1}`}</summary>
              <div className={styles.detailsBody}>
                {verification?.caseChecks?.filter((check) => check.name === testcase.name).map((check, checkIndex) => (
                  <p key={checkIndex}><strong>Benchmark request mapping: {check.status}</strong>{' — '}{check.detail}</p>
                ))}
                <SettingsTable settings={testcase} />
              </div>
            </details>
          ))}
          {!manifest.testcases?.length && <p>No test cases are declared in this manifest.</p>}
        </section>
        <FamilyParameters family={family} />
        {siblings.length > 0 && <section>
          <h2>Other recipes for this model</h2>
          <ul>{siblings.map((sibling) => <li key={sibling.sourcePath}><Link to={sibling.url}>{sibling.profile}</Link></li>)}</ul>
        </section>}
        <details className={styles.details}>
          <summary>Complete source manifest (JSON)</summary>
          <div className={styles.detailsBody}><pre><code>{JSON.stringify(manifest, null, 2)}</code></pre></div>
        </details>
      </article>
    </RecipePageLayout>
  );
}
