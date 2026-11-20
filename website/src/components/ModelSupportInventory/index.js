/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import React, {useState} from 'react';
import Link from '@docusaurus/Link';
import {usePluginData} from '@docusaurus/useGlobalData';
import styles from '../ModelRecipes/styles.module.css';

function parallelLabel(profile) {
  if (profile.parallelMode === 'single_device') return 'Single device';
  return `${profile.parallelMode === 'tensor_parallel' ? 'TP' : 'CP'}${profile.parallelSize}`;
}

function ProfileTable({profiles}) {
  const [query, setQuery] = useState('');
  const normalizedQuery = query.trim().toLowerCase().replaceAll('_', ' ').replaceAll('-', ' ');
  const filtered = profiles.filter((profile) =>
    [profile.hfId, profile.family, profile.profile, ...profile.tasks, profile.precision, profile.quantization]
      .join(' ').toLowerCase().replaceAll('_', ' ').replaceAll('-', ' ').includes(normalizedQuery));
  return (
    <>
      <label>
        Filter checkpoints, families, tasks, or precision
        <input type="search" value={query} onChange={(event) => setQuery(event.target.value)} className={styles.search} placeholder="For example: bert, text generation, or fp16" />
      </label>
      <p aria-live="polite">{filtered.length} of {profiles.length} checkpoint recipes</p>
      <div className={styles.tableScroll}>
    <table>
      <thead>
        <tr>
          <th>Checkpoint</th>
          <th>Family</th>
          <th>Task interface</th>
          <th>Build</th>
          <th>Source manifest</th>
        </tr>
      </thead>
      <tbody>
        {filtered.map((profile) => (
          <tr key={profile.sourcePath}>
            <td><Link to={profile.url}><code>{profile.hfId}</code></Link><small className={styles.note}>{profile.profile}</small></td>
            <td><Link to={profile.familyUrl}><code>{profile.family}</code></Link></td>
            <td>{profile.tasks.map((task, index) => <div key={task}><Link to={`/models-recipes/model-recipes/tasks/${profile.taskSlugs[index]}`}><code>{task}</code></Link></div>)}</td>
            <td>{profile.precision}; {parallelLabel(profile)}{profile.quantization !== 'none' && <>; <code>{profile.quantization}</code></>}</td>
            <td><a href={profile.sourceUrl}><code>{profile.sourcePath}</code></a></td>
          </tr>
        ))}
      </tbody>
    </table>
      </div>
      {!filtered.length && <p>No recipes match this filter.</p>}
    </>
  );
}

export default function ModelSupportInventory({variant = 'summary'}) {
  const inventory = usePluginData('model-support-inventory');
  if (variant === 'facts') {
    return (
      <ul>
        <li>{inventory.familyCount} self-contained family modules under <code>families/</code>.</li>
        <li>{inventory.manifestCount} family-owned E2E manifests under <code>families/*/tests/manifests/</code>.</li>
        <li>Every runtime DSO implements an abstract task interface directly.</li>
      </ul>
    );
  }
  if (variant === 'families') {
    return <ul className={styles.familyList}>{inventory.familyRecipes.map((family) => (
      <li key={family.family}><Link to={`/models-recipes/model-recipes/families/${family.slug}`}><code>{family.family}</code></Link>{' '}({family.profiles.length} recipes)</li>
    ))}</ul>;
  }
  if (variant === 'models' || variant === 'performance') {
    return <ProfileTable profiles={inventory.modelProfiles} />;
  }
  return (
    <p>
      The current checkout contains {inventory.familyCount} self-contained families and{' '}
      {inventory.manifestCount} family-owned E2E manifests.
    </p>
  );
}
