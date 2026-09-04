/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import {usePluginData} from '@docusaurus/useGlobalData';

function parallelLabel(profile) {
  if (profile.parallelMode === 'single_device') return 'Single device';
  return `${profile.parallelMode === 'tensor_parallel' ? 'TP' : 'CP'}${profile.parallelSize}`;
}

function ProfileTable({profiles}) {
  return (
    <table>
      <thead>
        <tr>
          <th>Checkpoint</th>
          <th>Family</th>
          <th>Task interface</th>
          <th>Build</th>
          <th>Owned manifest</th>
        </tr>
      </thead>
      <tbody>
        {profiles.map((profile) => (
          <tr key={profile.sourcePath}>
            <td><code>{profile.hfId}</code></td>
            <td><code>{profile.family}</code></td>
            <td><code>{profile.task}</code></td>
            <td>{profile.precision}; {parallelLabel(profile)}</td>
            <td><code>{profile.sourcePath}</code></td>
          </tr>
        ))}
      </tbody>
    </table>
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
    return <pre><code>{inventory.familyNames.join(', ')}</code></pre>;
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
