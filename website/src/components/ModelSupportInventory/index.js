/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import {usePluginData} from '@docusaurus/useGlobalData';

export default function ModelSupportInventory({variant = 'summary'}) {
  const inventory = usePluginData('model-support-inventory');

  if (variant === 'facts') {
    return (
      <ul>
        <li>
          {inventory.familyPluginCount} Python family plugins under{' '}
          <code>python/tensorrt_model_connect/families/</code>.
        </li>
        <li>
          {inventory.e2eManifestCount} E2E model manifests and{' '}
          {inventory.e2eFamilyIndexCount} family indexes under{' '}
          <code>tests/e2e/models/</code>.
        </li>
        <li>
          {inventory.runtimeStrategyKeyCount} unique C++ runtime strategy keys
          declared by model metadata under <code>src/runtime/models/</code>.
        </li>
      </ul>
    );
  }

  if (variant === 'families') {
    return (
      <pre>
        <code>{inventory.familyPluginNames.join(', ')}</code>
      </pre>
    );
  }

  return (
    <p>
      The current checkout contains {inventory.familyPluginCount} Python family
      plugins, {inventory.e2eManifestCount} E2E model manifests, and{' '}
      {inventory.e2eFamilyIndexCount} E2E family indexes.
    </p>
  );
}
