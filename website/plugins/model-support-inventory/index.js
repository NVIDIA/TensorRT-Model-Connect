/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

const fs = require('fs');
const path = require('path');

function readDirectory(directory, label) {
  let entries;
  try {
    entries = fs.readdirSync(directory, {withFileTypes: true});
  } catch (error) {
    throw new Error(`Unable to read ${label} at ${directory}: ${error.message}`);
  }
  return entries;
}

function collectModelSupportInventory(repoRoot) {
  const familiesDirectory = path.join(
    repoRoot,
    'python',
    'tensorrt_model_connect',
    'families'
  );
  const familyEntries = readDirectory(familiesDirectory, 'Python family plugins');
  const flatFamilies = familyEntries
    .filter(
      (entry) =>
        entry.isFile() &&
        entry.name.endsWith('.py') &&
        !entry.name.startsWith('_') &&
        entry.name !== 'base.py'
    )
    .map((entry) => path.basename(entry.name, '.py'));
  const packageFamilies = familyEntries
    .filter(
      (entry) =>
        entry.isDirectory() &&
        !entry.name.startsWith('_') &&
        fs.existsSync(path.join(familiesDirectory, entry.name, 'plugin.py'))
    )
    .map((entry) => entry.name);
  const familyPluginNames = [...new Set([...flatFamilies, ...packageFamilies])].sort();

  const e2eModelsDirectory = path.join(repoRoot, 'tests', 'e2e', 'models');
  const e2eEntries = readDirectory(e2eModelsDirectory, 'E2E model metadata');
  let e2eManifestCount = e2eEntries.filter(
    (entry) => entry.isFile() && entry.name.endsWith('.json')
  ).length;
  let e2eFamilyIndexCount = 0;
  for (const entry of e2eEntries.filter((candidate) => candidate.isDirectory())) {
    const familyDirectory = path.join(e2eModelsDirectory, entry.name);
    if (fs.existsSync(path.join(familyDirectory, 'MODEL.toml'))) {
      e2eFamilyIndexCount += 1;
    }
    const manifestsDirectory = path.join(familyDirectory, 'manifests');
    if (fs.existsSync(manifestsDirectory)) {
      e2eManifestCount += readDirectory(
        manifestsDirectory,
        `E2E manifests for ${entry.name}`
      ).filter(
        (candidate) => candidate.isFile() && candidate.name.endsWith('.json')
      ).length;
    }
  }

  const runtimeModelsDirectory = path.join(repoRoot, 'src', 'runtime', 'models');
  const runtimeStrategyKeys = new Set();
  const strategyArray = /^runtime_strategies\s*=\s*\[(.*?)\]/gms;
  const quotedValue = /"([^"]+)"/g;
  for (const entry of readDirectory(runtimeModelsDirectory, 'runtime model metadata')) {
    if (!entry.isDirectory()) {
      continue;
    }
    const metadataPath = path.join(runtimeModelsDirectory, entry.name, 'MODEL.toml');
    if (!fs.existsSync(metadataPath)) {
      continue;
    }
    const metadata = fs.readFileSync(metadataPath, 'utf8');
    for (const arrayMatch of metadata.matchAll(strategyArray)) {
      for (const valueMatch of arrayMatch[1].matchAll(quotedValue)) {
        runtimeStrategyKeys.add(valueMatch[1]);
      }
    }
  }

  return {
    familyPluginCount: familyPluginNames.length,
    familyPluginNames,
    e2eManifestCount,
    e2eFamilyIndexCount,
    runtimeStrategyKeyCount: runtimeStrategyKeys.size,
  };
}

function modelSupportInventoryPlugin(context) {
  return {
    name: 'model-support-inventory',
    loadContent() {
      return collectModelSupportInventory(path.resolve(context.siteDir, '..'));
    },
    contentLoaded({content, actions}) {
      actions.setGlobalData(content);
    },
  };
}

module.exports = modelSupportInventoryPlugin;
module.exports.collectModelSupportInventory = collectModelSupportInventory;
