/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const {
  collectModelSupportInventory,
} = require('./index');

function writeFixture(repoRoot, relativePath, content = '') {
  const fixturePath = path.join(repoRoot, relativePath);
  fs.mkdirSync(path.dirname(fixturePath), {recursive: true});
  fs.writeFileSync(fixturePath, content, 'utf8');
}

test('collects support inventory from repository metadata', (context) => {
  const repoRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'trtmc-model-support-'));
  context.after(() => fs.rmSync(repoRoot, {recursive: true, force: true}));

  writeFixture(
    repoRoot,
    'python/tensorrt_model_connect/families/alpha.py'
  );
  writeFixture(
    repoRoot,
    'python/tensorrt_model_connect/families/beta/plugin.py'
  );
  writeFixture(
    repoRoot,
    'python/tensorrt_model_connect/families/_private.py'
  );
  writeFixture(
    repoRoot,
    'python/tensorrt_model_connect/families/base.py'
  );
  writeFixture(repoRoot, 'tests/e2e/models/legacy.json', '{}');
  writeFixture(repoRoot, 'tests/e2e/models/alpha/MODEL.toml');
  writeFixture(repoRoot, 'tests/e2e/models/alpha/manifests/small.json', '{}');
  writeFixture(repoRoot, 'tests/e2e/models/beta/MODEL.toml');
  writeFixture(repoRoot, 'tests/e2e/models/beta/manifests/base.json', '{}');
  writeFixture(
    repoRoot,
    'src/runtime/models/alpha/MODEL.toml',
    'runtime_strategies = ["decoder", "shared"]\n'
  );
  writeFixture(
    repoRoot,
    'src/runtime/models/beta/MODEL.toml',
    'runtime_strategies = [\n  "shared",\n  "encoder",\n]\n'
  );

  assert.deepEqual(collectModelSupportInventory(repoRoot), {
    familyPluginCount: 2,
    familyPluginNames: ['alpha', 'beta'],
    e2eManifestCount: 3,
    e2eFamilyIndexCount: 2,
    runtimeStrategyKeyCount: 3,
  });
});

test('fails closed when a source-of-truth directory is missing', (context) => {
  const repoRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'trtmc-model-support-'));
  context.after(() => fs.rmSync(repoRoot, {recursive: true, force: true}));

  assert.throws(
    () => collectModelSupportInventory(repoRoot),
    /Unable to read Python family plugins/
  );
});
