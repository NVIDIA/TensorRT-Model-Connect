/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const {collectModelSupportInventory} = require('./index');

function repository() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'trtmc-inventory-'));
  for (const family of ['alpha', 'beta']) {
    const familyRoot = path.join(root, 'families', family);
    fs.mkdirSync(path.join(familyRoot, 'runtime'), {recursive: true});
    fs.mkdirSync(path.join(familyRoot, 'tests', 'manifests'), {recursive: true});
    fs.writeFileSync(path.join(familyRoot, 'support.py'), 'def describe(metadata):\n  return None\n');
    fs.writeFileSync(path.join(familyRoot, 'model.py'), 'def build(request, writer):\n  pass\n');
    fs.writeFileSync(path.join(familyRoot, 'runtime', 'CMakeLists.txt'), '# family owned\n');
  }
  fs.writeFileSync(
    path.join(root, 'families', 'alpha', 'tests', 'manifests', 'small.json'),
    JSON.stringify({
      name: 'alpha-small',
      hf_id: 'org/alpha',
      bundle: 'alpha-small.bundle',
      family: 'alpha',
      task: 'text_generation',
      precision: 'bf16',
      tensor_parallel_size: 1,
      testcases: [{name: 'generate'}],
    })
  );
  return root;
}

test('collects physical families and their owned manifests', () => {
  const root = repository();
  const inventory = collectModelSupportInventory(root);
  assert.equal(inventory.familyCount, 2);
  assert.deepEqual(inventory.familyNames, ['alpha', 'beta']);
  assert.equal(inventory.manifestCount, 1);
  assert.equal(inventory.modelProfiles[0].sourcePath,
    'families/alpha/tests/manifests/small.json');
  assert.equal(inventory.modelProfiles[0].task, 'text_generation');
  assert.equal(inventory.taskRecipes[0].families[0].family, 'alpha');
});

test('reports an explicit context-parallel profile', () => {
  const root = repository();
  const manifest = path.join(root, 'families', 'alpha', 'tests', 'manifests', 'small.json');
  const payload = JSON.parse(fs.readFileSync(manifest));
  payload.context_parallel_size = 2;
  fs.writeFileSync(manifest, JSON.stringify(payload));
  const [profile] = collectModelSupportInventory(root).modelProfiles;
  assert.equal(profile.parallelMode, 'context_parallel');
  assert.equal(profile.parallelSize, 2);
});

test('reports a family-owned prepared checkpoint without fake HF metadata', () => {
  const root = repository();
  const manifest = path.join(root, 'families', 'alpha', 'tests', 'manifests', 'small.json');
  const payload = JSON.parse(fs.readFileSync(manifest));
  delete payload.hf_id;
  fs.writeFileSync(manifest, JSON.stringify(payload));
  const [profile] = collectModelSupportInventory(root).modelProfiles;
  assert.equal(profile.hfId, 'prepared local checkpoint');
  assert.equal(profile.revision, 'not applicable');
});

test('reports an abstract task without inventing a CLI command', () => {
  const root = repository();
  const manifest = path.join(root, 'families', 'alpha', 'tests', 'manifests', 'small.json');
  const payload = JSON.parse(fs.readFileSync(manifest));
  payload.task = 'pose_hypothesis_refinement';
  fs.writeFileSync(manifest, JSON.stringify(payload));
  const [profile] = collectModelSupportInventory(root).modelProfiles;
  assert.deepEqual(profile.cliCommands, []);
});

test('rejects removed strategy metadata', () => {
  const root = repository();
  const manifest = path.join(root, 'families', 'alpha', 'tests', 'manifests', 'small.json');
  const payload = JSON.parse(fs.readFileSync(manifest));
  payload.runtime_strategy = 'legacy';
  fs.writeFileSync(manifest, JSON.stringify(payload));
  assert.throws(() => collectModelSupportInventory(root), /removed strategy field/);
});

test('rejects a manifest claimed by the wrong family', () => {
  const root = repository();
  const manifest = path.join(root, 'families', 'alpha', 'tests', 'manifests', 'small.json');
  const payload = JSON.parse(fs.readFileSync(manifest));
  payload.family = 'beta';
  fs.writeFileSync(manifest, JSON.stringify(payload));
  assert.throws(() => collectModelSupportInventory(root), /exact family/);
});

test('requires the complete family shape', () => {
  const root = repository();
  fs.rmSync(path.join(root, 'families', 'beta', 'runtime', 'CMakeLists.txt'));
  assert.throws(() => collectModelSupportInventory(root), /missing families\/beta\/runtime/);
});
