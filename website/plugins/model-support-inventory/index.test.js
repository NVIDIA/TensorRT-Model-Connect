/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const modelSupportInventoryPlugin = require('./index');
const {collectModelSupportInventory} = modelSupportInventoryPlugin;

test('head-score recipes are discovered without hidden or vocabulary claims', () => {
  const root = repository();
  addTaskManifest(root, 'text_to_head_scores');
  const recipe = collectModelSupportInventory(root).taskRecipes.find(row => row.task === 'text_to_head_scores');
  assert.equal(recipe.label, 'Text head scores');
  assert.equal(recipe.recipeCount, 1);
  assert.deepEqual(recipe.families[0].cliCommands, ['encode']);
  assert.equal(recipe.hfUrl, null);
});

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

function addTaskManifest(root, task, family = 'alpha') {
  fs.writeFileSync(
    path.join(root, 'families', family, 'tests', 'manifests', `${task}.json`),
    JSON.stringify({
      name: `${family}-${task}`,
      bundle: `${family}-${task}.bundle`,
      family,
      task,
      precision: 'fp32',
      tensor_parallel_size: 1,
      testcases: [{name: `${family}-${task}`}],
    })
  );
}

test('family testcase Tasks share one physical profile without mixing Task commands', (context) => {
  const root = repository();
  context.after(() => fs.rmSync(root, {recursive: true, force: true}));
  const file = path.join(root, 'families/alpha/tests/manifests/small.json');
  const payload = JSON.parse(fs.readFileSync(file));
  payload.task = 'text_to_pooled_features';
  payload.testcases = [
    {name: 'default'},
    {name: 'tokens', selected_task: 'text_to_token_features'},
    {name: 'tokens-again', selected_task: 'text_to_token_features'},
    {name: 'embedding', selected_task: 'text_to_embedding'},
  ];
  fs.writeFileSync(file, JSON.stringify(payload));
  const inventory = collectModelSupportInventory(root);
  assert.equal(inventory.manifestCount, 1);
  const [profile] = inventory.modelProfiles;
  assert.equal(profile.task, 'text_to_pooled_features');
  assert.deepEqual(profile.tasks, ['text_to_pooled_features', 'text_to_token_features', 'text_to_embedding']);
  assert.equal(inventory.taskRecipes.length, 3);
  for (const recipe of inventory.taskRecipes) {
    assert.equal(recipe.recipeCount, 1);
    assert.deepEqual(recipe.families[0].cliCommands,
      [recipe.task === 'text_to_embedding' ? 'embed' : 'encode']);
  }
  assert.deepEqual(profile.cliCommands, ['embed', 'encode']);
  for (const selected_task of [null, '', 'unknown_task', false]) {
    payload.testcases = [{name: 'invalid', selected_task}];
    fs.writeFileSync(file, JSON.stringify(payload));
    assert.throws(() => collectModelSupportInventory(root), /unknown selected_task/);
  }
});


function semanticTaskIds() {
  const headers = path.resolve(__dirname, '../../../core/api/include/trtmc');
  return fs.readdirSync(headers).filter((name) => name.endsWith('.h')).flatMap((name) => {
    const source = fs.readFileSync(path.join(headers, name), 'utf8').replace(/\\\r?\n/g, '');
    return [...source.matchAll(/^#define\s+TRTMC_TASK_\w+\s+"([a-z][a-z0-9_]*)"/gm)]
      .map((match) => match[1]);
  }).sort();
}

test('every public semantic Task can generate a family recipe and route', (context) => {
  const tasks = semanticTaskIds();
  assert.ok(tasks.includes('text_continuation'));
  assert.ok(tasks.includes('batch_initial_image_text_to_audio_video'));
  assert.equal(new Set(tasks).size, tasks.length);
  const root = repository();
  context.after(() => fs.rmSync(root, {recursive: true, force: true}));
  for (const task of tasks) addTaskManifest(root, task);

  const inventory = collectModelSupportInventory(root);
  assert.equal(inventory.familyCount, 2);
  assert.equal(inventory.manifestCount, tasks.length + 1);
  assert.deepEqual(inventory.taskRecipes.map((recipe) => recipe.task).sort(),
    [...tasks, 'text_generation'].sort());
  const routes = [];
  const plugin = modelSupportInventoryPlugin({siteDir: path.join(root, 'website'), baseUrl: '/docs/'});
  plugin.contentLoaded({content: inventory, actions: {
    setGlobalData: (data) => assert.equal(data, inventory),
    addRoute: (route) => routes.push(route),
  }});
  assert.equal(routes.length, tasks.length + 1 + inventory.familyCount);
  for (const task of tasks) {
    const recipe = inventory.taskRecipes.find((item) => item.task === task);
    assert.ok(recipe.label.length > 0, task);
    assert.ok(recipe.category.length > 0, task);
    assert.equal(recipe.recipeCount, 1, task);
    assert.equal(recipe.families[0].family, 'alpha', task);
    assert.ok(routes.some((route) => route.path ===
      `/docs/models-recipes/model-recipes/tasks/${task.replaceAll('_', '-')}`), task);
  }
});

test('family-only semantic switches preserve identity and use implemented commands', (context) => {
  const root = repository();
  context.after(() => fs.rmSync(root, {recursive: true, force: true}));
  const manifest = path.join(root, 'families', 'alpha', 'tests', 'manifests', 'small.json');
  const payload = JSON.parse(fs.readFileSync(manifest));
  for (const [task, command] of [
    ['text_translation', 'run'],
    ['series_to_point_forecast', 'forecast'],
    ['series_to_regression_values', 'forecast'],
    ['image_to_class_scores', 'classify'],
    ['images_text_to_text', 'run'],
    ['text_to_speech', 'generate-audio'],
    ['batch_speech_translation', 'transcribe-batch'],
    ['streaming_speech_transcription', 'transcribe-streaming'],
    ['duplex_speech_dialogue', 'speech-session'],
    ['masked_image_text_to_image', 'generate-image'],
    ['batch_text_to_image', 'generate-image-batch'],
    ['initial_image_text_to_video', 'generate-video'],
    ['image_text_to_instance_masks', 'segment-prompted'],
    ['frames_text_to_mask_tracks', 'video-segment'],
    ['image_state_to_action_chunk', 'control'],
    ['latent_denoising_step', 'solve'],
  ]) {
    fs.writeFileSync(manifest, JSON.stringify({...payload, task}));
    const inventory = collectModelSupportInventory(root);
    const [profile] = inventory.modelProfiles;
    assert.equal(profile.profile, payload.name, task);
    assert.equal(profile.family, payload.family, task);
    assert.equal(profile.hfId, payload.hf_id, task);
    assert.deepEqual(profile.cliCommands, [command], task);
    assert.deepEqual(inventory.familyRecipes[0].cliCommands, [command], task);
  }
});

test('SDK-only Tasks do not invent CLI workloads or taxonomy links', (context) => {
  const root = repository();
  context.after(() => fs.rmSync(root, {recursive: true, force: true}));
  for (const task of [
    'image_state_action_queue',
    'pose_hypotheses_crops_to_refined_poses',
    'batch_series_to_point_forecast',
    'recurrent_tokens_to_logits',
    'text_to_audio_video',
  ]) addTaskManifest(root, task);
  const inventory = collectModelSupportInventory(root);
  for (const recipe of inventory.taskRecipes.filter((item) => item.task !== 'text_generation')) {
    assert.equal(recipe.hfUrl, null, recipe.task);
    assert.deepEqual(recipe.families[0].cliCommands, [], recipe.task);
  }
  addTaskManifest(root, 'text_translation', 'beta');
  const translation = collectModelSupportInventory(root).taskRecipes
    .find((recipe) => recipe.task === 'text_translation');
  assert.equal(translation.hfUrl, 'https://huggingface.co/tasks/translation');
});

test('keeps existing detector and structure-prediction manifests supported', (context) => {
  const root = repository();
  context.after(() => fs.rmSync(root, {recursive: true, force: true}));
  addTaskManifest(root, 'object_detection');
  addTaskManifest(root, 'structure_prediction', 'beta');
  const inventory = collectModelSupportInventory(root);
  assert.deepEqual(inventory.modelProfiles.find((profile) => profile.task === 'object_detection')
    .cliCommands, ['detect']);
  assert.deepEqual(inventory.modelProfiles.find((profile) => profile.task === 'structure_prediction')
    .cliCommands, []);
});

test('rejects unknown Tasks including inherited object property names', (context) => {
  const root = repository();
  context.after(() => fs.rmSync(root, {recursive: true, force: true}));
  const manifest = path.join(root, 'families', 'alpha', 'tests', 'manifests', 'small.json');
  const payload = JSON.parse(fs.readFileSync(manifest));
  for (const task of ['text_translatoin', 'text_to_mesh_not_declared', 'constructor', '__proto__']) {
    fs.writeFileSync(manifest, JSON.stringify({...payload, task}));
    assert.throws(() => collectModelSupportInventory(root), /declares unknown task/, task);
  }
});

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
