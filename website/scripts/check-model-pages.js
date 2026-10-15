/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Check rendered output, so a successful compile cannot hide lost model links
// or configuration sections. No browser, GPU, or remote checkpoint is needed.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {collectModelSupportInventory} = require('../plugins/model-support-inventory');
const config = require('../docusaurus.config');

const siteDir = path.resolve(__dirname, '..');
const inventory = collectModelSupportInventory(path.dirname(siteDir));
const baseUrl = config.baseUrl.replace(/\/$/, '');
const buildDir = path.join(siteDir, 'build');

function readPage(route) {
  const relative = route.replace(/^\//, '');
  for (const file of [path.join(buildDir, relative, 'index.html'), path.join(buildDir, `${relative}.html`)]) {
    if (fs.existsSync(file)) return fs.readFileSync(file, 'utf8');
  }
  assert.fail(`Missing generated model page: ${route}`);
}

function links(html) {
  return new Set(Array.from(html.matchAll(/<a\b[^>]*\bhref="([^"]+)"/g),
    (match) => match[1].replace(/\/$/, '')));
}

function hasLink(anchors, route, context) {
  assert.ok(anchors.has(`${baseUrl}${route}`), `${context}: missing link to ${route}`);
}

function hasSection(html, id, context) {
  assert.ok(html.includes(`id="${id}"`), `${context}: missing ${id} section`);
}

function escapeHtml(value) {
  return String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;').replaceAll('"', '&quot;').replaceAll("'", '&#x27;');
}

const overview = links(readPage('/models-recipes/overview'));
const recipeIndex = links(readPage('/models-recipes/model-recipes'));
for (const family of inventory.familyRecipes) {
  hasLink(overview, family.url, 'Supported Models');
  hasLink(recipeIndex, family.url, 'Model Recipes');
  const html = readPage(family.url);
  const familyLinks = links(html);
  for (const id of ['cli-parameters', 'build-parameters', 'configuration-schema']) {
    hasSection(html, id, family.family);
  }
  const details = JSON.parse(fs.readFileSync(path.join(siteDir, '.docusaurus',
    'model-support-inventory', 'default', `family-${family.slug}.json`), 'utf8'));
  for (const schema of details.configClasses) {
    assert.ok(html.includes(escapeHtml(schema.name)), `${family.family}: missing schema ${schema.name}`);
    for (const field of schema.fields) {
      assert.ok(html.includes(`<code>${escapeHtml(field.name)}</code>`),
        `${family.family}: missing field ${schema.name}.${field.name}`);
      for (const value of [field.type, field.default].filter((value) => value !== null)) {
        assert.ok(html.includes(escapeHtml(value)), `${family.family}: missing declaration value ${value}`);
      }
    }
    if (schema.declaration) {
      assert.ok(html.includes(escapeHtml(schema.declaration)),
        `${family.family}: missing complete declaration for ${schema.name}`);
    }
  }
  for (const command of details.cli.commands) {
    for (const argument of command.arguments) {
      for (const flag of argument.flags || []) {
        assert.ok(html.includes(escapeHtml(flag)), `${family.family}: missing CLI flag ${flag}`);
      }
    }
  }
  for (const profile of family.profiles) {
    hasLink(overview, profile.url, 'Supported Models');
    hasLink(familyLinks, profile.url, family.family);
    const checkpoint = readPage(profile.url);
    hasLink(links(checkpoint), family.url, profile.profile);
    for (const id of ['recipe-settings', 'verification', 'test-coverage', 'cli-parameters',
      'build-parameters', 'configuration-schema']) {
      hasSection(checkpoint, id, profile.profile);
    }
    assert.ok(checkpoint.includes(profile.sourcePath), `${profile.profile}: missing exact manifest source`);
    // Metadata must remain visible even when no generic CLI flag exists for it.
    for (const key of Object.keys(profile.manifest).filter((key) => key !== 'testcases')) {
      assert.ok(checkpoint.includes(`<code>${key}</code>`), `${profile.profile}: lost manifest setting ${key}`);
    }
    for (const check of details.profileChecks[profile.sourcePath].caseChecks || []) {
      assert.ok(checkpoint.includes(escapeHtml(check.detail)),
        `${profile.profile}: missing runtime request mapping result for ${check.name}`);
    }
  }
}
for (const task of inventory.taskRecipes) {
  hasLink(recipeIndex, `/models-recipes/model-recipes/tasks/${task.slug}`, 'Model Recipes');
  const taskLinks = links(readPage(`/models-recipes/model-recipes/tasks/${task.slug}`));
  for (const family of task.families) {
    hasLink(taskLinks, `/models-recipes/model-recipes/families/${family.slug}`, task.label);
  }
}
console.log(`Verified rendered links and configuration sections for ${inventory.familyCount} families, `
  + `${inventory.manifestCount} recipes, and ${inventory.taskRecipes.length} tasks.`);
