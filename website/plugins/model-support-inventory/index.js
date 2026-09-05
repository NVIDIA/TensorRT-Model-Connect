/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

const fs = require('fs');
const path = require('path');

const TASKS = {
  audio_generation: ['Text to speech', 'Audio', 'text-to-speech', 'generate-audio'],
  classification: ['Image classification', 'Computer Vision', 'image-classification', 'classify'],
  embedding: ['Embedding', 'Natural Language Processing', 'feature-extraction', 'embed'],
  encoding: ['Encoding', 'Natural Language Processing', 'feature-extraction', 'encode'],
  image_edit: ['Image editing', 'Computer Vision', 'image-to-image', 'generate-image'],
  image_features: ['Image feature extraction', 'Computer Vision', 'image-feature-extraction', 'extract-features'],
  image_generation: ['Image generation', 'Computer Vision', 'text-to-image', 'generate-image'],
  image_generation_batch: ['Batch image generation', 'Computer Vision', 'text-to-image', 'generate-image-batch'],
  monocular_geometry: ['Monocular geometry', 'Computer Vision', 'depth-estimation', 'geometry'],
  pose_hypothesis_refinement: ['Pose hypothesis refinement', 'Robotics', 'robotics', null],
  prompted_segmentation: ['Prompted segmentation', 'Computer Vision', 'mask-generation', 'segment'],
  reranking: ['Text ranking', 'Natural Language Processing', 'text-ranking', 'rerank'],
  robot_control: ['Robot control', 'Robotics', 'robotics', 'control'],
  segmentation: ['Image segmentation', 'Computer Vision', 'image-segmentation', 'segment'],
  speech_session: ['Speech session', 'Audio', 'audio-to-audio', 'speak'],
  speech_to_speech: ['Speech to speech', 'Audio', 'audio-to-audio', 'speak'],
  stereo_disparity: ['Depth estimation', 'Computer Vision', 'depth-estimation', 'disparity'],
  text_generation: ['Text generation', 'Natural Language Processing', 'text-generation', 'run'],
  text_prompted_segmentation: ['Text-prompted segmentation', 'Computer Vision', 'mask-generation', 'segment'],
  time_series_forecast: ['Time-series forecasting', 'Time Series', 'time-series-forecasting', 'forecast'],
  transcription: ['Speech recognition', 'Audio', 'automatic-speech-recognition', 'transcribe'],
  transcription_streaming: ['Streaming speech recognition', 'Audio', 'automatic-speech-recognition', 'transcribe'],
  video_segmentation: ['Video segmentation', 'Computer Vision', 'image-segmentation', 'video-segment'],
  vision_language_generation: ['Vision-language generation', 'Multimodal', 'image-text-to-text', 'run'],
  world_model_generation: ['World-model generation', 'Computer Vision', 'image-to-video', 'generate-world'],
};

function readJson(filePath) {
  try {
    return JSON.parse(fs.readFileSync(filePath, 'utf8'));
  } catch (error) {
    throw new Error(`Unable to parse ${filePath}: ${error.message}`);
  }
}

function familyDirectories(repoRoot) {
  const root = path.join(repoRoot, 'families');
  return fs.readdirSync(root, {withFileTypes: true})
    .filter((entry) => entry.isDirectory() && !entry.name.startsWith('_'))
    .map((entry) => ({name: entry.name, root: path.join(root, entry.name)}))
    .sort((left, right) => left.name.localeCompare(right.name));
}

function requireFamilyShape(family) {
  for (const relative of ['support.py', 'model.py', 'runtime/CMakeLists.txt', 'tests']) {
    if (!fs.existsSync(path.join(family.root, relative))) {
      throw new Error(`${family.name} is missing families/${family.name}/${relative}`);
    }
  }
  if (fs.existsSync(path.join(family.root, 'MODEL.toml'))) {
    throw new Error(`${family.name} contains forbidden MODEL.toml metadata`);
  }
}

function manifestFiles(family) {
  const root = path.join(family.root, 'tests', 'manifests');
  if (!fs.existsSync(root)) return [];
  return fs.readdirSync(root, {withFileTypes: true})
    .filter((entry) => entry.isFile() && entry.name.endsWith('.json'))
    .map((entry) => path.join(root, entry.name))
    .sort();
}

function profileFromManifest(repoRoot, family, filePath) {
  const manifest = readJson(filePath);
  if (manifest.runtime_strategy !== undefined || manifest.task_strategy !== undefined) {
    throw new Error(`${filePath} uses a removed strategy field`);
  }
  if (!manifest.name || !manifest.bundle
      || manifest.family !== family.name || !manifest.task) {
    throw new Error(`${filePath} must declare name, bundle, exact family, and task`);
  }
  const task = TASKS[manifest.task];
  if (!task) throw new Error(`${filePath} declares unknown task ${manifest.task}`);
  if (typeof manifest.precision !== 'string' || !Number.isInteger(manifest.tensor_parallel_size)) {
    throw new Error(`${filePath} must declare precision and tensor_parallel_size`);
  }
  const tensorParallelSize = manifest.tensor_parallel_size;
  const contextParallelSize = manifest.context_parallel_size ?? 1;
  if (tensorParallelSize < 1 || !Number.isInteger(contextParallelSize)
      || contextParallelSize < 1 || (tensorParallelSize > 1 && contextParallelSize > 1)) {
    throw new Error(`${filePath} has invalid or overlapping parallel sizes`);
  }
  const parallelSize = Math.max(tensorParallelSize, contextParallelSize);
  const parallelMode = contextParallelSize > 1
    ? 'context_parallel'
    : tensorParallelSize > 1 ? 'tensor_parallel' : 'single_device';
  return {
    profile: manifest.name,
    hfId: manifest.hf_id || 'prepared local checkpoint',
    revision: manifest.hf_id ? manifest.hf_revision || 'not pinned' : 'not applicable',
    bundle: manifest.bundle,
    family: family.name,
    task: manifest.task,
    taskSlugs: [manifest.task.replaceAll('_', '-')],
    cliCommands: task[3] ? [task[3]] : [],
    precision: manifest.precision,
    quantization: manifest.quantization || 'none',
    parallelMode,
    parallelSize,
    testcases: Array.isArray(manifest.testcases)
      ? manifest.testcases.map((testcase) => testcase.name).filter(Boolean)
      : [],
    fp32Layers: Array.isArray(manifest.fp32_layers) ? manifest.fp32_layers : [],
    sourcePath: path.relative(repoRoot, filePath).replaceAll('\\', '/'),
  };
}

function taskRecipe(taskName, profiles) {
  const [label, category, hfSlug] = TASKS[taskName];
  const slug = taskName.replaceAll('_', '-');
  const byFamily = new Map();
  for (const profile of profiles) {
    if (!byFamily.has(profile.family)) byFamily.set(profile.family, []);
    byFamily.get(profile.family).push(profile);
  }
  return {
    task: taskName,
    slug,
    label,
    category,
    description: `Family-owned implementations of the ${label.toLowerCase()} task interface.`,
    hfUrl: `https://huggingface.co/tasks/${hfSlug}`,
    recipeCount: profiles.length,
    families: [...byFamily.entries()].map(([family, familyProfiles]) => ({
      family,
      slug: family.replaceAll('_', '-'),
      recipeCount: familyProfiles.length,
      hfIds: [...new Set(familyProfiles.map((profile) => profile.hfId))].sort(),
      cliCommands: [...new Set(familyProfiles.flatMap((profile) => profile.cliCommands))].sort(),
    })).sort((left, right) => left.family.localeCompare(right.family)),
  };
}

function collectModelSupportInventory(repoRoot) {
  const families = familyDirectories(repoRoot);
  const profiles = [];
  for (const family of families) {
    requireFamilyShape(family);
    for (const filePath of manifestFiles(family)) {
      profiles.push(profileFromManifest(repoRoot, family, filePath));
    }
  }
  profiles.sort((left, right) =>
    left.task.localeCompare(right.task) ||
    left.family.localeCompare(right.family) ||
    left.profile.localeCompare(right.profile));

  const familyRecipes = families.map((family) => {
    const owned = profiles.filter((profile) => profile.family === family.name);
    return {
      family: family.name,
      slug: family.name.replaceAll('_', '-'),
      profiles: owned,
      taskSlugs: [...new Set(owned.flatMap((profile) => profile.taskSlugs))].sort(),
      cliCommands: [...new Set(owned.flatMap((profile) => profile.cliCommands))].sort(),
    };
  });
  const profilesByTask = new Map();
  for (const profile of profiles) {
    if (!profilesByTask.has(profile.task)) profilesByTask.set(profile.task, []);
    profilesByTask.get(profile.task).push(profile);
  }
  const taskRecipes = [...profilesByTask.entries()]
    .map(([task, taskProfiles]) => taskRecipe(task, taskProfiles))
    .sort((left, right) => left.label.localeCompare(right.label));

  return {
    familyCount: families.length,
    familyNames: families.map((family) => family.name),
    manifestCount: profiles.length,
    modelProfiles: profiles,
    familyRecipes,
    taskRecipes,
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
      const routeBase = context.baseUrl.replace(/\/$/, '');
      const taskPage = path.join(context.siteDir, 'src/components/ModelRecipes/TaskPage.js');
      const familyPage = path.join(context.siteDir, 'src/components/ModelRecipes/FamilyPage.js');
      for (const task of content.taskRecipes) {
        actions.addRoute({
          path: `${routeBase}/models-recipes/model-recipes/tasks/${task.slug}`,
          component: taskPage,
          exact: true,
          props: {taskSlug: task.slug},
        });
      }
      for (const family of content.familyRecipes) {
        actions.addRoute({
          path: `${routeBase}/models-recipes/model-recipes/families/${family.slug}`,
          component: familyPage,
          exact: true,
          props: {familySlug: family.slug},
        });
      }
    },
  };
}

module.exports = modelSupportInventoryPlugin;
module.exports.collectModelSupportInventory = collectModelSupportInventory;
