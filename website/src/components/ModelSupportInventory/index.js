/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import Link from '@docusaurus/Link';
import {usePluginData} from '@docusaurus/useGlobalData';

const TASK_LABELS = {
  diffusion_media_generation: 'Image / video generation',
  diffusion_text_generation: 'Diffusion text generation',
  embedding: 'Embedding',
  encoder_only_nlp: 'Encoder NLP',
  image_classification: 'Image classification',
  image_feature_extraction: 'Image feature extraction',
  monocular_geometry: 'Monocular geometry',
  neural_operator: 'Time-series / neural operator',
  omni_multimodal: 'Omni multimodal',
  prompted_segmentation: 'Prompted segmentation',
  reranking: 'Reranking',
  segmentation: 'Segmentation',
  speech_to_speech: 'Speech to speech',
  speech_to_text: 'Speech to text',
  text_generation_causal: 'Text generation',
  text_to_audio: 'Text to audio',
  vision_language_generation: 'Vision-language generation',
};

const TASK_GROUPS = {
  text: ['text_generation_causal', 'encoder_only_nlp', 'embedding', 'reranking'],
  'multimodal-speech': [
    'vision_language_generation', 'omni_multimodal', 'speech_to_text',
    'text_to_audio', 'speech_to_speech',
  ],
  'image-video': [
    'diffusion_media_generation', 'image_classification', 'image_feature_extraction',
    'monocular_geometry', 'segmentation', 'prompted_segmentation',
  ],
  'time-series': ['neural_operator', 'diffusion_text_generation'],
};

function taskLabel(task) {
  return TASK_LABELS[task] || task.replaceAll('_', ' ');
}

function parallelLabel(profile) {
  if (profile.parallelMode === 'single_device') return 'Single device';
  const mode = profile.parallelMode === 'tensor_parallel' ? 'TP' : 'CP';
  return `${mode}${profile.parallelSize}`;
}

function ModelProfileTable({profiles, taskGroup}) {
  const allowedTasks = taskGroup ? TASK_GROUPS[taskGroup] || [] : null;
  const scopedProfiles = allowedTasks
    ? profiles.filter((profile) => allowedTasks.includes(profile.taskStrategy))
    : profiles;

  return (
    <>
      <p>{scopedProfiles.length} declared configurations.</p>
      <div>
        <table>
          <thead>
            <tr>
              <th>Hugging Face checkpoint</th>
              <th>Manifest profile</th>
              <th>Task</th>
              <th>Family / runtime</th>
              <th>Build configuration</th>
              <th>Runtime path</th>
            </tr>
          </thead>
          <tbody>
            {scopedProfiles.map((profile) => (
              <tr key={profile.sourcePath}>
                <td>
                  <code>{profile.hfId}</code>
                  <br />
                  <small>Revision: {profile.revision}</small>
                </td>
                <td>
                  <code>{profile.profile}</code>
                  <br />
                  <small>{profile.sourcePath}</small>
                </td>
                <td>{taskLabel(profile.taskStrategy)}</td>
                <td>
                  <code>{profile.family}</code>
                  <br />
                  <small>{profile.runtimeStrategy}</small>
                </td>
                <td>
                  {profile.precision}; {parallelLabel(profile)}
                  <br />
                  <small>Quantization: {profile.quantization}</small>
                </td>
                <td>Native TensorRT</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
function CellLines({lines}) {
  return lines.map((line, index) => (
    <React.Fragment key={`${line}-${index}`}>
      {index > 0 && <br />}
      {line}
    </React.Fragment>
  ));
}

function OptimizedRuntimeDispatch({lines}) {
  if (lines.length === 0 || lines[0] === '—') return <>No</>;
  const qualified = lines.some(
    (line) =>
      line.startsWith('Qualified TRTMC dispatch target:') &&
      !line.endsWith('Coming soon')
  );
  const status = qualified ? 'Yes' : 'Coming soon';
  return (
    <>
      <strong>{status}</strong><br />
      <CellLines lines={lines} />
    </>
  );
}

function HfArchitecture({architectures, source}) {
  if (architectures.length === 0) {
    return <><span>—</span><br /><small>Not declared by checkpoint metadata</small></>;
  }
  return (
    <>
      {architectures.map((architecture, index) => (
        <React.Fragment key={architecture}>
          {index > 0 && <br />}
          <code>{architecture}</code>
        </React.Fragment>
      ))}
      <br />
      <small>Source: <code>{source}</code></small>
    </>
  );
}

function PerformanceSnapshotTable({rows}) {
  return (
    <table>
      <thead>
        <tr>
          <th>TRTMC family</th>
          <th>HF <code>model_type</code></th>
          <th>HF architecture / pipeline class</th>
          <th>TRTMC task/head contract</th>
          <th>Hugging Face ID (<code>hf_id</code>, CLI input)</th>
          <th>Checkpoint ID (TRTMC profile)</th>
          <th>GB300 performance</th>
          <th>Precision</th>
          <th>Quantization</th>
          <th>Optimized runtime dispatch</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr key={row.profile}>
            <td>
              {row.family === 'not declared' ? row.family : (
                <Link to={`/models-recipes/model-recipes/families/${row.family.replaceAll('_', '-')}`}>
                  <code>{row.family}</code>
                </Link>
              )}
            </td>
            <td>{row.hfModelType === 'not declared' ? '—' : <code>{row.hfModelType}</code>}</td>
            <td>
              <HfArchitecture
                architectures={row.hfArchitectures}
                source={row.hfArchitectureSource}
              />
            </td>
            <td>{row.taskStrategy === 'not declared' ? '—' : <code>{row.taskStrategy}</code>}</td>
            <td>
              <code>{row.hfId}</code>
              {row.revision !== 'not pinned' && <><br /><small>Revision: <code>{row.revision}</code></small></>}
              {row.manifestHfId !== 'not declared' && row.manifestHfId !== row.hfId && (
                <><br /><small>Current manifest <code>hf_id</code>: <code>{row.manifestHfId}</code></small></>
              )}
              <br />
              <small>
                Architecture metadata: <code>{row.hfMetadataFile}</code> at{' '}
                <code>{row.hfMetadataRevision.slice(0, 12)}</code>
              </small>
            </td>
            <td><code>{row.profile}</code></td>
            <td>{row.performance}</td>
            <td><CellLines lines={row.precision} /></td>
            <td><CellLines lines={row.quantization} /></td>
            <td><OptimizedRuntimeDispatch lines={row.platformSpecialization} /></td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default function ModelSupportInventory({variant = 'summary', taskGroup}) {
  const inventory = usePluginData('model-support-inventory');

  if (variant === 'facts') {
    return (
      <ul>
        <li>{inventory.familyPluginCount} Python family plugins under <code>python/tensorrt_model_connect/families/</code>.</li>
        <li>{inventory.e2eManifestCount} E2E model manifests and {inventory.e2eFamilyIndexCount} family indexes under <code>tests/e2e/models/</code>.</li>
        <li>{inventory.runtimeStrategyKeyCount} unique C++ runtime strategy keys declared by model metadata under <code>src/runtime/models/</code>.</li>
      </ul>
    );
  }
  if (variant === 'families') {
    return <pre><code>{inventory.familyPluginNames.join(', ')}</code></pre>;
  }
  if (variant === 'models') {
    return <ModelProfileTable profiles={inventory.modelProfiles} taskGroup={taskGroup} />;
  }
  if (variant === 'performance') {
    return <PerformanceSnapshotTable rows={inventory.performanceSnapshot} />;
  }
  return (
    <p>
      The current checkout contains {inventory.familyPluginCount} Python family plugins,{' '}
      {inventory.e2eManifestCount} E2E model manifests, and {inventory.e2eFamilyIndexCount} E2E family indexes.
    </p>
  );
}
