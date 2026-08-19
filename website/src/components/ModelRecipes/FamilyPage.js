/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import Link from '@docusaurus/Link';
import Layout from '@theme/Layout';
import {usePluginData} from '@docusaurus/useGlobalData';
import RecipePageLayout from './RecipePageLayout';

const BUNDLE_CLI_REFERENCE = {
  build: {
    purpose: 'Build one exact model source into a TensorRT-Model-Connect bundle.',
    syntax: 'trtmc build <model-source> --output <bundle.bundle>',
  },
  inspect: {
    purpose: 'Inspect bundle metadata, runtime identity, and packaged sections.',
    syntax: 'trtmc inspect <bundle.bundle>',
  },
};

const COMMAND_ORDER = [
  'build', 'run', 'encode', 'embed', 'rerank', 'classify', 'segment',
  'segment-prompted', 'generate-audio', 'generate-video', 'transcribe',
  'speak', 'solve', 'inspect',
];

function parallelLabel(profile) {
  if (profile.parallelMode === 'single_device') return 'single device';
  const abbreviation = profile.parallelMode === 'tensor_parallel' ? 'TP' : 'CP';
  return `${abbreviation}${profile.parallelSize}`;
}

function ArchitectureNames({profile}) {
  if (profile.modelSourceKind === 'local_source_package') {
    return <><span>Family-owned source graph</span><br /><small>Not sourced from Hugging Face metadata</small></>;
  }
  if (profile.hfArchitectures.length === 0) {
    return <><span>—</span><br /><small>Not declared by checkpoint metadata</small></>;
  }
  return (
    <>
      {profile.hfArchitectures.map((architecture, index) => (
        <React.Fragment key={architecture}>
          {index > 0 && <br />}
          <code>{architecture}</code>
        </React.Fragment>
      ))}
      <br />
      <small>Source: <code>{profile.hfArchitectureSource}</code></small>
    </>
  );
}

function collectArchitectureContracts(profiles) {
  const contracts = new Map();
  for (const profile of profiles) {
    const key = JSON.stringify([
      profile.modelSourceKind,
      profile.hfModelType,
      profile.hfArchitectures,
      profile.hfArchitectureSource,
      profile.taskStrategy,
    ]);
    if (!contracts.has(key)) {
      contracts.set(key, {...profile, profileNames: []});
    }
    contracts.get(key).profileNames.push(profile.profile);
  }
  // Keep this explicit for the browser bundle. The loose iterable-spread
  // transform can compile `[...map.values()]` as `[map.values()]`, leaving a
  // Map iterator where the page expects architecture records.
  return Array.from(contracts.values());
}

function FamilyConfigReference({family, commands}) {
  if (family.configSchemas.length === 0) {
    return (
      <>
        <h2>Family-owned configuration</h2>
        <p>
          This family does not declare a family-owned <code>--set</code>{' '}
          namespace. Use the explicit CLI options shown below and the shared
          configuration namespaces documented in{' '}
          <Link to="/user-guides/configure-runtime">Configure Runtime Behavior</Link>.
        </p>
      </>
    );
  }

  const runtimeCommand = commands.find(
    (command) => !['build', 'inspect'].includes(command)
  ) || 'run';
  return (
    <>
      <h2>Family-owned configuration</h2>
      <p>
        These keys are extracted from the family&apos;s registered config schema.
        Pass one key per repeatable <code>--set namespace.field=value</code>{' '}
        argument. Build-time keys belong on <code>trtmc build</code>; session
        keys belong on the family&apos;s inference command.
      </p>
      {family.configSchemas.map((schema) => {
        const buildField = schema.fields.find((field) => field.surfaces.includes('build'));
        const sessionField = schema.fields.find(
          (field) => field.surfaces.includes('runtime') &&
            field.allowedLayers.includes('Session Request')
        );
        return (
          <section key={schema.namespace}>
            <h3><code>{schema.namespace}</code></h3>
            <p>
              Schema {schema.sourcePaths.length === 1 ? 'source' : 'sources'}:{' '}
              {schema.sourcePaths.map((sourcePath, index) => (
                <React.Fragment key={sourcePath}>
                  {index > 0 && <br />}
                  <code>{sourcePath}</code>
                </React.Fragment>
              ))}
            </p>
            <table>
              <thead>
                <tr>
                  <th><code>--set</code> key</th>
                  <th>Type</th>
                  <th>Default</th>
                  <th>CLI surface</th>
                  <th>Allowed configuration layers</th>
                </tr>
              </thead>
              <tbody>
                {schema.fields.map((field) => (
                  <tr key={field.key}>
                    <td><code>{field.key}</code></td>
                    <td><code>{field.type}</code></td>
                    <td><code>{field.defaultValue}</code></td>
                    <td>{field.surfaces.length > 0 ? field.surfaces.join(', ') : 'Schema metadata only'}</td>
                    <td>{field.allowedLayers.join(', ')}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {buildField && (
              <pre><code>{`trtmc build <model-source> --output <bundle.bundle> --set ${buildField.key}=<value>`}</code></pre>
            )}
            {sessionField && (
              <pre><code>{`trtmc ${runtimeCommand} <bundle.bundle> <task-inputs> --set ${sessionField.key}=<value>`}</code></pre>
            )}
          </section>
        );
      })}
      <p>
        The registry rejects unknown namespaces and fields. A{' '}
        <Link to="/user-guides/configure-runtime"><code>--config</code> file</Link>{' '}
        can set the same keys in JSON or YAML form.
      </p>
    </>
  );
}

export default function ModelFamilyRecipePage({familySlug}) {
  const {familyRecipes, taskRecipes} = usePluginData('model-support-inventory');
  const family = familyRecipes.find((candidate) => candidate.slug === familySlug);

  if (!family) {
    return <Layout title="Model family not found"><main className="container margin-vert--lg"><h1>Model family not found</h1></main></Layout>;
  }

  const taskBySlug = new Map(taskRecipes.map((task) => [task.slug, task]));
  const commands = COMMAND_ORDER.filter((command) => family.cliCommands.includes(command));
  const architectureContracts = collectArchitectureContracts(family.profiles);
  const cAbiProfiles = family.profiles.filter(
    (profile) => profile.runtimeApi?.kind === 'model_owned_c_abi'
  );

  return (
    <RecipePageLayout title={`${family.family} model recipes`} description={`Declared recipes, CLI commands, and configuration keys for the ${family.family} model family.`}>
      <article>
        <p><Link to="/models-recipes/model-recipes">← All model recipe tasks</Link></p>
        <h1><code>{family.family}</code> model family</h1>

        <h2>Tasks</h2>
        <ul>
          {family.taskSlugs.map((taskSlug) => {
            const task = taskBySlug.get(taskSlug);
            return (
              <li key={taskSlug}>
                <Link to={`/models-recipes/model-recipes/tasks/${taskSlug}`}>
                  {task?.label || taskSlug}
                </Link>
              </li>
            );
          })}
        </ul>

        <h2>Supported architectures and task heads</h2>
        <p>
          Hugging Face values are copied from checkpoint metadata at the recorded
          revision. Local source-package recipes are identified explicitly and
          use their family-owned graph instead. The TRTMC task contract comes
          from the exact E2E recipe; architecture identity does not imply that
          TRTMC reproduces every source-model head. For example, an encoder recipe
          may consume the base model and intentionally return hidden states instead
          of the checkpoint&apos;s pretraining or classification logits.
        </p>
        <table>
          <thead>
            <tr>
              <th>Source <code>model_type</code></th>
              <th>Architecture / pipeline class</th>
              <th>TRTMC task contract</th>
              <th>Exact recipe profiles</th>
            </tr>
          </thead>
          <tbody>
            {architectureContracts.map((contract) => (
              <tr key={`${contract.hfModelType}-${contract.hfArchitectureSource}-${contract.hfArchitectures.join('-')}-${contract.taskStrategy}`}>
                <td>{['not declared', 'not applicable'].includes(contract.hfModelType)
                  ? '—'
                  : <code>{contract.hfModelType}</code>}</td>
                <td>
                  <ArchitectureNames profile={contract} />
                  <br />
                  <small>
                    {contract.modelSourceKind === 'local_source_package' ? (
                      <>Local source package</>
                    ) : (
                      <>Metadata: <code>{contract.hfMetadataFile}</code> at{' '}
                        <code>{contract.hfMetadataRevision.slice(0, 12)}</code></>
                    )}
                  </small>
                </td>
                <td><code>{contract.taskStrategy}</code></td>
                <td>{contract.profileNames.map((profileName) => <div key={profileName}><code>{profileName}</code></div>)}</td>
              </tr>
            ))}
          </tbody>
        </table>

        <h2>Declared recipes</h2>
        <p>
          Each row comes from a model-owned E2E manifest. It declares a test
          recipe; it is not a live pass receipt for every hardware target.
        </p>
        <table>
          <thead>
            <tr>
              <th>Recipe</th>
              <th>Exact model source</th>
              <th>Task</th>
              <th>Build configuration</th>
              <th>Runtime interface</th>
              <th>Declared E2E cases</th>
              <th>Manifest</th>
            </tr>
          </thead>
          <tbody>
            {family.profiles.map((profile) => (
              <tr key={profile.sourcePath}>
                <td><code>{profile.profile}</code></td>
                <td>
                  <code>{profile.hfId}</code>
                  {profile.modelSourceKind === 'local_source_package'
                    ? <><br /><small>Local source package</small></>
                    : profile.revision !== 'not pinned' && <><br /><small>Revision: <code>{profile.revision}</code></small></>}
                </td>
                <td>
                  {profile.modelSourceKind === 'local_source_package' && (
                    <><code>{profile.taskStrategy}</code><br /><small>Nearest catalog category:</small></>
                  )}
                  {profile.hfTasks.map((taskSlug) => (
                    <div key={taskSlug}>
                      <Link to={`/models-recipes/model-recipes/tasks/${taskSlug}`}>
                        {taskBySlug.get(taskSlug)?.label || taskSlug}
                      </Link>
                    </div>
                  ))}
                </td>
                <td>
                  {profile.precision}; {parallelLabel(profile)}
                  {profile.fp32Layers.length > 0 && <><br /><small>FP32 layers: {profile.fp32Layers.join(', ')}</small></>}
                  {profile.quantization !== 'not declared' && <><br /><small>Quantization: {profile.quantization}</small></>}
                </td>
                <td>
                  {profile.runtimeApi?.kind === 'model_owned_c_abi' ? (
                    <>
                      Model-owned C ABI<br />
                      <small><code>{profile.runtimeApi.entrypoint}</code></small>
                    </>
                  ) : 'Task-specific CLI'}
                </td>
                <td>{profile.testcases.length > 0 ? profile.testcases.join(', ') : 'No named testcase'}</td>
                <td><code>{profile.sourcePath}</code></td>
              </tr>
            ))}
          </tbody>
        </table>

        <FamilyConfigReference family={family} commands={commands} />

        {cAbiProfiles.length > 0 && (
          <>
            <h2>Model-owned C ABI</h2>
            <p>
              These recipes intentionally have no generic task CLI command. The
              E2E runner loads the family DSO directly and calls its versioned C ABI.
            </p>
            {cAbiProfiles.map((profile) => (
              <section key={`${profile.profile}-${profile.runtimeApi.entrypoint}`}>
                <h3><code>{profile.runtimeApi.entrypoint}</code></h3>
                <p>
                  Recipe: <code>{profile.profile}</code><br />
                  Runtime library: <code>{profile.runtimeApi.library}</code><br />
                  Public header: <code>{profile.runtimeApi.header}</code>
                </p>
              </section>
            ))}
          </>
        )}

        <h2>Family-specific CLI contracts</h2>
        {family.commandContracts.length === 0 ? (
          <p>No task-specific CLI contract is declared for this family.</p>
        ) : (
          <p>
            Inputs and options below are filtered by the declared E2E task and
            the methods and configuration fields used by that family&apos;s native
            runtime implementation. The global CLI parser accepts a wider union
            of flags; flags absent here are not declared for this family.
          </p>
        )}
        {family.commandContracts.map((contract, index) => (
          <section key={`${contract.command}-${contract.profileNames.join('-')}-${index}`}>
            <h3><code>trtmc {contract.command}</code></h3>
            <p>{contract.purpose}</p>
            <p>
              Declared recipes:{' '}
              {contract.profileNames.map((profileName, profileIndex) => (
                <React.Fragment key={profileName}>
                  {profileIndex > 0 && ', '}
                  <code>{profileName}</code>
                </React.Fragment>
              ))}
            </p>
            <pre><code>{contract.syntax}</code></pre>
            <table>
              <thead>
                <tr>
                  <th>Supported input or option</th>
                  <th>Requirement</th>
                  <th>Runtime behavior</th>
                </tr>
              </thead>
              <tbody>
                {contract.options.map((cliOption) => (
                  <tr key={cliOption.flag}>
                    <td><code>{cliOption.flag}</code></td>
                    <td>{cliOption.requirement}</td>
                    <td>{cliOption.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <details>
              <summary>Code basis</summary>
              <p>
                Runtime provider{contract.runtimeOwners.length === 1 ? '' : 's'}:{' '}
                {contract.runtimeOwners.length > 0
                  ? contract.runtimeOwners.map((owner, ownerIndex) => (
                    <React.Fragment key={owner}>
                      {ownerIndex > 0 && ', '}
                      <code>{owner}</code>
                    </React.Fragment>
                  ))
                  : 'No native runtime owner declared'}
              </p>
              <ul>
                {contract.evidence.map((sourcePath) => (
                  <li key={sourcePath}><code>{sourcePath}</code></li>
                ))}
              </ul>
            </details>
          </section>
        ))}

        <h2>Bundle lifecycle commands</h2>
        <p>
          These commands apply to every declared recipe in this family; they
          do not add task inputs or model capabilities.
        </p>
        {['build', 'inspect'].map((command) => (
          <section key={command}>
            <h3><code>trtmc {command}</code></h3>
            <p>{BUNDLE_CLI_REFERENCE[command].purpose}</p>
            <pre><code>{BUNDLE_CLI_REFERENCE[command].syntax}</code></pre>
          </section>
        ))}
        <p>See the <Link to="/api/cli-reference">CLI Reference</Link> for all options and limitations.</p>
      </article>
    </RecipePageLayout>
  );
}
