/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import {usePluginData} from '@docusaurus/useGlobalData';
import styles from './styles.module.css';

export function SourceLink({source, children}) {
  const {sourceBaseUrl} = usePluginData('model-support-inventory');
  if (!source) return null;
  const path = typeof source === 'string' ? source : source.path;
  const line = typeof source === 'object' && source.line ? `#L${source.line}` : '';
  return <a href={`${sourceBaseUrl}/${path}${line}`}>{children || <code>{path}{line && `:${source.line}`}</code>}</a>;
}

export function ExactValue({value}) {
  const text = JSON.stringify(value, null, 2);
  return typeof value === 'object' && value !== null
    ? <pre className={styles.value}><code>{text}</code></pre>
    : <code className={styles.inlineValue}>{text}</code>;
}

export function SettingsTable({settings}) {
  return (
    <div className={styles.tableScroll}>
      <table className={styles.parameterTable}>
        <thead><tr><th>Setting</th><th>Declared value</th></tr></thead>
        <tbody>{Object.entries(settings).map(([name, value]) => (
          <tr key={name}><td><code>{name}</code></td><td><ExactValue value={value} /></td></tr>
        ))}</tbody>
      </table>
    </div>
  );
}

function FieldTable({fields}) {
  if (!fields?.length) return <p>No typed fields were extracted. See the source for configuration handling.</p>;
  return (
    <div className={styles.tableScroll}>
      <table className={styles.parameterTable}>
        <thead><tr><th>Parameter</th><th>Type</th><th>Declared default</th><th>Description</th></tr></thead>
        <tbody>{fields.map((field, index) => (
          <tr key={`${field.name}:${field.source?.path}:${field.source?.line}:${index}`}>
            <td><SourceLink source={field.source}><code>{field.name}</code></SourceLink>
              {field.readOnly ? <small className={styles.note}>Derived / read-only</small>
                : field.required ? <small className={styles.note}>Required</small> : null}
            </td>
            <td><code className={styles.inlineValue}>{field.type || 'Not annotated'}</code></td>
            <td>{field.default === null || field.default === undefined
              ? <span className={styles.muted}>{field.readOnly ? 'Computed' : 'Not declared'}</span>
              : <code className={styles.inlineValue}>{field.default}</code>}</td>
            <td>{field.description || '—'}</td>
          </tr>
        ))}</tbody>
      </table>
    </div>
  );
}

function CommandParameters({command, family}) {
  return (
    <details className={styles.details}>
      <summary><code>{['trtmc', family, command.name].filter(Boolean).join(' ')}</code> — {command.help || command.executor}</summary>
      <div className={styles.detailsBody}>
        <p>{command.executor === 'native' ? 'Native runtime command' : 'Python command'}{command.handler && <>; handler <code>{command.handler}</code></>}.
          {' '}<SourceLink source={command.handlerSource || command.source}>View implementation</SourceLink></p>
        {command.issues?.length > 0 && <ul>{command.issues.map((issue) => <li key={issue}>{issue}</li>)}</ul>}
        <div className={styles.tableScroll}>
          <table className={styles.parameterTable}>
            <thead><tr><th>Parameter / flags</th><th>Type</th><th>Default</th><th>Choices / description</th></tr></thead>
            <tbody>{(command.arguments || []).map((argument) => (
              <tr key={argument.name}>
                <td>
                  <code>{argument.name}</code>
                  <small className={styles.note}>{argument.flags?.join(', ') || 'Positional argument'}</small>
                  {(argument.required ?? !argument.flags) && <small className={styles.note}>Required</small>}
                </td>
                <td><code>{argument.type}</code>{argument.action && <small className={styles.note}>Action: <code>{argument.action}</code></small>}</td>
                <td>{Object.hasOwn(argument, 'default') ? <ExactValue value={argument.default} /> : 'Not declared'}</td>
                <td>
                  {argument.choices && <div><ExactValue value={argument.choices} /></div>}
                  {argument.help || (!argument.choices && '—')}
                </td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      </div>
    </details>
  );
}

export function SourceVerification({verification}) {
  return (
    <section id="verification" className={styles.verification}>
      <h2>Build configuration checks</h2>
      <p>
        {verification?.status === 'checked' ? 'The build-input source and CPU checks below passed for this recipe.'
          : verification?.status === 'issues' ? 'This recipe has configuration issues. Review them before using it.'
            : 'This recipe has not passed the available configuration checks.'}
        {' '}These checks do not establish a successful TensorRT build, runtime configuration validation, GPU execution, numerical parity, or performance.
      </p>
      {verification?.issues?.length > 0 && <ul>{verification.issues.map((issue) => <li key={issue}>{issue}</li>)}</ul>}
      {verification?.checks?.length > 0 && (
        <details className={styles.details}>
          <summary>Check results</summary>
          <div className={styles.detailsBody}>
            <ul>{verification.checks.map((check, index) => (
              <li key={`${check.name}-${index}`}><strong>{check.name}: {check.status}</strong>{check.detail && ` — ${check.detail}`}</li>
            ))}</ul>
          </div>
        </details>
      )}
    </section>
  );
}

const SCHEMA_KINDS = [
  ['checkpoint', 'Checkpoint configuration'],
  ['build', 'Build configuration'],
  ['runtime', 'Runtime configuration'],
  ['other', 'Other configuration'],
];

export function FamilyParameters({family}) {
  const {sharedBuildCommand} = usePluginData('model-support-inventory');
  const details = family.details;
  if (!details) return <p>Configuration details were not extracted for this family.</p>;
  const buildRequest = details.buildRequest;
  const cli = details.cli;
  return (
    <>
      {details.support && <section id="support-declarations">
        <h2>Family support declarations</h2>
        <p>{details.support.note}{' '}<SourceLink source={details.support.source} /></p>
        {details.support.declarations.map((declaration, index) => <div key={index}>
          <SettingsTable settings={declaration.values} />
        </div>)}
      </section>}
      <section id="cli-parameters">
        <h2>CLI parameters</h2>
        {cli?.commands?.length ? (
          <>
            <p>Commands and options declared by this family in <SourceLink source={cli.source} />.
              {' '}Defaults and choices below are copied from that declaration.</p>
            {cli.commands.map((command) => <CommandParameters key={command.name} command={command} family={family.family} />)}
          </>
        ) : <>
          <p>This family does not provide a valid <code>cli.json</code> command declaration.
            {' '}The shared build parser below describes available syntax; the selected family may reject individual options or values.
            {' '}<SourceLink source={cli?.fallbackSource} /></p>
          {sharedBuildCommand && <CommandParameters command={sharedBuildCommand} />}
        </>}
        {cli?.issues?.length > 0 && <ul>{cli.issues.map((issue) => <li key={issue}>{issue}</li>)}</ul>}
      </section>
      <section id="build-parameters">
        <h2>Build request parameters</h2>
        {buildRequest?.fields?.length ? (
          <>
            <p>{buildRequest.owner === 'shared' ? 'This family uses the shared Python build request.' : 'This family declares its own Python build request.'}
              {' '}Field presence does not mean that every value or combination is supported; check the source restrictions below.
              {' '}<SourceLink source={buildRequest.source} /></p>
            <details className={styles.details}>
              <summary>All {buildRequest.fields.length} build request fields</summary>
              <div className={styles.detailsBody}><FieldTable fields={buildRequest.fields} /></div>
            </details>
          </>
        ) : <p>No typed build request was extracted. Follow the implementation links below for the family's build interface.</p>}
      </section>
      <section id="configuration-schema">
        <h2>Configuration schemas</h2>
        <p>Fields, defaults, and declarations are read from the current Python and C++ source. Expressions are shown as declared.
          {' '}These include internal and derived fields; they are not all user-editable options. Schema defaults do not represent the downloaded checkpoint's architecture values or guarantee that every configuration is supported.</p>
        {SCHEMA_KINDS.map(([kind, label]) => {
          const schemas = (details.configClasses || []).filter((schema) => schema.kind === kind);
          if (!schemas.length) return null;
          return <section key={kind}>
            <h3>{label}</h3>
            {schemas.map((schema) => (
              <details key={`${schema.source.path}:${schema.source.line}:${schema.name}`} className={styles.details}>
                <summary><code>{schema.name}</code> · {schema.fields?.length || 0} fields</summary>
                <div className={styles.detailsBody}>
                  {schema.description && <p>{schema.description}</p>}
                  <p><SourceLink source={schema.source} /></p>
                  <FieldTable fields={schema.fields} />
                  {schema.declaration && <details className={styles.details}>
                    <summary>Complete source declaration</summary>
                    <div className={styles.detailsBody}><pre><code>{schema.declaration}</code></pre></div>
                  </details>}
                </div>
              </details>
            ))}
          </section>;
        })}
        {!details.configClasses?.length && <p>No typed configuration class was extracted. The family may read its checkpoint configuration directly; consult its implementation.</p>}
      </section>
      <section id="configuration-constraints">
        <h2>Source restrictions</h2>
        <p>Rejection guards extracted from the family source are listed for review. Their conditions may depend on the checkpoint or execution path; this is not an exhaustive validation of possible configurations.</p>
        {details.constraints?.length ? (
          <details className={styles.details}>
            <summary>{details.constraints.length} source guards</summary>
            <div className={styles.detailsBody}>
              {details.constraints.map((constraint, index) => (
                <div key={`${constraint.source.path}:${constraint.source.line}:${index}`} className={styles.constraint}>
                  <p><code>{constraint.exception}</code>{constraint.scope && <> in <code>{constraint.scope}</code></>}: <code className={styles.inlineValue}>{constraint.message}</code></p>
                  {constraint.condition && <p>When <code className={styles.inlineValue}>{constraint.condition}</code></p>}
                  <SourceLink source={constraint.source} />
                </div>
              ))}
            </div>
          </details>
        ) : <p>No rejection guards were extracted. See the implementation for additional conditions.</p>}
      </section>
      <section id="implementation-sources">
        <h2>Implementation sources</h2>
        <ul>{(details.sources || []).map((source) => <li key={`${source.path}:${source.line || 0}`}><SourceLink source={source} /></li>)}</ul>
      </section>
    </>
  );
}
