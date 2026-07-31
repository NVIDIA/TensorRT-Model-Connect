/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import {instance} from '@viz-js/viz';
import {existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync} from 'node:fs';
import {dirname, join, relative, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';

const websiteRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const sourceRoot = join(websiteRoot, 'diagram-sources');
const outputRoot = join(websiteRoot, 'static', 'img', 'diagrams');
const documentationRoot = join(websiteRoot, 'docs');
const checkOnly = process.argv.includes('--check');

const palette = {
  api: {fill: '#f5f3ff', stroke: '#7c3aed'},
  artifact: {fill: '#fffbeb', stroke: '#b45309'},
  build: {fill: '#eff6ff', stroke: '#2563eb'},
  neutral: {fill: '#ffffff', stroke: '#64748b'},
  runtime: {fill: '#ecfccb', stroke: '#4d7c0f'},
  validation: {fill: '#ecfeff', stroke: '#0e7490'},
  warning: {fill: '#fef2f2', stroke: '#b91c1c'},
};

function collectDotFiles(directory) {
  return readdirSync(directory, {withFileTypes: true})
    .flatMap((entry) => {
      const path = join(directory, entry.name);
      return entry.isDirectory() ? collectDotFiles(path) : [path];
    })
    .filter((path) => path.endsWith('.dot.txt'))
    .sort();
}

function collectSequenceFiles(directory) {
  return readdirSync(directory, {withFileTypes: true})
    .flatMap((entry) => {
      const path = join(directory, entry.name);
      return entry.isDirectory() ? collectSequenceFiles(path) : [path];
    })
    .filter((path) => path.endsWith('.sequence.json'))
    .sort();
}

function collectDocumentationFiles(directory) {
  return readdirSync(directory, {withFileTypes: true})
    .flatMap((entry) => {
      const path = join(directory, entry.name);
      return entry.isDirectory() ? collectDocumentationFiles(path) : [path];
    })
    .filter((path) => path.endsWith('.md') || path.endsWith('.mdx'))
    .sort();
}

function collectSvgFiles(directory) {
  return readdirSync(directory, {withFileTypes: true})
    .flatMap((entry) => {
      const path = join(directory, entry.name);
      return entry.isDirectory() ? collectSvgFiles(path) : [path];
    })
    .filter((path) => path.endsWith('.svg'))
    .sort();
}

function readMetadata(source, sourcePath) {
  const metadata = {};
  for (const line of source.split('\n')) {
    const match = line.match(/^\/\/\s*@([a-z-]+):\s*(.+?)\s*$/);
    if (match) metadata[match[1]] = match[2];
  }

  for (const key of ['title', 'description']) {
    if (!metadata[key]) {
      throw new Error(`${relative(websiteRoot, sourcePath)} is missing // @${key}: metadata`);
    }
  }
  return metadata;
}

function escapeXml(value) {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&apos;');
}

function makeAccessible(svg, metadata, id) {
  const titleId = `${id}-title`;
  const descriptionId = `${id}-description`;
  if (!svg.includes('<svg')) {
    throw new Error(`dot returned an unexpected SVG document for ${id}`);
  }

  const svgStart = svg.indexOf('<svg');
  const svgStartEnd = svg.indexOf('>', svgStart);
  const before = '<!-- Generated from the adjacent pinned DOT source by website/scripts/render-diagrams.mjs. -->\n';
  let opening = svg.slice(svgStart, svgStartEnd + 1);
  let body = svg.slice(svgStartEnd + 1);

  opening = opening
    .replace(/\s+width="[^"]*"/, '')
    .replace(/\s+height="[^"]*"/, '')
    .replace(/\s+xmlns:xlink="[^"]*"/, '');
  const viewBox = opening.match(/viewBox="\s*[0-9.]+\s+[0-9.]+\s+([0-9.]+)\s+([0-9.]+)"/);
  if (!viewBox) throw new Error(`dot returned an SVG without a numeric viewBox for ${id}`);
  const naturalWidth = Math.ceil(Number(viewBox[1]));
  const naturalHeight = Math.ceil(Number(viewBox[2]));
  opening = opening.replace(
    /^<svg\b/,
    `<svg width="${naturalWidth}" height="${naturalHeight}" role="img" aria-labelledby="${titleId} ${descriptionId}" preserveAspectRatio="xMidYMid meet"`,
  );

  body = body.replace(/\s*<title>.*?<\/title>\s*/s, '\n');
  const accessibleText = [
    `\n<title id="${titleId}">${escapeXml(metadata.title)}</title>`,
    `<desc id="${descriptionId}">${escapeXml(metadata.description)}</desc>`,
  ].join('\n');

  return `${before}${opening}${accessibleText}${body}`;
}

function renderSequence(spec, outputRelative) {
  for (const key of ['title', 'description', 'participants', 'messages']) {
    if (!spec[key]) throw new Error(`${outputRelative} sequence source is missing ${key}`);
  }
  if (!Array.isArray(spec.participants) || !Array.isArray(spec.messages) || spec.messages.length === 0) {
    throw new Error(`${outputRelative} must define non-empty participant and message arrays`);
  }
  if (spec.participants.length < 2 || spec.participants.length > 6) {
    throw new Error(`${outputRelative} must use 2-6 participants; split denser sequences`);
  }

  const participantIds = new Set();
  for (const participant of spec.participants) {
    if (typeof participant.id !== 'string' || !participant.id ||
        typeof participant.label !== 'string' || !participant.label) {
      throw new Error(`${outputRelative} participants require non-empty id and label strings`);
    }
    if (participantIds.has(participant.id)) {
      throw new Error(`${outputRelative} contains duplicate participant id ${participant.id}`);
    }
    if (participant.color && !palette[participant.color]) {
      throw new Error(`${outputRelative} uses unknown participant color ${participant.color}`);
    }
    participantIds.add(participant.id);
  }

  const explicitNumbers = new Set();
  for (const message of spec.messages) {
    if (typeof message.label !== 'string' || !message.label) {
      throw new Error(`${outputRelative} messages require a non-empty label`);
    }
    if (message.kind === 'section') continue;
    if (!participantIds.has(message.from) || !participantIds.has(message.to)) {
      throw new Error(`${outputRelative} message references an unknown participant`);
    }
    if (message.style && !['optional', 'return'].includes(message.style)) {
      throw new Error(`${outputRelative} uses unknown message style ${message.style}`);
    }
    if (message.number !== undefined) {
      if (!Number.isInteger(message.number) || message.number <= 0 || explicitNumbers.has(message.number)) {
        throw new Error(`${outputRelative} message numbers must be unique positive integers`);
      }
      explicitNumbers.add(message.number);
    }
  }

  const width = Math.min(1200, Math.max(960, 225 + spec.participants.length * 175));
  const top = 172;
  const rowHeight = 70;
  const height = top + spec.messages.length * rowHeight + 76;
  const participantHalfWidth = 72;
  const left = participantHalfWidth + 16;
  const usable = width - left * 2;
  const spacing = usable / (spec.participants.length - 1);
  const xById = new Map(spec.participants.map((participant, index) => [participant.id, left + spacing * index]));
  const markerId = `${outputRelative.replaceAll(/[^a-zA-Z0-9_-]/g, '-')}-arrow`;
  const titleId = `${markerId}-title`;
  const descriptionId = `${markerId}-description`;

  const parts = [
    '<!-- Generated from the adjacent pinned sequence source by website/scripts/render-diagrams.mjs. -->',
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="${titleId} ${descriptionId}" preserveAspectRatio="xMidYMid meet">`,
    `<title id="${titleId}">${escapeXml(spec.title)}</title>`,
    `<desc id="${descriptionId}">${escapeXml(spec.description)}</desc>`,
    '<defs>',
    `  <marker id="${markerId}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 10 5 0 10Z" fill="#64748b"/></marker>`,
    '</defs>',
    `<rect x="1" y="1" width="${width - 2}" height="${height - 2}" rx="18" fill="#fbfcfe" stroke="#cbd5e1"/>`,
    `<text x="40" y="50" font-family="Inter, Arial, sans-serif" font-size="28" font-weight="800" fill="#111827">${escapeXml(spec.title)}</text>`,
  ];
  if (spec.subtitle) {
    parts.push(`<text x="40" y="78" font-family="Inter, Arial, sans-serif" font-size="17" fill="#475569">${escapeXml(spec.subtitle)}</text>`);
  }

  for (const participant of spec.participants) {
    const x = xById.get(participant.id);
    const colors = palette[participant.color || 'neutral'] || palette.neutral;
    parts.push(
      `<rect x="${x - participantHalfWidth}" y="106" width="${participantHalfWidth * 2}" height="52" rx="12" fill="${colors.fill}" stroke="${colors.stroke}" stroke-width="2.5"/>`,
      `<text x="${x}" y="138" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="17" font-weight="700" fill="#111827">${escapeXml(participant.label)}</text>`,
      `<line x1="${x}" y1="158" x2="${x}" y2="${height - 38}" stroke="#94a3b8" stroke-width="2" stroke-dasharray="7 6"/>`,
    );
  }

  let messageNumber = 0;
  spec.messages.forEach((message, index) => {
    const y = top + index * rowHeight;
    if (message.kind === 'section') {
      parts.push(
        `<rect x="40" y="${y - 25}" width="${width - 80}" height="42" rx="10" fill="#f1f5f9" stroke="#94a3b8"/>`,
        `<text x="58" y="${y + 2}" font-family="Inter, Arial, sans-serif" font-size="17" font-weight="700" fill="#334155">${escapeXml(message.label)}</text>`,
      );
      return;
    }

    messageNumber += 1;

    const from = xById.get(message.from);
    const to = xById.get(message.to);
    if (from === undefined || to === undefined) {
      throw new Error(`${outputRelative} message ${index + 1} references an unknown participant`);
    }
    const dashed = message.style === 'return' || message.style === 'optional';
    const dash = dashed ? ' stroke-dasharray="7 6"' : '';
    const label = `${message.number || messageNumber}. ${message.label}`;
    const estimatedLabelWidth = label.length * 8;

    if (from === to) {
      const availableLabelWidth = width - (from + 56) - 20;
      if (estimatedLabelWidth > availableLabelWidth) {
        throw new Error(`${outputRelative} self-message label is too wide: ${message.label}`);
      }
      parts.push(
        `<path d="M ${from} ${y} h 48 v 30 h -48" fill="none" stroke="#64748b" stroke-width="2.5"${dash} marker-end="url(#${markerId})"/>`,
        `<text x="${from + 56}" y="${y + 20}" font-family="Inter, Arial, sans-serif" font-size="16" fill="#111827">${escapeXml(label)}</text>`,
      );
      return;
    }

    const direction = to > from ? 1 : -1;
    const startX = from + direction * 8;
    const endX = to - direction * 8;
    const availableLabelWidth = Math.abs(to - from) - 24;
    if (estimatedLabelWidth > availableLabelWidth) {
      throw new Error(`${outputRelative} message label is too wide: ${message.label}`);
    }
    parts.push(
      `<line x1="${startX}" y1="${y}" x2="${endX}" y2="${y}" stroke="#64748b" stroke-width="2.5"${dash} marker-end="url(#${markerId})"/>`,
      `<rect x="${Math.min(from, to) + 12}" y="${y - 27}" width="${Math.abs(to - from) - 24}" height="23" rx="5" fill="#fbfcfe"/>`,
      `<text x="${(from + to) / 2}" y="${y - 10}" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="16" fill="#111827">${escapeXml(label)}</text>`,
    );
  });

  parts.push('</svg>', '');
  return parts.join('\n');
}

function validateSvg(svg, outputRelative) {
  if (!/<svg\b[^>]*role="img"/.test(svg) || !/aria-labelledby="[^"]+"/.test(svg)) {
    throw new Error(`${outputRelative} is missing accessible root SVG attributes`);
  }
  if (!/<title\b/.test(svg) || !/<desc\b/.test(svg)) {
    throw new Error(`${outputRelative} is missing title or description`);
  }
  const viewBox = svg.match(/viewBox="\s*([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)"/);
  if (!viewBox || Number(viewBox[3]) <= 0 || Number(viewBox[4]) <= 0 || Number(viewBox[3]) > 1200) {
    throw new Error(`${outputRelative} must have a positive viewBox no wider than 1200 units`);
  }
  const dimensions = svg.match(/<svg\b[^>]*\swidth="([0-9.]+)"[^>]*\sheight="([0-9.]+)"/);
  if (!dimensions || Number(dimensions[1]) <= 0 || Number(dimensions[2]) <= 0) {
    throw new Error(`${outputRelative} must declare positive intrinsic dimensions`);
  }
  if (/<(?:script|foreignObject)\b|\son[a-z]+\s*=|(?:href|src)\s*=/i.test(svg)) {
    throw new Error(`${outputRelative} contains unsafe or external SVG content`);
  }
  const fontSizes = [...svg.matchAll(/font-size="([0-9.]+)(?:px|pt)?"/g)].map((match) => Number(match[1]));
  if (fontSizes.some((size) => size < 15)) {
    throw new Error(`${outputRelative} contains text smaller than 15 units`);
  }
  const ids = [...svg.matchAll(/\sid="([^"]+)"/g)].map((match) => match[1]);
  if (new Set(ids).size !== ids.length) {
    throw new Error(`${outputRelative} contains duplicate element IDs`);
  }
}

const viz = await instance();

function render(sourcePath) {
  const source = readFileSync(sourcePath, 'utf8');
  const metadata = readMetadata(source, sourcePath);
  const relativeSource = relative(sourceRoot, sourcePath);
  const outputRelative = relativeSource.replace(/\.dot\.txt$/, '.svg');
  const outputPath = join(outputRoot, outputRelative);
  const id = outputRelative.replaceAll(/[^a-zA-Z0-9_-]/g, '-');

  const rendered = viz.renderString(source, {engine: 'dot', format: 'svg'});
  const accessible = makeAccessible(rendered, metadata, id);
  validateSvg(accessible, outputRelative);

  if (checkOnly) {
    if (!existsSync(outputPath)) {
      throw new Error(`Missing generated diagram: ${relative(websiteRoot, outputPath)}`);
    }
    const current = readFileSync(outputPath, 'utf8');
    if (current !== accessible) {
      throw new Error(`Stale generated diagram: ${relative(websiteRoot, outputPath)}`);
    }
    return outputRelative;
  }

  mkdirSync(dirname(outputPath), {recursive: true});
  writeFileSync(outputPath, accessible);
  return outputRelative;
}

function renderSequenceFile(sourcePath) {
  const spec = JSON.parse(readFileSync(sourcePath, 'utf8'));
  const relativeSource = relative(sourceRoot, sourcePath);
  const outputRelative = relativeSource.replace(/\.sequence\.json$/, '.svg');
  const outputPath = join(outputRoot, outputRelative);
  const rendered = renderSequence(spec, outputRelative);
  validateSvg(rendered, outputRelative);

  if (checkOnly) {
    if (!existsSync(outputPath)) throw new Error(`Missing generated diagram: ${relative(websiteRoot, outputPath)}`);
    if (readFileSync(outputPath, 'utf8') !== rendered) {
      throw new Error(`Stale generated diagram: ${relative(websiteRoot, outputPath)}`);
    }
    return outputRelative;
  }

  mkdirSync(dirname(outputPath), {recursive: true});
  writeFileSync(outputPath, rendered);
  return outputRelative;
}

function validateDocumentationUsage(generatedOutputs) {
  const documentationFiles = collectDocumentationFiles(documentationRoot);
  const documentation = documentationFiles
    .map((path) => readFileSync(path, 'utf8'))
    .join('\n');

  if (/^(?:```|~~~)mermaid(?:\s+.*)?$/im.test(documentation)) {
    throw new Error('Documentation still contains Mermaid blocks; use a generated SVG and the Diagram component');
  }

  const generatedSet = new Set(generatedOutputs);
  for (const outputRelative of generatedOutputs) {
    const publicPath = `/img/diagrams/${outputRelative}`;
    if (!documentation.includes(publicPath)) {
      throw new Error(`Generated diagram is not referenced by documentation: ${publicPath}`);
    }
  }

  const referencedOutputs = new Set(
    [...documentation.matchAll(/\/img\/diagrams\/([a-zA-Z0-9._/-]+\.svg)/g)]
      .map((match) => match[1]),
  );
  for (const outputRelative of referencedOutputs) {
    if (!generatedSet.has(outputRelative)) {
      throw new Error(`Documentation references a diagram without a generated source: /img/diagrams/${outputRelative}`);
    }
  }

  for (const svgPath of collectSvgFiles(outputRoot)) {
    const svg = readFileSync(svgPath, 'utf8');
    const isGenerated = svg.startsWith('<!-- Generated from the adjacent pinned');
    const outputRelative = relative(outputRoot, svgPath);
    if (isGenerated && !generatedSet.has(outputRelative)) {
      throw new Error(`Generated diagram has no source: ${relative(websiteRoot, svgPath)}`);
    }
  }
}

const dotFiles = collectDotFiles(sourceRoot);
const sequenceFiles = collectSequenceFiles(sourceRoot);
if (dotFiles.length + sequenceFiles.length === 0) {
  throw new Error(`No diagram sources found below ${relative(websiteRoot, sourceRoot)}`);
}

const generatedOutputs = [
  ...dotFiles.map(render),
  ...sequenceFiles.map(renderSequenceFile),
];
validateDocumentationUsage(generatedOutputs);
console.log(`${checkOnly ? 'Verified' : 'Rendered'} ${dotFiles.length + sequenceFiles.length} SVG diagrams.`);
