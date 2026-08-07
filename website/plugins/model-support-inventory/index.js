/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

const fs = require('fs');
const path = require('path');

const HF_TASKS = [
  {
    slug: 'any-to-any',
    label: 'Any-to-Any',
    category: 'Multimodal',
    description: 'Models that accept or produce more than one modality through one task contract.',
    hfUrl: 'https://huggingface.co/tasks/any-to-any',
  },
  {
    slug: 'image-text-to-text',
    label: 'Image-Text-to-Text',
    category: 'Multimodal',
    description: 'Vision-language models that answer or generate text from an image and prompt.',
    hfUrl: 'https://huggingface.co/tasks/image-text-to-text',
  },
  {
    slug: 'text-generation',
    label: 'Text Generation',
    category: 'Natural Language Processing',
    description: 'Decoder, encoder-decoder, recurrent, and diffusion-style text generation.',
    hfUrl: 'https://huggingface.co/tasks/text-generation',
  },
  {
    slug: 'translation',
    label: 'Translation',
    category: 'Natural Language Processing',
    description: 'Text generation recipes whose declared user contract is translation.',
    hfUrl: 'https://huggingface.co/tasks/translation',
  },
  {
    slug: 'feature-extraction',
    label: 'Feature Extraction',
    category: 'Natural Language Processing',
    description: 'Encoder and embedding models that return vector representations.',
    hfUrl: 'https://huggingface.co/tasks/feature-extraction',
  },
  {
    slug: 'text-ranking',
    label: 'Text Ranking',
    category: 'Natural Language Processing',
    description: 'Models that score documents against a query.',
    hfUrl: 'https://huggingface.co/tasks/text-ranking',
  },
  {
    slug: 'image-classification',
    label: 'Image Classification',
    category: 'Computer Vision',
    description: 'Models that assign a class to an input image.',
    hfUrl: 'https://huggingface.co/tasks/image-classification',
  },
  {
    slug: 'image-segmentation',
    label: 'Image Segmentation',
    category: 'Computer Vision',
    description: 'Models that produce segmentation masks for an image.',
    hfUrl: 'https://huggingface.co/tasks/image-segmentation',
  },
  {
    slug: 'mask-generation',
    label: 'Mask Generation',
    category: 'Computer Vision',
    description: 'Prompted segmentation models that generate masks from points or text.',
    hfUrl: 'https://huggingface.co/tasks/mask-generation',
  },
  {
    slug: 'text-to-image',
    label: 'Text-to-Image',
    category: 'Computer Vision',
    description: 'Diffusion recipes that generate an image from text.',
    hfUrl: 'https://huggingface.co/tasks/text-to-image',
  },
  {
    slug: 'image-to-image',
    label: 'Image-to-Image',
    category: 'Computer Vision',
    description: 'Image-conditioned generation and editing recipes.',
    hfUrl: 'https://huggingface.co/tasks/image-to-image',
  },
  {
    slug: 'text-to-video',
    label: 'Text-to-Video',
    category: 'Computer Vision',
    description: 'Diffusion recipes that generate video frames from text.',
    hfUrl: 'https://huggingface.co/tasks/text-to-video',
  },
  {
    slug: 'image-to-video',
    label: 'Image-to-Video',
    category: 'Computer Vision',
    description: 'Image-conditioned video and world-model generation recipes.',
    hfUrl: 'https://huggingface.co/tasks/image-to-video',
  },
  {
    slug: 'automatic-speech-recognition',
    label: 'Automatic Speech Recognition',
    category: 'Audio',
    description: 'Speech-to-text models, including offline and streaming contracts.',
    hfUrl: 'https://huggingface.co/tasks/automatic-speech-recognition',
  },
  {
    slug: 'text-to-speech',
    label: 'Text-to-Speech',
    category: 'Audio',
    description: 'Models that synthesize audio from text.',
    hfUrl: 'https://huggingface.co/tasks/text-to-speech',
  },
  {
    slug: 'audio-to-audio',
    label: 'Audio-to-Audio',
    category: 'Audio',
    description: 'Speech-to-speech models that consume and generate audio.',
    hfUrl: 'https://huggingface.co/tasks/audio-to-audio',
  },
  {
    slug: 'time-series-forecasting',
    label: 'Time Series Forecasting',
    category: 'Time Series',
    description: 'Forecasting and neural-operator models that consume numerical sequences.',
    hfUrl: 'https://huggingface.co/models?pipeline_tag=time-series-forecasting',
  },
];

const TASK_BY_SLUG = new Map(HF_TASKS.map((task) => [task.slug, task]));

const CLI_COMMANDS_BY_TASK_STRATEGY = {
  diffusion_text_generation: ['run'],
  embedding: ['embed'],
  encoder_only_nlp: ['encode'],
  image_classification: ['classify'],
  neural_operator: ['solve'],
  omni_multimodal: ['generate-audio'],
  prompted_segmentation: ['segment-prompted'],
  reranking: ['rerank'],
  segmentation: ['segment'],
  speech_to_speech: ['speak'],
  speech_to_text: ['transcribe'],
  text_generation_causal: ['run'],
  text_to_audio: ['generate-audio'],
  vision_language_generation: ['run'],
};

function readDirectory(directory, label) {
  let entries;
  try {
    entries = fs.readdirSync(directory, {withFileTypes: true});
  } catch (error) {
    throw new Error(`Unable to read ${label} at ${directory}: ${error.message}`);
  }
  return entries;
}

function readJson(filePath) {
  try {
    return JSON.parse(fs.readFileSync(filePath, 'utf8'));
  } catch (error) {
    throw new Error(`Unable to parse E2E manifest ${filePath}: ${error.message}`);
  }
}

function collectHfModelMetadata(repoRoot) {
  const metadataPath = path.join(repoRoot, 'website', 'data', 'hf-model-metadata.json');
  let catalog;
  try {
    catalog = JSON.parse(fs.readFileSync(metadataPath, 'utf8'));
  } catch (error) {
    throw new Error(`Unable to parse Hugging Face model metadata ${metadataPath}: ${error.message}`);
  }
  if (catalog.schema_version !== 1 || !Array.isArray(catalog.checkpoints)) {
    throw new Error(`Unsupported Hugging Face model metadata schema in ${metadataPath}`);
  }

  const byHfId = new Map();
  for (const checkpoint of catalog.checkpoints) {
    if (
      !checkpoint ||
      typeof checkpoint.hf_id !== 'string' ||
      !checkpoint.hf_id ||
      !/^[0-9a-f]{40}$/.test(checkpoint.revision) ||
      !['declared', 'resolved'].includes(checkpoint.revision_source) ||
      (checkpoint.metadata_file !== null && typeof checkpoint.metadata_file !== 'string') ||
      (checkpoint.model_type !== null && typeof checkpoint.model_type !== 'string') ||
      !Array.isArray(checkpoint.architectures) ||
      checkpoint.architectures.some(
        (architecture) => typeof architecture !== 'string' || !architecture
      ) ||
      (checkpoint.architecture_source !== null &&
        typeof checkpoint.architecture_source !== 'string') ||
      (checkpoint.architectures.length > 0 &&
        (!checkpoint.metadata_file || !checkpoint.architecture_source)) ||
      (checkpoint.architectures.length === 0 && checkpoint.architecture_source !== null)
    ) {
      throw new Error(`Malformed Hugging Face model metadata entry in ${metadataPath}`);
    }
    if (byHfId.has(checkpoint.hf_id)) {
      throw new Error(`Duplicate Hugging Face model metadata for ${checkpoint.hf_id}`);
    }
    byHfId.set(checkpoint.hf_id, checkpoint);
  }
  return byHfId;
}

function hfMetadataFields(metadata) {
  return {
    hfModelType: metadata.model_type || 'not declared',
    hfArchitectures: metadata.architectures,
    hfArchitectureSource: metadata.architecture_source || 'not declared',
    hfMetadataRevision: metadata.revision,
    hfMetadataRevisionSource: metadata.revision_source,
    hfMetadataFile: metadata.metadata_file || 'not declared',
  };
}

function markdownCellLines(value) {
  return value
    .split(/<br\s*\/?\s*>/i)
    .map((line) => line.replace(/`([^`]*)`/g, '$1').trim())
    .filter(Boolean);
}

function collectPerformanceSnapshot(repoRoot) {
  const matrixPath = path.join(repoRoot, 'website', 'data', 'model-support-matrix.md');
  const source = fs.readFileSync(matrixPath, 'utf8');
  const header =
    '| Hugging Face model ID (`hf_id`, CLI input) | TRTMC profile | Build precision | Quantization | Platform specialization runtime provider | GB300 |';
  const headerIndex = source.indexOf(header);
  if (headerIndex === -1) {
    throw new Error(`Unable to find the supported-model table in ${matrixPath}`);
  }

  const tableLines = source
    .slice(headerIndex)
    .split(/\r?\n/)
    .filter((line, index) => index < 2 || line.startsWith('| '));
  const rows = [];
  for (const line of tableLines.slice(2)) {
    if (!line.startsWith('| ')) break;
    const cells = line.slice(1, -1).split('|').map((cell) => cell.trim());
    if (cells.length !== 6) {
      throw new Error(`Malformed supported-model row in ${readmePath}: ${line}`);
    }
    const [checkpointCell, profileCell, precisionCell, quantizationCell, providerCell, resultCell] =
      cells;
    const checkpointLines = markdownCellLines(checkpointCell);
    const revisionLine = checkpointLines.find((value) => value.startsWith('Revision: '));
    rows.push({
      hfId: checkpointLines[0],
      revision: revisionLine ? revisionLine.slice('Revision: '.length) : 'not pinned',
      profile: markdownCellLines(profileCell)[0],
      precision: markdownCellLines(precisionCell),
      quantization: markdownCellLines(quantizationCell),
      platformSpecialization: markdownCellLines(providerCell),
      performance: markdownCellLines(resultCell)[0],
    });
  }
  if (rows.length === 0) {
    throw new Error(`Supported-model table in ${matrixPath} has no data rows`);
  }
  return rows;
}

function testcaseHasImageInput(testcase) {
  const inputs = testcase.inputs || {};
  return Boolean(
    testcase.test_image || inputs.image || inputs.test_image || inputs.image_path
  );
}

function testcaseGeneratesVideo(testcase) {
  return (
    Number(testcase.video_num_frames || testcase.inputs?.video_num_frames || 1) > 1 ||
    String(testcase.user_contract || '').includes('video')
  );
}

function hfTasksForManifest(manifest) {
  const testcases = Array.isArray(manifest.testcases) ? manifest.testcases : [];
  switch (manifest.task_strategy) {
    case 'text_generation_causal':
      return testcases.some((testcase) => testcase.user_contract === 'translation')
        ? ['translation']
        : ['text-generation'];
    case 'diffusion_text_generation':
      return ['text-generation'];
    case 'encoder_only_nlp':
    case 'embedding':
      return ['feature-extraction'];
    case 'reranking':
      return ['text-ranking'];
    case 'vision_language_generation':
      return ['image-text-to-text'];
    case 'omni_multimodal':
      return ['any-to-any'];
    case 'speech_to_text':
      return ['automatic-speech-recognition'];
    case 'text_to_audio':
      return ['text-to-speech'];
    case 'speech_to_speech':
      return ['audio-to-audio'];
    case 'image_classification':
      return ['image-classification'];
    case 'segmentation':
      return ['image-segmentation'];
    case 'prompted_segmentation':
      return ['mask-generation'];
    case 'neural_operator':
      return ['time-series-forecasting'];
    case 'diffusion_media_generation': {
      const tasks = new Set();
      for (const testcase of testcases.length > 0 ? testcases : [{}]) {
        const video = testcaseGeneratesVideo(testcase);
        const imageInput = testcaseHasImageInput(testcase);
        tasks.add(video ? (imageInput ? 'image-to-video' : 'text-to-video') :
          (imageInput ? 'image-to-image' : 'text-to-image'));
      }
      return [...tasks].sort();
    }
    default:
      throw new Error(
        `No Hugging Face task mapping for task_strategy=${manifest.task_strategy}`
      );
  }
}

function cliCommandsForManifest(manifest, hfTasks) {
  if (manifest.task_strategy === 'diffusion_media_generation') {
    const commands = new Set();
    if (hfTasks.some((task) => task.endsWith('-video'))) commands.add('generate-video');
    if (hfTasks.some((task) => task.endsWith('-image'))) commands.add('run');
    return [...commands];
  }
  const commands = CLI_COMMANDS_BY_TASK_STRATEGY[manifest.task_strategy];
  if (!commands) {
    throw new Error(`No CLI mapping for task_strategy=${manifest.task_strategy}`);
  }
  return commands;
}

function familySlug(family) {
  return family.replace(/_/g, '-');
}

function layerLabel(layer) {
  return layer
    .toLowerCase()
    .split('_')
    .map((word) => word[0].toUpperCase() + word.slice(1))
    .join(' ');
}

function parsePythonDefault(expression) {
  const value = expression.replace(/#.*$/gm, '').trim();
  if (value === 'True') return 'true';
  if (value === 'False') return 'false';
  return /^-?[\d_]+(?:\.[\d_]+)?$/.test(value) ? value.replace(/_/g, '') : value;
}

function parsePythonConfigSchemas(repoRoot, sourcePath) {
  const absolutePath = path.join(repoRoot, sourcePath);
  if (!fs.existsSync(absolutePath)) return [];
  const source = fs.readFileSync(absolutePath, 'utf8');
  const layerSets = new Map();
  for (const match of source.matchAll(
    /^(_?[A-Z][A-Z0-9_]*)\s*=\s*frozenset\(\{([\s\S]*?)\}\)/gm
  )) {
    layerSets.set(
      match[1],
      [...match[2].matchAll(/Layer\.([A-Z_]+)/g)].map((layer) => layerLabel(layer[1]))
    );
  }

  const schemas = [];
  const lines = source.split(/\r?\n/);
  let currentSchema = null;
  for (let index = 0; index < lines.length; index += 1) {
    if (/\bSchema\($/.test(lines[index].trim())) currentSchema = {namespace: null, fields: []};
    if (!currentSchema) continue;
    const namespace = lines[index].match(/^\s*namespace="([^"]+)"/);
    if (namespace) {
      currentSchema.namespace = namespace[1];
      schemas.push(currentSchema);
    }
    if (!lines[index].includes('ConfigField(') || !currentSchema.namespace) continue;

    const fieldLines = [];
    let depth = 0;
    do {
      const line = lines[index];
      fieldLines.push(line);
      depth += (line.match(/\(/g) || []).length - (line.match(/\)/g) || []).length;
      index += 1;
    } while (index < lines.length && depth > 0);
    index -= 1;
    const fieldSource = fieldLines.join('\n');
    const cleanedFieldSource = fieldSource.replace(/#.*$/gm, '');
    const name = cleanedFieldSource.match(/name="([^"]+)"/);
    const type = cleanedFieldSource.match(/type_tag="([^"]+)"/);
    const defaultValue = cleanedFieldSource.match(/default=([\s\S]*?),\s*allowed_layers=/);
    const allowedLayers = cleanedFieldSource.match(/allowed_layers=(_?[A-Z][A-Z0-9_]*)/);
    if (!name || !type || !defaultValue || !allowedLayers) {
      throw new Error(`Unable to parse ConfigField in ${sourcePath}: ${fieldSource}`);
    }
    currentSchema.fields.push({
      name: name[1],
      key: `${currentSchema.namespace}.${name[1]}`,
      type: type[1],
      defaultValue: parsePythonDefault(defaultValue[1]),
      allowedLayers: layerSets.get(allowedLayers[1]) || [allowedLayers[1]],
    });
  }
  return schemas.map((schema) => ({...schema, sourcePath}));
}

function parseCppDefault(expression) {
  if (expression === 'std::string{}') return '""';
  const integer = expression.match(/std::(?:int32_t|int64_t)\{([^}]+)\}/);
  if (integer) return integer[1];
  return expression.replace(/F$/, '');
}

function parseCppConfigSchemas(repoRoot, sourcePath) {
  const absolutePath = path.join(repoRoot, sourcePath);
  if (!fs.existsSync(absolutePath)) return [];
  const source = fs.readFileSync(absolutePath, 'utf8');
  const layerSets = new Map();
  for (const match of source.matchAll(
    /const std::set<Layer>\s+(\w+)\s*=\s*\{([^}]+)\};/g
  )) {
    layerSets.set(
      match[1],
      [...match[2].matchAll(/Layer::([A-Za-z]+)/g)].map((layer) =>
        layer[1].replace(/([a-z])([A-Z])/g, '$1 $2')
      )
    );
  }

  const schemas = [];
  for (const schemaMatch of source.matchAll(
    /return Schema\{\s*"([^"]+)",\s*\{([\s\S]*?)\n\s*\},\s*\};/g
  )) {
    const namespace = schemaMatch[1];
    const fields = [];
    for (const fieldMatch of schemaMatch[2].matchAll(
      /ConfigField\{\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*std::any\{([\s\S]*?)\}\s*,\s*(\w+)\s*,/g
    )) {
      fields.push({
        name: fieldMatch[1],
        key: `${namespace}.${fieldMatch[1]}`,
        type: fieldMatch[2],
        defaultValue: parseCppDefault(fieldMatch[3].trim()),
        allowedLayers: layerSets.get(fieldMatch[4]) || [fieldMatch[4]],
      });
    }
    if (fields.length > 0) schemas.push({namespace, fields, sourcePath});
  }
  return schemas;
}

function collectFamilyConfigSchemas(repoRoot, family, runtimeOwners) {
  const schemas = parsePythonConfigSchemas(
    repoRoot,
    `python/tensorrt_model_connect/families/${family}/runtime_config_schema.py`
  );
  const namespaces = new Set(schemas.map((schema) => schema.namespace));
  for (const owner of runtimeOwners) {
    const metadataPath = path.join(repoRoot, 'src', 'runtime', 'models', owner, 'MODEL.toml');
    if (!fs.existsSync(metadataPath)) continue;
    const metadata = fs.readFileSync(metadataPath, 'utf8');
    const schemaArray = metadata.match(/^runtime_config_schemas\s*=\s*\[(.*?)\]/ms);
    if (!schemaArray) continue;
    for (const entry of schemaArray[1].matchAll(/"([^"|]+)(?:\|[^"]+)?"/g)) {
      const sourcePath = `src/runtime/models/${owner}/${entry[1]}`;
      for (const schema of parseCppConfigSchemas(repoRoot, sourcePath)) {
        if (!namespaces.has(schema.namespace)) {
          schemas.push(schema);
          namespaces.add(schema.namespace);
        }
      }
    }
  }
  return schemas.sort((left, right) => left.namespace.localeCompare(right.namespace));
}

const GENERATE_CONFIG_OPTIONS = {
  max_new_tokens: {
    flag: '--max-new-tokens <N>',
    description: 'Limit the number of generated tokens or audio frames.',
  },
  temperature: {
    flag: '--temperature <F>',
    description: 'Set sampling temperature.',
  },
  top_k: {
    flag: '--top-k <N>',
    description: 'Restrict sampling to the top K tokens.',
  },
  top_p: {
    flag: '--top-p <F>',
    description: 'Enable nucleus sampling at probability P.',
  },
  min_p: {
    flag: '--min-p <F>',
    description: 'Filter tokens below min-p times the maximum probability.',
  },
  seed: {
    flag: '--seed <N>',
    description: 'Set the sampling or diffusion seed.',
  },
  use_chat_template: {
    flag: '--chat-template',
    description: 'Apply the chat template packaged with the bundle.',
  },
  enable_thinking: {
    flag: '--no-thinking',
    description: 'Disable a supported reasoning or thinking mode.',
  },
  text_generation_mode: {
    flag: '--generation-mode <MODE>',
    description: 'Select the runtime-supported autoregressive or text-diffusion mode.',
  },
  block_length: {
    flag: '--block-length <N>',
    description: 'Set the text-diffusion or speculative generation block length.',
  },
  confidence_threshold: {
    flag: '--threshold <F>',
    description: 'Set the confidence threshold used by the selected generation mode.',
  },
  num_samples: {
    flag: '--num-samples <N>',
    description: 'Generate N independent text samples.',
  },
  num_steps: {
    flag: '--num-steps <N>',
    description: 'Set the number of diffusion or flow-matching steps.',
  },
  guidance_scale: {
    flag: '--guidance-scale <F>',
    description: 'Set classifier-free guidance strength.',
  },
  cfg_scale: {
    flag: '--cfg-scale <F>',
    description: 'Set the conditional classifier-free guidance scale.',
  },
  sde_gamma: {
    flag: '--sde-gamma <F>',
    description: 'Set the SDE or flow-matching gamma override.',
  },
  initial_latents: {
    flag: '--initial-latents-raw <PATH>',
    description: 'Load packed float32 initial latents.',
  },
  condition_latents: {
    flag: '--condition-latents-raw <PATH>',
    description: 'Load packed float32 conditioning latents.',
  },
  condition_mask: {
    flag: '--condition-mask-raw <PATH>',
    description: 'Load a float32 conditioning mask.',
  },
  sampling_steps: {
    flag: '--sampling-steps-raw <PATH>',
    description: 'Load an explicit float32 sampling schedule.',
  },
  sde_noises: {
    flag: '--sde-noise-raw <PATH>',
    description: 'Load precomputed float32 SDE noise.',
  },
  negative_prompt: {
    flag: '--negative-prompt <TEXT>',
    description: 'Override the bundle default negative prompt.',
  },
  height: {
    flag: '--height <N>',
    description: 'Override the generated image or video height.',
  },
  width: {
    flag: '--width <N>',
    description: 'Override the generated image or video width.',
  },
  tail_frames: {
    flag: '--tail-frames <N>',
    description: 'Generate extra speech frames after the input audio ends.',
  },
  lora_adapter_id: {
    flag: '--lora-adapter <DIR> --lora-adapter-id <ID>',
    description: 'Load and select a dynamic LoRA adapter.',
  },
  source_language_token_id: {
    flag: '--source-language-token-id <N>',
    description: 'Set the source-language token used by a multilingual encoder-decoder runtime.',
  },
  forced_bos_token_id: {
    flag: '--forced-bos-token-id <N>',
    description: 'Force the first decoder token for a multilingual encoder-decoder runtime.',
  },
};

const CAUSAL_GENERATION_FIELDS = [
  'max_new_tokens', 'temperature', 'top_k', 'top_p', 'min_p', 'seed',
  'use_chat_template', 'enable_thinking', 'lora_adapter_id',
  'source_language_token_id', 'forced_bos_token_id',
];
const TEXT_DIFFUSION_FIELDS = [
  ...CAUSAL_GENERATION_FIELDS,
  'text_generation_mode', 'block_length', 'confidence_threshold', 'num_samples',
  'num_steps', 'guidance_scale', 'cfg_scale', 'sde_gamma', 'initial_latents',
  'condition_latents', 'condition_mask', 'sampling_steps', 'sde_noises',
];
const MEDIA_GENERATION_FIELDS = [
  'seed', 'num_steps', 'guidance_scale', 'cfg_scale', 'initial_latents',
  'negative_prompt', 'height', 'width',
];

const TRANSCRIPTION_CONFIG_OPTIONS = {
  max_output_tokens: {
    flag: '--max-new-tokens <N>',
    description: 'Set the maximum number of decoder output tokens.',
  },
  beam_size: {
    flag: '--beam-size <N>',
    description: 'Select greedy decoding at 1 or beam search above 1.',
  },
  length_penalty: {
    flag: '--length-penalty <F>',
    description: 'Set the beam-search length-normalization exponent.',
  },
  beam_fallback_max_size: {
    flag: '--beam-fallback-max-size <N>',
    description: 'Retry an unterminated decode with larger beams up to N.',
  },
  source_language: {
    flag: '--source-language <TAG>',
    description: 'Set the source language for a multilingual transcription request.',
  },
  target_language: {
    flag: '--target-language <TAG>',
    description: 'Set the target language for a multilingual transcription request.',
  },
  task: {
    flag: '--task <transcribe|translate>',
    description: 'Select transcription or speech translation.',
  },
  punctuation: {
    flag: '--punctuation | --no-punctuation',
    description: 'Enable or disable punctuation in the result.',
  },
  timestamps: {
    flag: '--timestamps',
    description: 'Return timestamped transcription segments.',
  },
  max_input_duration_seconds: {
    flag: '--max-input-seconds <F>',
    description: 'Reject input longer than the specified duration.',
  },
  segment_duration_seconds: {
    flag: '--segment-length-seconds <F>',
    description: 'Split long audio into segments no longer than this duration.',
  },
  segment_min_duration_seconds: {
    flag: '--segment-min-seconds <F>',
    description: 'Set the minimum duration for dynamic segmentation windows.',
  },
  segment_overlap_seconds: {
    flag: '--segment-overlap-seconds <F>',
    description: 'Set overlap between adjacent transcription segments.',
  },
  lcs_merge: {
    flag: '--lcs-merge',
    description: 'Merge overlapping segment tokens with boundary-constrained LCS.',
  },
};

const TRANSCRIPTION_STREAM_CONFIG_OPTIONS = {
  att_context_left: {
    flag: '--att-context-size <L,R>',
    description: 'Set left and right streaming attention context.',
  },
  pad_and_drop_preencoded: {
    flag: '--pad-and-drop-preencoded',
    description: 'Enable the runtime pre-encoded padding and drop path.',
  },
  language: {
    flag: '--language <TAG>',
    description: 'Select a language from the bundle prompt dictionary.',
  },
};

function runtimeSourceFiles(repoRoot, owner) {
  const directory = path.join(repoRoot, 'src', 'runtime', 'models', owner);
  if (!fs.existsSync(directory)) return [];
  return readDirectory(directory, `runtime implementation for ${owner}`)
    .filter(
      (entry) => entry.isFile() && /\.(?:h|hpp|cpp|cc|cu)$/.test(entry.name)
    )
    .map((entry) => ({
      sourcePath: `src/runtime/models/${owner}/${entry.name}`,
      source: fs.readFileSync(path.join(directory, entry.name), 'utf8'),
    }));
}

function matchingSourcePaths(files, expression) {
  return files
    .filter((file) => expression.test(file.source))
    .map((file) => file.sourcePath)
    .sort();
}

function functionBodiesWithTypedConfig(file, typeName) {
  const bodies = [];
  const signature = new RegExp(
    `\\([^;{}]*const\\s+${typeName}\\s*&\\s*(\\w+)[^;{}]*\\)\\s*(?:const\\s*)?(?:override\\s*)?\\{`,
    'g'
  );
  for (const match of file.source.matchAll(signature)) {
    const bodyStart = match.index + match[0].length - 1;
    let depth = 0;
    for (let index = bodyStart; index < file.source.length; index += 1) {
      if (file.source[index] === '{') depth += 1;
      if (file.source[index] === '}') depth -= 1;
      if (depth === 0) {
        bodies.push({
          parameterName: match[1],
          body: file.source.slice(bodyStart, index + 1),
        });
        break;
      }
    }
  }
  return bodies;
}

function collectRuntimeCapabilities(repoRoot, owner) {
  const files = runtimeSourceFiles(repoRoot, owner);
  const combinedSource = files.map((file) => file.source).join('\n');
  const generateConfigFields = new Set();
  const transcriptionConfigFields = new Set();
  const transcriptionStreamConfigFields = new Set();
  const configEvidence = new Set();
  for (const file of files) {
    const generateBodies = functionBodiesWithTypedConfig(file, 'GenerateConfig');
    for (const {parameterName, body} of generateBodies) {
      let bodyUsesPublicConfig = false;
      const fieldExpression = new RegExp(`\\b${parameterName}\\.(\\w+)`, 'g');
      for (const match of body.matchAll(fieldExpression)) {
        if (Object.hasOwn(GENERATE_CONFIG_OPTIONS, match[1])) {
          generateConfigFields.add(match[1]);
          bodyUsesPublicConfig = true;
        }
      }
      if (bodyUsesPublicConfig) configEvidence.add(file.sourcePath);
    }
    for (const [typeName, definitions, fields] of [
      ['TranscriptionConfig', TRANSCRIPTION_CONFIG_OPTIONS, transcriptionConfigFields],
      ['TranscriptionStreamConfig', TRANSCRIPTION_STREAM_CONFIG_OPTIONS, transcriptionStreamConfigFields],
    ]) {
      for (const {parameterName, body} of functionBodiesWithTypedConfig(file, typeName)) {
        const fieldExpression = new RegExp(`\\b${parameterName}\\.(\\w+)`, 'g');
        let bodyUsesPublicConfig = false;
        for (const match of body.matchAll(fieldExpression)) {
          if (Object.hasOwn(definitions, match[1])) {
            fields.add(match[1]);
            bodyUsesPublicConfig = true;
          }
        }
        if (bodyUsesPublicConfig) configEvidence.add(file.sourcePath);
      }
    }
  }

  const textImageInputExpression =
    /TextResult\s+generate\s*\([\s\S]{0,240}?const\s+float\s*\*\s*image_pixels/;
  const imageGenerationExpression = /ImageResult\s+generate_image\s*\(/;
  const imageConditioningExpression =
    /ImageResult\s+generate_image\s*\([\s\S]{0,240}?const\s+float\s*\*\s*image_pixels/;
  const streamingAudioExpression =
    /int32_t\s+generate_audio_streaming\s*\(\s*const\s+std::string&/;
  const streamingTranscriptionExpression =
    /create_transcription_stream\s*\([\s\S]{0,160}?\)\s*override/;
  const promptedPointExpression =
    /PromptedSegmentationResult\s+segment_prompted\s*\(/;
  const promptedTextExpression =
    /PromptedSegmentationResult\s+segment_prompted_text\s*\(/;
  const loraExpression = /supports_lora_adapters\s*\([^)]*\)\s*const\s*override/;

  const capabilityEvidence = new Set();
  for (const expression of [
    textImageInputExpression,
    imageGenerationExpression,
    imageConditioningExpression,
    streamingAudioExpression,
    streamingTranscriptionExpression,
    promptedPointExpression,
    promptedTextExpression,
    loraExpression,
  ]) {
    matchingSourcePaths(files, expression).forEach((sourcePath) => capabilityEvidence.add(sourcePath));
  }
  configEvidence.forEach((sourcePath) => capabilityEvidence.add(sourcePath));

  return {
    owner,
    textImageInput: textImageInputExpression.test(combinedSource),
    imageGeneration: imageGenerationExpression.test(combinedSource),
    imageConditioning: imageConditioningExpression.test(combinedSource),
    audioStreaming: streamingAudioExpression.test(combinedSource),
    transcriptionStreaming: streamingTranscriptionExpression.test(combinedSource),
    promptedPoint: promptedPointExpression.test(combinedSource),
    promptedText: promptedTextExpression.test(combinedSource),
    dynamicLora: loraExpression.test(combinedSource),
    generateConfigFields: [...generateConfigFields].sort(),
    transcriptionConfigFields: [...transcriptionConfigFields].sort(),
    transcriptionStreamConfigFields: [...transcriptionStreamConfigFields].sort(),
    sourcePaths: [...capabilityEvidence].sort(),
  };
}

function mergeRuntimeCapabilities(capabilities) {
  const merged = {
    owners: [],
    textImageInput: false,
    imageGeneration: false,
    imageConditioning: false,
    audioStreaming: false,
    transcriptionStreaming: false,
    promptedPoint: false,
    promptedText: false,
    dynamicLora: false,
    generateConfigFields: new Set(),
    transcriptionConfigFields: new Set(),
    transcriptionStreamConfigFields: new Set(),
    sourcePaths: new Set(),
  };
  for (const capability of capabilities) {
    merged.owners.push(capability.owner);
    for (const key of [
      'textImageInput', 'imageGeneration', 'imageConditioning', 'audioStreaming',
      'transcriptionStreaming', 'promptedPoint', 'promptedText', 'dynamicLora',
    ]) {
      merged[key] ||= capability[key];
    }
    capability.generateConfigFields.forEach((field) => merged.generateConfigFields.add(field));
    capability.transcriptionConfigFields.forEach((field) => merged.transcriptionConfigFields.add(field));
    capability.transcriptionStreamConfigFields.forEach((field) => merged.transcriptionStreamConfigFields.add(field));
    capability.sourcePaths.forEach((sourcePath) => merged.sourcePaths.add(sourcePath));
  }
  return {
    ...merged,
    owners: [...new Set(merged.owners)].sort(),
    generateConfigFields: [...merged.generateConfigFields].sort(),
    transcriptionConfigFields: [...merged.transcriptionConfigFields].sort(),
    transcriptionStreamConfigFields: [...merged.transcriptionStreamConfigFields].sort(),
    sourcePaths: [...merged.sourcePaths].sort(),
  };
}

function option(flag, requirement, description) {
  return {flag, requirement, description};
}

function generateOptions(capability, allowedFields) {
  const implementedFields = new Set(capability.generateConfigFields);
  return allowedFields
    .filter(
      (field) => implementedFields.has(field) &&
        (field !== 'lora_adapter_id' || capability.dynamicLora)
    )
    .map((field) => option(
      GENERATE_CONFIG_OPTIONS[field].flag,
      'Optional',
      GENERATE_CONFIG_OPTIONS[field].description
    ));
}

function typedConfigOptions(implementedFields, definitions, requirement = 'Optional') {
  const supported = new Set(implementedFields);
  return Object.entries(definitions)
    .filter(([field]) => supported.has(field))
    .map(([, definition]) => option(definition.flag, requirement, definition.description));
}

function commandContractForProfile(profile, capability) {
  const evidence = [
    'src/cli/main.cpp',
    'include/trtmc/pipeline.h',
    profile.sourcePath,
    ...capability.sourcePaths,
  ].filter((value, index, values) => values.indexOf(value) === index);

  switch (profile.taskStrategy) {
    case 'text_generation_causal':
      return {
        command: 'run',
        purpose: 'Generate text from a text prompt.',
        syntax: 'trtmc run <bundle.trtfb> --prompt "<text>" [generation options]',
        options: [
          option('--prompt <TEXT>', 'Required', 'Text input for this causal language-model recipe.'),
          ...generateOptions(capability, CAUSAL_GENERATION_FIELDS),
          ...(capability.generateConfigFields.includes('temperature')
            ? [option('--greedy', 'Optional', 'Select deterministic greedy decoding by setting temperature to zero.')]
            : []),
          option('--num-samples <N>', 'Optional', 'Run N independent text generations.'),
          option('--output <PATH>', 'Optional', 'Write generated samples as JSON Lines.'),
          option('--benchmark <N>', 'Optional', 'Run N timed generation iterations.'),
          option('--warmup <N>', 'Optional with --benchmark', 'Warm-up iterations before generation timing.'),
        ],
        evidence,
      };
    case 'vision_language_generation': {
      const imageOptions = capability.textImageInput
        ? [option('--image <PATH>', 'Required by declared image recipes', 'Image input consumed by the runtime image-aware generate overload.')]
        : [];
      const imageSyntax = capability.textImageInput ? ' --image <input.png>' : '';
      return {
        command: 'run',
        purpose: capability.textImageInput
          ? 'Generate text from an image and text prompt.'
          : 'Generate text from the inputs implemented by this runtime.',
        syntax: `trtmc run <bundle.trtfb> --prompt "<text>"${imageSyntax} [generation options]`,
        options: [
          option('--prompt <TEXT>', 'Required', 'Text prompt for the vision-language recipe.'),
          ...imageOptions,
          ...generateOptions(capability, CAUSAL_GENERATION_FIELDS),
          ...(capability.generateConfigFields.includes('temperature')
            ? [option('--greedy', 'Optional', 'Select deterministic greedy decoding by setting temperature to zero.')]
            : []),
        ],
        evidence,
      };
    }
    case 'diffusion_text_generation':
      return {
        command: 'run',
        purpose: 'Generate text with the family\'s diffusion-style text runtime.',
        syntax: 'trtmc run <bundle.trtfb> --prompt "<text>" [text-diffusion options]',
        options: [
          option('--prompt <TEXT>', 'Required unless initial latents are supplied', 'Text conditioning input.'),
          ...generateOptions(capability, TEXT_DIFFUSION_FIELDS),
          option('--output <PATH>', 'Optional', 'Write generated samples as JSON Lines.'),
        ],
        evidence,
      };
    case 'diffusion_media_generation': {
      const video = profile.hfTasks.some((task) => task.endsWith('-video'));
      const hasImageInputRecipe = profile.hfTasks.some((task) => task.startsWith('image-to-'));
      const hasTextOnlyRecipe = profile.hfTasks.some((task) => task.startsWith('text-to-'));
      const imageInput = hasImageInputRecipe &&
        (video ? false : capability.imageConditioning);
      if (video) {
        return {
          command: 'generate-video',
          purpose: 'Generate video frames from a text prompt.',
          syntax: 'trtmc generate-video <bundle.trtfb> --prompt "<text>" --output <output-dir> [generation options]',
          options: [
            option('--prompt <TEXT>', 'Required', 'Text conditioning input.'),
            option('--output <DIR>', 'Optional', 'Directory for generated PNG frames.'),
            ...generateOptions(capability, MEDIA_GENERATION_FIELDS),
          ],
          evidence,
        };
      }
      return {
        command: 'run',
        purpose: imageInput
          ? 'Generate or edit an image from an input image and text prompt.'
          : 'Generate an image from a text prompt.',
        syntax: `trtmc run <bundle.trtfb> --prompt "<text>"${imageInput ? (hasTextOnlyRecipe ? ' [--image <input.png>]' : ' --image <input.png>') : ''} --output <output.png> [generation options]`,
        options: [
          option('--prompt <TEXT>', 'Required', 'Text conditioning input.'),
          ...(imageInput ? [option('--image <PATH>', hasTextOnlyRecipe ? 'Required for image-to-image; omit for text-to-image' : 'Required', 'Image conditioning input implemented by the runtime overload.')] : []),
          option('--output <PATH>', 'Optional', 'Output PNG path or directory.'),
          ...(!imageInput || hasTextOnlyRecipe ? [
            option('--num-images <N>', imageInput ? 'Text-to-image only' : 'Optional', 'Generate an image batch from one prompt.'),
            option('--prompts-file <PATH>', imageInput ? 'Text-to-image only' : 'Optional', 'Generate one image per line in a prompt file.'),
          ] : []),
          ...generateOptions(capability, MEDIA_GENERATION_FIELDS),
        ],
        evidence,
      };
    }
    case 'encoder_only_nlp':
      return {
        command: 'encode',
        purpose: 'Return the encoder hidden-state representation for text.',
        syntax: 'trtmc encode <bundle.trtfb> --prompt "<text>"',
        options: [option('--prompt <TEXT>', 'Required', 'Text input passed to the encoder.')],
        evidence,
      };
    case 'embedding':
      return {
        command: 'embed',
        purpose: 'Return an embedding for text.',
        syntax: 'trtmc embed <bundle.trtfb> --prompt "<text>"',
        options: [option('--prompt <TEXT>', 'Required', 'Text input passed to the embedding runtime.')],
        evidence,
      };
    case 'reranking':
      return {
        command: 'rerank',
        purpose: 'Score one document against a query.',
        syntax: 'trtmc rerank <bundle.trtfb> --prompt "<query>" --document "<document>"',
        options: [
          option('--prompt <TEXT>', 'Required', 'Query text.'),
          option('--document <TEXT>', 'Required', 'Candidate document text.'),
        ],
        evidence,
      };
    case 'image_classification':
      return {
        command: 'classify',
        purpose: 'Classify an input image.',
        syntax: 'trtmc classify <bundle.trtfb> --image <input.png>',
        options: [
          option('--image <PATH>', 'Required', 'Image input consumed by classify().'),
          option('--benchmark <N>', 'Optional', 'Run N timed classification iterations.'),
          option('--warmup <N>', 'Optional', 'Warm-up iterations before classification timing.'),
        ],
        evidence,
      };
    case 'segmentation':
      return {
        command: 'segment',
        purpose: 'Create a semantic segmentation mask for an image.',
        syntax: 'trtmc segment <bundle.trtfb> --image <input.png> --output <mask.png>',
        options: [
          option('--image <PATH>', 'Required', 'Image input consumed by segment().'),
          option('--output <PATH>', 'Optional', 'Output grayscale mask path.'),
        ],
        evidence,
      };
    case 'prompted_segmentation': {
      const promptOptions = [];
      if (capability.promptedText) {
        promptOptions.push(option('--prompt <TEXT>', 'One prompt mode', 'Text prompt consumed by segment_prompted_text().'));
      }
      if (capability.promptedPoint) {
        promptOptions.push(
          option('--point-x <F> --point-y <F>', 'One prompt mode', 'Normalized point coordinates consumed by segment_prompted().'),
          option('--background', 'Optional with a point', 'Treat the point as background instead of foreground.')
        );
      }
      return {
        command: 'segment-prompted',
        purpose: 'Create masks from an image and a runtime-supported prompt type.',
        syntax: `trtmc segment-prompted <bundle.trtfb> --image <input.png> --output <output-dir>${capability.promptedText ? ' --prompt "<object>"' : ' --point-x <F> --point-y <F>'}`,
        options: [
          option('--image <PATH>', 'Required', 'Image input.'),
          option('--output <DIR>', 'Optional', 'Directory for masks, scores, boxes, and overlay.'),
          ...promptOptions,
        ],
        evidence,
      };
    }
    case 'text_to_audio':
    case 'omni_multimodal':
      return {
        command: 'generate-audio',
        purpose: 'Generate audio from a text prompt.',
        syntax: 'trtmc generate-audio <bundle.trtfb> --prompt "<text>" --output <output.wav>',
        options: [
          option('--prompt <TEXT>', 'Required', 'Text input for audio generation.'),
          option('--output <PATH>', 'Optional', 'Output WAV path.'),
          ...generateOptions(capability, ['max_new_tokens']),
          ...(capability.audioStreaming ? [
            option('--stream', 'Optional', 'Stream raw float32 PCM while decoding.'),
            option('--chunk-frames <N>', 'Optional with --stream', 'Frames decoded per streaming chunk.'),
          ] : []),
        ],
        evidence,
      };
    case 'speech_to_speech':
      return {
        command: 'speak',
        purpose: 'Generate speech from input speech.',
        syntax: 'trtmc speak <bundle.trtfb> --audio-in <input.wav> --audio-out <output.wav>',
        options: [
          option('--audio-in <PATH>', 'Required', 'Input WAV file.'),
          option('--audio-out <PATH>', 'Optional', 'Output WAV file.'),
          ...generateOptions(capability, ['max_new_tokens', 'tail_frames']),
        ],
        evidence,
      };
    case 'speech_to_text': {
      const offlineOptions = typedConfigOptions(
        capability.transcriptionConfigFields,
        TRANSCRIPTION_CONFIG_OPTIONS
      );
      const hasMaxTokens = offlineOptions.some(
        (entry) => entry.flag === '--max-new-tokens <N>'
      );
      const streamOptions = capability.transcriptionStreaming
        ? typedConfigOptions(
          capability.transcriptionStreamConfigFields,
          TRANSCRIPTION_STREAM_CONFIG_OPTIONS,
          'Optional with --stream'
        )
        : [];
      return {
        command: 'transcribe',
        purpose: 'Transcribe an audio file.',
        syntax: `trtmc transcribe <bundle.trtfb> --audio <input.wav>${capability.transcriptionStreaming ? ' [--stream]' : ''}`,
        options: [
          option('--audio <PATH>', 'Required', 'Input WAV file; repeat for supported offline batches.'),
          ...(!hasMaxTokens ? [
            option('--max-new-tokens <N>', 'Optional', 'Maximum output tokens.'),
          ] : []),
          ...offlineOptions,
          ...(capability.transcriptionStreaming ? [
            option('--stream', 'Optional', 'Use the runtime streaming transcription implementation.'),
            option('--chunk-ms <N>', 'Optional with --stream', 'Streaming audio chunk duration.'),
            ...streamOptions,
          ] : []),
        ],
        evidence,
      };
    }
    case 'neural_operator':
      return {
        command: 'solve',
        purpose: 'Run the family\'s numerical forecasting or neural-operator contract.',
        syntax: 'trtmc solve <bundle.trtfb> --field-input <CSV>',
        options: [
          option('--field-input <CSV>', 'One input mode', 'Single numerical field input.'),
          option('--branch-input <CSV>', 'One input mode', 'Branch input for operator-style recipes.'),
          option('--trunk-input <CSV>', 'Optional with --branch-input', 'Trunk input for operator-style recipes.'),
        ],
        evidence,
      };
    default:
      throw new Error(`No family CLI contract for task_strategy=${profile.taskStrategy}`);
  }
}

function collectFamilyCommandContracts(repoRoot, profiles, runtimeOwnersByStrategy) {
  const capabilityByOwner = new Map();
  const grouped = new Map();
  for (const profile of profiles) {
    const owners = runtimeOwnersByStrategy.get(profile.runtimeStrategy) || [];
    const capabilities = owners.map((owner) => {
      if (!capabilityByOwner.has(owner)) {
        capabilityByOwner.set(owner, collectRuntimeCapabilities(repoRoot, owner));
      }
      return capabilityByOwner.get(owner);
    });
    const merged = mergeRuntimeCapabilities(capabilities);
    const contract = commandContractForProfile(profile, merged);
    const key = JSON.stringify({
      command: contract.command,
      purpose: contract.purpose,
      syntax: contract.syntax,
      options: contract.options,
      runtimeOwners: merged.owners,
    });
    if (!grouped.has(key)) {
      grouped.set(key, {...contract, runtimeOwners: merged.owners, profileNames: []});
    } else {
      contract.evidence.forEach((sourcePath) => {
        if (!grouped.get(key).evidence.includes(sourcePath)) {
          grouped.get(key).evidence.push(sourcePath);
        }
      });
      grouped.get(key).evidence.sort();
    }
    grouped.get(key).profileNames.push(profile.profile);
  }
  return [...grouped.values()].sort(
    (left, right) =>
      left.command.localeCompare(right.command) ||
      left.profileNames[0].localeCompare(right.profileNames[0])
  );
}

function collectRecipeCatalog(repoRoot, modelProfiles, runtimeOwnersByStrategy) {
  const families = new Map();
  for (const profile of modelProfiles) {
    if (!families.has(profile.family)) {
      families.set(profile.family, {
        family: profile.family,
        slug: familySlug(profile.family),
        profiles: [],
        taskSlugs: new Set(),
        cliCommands: new Set(['build', 'inspect']),
        hfIds: new Set(),
      });
    }
    const family = families.get(profile.family);
    family.profiles.push(profile);
    profile.hfTasks.forEach((task) => family.taskSlugs.add(task));
    profile.cliCommands.forEach((command) => family.cliCommands.add(command));
    family.hfIds.add(profile.hfId);
  }

  const familyRecipes = [...families.values()]
    .map((family) => {
      const runtimeOwners = [...new Set(
        family.profiles.flatMap((profile) => runtimeOwnersByStrategy.get(profile.runtimeStrategy) || [])
      )].sort();
      return {
        ...family,
        profiles: family.profiles.sort((left, right) => left.profile.localeCompare(right.profile)),
        taskSlugs: [...family.taskSlugs].sort(),
        cliCommands: [...family.cliCommands].sort(),
        hfIds: [...family.hfIds].sort(),
        configSchemas: collectFamilyConfigSchemas(repoRoot, family.family, runtimeOwners),
        commandContracts: collectFamilyCommandContracts(
          repoRoot,
          family.profiles,
          runtimeOwnersByStrategy
        ),
      };
    })
    .sort((left, right) => left.family.localeCompare(right.family));

  const taskRecipes = HF_TASKS.map((task) => {
    const taskFamilies = familyRecipes
      .filter((family) => family.taskSlugs.includes(task.slug))
      .map((family) => ({
        family: family.family,
        slug: family.slug,
        recipeCount: family.profiles.filter((profile) => profile.hfTasks.includes(task.slug)).length,
        hfIds: [...new Set(
          family.profiles
            .filter((profile) => profile.hfTasks.includes(task.slug))
            .map((profile) => profile.hfId)
        )].sort(),
        cliCommands: [...new Set(
          family.profiles
            .filter((profile) => profile.hfTasks.includes(task.slug))
            .flatMap((profile) => profile.cliCommands)
        )].sort(),
      }));
    return {
      ...task,
      families: taskFamilies,
      recipeCount: taskFamilies.reduce((sum, family) => sum + family.recipeCount, 0),
    };
  }).filter((task) => task.families.length > 0);

  return {familyRecipes, taskRecipes};
}

function collectPublicCliCommands(repoRoot) {
  const cliSourcePath = path.join(repoRoot, 'src', 'cli', 'args.cpp');
  const source = fs.readFileSync(cliSourcePath, 'utf8');
  const knownCommands = source.match(
    /static const char\* known_cmds\[\]\s*=\s*\{([\s\S]*?)nullptr\s*\};/
  );
  if (!knownCommands) {
    throw new Error(`Unable to find the public CLI registry in ${cliSourcePath}`);
  }
  const commands = new Set(['build', 'graph', 'version']);
  for (const match of knownCommands[1].matchAll(/"([^"]+)"/g)) {
    commands.add(match[1]);
  }
  return [...commands].sort();
}

function manifestBuildConfiguration(manifest) {
  const parallel = manifest.build_args?.parallel || {};
  const parallelMode = parallel.mode || 'single_device';
  const parallelSize =
    parallel.tp_size || parallel.cp_size || manifest.distributed_runtime?.world_size || 1;
  return {
    precision: manifest.precision || manifest.build_args?.precision || 'family default',
    quantization: manifest.quantization?.format || 'not declared',
    parallelMode,
    parallelSize,
  };
}

function collectManifestPaths(e2eModelsDirectory, entries) {
  const manifestPaths = entries
    .filter((entry) => entry.isFile() && entry.name.endsWith('.json'))
    .map((entry) => path.join(e2eModelsDirectory, entry.name));

  let e2eFamilyIndexCount = 0;
  for (const entry of entries.filter((candidate) => candidate.isDirectory())) {
    const familyDirectory = path.join(e2eModelsDirectory, entry.name);
    if (fs.existsSync(path.join(familyDirectory, 'MODEL.toml'))) {
      e2eFamilyIndexCount += 1;
    }
    const manifestsDirectory = path.join(familyDirectory, 'manifests');
    if (fs.existsSync(manifestsDirectory)) {
      manifestPaths.push(
        ...readDirectory(manifestsDirectory, `E2E manifests for ${entry.name}`)
          .filter((candidate) => candidate.isFile() && candidate.name.endsWith('.json'))
          .map((candidate) => path.join(manifestsDirectory, candidate.name))
      );
    }
  }

  return {e2eFamilyIndexCount, manifestPaths: manifestPaths.sort()};
}

function collectModelSupportInventory(repoRoot) {
  const familiesDirectory = path.join(
    repoRoot,
    'python',
    'tensorrt_model_connect',
    'families'
  );
  const familyEntries = readDirectory(familiesDirectory, 'Python family plugins');
  const hfMetadataById = collectHfModelMetadata(repoRoot);
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
  const {e2eFamilyIndexCount, manifestPaths} = collectManifestPaths(
    e2eModelsDirectory,
    e2eEntries
  );
  const modelProfiles = manifestPaths
    .map((manifestPath) => {
      const manifest = readJson(manifestPath);
      if (!manifest.name || !manifest.hf_id || !manifest.family || !manifest.task_strategy) {
        return null;
      }
      const hfMetadata = hfMetadataById.get(manifest.hf_id);
      if (!hfMetadata) {
        throw new Error(
          `Missing Hugging Face model metadata for ${manifest.hf_id} (${manifestPath})`
        );
      }
      if (manifest.hf_revision && manifest.hf_revision !== hfMetadata.revision) {
        throw new Error(
          `Hugging Face model metadata revision ${hfMetadata.revision} does not match ` +
          `${manifest.hf_id}@${manifest.hf_revision} (${manifestPath})`
        );
      }
      const hfTasks = hfTasksForManifest(manifest);
      return {
        profile: manifest.name,
        hfId: manifest.hf_id,
        revision: manifest.hf_revision || 'not pinned',
        bundle: manifest.bundle || `${manifest.name}.trtfb`,
        family: manifest.family,
        runtimeStrategy: manifest.runtime_strategy || 'not declared',
        taskStrategy: manifest.task_strategy,
        hfTasks,
        cliCommands: cliCommandsForManifest(manifest, hfTasks),
        testcases: Array.isArray(manifest.testcases)
          ? manifest.testcases.map((testcase) => testcase.name).filter(Boolean)
          : [],
        fp32Layers: Array.isArray(manifest.fp32_layers) ? manifest.fp32_layers : [],
        sourcePath: path.relative(repoRoot, manifestPath).replace(/\\/g, '/'),
        ...hfMetadataFields(hfMetadata),
        ...manifestBuildConfiguration(manifest),
      };
    })
    .filter(Boolean)
    .sort(
      (left, right) =>
        left.taskStrategy.localeCompare(right.taskStrategy) ||
        left.hfId.localeCompare(right.hfId) ||
        left.profile.localeCompare(right.profile)
    );

  const runtimeModelsDirectory = path.join(repoRoot, 'src', 'runtime', 'models');
  const runtimeStrategyKeys = new Set();
  const runtimeOwnersByStrategy = new Map();
  const strategyArray = /^runtime_strategies\s*=\s*\[(.*?)\]/gms;
  const quotedValue = /"([^"]+)"/g;
  for (const entry of readDirectory(runtimeModelsDirectory, 'runtime model metadata')) {
    if (!entry.isDirectory()) continue;
    const metadataPath = path.join(runtimeModelsDirectory, entry.name, 'MODEL.toml');
    if (!fs.existsSync(metadataPath)) continue;
    const metadata = fs.readFileSync(metadataPath, 'utf8');
    for (const arrayMatch of metadata.matchAll(strategyArray)) {
      for (const valueMatch of arrayMatch[1].matchAll(quotedValue)) {
        const strategy = valueMatch[1];
        runtimeStrategyKeys.add(strategy);
        if (!runtimeOwnersByStrategy.has(strategy)) runtimeOwnersByStrategy.set(strategy, []);
        runtimeOwnersByStrategy.get(strategy).push(entry.name);
      }
    }
  }

  const recipeCatalog = collectRecipeCatalog(repoRoot, modelProfiles, runtimeOwnersByStrategy);
  const publicCliCommands = collectPublicCliCommands(repoRoot);
  const missingCliCommands = [
    ...new Set(recipeCatalog.familyRecipes.flatMap((family) => family.cliCommands)),
  ].filter((command) => !publicCliCommands.includes(command));
  if (missingCliCommands.length > 0) {
    throw new Error(
      `Model recipe CLI mapping references unregistered commands: ${missingCliCommands.join(', ')}`
    );
  }
  const profileById = new Map(modelProfiles.map((profile) => [profile.profile, profile]));
  const performanceSnapshot = collectPerformanceSnapshot(repoRoot).map((row) => {
    const profile = profileById.get(row.profile);
    const hfMetadata = hfMetadataById.get(row.hfId);
    if (!hfMetadata) {
      throw new Error(`Missing Hugging Face model metadata for release row ${row.hfId}`);
    }
    if (row.revision !== 'not pinned' && row.revision !== hfMetadata.revision) {
      throw new Error(
        `Hugging Face model metadata revision ${hfMetadata.revision} does not match ` +
        `release row ${row.hfId}@${row.revision}`
      );
    }
    return {
      ...row,
      ...hfMetadataFields(hfMetadata),
      family: profile?.family || 'not declared',
      taskStrategy: profile?.taskStrategy || 'not declared',
      manifestHfId: profile?.hfId || 'not declared',
      manifestSourcePath: profile?.sourcePath || null,
    };
  });
  const referencedHfIds = new Set([
    ...modelProfiles.map((profile) => profile.hfId),
    ...performanceSnapshot.map((row) => row.hfId),
  ]);
  const staleMetadata = [...hfMetadataById.keys()].filter(
    (hfId) => !referencedHfIds.has(hfId)
  );
  if (staleMetadata.length > 0) {
    throw new Error(
      `Hugging Face model metadata has no manifest or release row: ${staleMetadata.join(', ')}`
    );
  }
  return {
    familyPluginCount: familyPluginNames.length,
    familyPluginNames,
    e2eManifestCount: manifestPaths.length,
    e2eFamilyIndexCount,
    runtimeStrategyKeyCount: runtimeStrategyKeys.size,
    modelProfiles,
    performanceSnapshot,
    publicCliCommands,
    ...recipeCatalog,
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
      const taskPageComponent = path.join(
        context.siteDir,
        'src',
        'components',
        'ModelRecipes',
        'TaskPage.js'
      );
      const familyPageComponent = path.join(
        context.siteDir,
        'src',
        'components',
        'ModelRecipes',
        'FamilyPage.js'
      );
      for (const task of content.taskRecipes) {
        actions.addRoute({
          path: `${routeBase}/models-recipes/model-recipes/tasks/${task.slug}`,
          component: taskPageComponent,
          exact: true,
          props: {taskSlug: task.slug},
        });
      }
      for (const family of content.familyRecipes) {
        actions.addRoute({
          path: `${routeBase}/models-recipes/model-recipes/families/${family.slug}`,
          component: familyPageComponent,
          exact: true,
          props: {familySlug: family.slug},
        });
      }
    },
  };
}

module.exports = modelSupportInventoryPlugin;
module.exports.collectModelSupportInventory = collectModelSupportInventory;
module.exports.collectRuntimeCapabilities = collectRuntimeCapabilities;
