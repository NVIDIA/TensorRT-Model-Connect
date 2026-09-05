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
  collectRuntimeCapabilities,
  externalModelSource,
} = require('./index');

test('rejects hf_revision for external model sources regardless of its value', () => {
  const manifest = {
    hf_id: '/work/model-artifacts/foundationpose/ngc-1.0.1',
    model_source: {
      kind: 'ngc',
      id: 'nvidia/isaac/foundationpose',
      revision: '1.0.1_onnx',
    },
  };

  assert.equal(externalModelSource(manifest, 'manifest.json').sourceKind, 'ngc');
  for (const hfRevision of ['', null]) {
    assert.throws(
      () => externalModelSource({...manifest, hf_revision: hfRevision}, 'manifest.json'),
      /Malformed external model_source/
    );
  }
});

test('accepts a digest-pinned S3 model source', () => {
  const manifest = {
    hf_id: '/work/model-artifacts/openfold3/openbind-v0.5.0-ubiquitin',
    model_source: {
      kind: 's3',
      id: 's3://openfold3-data/openfold3-parameters/of3-ob-2025-06-30-174k.pt',
      revision: 'bd43301c011d5f87580d3e8b548658869433e4488399feb03035ba248f8e29e4',
    },
  };

  assert.deepEqual(externalModelSource(manifest, 'manifest.json'), {
    hfId: manifest.model_source.id,
    revision: manifest.model_source.revision,
    sourceKind: 's3',
    buildInput: manifest.hf_id,
    hfModelType: 'not applicable',
    hfArchitectures: [],
    hfArchitectureSource: 'not applicable',
    hfMetadataRevision: manifest.model_source.revision,
    hfMetadataRevisionSource: 's3',
    hfMetadataFile: 'not applicable',
  });
  assert.throws(
    () => externalModelSource({
      ...manifest,
      model_source: {...manifest.model_source, revision: 'mutable'},
    }, 'manifest.json'),
    /Malformed external model_source/
  );
});

function writeFixture(repoRoot, relativePath, content = '') {
  const fixturePath = path.join(repoRoot, relativePath);
  fs.mkdirSync(path.dirname(fixturePath), {recursive: true});
  fs.writeFileSync(fixturePath, content, 'utf8');
}

test('collects support inventory from repository metadata', (context) => {
  const repoRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'trtmc-model-support-'));
  context.after(() => fs.rmSync(repoRoot, {recursive: true, force: true}));
  const alphaRevision = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa';
  const betaRevision = 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb';

  writeFixture(
    repoRoot,
    'python/tensorrt_model_connect/families/alpha.py'
  );
  writeFixture(
    repoRoot,
    'python/tensorrt_model_connect/families/alpha/runtime_config_schema.py',
    [
      '_SESSION = frozenset({Layer.SESSION_REQUEST, Layer.PLATFORM_PROFILE})',
      'SCHEMA = Schema(',
      '    namespace="alpha_runtime",',
      '    fields=(',
      '        ConfigField(',
      '            name="temperature",',
      '            type_tag="float",',
      '            default=0.5,',
      '            allowed_layers=_SESSION,',
      '        ),',
      '    ),',
      ')',
      '',
    ].join('\n')
  );
  writeFixture(
    repoRoot,
    'website/data/model-support-matrix.md',
    [
      '| Hugging Face model ID (`hf_id`, CLI input) | TRTMC profile | Build precision | Quantization | Platform specialization runtime provider | GB300 |',
      '| --- | --- | --- | --- | --- | --- |',
      `| \`example/alpha-small\`<br />Revision: \`${alphaRevision}\` | \`alpha-small\` | \`FP16\` | None | — | 🟢 Green |`,
      '',
    ].join('\n')
  );
  writeFixture(
    repoRoot,
    'website/data/hf-model-metadata.json',
    JSON.stringify({
      schema_version: 1,
      checkpoints: [
        {
          hf_id: 'example/alpha-small',
          revision: alphaRevision,
          revision_source: 'declared',
          metadata_file: 'config.json',
          model_type: 'alpha',
          architectures: ['AlphaForCausalLM'],
          architecture_source: 'config.architectures',
        },
        {
          hf_id: 'example/beta-base',
          revision: betaRevision,
          revision_source: 'resolved',
          metadata_file: 'config.json',
          model_type: 'beta',
          architectures: ['BetaModel'],
          architecture_source: 'config.architectures',
        },
      ],
    })
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
  writeFixture(
    repoRoot,
    'tests/e2e/models/alpha/manifests/small.json',
    JSON.stringify({
      name: 'alpha-small',
      hf_id: 'example/alpha-small',
      hf_revision: alphaRevision,
      family: 'alpha',
      runtime_strategy: 'decoder',
      task_strategy: 'text_generation_causal',
      precision: 'fp16',
    })
  );
  writeFixture(repoRoot, 'tests/e2e/models/beta/MODEL.toml');
  writeFixture(
    repoRoot,
    'tests/e2e/models/beta/manifests/base.json',
    JSON.stringify({
      name: 'beta-base-tp2',
      hf_id: 'example/beta-base',
      family: 'beta',
      runtime_strategy: 'encoder',
      task_strategy: 'encoder_only_nlp',
      build_args: {parallel: {mode: 'tensor_parallel', tp_size: 2}},
    })
  );
  writeFixture(
    repoRoot,
    'src/runtime/models/alpha/MODEL.toml',
    [
      'runtime_strategies = ["decoder", "shared"]',
      'runtime_config_schemas = ["config_schema.cpp|register_alpha_schema"]',
      '',
    ].join('\n')
  );
  writeFixture(
    repoRoot,
    'src/runtime/models/alpha/config_schema.cpp',
    [
      'Schema make_alpha_schema() {',
      '  const std::set<Layer> session = {Layer::SessionRequest, Layer::PlatformProfile};',
      '  return Schema{',
      '    "alpha_runtime",',
      '    {',
      '      ConfigField{"temperature", "float", std::any{0.5F}, session, nullptr},',
      '    },',
      '  };',
      '}',
      '',
    ].join('\n')
  );
  writeFixture(
    repoRoot,
    'src/runtime/models/alpha/pipeline.h',
    [
      'class AlphaPipeline final : public IPipeline {',
      '  TextResult generate(const std::string& prompt, const GenerateConfig& cfg) override;',
      '};',
      '',
    ].join('\n')
  );
  writeFixture(
    repoRoot,
    'src/runtime/models/alpha/pipeline.cpp',
    [
      'TextResult AlphaPipeline::generate(const std::string& prompt, const GenerateConfig& cfg) {',
      '  auto token_limit = cfg.max_new_tokens;',
      '  auto temperature = cfg.temperature;',
      '  auto repetition_penalty = cfg.repetition_penalty;',
      '  return {};',
      '}',
      '',
    ].join('\n')
  );
  writeFixture(
    repoRoot,
    'src/runtime/models/beta/MODEL.toml',
    'runtime_strategies = [\n  "shared",\n  "encoder",\n]\n'
  );
  writeFixture(
    repoRoot,
    'src/cli/args.cpp',
    'static const char* known_cmds[] = {"run", "inspect", "encode", nullptr};\n'
  );

  const inventory = collectModelSupportInventory(repoRoot);
  assert.deepEqual(
    {
      familyPluginCount: inventory.familyPluginCount,
      familyPluginNames: inventory.familyPluginNames,
      e2eManifestCount: inventory.e2eManifestCount,
      e2eFamilyIndexCount: inventory.e2eFamilyIndexCount,
      runtimeStrategyKeyCount: inventory.runtimeStrategyKeyCount,
    },
    {
      familyPluginCount: 2,
      familyPluginNames: ['alpha', 'beta'],
      e2eManifestCount: 3,
      e2eFamilyIndexCount: 2,
      runtimeStrategyKeyCount: 3,
    }
  );
  assert.deepEqual(inventory.modelProfiles, [
    {
      profile: 'beta-base-tp2',
      hfId: 'example/beta-base',
      revision: 'not pinned',
      bundle: 'beta-base-tp2.bundle',
      family: 'beta',
      runtimeStrategy: 'encoder',
      taskStrategy: 'encoder_only_nlp',
      hfTasks: ['feature-extraction'],
      cliCommands: ['encode'],
      testcases: [],
      fp32Layers: [],
      sourcePath: 'tests/e2e/models/beta/manifests/base.json',
      hfModelType: 'beta',
      hfArchitectures: ['BetaModel'],
      hfArchitectureSource: 'config.architectures',
      hfMetadataRevision: betaRevision,
      hfMetadataRevisionSource: 'resolved',
      hfMetadataFile: 'config.json',
      precision: 'family default',
      quantization: 'not declared',
      parallelMode: 'tensor_parallel',
      parallelSize: 2,
    },
    {
      profile: 'alpha-small',
      hfId: 'example/alpha-small',
      revision: alphaRevision,
      bundle: 'alpha-small.bundle',
      family: 'alpha',
      runtimeStrategy: 'decoder',
      taskStrategy: 'text_generation_causal',
      hfTasks: ['text-generation'],
      cliCommands: ['run'],
      testcases: [],
      fp32Layers: [],
      sourcePath: 'tests/e2e/models/alpha/manifests/small.json',
      hfModelType: 'alpha',
      hfArchitectures: ['AlphaForCausalLM'],
      hfArchitectureSource: 'config.architectures',
      hfMetadataRevision: alphaRevision,
      hfMetadataRevisionSource: 'declared',
      hfMetadataFile: 'config.json',
      precision: 'fp16',
      quantization: 'not declared',
      parallelMode: 'single_device',
      parallelSize: 1,
    },
  ]);
  assert.deepEqual(inventory.performanceSnapshot, [
    {
      hfId: 'example/alpha-small',
      revision: alphaRevision,
      profile: 'alpha-small',
      precision: ['FP16'],
      quantization: ['None'],
      platformSpecialization: ['—'],
      performance: '🟢 Green',
      hfModelType: 'alpha',
      hfArchitectures: ['AlphaForCausalLM'],
      hfArchitectureSource: 'config.architectures',
      hfMetadataRevision: alphaRevision,
      hfMetadataRevisionSource: 'declared',
      hfMetadataFile: 'config.json',
      family: 'alpha',
      taskStrategy: 'text_generation_causal',
      manifestHfId: 'example/alpha-small',
      manifestSourcePath: 'tests/e2e/models/alpha/manifests/small.json',
    },
  ]);
  assert.deepEqual(
    inventory.taskRecipes.map((task) => task.slug),
    ['text-generation', 'feature-extraction']
  );
  assert.deepEqual(
    inventory.familyRecipes.map((family) => family.family),
    ['alpha', 'beta']
  );
  assert.deepEqual(
    inventory.familyRecipes.find((family) => family.family === 'alpha').configSchemas,
    [
      {
        namespace: 'alpha_runtime',
        fields: [
          {
            name: 'temperature',
            key: 'alpha_runtime.temperature',
            type: 'float',
            defaultValue: '0.5',
            allowedLayers: ['Platform Profile', 'Session Request'],
            surfaces: ['runtime'],
          },
        ],
        sourcePaths: [
          'python/tensorrt_model_connect/families/alpha/runtime_config_schema.py',
          'src/runtime/models/alpha/config_schema.cpp',
        ],
      },
    ]
  );
  const alphaContract = inventory.familyRecipes
    .find((family) => family.family === 'alpha')
    .commandContracts[0];
  assert.equal(alphaContract.command, 'run');
  assert.equal(
    alphaContract.syntax,
    'trtmc run <bundle.bundle> (--prompt "<text>" | --prompts-file <PATH>) [generation options]'
  );
  assert.ok(!alphaContract.syntax.includes('--image'));
  assert.deepEqual(
    alphaContract.options.map((entry) => entry.flag),
    [
      '--prompt <TEXT>', '--prompts-file <PATH>', '--max-new-tokens <N>',
      '--temperature <F>', '--repetition-penalty <F>', '--greedy', '--num-samples <N>',
      '--output <PATH>', '--benchmark <N>', '--warmup <N>',
    ]
  );
  assert.deepEqual(alphaContract.runtimeOwners, ['alpha']);
  assert.deepEqual(inventory.publicCliCommands, [
    'build', 'encode', 'graph', 'inspect', 'run', 'version',
  ]);
});

test('fails closed when a source-of-truth directory is missing', (context) => {
  const repoRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'trtmc-model-support-'));
  context.after(() => fs.rmSync(repoRoot, {recursive: true, force: true}));

  assert.throws(
    () => collectModelSupportInventory(repoRoot),
    /Unable to read Python family plugins/
  );
});

test('derives image input and generation options from runtime code', (context) => {
  const repoRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'trtmc-runtime-capability-'));
  context.after(() => fs.rmSync(repoRoot, {recursive: true, force: true}));

  writeFixture(
    repoRoot,
    'src/runtime/models/text_only/pipeline.cpp',
    [
      'TextResult TextOnlyPipeline::generate(',
      '    const std::string& prompt, const GenerateConfig& request) {',
      '  auto max_tokens = request.max_new_tokens;',
      '  return {};',
      '}',
      '',
    ].join('\n')
  );
  writeFixture(
    repoRoot,
    'src/runtime/models/vision_language/pipeline.h',
    [
      'class VisionLanguagePipeline final : public IPipeline {',
      '  TextResult generate(const std::string& prompt, const GenerateConfig& cfg) override;',
      '  TextResult generate(const std::string& prompt, const float* image_pixels,',
      '                      int image_height, int image_width,',
      '                      const GenerateConfig& cfg) override;',
      '};',
      '',
    ].join('\n')
  );

  const textOnly = collectRuntimeCapabilities(repoRoot, 'text_only');
  const visionLanguage = collectRuntimeCapabilities(repoRoot, 'vision_language');
  assert.equal(textOnly.textImageInput, false);
  assert.deepEqual(textOnly.generateConfigFields, ['max_new_tokens']);
  assert.equal(visionLanguage.textImageInput, true);
  assert.deepEqual(visionLanguage.generateConfigFields, []);
});

test('publishes LFM2 and MoGe profiles in model recipes and the website sidebar', () => {
  const repoRoot = path.resolve(__dirname, '..', '..', '..');
  const inventory = collectModelSupportInventory(repoRoot);
  const family = inventory.familyRecipes.find((entry) => entry.family === 'lfm2');

  assert.ok(family, 'missing LFM2 family recipe');
  assert.equal(family.slug, 'lfm2');
  assert.deepEqual(
    family.profiles.map((profile) => profile.profile),
    [
      'lfm2-1.2b',
      'lfm2-2.6b',
      'lfm2-350m-bf16-model-card',
      'lfm2-350m-fp16',
      'lfm2-700m',
    ]
  );
  assert.deepEqual(
    [...new Set(family.profiles.map((profile) => profile.hfId))].sort(),
    [
      'LiquidAI/LFM2-1.2B',
      'LiquidAI/LFM2-2.6B',
      'LiquidAI/LFM2-350M',
      'LiquidAI/LFM2-700M',
    ]
  );
  assert.ok(
    family.profiles.every(
      (profile) =>
        profile.hfModelType === 'lfm2' &&
        profile.hfArchitectures.includes('Lfm2ForCausalLM') &&
        profile.runtimeStrategy === 'lfm2_hybrid_conv_attention' &&
        /^[0-9a-f]{40}$/.test(profile.revision)
    )
  );

  const textGeneration = inventory.taskRecipes.find(
    (task) => task.slug === 'text-generation'
  );
  const taskFamily = textGeneration?.families.find(
    (entry) => entry.family === 'lfm2'
  );
  assert.deepEqual(taskFamily, {
    family: 'lfm2',
    slug: 'lfm2',
    recipeCount: 5,
    hfIds: [
      'LiquidAI/LFM2-1.2B',
      'LiquidAI/LFM2-2.6B',
      'LiquidAI/LFM2-350M',
      'LiquidAI/LFM2-700M',
    ],
    cliCommands: ['run'],
  });

  const runContract = family.commandContracts.find(
    (contract) => contract.command === 'run'
  );
  assert.equal(
    runContract?.syntax,
    'trtmc run <bundle.bundle> (--prompt "<text>" | --prompts-file <PATH>) [generation options]'
  );
  assert.ok(
    runContract.options.some(
      (entry) => entry.flag === '--repetition-penalty <F>'
    )
  );
  assert.ok(
    runContract.options.some((entry) => entry.flag === '--prompts-file <PATH>')
  );
  assert.ok(
    !runContract.options.some((entry) => entry.flag === '--no-thinking')
  );

  const depthEstimation = inventory.taskRecipes.find(
    (task) => task.slug === 'depth-estimation'
  );
  assert.ok(
    depthEstimation?.families.some(
      (entry) =>
        entry.family === 'fast_foundation_stereo' &&
        entry.cliCommands.includes('disparity')
    )
  );
  const moge = depthEstimation?.families.find((entry) => entry.family === 'moge');
  assert.deepEqual(moge, {
    family: 'moge',
    slug: 'moge',
    recipeCount: 1,
    hfIds: ['Ruicheng/moge-2-vitl'],
    cliCommands: ['geometry'],
  });
  const mogeFamily = inventory.familyRecipes.find((entry) => entry.family === 'moge');
  assert.equal(
    mogeFamily?.commandContracts[0].syntax,
    'trtmc geometry <bundle.bundle> --image <input.png> --output <output-directory>'
  );

  const sidebar = require(path.join(repoRoot, 'website', 'sidebars.js'));
  const modelsCategory = sidebar.docs.find(
    (item) => item && typeof item === 'object' && item.label === 'Models & Recipes'
  );
  const recipesCategory = modelsCategory?.items.find(
    (item) => item.label === 'Model Recipes'
  );
  const textGenerationCategory = recipesCategory?.items.find(
    (item) => item.label === 'Text Generation'
  );
  const lfm2Link = textGenerationCategory?.items.find(
    (item) => item.label === 'lfm2'
  );
  assert.deepEqual(lfm2Link, {
    type: 'link',
    label: 'lfm2',
    href: '/models-recipes/model-recipes/families/lfm2',
    autoAddBaseUrl: true,
  });
});

test('publishes the external NGC FoundationPose recipe without HF metadata', () => {
  const repoRoot = path.resolve(__dirname, '..', '..', '..');
  const inventory = collectModelSupportInventory(repoRoot);
  const profile = inventory.modelProfiles.find(
    (candidate) => candidate.profile === 'foundationpose-ngc-1.0.1'
  );

  assert.ok(profile, 'missing FoundationPose NGC profile');
  assert.equal(profile.hfId, 'nvidia/isaac/foundationpose');
  assert.equal(profile.revision, '1.0.1_onnx');
  assert.equal(profile.sourceKind, 'ngc');
  assert.equal(profile.buildInput, '/work/model-artifacts/foundationpose/ngc-1.0.1');
  assert.deepEqual(profile.hfTasks, ['robotics']);
  assert.deepEqual(profile.cliCommands, []);
  assert.equal(profile.hfModelType, 'not applicable');
  assert.deepEqual(profile.hfArchitectures, []);
  assert.deepEqual(
    inventory.familyRecipes.find((candidate) => candidate.family === 'foundationpose')
      ?.commandContracts,
    []
  );
});

test('publishes the external S3 OpenFold3 protein-folding recipe', () => {
  const repoRoot = path.resolve(__dirname, '..', '..', '..');
  const inventory = collectModelSupportInventory(repoRoot);
  const profile = inventory.modelProfiles.find(
    (candidate) => candidate.profile === 'openfold3-ubiquitin-fp16-l0'
  );

  assert.ok(profile, 'missing OpenFold3 profile');
  assert.equal(profile.sourceKind, 's3');
  assert.deepEqual(profile.hfTasks, ['protein-folding']);
  assert.deepEqual(profile.cliCommands, ['predict-structure']);
  const family = inventory.familyRecipes.find(
    (candidate) => candidate.family === 'openfold3'
  );
  assert.equal(
    family?.commandContracts[0].syntax,
    'trtmc predict-structure <bundle.bundle> --input <request.json> --output <structure.cif>'
  );
});
