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
} = require('./index');

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
    'python/tensorrt_model_connect/models/alpha/model.py'
  );
  writeFixture(
    repoRoot,
    'python/tensorrt_model_connect/models/alpha/MODEL.toml',
    [
      'id = "alpha"',
      'runtime_strategies = ["decoder", "shared"]',
      'runtime_config_schemas = ["config_schema.cpp|register_alpha_schema"]',
      'test_manifests = ["tests/manifests/small.json"]',
      '',
    ].join('\n')
  );
  writeFixture(
    repoRoot,
    'python/tensorrt_model_connect/models/alpha/runtime_config_schema.py',
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
    'python/tensorrt_model_connect/models/beta/model.py'
  );
  writeFixture(
    repoRoot,
    'python/tensorrt_model_connect/models/beta/MODEL.toml',
    [
      'id = "beta"',
      'runtime_strategies = ["shared", "encoder"]',
      'test_manifests = ["tests/manifests/base.json"]',
      '',
    ].join('\n')
  );
  writeFixture(
    repoRoot,
    'python/tensorrt_model_connect/models/_private.py'
  );
  writeFixture(
    repoRoot,
    'python/tensorrt_model_connect/models/base.py'
  );
  writeFixture(
    repoRoot,
    'python/tensorrt_model_connect/models/alpha/tests/manifests/small.json',
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
  writeFixture(
    repoRoot,
    'python/tensorrt_model_connect/models/beta/tests/manifests/base.json',
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
    'python/tensorrt_model_connect/models/alpha/runtime/config_schema.cpp',
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
    'python/tensorrt_model_connect/models/alpha/runtime/pipeline.h',
    [
      'class AlphaPipeline final : public IPipeline {',
      '  TextResult generate(const std::string& prompt, const GenerateConfig& cfg) override;',
      '};',
      '',
    ].join('\n')
  );
  writeFixture(
    repoRoot,
    'python/tensorrt_model_connect/models/alpha/runtime/pipeline.cpp',
    [
      'TextResult AlphaPipeline::generate(const std::string& prompt, const GenerateConfig& cfg) {',
      '  auto token_limit = cfg.max_new_tokens;',
      '  auto temperature = cfg.temperature;',
      '  return {};',
      '}',
      '',
    ].join('\n')
  );
  writeFixture(
    repoRoot,
    'src/cli/args.cpp',
    'static const char* known_cmds[] = {"run", "inspect", "encode", nullptr};\n'
  );

  const inventory = collectModelSupportInventory(repoRoot);
  assert.deepEqual(
    {
      modelOwnerCount: inventory.modelOwnerCount,
      modelOwnerNames: inventory.modelOwnerNames,
      e2eManifestCount: inventory.e2eManifestCount,
      runtimeStrategyKeyCount: inventory.runtimeStrategyKeyCount,
    },
    {
      modelOwnerCount: 2,
      modelOwnerNames: ['alpha', 'beta'],
      e2eManifestCount: 2,
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
      sourcePath: 'python/tensorrt_model_connect/models/beta/tests/manifests/base.json',
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
      sourcePath: 'python/tensorrt_model_connect/models/alpha/tests/manifests/small.json',
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
      manifestSourcePath:
        'python/tensorrt_model_connect/models/alpha/tests/manifests/small.json',
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
          'python/tensorrt_model_connect/models/alpha/runtime/config_schema.cpp',
          'python/tensorrt_model_connect/models/alpha/runtime_config_schema.py',
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
    'trtmc run <bundle.bundle> --prompt "<text>" [generation options]'
  );
  assert.ok(!alphaContract.syntax.includes('--image'));
  assert.deepEqual(
    alphaContract.options.map((entry) => entry.flag),
    [
      '--prompt <TEXT>', '--max-new-tokens <N>', '--temperature <F>', '--greedy',
      '--num-samples <N>', '--output <PATH>', '--benchmark <N>', '--warmup <N>',
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
    /Unable to read unified model owners/
  );
});

test('derives image input and generation options from runtime code', (context) => {
  const repoRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'trtmc-runtime-capability-'));
  context.after(() => fs.rmSync(repoRoot, {recursive: true, force: true}));

  writeFixture(
    repoRoot,
    'python/tensorrt_model_connect/models/text_only/runtime/pipeline.cpp',
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
    'python/tensorrt_model_connect/models/vision_language/runtime/pipeline.h',
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
