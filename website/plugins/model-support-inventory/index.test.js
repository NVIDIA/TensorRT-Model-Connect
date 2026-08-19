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
  runtimeApiForManifest,
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
      modelSourceKind: 'huggingface',
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
      modelSourceKind: 'huggingface',
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

test('validates the model-owned C ABI inventory declaration', () => {
  assert.equal(runtimeApiForManifest({name: 'none'}), null);
  assert.deepEqual(
    runtimeApiForManifest({
      name: 'sam2-hoi-tracking',
      runtime_api: {
        kind: 'model_owned_c_abi',
        library: 'libtrtmc_model_sam2_hoi.so',
        header: 'trtmc/models/sam2_hoi_video.h',
        entrypoint: 'trtmc_sam2_hoi_video_run_jpeg_files_v1',
      },
    }),
    {
      kind: 'model_owned_c_abi',
      library: 'libtrtmc_model_sam2_hoi.so',
      header: 'trtmc/models/sam2_hoi_video.h',
      entrypoint: 'trtmc_sam2_hoi_video_run_jpeg_files_v1',
    }
  );
  assert.throws(
    () => runtimeApiForManifest({name: 'broken', runtime_api: {kind: 'model_owned_c_abi'}}),
    /Malformed model-owned C ABI declaration/
  );
});

test('publishes the local SAM2 HOI source-package recipe without fake HF metadata', (context) => {
  const repoRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'trtmc-sam2-hoi-support-'));
  context.after(() => fs.rmSync(repoRoot, {recursive: true, force: true}));
  const alphaRevision = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa';
  writeFixture(repoRoot, 'python/tensorrt_model_connect/families/alpha.py');
  writeFixture(repoRoot, 'python/tensorrt_model_connect/families/sam2_hoi/plugin.py');
  writeFixture(
    repoRoot,
    'website/data/model-support-matrix.md',
    [
      '| Hugging Face model ID (`hf_id`, CLI input) | TRTMC profile | Build precision | Quantization | Platform specialization runtime provider | GB300 |',
      '| --- | --- | --- | --- | --- | --- |',
      `| \`example/alpha\`<br />Revision: \`${alphaRevision}\` | \`alpha\` | \`FP16\` | None | — | 🟢 Green |`,
      '',
    ].join('\n')
  );
  writeFixture(
    repoRoot,
    'website/data/hf-model-metadata.json',
    JSON.stringify({
      schema_version: 1,
      checkpoints: [{
        hf_id: 'example/alpha',
        revision: alphaRevision,
        revision_source: 'declared',
        metadata_file: 'config.json',
        model_type: 'alpha',
        architectures: ['AlphaModel'],
        architecture_source: 'config.architectures',
      }],
    })
  );
  writeFixture(repoRoot, 'tests/e2e/models/alpha/MODEL.toml');
  writeFixture(
    repoRoot,
    'tests/e2e/models/alpha/manifests/alpha.json',
    JSON.stringify({
      name: 'alpha',
      hf_id: 'example/alpha',
      hf_revision: alphaRevision,
      family: 'alpha',
      runtime_strategy: 'alpha_runtime',
      task_strategy: 'encoder_only_nlp',
    })
  );
  writeFixture(repoRoot, 'tests/e2e/models/sam2_hoi/MODEL.toml');
  writeFixture(
    repoRoot,
    'tests/e2e/models/sam2_hoi/manifests/sam2-hoi.json',
    JSON.stringify({
      name: 'sam2-hoi-tracking',
      hf_id: 'artifacts/sam2_hoi/hoi',
      model_source_kind: 'local_source_package',
      family: 'sam2_hoi',
      runtime_strategy: 'sam2_hoi_video_tracking',
      task_strategy: 'hoi_video_tracking',
      runtime_api: {
        kind: 'model_owned_c_abi',
        library: 'libtrtmc_model_sam2_hoi.so',
        header: 'trtmc/models/sam2_hoi_video.h',
        entrypoint: 'trtmc_sam2_hoi_video_run_jpeg_files_v1',
      },
    })
  );
  writeFixture(
    repoRoot,
    'src/runtime/models/alpha/MODEL.toml',
    'runtime_strategies = ["alpha_runtime"]\n'
  );
  writeFixture(
    repoRoot,
    'src/runtime/models/sam2_hoi/MODEL.toml',
    'runtime_strategies = ["sam2_hoi_video_tracking"]\n'
  );
  writeFixture(
    repoRoot,
    'src/cli/args.cpp',
    'static const char* known_cmds[] = {"run", "inspect", "encode", nullptr};\n'
  );

  const inventory = collectModelSupportInventory(repoRoot);
  const profile = inventory.modelProfiles.find(
    (candidate) => candidate.profile === 'sam2-hoi-tracking'
  );

  assert.ok(profile);
  assert.equal(profile.modelSourceKind, 'local_source_package');
  assert.equal(profile.hfId, 'artifacts/sam2_hoi/hoi');
  assert.equal(profile.hfMetadataFile, 'not applicable');
  assert.deepEqual(profile.hfTasks, ['mask-generation']);
  assert.deepEqual(profile.cliCommands, []);
  assert.deepEqual(profile.runtimeApi, {
    kind: 'model_owned_c_abi',
    library: 'libtrtmc_model_sam2_hoi.so',
    header: 'trtmc/models/sam2_hoi_video.h',
    entrypoint: 'trtmc_sam2_hoi_video_run_jpeg_files_v1',
  });

  const family = inventory.familyRecipes.find(
    (candidate) => candidate.family === 'sam2_hoi'
  );
  assert.ok(family);
  assert.deepEqual(family.cliCommands, ['build', 'inspect']);
  assert.deepEqual(family.commandContracts, []);
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
