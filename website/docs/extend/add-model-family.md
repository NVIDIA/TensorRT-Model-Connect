---
title: Add a Model Family
---

Native support for a current model family is a three-sided, model-owned unit:

1. A Python build package.
2. A C++ runtime model DSO with its own strategy key.
3. An E2E family index with one or more concrete manifests.

Even if the new model implements an existing public task such as causal text
generation, do not reuse a retired generic runtime key such as
`decoder_kv_cache`. Give the family a concrete key such as
`example_decoder_kv_cache` and keep its runtime implementation in its own DSO.
Shared tooling groups compatible models through capabilities and
`task_strategy`.

An optional delegated optimized-runtime adapter is a separate, additive
support path for exact model/revision/target/options tuples. Add it under the
owning Python family only after the three native ownership surfaces are
understood; it uses an implementation manifest/profile and embedded
implementation DSO, not another native `runtime_strategy`.

## 1. Choose the closest existing owner

Choose an existing family with the same checkpoint layout and request-time
contract. Compare all three roots before copying:

```text
python/tensorrt_model_connect/families/qwen/
src/runtime/models/qwen/
tests/e2e/models/qwen/
```

Copy only the files the new family actually needs. Do not import or include a
sibling family's model-owned implementation in production code.

`scripts/new_family.py` is only a preliminary Python bootstrap in this
revision. It creates `plugin.py` and `__init__.py`, but it does not create the
required `MODEL.toml`, family-local builder modules, runtime DSO, or E2E
descriptor. It is not a complete onboarding command.

## 2. Add the Python family package

Create:

```text
python/tensorrt_model_connect/families/<builder-family>/
  MODEL.toml
  __init__.py
  plugin.py
  config.py
  checkpoint_mapper.py
  <family-owned builders and graph helpers>
```

A minimal Python descriptor is:

```toml
id = "example"
plugin = "example"
# Specialization/tooling metadata; runtime discovery imports the package.
module = "plugin"
aliases = ["example", "ExampleModel"]
prefixes = ["example"]
```

Use `architecture_patterns` for architecture-name matching,
`diffusion_pipeline_classes` for Diffusers discovery, or the other metadata
fields only when the family needs them. The descriptor's `module` field is
specialization/tooling metadata; it does not select an arbitrary discovery
module. Runtime discovery imports the family package and reads the package-level
`plugin` exported by `__init__.py`.

Keep the three discovery flows distinct:

1. A full config first uses `architecture_patterns` to import bounded
   descriptor candidates and evaluate `matches_config()`. If that does not
   resolve a plugin, the compatibility path imports every non-private family
   module/package with `pkgutil` and evaluates its predicates.
2. A string or `model_type` first attempts a direct descriptor-ID lookup, then
   alias/prefix candidates, and finally the same all-package compatibility
   fallback.
3. A Diffusers pipeline class uses only
   `diffusion_pipeline_classes` from descriptors. It imports matching packages
   and has no `pkgutil` fallback.

See [Build Pipeline](../architecture/build-pipeline.md#2-resolve-the-owning-family)
for the live flow.

`plugin.py` must provide:

- `name`, matching the logical family ID;
- `runtime_strategy`, matching exactly one strategy in the C++ runtime
  descriptor;
- `matches(model_type)`;
- `load_weights(...)`;
- `build_engine(...)` or the modality-specific component hooks used by the
  closest family.

Keep config adapters, checkpoint mapping, graph helpers, builders, calibration
policy, and optional debug hooks in this family package. The old repository-root
`graph_ops.py`, `graph_blocks.py`, and `standard_decoder_builder.py` ownership
model has been retired.

### Optional Python execution profile

If build or reference code needs Python packages that conflict with the common
environment, keep the declaration and exact pins in the owning family:

```toml
python_profile_specs = [
  "example_reference|families/example/python_profile_requirements/reference.lock.txt|families/example/python_profile_verify.py|true|true",
]
default_execution_profiles = [
  "reference|example_reference",
]
```

The fields are `name|requirements|verifier|system_site_packages|prebuild`.
`prebuild=true` means CI prepares the profile before a network-disabled proof;
it does not bake the profile into the shared base image or change the base
runtime fingerprint. Requirements must be exact public PyPI `name==version`
pins. CI downloads their artifacts with the reviewed base-image downloader,
then installs, source-builds, and verifies them in a separate offline container.

When an sdist must disable its own network-wheel lookup, declare only the
package build setting in the family descriptor:

```toml
python_profile_build_environment = [
  "example_reference|PACKAGE_FORCE_BUILD|TRUE",
]
```

This surface is for package build switches, not credentials, package indexes,
runtime configuration, or system dependencies. A new APT, CUDA, compiler, or
system-library requirement still changes the reviewed base runtime and needs
maintainer qualification.

### Optional split decoder contract

Opt into separate prefill/decode engines only when the family builder and
runtime implement both roles. Provide
`supports_split_decoder_roles(config) -> bool` (or the equivalent
`split_decoder_roles` family capability) and make the family builder honor the
internal prefill/decode role passed by the generic engine builder. A family
with `embed_input = True` must also set `supports_split_embed_input = True`
only after its prefill engine accepts the embedding-input contract and its
decode engine handles the matching one-token role.

The generic builder does not select split layout for tensor parallelism,
dynamic KV, or TriAttention. An unsupported split request falls back to the
family's existing single-engine path. Tests must inspect the emitted
`config.json.decoder_engine_layout`: an actual split result must contain both
`prefill_engine_plan` and decode `engine_plan`, while a fallback must not claim
split merely because the request asked for it.

## 3. Add the runtime model DSO

Create:

```text
src/runtime/models/<runtime-owner>/
  MODEL.toml
  plugin.cpp
  pipeline.h
  pipeline.cpp
  <model-owned state, sampler, helpers, and CUDA sources>
```

Use the same name for `<builder-family>`, `<runtime-owner>`, and
`<e2e-family>` unless an existing compatibility boundary requires a different
physical owner. Current exceptions map builder/E2E owner `magpie_tts` to
runtime owner `magpie`, and `wan_t2v` to `wan`. A minimal runtime descriptor
is:

```toml
id = "example"
runtime_library = "libtrtmc_model_example.so"
runtime_plugins = ["plugin.cpp|register_example_plugin"]
runtime_strategies = ["example_decoder_kv_cache"]
```

The plugin source registers the same key:

```cpp
REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(
    register_example_plugin,
    ExamplePlugin,
    "example_decoder_kv_cache");
```

Do not edit a central plugin list. CMake discovers the descriptor, creates the
`trtmc_model_example` target, generates its exported model entrypoint, and adds
the strategy-to-DSO index used by the loader.

If the runtime needs model-owned C++ tests or a model-owned config schema,
declare them in this same `MODEL.toml` with `runtime_tests` or
`runtime_config_schemas`.

## 4. Add the E2E ownership root

Create:

```text
tests/e2e/models/<e2e-family>/
  MODEL.toml
  manifests/<case-name>.json
  test_<e2e-family>_e2e.py
  runner.py
  e2e_plugins/
  <focused family tests and optional thresholds>
```

Copy the small `test_<e2e-family>_e2e.py` shim from the closest current family,
rename it for the new family, and keep its import of the sibling `runner.py`.
This entry point is required: `tools/test_impact.py` selects
`test_<e2e-family>_e2e.py::test_model_e2e[<manifest-name>]` for model-owned E2E
coverage. The descriptor and `runner.py` alone do not create that pytest node.

The E2E index declares every JSON manifest and the defaults for each task
strategy:

```toml
id = "example"
plugin = "example"
test_manifests = [
  "manifests/example-small-fp16.json",
]

[e2e_defaults.text_generation_causal]
reference_backend = "hf_transformers"
oracle_level = "L1_external_reference"
stages = [
  { name = "full_generation", required = true },
]
```

Use an existing JSON manifest for the same task contract as the schema
reference. Every new manifest needs, at minimum, a unique `name`, concrete
`hf_id`, output `bundle`, logical `family`, exact model-owned
`runtime_strategy`, `task_strategy`, precision/build settings, and a non-empty
`testcases` array. Each testcase must name a real registered reference family
and user contract; do not invent placeholder oracle names.

The E2E `task_strategy` is the generic contract (`text_generation_causal`,
`vision_language_generation`, `speech_to_text`, and so on). It is intentionally
different from the model-owned runtime dispatch key.

## 5. Validate ownership and build the model target

From the repository root:

```bash
python3 tools/model_ci.py validate

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target trtmc trtmc_model_example -j
```

`model_ci.py validate` must report the new logical model and no descriptor
errors. CMake configure must reject missing sources, duplicate strategies, or
invalid runtime manifest entries.

## 6. Build a smoke bundle, then run the family validator

Set the concrete model reference once. A direct public-CLI build is useful for
inspection, but it is a separate smoke artifact:

```bash
MODEL_REF=example-org/example-small

./build/trtmc build "$MODEL_REF" \
  -o /tmp/example-small.bundle \
  --precision fp16 \
  --max-cache-length 256

./build/trtmc inspect /tmp/example-small.bundle --list-engines
```

Run the family validator separately and give its output directory an explicit
location:

```bash
VALIDATION_DIR=/tmp/trtmc-example-validation
export ENGINE_DIR="$VALIDATION_DIR/e2e"
mkdir -p "$VALIDATION_DIR" "$ENGINE_DIR"

./scripts/validate_family.sh "$MODEL_REF" \
  --binary ./build/trtmc \
  --bundle-dir "$VALIDATION_DIR" \
  --max-cache-length 256 \
  --isolate-model-plugin
```

Inspect the direct smoke bundle before inference. Confirm that `family`,
`runtime_strategy`, section layout, precision, and TensorRT metadata match the
three descriptors.

`validate_family.sh` does not consume `/tmp/example-small.bundle`. It builds
`$VALIDATION_DIR/example-org_example-small.bundle` directly, then runs the
applicable inspection and parity checks. If a matching E2E manifest is found,
the script invokes that pytest node with `--rebuild-engines`; that E2E run
builds the manifest's configured checkpoint independently rather than testing
the validator-built bundle. A missing matching E2E manifest is reported as a
warning. `--isolate-model-plugin` additionally checks that the requested
strategy can be satisfied by the owning DSO rather than a stale installed
plugin.

## 7. Optionally rebuild and run the declared E2E case independently

For a separate manifest-driven rebuild, use the manifest `name`, not the
filename stem:

```bash
E2E_MODEL=example-small-fp16
E2E_FAMILY=example
ENGINE_DIR=/tmp/trtmc-example-engines

PYTHONPATH=python:. pytest \
  "tests/e2e/models/${E2E_FAMILY}/test_${E2E_FAMILY}_e2e.py::test_model_e2e[${E2E_MODEL}]" \
  -v \
  --e2e-model "$E2E_MODEL" \
  --engine-dir "$ENGINE_DIR" \
  --trtmc-binary ./build/trtmc \
  --model-plugin-dir ./build/models \
  --rebuild-engines
```

`--rebuild-engines` deliberately builds the manifest's configured checkpoint
into `ENGINE_DIR`; it does **not** revalidate the direct smoke bundle or the
bundle built by `validate_family.sh`. Treat this as independent manifest-route
evidence and retain the validator's artifacts separately. In either flow, the
evidence—not the presence of the three descriptors alone—is the acceptance
signal.

{/* Collaborative review anchor: batch 2. */}
