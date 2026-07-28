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

See [Python Builder Units](../unit-design/python-builder.md#family-plugins) for
the live flow.

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

## 6. Build a smoke bundle, then validate an exact staged candidate

Set the concrete model reference once. A direct public-CLI build is useful for
inspection, but it is a separate smoke artifact:

```bash
MODEL_REF=example-org/example-small

./build/trtmc build "$MODEL_REF" \
  -o /tmp/example-small.trtfb \
  --precision fp16 \
  --max-cache-length 256

./build/trtmc inspect /tmp/example-small.trtfb --list-engines
```

Run the family validator separately and give its candidate and E2E proof
directories explicit locations:

```bash
VALIDATION_DIR=/tmp/trtmc-example-validation

./scripts/validate_family.sh "$MODEL_REF" \
  --binary ./build/trtmc \
  --bundle-dir "$VALIDATION_DIR" \
  --engine-dir "$VALIDATION_DIR/e2e" \
  --e2e-model example-small-fp16 \
  --max-cache-length 256 \
  --isolate-model-plugin
```

Inspect the direct smoke bundle before inference. Confirm that `family`,
`runtime_strategy`, section layout, precision, and TensorRT metadata match the
three descriptors.

`validate_family.sh` does not consume `/tmp/example-small.trtfb`. It builds a
new candidate in a hidden staging directory under `--bundle-dir`, inspects and
tests that exact candidate, and exposes it to the selected E2E node through a
temporary engine directory without asking the harness to rebuild from the
manifest's `hf_id`. A missing matching E2E manifest is a failure. The script
publishes the candidate as
`$VALIDATION_DIR/example-org_example-small.trtfb` only after every gate passes;
on failure it leaves no newly published candidate at that destination.
`--isolate-model-plugin` additionally proves that the requested strategy can
be satisfied by the owning DSO rather than a stale installed plugin.

Add `--trust-remote-code` only after reviewing and pinning the model repository
and only when native tokenizer discovery or generation actually requires its
custom Hugging Face code. The native path forwards the flag to every shared
tokenizer probe and to family fallbacks that can load repository code; it is
false by default. An optimized adapter receives the public option separately
and may reject it.

### Optional tokenizer repair hook

If the native family requires tokenizer repair that standard Hugging Face
slow-to-fast conversion cannot provide, add:

```python
def ensure_tokenizer_json(
    self,
    model_dir,
    *,
    previous_error=None,
    trust_remote_code=False,
) -> bool:
    ...
```

Once repair is needed, standard conversion always runs first. The family hook
runs only after it fails and while any rejected original `tokenizer.json` is
quarantined in a durable hidden recovery directory. Do not depend on that
original remaining at its canonical path. Return true only after writing
`model_dir/tokenizer.json` as a non-empty regular, non-symlink file accepted by
the native tokenizer validator. A false return, exception, missing file,
unsafe file type, or incompatible content causes the outer transaction to
remove the candidate and restore the original.

The coordinator holds the shared, reentrant repair lock across standard
conversion, the family hook, validation, commit, and rollback. The persistent
`.trtmc-tokenizer-repair.lock` sentinel is created before the first canonical
tokenizer mutation and is intentionally retained, but is not bundled. A hook
must not delete or replace it. Family helpers that can also be called directly
must use the same repair-lock helper and revalidate `tokenizer.json` after
acquiring it; a transient valid-looking candidate is not a committed result
until the owning hook returns and the outer transaction validates it. A forked
child discards inherited repair ownership, so it must acquire fresh ownership
before modifying tokenizer state rather than relying on an inherited lexical
context.

Successful repair creates or replaces `tokenizer.json` in the resolved model
directory, so local checkpoint directories must be writable. When an original
existed, a failed-candidate cleanup or restore failure identifies the concrete
`original-tokenizer.json` path retained for manual recovery. With no original,
ordinary failed-candidate cleanup leaves the canonical path absent; if cleanup
itself fails, the unsuccessful candidate can remain there and the terminal
error reports that cleanup failure without claiming an original-recovery path.
If the initial recovery directory cannot be prepared or the initial move
fails, the canonical original remains untouched and repair stops. Cleanup of
an old artifact after a compatible replacement commits is best-effort. A
cleanup failure does not turn the successful repair into a hook or build
failure; the compatible canonical file remains installed and a warning names
the recovery directory where cleanup residue may remain.

For diffusion families,
`diffusion_tokenizer_bundle_sections()` owns tokenizer-directory priority and
must invoke its supplied `ensure_tokenizer_json` callback for each required
directory before returning sections. The builder then detects special tokens
and reconciles bundle config from the repaired files.

## 7. Optionally rebuild and run the declared E2E case independently

The validator above has already run the declared E2E node against its exact
staged candidate. For a separate manifest-driven rebuild, use the manifest
`name`, not the filename stem:

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
into `ENGINE_DIR`; it does **not** revalidate the staged candidate or the
published bundle from `validate_family.sh`. Treat this as independent
manifest-route evidence. For evidence about the exact `MODEL_REF` supplied to
the validator, retain the validator's E2E result, comparison artifacts, and
published validated bundle. In either flow, the evidence—not the presence of
the three descriptors alone—is the acceptance signal.
