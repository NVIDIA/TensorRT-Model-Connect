# Simple Model Plugin Encapsulation Plan

Goal: each model is a self-contained plugin. Adding or changing one model
should require touching only that model's three folders:

```text
builder/<model>/
runtime/<model>/
test/<model>/
```

For this repository, the current equivalents are:

```text
python/tensorrt_model_connect/families/<model>/
src/runtime/models/<model>/
tests/e2e/models/<model>/
```

Duplication is allowed. Prefer duplicated model-local code over shared helper
code that couples unrelated models.

## Hard Rules

1. A model must live entirely in its three folders:
   - builder code in `builder/<model>/`
   - runtime code in `runtime/<model>/`
   - E2E and model tests in `test/<model>/`
2. A model change must not require edits to another model's folders.
3. A model change must not require edits to central model maps, global source
   lists, or shared E2E runner/comparator/reference code.
4. Model-local builder code may duplicate graph helpers, layer builders, tensor
   naming helpers, config parsing, and validation logic.
5. Model-local runtime code may duplicate host-side preprocessing,
   postprocessing, scheduling, decode policy, and helper logic.
6. Model-local tests may duplicate runner, comparator, threshold, reference,
   and artifact code.
7. Shared infrastructure is limited to:
   - TRT model/API abstraction
   - plugin registration
   - plugin lookup/loading
8. Nothing else should be shared by default.

In particular, do not keep broad shared model helper modules such as
`python/tensorrt_model_connect/graph_ops.py` as required cross-model
infrastructure. Copy the needed logic into each model folder.

## Target Shape

Each model owns this structure:

```text
builder/<model>/
  MODEL.toml
  plugin.py
  build.py
  graph_ops.py
  graph_blocks.py
  weights.py
  tests/

runtime/<model>/
  MODEL.toml
  plugin.cpp
  pipeline.cpp
  pipeline.h
  local_helpers.cpp
  local_helpers.h
  tests/

test/<model>/
  MODEL.toml
  manifests/
  data/
  runner.py
  comparator.py
  reference.py
  thresholds/
  waives.txt
```

Names can vary, but the ownership rule cannot.

## Shared Infra Only

The shared host layer should expose only:

```text
core/
  trt_model_api
  trt_tensor_api
  trt_backend_api
  plugin_registry
  plugin_loader
```

Allowed shared responsibilities:

- load a plugin by id
- register plugin capabilities
- call a plugin through a stable API
- provide generic TRT engine/module/tensor wrappers
- report clean errors when a plugin is missing

Not allowed as shared model infrastructure:

- graph construction helpers
- model layer builders
- model config adapters
- token naming conventions
- runtime pipeline helpers
- audio/diffusion/VL helper libraries
- E2E runners
- E2E comparators
- E2E references
- model threshold logic

If two models need similar code, duplicate it first. Extract later only if it
is truly part of the generic TRT/API layer.

## Runtime Requirement

Each model runtime must build as its own shared library:

```text
libtrtmc_model_<model>.so
```

The main runtime must load only the needed model plugin for the selected
bundle/model.

Validation for each model:

```bash
cmake --build build --target trtmc_model_<model>
ldd build/models/<model>/libtrtmc_model_<model>.so
nm -D build/models/<model>/libtrtmc_model_<model>.so
```

Pass criteria:

- the model `.so` loads independently
- it does not link to another `libtrtmc_model_*.so`
- it registers only its own plugin id and strategies
- it works when unrelated model `.so` files are absent

## Builder Requirement

The Python builder must import only the target model builder and shared
TRT/API abstraction.

Validation for each model:

```bash
pytest builder/<model>/tests -v
python -m tensorrt_model_connect build <model-or-dir> -o /tmp/<model>.trtfb
```

Pass criteria:

- no global scan/import of every model family
- no dependency on another model's builder folder
- duplicated local graph/layer/helper code is acceptable

## Test Requirement

Each model owns its E2E runner, comparator, reference, thresholds, data, and
manifests.

Validation for each model:

```bash
pytest test/<model> -v
```

Current-repo equivalent during migration:

```bash
pytest tests/test_e2e.py --e2e-model <model> -v
```

Pass criteria:

- only that model's E2E cases are collected
- the model builds or resolves its bundle
- the runtime loads only `libtrtmc_model_<model>.so`
- inference passes
- comparison passes
- artifacts are written under that model's test output folder

## Implementation Steps

### Step 1 - Pick the Folder Contract

Decide the final names:

```text
builder/<model>/
runtime/<model>/
test/<model>/
```

or keep the repo's current long paths temporarily.

Done when every model has a clear builder/runtime/test owner path.

### Step 2 - Add Per-Model Metadata

Add a tiny `MODEL.toml` in each of the three model folders.

It should declare:

```toml
id = "<model>"
plugin = "<model>"
runtime_library = "libtrtmc_model_<model>.so"
runtime_strategies = []
test_manifests = []
```

Done when model ownership can be discovered from the three folders.

### Step 3 - Stop Global Python Discovery

Replace global Python family scanning with direct plugin lookup:

```text
model id -> builder/<model>/plugin.py
```

Done when building one model imports only that model's builder code.

### Step 4 - Build One Runtime `.so` Per Model

Move model runtime code out of the monolithic runtime target.

Each model gets:

```text
trtmc_model_<model>
```

as a CMake target producing:

```text
libtrtmc_model_<model>.so
```

Done when `trtmc_core` no longer compiles every model pipeline.

### Step 5 - Dynamic Plugin Loading

Change runtime dispatch from "all plugins registered at startup" to:

```text
read bundle/model plugin id
load libtrtmc_model_<model>.so
register that plugin
run that plugin
```

Done when unrelated model plugins can be removed and the selected model still
runs.

### Step 6 - Move Tests Under Each Model

Move model E2E assets into:

```text
test/<model>/
```

Each model owns its own runner, comparator, reference, data, thresholds, and
waives. Duplicate code freely.

Done when `pytest test/<model> -v` is the normal model E2E command.

### Step 7 - Make Impact Selection Path-Based

Impact rules become simple:

```text
builder/<model>/** -> test <model>
runtime/<model>/** -> test <model>
test/<model>/**    -> test <model>
shared infra       -> run shared API/plugin tests
```

Done when touching one model selects only that model.

### Step 8 - CI Runs Per Model

For a one-model PR, CI runs:

```bash
cmake --build build --target trtmc_model_<model>
pytest builder/<model>/tests -v
pytest runtime/<model>/tests -v
pytest test/<model> -v
```

Plus the small shared plugin-loader/API smoke tests.

Done when one-model PRs no longer run unrelated model E2E.

## Final Acceptance

For every supported model:

```bash
cmake --build build --target trtmc_model_<model>
pytest builder/<model>/tests -v
pytest runtime/<model>/tests -v
pytest test/<model> -v
```

Then prove isolation:

```bash
mkdir -p /tmp/only-<model>
cp build/models/<model>/libtrtmc_model_<model>.so /tmp/only-<model>/
trtmc run <model-bundle> --model-plugin-dir /tmp/only-<model>
```

Then save a reference copy of `origin/main` and build it:

```bash
git clone . /tmp/trtmc-origin-main
git -C /tmp/trtmc-origin-main fetch origin main
git -C /tmp/trtmc-origin-main checkout origin/main
cmake -S /tmp/trtmc-origin-main -B /tmp/trtmc-origin-main/build
cmake --build /tmp/trtmc-origin-main/build --target trtmc
```

Run the current model plugin through the real migrated `trtmc` binary:

```bash
pytest test/<model> -v \
  --trtmc-binary "$(command -v trtmc)" \
  --model-plugin-dir /tmp/only-<model>
```

Run the same user-contract test against the saved `origin/main` build:

```bash
pytest /tmp/trtmc-origin-main/tests/test_e2e.py::test_e2e[<model>] -v \
  --trtmc-binary /tmp/trtmc-origin-main/build/trtmc
```

The migrated isolated plugin result must match the saved `origin/main` result:
same collected model contract, same pass/fail/skip status, same required
artifacts, and same comparator result. If the `origin/main` baseline fails for
that model, the migrated isolated plugin must reproduce the same failure
signature until the model contract itself is intentionally fixed.

The migration is done when:

- every model passes by itself
- every model runtime is a separate `.so`
- no model `.so` depends on another model `.so`
- adding/changing one model touches only `builder/<model>`,
  `runtime/<model>`, and `test/<model>`
- CI tests only that model for model-local changes
- pytestE2E can load only that model plugin through `trtmc` and reproduce the
  saved `origin/main` user-contract result for that model
