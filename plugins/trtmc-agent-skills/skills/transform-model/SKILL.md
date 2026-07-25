---
name: transform-model
description: >-
  Use when adding or transforming a HuggingFace model into a
  TensorRT-Model-Connect `.trtfb` bundle. Distinguishes native
  family/strategy work from exact qualified optimized implementations, and
  drives build-validation loops, comparison output, qualification evidence,
  and PR-ready reporting.
---

# Transform Model

## Required Inputs

- `hf_model`: HuggingFace model ID or local model path.
- `branch`: short-lived branch name.
- `container`: optional Docker container name. Use the repo's current GB300 dev
  container pattern unless the user specifies another container.

## Ground Rules

- Confirm the exact model variant before expensive build work.
- Run GPU/TensorRT commands inside the dev container.
- Read current docs and source before implementing:
  `website/docs/features/model-families.md`,
  `website/docs/features/runtime-strategies.md`, family plugins, strategies, and
  the relevant runtime code.
- Keep changes scoped to the selected route: native family/strategy/manifest
  ownership, or a family-local optimized implementation/profile and its
  qualification proof.
- Do not relax thresholds without evidence.
- Do not push directly to `main`; follow AGENTS.md and open a GitHub PR from a
  short-lived branch on the remote whose URL is
  `https://github.com/NVIDIA/TensorRT-Model-Connect`.
- Stop after 10 build/fix iterations and report the blocker if comparison output
  still cannot be produced.

## Phase 0: Confirm Scope

Before implementation, identify and report:

- HF model ID.
- Immutable model revision and intended deployment target when evaluating an
  optimized profile.
- Architecture type: decoder, encoder-only, diffusion, speech/audio,
  vision-language, embedding/reranking, or other.
- Existing family plugin or closest reference family.
- Route:
  - native support, with a model-owned `runtime_strategy` and E2E
    `task_strategy`; or
  - an exact optimized implementation/profile for an existing family, with
    implementation ID, profile ID, target, and operation. Do not invent a
    native strategy for this route.
- Branch name.

Proceed when the user has confirmed or when the request already provides enough
specificity for a low-risk assumption.

## Phase 1: Setup

```bash
if git remote get-url github >/dev/null 2>&1; then
  GITHUB_REMOTE=github
else
  GITHUB_REMOTE=origin
fi
GITHUB_REMOTE_URL=$(git remote get-url "$GITHUB_REMOTE")
case "$GITHUB_REMOTE_URL" in
  https://github.com/NVIDIA/TensorRT-Model-Connect|\
  https://github.com/NVIDIA/TensorRT-Model-Connect.git|\
  git@github.com:NVIDIA/TensorRT-Model-Connect.git|\
  ssh://git@github.com/NVIDIA/TensorRT-Model-Connect.git) ;;
  *)
    echo "Refusing non-canonical remote: $GITHUB_REMOTE_URL" >&2
    exit 1
    ;;
esac
git fetch "$GITHUB_REMOTE" main
git switch -c <branch> "$GITHUB_REMOTE/main"
```

If working inside a container, ensure the container checkout is on the same
branch and points at the same commit.

## Phase 2: Implement

Common reference paths:

| Path | Purpose |
|------|---------|
| `python/tensorrt_model_connect/families/` | Family plugins and family-local builders |
| `python/tensorrt_model_connect/build_cli.py` | Build CLI entrypoint |
| `python/tensorrt_model_connect/engine_builder.py` | Build orchestration |
| `python/tensorrt_model_connect/families/base.py` | Family plugin protocol |
| `python/tensorrt_model_connect/families/<family>/MODEL.toml` | Python discovery metadata |
| `src/runtime/models/<family>/MODEL.toml` | Model DSO, registrar, strategy, schema, and C++ test ownership |
| `tests/e2e/models/<family>/MODEL.toml` | E2E manifest index and task defaults |
| `tests/e2e/models/<family>/manifests/` | E2E model contracts |
| `python/tensorrt_model_connect/families/<family>/<adapter>/IMPLEMENTATION.toml` | Optimized implementation identity, isolated adapter entrypoint, runtime DSO, and ABI |
| `python/tensorrt_model_connect/families/<family>/<adapter>/profiles/*.toml` | Exact model revision, target, options, artifact contract, and qualification state |
| `tests/e2e/models/<family>/<adapter>/QUALIFICATION.*.toml` | Optimized producer entrypoint, digest-pinned environment, target, profiles, and trigger ownership |

Choose one ownership contract before editing:

- For a new or changed native path, keep the Python, runtime, and E2E
  `MODEL.toml` descriptors aligned. The runtime descriptor must own the concrete
  family `runtime_strategy`, registrar, `libtrtmc_model_<owner>.so`, and C++
  tests.
- For a delegated optimized implementation of an existing family, inspect or
  add the family-local `IMPLEMENTATION.toml`, exact profile TOMLs, isolated
  adapter/runtime DSO, and matching `QUALIFICATION.*.toml`. A profile must bind
  the immutable model revision, target, effective public options, semantic
  source, and current qualification state. Do not add a synthetic
  `runtime_strategy` or native runtime `MODEL.toml` just to describe it.

Native guidance by type:

- Decoder models: copy the nearest family-owned decoder implementation and use
  a family-owned key such as `<family>_decoder_kv_cache`; do not reuse a
  retired generic key such as `decoder_kv_cache`.
- Encoder-only models: copy the nearest model-owned encoder implementation and
  declare a family-owned strategy such as `<family>_encoder_only`.
- Diffusion models: implement component builds in the family plugin and inspect
  known attention/mask issues before changing thresholds.
- Multimodal/audio models: read existing family plugins and E2E harness runners
  before introducing new dataflow.

Do not use `scripts/new_family.py` as a complete native onboarding generator. It
is a legacy Python-plugin sketch and does not create the runtime DSO, the three
`MODEL.toml` ownership descriptors, or the E2E/C++ test surfaces required by
the native contract. It also does not create an optimized
implementation/profile or its producer qualification contract.

## Phase 3: Build

```bash
docker exec <container> bash -c "cd <repo> && \
  /opt/venv/bin/python -m tensorrt_model_connect build <hf_model> \
  -o engines/<model>.trtfb \
  --max-cache-length 256 --verbose"

docker exec <container> bash -c "cd <repo> && \
  ./build/trtmc inspect engines/<model>.trtfb"
```

The public build API first resolves the family and probes only that family's
optimized implementations. Exactly one qualified model/revision/target/options
claim produces a bundle with `optimized_runtime.json` and an embedded
`libtrtmc_impl_*.so`. If zero profiles claim the request, build continues
through the native `FamilyPlugin` path. Once an optimized adapter is selected,
its build failure is terminal. Use the inspection result to record which path
actually ran.

If build fails, read the full error, inspect the selected family/strategy or
implementation/profile source, make the smallest fix, and retry. Log new
recurring issues in the PR body.

## Phase 4: Validate

Native text models:

```bash
docker exec <container> bash -c "cd <repo> && \
  ./build/trtmc run engines/<model>.trtfb \
  --prompt 'The capital of France is' \
  --max-new-tokens 20 \
  --hf-python /opt/venv/bin/python"

docker exec <container> bash -c "cd <repo> && \
  /opt/venv/bin/python tools/diff_logits.py \
  --model <hf_model> --atol 1e-2 --battery --verbose"
```

Diffusion/media models:

- Run at least two semantically distinct prompts.
- Save outputs with versioned names.
- Compare against HF or the existing E2E comparator where available.

If output is wrong, use `$debug-trt-mismatch` and iterate.

For an optimized implementation, do not substitute the generic native
logit/profiler workflow for its producer contract. Validate host-side routing
and capsule contracts, select the affected producer descriptor, and run its
declared entrypoint on the exact target:

```bash
PYTHONPATH=python:. python3 -m pytest \
  tests/tools/test_optimized_runtime_qualifications.py \
  tests/builder/test_optimized_runtime_orchestrator.py \
  tests/builder/test_optimized_runtime_capsules.py -q

python3 tools/ci/optimized_runtime_qualifications.py \
  --files python/tensorrt_model_connect/families/<family>/<adapter>/IMPLEMENTATION.toml
```

Read the selected `QUALIFICATION.*.toml` for its digest-pinned image,
`profile_glob`, target, and entrypoint. A successful exact-target producer run
is required for parity or performance claims; host-only tests are not a
substitute.

## Phase 5: Native E2E Manifest Or Optimized Qualification

For a native path, create or update
`tests/e2e/models/<family>/manifests/<model-name>.json` and list it in
`tests/e2e/models/<family>/MODEL.toml`:

```json
{
  "name": "<model-name>",
  "hf_id": "<hf_model>",
  "bundle": "<model-name>.trtfb",
  "family": "<family_name>",
  "runtime_strategy": "<family_owned_strategy>",
  "task_strategy": "<task_strategy>",
  "precision": "fp16",
  "max_cache_length": 256,
  "trust_remote_code": false,
  "testcases": [
    {
      "name": "<model-name>",
      "trace_id": "IT-E2E-<MODEL>-01",
      "reference_family": "<reference_family>",
      "user_contract": "<user_contract>",
      "prompt": "<test prompt>",
      "max_new_tokens": 20
    }
  ]
}
```

Copy the closest current manifest with the same `task_strategy`; fields differ
for audio, vision, diffusion, time-series, and other non-text contracts. The
manifest loader requires a known runtime/task-strategy pair and a non-empty
`testcases` list. This JSON example and its `runtime_strategy` requirement apply
only to the native route.

Run:

```bash
docker exec <container> bash -c "cd <repo> && \
  /opt/venv/bin/python -m pytest tests/test_e2e.py::test_e2e[<model-name>] -v \
  --engine-dir engines \
  --trtmc-binary ./build/trtmc \
  --hf-python /opt/venv/bin/python \
  --rebuild-engines"
```

For an optimized path, do not manufacture a native JSON manifest solely to
provide `runtime_strategy`. Update the exact family-local profile and matching
`QUALIFICATION.*.toml` producer contract instead. Confirm the produced bundle's
`optimized_runtime.json` names the expected implementation/profile and its
artifact tree contains the implementation DSO.

## Phase 6: Report

Before opening a PR, make sure the report includes:

- Model name and HF ID.
- Immutable revision and target.
- Native family plugin and strategy, or optimized implementation/profile and
  embedded DSO.
- Bundle path.
- Commands run and pass/fail results.
- Comparison metrics or saved media outputs.
- Native E2E manifest result or exact-target optimized producer result.
- Remaining risks or unavailable hardware.

Use `$submit-github-pr` for the final push and PR creation.
