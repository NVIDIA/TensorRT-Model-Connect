---
name: transform-model
description: >-
  Use when adding or transforming a HuggingFace model into a
  TensorRT-Model-Connect `.trtfb` bundle. Drives scoped family/strategy work,
  build-validation loops, comparison output, E2E manifest creation, and PR-ready
  reporting.
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
- Keep changes scoped to the model family, strategy, manifest, and tests needed
  for the requested model.
- Do not relax thresholds without evidence.
- Do not push directly to `main`; follow AGENTS.md and open a GitHub PR from a
  short-lived branch on the `github` remote.
- Stop after 10 build/fix iterations and report the blocker if comparison output
  still cannot be produced.

## Phase 0: Confirm Scope

Before implementation, identify and report:

- HF model ID.
- Architecture type: decoder, encoder-only, diffusion, speech/audio,
  vision-language, embedding/reranking, or other.
- Existing family plugin or closest reference family.
- Runtime strategy.
- Branch name.

Proceed when the user has confirmed or when the request already provides enough
specificity for a low-risk assumption.

## Phase 1: Setup

```bash
git fetch github main
git switch -c <branch> github/main
```

If working inside a container, ensure the container checkout is on the same
branch and points at the same commit.

## Phase 2: Implement

Common reference paths:

| Path | Purpose |
|------|---------|
| `python/tensorrt_model_connect/families/` | Family plugins |
| `python/tensorrt_model_connect/strategies/` | Build strategies |
| `python/tensorrt_model_connect/compiler.py` | Build orchestration |
| `python/tensorrt_model_connect/families/base.py` | Family plugin protocol |
| `scripts/new_family.py` | Family scaffold helper |
| `tests/e2e/models/` | E2E manifests |

Guidance by type:

- Decoder models: prefer the established decoder strategy and standard family
  plugin patterns.
- Encoder-only models: use the encoder-only strategy and BERT-style references.
- Diffusion models: implement component builds in the family plugin and inspect
  known attention/mask issues before changing thresholds.
- Multimodal/audio models: read existing family plugins and E2E harness runners
  before introducing new dataflow.

## Phase 3: Build

```bash
docker exec <container> bash -c "cd <repo> && \
  /opt/venv/bin/python -m tensorrt_model_connect build <hf_model> \
  -o engines/<model>.trtfb \
  --max-cache-length 256 --verbose"
```

If build fails, read the full error, inspect the relevant family/strategy/source
code, make the smallest fix, and retry. Log new recurring issues in the PR body.

## Phase 4: Validate

Text models:

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

## Phase 5: E2E Manifest

Create or update `tests/e2e/models/<family>/manifests/<model-name>.json` and list it in `tests/e2e/models/<family>/MODEL.toml`:

```json
{
  "name": "<model-name>",
  "hf_id": "<hf_model>",
  "bundle": "<model-name>.trtfb",
  "family": "<family_name>",
  "runtime_strategy": "<strategy>",
  "max_cache_length": 256,
  "prompt": "<test prompt>",
  "max_new_tokens": 20
}
```

Run:

```bash
docker exec <container> bash -c "cd <repo> && \
  /opt/venv/bin/python -m pytest tests/test_e2e.py::test_e2e[<model-name>] -v \
  --engine-dir engines \
  --trtmc-binary ./build/trtmc \
  --hf-python /opt/venv/bin/python \
  --rebuild-engines"
```

## Phase 6: Report

Before opening a PR, make sure the report includes:

- Model name and HF ID.
- Strategy and family plugin used.
- Bundle path.
- Commands run and pass/fail results.
- Comparison metrics or saved media outputs.
- E2E manifest and test result.
- Remaining risks or unavailable hardware.

Use `$submit-github-pr` for the final push and PR creation.
