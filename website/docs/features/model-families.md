---
title: Model Families
---

A model family is one complete vertical slice under `families/<family>/`:

```text
families/<family>/
  support.py
  model.py
  requirements.txt          # optional
  runtime/CMakeLists.txt
  runtime/*.cpp
  tests/test_e2e.py
  tests/manifests/*.json
  tests/thresholds/*.json    # optional
```

## Discovery and build

`support.py` performs dependency-free exact matching of root model metadata and
declares the supported tasks plus default. It must not import the family build,
TensorRT, PyTorch, Transformers, or another family.

After exactly one match, the core imports only that family's `model.py` and
calls its plain `build(request, writer)` function. The family owns config and
weight interpretation, TensorRT graph topology, precision/quantization policy,
engine construction, and bundle-section semantics.

## Runtime and validation

The directory name is also the runtime DSO identity:
`libtrtmc_model_<family>.so`. The family factory implements one or more
abstract interfaces in `trtmc/task.h` and owns all preprocessing,
postprocessing, dispatch, state, sampling, and engine binding.

Tests, exact checkpoint manifests, fixtures, thresholds, and reference/oracle
logic live in the same directory. A family with extra build or reference
dependencies declares them in its own plain `requirements.txt`.

## Live inventory

The website's [Models & Recipes](../models-recipes/overview.md) pages are
generated from `families/*/support.py` and `tests/manifests/*.json`. Validate the
physical inventory with:

```bash
python3 -m tools.model_ci validate
```

The project covers text decoders and encoders, MoE and recurrent networks,
vision-language, speech/audio, diffusion, perception, time-series, robotics,
and world-model tasks. Support remains exact and model-owned: a similar
architecture, parser option, or sibling family's passing test is not evidence
for another checkpoint.

## Isolation rule

Families never import, include, or link siblings. Similar model code is copied
by design so a team can implement, validate, change, and revert one family
without modifying another. Shared code is limited to model-agnostic contracts
and mechanics described in the
[Architecture](../architecture/ai-native-horizontal-scaling.md).
