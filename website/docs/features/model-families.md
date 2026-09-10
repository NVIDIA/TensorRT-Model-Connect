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

### K2-Horizon-Uno task contract

`IFM/K2-Horizon-7B-Uno` is an adapter-only checkpoint. The independent
`k2_horizon_uno` family resolves its pinned `IFM/K2-Horizon-7B` base and
compiles the rank-128 conditional LoRA into one BF16 TensorRT engine. Build it
directly from the adapter ID:

```bash
python -m tensorrt_model_connect build IFM/K2-Horizon-7B-Uno \
  --revision ec92bbd768f4a404319625204544782e3377bcd7 \
  --precision bf16 \
  -o k2-horizon-7b-uno.bundle
```

Run the qualified high-reasoning chat path with the family-owned linear mode:

```bash
trtmc run k2-horizon-7b-uno.bundle \
  --runtime-root /opt/trtmc/lib \
  --prompt "Reply with the word OK." \
  --max-new-tokens 26 \
  --temperature 0 \
  --top-k 1 \
  --generation-mode linear_spec_lora \
  --block-length 8 \
  --use-chat-template true \
  --enable-thinking true
```

The initial runtime supports batch-one deterministic greedy generation with
linear Psi-Spec and block lengths from 1 through 8. Block 8 is the qualified
default. It also supports the pinned single-user, high-reasoning chat template
and an autoregressive control mode. String prompts are currently limited to
ASCII and fail closed before tokenization otherwise. Stochastic sampling, tree
verification, batching, tensor parallelism, quantization, the 0.9B adapter, and
long-context qualification remain outside this contract. The
committed-token-per-forward receipt is an algorithmic diagnostic, not a
wall-clock speedup claim.

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
