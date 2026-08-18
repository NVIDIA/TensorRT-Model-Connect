---
title: Supported Models
description: Exact checkpoints in the release support snapshot, with direct paths to task and family recipes.
---

import ModelSupportInventory from '@site/src/components/ModelSupportInventory';

This page is the model-support entry point. Use the table to find an exact
Hugging Face checkpoint and TRTMC profile, then open
[Model Recipes](model-recipes.md) to browse declared E2E recipes by task and
model family.

:::info One table, one source

The table below is rendered from `website/data/model-support-matrix.md`; the
project README links to this page instead of carrying a second copy. The docs
build resolves each row&apos;s TRTMC family from the exact matching E2E manifest
profile. Hugging Face `model_type`, `architectures`, and Diffusers
pipeline-class values come from the revision-pinned metadata snapshot in
`website/data/hf-model-metadata.json`; the same snapshot renders every family
recipe page.

:::

## How to read support

- A row applies only to its exact `hf_id`, checkpoint resolution, TRTMC
  profile, and build configuration.
- `model_type` identifies the Hugging Face config family. Architecture names
  identify the checkpoint class or, for Diffusers checkpoints, the pipeline
  class. They are not interchangeable with the TRTMC family ID.
- Architecture metadata describes the source checkpoint. The supported TRTMC
  task/head contract is shown separately and documented on the linked family
  page. It may
  intentionally consume only the base graph, such as an encoder that returns
  hidden states rather than a pretraining head.
- For a manifest without `hf_revision`, the metadata snapshot SHA pins only
  the displayed configuration fields; it does not retroactively turn that
  unpinned recipe into an exact-revision support claim.
- If a retained release row and the current manifest disagree on `hf_id`, the
  table shows both values. The GB300 light remains attached to the release-row
  checkpoint; the current manifest value identifies today&apos;s declared CLI
  recipe.
- Untested fine-tunes from the same family are best-effort compatible; they are
  not verified supported checkpoints.
- Platform specialization identifies a provider integration for an exact
  checkpoint or tuple. It does not replace native-runtime platform support.
- Performance color is evidence for the dated release comparison described
  below, not a blanket compatibility or correctness guarantee.

## Release performance snapshot

This is the completed July 29, 2026 release comparison on NVIDIA GB300 at
source revision `508613d0bcc7003b123cf5be3d1b3f6e6c6cb667`. It covers 105
unique single-process release profiles across 76 families. Distributed and L0
smoke profiles are outside this matrix.

Each light compares TRTMC inference p50 with the row&apos;s declared reference
under matching workload, output, and timing contracts:

- **🟢 Green:** TRTMC is more than 5% faster than the reference.
- **🟡 Yellow:** TRTMC is within 5% of the reference.
- **🔴 Red:** TRTMC is more than 5% slower than the reference.
- **— Not supported:** the profile is explicitly unsupported on that platform.

Bundle preparation, model loading, compilation when used, and warmup are
excluded from infer-p50 values. Baselines are declared per row and are not
uniformly `torch.compile`. See the
[revision-matched comparison contract](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/508613d0bcc7003b123cf5be3d1b3f6e6c6cb667/benchmarks/performance/release.yaml)
and [benchmark documentation](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/508613d0bcc7003b123cf5be3d1b3f6e6c6cb667/benchmarks/performance/README.md)
for reproduction and interpretation details.

<ModelSupportInventory variant="performance" />

## Declared recipes beyond the release snapshot

The support table is a retained release snapshot. The current checkout may
contain additional E2E manifests, L0 replacements, and distributed variants.
Those declarations are available through [Model Recipes](model-recipes.md),
organized by a Hugging Face-derived task taxonomy → model family → exact
manifest recipe. Local source packages use the nearest applicable task
category without claiming a Hugging Face repository.

Platform specializations will roll out in phases aligned with model coverage
available in TensorRT Edge-LLM and TensorRT-Model-Connect releases. Each batch
must document the exact qualified model, target, precision, and configuration
tuple.

{/* Collaborative review anchor: batch 2. */}
