---
title: Supported Models
description: Family-owned build, runtime, and test implementations in this checkout.
---

import ModelSupportInventory from '@site/src/components/ModelSupportInventory';

Support is defined by files that live inside one self-contained
`families/<family>/` directory. Each row below points to that family&apos;s own
E2E manifest; there is no central model registry or duplicated support matrix.

<ModelSupportInventory variant="facts" />

## Declared recipes

An entry means the repository contains a builder, a native runtime DSO that
implements an abstract Task API, and a family-owned test recipe for that exact
checkpoint. It is not a promise that an unlisted fine-tune is compatible.

A manifest with `hf_id` names its Hugging Face source. A manifest without
`hf_id` intentionally represents a family-owned prepared local checkpoint; the
inventory labels it as such instead of inventing Hugging Face metadata.

<ModelSupportInventory variant="models" />

For task-oriented navigation, see [Model Recipes](model-recipes.md).
