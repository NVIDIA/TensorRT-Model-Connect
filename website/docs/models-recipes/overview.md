---
title: Supported Models
description: Family-owned build, runtime, and test implementations in this checkout.
---

import ModelSupportInventory from '@site/src/components/ModelSupportInventory';

Select a **checkpoint** to see its exact recipe: build settings, revision,
test inputs, and configuration checks. Select a **family** to browse its
recipes, command parameters, configuration fields, defaults, and constraints.
These pages are generated from the current `families/<family>/` sources.

<ModelSupportInventory variant="facts" />

## Declared recipes

An entry means the repository contains a builder, a native runtime DSO that
implements an abstract Task API, and a family-owned test recipe for that exact
checkpoint. It is not a promise that an unlisted fine-tune is compatible.

Configuration checks use the current command parser, build-request validation,
and family source declarations. A passing configuration check does not establish
a GPU build, inference result, or parity result. Each detail page reports the
checks that ran and any settings requiring further verification.

A manifest with `hf_id` names its Hugging Face source. A manifest without
`hf_id` intentionally represents a family-owned prepared local checkpoint; the
inventory labels it as such instead of inventing Hugging Face metadata.

<ModelSupportInventory variant="models" />

For task-oriented navigation, see [Model Recipes](model-recipes.md).

## All families

<ModelSupportInventory variant="families" />
