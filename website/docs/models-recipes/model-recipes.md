---
title: Model Recipes
description: Browse declared TensorRT-Model-Connect recipes by Hugging Face task and model family.
---

import ModelRecipeTaskIndex from '@site/src/components/ModelRecipes';

Model recipes are organized in three levels:

1. Choose a task using the Hugging Face task taxonomy below.
2. Open the task to see every model family with a declared recipe.
3. Open a family to see its exact manifest recipes, applicable `trtmc` CLI
   commands, Hugging Face `model_type` and architecture or pipeline class,
   TRTMC task/head contract, and every family-owned `--set` configuration key.

Every recipe row is generated from a model-owned manifest under
`tests/e2e/models/<family>/manifests/`. A manifest declares an executable test
contract; it is not, by itself, a current hardware pass receipt. Family config
tables are generated from registered Python or C++ config schemas rather than
maintained by hand. Use
[Supported Models](overview.md) for the retained release-support snapshot.

<ModelRecipeTaskIndex />
