---
title: Developer Guide
description: Understand the thin shared contracts and contribute an independently owned model family.
---

The project is organized around one rule: a model family owns its complete
build, runtime, and validation vertical slice. Shared code owns only the
contracts needed to discover that family, write/read a bundle, load its DSO,
and call it through abstract Task and Engine interfaces.

## Read by change type

| Change | Start here | Expected owner |
| --- | --- | --- |
| Understand dependencies and control transfer | [Architecture](../architecture/ai-native-horizontal-scaling.md) | No edit |
| Add or change a model | [Add a Model Family](../extend/add-model-family.md) | `families/<family>/**` |
| Implement family runtime behavior | [Runtime Extension](../extend/add-runtime-strategy.md) | `families/<family>/runtime/**` |
| Add family build configuration | [Family Configuration](../extend/add-config-schema.md) | `families/<family>/model.py` and family-owned helpers |
| Add an explicit external kernel | [Bring Your Own Kernel](../tutorials/advanced/bring-your-own-kernel.md) | public BYOK API plus the consuming family/example |
| Validate a contribution | [Model Validation](../extend/model-validation.md) | `families/<family>/tests/**` |
| Submit a pull request | [Contributor Quickstart](../extend/contributing.md) | the smallest owned diff |

## Dependency direction

```text
applications/examples/benchmark
            -> public build, load, Task, and BYOK APIs

family model.py -> build contract + BundleWriter + TensorRT build API
family runtime  -> factory + BundleReader + abstract Task/Engine APIs
backend         -> abstract Engine API
```

The arrows point from implementation to dependency. Core does not import or
link a concrete family, a family does not depend on another family, and core or
families do not depend on application code.

## Normal contribution shape

```text
families/my_family/
├── support.py
├── model.py
├── requirements.txt       # only when this family needs extra packages
├── runtime/
└── tests/
```

There is no central model registry or runtime-strategy switch to update. Python
builders are plain functions and cannot inherit from a base class. Similar
family code may be copied to preserve isolation.

Start validation with one minimal real path. Once checkpoint resolution,
family build, bundle load, Task execution, and output validation pass, add only
the additional cases required by the contribution.

User-facing task instructions belong in [User Guides](../user-guides/overview.md).
Progressive labs belong in [Tutorials](../learning-path.md). Keep implementation
and ownership details here so a normal user does not need to understand the
source tree.
