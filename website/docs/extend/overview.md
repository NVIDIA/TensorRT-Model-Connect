---
title: Extend TensorRT-Model-Connect
description: Choose the narrow owner for a family, application, backend, or public-contract change.
---

import Diagram from '@site/src/components/Diagram';

Start by choosing the owner of the behavior.

<Diagram
  src="/img/diagrams/trtmc-extension-decision.svg"
  alt="Extension work stays in one model family or application unless a proven model-independent public contract must change"
  caption="Most model contributions change one family directory only."
/>

## Add or change a model

Use [Add a Model Family](add-model-family.md). A family owns its checkpoint
matching, build, optional dependencies, bundle sections, native runtime, and
tests under `families/<family>/`.

Do not add the family to a central registry or source list. Do not import a
sibling family. A plain `model.py::build(request, writer)` function is the
build boundary; inheritance is forbidden.

## Add an application or example

Put reusable applications in `apps/` and focused demonstrations in
`examples/`. They depend one way on the public Python build API or C++ Task
API. Core and families must not depend on them.

The existing benchmark and BYOK paths are examples of this boundary.

## Add backend behavior

A backend implements the abstract Engine API without knowing any family. Use
this path only when the behavior is truly engine-wide. Model-specific plugins
and execution policy remain in the owning family.

See [Backends and Family Configuration](../features/config-and-backends.md).

## Change a public contract

A change to `BuildRequest`, `BundleReader`, `trtmc/task.h`, or the Engine
API affects every consumer. Make it only for a current, model-independent
requirement, keep the interface narrow, and validate representative users.

Code similarity, possible future reuse, or a desire to remove duplication is
not sufficient reason to add shared infrastructure.
