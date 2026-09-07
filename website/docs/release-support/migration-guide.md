---
title: Current-Tree Cutover
description: What users must rebuild after the pre-#1093 architecture was replaced.
---

PR #1093 replaced the earlier builder, registry, bundle, and runtime layout in
one cutover. There is no compatibility adapter or automated migration tool.

Use a fresh checkout and rebuild:

1. install or build the current package and native runtime;
2. install only the selected family's optional
   `families/<family>/requirements.txt`;
3. rebuild the bundle with the current `python -m tensorrt_model_connect build`
   command;
4. deploy the core, selected backend, selected family DSO, and CLI from the
   same product build; and
5. rerun the family-owned E2E or your application acceptance test.

Do not copy old descriptor files, runtime-strategy fields, central config
schemas, or compatibility readers into the current tree. The current ownership
and source paths are documented in [Source Layout](../reference/source-layout.md).

There is no released-version migration beyond this development cutover. Future
release compatibility will be documented only if and when the project makes an
explicit release promise.
