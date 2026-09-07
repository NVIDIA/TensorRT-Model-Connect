---
title: Premerge and Nightly Model Selection
description: How family-owned manifests choose fast premerge evidence without creating a central replacement registry.
---

The current architecture does not maintain a central L0-replacement table.
Each family owns all of its E2E cases under
`families/<family>/tests/manifests/` and marks at least one real testcase with:

```json
{
  "name": "small-real-case",
  "premerge": true
}
```

Premerge runs the owner-selected cases. Nightly can run every declared case,
including larger checkpoints, longer outputs, higher resolutions, additional
precisions, and multi-device configurations.

## Selection rules

- A premerge case must exercise the family's real build, bundle, DSO, Task, and
  output contract.
- A smaller checkpoint is valid only when it preserves the implementation path
  whose regression the case is meant to catch.
- Reduced prompt length, image size, frame count, decode budget, or denoising
  steps must remain explicit in the testcase.
- A fast case does not replace nightly scale, memory-pressure, topology, or
  long-context evidence.
- The family test must not silently fall through to a generic comparator or a
  different checkpoint.
- Failures are fixed or explicitly reported; selection is not changed merely
  to make a gate pass.

The manifest is authoritative. Do not copy a model list into this page or a
second CI registry.
