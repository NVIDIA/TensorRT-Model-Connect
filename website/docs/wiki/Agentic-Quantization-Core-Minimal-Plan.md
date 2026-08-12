---
title: Quantization Core Implementation Note
unlisted: true
pagination_next: null
pagination_prev: null
---

The reader-facing explanation moved to
[Quantization](/features/quantization). This unlisted compatibility page keeps
the source-level ownership contract consumed by repository checks.

## Ownership standard

### Family agent default scope

The family owns supported formats, graph integration, exclusions, calibration,
checkpoint semantics, and model evidence.

### Core agent scope

The shared core owns only model-independent quantization primitives and
contracts.

- Shared quantization code must not import specific family plugins.
- Shared quantization code must not branch on concrete family names.
- Family-specific quantization policy belongs in plugin hooks.

## Test enforcement

`tests/builder/test_quantization_ownership.py` enforces this boundary. Follow
the canonical [Quantization](/features/quantization) page for the complete
workflow and current command examples.

{/* Collaborative review anchor: batch 2. */}
