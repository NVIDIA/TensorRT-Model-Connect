---
title: Limitations / Known Issues
description: Current boundaries for support and compatibility claims.
---

- The site documents the current `main` development state; it is not an
  immutable release snapshot.
- Family manifests are executable specifications, not automatically fresh pass
  receipts for every target.
- Some manifests do not pin an `hf_revision`. Pin one for reproducible support
  or performance evidence.
- A parser option does not prove that every family supports the option.
- Bundle format 1 and the native interfaces have no cross-release
  compatibility promise. Use core, backend, family DSOs, and CLI from one
  product build and rebuild affected bundles after contract changes.
- `trtmc inspect` validates and prints the bounded container header and section
  table; it does not deserialize an engine or prove inference.
- Runtime loading requires an explicit runtime root. There is no environment,
  current-directory, installed-path, alias, or sibling-family fallback search.
- Multi-device behavior is family-owned. Generic TP/CP build fields do not
  establish support, and a replicated rank selection does not imply NCCL.
- Reference parity, target qualification, and performance are separate evidence
  tiers.

Use the current source, exact family manifest, and exact-head test evidence when
a historical page or external example differs.
