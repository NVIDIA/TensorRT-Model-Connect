---
title: Limitations / Known Issues
description: Current boundaries that users must account for when interpreting feature and support claims.
---

- The site is currently `Latest`; there is no immutable release documentation
  snapshot yet.
- E2E manifests are executable specifications, not automatically current pass
  receipts for every target.
- Many manifests do not pin `hf_revision`. Pin an immutable revision when
  producing reproducible support or performance evidence.
- A parser option is not proof that every family implements or qualifies that
  option.
- Native and complete-network platform offload have different artifact,
  dependency, configuration, and qualification boundaries.
- Native bundles depend on compatible installed core/backend/family DSOs and host
  libraries; they are not complete operating-system images.
- `trtmc inspect` prints header metadata and a section inventory, but it does
  not decode every family-specific section and has no engine-listing mode.
- Multi-device execution is currently model-owned and topology-fixed; generic
  TP/CP flags do not establish blanket support.
- A public task interface does not establish support without a family-owned
  implementation, manifest, and E2E evidence.

Use the current source, exact manifest, and test evidence when this page and a
newer implementation differ.

{/* Collaborative review anchor: batch 2. */}
