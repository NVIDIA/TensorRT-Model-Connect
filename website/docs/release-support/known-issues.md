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
- Native and platform-specialized runtime paths have different artifact,
  dependency, configuration, and qualification boundaries.
- Native bundles depend on compatible installed model/backend DSOs and host
  libraries; they are not complete operating-system images.
- `trtmc inspect --list-engines` recognizes native plan naming and can report no
  engines for a valid provider-owned optimized artifact layout.
- Multi-device execution is currently model-owned and topology-fixed; generic
  TP/CP flags do not establish blanket support.
- The public detection API does not establish a supported detector without a
  model-owned runtime strategy and E2E manifest.

Use the current source, exact manifest, and test evidence when this page and a
newer implementation differ.
