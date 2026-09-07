---
title: Change Policy
description: How user-visible contracts change while the project has no compatibility window.
---

TensorRT-Model-Connect has not established a released compatibility window.
Current development may replace a CLI, API, bundle, family, or manifest
contract directly when the old design is no longer required.

The repository does not keep shims, aliases, dual readers, fallbacks, or
migration layers solely to preserve an obsolete development contract. A change
should instead:

- document the new current behavior;
- remove the retired implementation and tests in the same pull request;
- update examples and family-owned tests to the new contract; and
- require users to rebuild affected artifacts when formats or runtime
  interfaces change.

If a future release establishes a compatibility commitment, its exact scope
and duration must be documented with that release. This page does not create
such a commitment retroactively.
