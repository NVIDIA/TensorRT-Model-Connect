---
name: setup-trtmc-environment
description: >-
  Prepare a TensorRT-Model-Connect development or validation environment from
  a fresh checkout on a host without a known working setup.
---

# Set Up the Environment

Start from the checkout and current repository documentation, not a remembered
machine or container name.

1. Resolve the repository root and read
   `website/docs/getting-started/source-build.md`, `Dockerfile`, and
   `apps/devtoolkit/README.md`.
2. Inspect host architecture, available disk, Docker access, GPU visibility,
   driver, and compute capability only as needed for the requested work.
3. Prefer the existing `apps/devtoolkit` API for a checkout-owned environment.
   Use its Docker path for the supported development image or its local path
   with an explicit existing Python interpreter.
4. Select one family when model work is requested. Install only that family's
   optional `families/<family>/requirements.txt`; a missing file means no
   family package step.
5. Verify the checkout mount, Python import, TensorRT import, GPU visibility,
   runtime root, and requested native target before starting the real task.

Do not reuse, replace, or remove an unrelated container. Do not install GPU
drivers, reconfigure the container runtime, or change runner settings without
separate authorization. Stop and explain the unsupported host boundary when
the documented paths do not apply.

Report environment preparation separately from compilation, tests, model
correctness, performance, and deployment qualification.
