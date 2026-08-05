---
name: setup-trtmc-environment
description: >-
  Prepare a TensorRT-Model-Connect development or deployment-validation
  environment from a fresh checkout on an unfamiliar host. Use before builds,
  tests, packaging, or runtime work when no working repo environment is known.
---

# Set Up The Environment

Start from the checkout, not from existing container names or machine-specific
workspace conventions.

1. Find the repository root with `git rev-parse --show-toplevel`.
2. Read the current environment guide, `Dockerfile`, and matching
   `scripts/docker_build_*.sh` / `scripts/docker_run_*.sh`. Treat them as the
   source of truth; do not copy assumptions from another host.
3. Inspect the host as needed (`uname`, `docker info`, `nvidia-smi`, free disk)
   and select a repo-supported path. If none matches, explain the gap and stop.
   Do not install drivers or reconfigure the container runtime without separate
   authorization.
4. Build the selected repo image when it is missing or stale, using the repo's
   build path. Do not prebuild unrelated model-family reference profiles.
5. Start a fresh container from that image with this checkout mounted
   read-write for development or read-only for deployment validation. Choose
   paths and names for this checkout; do not reuse, migrate, or remove unrelated
   containers.
6. Verify the checkout mount, required GPU visibility, and TensorRT import, then
   run the requested build, test, packaging, or runtime command.

Keep model-specific dependencies on demand and follow each family's own lock
and verification files. Report setup evidence separately from compilation,
tests, model parity, performance, and production-deployment evidence.
