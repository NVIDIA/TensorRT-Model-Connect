---
name: trtmc-agent-container
description: >-
  Find, provision, and use the TensorRT-Model-Connect development container
  that mounts the current checkout. Use for CMake builds, C++ tests, pytest,
  smoke tests, or other commands that require the repository CUDA, TensorRT,
  and Python environment. Reuses an existing matching container without
  renaming it; when none exists, rebuilds the standard development image from
  the current checkout and starts a new isolated container automatically.
---

# TRTMC Agent Container

Use the bundled runner instead of hand-writing container discovery and
provisioning commands.

## Quick Start

From anywhere inside a TensorRT-Model-Connect checkout:

```bash
RUNNER=plugins/trtmc-agent-skills/skills/trtmc-agent-container/scripts/trtmc_agent_container.py
python3 "$RUNNER" ensure --pretty
python3 "$RUNNER" run -- cmake --build build -j8
python3 "$RUNNER" run -- ctest --test-dir build --output-on-failure
```

`run` calls `ensure` automatically. Calling `ensure` explicitly is useful when
the user asks to prepare or inspect the environment before running work.

## Discovery And Provisioning Contract

1. Resolve the current repository root and reject unrelated repositories.
2. Inspect container mounts and reuse a single container that mounts this exact
   checkout. The container name is not used as proof of ownership.
3. Prefer the standard workspace name when more than one candidate exists.
   Stop on unresolved ambiguity instead of guessing.
4. Start a stopped matching container without renaming or recreating it.
5. When no matching container exists:
   - run `scripts/docker_build_gb300.sh` from the current checkout to rebuild
     `trtmc-dev-gb300:latest`;
   - start a detached container named `trtmc-dev-gb300-agent-N` for an
     `agent-N` checkout, or `trtmc-dev-gb300` otherwise;
   - mount the checkout at `/workspace/tensorrt-model-connect`.
6. Run the requested command from the mounted repository root.

Existing containers are never renamed, removed, or migrated by this skill.

## Common Commands

```bash
python3 "$RUNNER" run -- cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
python3 "$RUNNER" run -- cmake --build build -j8
python3 "$RUNNER" run -- ctest --test-dir build --output-on-failure -R <regex>
python3 "$RUNNER" run -- python3 -m pytest tests/builder -q
python3 "$RUNNER" run -- ./scripts/validate_family.sh <model-id>
```

Use `resolve --pretty` for read-only discovery. Use `run --print-only -- ...`
to inspect the resulting `docker exec` command without creating or starting a
container.

## Overrides

- `TRTMC_CONTAINER_NAME`: require or create a specific container name.
- `TRTMC_CONTAINER_WORKDIR`: set the mount destination for a newly created
  container, or require that destination on a reused container.
- `TRTMC_CONTAINER_IMAGE`: use an already-built custom development image. The
  automatic image build applies to the standard image only.
- `TRTMC_STORAGE_ROOT`: override shared engine/cache storage.
- `TRTMC_HF_CACHE`: override the Hugging Face cache mounted into the container.

## Guardrails

- Do not use this skill outside TensorRT-Model-Connect.
- Do not choose a container solely because its name looks plausible.
- Do not reuse a container whose mount source does not resolve to the current
  checkout.
- Do not delete or rename an existing container to repair discovery.
- If multiple non-standard containers mount the same checkout, require
  `TRTMC_CONTAINER_NAME` instead of guessing.
- Treat image construction and container creation as environment preparation,
  not as evidence that a build, test, GPU workload, or model proof passed.
