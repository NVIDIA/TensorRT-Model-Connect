# Cosmos 3 revival status

Last updated: 2026-08-19

## Objective

Revive the implementation from closed PR #763 on current `main`, add a one-click Docker Story Scene sample for Cosmos3-Nano generation, validate the combined change, and publish a draft pull request.

## Current state

- [x] Recovered PR #763 at `b40d8ba62e35bef3e095e09c31675ee5c9dd17e7`.
- [x] Preserved the unrelated dirty checkout by creating an isolated worktree.
- [x] Rebased the recovered Cosmos implementation onto current `github/main` with required DCO sign-off.
- [x] Resolved the current 84-family inventory without dropping newer model registrations.
- [x] Finished the Cosmos Story Scene backend, frontend, and Docker packaging.
- [x] Added an aarch64/GB10 image path and separated local CUDA rank from global collective rank.
- [x] Completed focused source, unit, container-configuration, and frontend checks.
- [x] Pushed the feature branch and opened draft PR #926 targeting `main`.
- [ ] Push the rebased head and obtain protected premerge CI on that exact SHA.
- [ ] Qualify CP1 on one DGX Spark and CP2 across two Sparks before claiming hardware support or speedup.

## Scope and constraints

- Native TRT-MC support is text-to-video only for `nvidia/Cosmos3-Nano`.
- The fixed model profile is BF16, 1280x720, 189 frames, 24 FPS, 35 steps, CFG 6, and flow shift 10.
- The app derives a 720x1280 social edit with FFmpeg; the model does not generate native portrait video in this integration.
- Model access remains gated. Credentials, checkpoints, bundles, generated frames, and videos are never committed or baked into the image.
- The native path has no integrated safety checker, so the sample is a developer preview rather than a production content-moderation solution.
- The built-in app launcher remains single-node. Dual-Spark CP requires one externally orchestrated rank per host, identical CP2 bundles, explicit `WORLD_SIZE`/`RANK`/`LOCAL_RANK` values, and a unique shared `TRTMC_NCCL_RENDEZVOUS` path.

## Validation log

- Focused Story Scene, CI-impact, 84-family inventory, and cancellation tests: `24 passed in 1.85s`.
- The CPU-only distributed-rank test compiled with `-Wall -Wextra -Werror` against CUDA 13 and passed.
- `docker compose config --quiet`, Bash syntax, JavaScript syntax, Python bytecode compilation, and `git diff --check` passed.
- The exact protected premerge suite has not run on the rebased head. The local account cannot access the Docker daemon, and the available host Python environment cannot collect the complete builder suite.
- Docker image build, TensorRT bundle compilation, CP1/CP2 generation, and performance measurement are pending. Docker access, gated-model credentials, and trusted peer launch access were not configured, so no Spark performance or speedup claim is made.

## Related pull requests

- Closed PR #763 is the recovered implementation source; draft PR #926 preserves that history while porting it onto current `main`.
- Open PR #211 overlaps parts of the Cosmos3 family and runtime surface. This work does not modify or close it; maintainers should choose the integration path.
- Closed PR #819 contains separate B200 optimization work and is not included in this revival.
