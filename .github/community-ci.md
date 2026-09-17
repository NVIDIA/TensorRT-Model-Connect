<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Stable and Dev Community CI

## Current rollout

Stable keeps the existing Community CI behavior: every pull request runs the
four parallel CPU stages, and automatic GPU execution remains disabled.
Maintainers can still request the existing experimental GPU smoke test manually.
The Internal CI bridge, required status, and maintainer trigger are unchanged.

Dev is the experiment on the protected `ci/developer` branch. Its independent
pipeline runs the same CPU stages first, followed by the Brev GPU stage for
GPU-impacting changes. GPU failures fail Dev and do not affect Stable.

Both branches use one workflow file, `.github/workflows/community-ci.yml`, named
**Community CI**. There is no separate Community premerge workflow.

```text
Stable: source quality / docs / ownership / C++ and Python units → CPU result
Dev:    source quality / docs / ownership / C++ and Python units → Brev GPU → result
```

## Paired source snapshots

The existing `pull_request` run remains the Stable pipeline. Its event type,
run title, and `Community CPU / Required` job stay compatible with the unchanged
Internal CI bridge and notifications.

The same workflow also handles `pull_request_target` metadata. It locates the
existing Stable run for this exact PR head, validates the merge commit's two
parents, and captures its head/base/merge/tree identity. It never checks out or
executes contributor code. Stable is observed, not dispatched a second time.
If Dev is enabled, it receives that exact captured snapshot through a dispatch
of `community-ci.yml` from the selected CI branch. A moving PR merge ref cannot
silently change Dev's source. A newer PR head invalidates an older request.

The metadata jobs follow the exact Stable and Dev run IDs and publish their
complete workflow conclusions as `Stable Community CI` and `Dev Community CI`.
This includes future stages added on Dev. Each result is published independently;
Stable does not wait for Dev. Only these metadata jobs need dispatch permission.

## Switch and branch selection

Configure Source **Settings > Secrets and variables > Actions > Variables**:

| Variable | Value | Effect |
| --- | --- | --- |
| `TRTMC_COMMUNITY_CI_DUAL_RUN` | Unset or `false` | Stable only; no Dev dispatch or GPU allocation |
| `TRTMC_COMMUNITY_CI_DUAL_RUN` | Exactly `true` | Compare Stable with Dev on new requests |
| `TRTMC_COMMUNITY_CI_DEV_REF` | `ci/developer` | Select the protected experimental implementation |
| `TRTMC_COMMUNITY_CI_DEV_REF` | Unset | Default to `main` for an initial main/main comparison |

The switch is captured once per request. Turning it off stops Dev on subsequent
requests; work already started finishes normally. A Dev branch selection alone
does not enable automatic dual running. Dev must not be a required merge check.

For manual qualification, select **Actions > Community CI > Run workflow** and
enter a PR number. Selecting `ci/developer` runs Dev against the existing Stable
PR snapshot; with dual running enabled it also publishes the paired Stable result.
Selecting `main` manually starts Stable and, when enabled, Dev. Keep the internal
snapshot, lane, and request inputs at their defaults. Manual starts require
maintain or admin access. `run_gpu_smoke` retains the existing manual GPU opt-in.

## Dev GPU experiment

GPU policy is versioned in the workflow, independently of the dual-run switch.
The rollout PR leaves `COMMUNITY_GPU_EXECUTION_ENABLED` at `"false"` on Stable.
A separate commit on `ci/developer` enables GPU after CPU passes and keeps
credentials out of the isolated instance running contributor code. It also
carries the GPU runner/runtime changes needed by that experiment.

The Dev workflow loads its GPU orchestration and runner from the selected CI
commit while building and testing the immutable PR source. Family manifests and
test criteria remain owned by the source under test. The GPU runner selects
`premerge` cases for changed families; shared changes cover BERT, GPT-2, Qwen,
timm ViT, Whisper, and directly changed or added families. Docs-only changes
skip GPU. Missing public assets fail visibly rather than count as coverage.

The existing `gpu-ci-dispatch` environment permits `main` and protected
`ci/developer`. Only administrators may update the latter. Additional CI refs
need protection and environment approval. Dev uses a disposable Brev instance,
with cleanup in the job and an independent cleanup job. Per-PR and per-lane
concurrency keeps independent PRs parallel; shared provider quota can still
prevent allocation. Dev GPU type and CUDA architecture default to L40 and 89.

## Qualification and promotion

1. Merge the dual-run infrastructure while Stable retains its current behavior.
   Keep the automatic GPU experiment on `ci/developer`.
2. Enable the switch and select `ci/developer` for a one- or two-day comparison.
   Record paired source snapshots, both CI commits, CPU/GPU results, failure
   isolation, and VM cleanup. Turning the switch off ends the comparison.
3. After Dev passes, promote its qualified implementation through a separate
   reviewed PR to `main`. Validate automatic fork PR execution as part of that
   cutover, including continued compatibility with the retained Internal CI
   bridge. A successful manual Dev run alone does not prove the cutover.
4. Verify the promoted Stable pipeline on fresh PRs, then disable dual running
   to save resources. Roll back a promotion through a revert PR.

This infrastructure PR does not promote the GPU experiment, change required
checks, retire Internal premerge, or modify Nightly. Those are separate decisions.
Promotion and a multi-day comparison are unproven until their live runs complete.
