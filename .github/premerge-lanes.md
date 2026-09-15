<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Stable and development premerge CI

Every authorized premerge trigger starts two independent Actions paths:

| Path | Source workflow | CI implementation | Source status | Required |
| --- | --- | --- | --- | --- |
| Stable | `internal-ci-bridge.yml` | CI `main` | `TRTMC Internal CI / Automated premerge gate` | Yes |
| Dev | `internal-ci-dev.yml` | `TRTMC_PREMERGE_DEV_REF`, default `main` | `TRTMC Internal CI / Dev premerge (non-blocking)` | No |

The existing maintainer authorization and successful Community CPU check still
apply. The stable bridge captures one PR head, merge commit, first parent, and
tree. Both paths receive that exact snapshot. Neither path executes Source PR
code in the credentialed bridge.

The dev launcher records the snapshot in an immutable artifact on the stable
bridge run. Dev verifies the original workflow, run attempt, and maintainer
authorization before downloading it. Manual dev dispatch accepts only that
bridge run identity; it cannot supply a different PR or Source revision.

Stable publishes its result without waiting for dev. The dev workflow has its
own concurrency group, result, and status publisher. A rejected dev dispatch,
failed experiment, missing runner, or timeout cannot change the stable verdict.
The CI executor also separates stable and dev concurrency groups, so a dev run
cannot cancel a stable run through the normal premerge dispatch path.

## Choose the development implementation

In Source **Settings > Secrets and variables > Actions > Variables**, set
`TRTMC_PREMERGE_DEV_REF` to a trusted CI branch, for example `ci/developer`.
Leaving it unset, or setting it to `main`, runs the main implementation in both
paths. The ref must contain the lane-aware `premerge.yml` dispatch contract.
This setting selects the CI implementation; it does not change the Source PR
being tested. There is deliberately no variable to redirect stable away from
`main`.

The CI repository separately configures `TRTMC_PREMERGE_DEV_RUNNER_LABELS`.
Its default is `["self-hosted","trtmc-ci-dev"]`. Provision separate GPU capacity
with that label before expecting dev GPU results. With no matching runner, dev
waits while stable continues on its existing pool. Pointing this setting at the
stable pool preserves status isolation but introduces resource contention.

## Promotion and rollback

1. Start the CI development branch from CI `main`, including the dual-run
   dispatch contract. Set `TRTMC_PREMERGE_DEV_REF` to that branch.
2. Compare stable and dev results for the same captured Source snapshots.
   Record the exact CI workflow commit used by each private run. A queued,
   skipped, or failed dev run is not promotion evidence.
3. Open a CI pull request promoting the validated implementation to `main`.
   Review the exact candidate and pass its controller checks before merging.
   If the candidate changes, collect new dev evidence.
4. Merging that CI pull request is the switch: subsequent stable runs use the
   promoted `main`. Keep the existing required status and branch rules unchanged.
5. Reset `TRTMC_PREMERGE_DEV_REF` to `main` for the next baseline, then create
   another development branch when needed.

Before promotion, stop an experiment by resetting the dev ref to `main`.
After promotion, revert the promotion through a CI pull request and rerun
premerge on the current Source head. Existing runs retain their workflow
revision; changing a branch or variable does not rewrite past results.

## Initial rollout

Merge the lane-aware CI executor first, then this Source bridge change. The
executor keeps the old dispatch inputs working during this transition. Keep
the dev ref unset for the initial main/main comparison and verify that only
the existing stable status is required. Enable development refs after that
baseline succeeds. Nightly is outside this change.
