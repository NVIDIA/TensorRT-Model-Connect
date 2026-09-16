<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Community premerge and internal Nightly

The target boundary is public Community CI for all premerge validation, and
internal CI for Nightly. During qualification, the existing internal premerge
workflow and required status remain in place. This change does not remove or
rename that gate.

## Contributor flow

Opening or updating a pull request runs the existing public CPU checks. When
they pass, the trusted `community-premerge.yml` controller captures the current
PR merge from GitHub and starts stable Community CI. A repository-wide switch
optionally adds a dev run for every subsequent premerge. The author
does not need an internal CI label or maintainer dispatch to start these public
runs. Pushing another commit automatically starts a new attempt.

Each enabled lane reruns the CPU stages on the captured merge, then uses
disposable Brev instances to run the complete unit stage, build/install/load
the native wheel, and execute the selected family premerge cases. Qwen remains a baseline for
every premerge, including documentation changes. The protected base's existing
premerge cases cannot disappear from a selected family's test plan. Skipped
stages and incomplete GPU results fail the executor.

The CPU screening run's files and artifacts are never executed by the trusted
controller. It obtains the real merge from GitHub, verifies the current PR head,
and passes the same head/base/merge/tree snapshot to both lanes. Each Brev VM
fetches that exact merge object, not a later moving PR merge ref.

## CI selection and results

| Lane | Workflow implementation | Public status |
| --- | --- | --- |
| Stable | Source `main` | `TRTMC Community CI / Premerge` |
| Dev | `TRTMC_COMMUNITY_CI_DEV_REF`, default `main` | `TRTMC Community CI / Dev premerge (non-blocking)` |

In Source **Settings > Secrets and variables > Actions > Variables**, configure:

| Variable | Value | Effect |
| --- | --- | --- |
| `TRTMC_COMMUNITY_CI_DUAL_RUN` | Unset or `false` (default) | Stable only; no new dev job, dispatch, status, or Brev instance |
| `TRTMC_COMMUNITY_CI_DUAL_RUN` | `true` | Stable and dev for every new premerge controller run |
| `TRTMC_COMMUNITY_CI_DEV_REF` | A trusted CI branch, e.g. `ci/developer` | Selects the test implementation when dual running is enabled; defaults to `main` |

Only the exact value `true` enables dual running. Other values keep stable
running alone. Setting a dev ref does not enable dual running by itself, and
stable always uses `main`.

The switch is captured once per controller run, before its job matrix is
created. Turning it off makes subsequent controller runs stable-only. Controllers
that already selected their lanes retain that selection, including queued jobs;
running executors finish normally. No workflow edit or per-PR label is needed
to turn the comparison period on or off.

Leave the dev ref unset for main/main comparison. Stable and dev use
separate workflow concurrency groups, statuses, and GPU environments. Stable
publication does not wait for the dev result.

Stable uses the existing `gpu-ci-dispatch` environment. Configure
`gpu-ci-dev-dispatch` with its own `BREV_API_KEY`, approved CI branch policy, and
separate provider capacity for dev. Without that configuration, dev may fail
while stable continues. Using the same Brev quota for both can still create
resource contention despite separate workflow groups.

`TRTMC_COMMUNITY_GPU_TYPE` and `TRTMC_COMMUNITY_CUDA_ARCHITECTURES` default to
`L40` and `89`. Select matching hardware and build architecture when qualifying
families that require different capacity.

## Qualification and promotion

1. Merge the workflow change. With the switch unset, premerge runs stable only.
   Configure the dev GPU environment before starting a comparison period.
2. Set `TRTMC_COMMUNITY_CI_DEV_REF` to the CI development branch and set
   `TRTMC_COMMUNITY_CI_DUAL_RUN=true`. Leave the dev ref unset if first comparing
   main/main. All newly triggered premerges now run both implementations.
3. Collect paired results over the next one or two days of pull requests, including
   unit, package, GPU inference, failure reporting, and Brev cleanup evidence.
4. Promote the validated CI implementation through a reviewed Source pull
   request to `main`, then set `TRTMC_COMMUNITY_CI_DUAL_RUN=false` (or delete the
   variable). Future premerges use the promoted main implementation with no dev
   resource cost. The dev ref may stay configured while the switch is off.
   Roll back through a revert pull request. Keep dev outside the required checks.
5. Only after Community premerge is qualified should a separate reviewed
   rollout make its stable status required and retire internal premerge.

Public premerge never forwards Hub, Brev, or repository credentials into PR
code. A gated model that cannot be downloaded publicly fails visibly; it is not
skipped or counted as covered. Public model assets and suitable GPU capacity
must be qualified before internal premerge can be retired. The existing
maintainer-only manual smoke path keeps its original access behavior.

Internal stable/dev development is limited to Nightly and has its own branch
selector. It is independent of this public premerge rollout.
