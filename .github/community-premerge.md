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
PR merge from GitHub and starts stable and dev Community CI runs. The author
does not need an internal CI label or maintainer dispatch to start these public
runs. Pushing another commit automatically starts a new attempt.

Both lanes rerun the CPU stages on the captured merge, then use disposable Brev
instances to run the complete unit stage, build/install/load the native wheel,
and execute the selected family premerge cases. Qwen remains a baseline for
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

Set `TRTMC_COMMUNITY_CI_DEV_REF` in Source **Settings > Secrets and variables >
Actions > Variables** to a trusted development branch containing the new
dispatch contract. Leave it unset for main/main comparison. Stable and dev use
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

1. Merge the workflow change and configure the dev GPU environment. Both CI
   implementations initially come from `main`.
2. Collect paired results on representative Source pull requests, including
   unit, package, GPU inference, failure reporting, and Brev cleanup evidence.
3. Develop CI settings on a separate Source branch selected by the dev ref.
   Promote a validated implementation through a reviewed Source pull request
   to `main`; reset the dev ref to `main` afterward. Roll back through a revert
   pull request. Keep dev outside the required checks.
4. Only after Community premerge is qualified should a separate reviewed
   rollout make its stable status required and retire internal premerge.

Public premerge never forwards Hub, Brev, or repository credentials into PR
code. A gated model that cannot be downloaded publicly fails visibly; it is not
skipped or counted as covered. Public model assets and suitable GPU capacity
must be qualified before internal premerge can be retired. The existing
maintainer-only manual smoke path keeps its original access behavior.

Internal stable/dev development is limited to Nightly and has its own branch
selector. It is independent of this public premerge rollout.
