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
For Dev qualification, `gpu_provider` can select AWS or Nebius explicitly.
Its default `auto` retains Brev's normal selection. The override is forwarded
only to Dev executions; Stable does not receive this experimental input.

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

A testcase may explicitly set `community_gpu: false` to keep a larger workload
in its existing Internal/Nightly qualification scope. Its `premerge` flag and
passing criteria remain unchanged. Community reports deferred cases by name and
does not claim to have qualified them. A fully deferred owner gets no checkpoint
staging or container. If every requested owner is fully deferred, normal CI runs
the existing five shared smoke families within the original requested-owner
budget; mixed selections run only their active owners. Missing premerge cases
and invalid flags remain errors. Dependency-image producers require actual
coverage of every selected owner, so shared smoke cannot qualify their images.

### Blocking GPU reservation

The workflow calls `python3 -m tools.brev_exec provision` once before admitting any
project build or model test. The entrypoint reserves a VM through the pinned Brev
CLI and waits for `RUNNING`, completed environment setup, and shell readiness.
It then checks SSH, the Docker daemon, host GPU visibility, available disk, and
GPU access from a digest-pinned CUDA probe container. Cloud-init state is
diagnostic information, not a requirement to rewrite or restart bootstrap.
Brev's create exit code alone is not a readiness guarantee.

CI requests Jupyter disabled because the job uses Docker rather than notebooks.
Transient inventory reads are retried on the same VM within the reservation
deadline. While Brev still reports initialization in progress, bounded read-only
diagnostics report cloud-init, service states, and fixed setup markers without
printing credentials, environment values, or complete bootstrap logs. These
diagnostics do not admit project or model execution.

The reservation waits up to 45 minutes on one named instance. The default is
64 GiB RAM on AWS `g6.4xlarge` or Nebius
`gpu-l40s-a.1gpu-16vcpu-64gb`. A qualified 128 GiB owner profile selects only
AWS `g6.8xlarge` or Nebius `gpu-l40s-a.1gpu-32vcpu-128gb`. All four types have
one GPU, request 500 GiB disk, and require 200 GiB free
at admission. Metadata READY is followed by the same functional probe; neither
a failed test nor a readiness timeout silently allocates a replacement.

Each failed probe records the CLI return code, whether stdout/stderr were
received, safe phase markers, and bounded-timeout partial evidence. Unrecognized
output remains unknown; diagnostic messages never relax the readiness receipt.

Every exit enters blocking deletion of the recorded allocation identity. Cleanup
is successful only after authenticated inventory confirms absence. A returned
DELETE response alone is insufficient. The independent cleanup job uses the same
lease and also waits for confirmed absence.

Once reservation succeeds, dependency setup and model validation use that VM.
The workflow invokes the family coordinator once and preserves its failure
result. Model failures cannot enter the provisioning retry loop. The readiness
gate checks the starting environment; family resource requirements and behavior
under load remain separate validation concerns.

### Sequential family containers

One GPU execution reserves one Brev VM and builds one base image. The coordinator
then starts a fresh container for each selected family, waits for it to finish,
removes it, and starts the next family. Shared smoke coverage remains BERT, GPT-2,
Qwen, timm ViT, and Whisper; directly changed or added families remain selected.
Failures are collected while the coordinator continues with the remaining
selected families. Checkpoint staging has a 15-minute per-family limit and the
family container has a 45-minute limit. The aggregate budget grows with selected
owners, up to three hours. Any families not admitted before that deadline remain
explicitly `not_run`, and the workflow fails with incomplete coverage. No testcase
selection or numerical passing criterion is reduced to fit this budget.

Dependency images are optional caches, admitted through family-owned
`families/<family>/ci/dependency-image.json` locks only after real qualification
and cleanup have passed. Promotion of a published candidate into a lock remains
manual. The trusted coordinator exports locks and recipe hashes from the chosen
CI commit's Git objects, never from the PR tree. It compares the base Dockerfile,
base requirements, and family requirements with the actual PR source. A changed
input uses the normal base image and ordinary family installation, so dependency
update PRs can still be tested before a new image is published. Invalid lock
metadata fails that owner explicitly while other owners continue.

Before reserving a VM, the trusted CI commit's selected owner locks determine
the host RAM profile. An omitted `resources.host_ram_gib` uses 64 GiB. The only
other supported value is 128, which requires the lock's existing qualification
proof and a matching `qualification_host` record: `ram_gib`, `gpu_count: 1`,
`arch: "x86_64"`, and the successful producer's decimal `run_id`. Admission of
that record waits for real workload qualification and confirmed cleanup. The
maximum profile among selected owners determines the one VM. Invalid resource
claims fail before allocation. A changed dependency input or image cache miss
retains the owner's host RAM profile; it never silently downgrades the host or
allocates a replacement. PR-owned locks and instance types do not select capacity.
This preallocation selector uses authorized requested owners before the PR's
execution plans are materialized. If a later plan defers an owner, its admitted
host profile may be retained conservatively.

Matching locks select private GHCR images by immutable digest. A read credential
is copied to the trusted VM only after its PR base image build; the coordinator
pulls all selected cached images using a temporary Docker configuration and
deletes both credential and configuration before any contributor container
starts. No registry token or Docker socket enters those containers. Native
builds, family installation, and every original E2E assertion still run. No lock
or image is admitted merely because the producer's local mechanics tests pass.

The image/setup/test step is capped at four hours inside the six-hour GPU job,
leaving at least an hour for release after the 45-minute reservation. Family
containers are restricted to three quarters of host RAM, no additional swap,
and all but two host CPUs. Docker OOM state is inspected before each container
is explicitly removed; it is reported as a resource failure rather than a
transport failure. Failed removal prevents admission of the next family.

Per-family results retain the failing stage, classification, duration, and every
selected E2E case's verdict. Missing, skipped, malformed, or incomplete results
cannot pass even when a container exits zero. The trusted host checks the exact
case inventory from manifests without importing contributor Python. After VM
cleanup, the workflow renders these results in the job summary and preserves the
raw durable-worker log and parsed summary as artifacts. Summary text is
diagnostic; command success and confirmed VM removal remain mandatory gates.

Each container owns its Python dependencies, native build, runtime directory,
temporary files, and checkpoint cache. The PR source and selected CI runner are
mounted read-only. Dependencies are installed before native configuration with
`--no-build-isolation`, so native package build hooks can use the image's PyTorch.
This pip option does not share environments between family containers. Repository
credentials and the Docker socket are not mounted into the test containers.

Develop and qualify GPU runner changes on `ci/developer`. A PR for this change
must target `ci/developer`, so merging it updates the implementation selected by
the Dev lane. After qualification, promote the reviewed Dev implementation to
`main` in a separate PR. Keep the Stable lane unchanged during development and
apply the main-only publisher guard from #1341 before promotion.

Container isolation addresses dependency and writable-state contamination. It
does not grant access to gated checkpoints, stage undeclared secondary assets,
repair family bundles or numerical mismatches, or establish the cause of a lost
Brev SSH connection. Those failures must remain visible during qualification.

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

## Candidate promotion compatibility

The Dev candidate includes the automatic Stable cutover. After its reviewed
promotion reaches `main`, the trusted metadata entry dispatches the complete
Stable CPU-to-GPU pipeline from `main`. The original PR run becomes a read-only
view of that pipeline's real CPU result, preserving the existing Internal CI
bridge without running the CPU suite twice. The trusted base's promotion marker
controls this transition, so a promotion PR still receives ordinary Stable CPU
tests before it is merged. Dev qualification from `ci/developer` continues to
reuse the current Stable PR run. The live cutover still requires post-merge proof.
