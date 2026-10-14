<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Stable and Dev Community CI

## Stable premerge

Every pull request targeting `main` automatically runs the promoted Community
pipeline from trusted `main`: source quality, documentation, ownership and
impact, and C++/Python unit tests; then plan admission, GPU execution, and
confirmed VM cleanup. `Stable Community GPU CI` is the aggregate automated merge
gate. A CPU-only result, successful dispatch, or `Dev Community CI` result cannot
replace it. Documentation-only changes need no GPU reservation and are reported
as such by the same aggregate.

The new status name is intentional: historical `Stable Community CI` statuses
may certify only CPU checks. They cannot satisfy the promoted GPU gate. Existing
PR heads need a fresh trusted Stable execution that publishes
`Stable Community GPU CI`; until then, the new required context remains missing
and blocks merging.

```text
PR snapshot → CPU checks → plan admission → VM readiness
            → shared base + selected family layers
            → sequential family containers → confirmed cleanup → Stable result
```

A successful result on the current PR head makes the PR ready for review by an
NVIDIA employee maintainer. It does not approve or merge the PR. Internal
premerge is no longer part of the normal contribution flow; private Nightly and
separately authorized diagnostics remain independent. Repository settings must
also replace the former Internal status requirement. Follow the
[cutover runbook](community-ci-cutover.md); merging workflow files alone cannot
change GitHub rulesets.

## Frozen source and status ownership

The trusted `pull_request_target` entry captures the PR head, base, merge and
tree identities and dispatches execution from trusted `main`. It does not execute
contributor code. The execution tests that immutable source snapshot with the
trusted orchestration implementation. A moving PR merge ref cannot silently
change its source, and a newer head invalidates an older result.

The original `pull_request` run provides a read-only view of the trusted
execution's real `Community CPU / Required` result without running the CPU suite
twice. Dispatched executions publish their own pending and final lane statuses.
Final GPU publication waits for confirmed cleanup. Observer timeouts, API errors
and delayed dispatch responses cannot replace the execution's classified result.
Stable execution and publication remain restricted to trusted `main`.

## Dev qualification

Automatic PR events run Stable only, including when the legacy
`TRTMC_COMMUNITY_CI_DUAL_RUN` variable remains set. Clear or disable that variable
when applying the cutover settings. Dev is optional manual qualification on the
protected `ci/developer` branch; it is not a required merge check and cannot
publish Stable's result.

To qualify a Dev change against an open PR, replace `PR_NUMBER` below:

```bash
gh workflow run community-ci.yml --repo NVIDIA/TensorRT-Model-Connect \
  --ref ci/developer -f ci_lane=dev -f pr_number=PR_NUMBER
```

Keep internal snapshot and request inputs empty. Manual starts require maintain
or admin access. This explicit Dev entry selects `ci/developer`, runs only Dev,
and accepts `gpu_provider=aws` or `gpu_provider=nebius` for provider qualification.
The default is AWS. The supported execution refs are `main` and protected
`ci/developer`; arbitrary branches and tags cannot run the credentialed owner.
Future changes can be qualified on Dev before a reviewed promotion to `main`.

## GPU selection and plan admission

CPU checks must pass before GPU plan admission. Trusted CI reads the frozen PR's
manifests without importing PR Python. Changed or added families select their
own `premerge` cases. Shared changes also select the existing five smoke families:
BERT, GPT-2, Qwen, timm ViT and Whisper. Documentation-only changes skip allocation.
Missing public assets fail visibly rather than count as coverage.

Ordinary PRs may directly change or add at most five model families. A larger
plan is rejected before allocation with instructions to split the change into
smaller PRs. The five shared smoke families do not consume this direct-family
limit, and directly changed families are never silently dropped. A change that
must span more families needs a separately reviewed execution approach; it
cannot bypass coverage by relabeling itself or omitting tests.

A testcase may explicitly set `community_gpu: false` to keep a larger workload
in its existing Internal/Nightly qualification scope. Its `premerge` flag and
passing criteria remain unchanged. Community reports deferred cases by name and
does not claim to have qualified them. A fully deferred owner gets no checkpoint
staging or container. If every requested owner is fully deferred, normal CI runs
the existing five shared smoke families with a budget for that effective plan;
mixed selections run only their active owners. Missing premerge cases
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

An explicit create HTTP 429 is retried at most three times in total, with a
60–90 second delay before each retry. The owner queries the original organization
and instance name first and reuses any recovered instance ID. Three rejected
attempts produce an infrastructure failure. Transport errors and unknown create
outcomes are not blindly retried. Cleanup reconciles every discovered owned ID,
including late or duplicate instances; a failed query or delete is never treated
as proof of release.

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

After CPU passes, trusted CI reads the frozen PR's declarative test manifests and
checks its test plan before reserving a VM. It does not import PR Python during
this preparation. The VM verifies the same frozen plan before execution.

A selected manifest declaring nonempty `external_files` must include a regular
`families/<family>/ci/Dockerfile.dependencies` in that same frozen PR source.
The family recipe prepares and validates those assets and supplies any directory
variables its tests require. Missing preparation recipes fail before allocation;
they do not defer cases or count as a successful test. Recipe admission alone
does not prove that preparation or model validation will succeed.

One GPU execution reserves one Brev VM and pulls one verified private shared
base by immutable digest. The base identity depends only on its common Dockerfile
and environment inputs, so a family dependency change does not rebuild the base.
For each selected family, the coordinator prepares its declared dependency layer
on that base, stages its checkpoints, then starts a fresh test container. It waits
for the container to finish and confirms removal before starting the next family.
Shared smoke coverage remains BERT, GPT-2,
Qwen, timm ViT, and Whisper; directly changed or added families remain selected.
Failures are collected while the coordinator continues with the remaining
selected families. Dependency preparation has a two-hour per-family limit,
separate from the test entry point. Checkpoint staging has a 15-minute limit and the
family container has a 45-minute limit. The aggregate budget grows with selected
owners, up to three hours. Any families not admitted before that deadline remain
explicitly `not_run`, and the workflow fails with incomplete coverage. No testcase
selection or numerical passing criterion is reduced to fit this budget.

The ordinary GPU path requires a verified shared-base entry in the protected
`gpu-ci-dispatch` environment's `TRTMC_COMMUNITY_DEPENDENCY_CATALOG` JSON secret.
Its shape is `schema_version: 1`, `registry_prefix`, `base`, and `families`.
The `base` record contains the immutable image identity and verified common
input hashes. Trusted CI compares those hashes with its own Git objects.
Missing or invalid base metadata, a missing reader token, or failure to resolve
the token owner through GitHub fails before allocation. Only trusted preparation
uses registry credentials to resolve the reader and pull the base; it erases them
before any family Dockerfile or container executes. No private coordinates belong
in public source, logs, or PR text.

Each family owns `requirements.txt` and may provide `ci/Dockerfile.dependencies`
with its additional public dependency inputs. Only selected owners prepare a
layer; the preparation context excludes project source, other families and
credentials. A family layer must extend the pulled common base. The public
Dockerfile and family requirements remain available for local reproduction
without access to the private registry.

A family may also declare `tests/reference-source.json` with exactly
`repository` (a public GitHub `owner/repo`) and `revision` (a full lowercase
40-character commit SHA). During dependency preparation, CI fetches that exact
commit inside the credential-free family build and sets
`TRTMC_REFERENCE_SOURCE_DIR=/opt/trtmc-reference`. The declaration and generated
recipe participate in the family layer identity. CI verifies the resulting
checkout and its receipt in a network-disabled container before test entry.
This declaration also works for families without extra Python requirements.
It does not install upstream requirements or change the shared interpreter:
any additional packages or independent reference interpreter belong in the
family's `ci/Dockerfile.dependencies`. Source preparation failures are environment
failures, and the PR entry point has not run in that case.

Previously qualified family image locks remain supported by the separate image
qualification path. Their admission still requires actual qualification and
cleanup; an ordinary dynamically prepared layer does not receive that claim.

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

Matching locks select private GHCR images by immutable digest. The separate
`TRTMC_COMMUNITY_REGISTRY_READ_TOKEN` secret provides read access. Use a classic
personal access token with `read:packages` whose account can read the private
package. Trusted preparation resolves its GitHub login through `GET /user`; no
separate username secret or additional `read:user` scope is required.
The normal GPU job has no package permission and never substitutes its public
repository `GITHUB_TOKEN`. Do not grant the public repository access to the
private package. A read credential
is copied to the trusted VM only after its PR base image build; the coordinator
pulls all selected cached images using a temporary Docker configuration and
deletes both credential and configuration before any contributor container
starts. No registry token or Docker socket enters those containers. Native
builds, family installation, and every original E2E assertion still run. No lock
or image is admitted merely because the producer's local mechanics tests pass.

`task=dependency-image` now qualifies an already private candidate; the public
Source workflow never builds or pushes that candidate and has no package write
permission. Reuse `audit_family` and `audit_digest` for this protected manual
task. Missing private registry configuration or a dedicated read credential
fails before a VM is reserved. Publication belongs to the private CPU controller
using the one public family recipe and locked dependency inputs.
The reusable-workflow caller passes only the named credentials; the called
`gpu-ci-dispatch` job resolves its protected environment values. The public
repository token is never a substitute for the separate private reader.

The qualifier verifies authenticated private visibility, pulls only the supplied
digest on the trusted host, and erases registry auth before any candidate
container starts. Its embedded public `build-inputs.json` must describe a locked
environment source commit in protected `main` or `ci/developer` history. All 11
input hashes are checked against that commit's Git objects, and family
requirements must also match the frozen protected MAIN used for the model tests.
`environment_source_sha` and `model_source_sha` are separate evidence fields.
The unchanged import/ABI probe, native BYOK test, and original selected family
E2Es then run against the actual pulled image. A fresh complete case receipt is
required; baseline substitution cannot qualify an owner image.

The owner exports only public source/input hashes, digest, ABI and actual case
outcomes. Its pending proof explicitly has `cleanup_confirmed: false` and
`admitted: false`. A final GPU proof is uploaded only after the owner and
independent backstop both confirm cleanup; a backstop fallback noop cannot
produce that proof. Candidate admission into the protected catalog remains a
separate manual review, including the matching positive/negative access checks.
Public result provenance contains only public source and input hashes, including
optional dependency helper, constraints, and resolved environment lock hashes.
When present, these additional inputs must match the PR or require a cold install.
Private registry access limits direct registry downloads. Authorized contributor
containers can read their image files and have network access, so dependency
layers must contain only public dependencies and build recipes, with no project
source, model weights, or secrets.

Maintainers can dispatch `task=dependency-image-access-check` with the existing
`audit_family` and immutable `audit_digest` inputs. Its registry prefix comes
only from the protected `TRTMC_COMMUNITY_REGISTRY` secret. The separate hosted
job uses anonymous access and the public repository token with `packages: read`,
without the private reader secret. Both must receive HTTP 401, 403, or 404 from
the registry token exchange or manifest check. Timeouts, malformed responses,
redirects, and server errors remain unknown and fail the diagnostic. The safe
receipt records only family, digest and access statuses; it neither admits an
image nor replaces the separate positive private-reader check of the same family
and digest. A 404 can also mean a missing image; denial evidence requires that
matching positive check before claiming the private package's access controls.

Maintainers can dispatch `task=dependency-image-audit` on a protected branch with
`audit_family` and an immutable `audit_digest`. This mode uses only package read
permission and records the package visibility, identity, linked repository, root
JSON keys, and matching version metadata. It rechecks the triggering actor before
the requests and preserves `dependency-image-audit.json` even when visibility is
not private. It allocates no VM and does not publish or admit an image; the
producer's private visibility requirement remains unchanged.

The image/setup/test step is capped at four hours inside the six-hour GPU job,
leaving at least an hour for release after the 45-minute reservation. Family
containers are restricted to three quarters of host RAM, no additional swap,
and all but two host CPUs. Docker OOM state is inspected before each container
is explicitly removed and retained as evidence. Failed removal prevents
admission of the next family.

Per-family results retain the failing stage, classification, duration, and every
selected E2E case's verdict. Missing, skipped, malformed, or incomplete results
cannot pass even when a container exits zero. The trusted host checks the exact
case inventory from manifests without importing contributor Python. After VM
cleanup, the workflow renders these results in the job summary and preserves the
raw durable-worker log and parsed summary as artifacts. Summary text is
diagnostic; command success and confirmed VM removal remain mandatory gates.

The responsibility boundary is the frozen PR's entry point. Preparation errors
are `infra_failure`. After environment and checkpoint checks, the wrapper records
entry and runs the PR's native build and `python3 -m tools.ci pipeline selective-e2e`.
Failures after entry default to `pr_failure`, including OOM for now; the receipt
retains the evidence needed for a later, more detailed diagnosis. Cleanup is an
independent infrastructure obligation and must still succeed. A proven skipped
reservation needs no Brev operation, while an uncertain allocation cannot be
reported as released.

Each container owns its Python dependencies, native build, runtime directory,
temporary files, and checkpoint cache. The PR source and selected CI runner are
mounted read-only. Dependencies are installed in the selected family's image
layer before test entry, with `--no-build-isolation` so native package build hooks
can use the image's PyTorch. Host checkpoint staging uses the common lock's exact
Hugging Face client version and prepares snapshot metadata as well as file bytes,
so offline consumers do not need reference metadata requests. Repository
credentials and the Docker socket are not mounted into the test containers.

Develop and qualify GPU runner changes on `ci/developer`. A PR for this change
must target `ci/developer`, so merging it updates the implementation selected by
the Dev lane. After qualification, promote the reviewed Dev implementation to
`main` in a separate PR. Keep the active Stable lane unchanged during development.
The integration must retain main-only Stable coordination, the explicit Dev-only
manual entry, approved implementation refs, and execution-owned publication.

Container isolation addresses dependency and writable-state contamination. It
does not grant access to gated checkpoints, stage undeclared secondary assets,
repair family bundles or numerical mismatches, or establish the cause of a lost
Brev SSH connection. Those failures must remain visible during qualification.

The existing `gpu-ci-dispatch` environment permits `main` and protected
`ci/developer`. Only administrators may update the latter. Additional CI refs
need explicit allowlist changes, protection, and environment approval. Each GPU
execution uses a disposable Brev instance, with owner and independent backstop
cleanup. Per-PR and per-lane concurrency keeps independent PRs parallel; shared
provider quota can still prevent allocation. The default profile is one AWS L4;
manual Dev qualification can select the configured Nebius L40S profile.

## Rollout and future changes

This promotion changes the ordinary premerge contract to `Stable Community GPU CI`.
Apply and verify the [settings migration](community-ci-cutover.md) as a separate
administrator action. Keep the status requirement in place throughout the
transition; do not temporarily remove all required CI.

The promotion PR itself still uses the trusted workflow and repository rules
already deployed on `main`. A candidate or manual Dev pass does not prove that
the new automatic Stable entry has run. After promotion, verify a fresh PR,
including a fork PR, through CPU admission, GPU execution, cleanup and final
current-head status. Verify failure classification with existing or controlled
failure cases as well. Only claim those observations after the runs complete.

Keep future runner changes on `ci/developer` until qualified, then promote them
through a reviewed PR. The integration must retain trusted Stable coordination,
manual Dev isolation, frozen source, execution-owned publication and confirmed
resource release. Private Nightly is not changed by this premerge promotion.
