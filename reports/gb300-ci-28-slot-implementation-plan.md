# GB300 CI 28-Slot Implementation Plan

Status: implementation in progress; acceptance evidence is not yet complete

Plan owner: TensorRT-Model-Connect CI

Target repository: NVIDIA/TensorRT-Model-Connect on GitHub

Target branch: <code>main</code>

Baseline audited: 2026-07-21 PDT

Baseline GitHub revision: <code>2b08f200c5602a369ac592bdeb3960bf4e5e5ce2</code>

This is the execution and acceptance plan for making 28 shared GB300 GPU slots
directly available to normal Pre-Merge and Nightly CI. It is intentionally a
plan, not a record that the rollout has already happened. Checkboxes should be
updated only when the linked evidence exists.

## 1. Goal and Definition of Success

At completion, the common production runner pool will expose exactly 28 online
runner listeners and exactly 28 node-local shared GPU slots:

| Node | Production runner listeners | Allowed GPUs | Slots per GPU | Shared slots |
| --- | ---: | --- | ---: | ---: |
| <code>gb300-nvl-019-compute01</code> | 16 | 0, 1, 2, 3 | 4 | 16 |
| <code>gb300-nvl-019-compute02</code> | 12 | 1, 2, 3 | 4 | 12 |
| Total | 28 | 7 GB300 GPUs | 4 | 28 |

The existing generic runner at
<code>/workspace/users/yifeif/model-connect-runner</code> on compute01 remains
outside this count. It may remain online only if it is technically CPU-only or
lease-aware. It must not receive the production proof label; otherwise it must
be stopped/reconfigured with the other conflicting schedulers before durable
28-slot admission.

“28 slots are available” means:

1. A Pre-Merge or Nightly model matrix has no fixed concurrency ceiling below
   fleet capacity.
2. GitHub may dispatch as many as 28 matching shared jobs at once.
3. Each dispatched job acquires one unique node-local GPU slot before using a
   GPU.
4. All 28 unique slots can be held concurrently in a hardware canary.
5. A 29th shared job cannot start GPU work until one of the first 28 releases
   its slot.
6. Every Nightly warms the full required Hugging Face cache once on every
   admitted GPU node before Nightly model proofs begin.
7. Pre-Merge and model-proof jobs never repair a cache by downloading from
   Hugging Face.
8. No scheduler or container outside the same <code>GpuLease</code> namespace
   can claim any of the seven CI GPUs while the 28 production listeners are
   admitted.

If a matrix contains fewer than 28 entries, it naturally uses fewer than 28
slots. If another trusted CI job is already consuming a matching runner, jobs
queue until capacity is free. The promise is “up to the full admitted fleet,”
not “manufacture 28 jobs regardless of workload.”

## 2. Complete Mental Model

### 2.1 Shared model-proof scheduling

~~~text
Pre-Merge or Nightly model matrix
                |
                v
GitHub Actions scheduler matches the common label
                |
                v
One idle generic proof-runner listener on one node
                |
                v
The job validates that node's host-local policy
                |
                v
GpuLease acquires (node, GPU, slot) through node-shared flock files
                |
                v
Docker is bound to that GPU index and model proof runs
                |
                v
Process exit releases the flock automatically
~~~

GitHub does not understand GPU slots. It only sees runner listeners and labels.
The repository's <code>GpuLease</code> allocator understands the allowed GPUs
and slots on the node where a job landed.

The listeners are generic at the node level. A runner named
<code>proof-07</code> is not permanently tied to GPU 1 slot 3. Any listener on
that node can acquire any currently free slot allowed by that node's policy.

For production, listener count and shared-slot count are deliberately kept
equal. This gives GitHub an honest view of usable concurrency:

~~~text
production listeners on a node
    = allowed GPU count × slots per GPU
~~~

The lock is the safety boundary. A listener is the GitHub scheduling boundary.
Neither one alone proves usable capacity.

### 2.2 Nightly cache scheduling

~~~text
Nightly legal + model inventory
                |
                v
Read the checked-in GB300 pool topology
                |
                v
Build one matrix row for every declared node label
                |
                v
GitHub schedules that row on any listener carrying that node label
                |
                v
Each node performs an online strict warm, then a network-disabled strict
local-only verification, and emits a receipt
                |
                v
All node receipts pass
                |
                v
Nightly model-proof matrix may start on the common 28-runner pool
~~~

No model job decides which node to warm. The single declarative source of truth
is <code>.github/ci/gb300-pool-topology.json</code>; the workflow contains no
separate hostname list. A declared node produces exactly one warm job, even
though every listener on that node may carry the node label. The warm job uses
only <code>runs-on: ${{ matrix.node_label }}</code>: it does not require
<code>self-hosted</code>, the common production label, or a fixed cache-anchor
runner. GitHub may choose any online listener on that node, and all listeners
on the node share the same host cache and cache lock.

Adding a physical node is intentionally semi-automatic: bootstrap its
node-labelled listeners outside the production pool, merge one topology-data
PR, warm and verify the declared node, and only then add the common production
label. Restarting or replacing a listener on an already-declared node does not
require a repository change.

### 2.3 What “slot” does and does not mean

A shared slot is a cooperative concurrency token. Four slots per GPU allow up
to four model proofs to share one physical GPU. This is not MIG, a VRAM quota,
or a hard performance partition. A single workload can still allocate too much
memory and cause contention. The 4-per-GPU policy therefore remains subject to
real workload canaries and OOM monitoring.

An <code>exclusive_gpu</code> request acquires all four slots on one GPU. The
local allocator guarantees safety and fairness on that node. It is not a
fleet-global packing scheduler: seven arbitrary exclusive jobs can be placed
unevenly by GitHub and are not guaranteed to immediately occupy all seven
physical GPUs.

<code>GpuLease</code> coordinates only cooperative participants. A GitLab
runner or Docker container that can launch CUDA work without taking the same
lock can collide with CI even when no process was visible during admission.
Therefore a temporary drain is enough to run a canary, but not enough to claim
durable 28-slot capacity.

For the initial rollout, the KISS final state dedicates compute01 GPUs 0
through 3 and compute02 GPUs 1 through 3 to the GitHub proof pool. Conflicting
external schedulers must remain disabled, be reconfigured away from those
GPUs, or be made to acquire the same exclusive lease before the 28 labels can
remain admitted. This includes the compute01 GitLab scheduler, the generic
compute01 GitHub runner if it can launch GPU work, and the four compute02
overlap containers. If owners restore an uncoordinated workload, first remove
the affected GitHub production labels and treat capacity as rolled back.

### 2.4 Trust boundary

All proof listeners on a node intentionally share the <code>yifeif</code>
account, lock files, cache, and Docker daemon. The <code>run-ci</code> label is
therefore an admission decision into a cooperative trusted queue, not an
adversarial tenant boundary. Only maintainers may admit same-repository test
PRs, and they must review workflow and executable changes before labeling.

The implementation injects the HF token only into one bounded, foreground
default-branch cache-warm container. Package, VLM assessment, Pre-Merge, model
proof, and canary containers receive no token; cache consumers run local-only
where model access is required. The Nightly VLM judge uses its model-owned
configuration rather than an external model-ID override, so the same dependency
is part of the per-node warm plan. The temporary read-only token mount keeps the
raw token out of Docker subprocess and container environments and command-line
arguments; normal exit, failure, and cancellation remove the exact container
before deleting the token file. A process that already controls the same host
account or Docker daemon is inside the trusted boundary and could inspect
another container. Hostile-PR isolation would require separate Unix users or
machines and is outside this shared-runner design.

## 3. KISS Design Decisions

This plan uses the smallest architecture that satisfies the requirements:

- One common production label: <code>trtmc-gb300-proof</code>.
- One unique identity label per node: <code>trtmc-node-*</code>.
- One declarative topology file:
  <code>.github/ci/gb300-pool-topology.json</code>.
- One Nightly warm matrix row per declared node.
- The existing GitHub scheduler selects a runner.
- The existing node-local <code>GpuLease</code> selects a GPU slot.
- The existing strict Hugging Face warm script remains the downloader and
  validator.
- The existing topology validator derives both cache-warm rows and capacity
  expectations from the same file.
- One node-wide file lock coordinates cache writers and readers.

The design intentionally does not add:

- a central GPU scheduler;
- a database or queue service;
- a dedicated cache daemon;
- a hostname conditional in YAML;
- a runner-inventory API call, GitHub App, or discovery credential;
- a fixed cache-anchor runner or cache-anchor label;
- a per-model cache-warm job;
- a recurring rsync between nodes;
- a literal fleet size of 28 in production workflows;
- a dedicated 29th or controller runner;
- runner deletion during rollout.

The number 28 appears in this rollout plan and acceptance canary because it is
the current acceptance target. Normal Pre-Merge and Nightly model-proof
workflows derive usable concurrency from online matching production runners
and contain no hardcoded 28. The topology file declaratively records the
physical nodes targeted for warm and admission so Nightly can warm each node
and canaries can verify the exact node/GPU split. During a controlled expansion
it may temporarily include a staged node that is not yet in the production
runner pool; it is data, not a live runner inventory or scheduler.

## 4. Historical Audited Starting Point

The following was observed on 2026-07-21 PDT before host and label staging. It
is retained as historical design input, not as a baseline that can be
reconstructed retroactively. The current sanitized state is recorded in
<code>reports/gb300-ci-rollout/2026-07-22-pre-merge-current-state.md</code> and
must be refreshed immediately before each mutating rollout step because runner
state is live. The prior production baseline is 12 listeners on compute02,
with 16 online compute01 listeners still excluded by their missing production
label. Those 12 listeners are a safe rollback target only while overlapping
external compute02 workloads remain stopped, reconfigured, or lease-aware.

### 4.1 GitHub runner registry

- 33 repository runners were registered.
- <code>gb300-nvl-019-compute01</code> was online and idle as the existing
  generic runner with only default labels.
- <code>gb300-nvl-019-compute01-proof-00</code> through
  <code>compute01-proof-15</code> were registered with the common proof label
  but all offline.
- <code>gb300-nvl-019-compute02-proof-00</code> through
  <code>compute02-proof-15</code> were online with the common proof label.
- compute02 proof runners 08 through 15 also had the legacy
  <code>trtmc-gb300-proof-compute02</code> label.
- All 32 proof registrations were created without GitHub's automatic
  <code>self-hosted</code>, OS, or architecture labels. Their scheduling
  contract is therefore intentionally based only on explicit custom labels.

At that point, the GitHub-visible pool was not the desired topology:
compute01 contributes zero active proof listeners, while compute02 advertises
16 listeners for only 12 allowed shared slots.

### 4.2 Physical hosts

Both nodes expose four NVIDIA GB300 GPUs, indices 0 through 3, each reporting
284208 MiB. Current proof policy reserves compute02 GPU 0 and allows only
compute02 GPUs 1, 2, and 3. The 28-slot target therefore uses seven physical
GPUs.

Both nodes already contain:

- proof runner directories
  <code>~/.local/share/trtmc-actions-runners/proof-00</code> through
  <code>proof-15</code>;
- user unit template
  <code>~/.config/systemd/user/trtmc-github-proof@.service</code>;
- <code>Linger=yes</code> for user <code>yifeif</code>.

The compute02 proof units 00 through 15 were enabled and active. The compute01
proof units were disabled and inactive. The current unit template has no
<code>EnvironmentFile</code>, so runner processes do not yet inherit
host-specific GPU policy.

### 4.3 Current repository behavior

- Both Pre-Merge and Nightly model matrices currently contain
  <code>max-parallel: 16</code>.
- Both call the same reusable <code>model-proof.yml</code>.
- The reusable workflow injects GPU IDs from the repository variable
  <code>TRTMC_MODEL_PROOF_GPU_IDS=1,2,3</code>. That global value is incorrect
  for compute01, which should use GPUs 0 through 3.
- <code>TRTMC_MODEL_PROOF_SLOTS_PER_GPU=4</code>.
- The shared GPU lock directory is
  <code>/workspace/users/yifeif/.cache/trtmc-ci/locks/gpu</code>.
- The Hugging Face cache is
  <code>/workspace/users/yifeif/.cache/huggingface</code>.
- The model-reference cache root is
  <code>/workspace/users/yifeif/.cache/trtmc-ci/model-references</code>.
- Model proof already performs an offline cache check before acquiring a GPU
  lease and runs the proof container with a selected GPU.
- Nightly already has a strict full-cache warm command, but it currently lands
  on only one arbitrary matching runner.

### 4.4 Historical rsync observation

A one-time compute02-to-compute01 rsync was active at audit time. It was copying
the approximately 1 TB Hugging Face cache and reporting permission errors for
some root-owned or unreadable <code>trees/*.json</code> files.

The cache is large because it contains weights and assets for the full Nightly
model inventory, not just one model. Manual transfer is not required for
correctness: the first strict warm on a cold node can download and cache the
same dependencies. Rsync is only a bootstrap optimization to avoid a very long
first download and unnecessary Hub traffic.

The rsync is not a readiness signal. It may complete with exit code 23 while
leaving a mostly useful cache. The node becomes ready only after the repository
warm script and a second strict local-only verification succeed.

The 2026-07-22 current-state refresh found no rsync process and found the
representative compute01 cache set complete. Do not restart the copy or perform
a broad redownload; Nightly's strict warm plus offline verification remains the
only readiness authority.

## 5. Target Runner and Label Contract

### 5.1 Labels

Every production proof runner has:

- common pool label: <code>trtmc-gb300-proof</code>;
- exactly one node label.

Do not assume that <code>self-hosted</code>, <code>Linux</code>, or
<code>ARM64</code> is present. The current proof registrations were created
with default labels disabled. Normal proof jobs use only the common pool
label; each cache-warm matrix row uses only its declared node label.

Node labels are:

- <code>trtmc-node-gb300-nvl-019-compute01</code>;
- <code>trtmc-node-gb300-nvl-019-compute02</code>.

Every proof listener on a node may carry that node label. A Nightly creates
only one matrix row for the label, so GitHub selects any one online listener
on the node to perform the node-wide warm. There is no
<code>trtmc-cache-anchor</code> label, no designated <code>proof-00</code>
runner, and no runner name in the routing contract.

The earlier fixed-anchor proposal is superseded by this declarative design.
A stale <code>trtmc-cache-anchor</code> label from pre-staging is inert because
no workflow selects it. Its eventual cleanup is optional and is not an
admission gate or a reason to mutate live runners during this rollout.

The four compute02 standby registrations, proof-12 through proof-15, must:

- be stopped and disabled;
- not have <code>trtmc-gb300-proof</code>;
- have a clear non-production label such as
  <code>trtmc-proof-standby</code>;
- retain their runner registrations and directories for rollback.

Merely leaving excess runners offline while retaining the production label is
not sufficient: an accidental service start would advertise capacity that the
node does not have.

Remove the legacy compute02-specific proof label after the new node-label
contract is verified.

### 5.2 Host-local runner environment

Create this non-secret file on each node:

<code>/workspace/users/yifeif/.config/trtmc/proof-node.env</code>

compute01:

~~~ini
TRTMC_NODE_ID=gb300-nvl-019-compute01
TRTMC_MODEL_PROOF_GPU_IDS=0,1,2,3
TRTMC_MODEL_PROOF_SLOTS_PER_GPU=4
TRTMC_MODEL_PROOF_GPU_LOCK_DIR=/workspace/users/yifeif/.cache/trtmc-ci/locks/gpu
TRTMC_HF_CACHE_LOCK_FILE=/workspace/users/yifeif/.cache/trtmc-ci/locks/hf-cache.lock
TRTMC_HF_CACHE=/workspace/users/yifeif/.cache/huggingface
TRTMC_HF_HUB_CACHE=/workspace/users/yifeif/.cache/huggingface/hub
HF_HOME=/workspace/users/yifeif/.cache/huggingface
HF_HUB_CACHE=/workspace/users/yifeif/.cache/huggingface/hub
HUGGINGFACE_HUB_CACHE=/workspace/users/yifeif/.cache/huggingface/hub
HF_MODULES_CACHE=/workspace/users/yifeif/.cache/huggingface/modules
TRTMC_MODEL_REFERENCE_CACHE_ROOT=/workspace/users/yifeif/.cache/trtmc-ci/model-references
~~~

compute02 is identical except:

~~~ini
TRTMC_NODE_ID=gb300-nvl-019-compute02
TRTMC_MODEL_PROOF_GPU_IDS=1,2,3
~~~

Create this drop-in on each node:

<code>~/.config/systemd/user/trtmc-github-proof@.service.d/10-node-env.conf</code>

~~~ini
[Service]
EnvironmentFile=/workspace/users/yifeif/.config/trtmc/proof-node.env
~~~

Do not prefix the environment file with a minus sign. A missing host contract
must prevent the listener from starting instead of silently using default GPUs.
Do not enable <code>PrivateTmp</code>; all listeners on a node must see the same
lock files.

After <code>systemctl --user daemon-reload</code>, each active proof unit must
be drained and restarted before it can be considered compliant. Merely writing
the file does not change an already-running listener's environment.

No Hugging Face token belongs in this file or in the runner service
environment.

## 6. Repository Implementation

Implementation must begin from the current <code>github/main</code> on a
short-lived branch, be pushed to the <code>github</code> remote, and enter
<code>main</code> through a reviewed PR. Do not use the existing experimental
dual-host patch wholesale; its per-model online warm behavior conflicts with
this plan.

### 6.1 <code>.github/workflows/trtmc-ci.yml</code>

- Remove <code>max-parallel: 16</code> from the Pre-Merge model-proof matrix.
- Keep <code>fail-fast: true</code>.
- Keep the reusable model-proof workflow and common runner label.
- Remove <code>secrets: inherit</code> from the model-proof call.
- Do not replace 16 with 28.

Result: an affected-model matrix can use every currently idle production
listener without a workflow edit when capacity changes.

### 6.2 <code>.github/workflows/nightly.yml</code>

Add a GitHub-hosted <code>cache-warm-plan</code> job that checks out the exact
certified Nightly revision and reads
<code>.github/ci/gb300-pool-topology.json</code>. It must:

1. validate the topology's exact minimal schema;
2. reject an empty node set, duplicate or malformed node labels, unsorted or
   duplicate GPU indices, an invalid slot count, and extra fields;
3. normalize nodes in a deterministic order;
4. emit exactly one matrix row containing only <code>node_label</code> for each
   declared node.

There is no runner-inventory API call, GitHub App, private-key secret, or
online-runner preflight. GitHub's normal job scheduling is the availability
check: if no listener with a declared node label is online, that node's warm
job cannot complete and the Nightly barrier remains closed.

Convert <code>cache-warm</code> into a dynamic per-node matrix:

- <code>fail-fast: false</code>, so all node failures are visible in one run;
- use only <code>runs-on: ${{ matrix.node_label }}</code>; do not add
  <code>self-hosted</code>, <code>trtmc-gb300-proof</code>, or a fixed runner
  selector;
- run the full active single-GPU Nightly cache plan on every node, not only
  models that later happen to land there;
- acquire the node-wide cache lock exclusively for warm and verification;
- use the existing strict command:

~~~bash
python -u scripts/warm_hf_cache.py \
  --exclude-ci-tier multi_device \
  --strict \
  --attempt-timeout-seconds 600
~~~

- immediately run a second
  <code>--local-only --strict</code> validation while network is disabled;
- emit and upload a node-specific cache receipt;
- make the Nightly model-proof stage depend on successful completion of every
  node warm entry.

Remove <code>max-parallel: 16</code> from the Nightly model matrix and do not
replace it with 28. Preserve <code>fail-fast: false</code> so Nightly still
captures the complete model failure set.

The report, required gate, and any release/publish stage must continue to fail
closed when topology planning or any cache-warm entry fails.

### 6.3 <code>.github/ci/gb300-pool-topology.json</code>

Keep one checked-in declarative file as the source of truth for per-node cache
warm and exact capacity verification:

~~~json
{
  "schema_version": 1,
  "kind": "trtmc_capacity_topology",
  "slots_per_gpu": 4,
  "rollback_baseline_node_label": "trtmc-node-gb300-nvl-019-compute02",
  "nodes": [
    {
      "node_label": "trtmc-node-gb300-nvl-019-compute01",
      "gpu_indices": [0, 1, 2, 3]
    },
    {
      "node_label": "trtmc-node-gb300-nvl-019-compute02",
      "gpu_indices": [1, 2, 3]
    }
  ]
}
~~~

This file is the only checked-in warm-and-admission target declaration. It may
temporarily include a staged node before that node receives the production
label. Do not duplicate its node list in workflow YAML, repository variables,
a runner name list, or a second cache configuration. A pure subcommand in the
existing <code>tools/ci/capacity_canary.py</code> validator derives the Nightly
matrix:

~~~json
{
  "include": [
    {"node_label": "trtmc-node-gb300-nvl-019-compute01"},
    {"node_label": "trtmc-node-gb300-nvl-019-compute02"}
  ]
}
~~~

The node label is a declarative scheduling label, while
<code>gpu_indices</code> and <code>slots_per_gpu</code> are the acceptance
contract. Production model jobs still route only through the common generic
pool label; they never select a node from this file.
<code>rollback_baseline_node_label</code> identifies the protected prior
production node for a data-derived rollback canary; it is not used by normal
scheduling or Nightly cache routing. Its 12-slot result is safe only while
external overlap workloads remain excluded from those GPUs.

### 6.4 <code>.github/workflows/model-proof.yml</code>

- Stop assigning <code>TRTMC_MODEL_PROOF_GPU_IDS</code> from a repository
  variable or workflow default.
- Require the value inherited from the runner service.
- Validate <code>TRTMC_NODE_ID</code>, GPU ID syntax, slot count, and shared
  lock path before expensive work.
- Keep all Hugging Face readiness checks local-only and strict.
- Keep cache preparation before GPU-lease acquisition, so a cache miss does
  not consume a GPU slot.
- Hold a shared cache lock while validating and copying from the shared cache.
- Release that cache lock after the private model cache view is ready.
- Mount the shared source cache read-only during the offline check and private
  cache-copy stage. The network-disabled proof container sees only its
  job-private cache view and never mounts the shared source cache.
- Do not declare, inherit, or reference an HF token.

### 6.5 <code>tools/ci/gpu_lease.py</code> and
<code>tools/ci/model_proof.py</code>

Do not replace or redesign the allocator. Make only the changes needed to:

- fail closed when host-local GPU IDs are absent;
- expose a reusable shared/exclusive cache-file-lock context where appropriate;
- enrich <code>gpu-lease.json</code> evidence with:
  - schema version 3 plus exact workflow run, run attempt, and reusable-workflow
    job identity;
  - runner name;
  - node ID and hostname;
  - GPU index and GPU UUID;
  - slot ID or all slot IDs;
  - resource class and slots per GPU;
  - lock namespace;
  - source revision;
  - acquisition and release timestamps.

The globally unique slot identity in evidence is:

~~~text
(node_id, gpu_uuid, slot_id)
~~~

Numeric GPU index alone is not unique across nodes.

The reusable-workflow singleton gate and combined report must bind every lease
to the real job ID <code>prove</code>, the current run ID, and the artifact's
run attempt. The combined report normalizes the validated receipts and fails
closed on runner-to-node movement, node/hostname/lock-namespace drift,
GPU-UUID/index aliasing, inconsistent per-node slot counts, a reused runner
during an overlapping interval, or overlapping ownership of the same physical
slot. Merely checking that these fields are present is not certification.

### 6.6 Cache receipt

Reuse <code>scripts/warm_hf_cache.py</code> as the only downloader. Its current
strict, local-only, retry, timeout, gated-model, and completeness behavior is
already the desired base.

Use its existing selected-repository evidence output when constructing the
receipt. If the existing interface cannot expose the required counts without
parsing human-readable logs, add only a structured summary-output option; do
not add a second downloader or change warm/pass behavior.

The workflow should create a compact
<code>cache-warm-receipt.json</code> after successful warm plus local-only
verification. It must contain:

- receipt schema version 2; node ID, hostname, the runner that actually
  executed the warm, exact run ID, run attempt, and job ID. The runner name is
  evidence only and is not a fixed routing identity;
- tested source revision;
- a cache-plan digest derived from the sorted selected dependency set;
- a resolved-cache digest derived from repository IDs and resolved local refs;
- expected, present, and missing counts;
- ordered warm and local-only verification start/end times;
- common expected/present/missing counts, warm downloaded/already-cached
  counts, and separately prefixed local-verification
  present/missing/downloaded/already-cached counts;
- final strict local-only result with zero downloads;
- canonical host cache root, host Hub-cache path, and cache-lock path. The
  local-only summary names the fixed read-only container path
  <code>/hf-cache/hub</code>; it is deliberately not required to equal the host
  path.

Write the receipt atomically only after success and upload it as an artifact.
Two node receipts from the same Nightly must have the same source revision,
run ID, run attempt, job ID, cache-plan digest, resolved-cache digest, expected
count, and standardized host cache paths. The verifier requires the exact
field set and rejects old schemas, missing/extra fields, unsafe paths, malformed
counts/timestamps, duplicate identities, or any mismatch before the Nightly
barrier opens.

Because the attempt binding is exact, a partial cache-warm failure must be
recovered with GitHub's <code>Re-run all jobs</code>, not
<code>Re-run failed jobs</code>. Otherwise successful nodes retain receipts
from the prior attempt and the fleet barrier correctly remains closed.

Do not put a token, signed URL, or command-line secret in a receipt.

### 6.7 Capacity canary

Add a trusted, default-branch-only, manually dispatched workflow such as
<code>.github/workflows/model-proof-capacity-canary.yml</code>. It must not
download a model or receive an HF token. It should reuse the real
<code>GpuLease</code>, launch a tiny Docker GPU UUID probe, hold the lease to an
absolute barrier time, and upload a receipt.

Every dispatch reads the checked-in
<code>.github/ci/gb300-pool-topology.json</code> from the protected main-branch
revision as its expected topology. The initial admission file is:

```json
{
  "schema_version": 1,
  "kind": "trtmc_capacity_topology",
  "slots_per_gpu": 4,
  "rollback_baseline_node_label": "trtmc-node-gb300-nvl-019-compute02",
  "nodes": [
    {
      "node_label": "trtmc-node-gb300-nvl-019-compute01",
      "gpu_indices": [0, 1, 2, 3]
    },
    {
      "node_label": "trtmc-node-gb300-nvl-019-compute02",
      "gpu_indices": [1, 2, 3]
    }
  ]
}
```

This is verification data and the Nightly per-node warm declaration, not
production model routing logic. The worker matrix still targets only the common
generic proof label, and GitHub remains the scheduler. The trusted preparation
job validates the exact minimal schema, derives seven GPUs and the 16 plus 12
capacities from the indices and slots/GPU, canonicalizes node order, and hashes
the result. Each GPU receipt carries that
digest; the GitHub-hosted postflight rejects a missing or different digest and
checks receipts against the normalized contract and exact worker job ID
<code>exercise</code>. For cross-workflow proof, the
two authenticated first-attempt source runs and the verifier must all bind to
the same digest. This prevents a total-only result from passing with the wrong
node split or with compute02 GPU 0, without putting a hostname branch or a
fleet-size repository variable in production CI.

Keep this canary after rollout. It is the fastest repeatable proof when a node
or runner is added, removed, or restarted.

The retained workflow has five manual modes:

- <code>shared-capacity</code> emits <code>expected_slots + 1</code> jobs for
  the 28/29 proof;
- <code>rollback-capacity</code> derives the single protected rollback node
  from <code>rollback_baseline_node_label</code> and emits
  <code>expected_slots + 1</code> jobs for its exact 12/13 proof;
- <code>shared-cohort</code> emits exactly <code>expected_slots</code> jobs and
  accepts a caller-supplied cohort ID and absolute barrier;
- <code>cross-workflow-verify</code> downloads receipts from two exact run IDs
  with read-only Actions permission and verifies their combined pool. It also
  authenticates both source-run API records as first-attempt, successful manual
  runs of this exact workflow on <code>main</code>, in this repository, at the
  verifier's current source revision;
- <code>exclusive-safety</code> proves same-GPU exclusive serialization.

Mode/contract relationships are deliberately small: shared-capacity and
exclusive-safety require <code>expected_slots == full capacity</code>;
rollback-capacity requires <code>expected_slots == protected rollback-node
capacity</code>; shared-cohort may request any positive subset up to full
capacity; and cross-workflow-verify requires two equal cohorts whose combined
size equals full capacity. A future node is represented by one additional row
in the topology data PR before it is warmed or admitted. Normal Pre-Merge and
Nightly model jobs do not consume the node rows for routing; Nightly cache
planning does.

The manual workflow exposes no topology override. Every capacity canary reads
the checked-in file from its protected-main checkout, preventing a
dispatch-time copy from becoming a second topology source of truth. Unit tests
cover synthetic contracts and failure injection without adding a production
workflow input.

Only the GitHub-hosted verifier job receives <code>actions: read</code>. GPU
workers receive no Actions-write permission, model input, or Hugging Face
credential. The workflow contains no <code>concurrency</code> group.

### 6.8 Tests

Update or add:

- <code>tests/tools/test_github_actions_ci.py</code>;
- <code>tests/tools/test_model_proof_runner.py</code>;
- <code>tests/tools/test_model_proof_security.py</code>;
- <code>tests/tools/test_generate_model_proof_report.py</code>;
- <code>tests/tools/test_capacity_canary.py</code>;
- <code>tests/tools/test_cache_warm_receipt.py</code>;
- <code>tests/tools/test_cache_lock.py</code>;
- <code>tests/tools/test_ci_container_secrets.py</code>;
- existing <code>GpuLease</code> fairness/cancellation tests;
- <code>tests/tools/test_warm_hf_cache_static.py</code>.

Tests must prove:

- neither model matrix has a fixed max-parallel value;
- both use the common reusable proof workflow;
- no normal Pre-Merge, Nightly model-proof, or reusable proof workflow contains
  a compute hostname or hardcoded fleet size; the single declarative topology
  file carries node labels and GPU indices for cache planning and acceptance;
- GPU IDs come only from the host environment;
- missing host policy fails;
- Pre-Merge does not inherit HF secrets;
- Pre-Merge rejects token variables and token-file indirection before checkout,
  scrubs them before subprocesses, and proves DNS plus numeric-IP HTTPS
  transport are unavailable inside the proof container;
- proof cache access is strict, local-only, and read-only at the shared-cache
  boundary;
- cache miss occurs before GPU acquisition;
- topology parsing rejects an empty set, duplicate or malformed nodes,
  duplicate/unsorted GPU indices, invalid slot density, and extra fields;
- cache planning emits exactly one node-label-only matrix entry per declared
  node and never requires <code>self-hosted</code>, a production label, a
  runner-inventory API, or a fixed runner name;
- warm failure blocks Nightly proof;
- malformed topology contracts, digest substitution, GPU-index/UUID aliasing,
  compute02 GPU 0, wrong slots per GPU, and a same-total but wrong node split
  all fail the capacity canary;
- shared or exclusive lease evidence from any job other than
  <code>exercise</code> fails the capacity canary;
- the accepted generic contract proves seven GPUs, four slots per GPU, and the
  exact 16 plus 12 per-node capacity without hostname-based scheduling;
- shared and exclusive lease safety, fairness, cancellation, and stale-ticket
  recovery remain correct;
- cache receipts reject stale schemas and are bound to exact run, attempt, job,
  source, node/executing-runner/hostname, paths, counts, timestamps, and
  digests;
- combined proof reports reject forged job identity, runner/node movement,
  topology drift, zero-length leases, and overlapping slot ownership.

Do not weaken any model or comparison acceptance criterion to make CI pass.

## 7. Declarative Topology Change Contract

The user-approved design is semi-automatic. It deliberately does not inspect
the repository runner inventory and therefore needs no GitHub App, App ID,
private-key secret, administration permission, or extra token.

The only warm-and-admission target declaration is
<code>.github/ci/gb300-pool-topology.json</code>. A topology-only PR is required
when any of these physical-capacity facts change:

- a new GPU node is added or an admitted node is removed;
- the allowed GPU indices on a node change;
- <code>rollback_baseline_node_label</code> changes;
- <code>slots_per_gpu</code> changes.

No topology PR is required to restart an existing listener, replace one runner
registration with the same node label and host policy, or change how many of
the already-authorized listeners are temporarily online. GitHub automatically
uses all online listeners carrying <code>trtmc-gb300-proof</code>.

The topology PR must be declarative data only unless the schema itself is
intentionally changing. CI validates it, derives one Nightly cache-warm row per
node, derives expected capacity, and rejects malformed or duplicate entries.
After merge, the new node must pass strict warm and local-only verification
before its listeners receive the common production label.

GitHub Actions still provides its normal per-job
<code>${{ github.token }}</code> where existing workflow operations require it;
that built-in token is unrelated to node discovery and this design adds no new
GitHub credential.

## 8. Safe Rollout Sequence

Each phase has an explicit exit gate. Do not skip forward because a runner
merely appears online.

Several labels, user services, and host-policy files are already pre-staged.
They are not accepted rollout evidence until the current safe baseline and raw
pre-admission evidence are retained. Phases 2, 3, and 6 therefore revalidate
and admit existing state instead of restarting services unnecessarily.

### Phase 0 — Snapshot and freeze the prior 12-slot production baseline

- [ ] Refresh GitHub main SHA and save the runner API JSON.
- [ ] Record every runner ID, name, status, busy state, and label.
- [ ] Record repository variables that affect model proof.
- [ ] Record both nodes' four GPU indices, UUIDs, health, memory, and active
      processes.
- [ ] Record checksums of the current user unit and any drop-ins.
- [ ] Record compute02 proof-00 through proof-11 as the only production-labeled
      listeners, compute02 proof-12 through proof-15 as offline standby, and
      compute01 proof-00 through proof-15 as online but not production-labeled.
- [ ] Record cache filesystem usage, inode usage, ownership exceptions, and
      the confirmed absence of an active rsync.
- [ ] Confirm no current job is using a runner before changing its label or
      service.

Evidence directory recommendation:

~~~text
artifacts/gb300-ci-rollout/<timestamp>-baseline/
~~~

Exit gate: the snapshot is saved and can reconstruct the previous safe state.

### Phase 1 — Implement and validate the repository PR

- [ ] Fetch <code>github/main</code>.
- [ ] Create a short-lived branch from the exact current GitHub main.
- [ ] Implement only the repository changes in Section 6.
- [ ] Confirm the patch contains no per-host branch and no literal production
      concurrency cap.
- [ ] Run focused tests and lint:

~~~bash
python3 -m ruff check --config ruff.toml \
  tools/ci/cache_warm_receipt.py tools/ci/capacity_canary.py \
  tools/ci/gpu_lease.py tools/ci/model_proof.py \
  tools/ci/model_proof_inner.py tools/ci/model_proof_security.py \
  scripts/generate_model_proof_report.py \
  tests/tools/test_cache_warm_receipt.py tests/tools/test_capacity_canary.py \
  tests/tools/test_generate_model_proof_report.py \
  tests/tools/test_github_actions_ci.py \
  tests/tools/test_model_proof_runner.py \
  tests/tools/test_model_proof_security.py
PYTHONPATH=python:. python3 -m pytest \
  tests/tools/test_cache_lock.py \
  tests/tools/test_cache_warm_receipt.py \
  tests/tools/test_capacity_canary.py \
  tests/tools/test_ci_container_secrets.py \
  tests/tools/test_github_actions_ci.py \
  tests/tools/test_model_proof_runner.py \
  tests/tools/test_generate_model_proof_report.py \
  tests/tools/test_model_proof_security.py \
  tests/tools/test_warm_hf_cache_static.py -q
git diff --check
actionlint .github/workflows/nightly.yml .github/workflows/trtmc-ci.yml \
  .github/workflows/model-proof.yml \
  .github/workflows/model-proof-capacity-canary.yml
~~~

- [ ] Push to the <code>github</code> remote and open a PR targeting
      <code>main</code>.
- [ ] Wait for GitHub CI and review.

Exit gate: PR code is green and reviewable, but it is not merged until current
production runners have a compatible host environment.

### Phase 2 — Stage the declarative node-label contract

- [ ] Confirm the PR contains exactly one warm-and-admission target declaration
      at <code>.github/ci/gb300-pool-topology.json</code> and that it encodes
      the exact 16 plus 12 target from Section 6.3.
- [ ] Add exactly one node label to every current proof registration.
- [ ] Remove the common production label from all compute01 proof
      registrations before starting any compute01 proof service.
- [ ] Remove the common production label from compute02 proof-12 through
      proof-15, wait for <code>busy=false</code>, then stop and disable them.
- [ ] Give compute02 proof-12 through proof-15 only the standby label.
- [ ] Remove the legacy compute02-specific label after verifying the new labels.
- [ ] Confirm any pre-staged <code>trtmc-cache-anchor</code> label is inert;
      defer optional cleanup rather than mutate live runners for this rollout.

Exit gate: labels describe nodes unambiguously; compute02 exposes no more than
12 production listeners; compute01 exposes no production capacity yet; and no
fixed runner is required for cache routing.

### Phase 3 — Install and verify host-local policy

On both nodes:

- [ ] Create the standard cache and lock directories as user
      <code>yifeif</code>.
- [ ] Write <code>proof-node.env</code> with the policy in Section 5.
- [ ] Add the systemd user-unit drop-in.
- [ ] Run <code>systemctl --user daemon-reload</code>.
- [ ] Verify <code>Linger=yes</code>; do not rerun sudo merely for reassurance.
- [ ] On compute02, drain production listeners one at a time, restart each,
      inspect the listener process environment, and restore its common label.
- [ ] Start or select at least one compute01 node-labelled listener and inspect
      its inherited environment. It must still lack the common production
      label; it need not be proof-00.
- [ ] Query GPU UUIDs from the exact indices named in each node policy.
- [ ] Confirm all same-node listeners resolve the same GPU-lock directory inode
      and lock namespace.

Exit gate: every active production listener has the correct host-local GPU
policy, and at least one compute01 node-labelled listener can accept its future
warm row without advertising proof capacity.

### Phase 4 — Clear every pre-merge gate, then merge the scheduling and cache PR

- [ ] On compute02, rerun the measured ownership/read-write audit and require
      zero ownership, unreadable-file, unwritable-file, and
      unwritable-directory exceptions. Run the exact targeted repair in
      Section 13 only if that fresh audit finds an exception; do not
      recursively chown the approximately 1 TB cache.
- [ ] Retain the raw pre-admission runner and host snapshot, including hashes,
      in the access-controlled rollout evidence store.
- [ ] Confirm no runner-discovery App variable, private-key secret, or runner
      inventory permission is required by the patch.
- [ ] Obtain the workload owners' confirmation of a declared drained
      acceptance window for the compute01 GitLab scheduler and compute02
      external Docker GPU claims. The same approval must select a durable
      post-acceptance state: dedicate the seven GPUs to GitHub CI, reconfigure
      the workloads away from them, or make every workload lease-aware. A
      bounded drain followed by uncoordinated restoration is not sufficient.
- [ ] Confirm the generic compute01 GitHub runner is technically CPU-only or
      include it in the same owner-approved drain/reconfiguration plan; its
      absence from the proof label alone does not prevent Docker GPU access.
- [ ] Rebase the implementation branch on current <code>github/main</code>.
- [ ] Rerun focused tests and CI.
- [ ] Require exact-head CI to be green and resolve all review feedback while
      the PR remains draft.
- [ ] Only after every pre-merge gate above passes, mark the PR ready for
      review/merge and obtain the required approvals.
- [ ] Refresh the immutable runner/host/container snapshot and exact drain
      manifest. Freeze new <code>run-ci</code> authorizations and manual
      Nightly/canary dispatches, avoid the scheduled Nightly window, and
      require zero nonterminal Pre-Merge, Nightly, or capacity-canary runs that
      could later enqueue proof work. Require the production pool to report 12
      online and zero busy in two polls, no proof container or held GPU/cache
      lock, and no CI-GPU process.
- [ ] After review and workload-owner approval, start the same bounded
      acceptance window before merge: stop the compute01 GitLab scheduler,
      stop only the freshly validated compute02 overlap containers, and prove
      that all seven CI GPUs have no external compute process.
- [ ] Merge through the normal GitHub ruleset using squash or rebase.
- [ ] Record the merged main SHA.
- [ ] Retain the repository-level
      <code>TRTMC_MODEL_PROOF_GPU_IDS=1,2,3</code> variable as a compatibility
      guard for old-ref workflow dispatches and reruns. The merged workflow
      must not reference it and must use host-local policy. Deleting it would
      make an old workflow fall back to GPU 0 on compute02, so removal requires
      a separate fail-closed compatibility change rather than this rollout.
- [ ] Confirm an existing compute02 proof still selects only GPU 1, 2, or 3.

Exit gate: every documented pre-merge blocker is cleared, the acceptance
window is active, production workflow code uses host-local GPU topology, and
the existing compute02 pool remains functional. Until this gate is satisfied,
the PR remains draft and unmerged.

### Phase 5 — Finish cache bootstrap and warm every node

- [ ] Retain the current snapshot showing that no rsync is active; do not start
      or restart one.
- [ ] Do not copy or broadly redownload the already complete representative
      compute01 cache.
- [ ] Reconfirm that the pre-merge compute02 ownership/read-write audit remains
      clean; any new exception stops rollout and returns to the targeted repair
      gate in Phase 4.
- [ ] Manually dispatch trusted Nightly cache planning/warm from the merged
      main revision.
- [ ] Treat this as a full Nightly, not a warm-only operation: the workflow has
      no cache-only dispatch. Do not cancel it after the cache-ready barrier.
      Keep compute01 unlabelled until this run is terminal, so all model jobs
      use the existing 12-slot pool. This is pre-admission bootstrap evidence,
      not the final clean 28-slot Nightly.
- [ ] Verify the checked-in topology produces exactly two warm matrix entries,
      one for each declared node label.
- [ ] Verify each warm job was scheduled by its node label alone and may use any
      listener on that physical node.
- [ ] Verify both perform the full active single-GPU Nightly plan.
- [ ] Verify strict warm succeeds on both nodes.
- [ ] Verify the second network-disabled local-only check downloads zero bytes.
- [ ] Compare the two cache receipts and require identical source, plan, and
      resolved-cache digests.
- [ ] Let the bootstrap Nightly finish and retain its result separately before
      draining the proof pool for hardware canaries. Do not count it as the
      clean 28-slot acceptance Nightly.

Exit gate: both node caches are strict-ready for the same plan. The compute01
node listeners may now be admitted to production.

### Phase 6 — Admit exactly 28 production listeners

Durable external ownership, before adding any compute01 production label:

- [ ] Activate the workload-owner-approved final state: the compute01 GitLab
      scheduler is disabled, GPU-inaccessible, or lease-aware; the generic
      compute01 runner is proven CPU-only/lease-aware or stopped; and the
      compute02 overlap workloads are stopped with their deployment/restart
      source made unable to reacquire GPUs 1 through 3, reconfigured to GPU 0,
      or lease-aware.
- [ ] Exercise every relevant restart source—or perform an owner-approved host
      reboot—and prove that none can regain an unleased CI GPU. Record service,
      deployment, process, and GPU state. If this cannot be proved, keep
      compute01 unlabelled and do not claim durable 28-slot capacity.

compute01:

- [ ] Add <code>trtmc-gb300-proof</code> only after node cache readiness.
- [ ] Revalidate proof-00 through proof-15 as active/enabled with the inherited
      host-local policy; do not restart them unnecessarily.
- [ ] Add the common label to each only after its unit, environment, Docker,
      GPU probe, lock path, and workspace checks pass.
- [ ] Enable proof-00 through proof-15 for user-systemd restart.

compute02:

- [ ] Keep only proof-00 through proof-11 enabled and production-labeled.
- [ ] Keep proof-12 through proof-15 disabled and standby-labeled.
- [ ] Confirm GPU 0 is absent from every active listener environment.

Fleet:

- [ ] Query the runner API in one snapshot.
- [ ] Require exactly 28 online production-labeled runners, regardless of
      whether an individual runner is idle or busy at the snapshot instant:
      16 on compute01 and 12 on compute02.
- [ ] Require no 29th production-labeled runner, whether online or offline.
- [ ] Require every production runner to have exactly one node label and no
      routing dependency on a fixed cache runner.
- [ ] Confirm the compute01 generic runner remains separate and is either
      healthy under proven CPU-only/lease-aware enforcement or intentionally
      stopped.

Exit gate: durable external ownership is active, and GitHub registry topology
and node-local capacity both equal 28.

### Phase 7 — Hardware and concurrency acceptance

Run the tests in Section 9. Any duplicate lease, wrong GPU, wrong node count,
cache miss, secret exposure, or inability to reach 28 concurrent slots blocks
completion.

- [ ] Dispatch the 28/29 mode using the exact checked-in Section 6.3 topology
      file; the workflow exposes no topology override.
- [ ] Save the normalized contract and digest uploaded with the verification.
- [ ] Require the result to report seven GPUs, four slots per GPU, and the
      exact 16 plus 12 node distribution, with compute02 index 0 absent.
- [ ] Reuse byte-for-byte equivalent contract data for both 14-slot source
      cohorts and their cross-workflow verifier.
- [ ] After the primary canaries pass, perform the protected 12/13 rollback
      rehearsal from Section 9.8, restore exact 28, and rerun the 28/29 proof.

Exit gate: every acceptance artifact passes and the results are linked in this
document.

### Phase 8 — Normal CI acceptance

- [ ] Create several small, test-specific PRs that trigger different affected
      model matrices without weakening any production acceptance criterion.
- [ ] Use those PRs to prove normal Pre-Merge scheduling reaches both nodes,
      uses distinct real slot receipts, queues safely when the pool is full,
      and remains healthy across repeated runs.
- [ ] Close the test-specific PRs after their durable run and artifact links
      have been recorded; do not merge test-only payloads into
      <code>main</code> unless they are independently useful repository tests.
- [ ] Run one complete Nightly on the same main-line design.
- [ ] Start this acceptance Nightly only after all 28 production labels are
      admitted and the durable external-workload ownership state is active;
      do not count the bootstrap Nightly from Phase 5 as this run.
- [ ] Confirm Nightly warmed each node before any Nightly model proof.
- [ ] Confirm Nightly model proofs consume the same generic pool used by the
      test-specific Pre-Merge PRs, rather than a separate host-specific path.
- [ ] Confirm model-proof logs contain no Hugging Face download.
- [ ] Run the next unchanged cache plan and confirm the warm stage performs
      zero downloads on both nodes.
- [ ] Monitor GPU OOM, runner loss, disk headroom, and queue time for at least
      one additional Nightly cycle.

Exit gate: several real PR runs and a complete Nightly use the pool normally,
not only the synthetic canary.

## 9. Acceptance Tests

### 9.1 Static and unit acceptance

All tests from Phase 1 pass on the exact merged main SHA. Static assertions
also confirm:

- no hidden matrix cap below fleet capacity;
- no hostname routing;
- no current workflow reference to the compatibility-only global GPU-ID
  variable;
- no HF token in Pre-Merge or reusable model proof;
- warm failure is a required Nightly failure;
- every cache read happens before GPU acquisition;
- all proof jobs use the same node-local lock namespace.

### 9.2 Runner topology acceptance

One runner API snapshot must show:

| Check | Required result |
| --- | --- |
| Production proof runners | Exactly 28 registered and online |
| compute01 production runners | Exactly 16 |
| compute02 production runners | Exactly 12 |
| Node labels | Exactly one on every production runner |
| Declared warm nodes | Exactly compute01 and compute02 in the topology file |
| Cache-warm selector | Node label only; no fixed runner or default label |
| compute02 proof-12..15 | Standby-labeled, disabled, no production label |
| compute02 GPU 0 | Absent from policy and all receipts |
| Generic compute01 runner | Online only if CPU-only or lease-aware; never production-labeled |

### 9.3 28/29 shared-slot canary

The canary preparation job selects an absolute barrier roughly 15 minutes in
the future and launches a 29-leg matrix on the common proof label. Run it in a
declared acceptance window after other jobs using the proof pool have drained;
otherwise it measures unrelated contention instead of admitted capacity.
Use the exact checked-in topology contract from Section 6.3. The workflow
exposes no manual topology override. The
preparation job must publish only its normalized JSON and digest to downstream
jobs; workers must record the digest but must not use node rows for routing.

Each leg:

1. validates host policy;
2. acquires a real shared <code>GpuLease</code>;
3. records node, runner, GPU index, GPU UUID, slot, namespace, and timestamps;
4. launches a tiny container bound to the leased GPU;
5. verifies the container-observed UUID matches the receipt;
6. holds the lease until the common barrier;
7. releases and uploads its receipt.

A GitHub-hosted postflight downloads all receipts and verifies their runner,
lease, barrier, start/release timestamps, and topology-contract digest.

Pass criteria:

- exactly 28 leases overlap at the barrier;
- all 28 <code>(node_id, gpu_uuid, slot_id)</code> tuples are unique;
- all receipts contain the canonical digest of the checked-in topology file;
- each node ID maps to exactly one hostname and lock namespace, each hostname
  maps to one node ID, and each GPU UUID maps to one node ID;
- each <code>(node_id, gpu_index)</code> maps to exactly one GPU UUID and each
  <code>(node_id, gpu_uuid)</code> maps to exactly one GPU index;
- the compute01 node label covers indices 0 through 3, four GPU UUIDs, four
  slots per GPU, and exactly 16 receipts;
- the compute02 node label covers only indices 1 through 3, three GPU UUIDs,
  four slots per GPU, and exactly 12 receipts; index 0 is rejected;
- the observed node set, seven-GPU total, per-node counts, and 28-slot total
  exactly equal the contract rather than merely summing to the requested size;
- exactly 28 distinct proof runner names execute during the first wave;
- the remaining leg's worker start timestamp is after a first-wave lease
  releases, proving no matching listener was available earlier;
- the 29th leg then starts, acquires, and completes;
- all 29 jobs finish with no timeout or leftover lock.

This is the primary proof of the 28-slot Goal. Runner count alone is not enough.

### 9.4 Cross-workflow sharing

Start two trusted canary runs without a shared workflow concurrency group, each
requesting 14 shared leases and holding to the same absolute barrier.

Pass criteria:

- combined concurrency is 28, not 14;
- all combined tuples are unique;
- neither workflow has a fleet-global serialization lock;
- both runs receive capacity;
- the two runs share one total pool of 28 rather than each assuming it owns 28.

Cancel one holder and verify its flock releases automatically and a queued job
acquires the freed slot within the configured timeout.

Execution uses the retained workflow rather than a second scheduler:

1. Choose one cohort ID and future barrier, then dispatch two
   <code>shared-cohort</code> runs with <code>expected_slots=14</code> and the
   checked-in topology, from the same main revision.
2. After both finish, dispatch <code>cross-workflow-verify</code> with those two
   exact run IDs, the same cohort ID, barrier, 14 slots per run, and the exact
   same checked-in topology. Receipt-digest validation must happen before
   combined placement is accepted; all three runs must bind to its same digest.
3. For cancellation recovery, dispatch a 27-slot fill run and a separate
   one-slot holder to the same barrier, then dispatch a one-slot waiter.
4. Save the holder and waiter run and job URLs plus raw Actions API JSON proving
   that the waiter was queued before the cancellation request. Record the
   regular-cancel command and timestamp, then save raw API JSON showing the
   holder concluded as <code>cancelled</code>.
5. Save the cancelled holder log containing exactly one acquisition marker and
   the completed waiter receipt. Manually compare revision, cohort, node,
   hostname, GPU UUID, slot, lock namespace, and acquisition timestamps; require
   the waiter to acquire the released slot within the declared timeout.
6. Record the evidence URLs and SHA256 digests of every saved API response, log,
   command transcript, and receipt in the rollout evidence log.

Each holder flushes a compact machine-readable acquisition marker immediately
after the network-free container UUID probe and before sleeping to the barrier,
so cancellation does not erase the holder identity needed by this study. This
cancellation check is an explicit operator-reviewed rollout gate, not a
machine-authenticated verifier or a claim derived from caller-authored JSON.

### 9.5 Exclusive-GPU safety

Use tiny probes, not model downloads, to validate:

- an exclusive lease contains exactly slots 0, 1, 2, and 3;
- a GPU has at most one exclusive holder;
- shared work cannot overlap an exclusive holder on the same GPU;
- an exclusive request waits for older shared holders;
- younger shared requests do not bypass a queued exclusive request;
- shared work resumes after release;
- every one of the seven allowed GPU UUIDs can run a bound container probe.

The retained capacity workflow provides an <code>exclusive-safety</code> mode.
One generic runner acquires a real exclusive lease, launches a second real
exclusive contender pinned dynamically to the selected GPU, proves that the
contender stays queued, releases the primary, and requires the contender to
acquire, UUID-probe, and release. Its receipt is deliberately scoped to one
scheduler-selected node. The shared 28/29 mode supplies bound-container UUID
coverage for every admitted GPU; real shared/exclusive test PRs and the
deterministic allocator tests supply the remaining cross-class evidence.

A large 28-leg exclusive canary can demonstrate seven simultaneous exclusive
leases, with later jobs waiting locally. Do not interpret this as a guarantee
that any arbitrary set of seven exclusive jobs will be optimally spread by
GitHub.

### 9.6 Cache acceptance and failure injection

For each Nightly cache plan:

- the checked-in topology produces exactly one entry per declared node;
- each entry selects only its node label, with no fixed runner name,
  <code>self-hosted</code>, or common production label;
- each node receives the same plan;
- every receipt has <code>missing_count=0</code>;
- both receipts have identical plan and resolved-cache digests;
- node IDs map to distinct physical hostnames, and every receipt is bound to
  the current Nightly run ID, run attempt, <code>cache-warm</code> job ID, and
  certified source revision;
- host cache root, Hub-cache, and lock paths are canonical and consistent;
- a second strict local-only run succeeds with network disabled and zero
  downloaded bytes;
- Nightly model proof starts only after both receipts pass;
- the workflow injects the HF token only into the trusted default-branch warm
  container; package, VLM, proof, canary, and Pre-Merge jobs receive no token;
- the token never appears in command lines, logs, receipts, or artifacts.

Inject and verify fail-closed behavior:

| Injected failure | Required result |
| --- | --- |
| Empty, duplicate, or malformed topology node | Planning fails; no warm starts |
| Declared node has no online node-labelled listener | Its warm cannot complete; Nightly barrier stays closed |
| Warm lands on a runner whose host node ID differs from the label | Host-policy validation fails |
| Hugging Face 401/403 or missing dependency | Warm fails; no new ready receipt |
| Disk full | Warm fails; previous receipt is not overwritten |
| Corrupt or incomplete cache entry | Strict local-only validation fails |
| Pre-Merge cache miss | Job fails before GPU lease and makes no network request |
| Warm overlaps a proof cache read | Exclusive/shared cache lock prevents corruption |

### 9.7 Pre-Merge security acceptance

At runtime prove:

- host job and proof container lack <code>HF_TOKEN</code> and
  <code>HUGGING_FACE_HUB_TOKEN</code>, and token-file indirection is rejected;
- the shared cache cannot be modified through its read-only mount;
- DNS and HTTPS access fail in the proof container;
- cached models still pass strict local-only validation and real proof;
- an intentionally absent canary dependency fails rather than downloading,
  skipping, or warning-pass.

### 9.8 Semi-automatic scale acceptance

First prove that listener availability on an already-declared node needs no
repository change:

1. drain and stop one admitted production listener;
2. verify fleet audit reports 27;
3. restart the same already-admitted listener;
4. verify GitHub reports 28;
5. rerun the 28/29 canary and recover the full result.

Then rehearse the complete known-safe rollback without a topology override:

1. freeze new dispatch and wait for the production pool to become idle;
2. remove the common production label from all 16 compute01 proof listeners;
3. require a single runner snapshot with exactly the 12 compute02 production
   listeners and no other production-labelled registration;
4. dispatch <code>rollback-capacity</code> with
   <code>expected_slots=12</code>. The protected topology selector must derive
   exactly compute02 GPUs 1 through 3, the first wave must contain 12 unique
   slots/runners, and the 13th leg must start only after a release;
5. preflight and re-add the 16 compute01 production labels, require exact
   16 plus 12 capacity, then rerun <code>shared-capacity</code> with 28.

Any unrelated production-labelled runner, compute02 GPU 0 receipt, unexpected
node, or inability to recover exact 28 fails the rehearsal. The workflow has
no node or topology input; the protected main-branch topology selects both the
full fleet and rollback baseline.

Then review or rehearse the new-node path: bootstrap node-labelled listeners
without the production label, add exactly one node row in a declarative
topology PR, merge it, require that node's Nightly warm receipt, and only then
add the production label. No workflow code, repository secret, GitHub App, or
runner-name list changes.

This proves CI capacity follows matching online admitted listeners while a new
physical node requires only one explicit, reviewable topology-data change.

### 9.9 Test-specific PR acceptance

Create at least three narrowly scoped PRs from independent short-lived
branches. Their changes must be safe to close without merging and must not
change comparison thresholds, skip tests, or reduce required coverage. Select
small manifest or test fixtures that exercise different model-proof matrix
sizes and resource classes.

Pass criteria across the PR set:

- normal Pre-Merge dispatch observes both node IDs and more than one GPU UUID;
- every recorded <code>(node_id, gpu_uuid, slot_id)</code> tuple is internally
  consistent and no overlapping jobs duplicate a tuple;
- one run creates enough simultaneous work to demonstrate that no hidden
  <code>max-parallel: 16</code> limit remains;
- a queued excess leg begins after an earlier lease releases;
- an exclusive-GPU test does not overlap shared work on its selected GPU;
- rerunning an unchanged PR uses the already-warmed node cache and performs no
  Hugging Face download;
- all required PR checks and artifact uploads finish normally;
- every PR and run URL is recorded before the test PR is closed.

## 10. Resource and Operational Budget

At full shared concurrency:

- up to 28 proof containers can exist at once;
- compute01 can run 16 shared jobs across four GPUs;
- compute02 can run 12 shared jobs across three GPUs;
- with <code>TRTMC_MODEL_PROOF_BUILD_JOBS=2</code>, the theoretical build
  worker ceiling is 32 on compute01 and 24 on compute02;
- each node stores its own roughly 1 TB Hugging Face cache;
- each proof may create a private reflink/copy view, build tree, logs, and
  artifacts;
- four logical slots do not cap VRAM, CPU, RAM, disk I/O, or PCIe bandwidth.

Before admission, record:

- CPU count and load;
- system RAM and swap;
- free disk and inodes on cache, runner work, Docker, and artifact filesystems;
- Docker storage usage;
- GPU free memory and active processes.

During canaries and the first two Nightlies, collect peak:

- per-GPU memory and utilization;
- host RAM and load;
- disk throughput and free space;
- container count;
- queue time and lease wait time;
- OOM, Xid, Docker, and runner-service errors.

If four shared jobs per GPU is not stable, stop and revisit the slot density.
Do not hide the problem by changing model pass criteria.

## 11. Future Node Scale-Out Contract

After this design is merged, adding a physical node requires no workflow-code
or credential change. It requires one reviewable declarative-data PR, followed
by warm verification and production admission:

1. Verify GPU, driver, Docker, storage, network, and runner prerequisites.
2. Choose allowed GPU indices.
3. Create the standard cache, reference-cache, and lock paths.
4. Install <code>proof-node.env</code> with a unique
   <code>TRTMC_NODE_ID</code>.
5. Register
   <code>allowed GPU count × slots_per_gpu</code> generic proof listeners
   without <code>trtmc-gb300-proof</code>.
6. Choose one <code>trtmc-node-*</code> label that is unique to the physical
   node, give that same label to every listener on the node, and start enough
   node-labelled listeners for host validation and cache warm.
7. Submit a topology-only PR adding one sorted node row with that node label
   and its allowed GPU indices to
   <code>.github/ci/gb300-pool-topology.json</code>. Change no workflow YAML and
   add no runner name, GitHub App, or secret.
8. Let CI validate the topology schema, derived capacity, and Nightly matrix;
   then merge the PR through the normal ruleset.
9. Run trusted Nightly cache warm from the merged revision. The single new
   matrix row lands on any online listener carrying the new node label.
10. Require strict warm and network-disabled local-only verification, then
    verify its cache receipt, GPU UUID probes, lock namespace, disk headroom,
    and listener environment.
11. Add <code>trtmc-gb300-proof</code> to exactly the calculated listener count.
12. Run the capacity canary from main and require the new exact node
    distribution and fleet total from the checked-in topology.

After admission, Pre-Merge and Nightly model matrices automatically use the new
matching runner capacity. The Nightly warm matrix continues to generate one
row for every node in the same topology file. No production hostname branch,
repository capacity variable, runner-inventory API, fixed cache runner, or
runner-selection rule changes.

Adding, restarting, or replacing a listener on an already-declared node does
not need another PR as long as its node label and host policy are unchanged and
the number of production listeners never exceeds the declared slot capacity.
Changing the physical node set, allowed GPU indices, or slots per GPU does need
another topology-data PR.

Starting an arbitrary unconfigured runner is intentionally not sufficient. A
node must first satisfy the admission contract; otherwise it could expose a
cold cache, wrong GPU set, wrong lock directory, or false listener count.

If onboarding becomes frequent, package the host bootstrap and local preflight
as one idempotent script. That script is a convenience, not a new scheduler and
not required for the initial 28-slot rollout.

## 12. Rollback

Rollback is label-first and non-destructive:

1. Remove <code>trtmc-gb300-proof</code> from affected runners so GitHub cannot
   dispatch new work.
2. Wait for every affected runner to report <code>busy=false</code>.
3. Stop and disable only the affected proof user units.
4. Preserve runner registrations, runner directories, cache, topology history,
   and evidence.
5. Keep compute02 proof-00 through proof-11 as the prior 12-slot production
   baseline only if every overlapping external workload remains stopped,
   reconfigured, or lease-aware.
6. Revert repository workflow code through a normal GitHub PR if the defect is
   in code.
7. Keep the compatibility-only global GPU-ID repository variable unchanged;
   both the 28-slot rollout and its rollback leave it in place for old refs.

External-workload restoration is conditional, not automatic. Restore the
compute01 GitLab scheduler or a compute02 overlap container only after the
GitHub production labels that protect its GPUs have been withdrawn, unless
the workload was reconfigured away from those GPUs or now acquires the same
lease. Restoring an uncoordinated workload while retaining all 28 production
labels is an immediate rollback failure.

There are therefore two explicit rollback outcomes. A CI-only rollback keeps
the overlap workloads excluded, retains the 12 compute02 labels, and proves
that pool with <code>rollback-capacity</code>. A full external-workload
restoration first withdraws every production label whose GPU can be reached by
the restored workload; it may leave fewer than 12 CI slots and is not accepted
as the 12-slot rollback canary state.

Do not delete the approximately 1 TB cache. Do not delete runner registrations
to roll back capacity.

Immediate rollback triggers:

- duplicate slot lease;
- compute02 GPU 0 appears in a proof;
- container UUID does not match the lease;
- runner count differs from allowed GPU slots;
- shared/exclusive overlap;
- cache corruption;
- Pre-Merge receives an HF token or network access;
- 28/29 canary cannot prove the 16 plus 12 distribution;
- normal CI shows a new infrastructure regression.

The CI-only 12-slot rollback rehearsal is successful when:

- compute02 has exactly 12 production runners online;
- <code>rollback-capacity</code> passes with
  <code>expected_slots=12</code>, proving the exact 12/13 behavior;
- existing Pre-Merge can proceed;
- no stale lock prevents subsequent work;
- compute01 proof registrations and cache remain available for diagnosis.

A full external-workload restoration is successful only when all conflicting
production labels are absent before those workloads restart and the resulting
smaller CI capacity is reported honestly.

## 13. Human and Sudo Checkpoints

No sudo is required for linger, the user proof services, or the runner
configuration: both nodes already report <code>Linger=yes</code>, and those
resources are under the user account.

The initial compute02 cache audit found 2,781 entries not owned by
<code>yifeif</code>, including 110 unreadable files, 1,195 unwritable files,
and 661 unwritable directories. The narrow repair below was completed. A fresh
audit now reports zero exceptions in all four categories, and both nodes pass
the same 113-entry local-only plan with zero downloads. Do not rerun the repair
unless a later measured audit finds a new non-user-owned entry.

Run only on compute02:

~~~bash
sudo find \
  /workspace/users/yifeif/.cache/huggingface/hub \
  /workspace/users/yifeif/.cache/huggingface/modules \
  -xdev ! -user yifeif \
  -exec chown --no-dereference yifeif -- {} +
~~~

This targets only measured non-user-owned entries. Do not recursively change
ownership of the full approximately 1 TB cache. After any future repair, rerun
the read/write audit and strict warm/local-only checks before treating the node
as ready.

Human GitHub administration is required to:

- approve and merge the repository PR under the normal ruleset;
- administer runner labels during staged admission.

No GitHub App, runner-inventory permission, App ID, or new private-key secret is
required by this semi-automatic topology design.

Human workload coordination is also required for a declared drained acceptance
window: compute01 has a root GitLab runner service that could claim all GPUs,
and compute02 has restartable external Docker GPU claims. Their owners must
approve the exact conflict set, bounded window, and restoration path before
cache warm, production admission, or full hardware canaries.

The retained preparatory acceptance-window runbook predates the declarative
topology design: it still names the removed GitHub App gate and its manifest
has <code>execution_authorized=false</code>. Treat it only as historical target
evidence. After final review and owner approval, regenerate a fresh immutable
manifest and runbook without the App condition, bind them to the current PR
head and live container device requests, and set execution authorization
explicitly. Never execute the retained draft as-is.

After that approval, the first temporary sudo operation is the reversible stop
of the compute01 system-level GitLab scheduler. Run this on compute01 at the
start of the accepted window, then verify that it is inactive:

~~~bash
sudo systemctl stop gitlab-runner.service
systemctl is-active gitlab-runner.service | grep -Fx inactive
~~~

Do not disable the service merely to run the acceptance window. If the rollout
fails or is cancelled before durable ownership is established, first withdraw
the affected GitHub production labels, then restore the service and verify
both active and enabled state:

~~~bash
sudo systemctl start gitlab-runner.service
systemctl is-active gitlab-runner.service | grep -Fx active
systemctl is-enabled gitlab-runner.service | grep -Fx enabled
~~~

If the workload owner approves dedicating compute01 to the 28-slot GitHub
pool, convert the temporary stop into a reboot-safe final state before claiming
completion:

~~~bash
sudo systemctl disable --now gitlab-runner.service
systemctl is-active gitlab-runner.service | grep -Fx inactive
systemctl is-enabled gitlab-runner.service | grep -Fx disabled
~~~

That change is reversible with
<code>sudo systemctl enable --now gitlab-runner.service</code>, but only after
withdrawing the compute01 production labels or installing and validating an
equivalent shared-lease integration.

The four compute02 containers whose device requests overlap CI GPUs 1 through
3 are a separate non-sudo drain owned by that node's workload owners. Rebuild
and revalidate the exact container manifest immediately before the window;
never stop the two GPU0-only containers or widen the target set by hand. A
successful permanent 28-slot rollout must also change the overlap containers'
owner-controlled restart/deployment policy so a Docker or host restart cannot
reintroduce access to GPUs 1 through 3. Otherwise keep them stopped only during
the window and withdraw the affected production labels before restoration.

Outside that approved GitLab-service drain, ask the user for another exact sudo
command only if a new measured blocker requires it:

- a required cache path is owned by root and strict warm cannot read, replace,
  or repair it;
- user <code>yifeif</code> lacks Docker or NVIDIA device access;
- a host prerequisite truly requires a system-level service or permission.

Before any sudo request:

1. identify the exact path, device, or permission;
2. show the non-sudo failure;
3. propose the narrowest command;
4. avoid blanket recursive ownership changes;
5. explain rollback.

## 14. Goal Completion Checklist

The Goal is complete only when all boxes below refer to the same merged
<code>github/main</code> design:

- [ ] Repository changes merged to GitHub <code>main</code>.
- [ ] Focused static and unit tests green.
- [ ] Exactly 28 production proof runners online: compute01 16, compute02 12.
- [ ] The single topology file declares exactly compute01 GPUs 0..3 and
      compute02 GPUs 1..3 at four slots per GPU.
- [ ] Nightly derives exactly one node-label-only warm row per declared node,
      with no fixed runner or runner-inventory credential.
- [ ] All seven allowed GB300 UUIDs healthy.
- [ ] compute02 GPU 0 absent from policy and proof receipts.
- [ ] Every node's listener count equals allowed GPUs times four.
- [ ] Same-node runners share one real GPU-lock namespace.
- [ ] Both nodes have strict-ready receipts for the same cache plan.
- [ ] Second cache validation succeeds offline with zero downloads.
- [ ] Pre-Merge has no HF token, no proof-container network, and no writable
      shared-cache mount.
- [ ] 28/29 shared canary passes with 16 plus 12 unique slot receipts.
- [ ] Canary artifact contains the normalized trusted topology and matching
      receipt digest; exact node indices, seven GPUs, and four slots/GPU pass.
- [ ] Two concurrent workflows share 28 unique slots without collision.
- [ ] Cancellation releases a slot and queued work recovers.
- [ ] Shared/exclusive hardware safety canary passes.
- [ ] At least three test-specific PRs prove normal Pre-Merge scheduling,
      queueing, cache reuse, and shared/exclusive safety on the common pool.
- [ ] One complete Nightly succeeds after warming both nodes.
- [ ] An unchanged following Nightly downloads zero cache bytes.
- [ ] The conditional compute02 12-slot rollback baseline has been rehearsed
      with <code>rollback-capacity</code>, then exact 28 has been restored.
- [ ] No scheduler or container outside the shared lease namespace can regain
      access to the seven CI GPUs after reboot while 28 production labels are
      admitted; owner approval and the durable host/deployment state are
      recorded.
- [ ] Run URLs, runner snapshots, receipts, and canary artifacts are recorded
      below.

## 15. Execution Evidence Log

Fill this table during rollout. A checkbox without a durable link is not
evidence.

| Evidence | Revision or timestamp | URL or artifact path | Result |
| --- | --- | --- | --- |
| Baseline runner inventory |  |  |  |
| Host/GPU baseline |  |  |  |
| Pre-merge current-state snapshot (not Phase-0 baseline or 28-slot acceptance) | 2026-07-22T04:52:58Z | <code>reports/gb300-ci-rollout/2026-07-22-pre-merge-current-state.md</code> | Historical sanitized snapshot retained; current supersession note records cleared cache/CI blockers |
| Implementation PR | <code>c200eed7</code> before the latest review/hardening delta | [Draft PR #519](https://github.com/NVIDIA/TensorRT-Model-Connect/pull/519) | Attempt-1 exact-head CI green; every later pushed head still requires exact-head CI |
| Merged main SHA |  |  |  |
| Historical focused test log | 2026-07-22T05:49:14Z | <code>reports/gb300-ci-rollout/2026-07-22-focused-validation.md</code> | Superseded anchor-design repository evidence; not current topology-design proof |
| Topology-design focused test log | 2026-07-22 | <code>reports/gb300-ci-rollout/2026-07-22-topology-validation.md</code> | Repository validation, <code>c200eed7</code> exact-head CI, and two-node local-only cache readiness green; hardware/Nightly acceptance pending |
| compute01 cache receipt |  |  |  |
| compute02 cache receipt |  |  |  |
| Bootstrap pre-admission Nightly |  |  |  |
| Durable external GPU ownership and restart/reboot proof |  |  |  |
| Final 28-runner inventory |  |  |  |
| 28/29 shared canary |  |  |  |
| Cross-workflow canary |  |  |  |
| Exclusive safety canary |  |  |  |
| Cancellation recovery |  |  |  |
| Test PR 1 / normal shared Pre-Merge |  |  |  |
| Test PR 2 / queue and cache reuse |  |  |  |
| Test PR 3 / shared-exclusive safety |  |  |  |
| First complete Nightly |  |  |  |
| Zero-download following Nightly |  |  |  |
| Rollback rehearsal |  |  |  |

Terminal evidence requires the durable ownership/restart proof, exact 28-runner
snapshot, 28/29 and paired cross-workflow canaries, cancellation and exclusive
safety, a protected 12/13 rollback rehearsal followed by recovered 28/29,
three real test PRs, and both a complete post-admission Nightly and its
unchanged zero-download successor on the same common runner pool.
