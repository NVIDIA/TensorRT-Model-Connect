# Community GPU recovery

Community GPU CI reports a terminal infrastructure error when resource recovery
exceeds its budget. An unfinished lease remains unresolved. It never becomes
evidence that no VM was created, that deletion succeeded, or that model tests
passed.

## Budgets

The owner waits at most 15 minutes for a known instance and 90 seconds for an
uncertain allocation without an instance ID. Independent recovery and rerun
preflight have a 20-minute overall budget, reduced to three minutes while all
allocations lack IDs. Individual recovery turns remain bounded and preserve
their learned instance identities and deletion progress.

These values leave margin over the observed successful owner cleanup sample:
91 runs, median 495 seconds, maximum and empirical p99 736 seconds. The workflow
also provides an outer step timeout, allowing time to upload recovery progress.
Allocation retains its existing 45-minute software budget, with a 47-minute
step limit. Model validation criteria and its execution budget are unchanged.

## Checkpoint authority and retention

The default-branch `Community GPU recovery` workflow executes only a resolved
commit of the trusted `main` or `ci/developer` worker. It never checks out a PR
revision or executes PR-supplied recovery code. The existing `gpu-ci-dispatch`
environment must allow `main` and the selected worker branch.

Each new recovery run publishes one immutable journal with 90-day retention.
Retry recovery by starting a new manual run; the recovery workflow refuses to
rerun an old run and overwrite its journal. Original Community CI reruns remain
supported. The reader validates the repository, workflow, default
branch, job, successful upload, owner time window, and archive digest. A rerun
can still read the original completed job's receipt if an old run is retried.
Journal lookup uses GitHub's exact artifact-name filter, so unrelated CI
artifacts cannot hide the recovery history during an outage.

Every checkpoint carries all unresolved original leases forward, including
their original run/attempt, producer job ID, source revision, original artifact
digest, learned IDs, and deletion progress. An unresolved lease is never removed
because it is old. Completed receipts remain available beyond GitHub's rerun
window. The next checkpoint refreshes the retention of unresolved evidence; it
does not depend on the original owner artifact remaining downloadable forever.

The owner keeps its original immutable intent and final lease. Independent
cleanup also publishes its authenticated progress, even when cleanup fails.
These updates have a separate provenance rule tied to the cleanup job and the
same original producer/resource identity. Later sweeps and reruns consume them.

Missing, corrupt, expired, or ambiguous checkpoint history fails closed. It is
never replaced with an empty journal. If the scheduler is unavailable long
enough to lose all retained checkpoints, restore verified evidence before
allowing another allocation.

## Recovery and admission

The scheduler runs every 15 minutes and on an explicit manual dispatch. A single
concurrency group prevents overlapping sweeps. Each round is bounded; incomplete
discovery and unresolved records are checkpointed for another pass. GitHub
discovery cannot consume the entire provider-deletion budget. A live owner is
never used as authority to delete its resources.

Discovery initially inventories the preceding 35 days, the retained original
lease and rerun window. Historical leases outside this bootstrap window require
an explicit inventory audit before enabling new allocations. Once discovered,
unresolved records remain in every checkpoint regardless of their age.

Before preparation and again before reservation, admission checks the exact
open PR/head and a fresh, complete, apply-mode recovery checkpoint. A checkpoint
older than one hour blocks admission, exposing a failed scheduler heartbeat.
Admission also checks recent and active owner metadata, including reruns of old
runs, so a newly failed reservation is not hidden by the previous checkpoint.
Healthy concurrent reservations are allowed. An authenticated prior-cleanup
receipt permits its current rerun immediately, without waiting for the next
sweep. Progress merges preserve known IDs, deletion receipts and the cursor;
an untouched backstop export does not invent a newer observation timestamp.

Discovery errors are isolated per source run. A bad run remains visible and
keeps admission closed, but cannot prevent other owned resources being found
and recovered. Known discovery gaps are retried individually even after their
source runs age out of the normal discovery window; they are never cleared by
age alone. An optional empty backstop upload is not a discovery error and does
not erase the original owner's evidence. The heartbeat measures the recovery worker, not provider
availability; unresolved-resource limits and cooldown remain separate guards.

Repository variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `TRTMC_GPU_RECOVERY_WORKER_REF` | `ci/developer` | Trusted recovery worker; only `main` and `ci/developer` are accepted. |
| `TRTMC_GPU_RECOVERY_APPLY` | `false` | Scheduled sweeps observe by default; `true` enables ownership-verified deletion. |
| `TRTMC_GPU_MAX_UNRESOLVED` | `5` | Stop new allocations at this unresolved-lease count. |
| `TRTMC_GPU_ALLOCATION_COOLDOWN` | `900` | Cooldown in seconds after an unresolved producer completes. |

The admission inventory is a conservative guard, not a replacement for the
provider's account quota. Existing live model tests retain their per-PR
concurrency. No increase in machine allocations is inferred from a timeout or
an empty/incomplete API response.

## Rollout

1. Merge the worker changes into `ci/developer`. Its new admission guard blocks
   new allocation until the default-branch recovery entry is installed and
   healthy. This avoids bounded cleanup running without its recovery backstop.
2. Install `community-gpu-recovery.yml` on `main` as a small companion change.
   A schedule on `ci/developer` alone will not execute.
3. Run in observe mode. Check discovery completeness, original producer
   identities, unresolved leases, and the actual provider inventory. Check older
   outstanding resources outside the bootstrap window separately.
4. A maintainer/admin may manually select `apply` to validate a recovery round.
   Set `TRTMC_GPU_RECOVERY_APPLY=true` for subsequent automated recovery only
   after the observed inventory is understood. A fresh, complete apply-mode
   checkpoint is required to admit new model jobs.
5. Promote the worker with the Community GPU pipeline. After promotion, the
   worker variable may select `main`.

There is no automatic expiry of an uncertain request. HTTP 400, HTTP 500, a
transport timeout, or an empty inventory alone does not prove that the provider
will never create the VM later. Existing bounded 429 backoff and reconciliation
remain in place; generic errors are not reclassified as guaranteed rejection.

## Unresolved resources and outages

- Inspect the recovery workflow's journal and the original producer's report.
  Confirm the original repository/run/attempt, exact instance name, organization,
  SKU, and IDs before taking action. Never delete by name prefix alone.
- A late-created instance is discovered and deleted on a later bounded pass.
  Known owned IDs remain actionable when fresh discovery is degraded.
- If the provider never exposes an ID, retain the unresolved record and obtain
  the provider's authoritative request disposition. Do not edit a journal to
  mark it deleted, infer absence from its age, or raise allocation limits merely
  to bypass missing evidence.
- An ownership conflict blocks deletion and requires investigation. A deadline
  expiry is a different error and never sets the persistent ownership-conflict
  flag.
- Preserve checkpoints through authentication, GitHub, or provider outages.
  Restore the recovery worker/credentials and retry recovery before rerunning
  allocation. `--cleanup-prior` must resolve every earlier attempt before a
  rerun may submit a new request.
- Public diagnostics retain only the existing allowlisted status-code, enum,
  byte-count and hash fields. Do not publish raw provider responses, credentials,
  or account information to diagnose the underlying 400/500 cause.

Background recovery never changes an original model-test failure into success.
A successful recovery receipt permits a later explicit rerun; it does not
replace that rerun's validation.
