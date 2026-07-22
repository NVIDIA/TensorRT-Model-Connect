# GB300 CI: 28-slot KISS rollout

## Outcome

Use one generic GitHub Actions label, `trtmc-gb300-proof`, for all model-proof
work. After rollout the pool contains:

- `gb300-nvl-019-compute01`: GPUs `0,1,2,3`, four shared slots per GPU,
  16 runner listeners.
- `gb300-nvl-019-compute02`: GPUs `1,2,3`, four shared slots per GPU,
  12 runner listeners.

Normal Pre-Merge and Nightly model jobs do not select a host. GitHub assigns
each matrix job to any idle runner carrying the generic label. The existing
host-local GPU lease then selects one of that host's shared GPU slots.

## Minimal repository change

1. Stop overriding GPU IDs in `model-proof.yml`. That value comes from each
   runner service environment.
2. Remove the fixed 16-job Pre-Merge and Nightly model matrix limits so the
   matrix can use every available generic runner when it has enough jobs.
3. Turn the existing Nightly cache-warm job into a two-row static matrix, one
   node label per host. The existing strict cache warmer runs once on each
   host before Nightly model proofs begin.

No new scheduler, GitHub App, runner-discovery service, cache-copy process, or
permanent capacity-canary framework is required.

## Runner configuration

Only the usable GPU list differs between the two nodes, so that is the only
runner-local value this patch needs to consume:

```text
# compute01
TRTMC_MODEL_PROOF_GPU_IDS=0,1,2,3

# compute02
TRTMC_MODEL_PROOF_GPU_IDS=1,2,3
```

The existing workflow continues to use four slots per GPU and the same lock
and Hugging Face cache path strings on both hosts. Those paths resolve on each
host's local filesystem; the two hosts do not share one cache.

Runner labels have two purposes:

- `trtmc-gb300-proof`: generic production scheduling.
- `trtmc-node-<hostname>`: route the one-per-node Nightly cache warm.

## Safe rollout

1. Keep the 16 compute01 proof runners online with only their node label while
   the pull request is under review.
2. Verify all four compute01 GPUs with a disposable, network-disabled CUDA
   probe and verify all 16 listener services are active and enabled.
3. Run the pull request's exact-head CI on the existing production pool.
4. Merge through the GitHub ruleset only after the required checks pass.
5. Run Nightly from merged `main`; require both node-specific cache-warm matrix
   rows to pass.
6. Add `trtmc-gb300-proof` to the exact 16 compute01 runner registrations with
   the GitHub runner-label API. Adding a label does not require `sudo` or a
   runner restart.
7. Assert the production label resolves to exactly 28 online runners:
   16 on compute01 and 12 on compute02.

If admission fails, remove only `trtmc-gb300-proof` from the 16 compute01
runners. Their services and node labels remain intact, so the rollback is
immediate and reversible.

## Acceptance tests

After merge, use short-lived test-only pull requests. Do not merge their test
workflow into `main`.

1. Capacity test: start 29 tiny jobs on `trtmc-gb300-proof`. The first 28 hold
   one real `GpuLease`; GitHub must report 16 jobs on compute01, 12 on
   compute02, and the 29th queued until a slot is released.
2. Ordinary Pre-Merge test: run a small model-owned change and confirm it can
   land on compute01 without any host-specific workflow selector.
3. Nightly test: dispatch merged `main`, confirm one successful cache-warm job
   on each node, then confirm model proofs consume the generic 28-runner pool.

Close the test pull requests after collecting run IDs, runner names, node
distribution, GPU/slot receipts, and final conclusions.

## Adding another GPU node

Expansion is intentionally semi-automatic and declarative:

1. Install and start the desired number of runner listeners on the new host.
2. Give them the generic production label and one shared node label.
3. Configure that host's usable GPU IDs in its runner service environment and
   use the shared four-slots-per-GPU, lock-path, and cache-path contract.
4. Add one row for the node label to the Nightly cache-warm matrix in a normal
   pull request.
5. Repeat the temporary `N+1` capacity test.

Normal Pre-Merge and Nightly model scheduling needs no further host-specific
code. The only per-node repository data is the cache-warm matrix row.
