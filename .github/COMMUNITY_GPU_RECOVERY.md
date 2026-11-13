# Community GPU cleanup

Community GPU CI bounds cleanup instead of occupying a runner until deletion
is confirmed. This workflow uses the existing owner and independent cleanup
jobs. Unresolved requests require maintainer follow-up; there is no scheduled
recovery service or recovery-heartbeat dependency.

## Cleanup budgets

- The owner allows 15 minutes for a known instance, or 90 seconds while the
  original creation has no instance ID.
- Independent cleanup and rerun preflight allow 20 minutes overall, shortened
  to three minutes while no instance IDs are known. GitHub evidence calls and
  provider operations share finite budgets.
- Known deletion budgets leave margin over the observed successful sample:
  91 runs, median 495 seconds, maximum 736 seconds.
- Allocation retains its 45-minute budget. Model validation criteria and its
  execution budget are unchanged.

An exhausted budget produces a terminal infrastructure error. It never means
that an unknown creation was rejected, that deletion succeeded, or that a model
passed. Original intent/final lease artifacts remain available when release is
unconfirmed. The same original run's rerun must release its older attempts
before submitting a new request.

## Manual cleanup without running tests

After the source run has completed, a maintainer or administrator can dispatch
the same workflow with `task=cleanup` and `cleanup_run_id=<original run ID>`:

```bash
gh workflow run community-ci.yml \
  --repo NVIDIA/TensorRT-Model-Connect --ref ci/developer \
  -f task=cleanup -f cleanup_run_id=123456789
```

Replace `123456789` with the original run ID. Use `main` after this change is promoted. No PR number is required. This path
runs only the existing cleanup job: it never allocates a VM or runs models, and
it does not change the original test result. The current trusted workflow
resolves the original source SHA and attempt from GitHub. Only completed
Community CI dispatches from this repository's `main` or `ci/developer` are
accepted. Each original producer and immutable lease still undergoes the
existing provenance and resource-ownership checks. Running owners are rejected.

If release remains unconfirmed, preserve the original lease and consult the
provider's actual inventory and request disposition. Retry cleanup when the
provider is healthy; a VM created late may then be discoverable. Missing,
expired, corrupt, or conflicting evidence never authorizes deletion. Existing
lease/report artifacts have 35-day retention; investigate before they expire.
Evidence lost to retention requires manual recovery of authoritative ownership.

## Pause new GPU requests

Set repository variable `TRTMC_COMMUNITY_GPU_PAUSED=true` to prevent new GPU
requests. The default is `false`. The check runs before creating allocation
intent, after prior-attempt cleanup. A paused request reports an infrastructure
failure; it is not a skipped GPU pass. Already-running model jobs and all
cleanup paths continue. Return the variable to `false` after investigation.
This is a manual operational control, not an automatic circuit-breaker.

## Operational limits

There is no guaranteed future cleanup after these jobs end. Someone must
follow up unresolved requests, including late-created instances and cancelled
runs. Until then, a residual VM can continue incurring charges. Prior-attempt
checks protect the same original run; unrelated PRs or new workflow runs can
still allocate, so pause new requests during a widespread provider incident.

Generic HTTP 400/500, a transport timeout, or an empty inventory does not prove
that a submitted request cannot create a VM later. Uncertain creates are not
blindly retried; existing bounded retries for verified HTTP 429 rejections are
preserved. Public diagnostics remain limited to allowlisted fields; do not
publish raw provider responses or credentials. The underlying creation error
needs separate provider/client investigation.
