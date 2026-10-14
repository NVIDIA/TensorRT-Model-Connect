<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Activate Stable Community GPU CI

The cutover promotes the qualified Dev implementation to trusted `main` and
retires the public Internal premerge bridge. Ordinary PRs run CPU checks, plan
admission, GPU tests when required, and confirmed resource cleanup. The intended
single automated merge gate is `Stable Community GPU CI` on the current PR head.
Passing it makes a PR ready for review by an NVIDIA employee maintainer; human
acceptance and merge authorization remain separate.

Do not reuse the legacy `Stable Community CI` context. Existing PR heads can
already carry a success under that name from the previous CPU-only pipeline.
The distinct `Stable Community GPU CI` context prevents that historical success
from satisfying the new gate.

**A source PR cannot change GitHub repository settings.** The workflow promotion
and the required-status migration below are separate deployment steps. Preparing
or reviewing these commands does not apply them. An authorized repository
administrator performs the settings update after the new trusted Stable path
has been verified.

## Before activation

1. Review the promotion PR and its exact-head validation under the currently
   deployed rules. Keep the old required status in place while preparing and
   merging the approved workflow change; do not open an unprotected interval.
2. After the reviewed change reaches `main`, run a fresh ordinary PR through the
   automatic Stable path. Verify the pinned source identities, CPU checks,
   selected GPU cases, confirmed cleanup and final `Stable Community GPU CI=success`
   on its current head. Include a fork PR in the live validation. A manual Dev
   result does not prove the automatic Stable path.
3. Check a preparation failure and a failure after the family entry point using
   existing or controlled runs. Confirm the final classifications remain visible
   and all allocated resources are recovered. An older head's receipt is not
   evidence for the new head.
4. Check that a plan changing more than five families is rejected before GPU
   allocation, that ordinary small plans retain all selected cases, and that a
   shared change still selects the five shared smoke families.
5. Keep the old gate required until these checks pass. Since its bridge has been
   retired, new PRs may remain merge-blocked during this brief activation window;
   do not bypass the old gate or claim cutover is complete prematurely.

The private shared-base catalog and reader credential must already be configured
in the protected GPU environment. Keep registry coordinates, tokens and private
package metadata out of public configuration, logs and PR text.

## Prepare the settings payload

These commands only read GitHub settings and write local JSON. They identify the
existing Internal-status ruleset and preserve its conditions, enforcement,
bypass policy and unrelated rule types. They intentionally stop if another
required-status context is present; inspect unexpected policy before editing it.

```bash
set -euo pipefail
REPOSITORY=NVIDIA/TensorRT-Model-Connect
CUTOVER_STATE=$(mktemp -d)

gh api "repos/$REPOSITORY/rules/branches/main" \
  > "$CUTOVER_STATE/effective-before.json"
RULESET_ID=$(jq -er '
  [.[] | select(.type == "required_status_checks")
   | select(any(.parameters.required_status_checks[];
       .context == "TRTMC Internal CI / Automated premerge gate"))
   | .ruleset_id] | unique
  | if length == 1 then .[0] else error("Expected one Internal gate ruleset") end
' "$CUTOVER_STATE/effective-before.json")
jq -e '
  [.[] | select(.type == "required_status_checks")
   | .parameters.required_status_checks[].context] | unique
  == ["TRTMC Internal CI / Automated premerge gate"]
' "$CUTOVER_STATE/effective-before.json" >/dev/null

gh api "repos/$REPOSITORY/rulesets/$RULESET_ID" \
  > "$CUTOVER_STATE/ruleset-before.json"
jq -e '
  .source_type == "Repository" and .target == "branch"
  and .enforcement == "active"
  and .conditions.ref_name.include == ["refs/heads/main"]
  and .conditions.ref_name.exclude == []
' "$CUTOVER_STATE/ruleset-before.json" >/dev/null

jq '{name, target, enforcement, bypass_actors, conditions, rules}' \
  "$CUTOVER_STATE/ruleset-before.json" \
  > "$CUTOVER_STATE/restore-required-status.json"
jq '
  .name = "Require Stable Community GPU CI on main"
  | .rules |= map(
      if .type == "required_status_checks" then
        .parameters.required_status_checks = [{context: "Stable Community GPU CI"}]
      else . end)
' "$CUTOVER_STATE/restore-required-status.json" \
  > "$CUTOVER_STATE/activate-required-status.json"

cat "$CUTOVER_STATE/activate-required-status.json"
printf 'Settings review files: %s\n' "$CUTOVER_STATE"
```

Review the proposed JSON and retain the restore payload privately with the
activation record. No new bypass actor or temporary exemption is needed. If the
live rules have changed, regenerate the payload from a fresh read instead of
replaying an old snapshot.

## Apply after authorization and live validation

Run this in the same shell after the preparation above and the prerequisite
validation. The PUT replaces the existing required-status rule in one update;
it does not delete the gate first.

```bash
gh api "repos/$REPOSITORY/rulesets/$RULESET_ID" \
  > "$CUTOVER_STATE/ruleset-immediately-before.json"
jq '{name, target, enforcement, bypass_actors, conditions, rules}' \
  "$CUTOVER_STATE/ruleset-immediately-before.json" \
  > "$CUTOVER_STATE/current-required-status.json"
cmp "$CUTOVER_STATE/restore-required-status.json" \
  "$CUTOVER_STATE/current-required-status.json"

gh api --method PUT "repos/$REPOSITORY/rulesets/$RULESET_ID" \
  --input "$CUTOVER_STATE/activate-required-status.json" \
  > "$CUTOVER_STATE/activation-response.json"
gh variable set TRTMC_COMMUNITY_CI_DUAL_RUN \
  --repo "$REPOSITORY" --body false

gh api "repos/$REPOSITORY/rules/branches/main" \
  > "$CUTOVER_STATE/effective-after.json"
jq -e '
  [.[] | select(.type == "required_status_checks")
   | .parameters.required_status_checks[].context] | unique
  == ["Stable Community GPU CI"]
' "$CUTOVER_STATE/effective-after.json" >/dev/null
```

The promoted automatic workflow already runs Stable only even if the old dual
run variable remains true. Setting it to false records the intended deployment
state. Manual Dev qualification remains available and cannot satisfy or replace
the Stable merge status.

Inspect the effective rules and a fresh ordinary PR again. Confirm its current
head requires Stable, never requires Internal premerge, and cannot obtain a
passing Stable result while selected GPU coverage or cleanup is incomplete.
DCO, PR Metadata and optional qualification may remain visible; they are not
additional required CI statuses under this cutover policy.

Existing open PR heads without the new context stay blocked even when the old
`Stable Community CI` context is green. A maintainer can rerun or manually
dispatch a fresh trusted Stable execution for a selected open PR and verify the
new context on its current head. Do not copy old statuses, fabricate a new
success, or bulk-start every open PR as part of this settings migration. Each
GPU execution still requires its own allocation and confirmed cleanup.

## NVIDIA maintainer review

A passing CI aggregate is a handoff for human review, not automatic merging.
The repository's existing CODEOWNERS file names its owner, but this promotion
does not create an employee team or configure required employee-only approvals.
GitHub organization association, an email address or repository write access is
not an authoritative employment roster.

Keep review with NVIDIA employee maintainers under the contribution process.
If machine-enforced employee approval is desired, first establish a maintained
reviewer team with suitable repository access and enough members to review one
another's PRs, then review a separate approval-rule update. Requiring approval
from a sole code owner can block that owner's own PRs. Do not describe the
procedural human step as a deployed GitHub enforcement rule.

## Rollback

Do not disable all required checks to recover from a bad rollout. Prepare and
review a workflow revert, restore a working trusted Internal bridge if reverting
to the old premerge flow, and validate its current-head status before restoring
the saved required-status payload. Keep the current required gate until the
replacement is usable. An authorized administrator can then use one PUT with
`restore-required-status.json` and verify the effective rules again. Private
Nightly requires no change for this premerge rollback.
