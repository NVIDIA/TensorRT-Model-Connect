# Community CPU execution policy

The repository's contributor-facing `/run-ci` command depends on a narrowly
scoped GitHub Actions workflow execution protection. The policy is configured
in GitHub, not in this repository, so changing the workflow alone does not make
external comments executable.

## Required policy boundary

First identify the enterprise or organization ruleset that currently rejects
external actors. Applicable rulesets aggregate and the most restrictive rule
wins, so adding a repository-level allow rule cannot override that denial. At
the controlling level, retarget the existing restriction to exclude this
repository, then create a replacement workflow execution protection with all
of these properties:

- target only `NVIDIA/TensorRT-Model-Connect`;
- permit the `issue_comment` event for the narrowest actor selector that
  includes unaffiliated external contributors;
- retain the existing restrictions for every other repository and event; and
- start in **Evaluate** mode before changing it to **Active**.

GitHub currently documents actor and event rules, plus repository targeting,
but not workflow-file targeting. Consequently, `community-cpu.yml` must remain
the only workflow in this repository that listens to `issue_comment`. The
repository contract tests enforce that invariant, and CODEOWNERS requires the
same review boundary for workflow and Community CPU controller changes.
The central rule cannot determine whether a commenter authored a pull request;
the workflow's read-only authorization job enforces that relationship before
any pull-request code is checked out.

See GitHub's public
[Workflow execution protections documentation](https://docs.github.com/en/organizations/managing-organization-settings/actions-policies/workflow-execution-protections)
and [ruleset layering documentation](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/about-rulesets#about-rule-layering)
for the current settings interface and composition semantics. The feature is in
public preview, so validate the available actor selector in the target
organization instead of assuming that a similarly named selector has the
required scope.

## Canary and activation

1. In Evaluate mode, inspect rule insights for a fork pull request whose author
   has no repository role. Confirm that the proposed repository boundary would
   permit the author's exact `/run-ci` event after the controlling restriction
   is retargeted.
2. Confirm that the workflow itself rejects a non-author with no maintain/admin
   role, a non-PR issue comment, a closed PR, a PR targeting another branch,
   and any body other than the exact command.
3. Confirm that the permitted request runs only on a GitHub-hosted runner, has
   no secrets, binds checks to the captured merge SHA, and executes the trusted
   base controller and baseline tests.
4. Confirm that a fourth failed request for the same merge SHA is rejected.
5. Confirm that the `main` ruleset requires CODEOWNER approval for the workflow
   and controller paths and does not grant ordinary writers a bypass.
6. Activate the replacement repository boundary while the controlling deny
   still covers this repository, verify it in rule insights, and only then
   exclude this repository from the controlling deny. For rollback, restore the
   controlling deny target before disabling the replacement boundary.
7. Repeat the external-author canary and inspect every event permitted by the
   replacement policy. Actor and event allowlists may combine across the policy,
   so verify that an unaffiliated actor cannot reach another existing trigger.

Every issue comment can create a lightweight workflow record after activation;
only an exact authorized `/run-ci` comment reaches the test jobs. This expected
Actions UI noise is the tradeoff for keeping the original contributor identity
and avoiding a bot credential or webhook service.

The workflow's comment-history budget is a soft throttle because authors can
edit or delete their own comments. Durable per-merge checks cap repeated
attempts for one revision, and actor-scoped concurrency bounds one account's
simultaneous work, but organization runner quotas remain the hard
repository-wide cost ceiling.
