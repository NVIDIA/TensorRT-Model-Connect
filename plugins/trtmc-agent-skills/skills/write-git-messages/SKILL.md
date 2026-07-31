---
name: write-git-messages
description: Draft, revise, or review Git commit messages, PR titles, PR descriptions, and squash or rebase merge messages. Use when Codex needs to summarize a diff for reviewers, convert rough notes into a commit or PR message, check a message against Git and Conventional Commits style, or prepare repository contribution text before pushing/opening a PR.
---

# Write Git Messages

## Workflow

1. Identify the artifact and exact comparison: commit, PR, squash/rebase
   message, or review response; base ref/SHA; head ref/SHA; and intended target.
2. Inspect the actual change with `git status --short`, `git diff --stat
   <base>...<head>`, `git diff --name-status <base>...<head>`, and focused
   diffs. Include staged and untracked scope when drafting a new commit. If only
   a user summary is available, state that limitation.
3. Follow the repository's existing convention first. If it has no clear
   convention, default to Conventional Commits for commit and PR titles.
4. Separate implemented behavior, static/CPU validation, target-hardware proof,
   model parity, package evidence, and performance/qualification. Do not promote
   one tier into another.
5. Keep claims evidence-based. Do not invent issue numbers, benchmark results,
   tests, generated artifacts, reviewer decisions, or CI state.
6. Return ready-to-use text followed by notes only when a limitation or choice
   matters.

For source rationale, read `references/source-notes.md` only when you need to explain or adjust the standard.

## Commit Messages

Use this default shape:

```text
type(scope): imperative summary

Why this change is needed and what behavior changes.
Mention important constraints, migrations, or risk.

Refs: #123
BREAKING CHANGE: explain incompatible behavior
```

Guidelines:

- Keep the first line short and self-contained; aim for 50 characters when practical.
- Separate the title from the body with a blank line.
- Use imperative mood in the summary: `add`, `fix`, `document`, `remove`.
- Use a meaningful type: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `build`, `ci`, `chore`, or `revert`.
- Add a scope only when it helps route ownership or understand impact.
- Put the "why" and user-visible behavior in the body; do not restate every file changed.
- Use footers for issue links, co-authors, and breaking changes.
- Do not add a co-author unless that person or tool actually authored part of
  the change and the user wants the attribution.
- Split unrelated work into separate commits when possible. If not possible, choose the dominant type and explain the combined scope in the body.
- Avoid vague titles such as `update files`, `misc fixes`, `work in progress`, or `fix stuff`.
- In this repo, never include `Claude` in commit messages because the GitHub ruleset rejects it.

## PR Messages

Write PR bodies as a one-minute review brief in plain English. A reviewer should be able to understand why the PR exists, what done means, how it works, how it was validated, and what future readers should know without reconstructing the task from the diff.

Use this default PR body:

```markdown
## Background
<Why this task exists. Include the problem, prior failure, user request, policy,
or workflow gap that motivated the change.>

## Exit Criteria
- <objective condition that means this task is done>
- <any explicit non-goal or boundary if useful>

## Implementation
- <what changed, grouped by behavior or component rather than every file>
- <how the design works and why this approach was chosen>

## Validation
- `<command>`: <result>
- `<command>`: <result>

## Notes For Future Readers
- <follow-up, limitation, reviewer order, operational note, or reason a future
  maintainer should not remove or duplicate this change>
```

For tiny PRs, keep all sections but make each one one or two sentences. For larger PRs, use bullets inside sections. Omit only sections that are truly inapplicable, not just inconvenient to fill.

When reporting validation:

- Include exact commands and whether they passed, failed, or were not run.
- Include the result in plain English, not only the command.
- If validation was not run, say why and describe the residual risk.
- Tie remote CI claims to the exact head SHA. A check on an older head is not
  evidence for the current diff.
- Distinguish a skipped check from a passing check and a dry run from execution.
- In this repository, say premerge passed only when the current head has
  `trtmc/premerge/required=PASS`. A successful Internal CI Bridge dispatch or
  Source workflow is not the premerge result.
- Do not quote or link private Internal CI logs, artifacts, runner details,
  package coordinates, or internal URLs in Source PR text. Use the sanitized
  exact-head status and separately reproducible public evidence.

Guidelines:

- Make the PR title mirror the expected squash or merge commit title.
- Check the final title and squash/rebase message for the repository's banned
  terms; in this repo that includes `Claude`.
- Keep the PR focused on one purpose. If the diff is broad, say why it could not be split.
- In the body, include purpose, done criteria, change overview, relevant issue links or prior discussions, and any requested feedback.
- For multi-file or multi-layer changes, tell reviewers where to start and what order to review in the future-reader notes.
- Call out security, dependency, migration, compatibility, and rollback concerns when present.
- Omit empty sections rather than leaving placeholders.

When the PR depends on another PR, names a merge order, or changes ownership,
put that relationship in `Notes For Future Readers`. When a branch was rebased
or force-pushed, do not claim current CI is green until the checks on the new
head complete.

## Review Checklist

Before returning a message, verify:

- The title says what changed without relying on the body.
- The body explains why the change exists when the reason is not obvious.
- The body defines exit criteria clearly enough that a reviewer can tell whether the task is complete.
- The implementation section explains the approach in human terms, not just filenames.
- The type and scope match the diff.
- Test claims match commands actually run or user-provided evidence.
- Validation includes both commands and pass/fail results.
- Base/head and changed-file scope match the artifact being described.
- CI and model-proof claims refer to the current head SHA.
- Source/Internal CI wording respects the public evidence boundary and does
  not call a bridge dispatch a test pass.
- Failures, skips, unrun checks, and residual risks are explicit.
- The PR title and expected squash/rebase title are aligned and contain no
  repository-banned terms.
- Future-reader notes capture any review order, limitation, or maintenance warning that is not obvious from the diff.
