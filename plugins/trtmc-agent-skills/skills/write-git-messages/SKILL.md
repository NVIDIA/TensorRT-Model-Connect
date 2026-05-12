---
name: write-git-messages
description: Draft, revise, or review Git commit messages, PR titles, PR descriptions, and squash or rebase merge messages. Use when Codex needs to summarize a diff for reviewers, convert rough notes into a commit or PR message, check a message against Git and Conventional Commits style, or prepare repository contribution text before pushing/opening a PR.
---

# Write Git Messages

## Workflow

1. Inspect the actual change before writing: use `git diff --stat`, `git diff --name-only`, and focused diffs for changed files. If only a user summary is available, state that the message is based on the provided summary.
2. Match the requested artifact: commit message, PR title/body, squash merge message, rebase todo wording, or review response.
3. Follow the repository's existing convention first. If the repo has no clear convention, default to Conventional Commits for commit titles and PR titles.
4. Keep claims evidence-based. Do not invent issue numbers, benchmark results, tests, generated artifacts, or reviewer decisions.
5. Prefer ready-to-use text. Return the final message in a fenced `text` or `markdown` block, followed by short notes only if tradeoffs matter.

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

Guidelines:

- Make the PR title mirror the expected squash or merge commit title.
- Keep the PR focused on one purpose. If the diff is broad, say why it could not be split.
- In the body, include purpose, done criteria, change overview, relevant issue links or prior discussions, and any requested feedback.
- For multi-file or multi-layer changes, tell reviewers where to start and what order to review in the future-reader notes.
- Call out security, dependency, migration, compatibility, and rollback concerns when present.
- Omit empty sections rather than leaving placeholders.

## Review Checklist

Before returning a message, verify:

- The title says what changed without relying on the body.
- The body explains why the change exists when the reason is not obvious.
- The body defines exit criteria clearly enough that a reviewer can tell whether the task is complete.
- The implementation section explains the approach in human terms, not just filenames.
- The type and scope match the diff.
- Test claims match commands actually run or user-provided evidence.
- Validation includes both commands and pass/fail results.
- Future-reader notes capture any review order, limitation, or maintenance warning that is not obvious from the diff.
