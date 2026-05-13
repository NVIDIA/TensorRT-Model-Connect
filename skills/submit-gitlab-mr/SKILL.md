---
name: submit-gitlab-mr
description: Use when pushing a branch and creating a GitLab merge request. Trigger when user says "submit MR", "create MR", "open merge request", or "push and create MR". Requires glab CLI.
---

# Submit GitLab MR

## Overview

Use `glab` (not `gh`) to create GitLab merge requests. The `glab mr create` command requires `dangerouslyDisableSandbox: true` because it writes to `.git/config`.

## Quick Reference

```bash
# Push branch first
git push -u origin <branch-name>

# Create MR (requires dangerouslyDisableSandbox: true)
glab mr create \
  --source-branch <branch> \
  --target-branch master \
  --title "feat: short title" \
  --description "$(cat <<'EOF'
## Summary
- bullet points

## Test plan
- [x] done item
- [ ] todo item
EOF
)" \
  --remove-source-branch
```

## ADR Creation (Automatic)

After pushing the branch but BEFORE calling `glab mr create`, check if the
diff warrants an Architecture Decision Record.

### High-Confidence Triggers

Analyze the diff against master:

```bash
git diff --name-only master...HEAD
```

Create an ADR if ANY of these are found in the diff:

| Signal | Detection |
|--------|-----------|
| New runtime strategy | New file in `src/runtime/plugins/` containing `PluginRegistrar` |
| New family plugin | New `.py` file in `tensorrt_model_connect/tensorrt_model_connect/families/` (not `__init__.py` or `base.py`) |
| New pipeline class | New `.cpp` or `.h` in `src/runtime/pipelines/` |
| Config schema change | New field parsed in `src/runtime/registry/pipeline_plugin.cpp` or `tensorrt_model_connect/tensorrt_model_connect/config.py` |
| New E2E task strategy | New file in `tests/e2e_harness/runners/` |
| New comparator/reference | New file in `tests/e2e_harness/comparators/` or `references/` |
| Architectural refactor | 5+ files moved/renamed under `src/runtime/` |

Do NOT create ADRs for: bug fixes, new tests without architectural impact,
dependency updates, documentation-only changes, new E2E model manifests for
existing families.

### If Triggers Fire

1. **Determine the next ADR number:**

```bash
LAST_NUM=$(ls website/docs/context/adr/[0-9]*.md 2>/dev/null | sort -V | tail -1 | grep -oP '\d{4}')
NEXT_NUM=$(printf "%04d" $((10#${LAST_NUM:-0} + 1)))
```

2. **Generate a slug** from the change (e.g., `registry-based-dispatch`,
   `t5-encoder-decoder-plugin`). Use lowercase, hyphens, max 50 chars.

3. **Write the ADR** to `website/docs/context/adr/${NEXT_NUM}-${SLUG}.md`:

```markdown
---
number: <NNNN>
title: <concise title describing the decision>
status: Proposed
date: <YYYY-MM-DD>
source_commits: [<sha1>, <sha2>, ...]
---

## Context

<Why was this change needed? What problem does it solve? Include the
motivation from the conversation/commit messages. 2-4 sentences.>

## Decision

<What was decided? Describe what the code now does. Reference specific
files and patterns. 2-4 sentences.>

## Considered Alternatives

<If the conversation or code comments mention rejected approaches, list
them here. Otherwise: "Not captured — review and add if known.">

## Consequences

<What are the implications? New capabilities, constraints, maintenance
burden, performance characteristics. 2-4 sentences.>
```

4. **Update the ADR index** — add a row to `website/docs/context/adr/README.md`:

```markdown
| <NNNN> | <title> | Proposed | <YYYY-MM-DD> |
```

5. **Commit to the feature branch:**

```bash
git add website/docs/context/adr/${NEXT_NUM}-${SLUG}.md website/docs/context/adr/README.md
git commit -m "docs(adr): ADR-${NEXT_NUM} — <title>"
```

6. **Proceed with MR creation as normal.** The ADR is part of the MR diff.

### Opt-Out

If the user passes `--no-adr` to the submit command, skip ADR detection and
creation entirely.

## Key Flags

| Flag | Purpose |
|------|---------|
| `-s, --source-branch` | Branch to merge from |
| `-b, --target-branch` | Branch to merge into (default: project default) |
| `-t, --title` | MR title |
| `-d, --description` | MR body (use heredoc for multiline) |
| `--remove-source-branch` | Delete branch after merge |
| `--draft` | Mark as draft/WIP |
| `-a, --assignee` | Assign by username |
| `--reviewer` | Request review by username |
| `-l, --label` | Add labels (comma-separated) |
| `-f, --fill` | Auto-fill title/description from commits |

## Common Mistakes

- **Using `gh` instead of `glab`**: This is a GitLab project. `gh` is for GitHub.
- **Running without `dangerouslyDisableSandbox: true`**: `glab` writes to `.git/config`, which the sandbox blocks. You'll get "Device or resource busy" errors.
- **Forgetting to push first**: `glab mr create` needs the branch on the remote. Push with `-u` before creating.
- **Heredoc quoting**: Use `'EOF'` (quoted) to prevent shell expansion in the description body.
