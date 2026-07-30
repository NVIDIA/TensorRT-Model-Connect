---
name: doc-sync
description: >-
  Use for documentation maintenance scans that keep ADRs, website wiki pages,
  and traceability matrices aligned with the current codebase. Produces focused
  GitHub PRs for ADR maintenance, wiki drift repair, and traceability audit.
---

# Doc Sync

## Purpose

Scan documentation drift and produce focused GitHub PRs. Process these phases in
order:

1. ADR maintenance: keep existing ADRs accurate.
2. Wiki drift repair: fix stale factual claims in `website/docs/wiki/*.md`.
3. Traceability audit: fix gaps in the ARCH/UD/UT/IT trace system.

Do not guess based on filenames. For behavioral documentation, read the source
files being described before editing prose.

## Change Set

Read the last scan SHA:

```bash
LAST_SHA=""
if [ -f website/docs/context/.last_scan_sha ]; then
  LAST_SHA=$(cat website/docs/context/.last_scan_sha)
fi
```

For a first scan, bootstrap from the oldest commit in the last 7 days:

```bash
LAST_SHA=$(git log --since="7 days ago" --format="%H" --reverse | head -1)
```

Fetch current GitHub main:

```bash
git fetch github main
CURRENT_SHA=$(git rev-parse github/main)
```

If there is no change set, update the marker only when that marker update is
part of the requested maintenance work. Otherwise report nothing to do.

Inspect changes:

```bash
git log --oneline "$LAST_SHA".."$CURRENT_SHA"
git diff --name-only "$LAST_SHA".."$CURRENT_SHA"
```

## Branch And PR Flow

Create one branch per phase that produces changes:

```bash
PHASE_NAME="adr"  # or "wiki" or "traceability"
DATE=$(date +%Y-%m-%d)
BRANCH="doc-sync/${PHASE_NAME}-${DATE}"

git fetch github main
git switch -c "$BRANCH" github/main
```

Skip a phase if the remote branch already exists:

```bash
git ls-remote --heads github "$BRANCH"
```

Push and open a GitHub PR:

```bash
git push -u github HEAD
gh pr create \
  --repo NVIDIA/TensorRT-Model-Connect \
  --base main \
  --head "$BRANCH" \
  --title "docs(${PHASE_NAME}): automated doc sync $(date +%Y-%m-%d)" \
  --body-file <pr-body.md> \
  --label documentation
```

Use `$write-git-messages` for the PR body.

## Phase 1: ADR Maintenance

Scan `website/docs/context/adr/*.md` excluding `README.md`.

Checks:

- Parse frontmatter: `number`, `title`, `status`, `date`, `source_commits`,
  and `superseded_by`.
- Verify backtick-quoted file paths under `src/`,
  `python/tensorrt_model_connect/`, `tests/`, and `include/` exist. For moved
  paths, use `git log --follow --diff-filter=R -- <old_path>`.
- Verify backtick-quoted class, function, and strategy names exist.
- Update numeric claims, including runtime strategy count, family plugin count,
  and E2E manifest count.
- Promote old `Proposed` ADRs to `Accepted` only when the described decision is
  clearly implemented.
- Mark ADRs `Superseded` when a newer ADR describes the replacement.
- If a `source_commits` SHA is not an ancestor of current `main`, mark the ADR
  `Deprecated` with a short note.
- Rebuild `website/docs/context/adr/README.md` from ADR frontmatter, sorted by
  number.

Ground truth commands:

```bash
rg -n "PluginRegistrar" src/plugins --glob "*.cpp"
find python/tensorrt_model_connect/families/ -name "*.py" \
  -not -name "__init__.py" -not -name "base.py" | sort
find tests/e2e/models/ -name "*.json" | sort
```

PR body should summarize changed ADR numbers, status changes, index updates, and
any deprecated/superseded rationale.

## Phase 2: Wiki Drift Repair

Scan all markdown files in `website/docs/wiki/`.

Collect reusable ground truth first:

```bash
rg -n "PluginRegistrar" src/plugins --glob "*.cpp"
find python/tensorrt_model_connect/families/ -name "*.py" \
  -not -name "__init__.py" -not -name "base.py" | sort
find tests/e2e/models/ -name "*.json" | sort
find src/runtime/domains/ -name "*.h" -o -name "*.cpp" | sort
find tests/builder/ tests/tools/ -name "test_*.py" | sort
find tests/cpp/ -name "test_*.cpp" | sort
find src/ include/ -type f \( -name "*.cpp" -o -name "*.h" \) | sort
find python/tensorrt_model_connect/ -type f -name "*.py" | sort
```

Rules:

- Counts and file paths can be fixed mechanically.
- Behavioral descriptions require reading the implementation first.
- Strategy tables must match the actual registry.
- Family/plugin lists must match current family files.
- Manifest schema examples must match real manifests.
- File paths and symbols added to docs must exist.

Page-specific checks:

| Page | Checks |
|------|--------|
| `Architecture-Overview.md` | Strategy/family counts, strategy tables, dispatch description, dead paths/symbols |
| `Source-Layout.md` | Directory tree, plugin/pipeline tables, file counts |
| `Testing-and-Validation.md` | Test layers, key files, test counts, E2E manifest count/schema |
| `Traceability-Matrix.md` | Trace IDs, source/test paths, strategy lists, E2E manifest references |
| `Static-Design.md` | Class/interface/method names, diagrams, relationships |
| `Pipeline-Deep-Dive.md` | Dispatch flow, legacy normalization, per-pipeline behavior |

Before committing, spot-check at least three behavioral rewrites by rereading the
source, verify every newly added path exists, and search modified docs for stale
old counts.

## Phase 3: Traceability Audit

Build the current trace state from `website/docs/wiki/Traceability-Matrix.md`.

Search both trace annotation formats:

```bash
rg -n "Trace:|Trace ID:" tests/builder tests/tools --glob "*.py"
rg -n "Trace:|Trace ID:" tests/cpp --glob "*.cpp"
```

Scan E2E manifests and source files:

```bash
find tests/e2e/models/ -name "*.json" | sort
find src/ -name "*.cpp" -o -name "*.h" | sort
find python/tensorrt_model_connect/ -name "*.py" -not -name "__init__.py" | sort
```

Fixes:

- Add missing test trace IDs after reading enough of the test to understand what
  behavior it validates.
- Remove or update orphaned `UT-*`/`IT-*` rows whose files moved or vanished.
- Link unverified `ARCH-*` entries when tests exist; otherwise record a gap.
- Add missing `UD-*` rows for uncovered source files.
- Add missing `IT-E2E-*` rows for unlinked manifests.
- Fix malformed trace IDs.
- Recalculate matrix coverage metrics after edits.

Use specific architecture IDs. Do not default to a catch-all. If no obvious ID
fits, read the source and matrix context before creating or assigning one.

## Validation

Run at minimum:

```bash
git diff --check
```

Then run any documentation generation, link, formatting, or focused tests
available for the files touched. If no doc validation exists, state that
explicitly in the PR body and describe the manual checks performed.

## Summary Report

End each run with:

```text
Doc-Sync Summary (YYYY-MM-DD)
Commits scanned: N (<last>..<current>)
ADR maintenance: <changes or no changes> <PR URL or skipped>
Wiki drift: <changes or no changes> <PR URL or skipped>
Traceability: <changes or no changes> <PR URL or skipped>
Blockers: <none or details>
```
