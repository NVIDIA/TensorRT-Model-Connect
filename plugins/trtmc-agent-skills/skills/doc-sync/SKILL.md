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

Resolve the configured remote that points at the canonical GitHub repository.
Prefer the repo-standard `github` name, but support checkouts where the same
repository is configured only as `origin`:

```bash
DOC_SYNC_REMOTE="github"
if ! git remote get-url "$DOC_SYNC_REMOTE" >/dev/null 2>&1; then
  DOC_SYNC_REMOTE="origin"
fi
DOC_SYNC_REMOTE_URL=$(git remote get-url "$DOC_SYNC_REMOTE")
case "$DOC_SYNC_REMOTE_URL" in
  https://github.com/NVIDIA/TensorRT-Model-Connect|\
  https://github.com/NVIDIA/TensorRT-Model-Connect.git|\
  git@github.com:NVIDIA/TensorRT-Model-Connect.git|\
  ssh://git@github.com/NVIDIA/TensorRT-Model-Connect.git) ;;
  *)
    echo "Refusing non-canonical remote: $DOC_SYNC_REMOTE_URL" >&2
    exit 1
    ;;
esac

git fetch "$DOC_SYNC_REMOTE" main
CURRENT_SHA=$(git rev-parse "$DOC_SYNC_REMOTE/main")
```

Read the last scan marker only after fetching, and accept it only when it names
a commit in the current canonical-main history:

```bash
DOC_SYNC_MARKER="website/docs/context/.last_scan_sha"
LAST_SHA=""
if [ -s "$DOC_SYNC_MARKER" ]; then
  read -r LAST_SHA < "$DOC_SYNC_MARKER"
  if ! git cat-file -e "$LAST_SHA^{commit}" 2>/dev/null ||
     ! git merge-base --is-ancestor "$LAST_SHA" "$CURRENT_SHA"; then
    echo "Ignoring invalid or unrelated doc-sync marker: $LAST_SHA" >&2
    LAST_SHA=""
  fi
fi
```

For a first scan or invalid marker, use the latest commit at or before the
seven-day boundary. If the repository is younger than that, use its root
commit:

```bash
if [ -z "$LAST_SHA" ]; then
  LAST_SHA=$(git rev-list --max-count=1 --before="7 days ago" "$CURRENT_SHA")
fi
if [ -z "$LAST_SHA" ]; then
  LAST_SHA=$(git rev-list --max-parents=0 "$CURRENT_SHA" | tail -n 1)
fi
```

If there is no change set, update the marker only when that marker update is
part of the requested maintenance work. Otherwise report nothing to do.

Inspect changes:

```bash
git log --oneline "$LAST_SHA".."$CURRENT_SHA"
git diff --name-only "$LAST_SHA".."$CURRENT_SHA"
```

Do not advance the marker until all requested phases and validations succeed.
When a marker update is explicitly in scope, write the exact scanned SHA:

```bash
printf '%s\n' "$CURRENT_SHA" > "$DOC_SYNC_MARKER"
```

## Branch And PR Flow

Create one branch per phase that produces changes:

```bash
PHASE_NAME="adr"  # or "wiki" or "traceability"
DATE=$(date +%Y-%m-%d)
BRANCH="doc-sync/${PHASE_NAME}-${DATE}"

git fetch "$DOC_SYNC_REMOTE" main
git switch -c "$BRANCH" "$DOC_SYNC_REMOTE/main"
```

Skip a phase if the remote branch already exists:

```bash
git ls-remote --heads "$DOC_SYNC_REMOTE" "$BRANCH"
```

Push and open a GitHub PR:

```bash
git push -u "$DOC_SYNC_REMOTE" HEAD
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
- Verify backtick-quoted file paths such as `src/...`,
  `python/tensorrt_model_connect/...`, `tests/...`, and `include/...` exist.
  For moved paths, use `git log --follow --diff-filter=R -- <old_path>`.
- Verify backtick-quoted class, function, and strategy names exist.
- Update numeric claims while keeping inventories distinct: native runtime
  strategy/family/E2E-manifest counts versus exact qualified optimized
  implementation/profile counts.
- Promote old `Proposed` ADRs to `Accepted` only when the described decision is
  clearly implemented.
- Mark ADRs `Superseded` when a newer ADR describes the replacement.
- If a `source_commits` SHA is not an ancestor of current `main`, mark the ADR
  `Deprecated` with a short note.
- Rebuild `website/docs/context/adr/README.md` from ADR frontmatter, sorted by
  number.

Ground truth commands:

```bash
rg -n "PluginRegistrar|trtmc_register_model_plugin" \
  include/trtmc/runtime src/runtime/registry src/runtime/models \
  src/runtime/providers \
  --glob "*.h" --glob "*.cpp"
find python/tensorrt_model_connect/families -mindepth 2 -maxdepth 2 \
  \( -name "MODEL.toml" -o -name "plugin.py" \) | sort
find src/runtime/models -mindepth 2 -maxdepth 2 \
  \( -name "MODEL.toml" -o -name "plugin.cpp" \) | sort
find tests/e2e/models -mindepth 3 -maxdepth 3 \
  -path "*/manifests/*.json" | sort
find python/tensorrt_model_connect/families -mindepth 3 -maxdepth 3 \
  -name "IMPLEMENTATION.toml" | sort
find python/tensorrt_model_connect/families -mindepth 4 -maxdepth 4 \
  -path "*/profiles/*.toml" | sort
find tests/e2e/models -mindepth 3 -maxdepth 3 \
  -name "QUALIFICATION.*.toml" | sort
```

The three `MODEL.toml` roots and `runtime_strategy` inventory define native
support. A delegated optimized implementation for an existing family instead
uses its family-local `IMPLEMENTATION.toml`, exact profile TOMLs, isolated
adapter/runtime DSO, and `QUALIFICATION.*.toml`; do not require or document a
synthetic native strategy for it.

PR body should summarize changed ADR numbers, status changes, index updates, and
any deprecated/superseded rationale.

## Phase 2: Wiki Drift Repair

Scan all markdown files in `website/docs/wiki/`.

Collect reusable ground truth first:

```bash
# One family root is counted once even though it normally owns both files.
find python/tensorrt_model_connect/families -mindepth 2 -maxdepth 2 \
  \( -name "MODEL.toml" -o -name "plugin.py" \) \
  -printf '%h\n' | sort -u

find src/runtime/models -mindepth 2 -maxdepth 2 -name "MODEL.toml" | sort
find tests/e2e/models -mindepth 3 -maxdepth 3 \
  -path "*/manifests/*.json" | sort
find python/tensorrt_model_connect/families -mindepth 3 -maxdepth 3 \
  -name "IMPLEMENTATION.toml" | sort
find python/tensorrt_model_connect/families -mindepth 4 -maxdepth 4 \
  -path "*/profiles/*.toml" | sort
find tests/e2e/models -mindepth 3 -maxdepth 3 \
  -name "QUALIFICATION.*.toml" | sort
find src/runtime/models src/runtime/registry src/runtime/providers -type f \
  \( -name "*.h" -o -name "*.cpp" \) | sort
find tests/builder tests/tools -type f -name "test_*.py" | sort
find tests/cpp -type f -name "test_*.cpp" | sort
find src include -type f \( -name "*.cpp" -o -name "*.h" \) | sort
find python/tensorrt_model_connect -type f -name "*.py" | sort
```

Compute count claims from the same authoritative sets:

```bash
find python/tensorrt_model_connect/families -mindepth 2 -maxdepth 2 \
  \( -name "MODEL.toml" -o -name "plugin.py" \) \
  -printf '%h\n' | sort -u | wc -l
find src/runtime/models -mindepth 2 -maxdepth 2 -name "MODEL.toml" | wc -l
find tests/e2e/models -mindepth 3 -maxdepth 3 \
  -path "*/manifests/*.json" | wc -l
find python/tensorrt_model_connect/families -mindepth 3 -maxdepth 3 \
  -name "IMPLEMENTATION.toml" | wc -l
find python/tensorrt_model_connect/families -mindepth 4 -maxdepth 4 \
  -path "*/profiles/*.toml" | wc -l
find tests/e2e/models -mindepth 3 -maxdepth 3 \
  -name "QUALIFICATION.*.toml" | wc -l
find tests/builder -type f -name "test_*.py" | wc -l
find tests/tools -type f -name "test_*.py" | wc -l
find tests/cpp -type f -name "test_*.cpp" | wc -l
```

Rules:

- Counts and file paths can be fixed mechanically.
- Behavioral descriptions require reading the implementation first.
- Strategy tables must match the actual registry.
- Native family/plugin lists must match the three current `MODEL.toml` roots.
- Native E2E examples must match real JSON manifests. Optimized examples must
  separately match `IMPLEMENTATION.toml`, profile TOMLs, and
  `QUALIFICATION.*.toml`.
- Python discovery descriptions must keep the three actual flows distinct:
  a full config tries architecture-pattern descriptor candidates and then the
  all-package `pkgutil` compatibility fallback; a string or `model_type` tries
  direct descriptor ID, alias/prefix candidates, then that fallback; a
  Diffusers pipeline class uses descriptor `diffusion_pipeline_classes` only
  and never falls back to `pkgutil`. Descriptor routes import the package-level
  `plugin` exported by `__init__.py`; descriptor `module` is
  specialization/tooling metadata, not an arbitrary runtime import selector.
- Build descriptions must state that optimized discovery is bounded to the
  selected family, zero claims continue to native, and a selected adapter's
  failure is terminal.
- Optimized bundles use `optimized_runtime.json` and an embedded
  `libtrtmc_impl_*.so`; `runtime_strategy` may be empty.
- File paths and symbols added to docs must exist.

Page-specific checks:

| Page | Checks |
|------|--------|
| `Architecture-Overview.md` | Strategy/family counts, strategy tables, dispatch description, dead paths/symbols |
| `Architecture-Extensibility-Assessment.md` | Native three-descriptor ownership versus exact optimized implementation/profile ownership; full-config, string/model-type, and Diffusers discovery flows |
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
find tests/e2e/models -mindepth 3 -maxdepth 3 \
  -path "*/manifests/*.json" | sort
find python/tensorrt_model_connect/families -mindepth 3 -maxdepth 3 \
  -name "IMPLEMENTATION.toml" | sort
find python/tensorrt_model_connect/families -mindepth 4 -maxdepth 4 \
  -path "*/profiles/*.toml" | sort
find tests/e2e/models -mindepth 3 -maxdepth 3 \
  -name "QUALIFICATION.*.toml" | sort
find src/runtime/providers -type f \( -name "*.cpp" -o -name "*.h" \) | sort
find src -type f \( -name "*.cpp" -o -name "*.h" \) | sort
find python/tensorrt_model_connect -type f -name "*.py" \
  -not -name "__init__.py" | sort
```

Fixes:

- Add missing test trace IDs after reading enough of the test to understand what
  behavior it validates.
- Remove or update orphaned `UT-*`/`IT-*` rows whose files moved or vanished.
- Link unverified `ARCH-*` entries when tests exist; otherwise record a gap.
- Add missing `UD-*` rows for uncovered source files.
- Add missing `IT-E2E-*` rows for unlinked manifests.
- Link optimized implementation/profile behavior to its capsule contract tests
  and producer qualification descriptor; do not count those descriptors as
  native E2E JSON manifests.
- Fix malformed trace IDs.
- Recalculate matrix coverage metrics after edits.

Use specific architecture IDs. Do not default to a catch-all. If no obvious ID
fits, read the source and matrix context before creating or assigning one.

## Validation

The repository has a required documentation workflow at
`.github/workflows/docs-validation.yml`; do not describe documentation
validation as absent or optional. Reproduce every gate from the repository
root:

```bash
PYTHONPATH=python:. python3 -m pytest \
  tests/tools/test_check_doc_file_references.py \
  tests/tools/test_check_doc_commands.py \
  tests/tools/test_runtime_strategy_matrix_checker.py \
  tests/tools/test_model_owned_validation_scripts.py \
  tests/tools/test_test_impact.py \
  tests/tools/test_trtmc_validate.py \
  tests/tools/test_perf_matrix.py::test_release_suite_covers_every_non_l0_ready_model_profile \
  -q
PYTHONPATH=python:. python3 tools/test_impact.py --validate
python3 tools/check_doc_file_references.py --strict --tracked
python3 tools/check_doc_commands.py
PYTHONPATH=python:. python3 tools/check_runtime_strategy_matrix.py
npm --prefix website ci
npm --prefix website run build
git diff --check
```

Install `pytest` and `PyYAML` first if the validation environment does not
provide them. Use Node 20 for the website commands to match CI.

The pytest bundle protects the file-reference and command checkers, the live
runtime-strategy matrix, model-owned validation scripts, selective-test
classification, canonical reference-consistency and report behavior, and
release-suite coverage for every non-L0-ready model profile.
`tools/test_impact.py --validate` protects selective-test ownership; the strict
reference checker verifies tracked paths and numeric claims; the command
checker validates documented shell syntax and known argument contracts; the
matrix checker compares strategies with descriptors, source, tests, and runner
commands; and the clean install plus production build validates the complete
Docusaurus site. `git diff --check` is an additional local patch-quality check,
not a step in the GitHub workflow. Use Node 20 to match CI. Run additional
focused tests when a behavioral rewrite depends on another contract, and
record any environment-dependent gate that could not run rather than calling
the mandatory workflow nonexistent.

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
