---
name: doc-sync
description: >-
  Use for documentation maintenance scans that keep the canonical website
  journey, API reference, architecture and design, extension guides, feature
  context, ADRs, and traceability status aligned with the current codebase.
  Treat Wiki, Operations, Unit Design, and retired architecture pages only as
  compatibility redirects.
---

# Doc Sync

## Purpose

Scan documentation drift against the implementation and produce focused
GitHub PRs. Process these phases in order:

1. Canonical documentation: keep the six website sections and newcomer path
   accurate.
2. Architecture context and traceability: maintain ADRs and evidence status
   without upgrading incomplete evidence into a compliance claim.
3. Compatibility redirects: preserve retired routes without restoring a
   second source of truth.

Do not guess based on filenames. Read the implementation, tests, manifests,
and live command help before changing behavioral prose or command examples.

## Canonical Information Architecture

The maintained website sources are:

| Section | Source |
| --- | --- |
| Getting Started | `website/docs/getting-started/` |
| Learn & Tutorials | `website/docs/learning-path.md`, `website/docs/tutorials/` |
| API Reference | `website/docs/api/` |
| Architecture & Design | `website/docs/architecture/` |
| Contribute & Extend | `website/docs/extend/` |
| Feature Reference & Context | `website/docs/features/`, `website/docs/context/`, `website/docs/reference/` |

The navigation in `website/sidebars.js` and `website/docusaurus.config.js`
must expose those sources consistently. `website/docs/wiki/`,
`website/docs/operations/`, `website/docs/unit-design/`, and retired
architecture pages are compatibility routes, not maintained content sources.

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
PHASE_NAME="canonical"  # or "context" or "redirects"
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
  --body-file pr-body.md \
  --label documentation
```

Creating or pushing the PR does not start premerge. After confirming that the
PR's `headRefOid` equals the pushed SHA, follow `$submit-github-pr` and
`$pr-babysitter`: an authorized collaborator applies the one-shot
`run-internal-ci` label, and the exact head must receive a successful
`trtmc/premerge/required` status. Never use the retired `run-ci` label or
publish private Internal CI logs, artifacts, runner details, or URLs in Source.

Use `$write-git-messages` for the commit and PR text.

## Phase 1: Canonical Documentation

Start from current repository ground truth:

```bash
find python/tensorrt_model_connect/families/ -name "MODEL.toml" | sort
find src/runtime/models/ -name "MODEL.toml" | sort
find tests/e2e/models/ -path "*/manifests/*.json" | sort
find src/runtime/registry/ -type f \( -name "*.cpp" -o -name "*.h" \) | sort
find src/runtime/domains/ -type f \( -name "*.cpp" -o -name "*.h" \) | sort
find include/trtmc/ -type f -name "*.h" | sort
python3 tools/model_ci.py validate
python3 tools/test_impact.py --validate
```

Check the newcomer journey in order:

1. `getting-started/environment-and-repro.md` states supported host,
   container, Python, CUDA, TensorRT, and hardware assumptions.
2. `getting-started/installation.md` installs the prerequisites used by the
   first inference.
3. `getting-started/quick-start.md` contains one canonical NLP build,
   inspection, and inference flow.
4. `learning-path.md` continues from that exact result instead of restarting
   with a competing quick start.

Run every changed command locally when the required dependencies and hardware
are available. Otherwise validate the parser or `--help`, inspect the
implementation that consumes the argument, and record the missing runtime or
hardware proof explicitly.

Review the remaining canonical sections:

| Section | Checks |
| --- | --- |
| API Reference | Public Python, CLI, and C++ names and signatures match `python/`, `include/trtmc/`, and the live help output |
| Architecture & Design | Blocks, ownership, build pipeline, bundle format, runtime lifecycle, and validation diagrams match the implementation |
| Contribute & Extend | Model, runtime-strategy, optimized-runtime, schema, and validation guides use current extension points and tests |
| Feature Reference | Feature status, ownership, limitations, and historical context are clearly distinguished |
| Reference | Source layout, testing commands, benchmarking, and profiling paths exist and remain reproducible |

Behavioral descriptions require reading the implementation first. A parser
accepting an option is not evidence that every family supports or qualifies
the feature.

## Phase 2: Architecture Context And Traceability

### ADR maintenance

Scan `website/docs/context/adr/*.md` excluding `README.md`.

- Parse frontmatter: `number`, `title`, `status`, `date`, `source_commits`,
  and `superseded_by`.
- Verify referenced paths, symbols, strategies, and counts against the current
  tree.
- For a moved path, set a concrete value such as
  `OLD_PATH=src/runtime/old_file.cpp`, then run
  `git log --follow --diff-filter=R -- "$OLD_PATH"`.
- Promote `Proposed` ADRs only when the decision is clearly implemented.
- Mark an ADR `Superseded` only when a maintained ADR names the replacement.
- If a `source_commits` SHA is not an ancestor of GitHub `main`, document the
  provenance problem rather than silently rewriting history.
- Rebuild `website/docs/context/adr/README.md` from frontmatter, sorted by
  number.

### Traceability and safety status

`website/docs/context/traceability-and-safety.md` is the canonical status and
gap report. It is not a normative traceability matrix or a safety case.

Recompute its snapshot from the manifests and test annotations:

```bash
find tests/e2e/models/ -path "*/manifests/*.json" | sort
rg -n "Trace:|Trace ID:" tests/builder tests/tools --glob "*.py"
rg -n "Trace:|Trace ID:" tests/cpp --glob "*.cpp"
```

Verify manifest count, testcase count, empty trace IDs, duplicate IDs, source
revision, and date. Keep evidence tiers separate: an implemented check, a
passing CPU/static test, a GPU E2E result, qualified performance, and a
certification claim are not interchangeable.

Do not add or reassign source/test trace IDs as part of a documentation-only
PR. Report those implementation and test gaps for a separately reviewed
change.

## Phase 3: Compatibility Redirects

Audit every Markdown file below:

```text
website/docs/wiki/
website/docs/operations/
website/docs/unit-design/
```

Also audit retired canonical paths such as
`website/docs/architecture/runtime-plugins.md`.

Each compatibility page must:

- set `unlisted: true`;
- render a real Docusaurus `<Redirect>` to an existing canonical route;
- avoid navigation, pagination, diagrams, or maintained explanatory content;
- contain no unique factual or policy source that is absent from its
  destination; and
- build successfully so old external links do not become 404s.

If a test or tool still consumes a retired page as normative input, migrate
that consumer to the canonical page in the same logical change. Until that
consumer is migrated, report the retired route as a compatibility blocker
rather than treating its duplicated text as canonical.

Do not add new pages under the retired directories.

## Validation

Run at minimum:

```bash
git diff --check
python3 tools/check_doc_file_references.py --strict website/docs
python3 tools/model_ci.py validate
python3 tools/test_impact.py --validate
npm --prefix website ci
npm --prefix website run test:model-support
npm --prefix website run build
```

Run focused Python or C++ tests for commands, contracts, tests, and tooling
touched by the documentation change. Use the Node version and GitHub Pages
environment documented in `website/docs/reference/testing.md` when reproducing
the production website build.

A green Docusaurus build proves that the site compiles; it does not prove GPU
inference, parity, output quality, performance, or model qualification. Record
what ran, the exact revision, and every environment boundary in the PR body.

## Summary Report

End each run with:

```text
Doc-Sync Summary (YYYY-MM-DD)
Commits scanned: N (<last>..<current>)
Canonical docs: <changes or no changes> <PR URL or skipped>
Architecture context: <changes or no changes> <PR URL or skipped>
Redirect compatibility: <changes or no changes> <PR URL or skipped>
Command evidence: <tests and runtime boundaries>
Blockers: <none or details>
```
