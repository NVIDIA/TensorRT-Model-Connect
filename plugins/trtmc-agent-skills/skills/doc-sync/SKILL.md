---
name: doc-sync
description: >-
  Use for documentation maintenance scans that keep the canonical website
  journey, repo-local skills, commands, API reference, architecture and
  design, extension guides, feature context, ADRs, and traceability status
  aligned with the current GitHub main branch. Covers the Source/Internal CI
  boundary and model-owned validation without exposing private evidence.
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

Keep evidence tiers distinct:

1. a path, symbol, option, or command exists;
2. static and documentation checks pass;
3. CPU tests pass;
4. target-hardware execution passes; and
5. model parity, performance, or qualification is proven.

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

Fetch current GitHub main:

```bash
git fetch github main
DOC_SYNC_CURRENT_SHA=$(git rev-parse github/main)
```

If `website/docs/context/.last_scan_sha` exists, use it only after proving that
it names a commit and is an ancestor of the current main SHA:

```bash
DOC_SYNC_LAST_SHA=$(cat website/docs/context/.last_scan_sha)
git cat-file -e "${DOC_SYNC_LAST_SHA}^{commit}"
git merge-base --is-ancestor \
  "$DOC_SYNC_LAST_SHA" "$DOC_SYNC_CURRENT_SHA"
git diff --name-status \
  "$DOC_SYNC_LAST_SHA".."$DOC_SYNC_CURRENT_SHA"
```

When the marker is missing or invalid, choose and report an explicit baseline.
Do not silently substitute a date heuristic. Update the marker only when that
state change was requested and the scan completed. If there is no change set,
report that result instead of inventing documentation churn.

## Route Changed Surfaces

| Source of truth | Documentation consumers |
| --- | --- |
| `.github/workflows/internal-ci-bridge.yml`, `tools/ci/README.md` | premerge trigger and public/private evidence boundary |
| `.github/workflows/legal.yml`, `.github/workflows/pages.yml` | retained Source workflows |
| Python family `MODEL.toml` and family code | build selection and family behavior |
| C++ model `MODEL.toml`, registry, and runtime code | native strategies and dispatch |
| E2E `MODEL.toml`, manifests, and sidecars | model support and validation |
| `tests/validation/*.yaml`, `tools/validation/` | validation workloads and engine |
| `benchmarks/performance/*.yaml`, model perf profiles | performance contracts and reports |
| public headers and executable `--help` | API and CLI references |
| `plugins/trtmc-agent-skills/skills/` | repo-local automation guidance |

Current model ownership spans:

- `python/tensorrt_model_connect/families/<family>/`;
- `src/runtime/models/<family>/`; and
- `tests/e2e/models/<family>/`.

Do not resurrect removed Source workflows, retired task-evaluation commands,
root graph helpers, or paths copied from another branch.

## CI Documentation Contract

Source contains only the Internal CI Bridge, Legal Compliance, and Pages
workflows. Premerge and nightly orchestration are private Internal CI.

- `run-internal-ci` is the one-shot trusted trigger; `run-ci` is retired.
- The bridge verifies event, PR metadata, and source branch heads before
  dispatching the exact PR head.
- Source receives only `trtmc/premerge/required` as `PENDING`, `PASS`, or
  `FAIL` on that exact head.
- A successful bridge dispatch is not a successful premerge result.
- Never copy private logs, artifacts, package coordinates, runner details, or
  internal URLs into Source docs, Actions, Pages, or PR comments.

Read the workflow and `tools/ci/README.md` at the target SHA rather than
hard-coding private job names or topology.

## Scan Classes

For paths, symbols, counts, and commands:

- verify repository-relative paths on disk;
- verify symbols, manifest fields, runtime strategies, and CLI options in code
  or executable help;
- recompute counts and label snapshot values as such;
- execute safe `--help`, `--list`, `--dry-run`, and parser checks; and
- search modified docs and skills for removed names and paths.

```bash
python3 tools/check_doc_file_references.py --strict <changed-doc-directory>
```

For model-support claims, cross-check all three ownership descriptors and the
exact bundle/runtime path. Registration, build success, dry-run planning,
parity, performance, and qualification are different claims.

The supported Dev/QA entry point is `tools/trtmc_validate.py`. Workloads live
in `tests/validation/workloads.yaml`, bindings in
`tests/validation/model_workloads.yaml`, and implementation in
`tools/validation/`. The persisted `task_eval` artifact key is intentional; do
not describe it as a live legacy CLI.

## Branch And PR Flow

Create one short-lived branch per coherent documentation owner or review
concern. Split phases when their ownership or validation differs; keep them
together when one focused review can cover the change:

```bash
DOC_SYNC_PHASE_NAME="canonical"  # or "context" or "redirects"
DOC_SYNC_DATE=$(date +%Y-%m-%d)
DOC_SYNC_BRANCH="doc-sync/${DOC_SYNC_PHASE_NAME}-${DOC_SYNC_DATE}"

git fetch github main
git switch -c "$DOC_SYNC_BRANCH" github/main
```

Skip a phase if the remote branch already exists:

```bash
git ls-remote --heads github "$DOC_SYNC_BRANCH"
```

Push and open a GitHub PR:

```bash
git push -u github HEAD
gh pr create \
  --repo NVIDIA/TensorRT-Model-Connect \
  --base main \
  --head "$DOC_SYNC_BRANCH" \
  --title "docs(${DOC_SYNC_PHASE_NAME}): automated doc sync $(date +%Y-%m-%d)" \
  --body-file pr-body.md
```

Do not attach labels, update scan markers, create issues, or publish private
evidence unless the user requested that state change.

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
python3 tools/check_doc_file_references.py \
  --strict plugins/trtmc-agent-skills/skills/doc-sync
python3 tools/model_ci.py validate
python3 tools/test_impact.py --validate
npm --prefix website ci
npm --prefix website run test:model-support
npm --prefix website run build
```

For a broad documentation contract change, the representative focused Python
set is:

```bash
PYTHONPATH=python:. python3 -m pytest \
  tests/tools/test_check_doc_file_references.py \
  tests/tools/test_github_actions_ci.py \
  tests/tools/test_runtime_strategy_matrix_checker.py \
  tests/tools/test_model_owned_validation_scripts.py \
  tests/tools/test_validation_engine.py \
  tests/tools/test_test_impact.py \
  tests/tools/test_trtmc_validate.py \
  tests/tools/test_perf_matrix.py::test_release_suite_covers_every_non_l0_ready_model_profile \
  -q
```

Run focused Python or C++ tests for commands, contracts, tests, and tooling
touched by the documentation change. Use the Node version and GitHub Pages
environment documented in `website/docs/reference/testing.md` when reproducing
the production website build.

A green Docusaurus build proves that the site compiles; it does not prove GPU
inference, parity, output quality, performance, or model qualification. Record
what ran, the exact revision, and every environment boundary in the PR body.
`npm ci` needs network access and changes the local dependency tree; report
missing dependencies, baseline failures, and unrun evidence tiers explicitly.

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

State the baseline and target SHA, corrected facts and source paths, exact
checks and outcomes, unrun evidence tiers, known baseline gaps, and PR URLs.
“No change” is valid when the current documentation matches current evidence.

<!-- Collaborative review anchor. -->
