# Architecture Decision Records

No numbered ADRs are currently tracked in this directory. The empty table is
intentional; it is not evidence that the project has made no architectural
decisions.

| Number | Title | Status | Date |
|--------|-------|--------|------|

## Current decision sources

Until a decision is captured as an ADR, use the implementation and its tests as
the source of truth. The maintained
[architecture overview](../../architecture/overview.md) describes system
boundaries, and the [unit-design overview](../../unit-design/overview.md)
links to component-level contracts. Family-owned `MODEL.toml` descriptors,
runtime registrations, E2E manifests, and the Git history provide the
authoritative details for a particular model path. Status-labeled documents in
`website/docs/context/` may explain active plans or investigations, but they
are not accepted ADRs.

## When to add an ADR

The repository's
`plugins/trtmc-agent-skills/skills/submit-github-pr/SKILL.md` workflow requires
an ADR when a change introduces or substantially changes a runtime strategy,
family plugin, runtime pipeline class, persisted config schema, E2E task
strategy, comparator/reference mechanism, or broad architectural contract. Do
not create an ADR for routine bug fixes, dependency bumps, docs-only changes,
tests without architectural impact, or another manifest for an existing
family.

## Add an ADR with the PR workflow

1. Run the repo-local `$submit-github-pr` skill and perform its ADR check
   against the resolved GitHub `main`.
2. Determine the next four-digit number:

   ```bash
   LAST_NUM=$(ls website/docs/context/adr/[0-9]*.md 2>/dev/null | sort -V | tail -1 | grep -oE '[0-9]{4}' || true)
   NEXT_NUM=$(printf "%04d" $((10#${LAST_NUM:-0} + 1)))
   ```

3. Create `website/docs/context/adr/${NEXT_NUM}-<slug>.md` with frontmatter
   fields `number`, `title`, `status: Proposed`, `date`, and
   `source_commits`. Include `Context`, `Decision`, `Considered Alternatives`,
   and `Consequences` sections.
4. Add the ADR to the table above, including its status and date.
5. Keep the ADR on the same short-lived branch as the architectural change,
   validate the docs, and include it before opening the pull request to
   GitHub `main`. Do not push directly to `main`.

The `doc-sync` and `submit-github-pr` skills maintain this index as part of
their documentation and PR workflows.
