# Architecture Decision Records

No numbered ADRs are currently tracked in this directory. The empty table is
intentional; it does not mean that the project has made no architectural
decisions.

| Number | Title | Status | Date |
| --- | --- | --- | --- |

## Current decision sources

Use the implementation and its tests as the source of truth. The
[AI-Native Horizontal Scaling Architecture](../../architecture/ai-native-horizontal-scaling.md)
records the repository-wide ownership and dependency rules. The most important
concrete boundaries are:

- `families/<family>/support.py` for exact checkpoint ownership and supported
  tasks;
- `families/<family>/model.py` for the plain Python build entrypoint;
- `families/<family>/runtime/` for the family DSO and Task implementation;
- `families/<family>/tests/` for manifests and model-owned validation; and
- `core/` for the deliberately small shared contracts and loaders.

Context pages may preserve why an earlier design was tried or retired. They are
not current contracts unless the implementation and current architecture page
say the same thing.

## When an ADR is useful

Add an ADR only for a durable, repository-wide decision that cannot be
expressed clearly in the owning code and architecture documentation. Routine
family additions, bug fixes, tests, and documentation changes do not need one.

An ADR should contain `Context`, `Decision`, `Alternatives`, and `Consequences`,
and it should land in the same pull request as the decision it records.
