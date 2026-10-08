# Family case migrations

When a model moves to another family, keep its validation cases under the new
family and declare the move in `families/<new-family>/tests/migrations/<old-family>.json`:

```json
{
  "schema_version": "trtmc.family-case-migration/v1",
  "from_family": "old_family",
  "cases": ["existing-case-a", "existing-case-b"]
}
```

List every case declared by the old family in the protected base. The directory
identifies the new owner; the filename must match `from_family`. A declaration
cannot split one old family across multiple new owners.

The premerge verifier requires the old family to be gone, each listed case to
have exactly one new owner, and protected case, semantic, premerge, and
single-GPU nightly coverage to remain intact. Missing or extra cases and
undeclared moves fail. Keep the declaration after merging: it becomes a dormant
record once the new family is in the protected base.
