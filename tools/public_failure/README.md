# Protected CI failure report prototype

This package builds a local preview of the limited information that protected
CI may disclose after a failed run. It is a P0/shadow implementation: no Source
workflow imports it, and it has no storage, GitHub status, or PR comment client.

The pipeline is deterministic:

1. `export_failure` constructs a new object from explicitly approved fields.
2. `validate_public_failure` enforces the closed `public-failure-v1` contract.
3. `render_failure_report` produces one script-free, self-contained HTML file.
4. `assert_public_payload_safe` scans the JSON and decoded HTML as a final
   defense-in-depth check.

Unknown input fields are ignored. Unknown names and unsafe test IDs become
fixed placeholders. Raw logs, commands, stack traces, environment data, paths,
URLs, and arbitrary messages are not part of the contract.

## Local preview

Generate the synthetic review sample:

```bash
python3 -m tools.public_failure \
  --input tests/tools/fixtures/public_failure/internal-failure.json \
  --context tests/tools/fixtures/public_failure/context.json \
  --output-dir /tmp/trtmc-public-failure-preview
```

The command writes `public-failure.json` and `report.html` only to the selected
local directory. It does not publish either file.

The synthetic input intentionally contains fake internal-looking values. The
tests prove that adding those unknown fields does not change the serialized
public output.

## Integration boundary

A future private-CI finalizer may call `build_failure_artifacts` from a pinned,
merged Source revision. Upload, anonymous verification, stale-SHA checks,
required-status updates, and failure-comment upsert belong to that private
integration and are deliberately absent here.
