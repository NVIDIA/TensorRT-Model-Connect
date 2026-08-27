# Protected CI failure report prototype

This package builds the limited information that protected CI may disclose
after a failed run. The same validator and renderer are used by the local CLI
and the trusted Source relay workflow.

The pipeline is deterministic:

1. `export_failure` constructs a new object from explicitly approved fields.
2. `validate_public_failure` enforces the closed `public-failure-v1` contract.
3. `render_failure_report` produces one deterministic UTF-8 text log.
4. `assert_public_payload_safe` scans the JSON and rendered text as a final
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

The command writes `public-failure.json` and `public-failure.log` only to the
selected local directory. The text log is the user-facing representation; no
HTML renderer is required. It does not publish either file.

The synthetic input intentionally contains fake internal-looking values. The
tests prove that adding those unknown fields does not change the serialized
public output.

## Integration boundary

The Source relay workflow accepts only the closed report contract, revalidates
it on the default branch, prints `public-failure.log` into a public Actions log,
and updates the existing automated status context for the exact PR head. The
private finalizer must send already structured fields; raw protected logs are
never accepted by the relay.
