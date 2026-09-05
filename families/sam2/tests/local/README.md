# SAM2 caller-provisioned L4 qualification

This qualification is intentionally outside `tests/manifests/`. Private
premerge and nightly jobs therefore do not select it or assume access to its
caller-provisioned bundle, five RGB8 frames, or golden evidence.

Run it explicitly from the repository root:

```bash
PYTHONPATH=core/builder:. python3 -m families.sam2.tests.l4_local_runner \
  --probe /absolute/build/sam2_operational_probe \
  --bundle /absolute/artifacts/sam2-l4-local.bundle \
  --runtime-root /absolute/runtime \
  --fixture-dir /absolute/sam2-l4-fixtures
```

Every path is required. Missing, symlinked, malformed, or incomplete inputs
fail the qualification; there is no automated asset lookup or alternate
execution path.
