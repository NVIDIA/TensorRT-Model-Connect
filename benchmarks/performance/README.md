<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Release performance matrix

This matrix compares the public TRTMC operation measured by `trtmc-bench` with the
baseline named in each row of `release.yaml`. The release suite contains exactly one
representative for every ready `(family, operation)` in the benchmark catalog. A catalog
change therefore makes suite validation fail until the new row is reviewed.

## Manual runs

Run one row first:

```bash
python3 tools/perf_release.py benchmarks/performance/release.yaml \
  --case gpt2.generate \
  --output artifacts/perf
```

Run the fast rows, resuming an interrupted campaign:

```bash
python3 tools/perf_release.py benchmarks/performance/release.yaml \
  --priority fast \
  --bundle-cache /path/to/bundle-cache \
  --output artifacts/perf \
  --resume
```

`--priority normal` includes fast and normal rows. `--priority slow` includes the complete
matrix. An explicit `--case` takes precedence over the priority ceiling. `--only trtmc` and
`--only baseline` are debugging modes. `--dry-run` resolves and records both commands without
loading a model.

The command writes only three persistent files:

```text
artifacts/perf/results.json
artifacts/perf/report.html
artifacts/perf/reproduce.py
```

`results.json` is internal evidence and contains raw samples. `report.html` intentionally
contains only green/yellow/red/white categories. Temporary per-backend files are merged into
`results.json` and removed after the run.

Replay a recorded command without the matrix orchestrator:

```bash
python3 artifacts/perf/reproduce.py gpt2.generate
python3 artifacts/perf/reproduce.py gpt2.generate baseline
python3 artifacts/perf/reproduce.py gpt2.generate trtmc --print
```

The generated replay program uses only the Python standard library and directly executes the
recorded `trtmc-bench` or baseline argv.

## CI runs

The `TRTMC Performance Matrix` workflow is callable and manually dispatchable. It runs the
same Python entry point with `--ci`, then uploads the three result files even when a row fails.
Red and yellow comparisons are valid data and do not fail CI. Unexpected command failures,
workload/output mismatches, or an eager fallback from a declared `torch.compile` baseline do.

The self-hosted runner configures the executable, worker, runtime directories, bundle roots,
bundle cache, and Python profile cache through repository variables. Heavy Transformers
dependencies remain in the existing reference Python profiles; they are not added to
`trtmc-bench`.

## Adding or changing a row

Every setting that changes comparison semantics is visible in `release.yaml`: representative
model, priority, measurement count, baseline mode, compile scope, precision/padding or MoE
implementation exceptions, and optional request overrides. Seq2seq rows also declare how
decoder-start/EOS framing maps
to the corresponding TRTMC public result. Generation defaults to exact token-ID equality;
chat-response rows may explicitly require exact public-text equality instead. The exact
manifest-derived request and the resolved
model revision are recorded in `results.json`. Outside a Git checkout, set
`TRTMC_PERF_SOURCE_REVISION` to the tested commit; GitHub Actions supplies `GITHUB_SHA`
automatically.

For example, a row can override its manifest input without changing Python code:

```yaml
- id: gpt2.generate
  model: distilgpt2
  request:
    prompt: The capital of France is
    max_new_tokens: 16
```

The shared `hf-transformers` runner supports encoder, causal-LM, and seq2seq-LM execution.
Models that require another task surface remain explicit white `unsupported` rows until an
aligned runner exists. Do not silently substitute HF eager for a `torch.compile` row; change
the row's `baseline.mode` to `hf-eager` so the report labels it honestly.
