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

For a non-text row the baseline command is equally explicit. For example:

```bash
python3 tools/perf_release.py benchmarks/performance/release.yaml \
  --case whisper.transcribe \
  --output artifacts/perf \
  --dry-run
python3 artifacts/perf/reproduce.py whisper.transcribe baseline --print
```

Remove `--print` to execute that baseline command without the matrix orchestrator. The command
contains the adapter, model, manifest, exact resolved request JSON, precision, warmup/iteration
counts, resolved runtime JSON, workload digest, and output path.

## CI runs

The `TRTMC Performance Matrix` workflow is callable and manually dispatchable. It runs the
same Python entry point with `--ci`, then uploads the three result files even when a row fails.
Red and yellow comparisons are valid data and do not fail CI. Unexpected command failures,
workload/output mismatches, or an eager fallback from a declared `torch.compile` baseline do.

The self-hosted runner configures the executable, worker, runtime directories, bundle roots,
bundle cache, and Python profile cache through repository variables. Reference dependencies
stay in the separate baseline process and its selected Python profile or upstream checkout;
they are not added to `trtmc-bench`.

The baseline process is deliberately separate from `trtmc-bench`. Text rows use the shared
`hf-transformers` runner. Task rows use one of the explicit `task-reference` adapters for
Diffusers, ASR, TTS, VLM, embedding, reranking, vision, time series, Qwen3-Omni, PersonaPlex,
ELF, or Lance. Each task result records whether preprocessing is inside the timed call. Model
loading and warmup are always outside measured samples.

The complete slow matrix expects the same reference prerequisites as model E2E validation.
Special upstream code locations are supplied without changing `release.yaml`:

```text
TRTMC_ELF_REFERENCE_REPO=/path/to/pinned/ELF/checkout
TRTMC_LANCE_REFERENCE_REPO=/path/to/bytedance/Lance/checkout
PERSONAPLEX_OFFICIAL_REPO=/path/to/pinned/personaplex/moshi/checkout
```

The ELF and Lance checkout commits and resolved Hugging Face snapshot revisions are recorded in
`results.json`. CI should prebuild the declared Python profiles and set
`TRTMC_PYTHON_PROFILE_PREBUILT_ONLY=1`; this keeps dependency installation out of a measured
campaign and fails immediately when a required profile is missing.

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

The shared `hf-transformers` runner supports encoder, causal-LM, and seq2seq-LM execution. All
other release rows declare a `task-reference` adapter and `reference_backend`; the suite does
not accept `unsupported` placeholders. When adding a new family operation, either select an
existing adapter or add a focused adapter and its contract test. Do not silently substitute HF
eager for a `torch.compile` row; change the row's `baseline.mode` to `hf-eager` so the report
labels it honestly.
