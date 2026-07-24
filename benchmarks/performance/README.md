<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Release performance matrix

This matrix compares the public TRTMC operation measured by `trtmc-bench` with the
baseline named in each row of `release.yaml`. The release suite contains exactly one
representative for every ready `(family, operation)` in the benchmark catalog. The current
suite has 79 family-operation comparisons across 78 unique families because `eagle_vlm`
exposes both `embed` and `rerank`. A catalog change makes suite validation fail until the
new row is reviewed.

## Timing contract

The reference implementation defines the measured boundary for each row. The suite records
that boundary plus two explicit booleans: whether input preparation and external asset loading
are timed. TRTMC then selects one of two matching scopes:

- `public_pipeline_call_wall` measures the public pipeline call, including its preprocessing
  and returned output.
- `model_call_wall` starts at the first TensorRT module invocation and ends when the public
  operation returns. This excludes pipeline preprocessing prepared before the model call.

Timing alignment is fail-closed. Suite loading checks every row against the shared reference
contract registry. Before any benchmark command executes, `trtmc-bench --dry-run` must resolve
the requested TRTMC scope and asset-loading policy for every selected row. The baseline runner
checks its implemented boundary before collecting samples, and final classification checks
the recorded policy from both processes. A mismatch produces no performance light.

The candidate worker is also checked before execution. Its `--metadata` response must identify
a `Release` build from the exact source revision being reported. Configure source builds with
`cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`; stale, Debug, and unlabelled workers are
rejected instead of producing performance evidence.

Non-text rows declare an output-shape contract in `release.yaml`. Segmentation compares the
materialized mask count and source geometry, audio compares generated sample count and sample
rate, and image/video generation compares frame count and geometry. These contracts prevent
raw and postprocessed outputs, or different generation lengths, from receiving a traffic light.

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

To carry forward already-aligned evidence after a timing-contract correction, seed a new
campaign from the earlier result:

```bash
python3 tools/perf_release.py benchmarks/performance/release.yaml \
  --priority slow \
  --reuse-aligned-from artifacts/previous/results.json \
  --output artifacts/perf
```

This mode preflights the complete current matrix. It reuses a terminal row only when the
resolved workload, measurement settings, non-timing baseline settings, TRTMC boundary, and
reference boundary all still match. Every other row runs again. The new `results.json`
records the source hash, reused row IDs, and the reason each non-reused row was rerun.

The command writes only two persistent files:

```text
artifacts/perf/results.json
artifacts/perf/report.html
```

`results.json` is internal evidence and contains raw samples. `report.html` shows the p50
wall time calculated from those samples, the sample count, the green/yellow/red/white category,
and the validated timing description for each side. The description states the exact measured
call boundary and the work included and excluded. A side without samples is labeled
`No timing result`. The report does not expose the individual raw samples. Temporary
per-backend files are merged into `results.json` and removed after the run.

Expand `Commands` for any row in `report.html` to see the original `trtmc-bench` and baseline
commands exactly as they were executed. The report also shows the recorded working directory;
copy either command directly into that environment without going through the matrix
orchestrator or a generated replay script.

For a non-text row the baseline command is equally explicit. For example:

```bash
python3 tools/perf_release.py benchmarks/performance/release.yaml \
  --case whisper.transcribe \
  --output artifacts/perf \
  --dry-run
```

Open the resulting report and expand that row. The raw baseline command contains the adapter,
model, manifest, exact resolved request JSON, precision, warmup/iteration counts, resolved
runtime JSON, workload digest, and output path.

## CI runs

The `TRTMC Performance Matrix` workflow is callable and manually dispatchable. It runs the
same Python entry point with `--ci`, then uploads both result files even when a row fails.
Red and yellow comparisons are valid data and do not fail CI. Unexpected command failures,
workload/output mismatches, or an eager fallback from a declared `torch.compile` baseline do.

The self-hosted runner configures the executable, worker, runtime directories, bundle roots,
bundle cache, and Python profile cache through repository variables. Reference dependencies
stay in the separate baseline process and its selected Python profile or upstream checkout;
they are not added to `trtmc-bench`.

The baseline process is deliberately separate from `trtmc-bench`. Text rows use the shared
`hf-transformers` runner. Task rows use one of the explicit `task-reference` adapters for
Diffusers, ASR, TTS, VLM, embedding, reranking, vision, time series, Qwen3-Omni, PersonaPlex,
ELF, or Lance. Each task result records whether preprocessing and external asset loading are
inside the timed call. Model loading and warmup are always outside measured samples.

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
model, priority, measurement count, baseline mode, timing scope, input/asset inclusion,
compile scope, precision/padding or MoE implementation exceptions, and optional request
overrides. Seq2seq rows also declare how decoder-start/EOS framing maps
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
