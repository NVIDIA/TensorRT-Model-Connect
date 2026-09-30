# trtmc-aiperf-qual

Accuracy and performance qualification of every ready catalog model against its native
(unconverted) Hugging Face / PyTorch model, driven by [AIPerf](https://github.com/ai-dynamo/aiperf).

- **Acc**: the TRTMC output must match the native model's output for the same input, sample by
  sample, within Task-specific tolerances.
- **Perf**: TRTMC must be faster than the native model (eager and, where listed, `torch.compile`)
  at the candidate's precision.

## Design

| Layer | What | Model-specific? |
|---|---|---|
| Execution | Two HTTP servers speaking the same `/v1/tasks/{operation}` protocol: TRTMC (`trtmc-perf-serve --backend trtmc`, later `trtmc-server`) and the native reference (`--backend reference` generic adapters, or `--backend script` for the family's declared reference). AIPerf sends every request and computes the statistics. | No |
| Task | `config/tasks.yaml`: per catalog Task, the Acc suites, grader, gate, and Perf settings. | No |
| Model | Derived from the catalog (checkpoint, revision, precision, bundle, Task, sequence limit). `config/models/<profile>.yaml` holds only exceptions. | Only exceptions |

`trtmc-aiperf-qual plan` prints the derived configuration of every model.

### Accuracy

1. **Goldens**: the native model at the golden precision (fp32 unless the Task says otherwise),
   deterministic numerics (no TF32). Stored per suite, reference identity, and reference platform
   (GPU architecture plus the reference environment's framework versions) under
   `<golden_store>/<suite>/<platform>/<key>/`; `publish-goldens` copies them to a shared golden store.
2. **Noise floor**: the native model at the candidate precision (fp16/bf16) graded against the
   goldens. When the native model itself misses the gate at that precision, the gate becomes its
   own result minus a slack of one sample (2% for large suites); otherwise the gate stands. A
   candidate that meets the original gate passes; one that passes only because the relaxed gate
   requires nothing is `acc-inconclusive`.
3. **Candidate**: the suites run on a persistent TRTMC session. Failing suites are re-run on an
   isolated session (fresh worker per request): passing there means state leaks across requests
   (`acc-session-state`), failing there too means a numerical problem (`acc-issue`).

Text prompts are left-truncated with the model's tokenizer so prompt plus generated tokens fit the
bundle's sequence limit; both sides receive the same text.

| Task | Suite | Grader |
|---|---|---|
| text generation | MMLU zero-shot (10 subjects, answer line); HumanEval (10, token IDs) | `parity_answer_line`, `parity_token_exact` |
| vision-language | Imagenette, one image per class, answer chosen from the ten class names; OCR models: OCRBench v2 (10 task types) | `parity_edit_distance` |
| classification | Imagenette 100 (stride over all classes) | `parity_top1` |
| encoding / embedding | STS-B 50 sentences | `parity_vector` (cosine >= 0.999) |
| image features | Beans 12 (DINOv3 kNN bank) | `parity_numeric` |
| detection | COCO 2017, 100 images | `parity_boxes` |
| segmentation / prompted segmentation | Imagenette 10 | `parity_mask` |
| transcription | LibriSpeech 10 (AIPerf dataset) + corpus WER vs labels | `parity_wer` |
| audio generation | SeedTTS English 5 around the catalog request | `parity_audio` (duration, RMS, spectrum) |
| image / video generation | PartiPrompts 3 around the catalog request | `parity_image` (geometry, thumbnail PSNR/SSIM) |
| image edit, world model, text-prompted segmentation | catalog testcase | `parity_image`, `parity_numeric` |
| reranking | BEIR SciFact 10 queries x 5 abstracts | `parity_scores` (document order) |
| time series | ETTh1 10 seeded windows (the family qualification's window) | `parity_numeric` |
| robot control, stereo, geometry | catalog testcase | `parity_numeric` (shapes exact, vectors by cosine) |

A suite with `base: catalog` overrides the profile's catalog request (size, steps, seed, ...) with
its dataset fields. Exceptions stay in `config/models/<profile>.yaml`:

- **Quantized candidates** (catalog `quantization`, or `candidate.quantization` for a quantized
  checkpoint) use the Task's `quantized` block: for text generation, the answer and the first eight
  generated tokens must agree on 8 of 10 samples instead of exact parity.
- **Sampled generation** (`sampled: true` on a suite, for example Bark): the model always samples
  and TRTMC does not replay PyTorch's random stream, so a failing suite is `acc-inconclusive`.
- **Timing precision** (`reference.timing_precision`): when the native model does not produce a
  valid output at the candidate precision, Perf times it at this precision (the noise floor stays
  at the candidate precision).

Generated media leave the servers as digests (`trtmc_perf_serving.digests`): 64x64 thumbnails of
the first/middle/last frame and a banded log spectrum for audio, so goldens stay small.

### Performance (L1)

The catalog testcase request is sent repeatedly to each server (warmup, then N requests, R runs).
The metric is the server-side model-call time (`trtmc_model_call_time`: `public_task_call_wall`
for TRTMC, the model call after input preparation for references). Per side, the per-run p50
values give a mean and a 95% Student-t CI; `torch.compile` is credited with its best run. Lights
follow the perf matrix rule: green faster by more than 5%, red slower by more than 5%. For generated
media the output check compares geometry only (diffusion content diverges across precisions;
content parity is the Acc suites' job). A run is
white when outputs differ from the reference or a mean side's CI exceeds 5% (and 0.05 ms).
Timing phases hold the host GPU lock (`gpu_lock`), which bundle builds on the same host also take.

### Reference environments

Families declare reference requirements in their benchmark qualification case
(`families/<family>/tests/benchmark/*.yaml`, `reference_environment`). The reference server runs in
that family's environment, created and cached by `trtmc-perf-serve reference-env` (same digest
rules as benchmark qualification, layered on the base interpreter). Bundles are built in the same
environment, which carries the family's declared requirements as CI installs them. The orchestrator
and AIPerf run in their own environment; the TRTMC server needs no Python model packages.

### Reference backends

`reference` (generic adapters, persistent, eager/compile) serves every operation it supports; if it
cannot load or run a model, the family's declared qualification reference (`script`) takes over.
Image editing, world-model generation, and text-prompted segmentation always use the family
reference, because the generic adapters would ignore their extra inputs. A `script` reference
starts one process per request and reports its own warmup/iteration p50.

## Setup

```bash
apps/aiperf_qual/setup.sh /path/to/aiperf-venv /path/to/trtmc-venv/bin/python   # orchestrator + serving packages
cp config/environments/example.yaml config/environments/<machine>.yaml         # fill in this machine's paths
trtmc-aiperf-qual doctor --environment config/environments/<machine>.yaml       # paths, packages, GPU, model list
```

## Commands

```bash
E=config/environments/gb300-perf-serving.yaml
trtmc-aiperf-qual plan --environment $E
trtmc-aiperf-qual run --profile qwen3-0.6b-fp16 --environment $E --out out/qwen3-0.6b-fp16
trtmc-aiperf-qual run-all --environment $E --out-root out/ --shard 0/2      # host 1 of 2
trtmc-aiperf-qual summary gb300-1=nvidia@host1:/runs/out gb300-2=nvidia@host2:/runs/out \
    --ssh "ssh -J jump" --output qualification.md                              # remote roots over ssh
trtmc-aiperf-qual rejudge out/*/                                            # re-apply the judge, no model runs
trtmc-aiperf-qual publish-goldens --source /state/golden-store --store-cli "<storage CLI>"
```

`run` builds the candidate bundle when `bundle_root` does not hold it yet (trtmc-bench, in the
family environment, under the GPU lock), qualifies, and applies the bundle retention policy.
`report.md` / `report.json` in the output directory hold the verdict: `pass`, `acc-issue`,
`acc-session-state`, `acc-inconclusive`, `perf-issue` (red/yellow), `perf-inconclusive` (white),
`not-comparable`, or `error` (a phase failed; see "Phase errors" and `phase-errors.log`);
`build.json` records the build (`build-failed` when it failed).

`run-all` runs profiles one after another (profiles sharing a checkpoint back to back), appends one
line per profile to `<out-root>/campaign.jsonl`, and skips profiles that already have a result
(`--rerun` keeps the old directory as `<profile>.<timestamp>`). `--shard INDEX/COUNT` splits the
catalog across hosts; `summary` merges the result roots afterwards. A remote root
`[NAME=][USER@]HOST:/PATH` is fetched over `--ssh` (result files only: `report.json`, `build.json`,
`error.json`, `excluded.json`); local paths are read in place.

### Per-machine model list

GPU memory decides which models a machine can run. `models` in its environment file selects them
for `run-all` and `plan` (an explicit `--profile` overrides it):

```yaml
models:
  include: all                    # all ready catalog profiles (default), or names / glob patterns
  exclude:                        # a reason is required
    - {profile: "flux-2-dev*", reason: "does not fit in 80 GB"}
  max_checkpoint_gib: 30          # optional: also exclude larger checkpoints (unknown sizes are kept)
```

Excluded profiles are written to `<out-root>/excluded.json` and appear in `summary` as `excluded`
with their reason, unless another result root holds a result for them.

### Disk retention

Checkpoints and bundles dominate disk use. `retention` in the environment file frees them as the
batch advances (both default to `retain`):

| key | values | effect |
|---|---|---|
| `bundle` | `retain`, `delete_on_pass`, `delete_unless_error` | delete `bundle_root/<name>/` after the model's run; an `error` verdict always keeps it for the rerun |
| `hf_cache` | `retain`, `delete_unused` | `run-all` deletes a checkpoint repository from `hf_hub_cache` once no remaining profile of the batch uses it |

Deletion never leaves those roots. Reference environments, goldens, and reports are kept (`rejudge`
needs only the reports). The peak is the largest single model (checkpoint plus bundle) plus the
next profile's checkpoint, which `run-all` downloads during the current run (`--no-prefetch` avoids
it).

## Extending

- New model of a known Task: nothing to add.
- New Task: one entry in `config/tasks.yaml` (suite, grader, gate); add a grader to
  `plugins/trtmc_aiperf_plugins/accuracy.py` only if no existing one fits.
- Model needing a different input or reference option: `config/models/<profile>.yaml`.
- Model needing a differently built bundle: `candidate.build` in `config/models/<profile>.yaml`
  overrides catalog manifest fields (give it its own `name` and `bundle`; see `detr-resnet-50.yaml`).
- New machine: a new file under `config/environments/` (paths, Python interpreters, ports, lock, model list,
  retention); Docker or bare metal only differ in these paths.
