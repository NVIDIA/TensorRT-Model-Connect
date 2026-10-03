# trtmc-aiperf-qual design (decoupled, 24-hour scheme)

Status: approved (codex review round 4, 2026-10-03); revision 7 records what the implementation, its review,
and the smoke run settled (Section 12). Supersedes the v7 campaign
scheme; its results are archived.

## 1. Requirements and scope

1. Qualify every ready catalog model (174 today): TRTMC must be as accurate as the native (unconverted
   Hugging Face / PyTorch) model on the qualified workloads and faster than it.
2. Acc and Perf verdicts are statistically defined and reproducible; the scheme stays simple and a new
   model of a known Task needs no configuration.
3. One GB300 qualifies all ready models within 24 hours (Section 9 states what is inside the budget).
4. No dependency on `qualification_tests/benchmark_qualification`: no imports, no reading of
   `families/*/tests/benchmark/*`. `apps/benchmark` (catalog, `resolve_case`, `trtmc-bench` builds, the
   worker) is the TRTMC side and stays.
5. Datasets: AIPerf's benchmarks and loaders where they fit; otherwise a public dataset pinned by
   revision or sha256 with a scripted acquisition recipe. AIPerf is bypassed only where it cannot carry
   the check.
6. A smoke mode exercises the whole pipeline for every model during the rework; it never yields a verdict.

**What a pass means.** The verdict qualifies one bundle (the catalog bundle as shipped, its sequence
length, image size, and precision) on the workloads listed for its Task in Section 6, against the
native model at the candidate precision. It does not qualify long-context behavior beyond the shipped
length, few-shot prompting, extended reasoning, or dimensions Section 6 marks as not covered. A model
with a `candidate.build` exception is qualified for that bundle only, and the report names it.

## 2. The candidate: the shipped bundle

- 51 of 66 text models ship 256-token bundles. MMLU 0-shot fits them: with Qwen3's tokenizer, 1,080 of
  1,140 problems (20 per subject; 54 of 57 subjects) fit 256 tokens with a 128-token chat allowance and
  all 1,140 fit 512. Problems are filtered by their **rendered** length (the chat template applied, plus
  special and reserved output tokens), not a fixed allowance; the report lists retained problems per
  subject.
- A near-capacity request per text model (Section 4.3) exercises the shipped length.
- The v7 `-qual` rebuilds (4,096 / 2,048 tokens) are not repeated: 17 families could not serve them;
  that remains a tracked TRTMC finding and a later `long_context` check, not part of this verdict.

## 3. Decoupling: removal inventory and replacements

| Removed (current dependency) | Replacement |
|---|---|
| `models._qualification_cases` / `_qualified_build` / `_qualification_case`: accuracy and performance cases, ETTh1 windows, build and bundle overrides, revision fallback, `trust_remote_code`, declared native model / revision / precision, script fallback, performance request, `model_directory` | catalog entry + Task defaults + explicit `config/models/<profile>.yaml` exceptions (`reference.requirements`, `reference.trust_remote_code`, `reference.model`/`revision`, `candidate.revision`, `candidate.build`, ETTh1 `window`), migrated once from the inventory and reviewed per model |
| `family.py` (family accuracy cases, reference attribution, isolated re-check), `cli.py` family refresh | deleted; Acc is Section 6 only |
| `suites._qualification_perf_records` (family performance request) | the catalog testcase (`trtmc-perf-serve payload`, i.e. `resolve_case`) plus the near-capacity request; per-model `performance.l1` overrides where a catalog request leaves work unstated |
| `gold_metrics.coco_map` importing `benchmark_qualification.accuracy` | `pycocotools` COCOeval |
| `perf_serving/backends/script.py` (family reference scripts via `reference_harness`) and native fallback loops (`absolute.run_native`, `generation.py`, `runner._reference_perf`) | generic adapters by operation; a family's own native pipeline in `families/<family>/tests/native_reference.py` behind the same adapter interface (Section 7) |
| `perf_serving reference-env` reading qualification cases | `reference-env --requirements families/<family>/requirements.txt` (the family's own dependency file) layered on the serving interpreter; `--no-build-isolation` where declared |
| `bundles._descriptor` `model_directory` (an environment hook prepared upstream checkouts) | catalog builds with the serving interpreter; native upstream checkouts by the family's `native_prepare.py` (Section 7); a model that cannot build from its catalog entry is a recorded build failure |
| tests and docs asserting family cases, windows, script backends | rewritten |

Acceptance: a unit test scans `apps/aiperf_qual` and `apps/perf_serving` sources for
`qualification_tests` and `tests/benchmark` and fails on any hit; `trtmc-aiperf-qual matrix` resolves
all 174 ready profiles without them (Section 7).

## 4. Verdict rules

### 4.1 Contrast, units, outcomes

For every benchmark the **regression** `R` is positive when TRTMC is worse: `native - TRTMC` for
higher-is-better scores, `TRTMC - native` for lower-is-better ones (WER, MSE). Scores are in points
(accuracy, pass rate, chrF, mAP, IoU, nDCG, Spearman x 100); the margin `delta` is in the same points,
or `max(delta, relative x |native score|)` where a relative margin is declared. Gates use unrounded values.

Each benchmark ends in one of:

- `pass`: non-inferiority shown, the one-sided 95% upper bound of `R` is below the margin;
- `fail` (regression established): the one-sided 95% lower bound of `R` exceeds the margin;
- `inconclusive`: neither;
- `not-comparable`: the native score is below the benchmark's suitability floor `min_native`;
- `error`: a problem without an answer on either side, or a phase failure.

A model passes Acc only when **every** mandatory benchmark of its Task passes (intersection-union: no
multiplicity correction is needed for the joint claim, and no subset may pass on its own). Claims are
per model and per benchmark, not simultaneous across the fleet.

### 4.2 Binary metrics: paired score test

Per problem each side is right or wrong; `b` = only native right, `c` = only TRTMC right, `n` problems;
the margin in points converts to a fraction, `delta = margin / 100`. The statistic is Tango's paired
score statistic at the margin:
`Z = (b - c - n delta) / sqrt(n (2 t + delta (1 - delta)))`, with `t` the restricted maximum-likelihood
estimate of the TRTMC-only probability under `R = delta`: `t = (-B + sqrt(B^2 - 4 A C)) / (2 A)`,
`A = 2n`, `B = -(b + c) + (2n - b + c) delta`, `C = -c delta (1 - delta)`. Pass iff `Z < -k`
(non-inferiority, `H0: R >= delta` rejected); fail iff `Z > k` (`H0: R <= delta` rejected).

The cutoff `k(n, delta)` is exact for the frozen, post-filtering `n`: the smallest value on a 0.005 grid
from 1.645 whose pass **and** fail probabilities at the margin (`R = delta`), enumerated over the
trinomial distribution of `(b, c)`, are both at most 4.9% on a dense discordance grid: from `delta` (no
TRTMC-only answers) to `delta + 0.05` in steps of 0.0005 and on to `delta + 0.4` in steps of 0.005
(the 0.1-point allowance below 5% covers the size between grid points; discordance above `delta + 0.4`
is outside the admissible range: v7's largest unbiased discordance was 9.3%). It is computed when judging
(under 20 s at n = 10,000) and recorded. Examples: n = 100, delta = 5 points: k = 1.84; n = 287,
delta = 1: 1.79; n = 2,280, delta = 1: 1.73; n = 5,153, delta = 0.5: 1.73. The test does not collapse
at zero discordance: n = 100, delta = 3 points passes `b = c = 0` (Z = -1.76 < -1.745) and leaves
`b = 1, c = 0` inconclusive (Z = -1.17).

### 4.3 Sampled models, corpus metrics, clusters

- **Sampled models** (catalog request samples): three seeds per side; per problem `d_i` = mean over
  seeds of (native right) - mean over seeds of (TRTMC right); the bound is the one-sided Student-t
  bound over problems, `mean(d) +/- t(0.95, n-1) sd(d) / sqrt(n)` (seeds stay inside their problem).
- **Corpus metrics** (WER, chrF, mAP, mIoU, Spearman, nDCG, MSE, CLIP scores): paired percentile
  bootstrap of `R - margin(native)` (the relative margin recomputed per resample), 2,000 resamples
  (COCO / ADE20K: 200, each a full evaluation with unique image ids per draw), fixed seed; pass iff the
  95th percentile is below 0, fail iff the 5th percentile is above 0. The statistic is recomputed on
  each resample, never averaged from per-sample surrogates. A zero native score with a relative margin
  is `not-comparable`.
- **Clusters**: resampling units are independent groups: LibriSpeech speakers, translation documents,
  ETTh1 moving blocks of `ceil((context + horizon) / stride)` consecutive windows. RefCOCO keeps one
  referring expression per image so its pairs are independent.

### 4.4 Suitability and selection

- `min_native` per benchmark, set from reference-only calibration (v7 native scores) before the formal
  run: MMLU and MMStar 30 (chance 25), LAMBADA and TinyStories 10, OCRBench 10, code 5, GenEval 10.
  A model below it uses its Task's declared alternative (for example LAMBADA for a base LM) in
  `config/models`; at run time it is `not-comparable`, which blocks qualification.
- Selection is seeded stratified random sampling (seed 20261003) within subject / class / speaker after
  length filtering, never a dataset prefix; the selection manifest (sample ids and their sha) is in the
  report, and counts are frozen before the formal run.

### 4.5 Sample sizes (power)

`delta` is the largest regression accepted as numerical noise of a correct conversion, from v7's paired
data: healthy models' `|R|` stayed within 0.44 points on MMLU and 0.12 on LAMBADA, with discordance
`q` of 0.1-1.4%; defects were one-sided and larger (MMLU 4-50 points, LAMBADA 0.9-2.6). `n` then gives
an unbiased model at least 90% probability to pass at q = 2% (exact enumeration at the exact cutoff);
LAMBADA, whose healthy discordance in v7 stayed at or below 1%, is sized at 1%:

| Benchmark | n | delta | k | P(pass), unbiased, q = 1 / 2 / 5 / 10% |
|---|---|---|---|---|
| MMLU 0-shot, 40 per subject | <= 2,280 | 1 (quantized 2) | 1.73 | 1.00 / 0.94 / 0.65 / 0.41 |
| LAMBADA | 5,153 | 0.5 | 1.73 | 0.96 / 0.78 / 0.45 / 0.28 |
| MMStar | 1,500 | 1.5 | 1.81 | 1.00 / 0.98 / 0.77 / 0.50 |
| OCRBench | 1,000 | 2 | 1.81 | 1.00 / 0.99 / 0.83 / 0.57 |
| code (HumanEval + MBPP) | 664 | 2 | 1.865 | 0.99 / 0.92 / 0.64 / 0.40 |
| ImageNetV2 (all) | 10,000 | 0.5 | 1.705 | 1.00 / 0.96 / 0.70 / 0.45 |
| GenEval (latent replay) | 200 | 5 | 1.72 | 1.00 / 0.99 / 0.88 / 0.67 |

A model with 5-10% discordance (v7: minicpm5-2b, nemotron-labs-diffusion-8b) is often `inconclusive`:
symmetric but frequent disagreement is reported, not passed. The v7 defects all `fail` under these
margins (gpt2 LAMBADA, b = 53, c = 0: Z = 5.2 > 1.73). Corpus-metric sizes are confirmed from the
smoke and pilot runs (bootstrap width at the formal `n`) before the counts are frozen. A budget
shortfall is resolved by execution, never by a smaller `n` (Section 9).

### 4.6 Performance

- Workload: the catalog testcase (`resolve_case`), plus for text generation a near-capacity request
  (a public passage filling the bundle minus 32 generated tokens, greedy). A catalog request leaving
  generation work to each side's default (`num_steps`, `guidance_scale`, `num_frames` at -1) is an
  error until `config/models` states it.
- Measurement per request and side: warmup 3, 12 requests per run, 5 runs (checkpoints above the large
  size: warmup 1, 3 requests, 5 runs; generative media and speech output: warmup 1, 3 requests, 3 runs); statistic:
  the server-side model-call p50 per run.
- Speedup `S = native / TRTMC`; its 90% two-sided interval (95% one-sided per bound) is Welch's t
  interval of `log(native) - log(TRTMC)` over the runs. **green**: lower bound > 1.05; **red**: upper
  bound < 0.95; **perf-inconclusive** (white): either side's 95% half-width exceeds 5% of its mean, the
  work differs, the native model ran at another precision, or the GPU was busy or not measured; **yellow** (not
  demonstrably faster): otherwise. Every request of the model must be green to pass.
- Work equivalence is checked on **every** timed response, not the first: equal output tokens for text,
  equal height/width/frames/steps for media, equal audio length (10 ms) for speech; any mismatch is
  `perf-inconclusive`. A catalog request that samples is timed as its greedy variant (temperature 0,
  same lengths and shapes) so both sides do the same work; sampling stays in Acc. A model that cannot
  decode greedily (Bark) is timed as shipped and is `perf-inconclusive` whenever lengths differ.
- Order: the native model is timed before TRTMC. Before the formal run, five models (one per server
  size class, including the largest) are timed twice in both orders. Where both servers fit, an order
  effect above 2% switches to interleaved runs; where they do not, the native model is timed again after
  TRTMC (native, TRTMC, native) and the model is `perf-inconclusive` if its two native timings differ by
  more than 2%.
- `torch.compile` and the L2 serving sweep are opt-in reports outside the category.

### 4.7 Model category

`error` > `acc-issue` (any `fail`) > `not-covered` (no native path) > `acc-inconclusive` /
`not-comparable` > `perf-issue` (red / yellow) > `perf-inconclusive` > `pass`. Only `pass` qualifies.
Perf-only models (random weights) have conversion parity as their Acc evidence: on the catalog and the
near-capacity request, TRTMC's greedy token ids equal the native ones for the first 8 tokens (or the
whole text is equal); a mismatch is `acc-issue`.

## 5. Datasets and AIPerf usage

AIPerf-native: MMLU (`mmlu`, `lighteval/mmlu` pinned) and LAMBADA (AIPerf accuracy mode with our pinned
loader plugin). AIPerf-carried (our pinned dataset through the `trtmc_task` endpoint, scored here):
everything else. AIPerf loaders reused where they exist: `mmstar`, `librispeech`, the
`spec_al_humaneval` / `spec_al_mbpp` prompts joined by task id to the pinned `openai_humaneval` /
`mbpp` tests. Locally prepared files (STS-B, SciFact, HumanEval) are replaced by pinned Hugging Face
revisions (`mteb/stsbenchmark-sts`, `mteb/scifact`) with the transformation in the suite definition.

## 6. Per-Task contracts

`n` is the formal count (frozen after the pilot); all binary entries use 4.2, corpus entries 4.3.

| Task (models) | Benchmark, source | n, sampling | Metric (direction), margin | Notes |
|---|---|---|---|---|
| text generation, instruction / base LMs (~45) | MMLU 0-shot, AIPerf `mmlu` with an answer-format instruction (Section 12) | 40 per subject, stratified | accuracy (up), 1 pt; quantized 2 | min_native 30; chat route per catalog request |
| small base LMs (~14) | LAMBADA OpenAI | all 5,153 | last-word accuracy (up), 0.5 pt | completions |
| code (codegen, starcoder2) | HumanEval 164 + MBPP test 500 | all 664 | pass@1 greedy (up), 2 pt | MBPP prompt as bigcode-evaluation-harness writes it for base models: the description and the first test in a docstring (names the function); execution in Section 8 |
| translation (marian, t5, nllb, riva) | newstest2019 en-ru / WMT14 en-de / FLORES-200 en-fr | all | chrF++ (up), 1 pt | document clusters |
| BART | WikiText-103 span denoising | 1,000 | chrF++ of the reconstruction (up), 1 pt | replaces exact-span accuracy (native 5.1%) |
| TinyStories model | TinyStories last word | 2,000 | accuracy (up), 1 pt | |
| random-weight LMs (2) | none | - | conversion parity of L1 outputs | Perf-only |
| VLMs (6) | MMStar val (AIPerf `mmstar`) | all 1,500 | accuracy (up), 1.5 pt | min_native 30 |
| OCR VLMs (deepseek-ocr, VLMs with OCR) | OCRBench | all 1,000 | contains-match (up), 2 pt | |
| grounding VLM (locateanything) | RefCOCO val, one expression per image | 1,500 | IoU >= 0.5 (up), 1.5 pt | |
| ASR (5) | LibriSpeech test-clean (AIPerf `librispeech`) | 1,000, 25 per speaker | WER (down), max(0.3 pt, 5% of native) | speaker clusters |
| streaming ASR (2) | same | 1,000 | WER (down), max(0.5 pt, 10%) | TRTMC streams; native streams when its adapter supports it, else offline (labelled) |
| classification (37) | ImageNetV2 matched-frequency | all 10,000 | top-1 (up), 0.5 pt | |
| image features (DINOv3, 2) | ImageNetV2 kNN: gallery 5 per class, disjoint test 5 per class | 5,000 | top-1 (up), 1 pt | L2-normalized pooled feature, k = 20 cosine-weighted (T = 0.07), each side its own gallery |
| sentence embedders (7) | STS-B test (`mteb/stsbenchmark-sts`) + SciFact retrieval (`mteb/scifact`, full 5,183-doc corpus, 300 test queries) | all | Spearman (up), 0.5 pt; nDCG@10 (up), 1 pt | the model's own pooling (sentence-transformers config) on both sides |
| raw encoders (13: BERT, RoBERTa, ...) | STS-B test sentences | 400 | conversion parity on every sentence: equal shapes, finite values, mean token cosine of the last hidden state >= 0.999 and relative L2 error <= 2% over the non-padding positions | STS-B is not their task (native Spearman 9-31) |
| reranker (1) | SciFact, fixed BM25 top-20 candidates per query (pinned `rank_bm25`) | 300 queries | nDCG@10 (up), 1 pt | labelled "rerank of fixed candidates" |
| detection (7) | COCO val2017, crowd and area kept | 500 images | mAP@[.5:.95] (up), 1 pt | pycocotools, image bootstrap |
| semantic segmentation | ADE20K val | 500 | dataset-level class mIoU (up), 1 pt | image bootstrap |
| prompted segmentation (SAM) | RefCOCO, one object per image, point prompt | 500 | mean mask IoU (up), 1 pt | |
| text-prompted segmentation (SAM3) | RefCOCO, one object per image, text prompt | 500 | mean mask IoU (up), 1 pt | native: transformers `Sam3Model` |
| time series (4) | ETTh1 standard split, StandardScaler fit on rows 0-8,639, test targets 11,520-14,399, stride 24 | all windows | MSE on the scaled values (down), 1% of native | the model's columns / context / horizon in `config/models`; moving-block bootstrap |
| TTS (3) | Seed-TTS eval English | 200 | Whisper large-v3-turbo round-trip WER (down), max(2 pt, 10%) | validity of every output: not silent (RMS > -50 dBFS), duration within 0.5-2x native, finite; voice identity and prosody not covered |
| text-to-image, latent replay (flux, pixart, qwen-image, z-image) | GenEval prompts | 200, stratified by tag | pass rate (up), 5 pt | OWLv2 detector + CLIP color (an approximation of GenEval, labelled); exact counts; same initial noise on both sides |
| text-to-image/video without replay (minimax-h3, wan22) and videos (wan21) | GenEval / VBench object prompts | 100 / 20 | CLIP-T score (up), 1 pt, prompt-paired bootstrap | videos: frame count equal and not frozen (mean inter-frame change >= 0.25x native) on every video; motion quality, temporal order, and flicker are not covered |
| image edit (qwen-image-edit) | MagicBrush dev, first turn | 100 | CLIP-I and DINO to the human target (up), 1 pt each | no-edit baseline reported; an output equal to its source (mean abs diff < 1/255) fails; whether the edit follows the instruction is not covered beyond target similarity |
| monocular geometry (moge) | COCO val2017 images | 50 | conversion parity on every image: median relative depth error <= 1%, valid-mask IoU >= 0.99 | native: `moge` package (family adapter) |
| stereo (fast-foundation-stereo) | Middlebury 2014 half resolution | 15 pairs | conversion parity on every pair: mean end-point error between the sides <= 0.1 px | upstream checkout (family adapter) |
| robot control (ACT) | `lerobot/aloha_sim_transfer_cube_human` observations | 50 | conversion parity on every observation: max abs action error <= 1e-3 of the action range | native: `lerobot` (family adapter) |
| speech-to-speech (personaplex) | LibriSpeech test-clean utterances as the user turn | 20 | output parity on every input: duration 0.8-1.2x, RMS 0.5-2x native, log-spectral distance <= 3 dB | upstream checkout (family adapter); conversational quality is not covered |
| world model (sana-wm) | COCO val2017 images with the catalog action sequence | 10 | validity (frame count equal, not frozen) and coarse agreement with the native video at the same seed: mean PSNR >= 5 dB, SSIM >= 0.1 | upstream checkout (family adapter); action-conditional fidelity is not covered |

## 7. Execution matrix and native paths

`trtmc-aiperf-qual matrix` writes one row per ready profile: Task, bundle, native adapter and its
environment (requirements, remote code, checkpoint and revision), workloads, mandatory checks, and
whether every path exists. It is generated from the configuration, so it cannot drift. The formal run
starts only when every row has an executable native path and mandatory checks (`not-covered` is a
rework state, never a formal outcome).

Native paths: the generic adapters in `perf_serving/backends/reference/` (model-agnostic, by operation);
where a family's native pipeline needs its own code (SAM3, MoGe, ACT, FoundationStereo, PersonaPlex,
Sana-WM today; any further gap the smoke run finds), the family owns it in
`families/<family>/tests/native_reference.py` (outside `tests/benchmark`), implementing the adapter interface
(`Adapter(spec).invoke(request, artifact_base) -> Invocation`), named by `reference.adapter` in
`config/models`. A family that needs an upstream checkout provides `families/<family>/tests/native_prepare.py`,
run once when its reference environment is created (`reference.prepare`).

## 8. Reproducibility

- **Bundles**: every run asks trtmc-bench to prepare the bundle; it reuses an existing one only when the
  bundle's build receipt matches (sha256 of the catalog manifest or of the descriptor carrying the build
  overrides, the resolved checkpoint snapshot, the build command, the TRTMC core and family source digest,
  package versions) and rebuilds it otherwise (a local model directory has no immutable identity: always
  rebuilt).
- **Runs** carry a run key (sha of the resolved model configuration, the harness sources, the code the run
  executes: the serving package, TRTMC core and trtmc-bench sources, the worker binary and runtime
  libraries, and the model's family directory; the mode smoke/formal; and the `pip freeze` digests of the
  serving, AIPerf, and reference interpreters); `run-all` skips a profile only when its finished report has
  the same key. Re-judging and re-checking keep the run's own report as `report.original.json` (never
  overwritten) and write the new one as `report.json`; a smoke report stays a smoke report.
- **Reference environments** are keyed by the requirements file, the preparation script, and the
  interpreter; at creation their `pip freeze` is stored with them and every reuse verifies it is unchanged.
  An environment whose freeze differs or was never recorded is left as it is and a fresh one is created
  next to it; each report records the interpreter used. References may download what their checkpoint
  does not carry; one whose gated repository refuses even optional-file checks (SAM3) declares
  `reference.offline` and reads the hub cache only.
- **Code execution** (HumanEval, MBPP): the GB300 container forbids namespaces (`unshare`: operation not
  permitted, verified), so each program runs as `nobody` through `setpriv` (groups cleared, no
  capabilities, no new privileges), with an empty environment, a temporary home and working directory,
  the human-eval reliability guard, CPU / memory / file-size / process / open-file limits, and a timeout, in
its own process group that is killed whole after the run. The boundary is
  file permissions: the program can read world-readable files and write only where any user may (its
  directory, `/tmp`, `/dev/shm`); before each run the harness verifies that `nobody` can neither write
  the result, bundle, cache, and data roots nor read root's home, and fails closed (an error, never an
  unprivileged-but-unchecked or root run) otherwise. Network egress is not blocked; the programs are the
  models' completions of public prompts. A pass needs a per-run nonce printed after the tests complete
  (an early `exit(0)` fails).

## 9. 24-hour budget

Inside the budget: bundle builds (v7: 94 builds, mean 109 s, 2.8 h in total), server starts, all Acc
requests at the frozen `n`, scorers, bootstrap, Perf, and a failure allowance (one retry of a phase that
fails before producing a result; 10% of the ledger). Every phase has a deadline of three times its
ledger time (at least 10 minutes); a phase past its deadline is stopped, counted as failed, and retried
once, so a hung phase costs at most its deadline twice. Outside: one-time preparation (checkpoint and
dataset downloads, reference environments, scorer checkpoints). The smoke run records every model's
build, start, peak memory, and per-request times; a pilot runs one representative model per Task at the
formal `n`; together they give a per-model ledger, and the formal run starts only when the ledger
predicts at most 22 hours. Memory: a native model whose weights exceed 240 GB (MiniMax-H3, 351 GB) runs
with layers offloaded to host memory (Accelerate `device_map`), its Perf labelled as against an
offloaded baseline. Levers when the ledger exceeds 22 hours: one native server reused for L1 and Acc,
native replicas where the measured throughput gain is real, TRTMC replicas for the Acc answers (Section 12),
the candidate probe reused as the candidate server. If it still exceeds 22 hours, the excess Tasks are reported to the owner; `n` is not reduced
silently.

## 10. Smoke mode

`run --smoke` / `run-all --smoke`: one logical problem per benchmark (an STS pair needs three pairs for
Spearman; a retrieval query brings its relevant documents and nine others), every scorer runs and
reports `n/a` where a statistic is undefined, L1 with warmup 0, 1 request, 1 run. Results go to a
separate `smoke/` namespace with their own run key and never satisfy a formal run; the verdict is
`smoke-pass` / `smoke-fail` with the failing phases, and `run` / `run-all --smoke` exit non-zero when any
model is not `smoke-pass`.

## 11. Review gates

codex (`gpt-6-astra`) reviews this design, the decoupling change, the per-model configuration, the
smoke results, and the formal run before each lands in PR #1550.

## 12. Settled during implementation (revision 5)

- **Build exceptions** (Section 2), each named in `config/models` and in the report: DETR's catalog engine is
  796x1333 and rejects every COCO image whose DETR resize (shortest edge 800) is taller than 796 rows, so COCO runs
  on a 1333x1333 build of the same checkpoint; Fast-FoundationStereo and SANA-WM build from the upstream model
  directory their family's `native_prepare.py` prepares (`candidate.model_directory`), as the catalog cannot build
  them from the Hugging Face repository alone.
- **Native paths** (Section 7) under `families/<family>/tests/native_reference.py` (with `native_prepare.py` and
  `native_inputs.py` next to it): the family's `tests/` directory is where the repository keeps reference code
  (its architecture rules require family production code to be reachable from `model.py` and free of
  environment side channels): GLM-ASR (a speech-conditioned causal
  LM), Magpie TTS, Canary and the Nemotron streaming ASR models (NeMo archives), DeepSeek-OCR (its own
  `model.infer`, bf16 only, so its Perf is perf-inconclusive), LocateAnything (its official loading and
  prompt contract, fp32 only), Phi-4 Multimodal (its own processor and causal LM), Nemotron Labs Diffusion
  (`ar_generate`, the catalog's autoregressive mode, which every request carries), SAM3, MoGe-2, ACT, Fast-FoundationStereo, PersonaPlex, SANA-WM, YOLOv5 / v8 / v10 / 11 (Ultralytics
  archives), Chronos-Bolt and TimesFM (no generic time-series adapter). Inputs a family must prepare itself
  (decoded LeRobot frames, the Middlebury 700x700 profile) come from its `native_inputs.py` (`family_inputs`
  suites), run in its reference environment.
- **Raw encoders**: TRTMC's `encode` returns the first-token hidden state, so the conversion parity compares that
  vector (cosine and relative L2 error), not every token.
- **Perf requests**: the near-capacity request fills at most 16,384 prompt tokens (native eager prefill memory);
  time-series models time their first ETTh1 window (their catalog testcases hold a few values, which the native
  models reject); a sampling text request is timed as its greedy variant.
- **Datasets**: STS-B (`mteb/stsbenchmark-sts`), SciFact (`mteb/scifact`, full corpus; BM25 top-20 for the
  reranker), HumanEval and MBPP (`openai/openai_humaneval`, `google-research-datasets/mbpp`) load from pinned
  Hugging Face revisions (AIPerf's `spec_al_*` loaders are not pinned). The translation sets stay sha256-pinned
  files: the sacreBLEU test sets (WMT14 en-de, newstest2019 en-ru, FLORES-200 devtest en-fr) with each model's
  request format; they carry no document ids, so sentences are the bootstrap units. The pinned COCO export has
  no crowd regions; COCOeval runs without them. BART is scored as corpus chrF++ of its reconstructed sentences
  from AIPerf's records.
- **DINOv3** (2 profiles) is blocked outside the harness: its checkpoints are gated, the hosts' token is refused
  (HTTP 403), and the cached weights are gone. The kNN contract (Section 6) is **not implemented**: neither the
  native nor the TRTMC feature output can be inspected without the weights. Until access exists both profiles
  resolve to `accuracy_source: missing` (an `error` entry, never a pass); the contract is implemented and
  smoke-verified once it does.
- **Work evidence** (Section 4.6), per operation from fields both backends report: for text (generation,
  translation, transcription) the generated token count or the generated text, all responses agreeing on
  either (equal counts are the same decode steps; equal texts the same tokens, whichever way a backend counts
  the end-of-sequence token: Marian's TRTMC counts it, the native path does not; the native Whisper-style
  transcriber reports its decoding steps, counted by a logits processor that generation calls once per generated
  token (the end token and generated special tokens included, the forced prompt not), as TRTMC counts its generated
  ids, so two transcripts of equal decoding length still compare; it runs TRTMC's fixed decoder prompt, English
  transcription declared as `reference.options`, so neither side spends a language-detection pass the other skips); `media_digest` frames /
  height / width for generated media; `audio_digest` length (10 ms) for generated speech; nothing for
  operations whose input fixes the work. TRTMC requests cannot force a generation length (no ignore-EOS), so a
  greedy text request whose two outputs end at different points is `perf-inconclusive`.
  Denoising steps are request parameters both sides receive explicitly (an unstated one is an error). Every
  timed response must carry its model-call time and its evidence, and all responses of both sides one and the
  same signature. A native model timed at another precision than the candidate's (a declared
  `timing_precision`, as for Z-Image and Wan 2.1) is `perf-inconclusive`, like a fallback. The opt-in
  `torch.compile` reference, aggregated by its best run, is not held to the 5% half-width.
- **Near-capacity request** (text-only requests of the text generation Task; vision-language models' image
  tokens share the bundle length): the passage is sized so that the rendered prompt (the chat template when the
  request asks for one, else the tokenizer with its special tokens) is exactly the bundle length minus 32. TRTMC
  tokenizes the passage itself and can count a few more tokens (TinyLlama: 2), and a bundle's prefill profile can
  be shorter than its sequence length (InternLM2): the TRTMC probe server checks the request first, and one it
  rejects for length is shortened to the longest passage it accepts (binary search); both sides time that
  request, and the report records its rendered length next to the budget.
- **Stated generation controls**: `config/models` `request` states what a catalog request leaves at -1 (guidance,
  CFG, steps, frames); it applies to every suite built on the catalog request (the timed request, GenEval, replay
  parity, world-model parity), so Acc generations do not rest on each side's own default either. The value is the
  one TRTMC's family uses, read where each side actually takes it: PixArt-Sigma 4.5 and Wan 5.0 (both defaults);
  Qwen-Image 4.0 as guidance_scale (TRTMC's true CFG field; Diffusers takes cfg_scale); FLUX.2-dev 3.5 (TRTMC's
  default; Diffusers would use 4.0); FLUX.1-schnell 0.0 (no guidance embedding); Z-Image and MiniMax-H3 without CFG
  (Turbo / CFG-distilled weights).
- **Qualifying lights**: one eager light per timed request decides Perf; a missing, unavailable, or `error` eager
  light is an `error`. The opt-in `torch.compile` lights are reported only. Every successful timed response must
  carry its model-call time (a success without one, or without a body, makes the run incomplete: `error`).
- **Native timing scope**: input files are read and decoded before the timed call (the backend preloads every
  `*_path` input; the MoGe, Fast-FoundationStereo, and ACT adapters read the preloaded data), and output artifacts
  (MoGe, Fast-FoundationStereo, SANA-WM) are written after it, as on the TRTMC side. Time-series references return
  their forecast values inline, as TRTMC does.
- **Missing outputs** are missing answers (an `error`), never a wrong answer or a zero score: generated media
  (GenEval, edits) without an image, and for every corpus metric an output without the field it reads, well
  formed (a forecast of the horizon's length with finite values, a finite vector, a mask of the image's size).
- **Smoke coverage**: a sampled model's smoke run keeps every configured seed (one problem), so the seed-mean
  scorer runs; the combined code suite sends one HumanEval and one MBPP problem. A conversion-parity output that is
  missing or unreadable on either side (no vector or action chunk, no size, an unreadable artifact, no audio digest,
  a video without frames) is missing evidence: an `error`, never a `fail`, in smoke and formal runs alike; a smoke
  verdict whose Acc or Perf is `error` is `smoke-fail`. Each side's depth, mask, or disparity artifact is read at
  that side's own size before the sizes are compared, so a size difference fails only between readable outputs.
- **Order check** (Section 4.6): `trtmc-aiperf-qual order-check --profile P` times a profile's L1 requests in both
  orders (native then TRTMC, TRTMC then native) and reports each side's order effect (its second timing relative to
  its first). Before the formal run it runs on five profiles spanning the server sizes, including the largest
  dense one; an effect above 2% switches that size class to interleaved runs or native, TRTMC, native timing.
  The effect counts only when all four measurements are valid on their own (every request succeeded and is timed,
  every response carries work evidence, the GPU was measured idle before every run, at least two runs within
  `max_ci_percent`) and each order's two sides did the same work; otherwise, or when a measurement or the check
  fails (kept as the measurement's failure; the other measurements still run), the check is `unresolved` and is
  repeated, never read as within the limit. Each run replaces the profile's previous `order.json`. A GPU
  utilization reading that fails is no evidence of an idle GPU: in the formal verdict it makes the Perf light
  white, as a busy GPU does.
- **Smoke namespace**: `run --smoke --out <dir>/<profile>` writes to `<dir>/smoke/<profile>`, as `run-all --smoke`
  does.
- **Lance** (lance-3b-x2t-image) is `not-covered`, declared with its reason (`reference.not_covered`): the upstream
  inference is a batch command that loads the model on every invocation, so serving it per request needs a resident
  adapter around its internals, not written yet. A rework item before the formal run (Section 7).
- **s1-mini** does not build from the catalog (`trtmc build` cannot choose between the `qwen` and `s1_mini`
  families): a TRTMC finding, reported as a build failure.
- **Replicas for Acc answers**: the native adapter (`native_replicas`) and the TRTMC server (`candidate_replicas`)
  answer Acc problems as up to four copies that fit the GPU's free memory (the first copy measures one copy's
  footprint; a copy that fails to start leaves the ones running). Each copy loads the same checkpoint or bundle and
  answers one request at a time, with nothing batched across requests, so the answers equal one server's; the pilot
  measures the throughput gain. L1 is timed on a single server started after the copies stopped, and the Acc
  requests' own model-call times (`workload_perf`) are white when either side ran as copies. Smoke runs use one
  copy. Copies of separate processes time-slice the GPU: the pilot measured about 4x for native eager copies,
  2x for TRTMC's falcon-rw-1b and none for its lfm2-350m or qwen35-4b. `acc_mps` (environment; on for GB300)
  starts a CUDA MPS daemon of the copies' own (a private pipe directory: only the copies are its clients, L1
  servers and the GPU-idle reading are not) so their kernels share the SMs. The daemon and its clients name the
  GPU by UUID (MPS renumbers ordinals; ambiguous ordinals mean no MPS). It quits after the copies stop and its
  control and server processes must have exited (signalled if they outlive the quit; only processes whose program
  is MPS's and whose environment names the attempt's own pipe directory, each attempt in a directory of its own)
  before anything else runs: one that survives stops the run (no phase, fallback, or per-model handler absorbs
  it). A daemon that does not start is stopped too and leaves the copies time-sliced. Measured on lfm2-350m (four copies a side, the same
  bundle): TRTMC MMLU 199 -> 88 s and LAMBADA 345 -> 150 s with every answer identical (2,278 and 5,151); native
  LAMBADA 189 -> 147 s; falcon-rw-1b TRTMC LAMBADA 84 -> 43 s. TensorRT builds are not bit-reproducible: falcon's
  bundle, rebuilt between the two runs, changed 58 of 5,153 LAMBADA grades (55.95% -> 55.99%), so a verdict is
  for the bundle its run built. Native eager answers vary between runs with or without MPS alike (two runs without it agree on
  2,271 of 2,278 MMLU answers; with it, 2,269 to 2,272); TRTMC's copies, with or without MPS, reproduce a single
  server's answers exactly (lfm2-350m, qwen35-4b: 2,227 of 2,227).
- **Five timed runs** (Section 4.6): the pilot timed ResNet-50 at a 9.7x speedup with run p50s 2% apart, and the
  CI gate turned it white: with three runs the 95% half-width is t(0.975, 2) = 4.30 standard errors, so ordinary
  run-to-run noise exceeds 5%. The same number of timed requests now runs as five runs (t = 2.78): 12 requests per
  run instead of 20 (large checkpoints 3 instead of 5). Generative media and speech output keep three runs of three
  long requests, a cost choice resting on the pilot so far (half-widths with three runs: FLUX.1-schnell TRTMC 0.63%,
  native 0.21%; Bark-small 1.73%, 2.19%): every media and speech profile of the pilot is checked before the counts
  freeze (Section 12.1), and if any is white on the CI gate those categories move to five runs too.
- **MMLU answer format**: lighteval's 0-shot prompt gets one added sentence, "Answer with the letter of the
  correct option only.", and an 8-token answer budget (lighteval: 5), on the completion and chat routes alike;
  AIPerf's grader is unchanged. Without it, instruction-tuned models open with an explanation and give no letter
  within the budget: the pilot's qwen35-4b scored 0.7 natively (not-comparable), and the smoke runs' native
  answers began "To find ...", "We are given ..." for about 30 of 43 chat models. A CPU probe (60 problems, 3 per
  subject over 20 subjects, the native chat rendering): Qwen3.5-0.8B 5.0 -> 53.3 (5 -> 57 parsed), Qwen3-0.6B
  6.7 -> 48.3, MiniCPM5-2B 11.7 -> 68.3, and the base Falcon3-1B 53.3 -> 51.7 (60 parsed either way). Five-shot
  prompts, which make chat models answer with a letter too, do not fit the 256-token bundles (Section 2).
- **Retries and deadlines**: a failed GPU phase runs once more (not in smoke mode); every AIPerf run (Acc, Perf,
  generation for the media checks, L2) takes the ledger's per-profile deadline from the environment's
  `deadlines` map.

### 12.1 Gates before the formal run

The formal run starts only when every gate holds; each is recorded in PR #1550.

1. **Pilot**: one representative profile per Task completes at the formal `n` (Section 9), on the final code.
2. **Ledger**: the per-model ledger from the smoke run and the pilot's steady-state request times predicts at most
   22 hours on one GB300, failure allowance included; otherwise the levers of Section 9, then a report to the owner.
3. **Order**: the order check (above) runs on five profiles across the server sizes with a resolved result each;
   where an effect exceeds 2%, the size class's mitigation (interleaved runs, or native, TRTMC, native timing that is
   `perf-inconclusive` when the two native timings differ by more than 2%) is implemented and tested.
4. **Blocked profiles**: DINOv3's kNN scorer once the checkpoints are accessible; Lance's resident adapter; s1-mini
   once the catalog builds it. A profile still blocked is reported with its reason, not dropped.
5. **Smoke rerun**: every ready profile `smoke-pass` (or an accepted, reported finding) on the final code.
6. **Frozen inputs**: the per-suite counts, sample manifests, and pinned revisions frozen; the execution matrix
   regenerated from them.
7. **Review**: codex approves the final code, the smoke results, the pilot, and the ledger (Section 11).
