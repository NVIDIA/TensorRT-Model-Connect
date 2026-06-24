# Task Eval Suite Workflow

Use this checklist when adding or changing a `tools/task_eval.py` validation
suite.

## 1. Keep Dataset Shape Separate From Runtime Business Logic

- Put suite configuration in `tests/task_eval/validation_suites.yaml`.
- Put dataset conversion scripts next to downloaded benchmark assets under
  `.benchmarks/<dataset>/`.
- Convert raw datasets into a framework-neutral manifest before wiring them
  into `task_eval.py`.
- Do not put TRTMC-only suite YAML or model-selection logic inside dataset
  directories.
- Do not commit benchmark images, parquet shards, or other large downloaded
  assets unless the repository already tracks that asset class.

## 2. Define The Evaluation Contract First

Before editing model selectors, write down:

- the user-visible task contract, such as `vl_answer`, `ocr_text`, or a new
  contract if the existing ones do not fit;
- the dataset kind consumed by `prepare_task_dataset`;
- the scorer used for absolute task quality;
- the HF/reference behavior: cached HF predictions, force-rerun support, or a
  documented reference backend exception;
- whether agreement means text equality, token parity, or correctness parity.

Do not reuse a broad VLM suite for a specialized task unless the scoring and
prompt contract are also valid for that task.

## 3. Add A Suite In YAML

Every suite entry should include:

- stable `id`;
- `description` explaining the task and non-goals;
- `user_contract`;
- `dataset.kind`, `default_path`, prompt/answer/media fields, and any
  normalization contract;
- `selectors` that pick only the intended model families;
- deterministic `generation` settings;
- `scoring.scorer`;
- `build.min_max_cache_length` if the dataset prompt/image prefill requires it;
- `ci` status and notes.

Keep model-family selection narrow. If a model family is already covered by a
more appropriate suite, do not add it to a specialized suite just because it can
produce text.

## 4. Add Or Reuse A Dataset Adapter

Prefer existing dataset kinds:

- `mmlu_five_shot_json` for text multiple-choice prompts;
- `vlm_chat_json` for image-conditioned VLM chat samples;
- `vlm_unified_json` for framework-neutral VLM/OCR manifests with media,
  messages, answers, and metadata.

When a new shape is needed, add a focused `prepare_*_dataset` function, validate
all image/media assets up front, and write `answers.json`, `prompts.jsonl`, and
`manifest.json` with enough metadata for offline scoring.

## 5. Add Scoring With Tests

Scorers must be explicit and unit-tested. Add tests for:

- correct and incorrect predictions;
- aliases or task-specific gold fields;
- skipped/error outputs;
- HF/TRTFB comparison behavior;
- offline `score` or `compare` reuse when applicable.

If a benchmark's official metric requires heavy optional dependencies, either
make the dependency explicit or mark any built-in approximation in the sample
metadata so reviewers do not mistake it for the official metric.

## 6. Validate Locally Before PR

Run the narrow unit test first:

```bash
python -m pytest tests/tools/test_task_eval.py -q
```

Then run at least one CLI smoke on a small slice:

```bash
python tools/task_eval.py prepare --suite <suite_id> --dataset <dataset.json> --work-dir /tmp/<suite> --limit 2
python tools/task_eval.py score --work-dir /tmp/<suite> --scorer <scorer>
```

For p2021/container validation, use the container Python and dataset path used
by the target environment rather than changing the CI venv.

