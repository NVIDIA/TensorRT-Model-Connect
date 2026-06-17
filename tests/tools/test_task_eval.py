from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools import task_eval


def _write_mmlu(path: Path) -> None:
    payload = {
        "apply_chat_template": False,
        "batch_size": 1,
        "max_generate_length": 1,
        "temperature": 1.0,
        "top_k": 50,
        "top_p": 1.0,
        "requests": [
            {
                "messages": [{"role": "user", "content": "Question one\nA. a\nB. b\nAnswer:"}],
                "answer": "B",
                "subject": "subject_a",
            },
            {
                "messages": [{"role": "user", "content": "Question two\nA. a\nB. b\nAnswer:"}],
                "answer": "A",
                "subject": "subject_b",
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_vlm_mmmu_pro_vision(path: Path) -> None:
    image_path = path.parent / "images" / "sample.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"fake image bytes")
    payload = {
        "batch_size": 1,
        "max_generate_length": 8,
        "temperature": 1.0,
        "top_k": 1,
        "top_p": 1.0,
        "requests": [
            {
                "messages": [
                    {
                        "role": "system",
                        "content": "Answer with the option letter.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": "mmmu_pro_vision/images/sample.jpg",
                            },
                            {
                                "type": "text",
                                "text": "Which letter is correct?\nA. no\nJ. yes\n\nAnswer directly.",
                            },
                        ],
                    },
                ],
                "answer": "J",
                "id": "test_case_1",
                "subject": "History",
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_plan_selects_chat_text_generation_manifests() -> None:
    suites = task_eval.load_suites()
    models = task_eval.load_manifest_records()

    rows = task_eval.build_plan(suites, models, suite_id="mmlu_five_shot_mcq")

    selected = {row["model"]: row for row in rows}
    assert "qwen3-0.6b-fp16" in selected
    assert selected["qwen3-0.6b-fp16"]["runtime_strategy"] == "decoder_kv_cache"
    assert selected["qwen3-0.6b-fp16"]["user_contract"] == "chat_response"
    assert "gpt2-125m" not in selected
    assert "codegen-350m" not in selected


def test_plan_selects_vlm_mmmu_pro_vision_models() -> None:
    suites = task_eval.load_suites()
    models = task_eval.load_manifest_records()

    rows = task_eval.build_plan(suites, models, suite_id="vlm_mmmu_pro_vision_mcq")

    selected = {row["model"]: row for row in rows}
    assert "qwen25vl-3b" in selected
    assert selected["qwen25vl-3b"]["runtime_strategy"] == "vision_language"
    assert "qwen3-vl-2b" in selected
    assert "internvl3-2b" in selected
    assert "deepseek-ocr-l0" not in selected
    assert "locateanything-3b" not in selected


def test_prepare_mmlu_writes_answers_and_trtfb_jsonl(tmp_path: Path) -> None:
    dataset = tmp_path / "mmlu.json"
    _write_mmlu(dataset)
    suite = task_eval.suite_by_id(task_eval.load_suites(), "mmlu_five_shot_mcq")

    outputs = task_eval.prepare_mmlu_dataset(
        dataset_path=dataset,
        work_dir=tmp_path / "work",
        suite=suite,
        limit=1,
    )

    answers = json.loads(outputs["answers"].read_text(encoding="utf-8"))
    prompts = task_eval.load_jsonl(outputs["prompts"])
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))

    assert len(answers["requests"]) == 1
    assert prompts == [{
        "sample_id": "mmlu_000000",
        "dataset_index": 0,
        "eval_index": 0,
        "subject": "subject_a",
        "answer": "B",
        "prompt": "Question one\nA. a\nB. b\nAnswer:",
    }]
    assert manifest["suite"] == "mmlu_five_shot_mcq"
    assert manifest["request_count"] == 1


def test_prepare_vlm_mmmu_pro_vision_writes_image_prompt_jsonl(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "MMMU_Pro_vision"
    dataset_dir.mkdir()
    dataset = dataset_dir / "mmmu_pro_vision_dataset.json"
    _write_vlm_mmmu_pro_vision(dataset)
    suite = task_eval.suite_by_id(task_eval.load_suites(), "vlm_mmmu_pro_vision_mcq")

    outputs = task_eval.prepare_vlm_chat_dataset(
        dataset_path=dataset,
        work_dir=tmp_path / "work",
        suite=suite,
        limit=1,
    )

    answers = json.loads(outputs["answers"].read_text(encoding="utf-8"))
    prompts = task_eval.load_jsonl(outputs["prompts"])
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))

    assert len(answers["requests"]) == 1
    assert prompts == [{
        "sample_id": "test_case_1",
        "dataset_index": 0,
        "eval_index": 0,
        "subject": "History",
        "answer": "J",
        "prompt": "Answer with the option letter.\n\nWhich letter is correct?\nA. no\nJ. yes\n\nAnswer directly.",
        "images": [str(dataset_dir / "images" / "sample.jpg")],
    }]
    assert manifest["suite"] == "vlm_mmmu_pro_vision_mcq"
    assert manifest["dataset_kind"] == "vlm_chat_json"
    assert manifest["request_count"] == 1
    assert manifest["image_count"] == 1
    assert "reference" not in manifest


def test_prepare_vlm_fixed_suite_normalizes_image_and_messages(
    tmp_path: Path, monkeypatch
) -> None:
    dataset_dir = tmp_path / "MMMU_Pro_vision"
    dataset_dir.mkdir()
    dataset = dataset_dir / "mmmu_pro_vision_dataset.json"
    _write_vlm_mmmu_pro_vision(dataset)
    resize_calls: list[tuple[Path, Path, int]] = []

    def fake_resize(src: Path, dst: Path, image_size: int) -> None:
        resize_calls.append((src, dst, image_size))
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"fixed image")

    monkeypatch.setattr(task_eval, "_resize_image_to_square", fake_resize)
    suite = task_eval.suite_by_id(task_eval.load_suites(), "vlm_mmmu_pro_vision_fixed_mcq")

    outputs = task_eval.prepare_vlm_chat_dataset(
        dataset_path=dataset,
        work_dir=tmp_path / "work",
        suite=suite,
        limit=1,
    )

    answers = json.loads(outputs["answers"].read_text(encoding="utf-8"))
    prompts = task_eval.load_jsonl(outputs["prompts"])
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    fixed_image = tmp_path / "work" / "images" / "test_case_1.png"
    merged_prompt = (
        "Answer with the option letter.\n\n"
        "Which letter is correct?\nA. no\nJ. yes\n\nAnswer directly."
    )

    assert fixed_image.is_file()
    assert fixed_image.read_bytes() == b"fixed image"
    assert resize_calls == [(dataset_dir / "images" / "sample.jpg", fixed_image, 448)]
    assert prompts[0]["prompt"] == merged_prompt
    assert prompts[0]["images"] == [str(fixed_image)]
    assert answers["requests"][0]["messages"] == [{
        "role": "user",
        "content": [
            {"type": "image", "image": str(fixed_image)},
            {"type": "text", "text": merged_prompt},
        ],
    }]
    assert manifest["normalization"] == {
        "image_size": 448,
        "prompt_contract": "single_user_image_first",
    }


def test_vlm_reference_prompt_uses_native_messages() -> None:
    class FakeProcessor:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            assert tokenize is False
            assert add_generation_prompt is True
            return json.dumps(messages)

    request = {
        "messages": [
            {"role": "system", "content": "system text"},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "original.jpg"},
                    {"type": "text", "text": "user text"},
                ],
            },
        ]
    }

    rendered = task_eval._vlm_chat_text(
        FakeProcessor(),
        request,
        "flattened prompt",
        "Qwen/Qwen3-VL-2B-Instruct",
    )

    messages = json.loads(rendered)
    assert messages == request["messages"]


def test_prepare_cli_accepts_vlm_dataset_kind(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "MMMU_Pro_vision"
    dataset_dir.mkdir()
    dataset = dataset_dir / "mmmu_pro_vision_dataset.json"
    _write_vlm_mmmu_pro_vision(dataset)
    work_dir = tmp_path / "work"

    rc = task_eval.cmd_prepare(argparse.Namespace(
        suites=str(task_eval.DEFAULT_SUITES),
        suite="vlm_mmmu_pro_vision_mcq",
        dataset=str(dataset),
        work_dir=str(work_dir),
        limit=1,
        subject="",
        sample_seed=None,
    ))

    assert rc == 0
    assert task_eval.load_jsonl(work_dir / "prompts.jsonl")[0]["images"] == [
        str(dataset_dir / "images" / "sample.jpg")
    ]


def test_continuation_parity_exact_and_first_divergence() -> None:
    hf = {"responses": [
        {"sample_id": "a", "output_text": "the cat sat"},
        {"sample_id": "b", "output_text": "hello world"},
    ]}
    trtfb = {"responses": [
        {"sample_id": "a", "output_text": "the cat sat"},
        {"sample_id": "b", "output_text": "hello there"},
    ]}

    summary = task_eval.compare_continuation_sets(hf, trtfb, tokenize=lambda s: s.split())

    assert summary["count"] == 2
    assert summary["exact_match_rate"] == 0.5          # "a" exact, "b" not
    assert summary["samples"][0]["first_divergence"] == 3  # all 3 tokens match
    assert summary["samples"][1]["first_divergence"] == 1  # diverge at token index 1
    # matched prefixes 3 + 1 = 4, ref token counts 3 + 2 = 5
    assert abs(summary["token_prefix_agreement"] - 4 / 5) < 1e-9


def test_continuation_parity_prefers_generated_token_ids() -> None:
    hf = {"responses": [
        {"sample_id": "a", "output_text": "same text", "generated_token_ids": [10, 20]},
        {"sample_id": "b", "output_text": "same text", "generated_token_ids": [1, 2, 3]},
    ]}
    trtfb = {"responses": [
        {"sample_id": "a", "output_text": "same text", "generated_token_ids": [10, 20]},
        {"sample_id": "b", "output_text": "same text", "generated_token_ids": [1, 2, 4]},
    ]}

    summary = task_eval.compare_continuation_sets(hf, trtfb, require_token_ids=True)

    assert summary["comparison_granularity"] == "generated_token_ids"
    assert summary["exact_match_rate"] == 0.5
    assert summary["token_id_exact_match_rate"] == 0.5
    assert summary["text_exact_match_rate"] == 1.0
    assert summary["samples"][1]["first_divergence"] == 2
    assert summary["samples"][1]["hf_token_at_divergence"] == 3
    assert summary["samples"][1]["trtfb_token_at_divergence"] == 4


def test_continuation_parity_requires_token_ids_when_requested() -> None:
    hf = {"responses": [{"sample_id": "a", "output_text": "x"}]}
    trtfb = {"responses": [{"sample_id": "a", "output_text": "x"}]}

    try:
        task_eval.compare_continuation_sets(hf, trtfb, require_token_ids=True)
    except ValueError as exc:
        assert "generated_token_ids" in str(exc)
    else:
        raise AssertionError("expected missing token-id validation failure")


def test_validation_suites_keep_continuation_and_drop_trace_cloze() -> None:
    suites = task_eval.load_suites()
    ids = {suite["id"] for suite in suites}
    continuation = task_eval.suite_by_id(suites, "mmlu_continuation_parity")

    assert "mmlu_continuation_parity" in ids
    assert "mmlu_trace_cloze" not in ids
    assert continuation["dataset"]["kind"] == "mmlu_five_shot_json"
    assert continuation["scoring"]["scorer"] == "continuation"
    assert continuation["user_contract"] == "continuation_parity"


def test_compare_continuation_cli_writes_json_summary(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "hf_predictions.json").write_text(
        json.dumps({
            "responses": [
                {"sample_id": "a", "output_text": "same", "generated_token_ids": [1, 2]},
                {"sample_id": "b", "output_text": "left", "generated_token_ids": [3, 4]},
            ]
        }),
        encoding="utf-8",
    )
    (work_dir / "trtfb_predictions.json").write_text(
        json.dumps({
            "responses": [
                {"sample_id": "a", "output_text": "same", "generated_token_ids": [1, 2]},
                {"sample_id": "b", "output_text": "right", "generated_token_ids": [3, 5]},
            ]
        }),
        encoding="utf-8",
    )
    output = tmp_path / "continuation.json"

    rc = task_eval.cmd_compare_continuation(argparse.Namespace(
        work_dir=str(work_dir),
        hf_predictions="",
        trtfb_predictions="",
        model="",
        trust_remote_code=False,
        local_files_only=False,
        output=str(output),
    ))

    summary = json.loads(output.read_text(encoding="utf-8"))
    assert rc == 0
    assert summary["comparison_granularity"] == "generated_token_ids"
    assert summary["exact_match_rate"] == 0.5
    assert summary["token_prefix_agreement"] == 0.75
    assert summary["samples"][1]["first_divergence"] == 1


def test_convert_trtfb_uses_generated_text_field(tmp_path: Path) -> None:
    raw = tmp_path / "trtfb_raw.jsonl"
    raw.write_text(
        json.dumps({
            "sample_id": "mmlu_000000",
            "gold_answer": "B",
            "pred_answer": "",
            "text": "Answer: B",
            "generated_tokens": 1,
            "generated_token_ids": [42],
            "wall_ms": 3.5,
        }) + "\n",
        encoding="utf-8",
    )
    predictions = tmp_path / "predictions.json"

    task_eval.convert_trtfb_jsonl_to_predictions(raw, predictions)

    payload = json.loads(predictions.read_text(encoding="utf-8"))
    assert payload["responses"][0]["output_text"] == "Answer: B"
    assert payload["responses"][0]["generated_token_ids"] == [42]
    assert payload["responses"][0]["source"] == "trtfb"


def test_score_and_compare_mmlu_predictions(tmp_path: Path) -> None:
    dataset = tmp_path / "mmlu.json"
    _write_mmlu(dataset)
    answers = json.loads(dataset.read_text(encoding="utf-8"))
    hf = {
        "responses": [
            {"sample_id": "mmlu_000000", "output_text": "b"},
            {"sample_id": "mmlu_000001", "output_text": "Answer: A"},
        ]
    }
    trtfb = {
        "responses": [
            {"sample_id": "mmlu_000000", "output_text": "B<|im_end|>"},
            {"sample_id": "mmlu_000001", "output_text": "(B)"},
        ]
    }

    hf_score = task_eval.score_predictions(hf, answers)
    summary = task_eval.compare_prediction_sets(hf, trtfb, answers)

    assert hf_score["overall_accuracy"] == 1.0
    assert summary["hf"]["overall_accuracy"] == 1.0
    assert summary["trtfb"]["overall_accuracy"] == 0.5
    assert summary["accuracy_delta_trtfb_minus_hf"] == -0.5
    assert summary["prediction_agreement_rate"] == 0.5
    assert summary["buckets"]["hf_correct_trtfb_wrong"] == 1


def test_score_predictions_parses_vlm_a_to_j_choices() -> None:
    answers = {"requests": [{"answer": "J", "subject": "History"}]}
    predictions = {"responses": [{"sample_id": "test_case_1", "output_text": "Answer: J"}]}

    score = task_eval.score_predictions(predictions, answers)

    assert score["overall_accuracy"] == 1.0
    assert score["samples"][0]["parsed_prediction"] == "J"


def test_selected_models_for_suite_accepts_manifest_name() -> None:
    suite = task_eval.suite_by_id(task_eval.load_suites(), "mmlu_five_shot_mcq")
    models = task_eval.load_manifest_records()

    selected = task_eval.selected_models_for_suite(
        suite,
        models,
        selectors=["qwen3-0.6b-fp16"],
        single_device_only=True,
    )

    assert [model["name"] for model in selected] == ["qwen3-0.6b-fp16"]


def test_waives_exclude_default_selection_but_explicit_model_can_debug(tmp_path: Path) -> None:
    suite = {
        "id": "mmlu_five_shot_mcq",
        "selectors": {
            "task_strategies": ["text_generation_causal"],
            "runtime_strategies": ["decoder_kv_cache"],
            "user_contracts": ["chat_response"],
        },
    }
    models = [
        {
            "name": "internlm2-1.8b",
            "hf_id": "internlm/internlm2-math-plus-1_8b",
            "bundle": "internlm2-1.8b.trtfb",
            "runtime_strategy": "decoder_kv_cache",
            "task_strategy": "text_generation_causal",
            "reference_family": "chat_instruct_template",
            "user_contract": "chat_response",
            "family": "internlm",
            "ci_tier": "default",
            "requires_multi_device": False,
            "manifest": "tests/e2e/models/internlm2-1.8b.json",
            "skip": "",
        },
        {
            "name": "tinyllama-1.1b",
            "hf_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "bundle": "tinyllama-1.1b.trtfb",
            "runtime_strategy": "decoder_kv_cache",
            "task_strategy": "text_generation_causal",
            "reference_family": "chat_instruct_template",
            "user_contract": "chat_response",
            "family": "llama",
            "ci_tier": "default",
            "requires_multi_device": False,
            "manifest": "tests/e2e/models/tinyllama-1.1b.json",
            "skip": "",
        },
    ]
    waives_path = tmp_path / "waives.txt"
    waives_path.write_text(
        "internlm2-1.8b  SKIP  (HF remote code incompatible)\n",
        encoding="utf-8",
    )
    waives = task_eval.load_waives(waives_path)

    selected = task_eval.selected_models_for_suite(suite, models, waives=waives)
    explicit = task_eval.selected_models_for_suite(
        suite,
        models,
        selectors=["internlm2-1.8b"],
        waives=waives,
    )
    rows = task_eval.build_plan([suite], models, include_non_matching=True, waives=waives)
    internlm_row = next(row for row in rows if row["model"] == "internlm2-1.8b")

    assert [model["name"] for model in selected] == ["tinyllama-1.1b"]
    assert [model["name"] for model in explicit] == ["internlm2-1.8b"]
    assert internlm_row["selected"] is False
    assert "waived SKIP" in internlm_row["reason"]


def test_build_bundle_command_uses_manifest_build_settings(tmp_path: Path) -> None:
    model = {
        "name": "case",
        "hf_id": "org/model",
        "max_cache_length": 512,
        "precision": "bf16",
        "trust_remote_code": True,
        "build_args": {"backend": "trt", "parallel": {"mode": "tensor_parallel", "tp_size": 2}},
        "quantization": {"format": "fp8", "calibration_samples": 4},
    }

    cmd = task_eval.build_bundle_command(
        model,
        trtmc_binary="build/trtmc",
        bundle_path=tmp_path / "case.trtfb",
        extra_build_args=["--verbose"],
    )

    assert cmd[:4] == ["build/trtmc", "build", "org/model", "-o"]
    assert "--max-cache-length" in cmd
    assert "512" in cmd
    assert ["--method", "trt"] == cmd[cmd.index("--method"):cmd.index("--method") + 2]
    assert ["--tp-size", "2"] == cmd[cmd.index("--tp-size"):cmd.index("--tp-size") + 2]
    assert ["--precision", "bf16"] == cmd[cmd.index("--precision"):cmd.index("--precision") + 2]
    assert "--trust-remote-code" in cmd
    assert "--verbose" in cmd


def test_suite_build_cache_minimum_overrides_manifest_cache() -> None:
    suite = {"build": {"min_max_cache_length": 1024}}
    model = {"max_cache_length": 256}

    assert task_eval.requested_build_max_cache_length(suite, model) == 1024
    assert task_eval.requested_build_max_cache_length(suite, model, prompt_max_tokens=2048) == 2048
    assert task_eval.requested_build_max_cache_length(suite, model, 512) == 512


def test_prompt_length_validation_rejects_over_cache(tmp_path: Path, monkeypatch) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "prompts.jsonl").write_text(
        json.dumps({"prompt": "long prompt"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(task_eval, "max_prompt_token_length", lambda **_kwargs: 513)

    try:
        task_eval.validate_prompt_lengths_for_cache(
            model={"name": "case", "hf_id": "org/model"},
            work_dir=work_dir,
            max_cache_length=512,
        )
    except RuntimeError as exc:
        assert "max_prompt_tokens=513" in str(exc)
    else:
        raise AssertionError("expected prompt length validation failure")


def test_eval_one_model_reuses_hf_builds_bundle_and_reruns_trtfb(
    tmp_path: Path, monkeypatch
) -> None:
    dataset = tmp_path / "mmlu.json"
    _write_mmlu(dataset)
    suite = task_eval.suite_by_id(task_eval.load_suites(), "mmlu_five_shot_mcq")
    model = {
        "name": "gpt2-125m",
        "hf_id": "openai-community/gpt2",
        "bundle": "gpt2-125m.trtfb",
        "max_cache_length": 256,
        "precision": "fp32",
        "trust_remote_code": False,
        "build_args": {},
        "quantization": {},
    }
    work_dir = tmp_path / "work" / suite["id"] / model["name"]
    work_dir.mkdir(parents=True)
    task_eval.prepare_mmlu_dataset(
        dataset_path=dataset,
        work_dir=work_dir,
        suite=suite,
    )
    (work_dir / "hf_predictions.json").write_text(
        json.dumps({
            "responses": [
                {"sample_id": "mmlu_000000", "output_text": "B"},
                {"sample_id": "mmlu_000001", "output_text": "A"},
            ]
        }),
        encoding="utf-8",
    )
    calls: list[str] = []

    def fake_run_hf(_args):
        calls.append("hf")
        raise AssertionError("HF should be reused")

    monkeypatch.setattr(task_eval, "max_prompt_token_length", lambda **_kwargs: 405)

    def fake_ensure_bundle(*_args, **kwargs):
        calls.append("build")
        assert kwargs["max_cache_length"] == 405
        bundle = kwargs["bundle_path"]
        bundle.parent.mkdir(parents=True, exist_ok=True)
        bundle.write_bytes(b"bundle")
        return bundle, True

    def fake_run_trtfb(args):
        calls.append(f"trtfb-seed={args.seed}")
        Path(args.work_dir, "trtfb_predictions.json").write_text(
            json.dumps({
                "responses": [
                    {"sample_id": "mmlu_000000", "output_text": "B"},
                    {"sample_id": "mmlu_000001", "output_text": "B"},
                ]
            }),
            encoding="utf-8",
        )

    monkeypatch.setattr(task_eval, "run_hf_reference", fake_run_hf)
    monkeypatch.setattr(task_eval, "ensure_bundle", fake_ensure_bundle)
    monkeypatch.setattr(task_eval, "run_trtfb", fake_run_trtfb)

    args = argparse.Namespace(
        work_root=str(tmp_path / "work"),
        dataset=str(dataset),
        limit=0,
        subject="",
        sample_seed=None,
        force_hf=False,
        force_build=False,
        build_max_cache_length=None,
        skip_prompt_length_check=False,
        bundle="",
        model=["gpt2-125m"],
        engine_dir="",
        trtmc_binary="build/trtmc",
        extra_build_arg=[],
        hf_dtype="auto",
        hf_device="cuda",
        hf_device_map="",
        hf_attn_impl="",
        trust_remote_code=False,
        local_files_only=True,
        do_sample=False,
        apply_chat_template=False,
        max_new_tokens=None,
        temperature=None,
        top_k=None,
        top_p=None,
        min_p=None,
        seed=123,
        benchmark_binary="build/trtmc_dataset_benchmark",
        hf_python="",
        backend_dir="",
        kv_cache_size="",
        config="",
        set=[],
        cuda_visible_devices="",
        chat_template=False,
    )

    result = task_eval.eval_one_model(suite=suite, model=model, args=args)

    assert calls == ["build", "trtfb-seed=123"]
    assert result["hf_reused"] is True
    assert result["bundle_built"] is True
    assert result["trtfb_accuracy"] == 0.5
    assert (work_dir / "summary.json").is_file()


def test_eval_one_model_uses_vlm_prepare_outputs_for_vlm_suite(
    tmp_path: Path, monkeypatch
) -> None:
    dataset_dir = tmp_path / "MMMU_Pro_vision"
    dataset_dir.mkdir()
    dataset = dataset_dir / "mmmu_pro_vision_dataset.json"
    _write_vlm_mmmu_pro_vision(dataset)
    suite = task_eval.suite_by_id(task_eval.load_suites(), "vlm_mmmu_pro_vision_mcq")
    model = {
        "name": "qwen25vl-3b",
        "hf_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "bundle": "qwen25vl-3b-vl.trtfb",
        "max_cache_length": 512,
        "precision": "fp32",
        "trust_remote_code": False,
        "build_args": {},
        "quantization": {},
    }
    calls: list[str] = []

    def fake_run_hf(_args, _model, work_dir):
        calls.append("hf")
        prompts = task_eval.load_jsonl(work_dir / "prompts.jsonl")
        assert prompts[0]["images"] == [str(dataset_dir / "images" / "sample.jpg")]
        Path(work_dir, "hf_predictions.json").write_text(
            json.dumps({"responses": [{"sample_id": "test_case_1", "output_text": "J"}]}),
            encoding="utf-8",
        )

    def fake_ensure_bundle(*_args, **kwargs):
        calls.append("build")
        bundle = kwargs["bundle_path"]
        bundle.parent.mkdir(parents=True, exist_ok=True)
        bundle.write_bytes(b"bundle")
        return bundle, True

    def fake_run_trtfb(args):
        calls.append("trtfb")
        prompts = task_eval.load_jsonl(Path(args.work_dir) / "prompts.jsonl")
        assert prompts[0]["images"] == [str(dataset_dir / "images" / "sample.jpg")]
        Path(args.work_dir, "trtfb_predictions.json").write_text(
            json.dumps({"responses": [{"sample_id": "test_case_1", "output_text": "Answer: J"}]}),
            encoding="utf-8",
        )

    monkeypatch.setattr(task_eval, "run_hf_reference_subprocess", fake_run_hf)
    monkeypatch.setattr(task_eval, "ensure_bundle", fake_ensure_bundle)
    monkeypatch.setattr(task_eval, "run_trtfb", fake_run_trtfb)
    monkeypatch.setattr(task_eval, "max_prompt_token_length", lambda **_kwargs: 128)

    args = argparse.Namespace(
        work_root=str(tmp_path / "work"),
        dataset=str(dataset),
        limit=1,
        subject="",
        sample_seed=None,
        force_hf=False,
        force_build=False,
        build_max_cache_length=None,
        skip_prompt_length_check=False,
        bundle="",
        model=["qwen25vl-3b"],
        engine_dir="",
        trtmc_binary="build/trtmc",
        extra_build_arg=[],
        hf_dtype="auto",
        hf_device="cuda",
        hf_device_map="",
        hf_attn_impl="",
        trust_remote_code=False,
        local_files_only=True,
        do_sample=False,
        apply_chat_template=False,
        max_new_tokens=None,
        temperature=None,
        top_k=None,
        top_p=None,
        min_p=None,
        seed=123,
        benchmark_binary="build/trtmc_dataset_benchmark",
        hf_python="",
        backend_dir="",
        kv_cache_size="",
        config="",
        set=[],
        cuda_visible_devices="",
        chat_template=False,
    )

    result = task_eval.eval_one_model(suite=suite, model=model, args=args)

    assert calls == ["hf", "build", "trtfb"]
    assert result["trtfb_accuracy"] == 1.0
    assert result["prediction_agreement_rate"] == 1.0


def test_eval_records_model_failure_and_continues(tmp_path: Path, monkeypatch) -> None:
    suite = {"id": "mmlu_five_shot_mcq", "dataset": {"kind": "mmlu_five_shot_json"}}
    models = [
        {"name": "gated", "hf_id": "org/gated", "bundle": "gated.trtfb"},
        {"name": "ok", "hf_id": "org/ok", "bundle": "ok.trtfb"},
    ]

    monkeypatch.setattr(task_eval, "load_suites", lambda *_args, **_kwargs: [suite])
    monkeypatch.setattr(task_eval, "load_manifest_records", lambda *_args, **_kwargs: models)
    monkeypatch.setattr(
        task_eval,
        "selected_models_for_suite",
        lambda *_args, **_kwargs: models,
    )

    def fake_eval_one_model(*_args, model, **_kwargs):
        if model["name"] == "gated":
            raise RuntimeError("gated repo")
        return {
            "suite": suite["id"],
            "model": "ok",
            "hf_id": "org/ok",
            "work_dir": str(tmp_path / "work" / suite["id"] / "ok"),
            "bundle": str(tmp_path / "bundles" / "ok.trtfb"),
            "hf_accuracy": 1.0,
            "trtfb_accuracy": 1.0,
            "prediction_agreement_rate": 1.0,
            "hf_reused": False,
            "bundle_built": True,
        }

    monkeypatch.setattr(task_eval, "eval_one_model", fake_eval_one_model)

    args = argparse.Namespace(
        suites="",
        suite=suite["id"],
        models_dir="",
        waives="",
        waive_platform="",
        include_waived=False,
        model=[],
        single_device_only=True,
        bundle="",
        work_root=str(tmp_path / "work"),
        engine_dir=str(tmp_path / "bundles"),
        fail_fast=False,
        disable_model_process_isolation=True,
    )

    assert task_eval.cmd_eval(args) == 0

    summary = json.loads(
        (tmp_path / "work" / suite["id"] / "eval_summary.json").read_text(encoding="utf-8")
    )
    assert summary["count"] == 2
    assert summary["passed_count"] == 1
    assert summary["failed_count"] == 1
    assert summary["results"][0]["status"] == "failed"
    assert summary["results"][0]["error"] == "gated repo"
    assert summary["results"][1]["status"] == "passed"
    assert summary["results"][1]["model"] == "ok"


def test_eval_stops_after_oom_when_gpu_cleanup_is_not_confirmed(tmp_path: Path, monkeypatch) -> None:
    suite = {"id": "mmlu_five_shot_mcq", "dataset": {"kind": "mmlu_five_shot_json"}}
    models = [
        {"name": "oom", "hf_id": "org/oom", "bundle": "oom.trtfb"},
        {"name": "next", "hf_id": "org/next", "bundle": "next.trtfb"},
    ]

    monkeypatch.setattr(task_eval, "load_suites", lambda *_args, **_kwargs: [suite])
    monkeypatch.setattr(task_eval, "load_manifest_records", lambda *_args, **_kwargs: models)
    monkeypatch.setattr(
        task_eval,
        "selected_models_for_suite",
        lambda *_args, **_kwargs: models,
    )
    calls: list[str] = []

    def fake_run_worker(*_args, model, **_kwargs):
        calls.append(model["name"])
        return {
            "suite": suite["id"],
            "model": model["name"],
            "hf_id": model["hf_id"],
            "work_dir": str(tmp_path / "work" / suite["id"] / model["name"]),
            "bundle": str(tmp_path / "bundles" / model["bundle"]),
            "status": "failed",
            "error_type": "RuntimeError",
            "error": "CUDA out of memory",
            "worker_log": str(tmp_path / "work" / suite["id"] / model["name"] / "eval_worker.log"),
            "gpu_cleanup_confirmed": False,
        }

    monkeypatch.setattr(task_eval, "run_eval_model_worker", fake_run_worker)

    args = argparse.Namespace(
        suites="",
        suite=suite["id"],
        models_dir="",
        waives="",
        waive_platform="",
        include_waived=False,
        model=[],
        single_device_only=True,
        bundle="",
        work_root=str(tmp_path / "work"),
        engine_dir=str(tmp_path / "bundles"),
        fail_fast=False,
    )

    assert task_eval.cmd_eval(args) == 0

    summary = json.loads(
        (tmp_path / "work" / suite["id"] / "eval_summary.json").read_text(encoding="utf-8")
    )
    assert calls == ["oom"]
    assert summary["count"] == 2
    assert summary["failed_count"] == 1
    assert summary["skipped_count"] == 1
    assert summary["model_process_isolation"] is True
    assert summary["results"][1]["status"] == "skipped"
    assert "GPU cleanup" in summary["results"][1]["reason"]
