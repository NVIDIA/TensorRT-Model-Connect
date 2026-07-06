# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import struct
import wave
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


def _write_pcm_wav(path: Path, *, seconds: float = 1.0, sample_rate: int = 24000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = [1000 if index % 2 else -1000 for index in range(int(seconds * sample_rate))]
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(struct.pack(f"<{len(samples)}h", *samples))


def _write_seedtts(path: Path) -> None:
    reference_wav = path.parent / "reference.wav"
    _write_pcm_wav(reference_wav)
    payload = {
        "speaker": "ryan",
        "requests": [
            {
                "id": "seedtts-1",
                "messages": [{"role": "assistant", "content": "The test sentence."}],
                "reference": "The test sentence.",
                "reference_wav": "reference.wav",
                "prompt_text": "A speaker prompt.",
            }
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


def _write_ocrbench_unified(path: Path) -> None:
    image_path = path.parent / "images" / "ocrbench_v2_000000.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"fake image bytes")
    payload = {
        "schema_version": "1.0",
        "dataset": "OCRBench_v2",
        "samples": [
            {
                "id": "ocrbench_v2_000000",
                "source_index": 0,
                "dataset_name": "rico",
                "category": "APP agent en",
                "type": "APP agent en",
                "question": "What is the wrong answer 2?",
                "media": [{"type": "image", "path": "images/ocrbench_v2_000000.jpg"}],
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "path": "images/ocrbench_v2_000000.jpg"},
                            {"type": "text", "text": "What is the wrong answer 2?"},
                        ],
                    }
                ],
                "answer": {"primary": "enabled", "aliases": ["enabled", "on"]},
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_asr_librispeech(path: Path) -> None:
    audio_path = path.parent / "audio" / "sample.wav"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.write_bytes(b"fake wav bytes")
    payload = {
        "dataset": "librispeech_clean_test",
        "requests": [
            {
                "id": "clean_000000",
                "subset": "test-clean",
                "reference": "The quick brown fox",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio", "audio": "audio/sample.wav"},
                            {"type": "text", "text": "Transcribe this audio."},
                        ],
                    }
                ],
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_default_suites_include_ocrbench_v2_unified() -> None:
    suites = task_eval.load_suites()
    suite = task_eval.suite_by_id(suites, "ocrbench_v2_unified")

    assert suite["dataset"]["kind"] == "vlm_unified_json"
    assert suite["scoring"]["scorer"] == "ocrbench_v2"
    assert suite["selectors"]["runtime_strategies"] == ["deepseek_ocr_vision_language"]
    assert suite["selectors"]["families"] == ["deepseek_ocr"]


def test_default_suites_include_librispeech_clean_asr() -> None:
    suites = task_eval.load_suites()
    suite = task_eval.suite_by_id(suites, "librispeech_clean_asr")

    assert suite["dataset"]["kind"] == "asr_chat_json"
    assert suite["scoring"]["scorer"] == "asr_transcript"
    assert suite["selectors"]["runtime_strategies"] == [
        "whisper_speech_to_text",
        "canary_speech_to_text",
    ]
    assert suite["selectors"]["families"] == ["whisper", "canary"]


def test_default_suites_do_not_split_librispeech_asr_by_family() -> None:
    suite_ids = {suite["id"] for suite in task_eval.load_suites()}

    assert "librispeech_clean_asr" in suite_ids
    assert "librispeech_clean_asr_whisper" not in suite_ids
    assert "librispeech_clean_asr_canary" not in suite_ids


def test_default_suites_include_seedtts_tts_intelligibility() -> None:
    suite = task_eval.suite_by_id(task_eval.load_suites(), "seedtts_en_tts_intelligibility")

    assert suite["dataset"]["kind"] == "seedtts_json"
    assert suite["scoring"]["scorer"] == "tts_intelligibility"
    assert suite["default_model_names"] == [
        "bark-large",
        "bark-small",
        "magpie-tts-357m",
    ]


def test_default_suites_include_librispeech_clean_asr_streaming() -> None:
    suites = task_eval.load_suites()
    suite = task_eval.suite_by_id(suites, "librispeech_clean_asr_streaming")

    assert suite["dataset"]["kind"] == "asr_chat_json"
    assert suite["scoring"]["scorer"] == "asr_transcript"
    assert suite["default_model_names"] == ["nemotron-speech-streaming-en-0.6b"]
    assert suite["selectors"]["runtime_strategies"] == [
        "nemotron_speech_streaming_speech_to_text_rnnt"
    ]
    assert suite["selectors"]["families"] == ["nemotron_speech_streaming"]
    non_streaming = task_eval.suite_by_id(suites, "librispeech_clean_asr")
    assert "nemotron_speech_streaming" in non_streaming["selectors"]["exclude_families"]


def test_custom_suite_file_does_not_add_builtin_suites(tmp_path: Path) -> None:
    custom = tmp_path / "suites.json"
    custom.write_text(
        json.dumps(
            {
                "suites": [
                    {
                        "id": "custom_only",
                        "dataset": {"kind": "mmlu_five_shot_json"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    suites = task_eval.load_suites(custom)

    assert [suite["id"] for suite in suites] == ["custom_only"]


def test_plan_selects_chat_text_generation_manifests() -> None:
    suites = task_eval.load_suites()
    models = [
        {
            "name": "decoder-chat",
            "hf_id": "example-org/decoder-chat",
            "bundle": "decoder-chat.trtfb",
            "runtime_strategy": "decoder_family_decoder_kv_cache",
            "task_strategy": "text_generation_causal",
            "reference_family": "chat_instruct_template",
            "user_contract": "chat_response",
            "family": "decoder_family",
            "ci_tier": "default",
            "requires_multi_device": False,
            "manifest": "tests/e2e/models/decoder_family/decoder-chat.json",
            "skip": "",
        },
        {
            "name": "decoder-continuation",
            "hf_id": "example-org/decoder-continuation",
            "bundle": "decoder-continuation.trtfb",
            "runtime_strategy": "decoder_family_decoder_kv_cache",
            "task_strategy": "text_generation_causal",
            "reference_family": "causal_base_continuation",
            "user_contract": "continuation_parity",
            "family": "decoder_family",
            "ci_tier": "default",
            "requires_multi_device": False,
            "manifest": "tests/e2e/models/decoder_family/decoder-continuation.json",
            "skip": "",
        },
    ]

    rows = task_eval.build_plan(
        suites,
        models,
        suite_id="mmlu_five_shot_mcq",
        use_default_models=False,
    )

    selected = {row["model"]: row for row in rows}
    assert any(
        row["runtime_strategy"] == "decoder_family_decoder_kv_cache"
        and row["user_contract"] == "chat_response"
        for row in selected.values()
    )
    assert "decoder-chat" in selected
    assert "decoder-continuation" not in selected


def test_load_manifest_records_discovers_model_owned_manifests(tmp_path: Path) -> None:
    family_dir = tmp_path / "example_decoder"
    manifest_dir = family_dir / "manifests"
    manifest_dir.mkdir(parents=True)
    (family_dir / "MODEL.toml").write_text(
        'test_manifests = ["manifests/example-decoder.json"]\n',
        encoding="utf-8",
    )
    (manifest_dir / "example-decoder.json").write_text(
        json.dumps(
            {
                "name": "example-decoder",
                "hf_id": "example-org/example-decoder",
                "bundle": "example-decoder.trtfb",
                "family": "example_decoder",
                "runtime_strategy": "example_decoder_decoder_kv_cache",
                "task_strategy": "text_generation_causal",
                "reference_family": "chat_example",
                "user_contract": "chat_response",
                "task_eval": {
                    "vlm_fallback_prompt_template": "<image>{prompt}",
                },
            }
        ),
        encoding="utf-8",
    )

    records = task_eval.load_manifest_records(tmp_path)

    assert [record["name"] for record in records] == ["example-decoder"]
    assert records[0]["manifest"].endswith("example_decoder/manifests/example-decoder.json")
    assert records[0]["task_eval"] == {
        "vlm_fallback_prompt_template": "<image>{prompt}",
    }


def test_default_model_names_match_selected_plan_models() -> None:
    suites = task_eval.load_suites()
    models = task_eval.load_manifest_records()

    for suite in suites:
        rows = task_eval.build_plan(suites, models, suite_id=suite["id"])
        selected_names = {row["model"] for row in rows if row["selected"]}

        assert selected_names == set(suite["default_model_names"]), suite["id"]


def test_plan_selects_vlm_mmmu_pro_vision_models() -> None:
    suites = task_eval.load_suites()
    suite = dict(task_eval.suite_by_id(suites, "vlm_mmmu_pro_vision_mcq"))
    suite.pop("default_model_names")
    suite["selectors"] = {
        **suite["selectors"],
        "runtime_strategies": ["vision_family_vision_language"],
        "families": ["vl_family_primary", "vl_family_secondary"],
        "exclude_families": ["excluded_vl_family"],
    }
    models = [
        {
            "name": "vl-primary",
            "hf_id": "example-org/vl-primary",
            "bundle": "vl-primary.trtfb",
            "runtime_strategy": "vision_family_vision_language",
            "task_strategy": "vision_language_generation",
            "reference_family": "vl_instruct_qa",
            "user_contract": "vl_answer",
            "family": "vl_family_primary",
            "ci_tier": "default",
            "requires_multi_device": False,
            "manifest": "tests/e2e/models/vl_family_primary/manifests/vl-primary.json",
            "skip": "",
        },
        {
            "name": "vl-secondary",
            "hf_id": "example-org/vl-secondary",
            "bundle": "vl-secondary.trtfb",
            "runtime_strategy": "vision_family_vision_language",
            "task_strategy": "vision_language_generation",
            "reference_family": "vl_instruct_qa",
            "user_contract": "vl_answer",
            "family": "vl_family_secondary",
            "ci_tier": "default",
            "requires_multi_device": False,
            "manifest": "tests/e2e/models/vl_family_secondary/manifests/vl-secondary.json",
            "skip": "",
        },
        {
            "name": "vl-excluded",
            "hf_id": "example-org/vl-excluded",
            "bundle": "vl-excluded.trtfb",
            "runtime_strategy": "vision_family_vision_language",
            "task_strategy": "vision_language_generation",
            "reference_family": "vl_instruct_qa",
            "user_contract": "vl_answer",
            "family": "excluded_vl_family",
            "ci_tier": "default",
            "requires_multi_device": False,
            "manifest": "tests/e2e/models/excluded_vl_family/manifests/vl-excluded.json",
            "skip": "",
        },
        {
            "name": "text-decoder",
            "hf_id": "example-org/text-decoder",
            "bundle": "text-decoder.trtfb",
            "runtime_strategy": "decoder_family_decoder_kv_cache",
            "task_strategy": "text_generation_causal",
            "reference_family": "chat_instruct_template",
            "user_contract": "chat_response",
            "family": "decoder_family",
            "ci_tier": "default",
            "requires_multi_device": False,
            "manifest": "tests/e2e/models/decoder_family/manifests/text-decoder.json",
            "skip": "",
        },
    ]

    rows = task_eval.build_plan([suite], models)

    selected = {row["model"]: row for row in rows}
    assert "vl-primary" in selected
    assert selected["vl-primary"]["runtime_strategy"] == "vision_family_vision_language"
    assert "vl-secondary" in selected
    assert "vl-excluded" not in selected
    assert "text-decoder" not in selected


def test_plan_selects_ocrbench_v2_unified_models() -> None:
    suites = task_eval.load_suites()
    models = task_eval.load_manifest_records()

    rows = task_eval.build_plan(suites, models, suite_id="ocrbench_v2_unified")

    selected = {row["model"]: row for row in rows}
    model_by_name = {model["name"]: model for model in models}
    assert set(selected) == {"deepseek-ocr"}
    assert model_by_name["deepseek-ocr"]["reference_backend"] == "hf_transformers"
    assert "qwen25vl-3b" not in selected
    assert "internvl3-2b" not in selected
    assert "locateanything-3b" not in selected


def test_plan_selects_librispeech_asr_models() -> None:
    suites = task_eval.load_suites()
    models = task_eval.load_manifest_records()

    rows = task_eval.build_plan(suites, models, suite_id="librispeech_clean_asr")

    selected = {row["model"]: row for row in rows}
    assert "whisper-tiny-fp16" in selected
    assert selected["whisper-tiny-fp16"]["runtime_strategy"] == "whisper_speech_to_text"
    assert "canary-1b-v2" in selected
    assert selected["canary-1b-v2"]["runtime_strategy"] == "canary_speech_to_text"
    assert set(selected) == {
        "whisper-tiny-fp16",
        "whisper-large-v3-turbo",
        "canary-1b-v2",
    }
    assert "nemotron-nano-v2-speech-embedded" not in selected


def test_plan_selects_librispeech_streaming_asr_models() -> None:
    suites = task_eval.load_suites()
    models = task_eval.load_manifest_records()

    rows = task_eval.build_plan(suites, models, suite_id="librispeech_clean_asr_streaming")

    selected = {row["model"]: row for row in rows}
    assert "nemotron-speech-streaming-en-0.6b" in selected
    assert selected["nemotron-speech-streaming-en-0.6b"]["runtime_strategy"] == (
        "nemotron_speech_streaming_speech_to_text_rnnt"
    )
    assert not any("-asr-probe" in name for name in selected)
    assert "whisper-tiny-fp16" not in selected
    assert "canary-1b-v2" not in selected


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
    assert prompts == [
        {
            "sample_id": "mmlu_000000",
            "dataset_index": 0,
            "eval_index": 0,
            "subject": "subject_a",
            "answer": "B",
            "prompt": "Question one\nA. a\nB. b\nAnswer:",
        }
    ]
    assert manifest["suite"] == "mmlu_five_shot_mcq"
    assert manifest["request_count"] == 1


def test_prepare_seedtts_writes_resolved_audio_and_scoring_contract(tmp_path: Path) -> None:
    dataset = tmp_path / "SeedTTS_en_meta" / "seedtts_en_meta.json"
    dataset.parent.mkdir()
    _write_seedtts(dataset)
    suite = task_eval.suite_by_id(task_eval.load_suites(), "seedtts_en_tts_intelligibility")

    outputs = task_eval.prepare_seedtts_dataset(
        dataset_path=dataset,
        work_dir=tmp_path / "work",
        suite=suite,
        limit=1,
    )

    answers = json.loads(outputs["answers"].read_text(encoding="utf-8"))
    prompts = task_eval.load_jsonl(outputs["prompts"])
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    reference_wav = str((dataset.parent / "reference.wav").resolve())

    assert answers["requests"][0]["answer"] == "The test sentence."
    assert answers["requests"][0]["reference_wav"] == reference_wav
    assert answers["scoring"]["max_wer"] == 0.25
    assert prompts == [
        {
            "sample_id": "seedtts-1",
            "dataset_index": 0,
            "eval_index": 0,
            "subject": "en",
            "answer": "The test sentence.",
            "prompt": "The test sentence.",
            "reference_wav": reference_wav,
            "language": "en",
        }
    ]
    assert manifest["dataset_kind"] == "seedtts_json"
    assert manifest["scoring"]["scorer"] == "tts_intelligibility"


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
    assert prompts == [
        {
            "sample_id": "test_case_1",
            "dataset_index": 0,
            "eval_index": 0,
            "subject": "History",
            "answer": "J",
            "prompt": "Answer with the option letter.\n\nWhich letter is correct?\nA. no\nJ. yes\n\nAnswer directly.",
            "images": [str(dataset_dir / "images" / "sample.jpg")],
        }
    ]
    assert manifest["suite"] == "vlm_mmmu_pro_vision_mcq"
    assert manifest["dataset_kind"] == "vlm_chat_json"
    assert manifest["request_count"] == 1
    assert manifest["image_count"] == 1
    assert "reference" not in manifest


def test_prepare_ocrbench_unified_writes_image_prompt_jsonl(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "OCRBench_v2" / "unified"
    dataset_dir.mkdir(parents=True)
    dataset = dataset_dir / "dataset.json"
    _write_ocrbench_unified(dataset)
    suite = task_eval.suite_by_id(task_eval.load_suites(), "ocrbench_v2_unified")

    outputs = task_eval.prepare_vlm_unified_dataset(
        dataset_path=dataset,
        work_dir=tmp_path / "work",
        suite=suite,
        limit=1,
    )

    answers = json.loads(outputs["answers"].read_text(encoding="utf-8"))
    prompts = task_eval.load_jsonl(outputs["prompts"])
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))

    assert len(answers["requests"]) == 1
    assert "samples" not in answers
    assert answers["requests"][0]["answer"] == "enabled"
    assert answers["requests"][0]["answer_aliases"] == ["enabled", "on"]
    assert answers["requests"][0]["ocrbench_type"] == "APP agent en"
    assert answers["requests"][0]["ocrbench_answers"] == ["enabled", "on"]
    assert answers["requests"][0]["ocrbench_eval"] is None
    assert answers["requests"][0]["messages"][0]["content"][0] == {
        "type": "image",
        "image": "images/ocrbench_v2_000000.jpg",
    }
    assert prompts == [
        {
            "sample_id": "ocrbench_v2_000000",
            "dataset_index": 0,
            "eval_index": 0,
            "subject": "APP agent en",
            "answer": "enabled",
            "prompt": "What is the wrong answer 2?",
            "images": [str(dataset_dir / "images" / "ocrbench_v2_000000.jpg")],
        }
    ]
    assert manifest["suite"] == "ocrbench_v2_unified"
    assert manifest["dataset_kind"] == "vlm_unified_json"
    assert manifest["request_count"] == 1
    assert manifest["image_count"] == 1


def test_prepare_ocrbench_unified_reports_missing_images(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "OCRBench_v2" / "unified"
    dataset_dir.mkdir(parents=True)
    dataset = dataset_dir / "dataset.json"
    _write_ocrbench_unified(dataset)
    (dataset_dir / "images" / "ocrbench_v2_000000.jpg").unlink()
    suite = task_eval.suite_by_id(task_eval.load_suites(), "ocrbench_v2_unified")

    try:
        task_eval.prepare_vlm_unified_dataset(
            dataset_path=dataset,
            work_dir=tmp_path / "work",
            suite=suite,
            limit=1,
        )
    except FileNotFoundError as exc:
        message = str(exc)
        assert "1 missing image asset" in message
        assert "ocrbench_v2_000000" in message
        assert "images/ocrbench_v2_000000.jpg" in message
    else:
        raise AssertionError("expected missing-image validation failure")


def test_prepare_asr_chat_dataset_writes_audio_prompt_jsonl(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "librispeech_clean_test"
    dataset_dir.mkdir()
    dataset = dataset_dir / "librispeech_clean_test.json"
    _write_asr_librispeech(dataset)
    suite = task_eval.suite_by_id(task_eval.load_suites(), "librispeech_clean_asr")

    outputs = task_eval.prepare_asr_chat_dataset(
        dataset_path=dataset,
        work_dir=tmp_path / "work",
        suite=suite,
        limit=1,
    )

    answers = json.loads(outputs["answers"].read_text(encoding="utf-8"))
    prompts = task_eval.load_jsonl(outputs["prompts"])
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    prepared_audio = tmp_path / "work" / "audio" / "clean_000000.wav"

    assert prepared_audio.is_file()
    assert len(answers["requests"]) == 1
    assert answers["requests"][0]["answer"] == "The quick brown fox"
    assert answers["requests"][0]["subject"] == "test-clean"
    assert answers["requests"][0]["audio"] == str(prepared_audio)
    assert prompts == [
        {
            "sample_id": "clean_000000",
            "dataset_index": 0,
            "eval_index": 0,
            "subject": "test-clean",
            "answer": "The quick brown fox",
            "prompt": "Transcribe this audio.",
            "audio": str(prepared_audio),
        }
    ]
    assert manifest["suite"] == "librispeech_clean_asr"
    assert manifest["dataset_kind"] == "asr_chat_json"
    assert manifest["request_count"] == 1
    assert manifest["audio_count"] == 1


def test_prepare_asr_chat_dataset_reports_missing_audio(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "librispeech_clean_test"
    dataset_dir.mkdir()
    dataset = dataset_dir / "librispeech_clean_test.json"
    _write_asr_librispeech(dataset)
    (dataset_dir / "audio" / "sample.wav").unlink()
    suite = task_eval.suite_by_id(task_eval.load_suites(), "librispeech_clean_asr")

    try:
        task_eval.prepare_asr_chat_dataset(
            dataset_path=dataset,
            work_dir=tmp_path / "work",
            suite=suite,
            limit=1,
        )
    except FileNotFoundError as exc:
        message = str(exc)
        assert "1 missing audio asset" in message
        assert "clean_000000" in message
        assert "audio/sample.wav" in message
    else:
        raise AssertionError("expected missing-audio validation failure")


def test_prepare_vlm_fixed_suite_normalizes_image_and_messages(tmp_path: Path, monkeypatch) -> None:
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
    assert answers["requests"][0]["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(fixed_image)},
                {"type": "text", "text": merged_prompt},
            ],
        }
    ]
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
        "example-org/vision-chat",
    )

    messages = json.loads(rendered)
    assert messages == request["messages"]


def test_vlm_reference_prompt_uses_manifest_owned_fallback_template() -> None:
    class FakeProcessor:
        pass

    rendered = task_eval._vlm_chat_text(
        FakeProcessor(),
        {},
        "Which option matches the image?",
        "<IMG_CONTEXT>\n{prompt}",
    )

    assert rendered == "<IMG_CONTEXT>\nWhich option matches the image?"


def test_prepare_cli_accepts_vlm_dataset_kind(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "MMMU_Pro_vision"
    dataset_dir.mkdir()
    dataset = dataset_dir / "mmmu_pro_vision_dataset.json"
    _write_vlm_mmmu_pro_vision(dataset)
    work_dir = tmp_path / "work"

    rc = task_eval.cmd_prepare(
        argparse.Namespace(
            suites=str(task_eval.DEFAULT_SUITES),
            suite="vlm_mmmu_pro_vision_mcq",
            dataset=str(dataset),
            work_dir=str(work_dir),
            limit=1,
            subject="",
            sample_seed=None,
        )
    )

    assert rc == 0
    assert task_eval.load_jsonl(work_dir / "prompts.jsonl")[0]["images"] == [
        str(dataset_dir / "images" / "sample.jpg")
    ]


def test_continuation_parity_exact_and_first_divergence() -> None:
    hf = {
        "responses": [
            {"sample_id": "a", "output_text": "the cat sat"},
            {"sample_id": "b", "output_text": "hello world"},
        ]
    }
    trtfb = {
        "responses": [
            {"sample_id": "a", "output_text": "the cat sat"},
            {"sample_id": "b", "output_text": "hello there"},
        ]
    }

    summary = task_eval.compare_continuation_sets(hf, trtfb, tokenize=lambda s: s.split())

    assert summary["count"] == 2
    assert summary["exact_match_rate"] == 0.5  # "a" exact, "b" not
    assert summary["samples"][0]["first_divergence"] == 3  # all 3 tokens match
    assert summary["samples"][1]["first_divergence"] == 1  # diverge at token index 1
    # matched prefixes 3 + 1 = 4, ref token counts 3 + 2 = 5
    assert abs(summary["token_prefix_agreement"] - 4 / 5) < 1e-9


def test_continuation_parity_prefers_generated_token_ids() -> None:
    hf = {
        "responses": [
            {"sample_id": "a", "output_text": "same text", "generated_token_ids": [10, 20]},
            {"sample_id": "b", "output_text": "same text", "generated_token_ids": [1, 2, 3]},
        ]
    }
    trtfb = {
        "responses": [
            {"sample_id": "a", "output_text": "same text", "generated_token_ids": [10, 20]},
            {"sample_id": "b", "output_text": "same text", "generated_token_ids": [1, 2, 4]},
        ]
    }

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
        json.dumps(
            {
                "responses": [
                    {"sample_id": "a", "output_text": "same", "generated_token_ids": [1, 2]},
                    {"sample_id": "b", "output_text": "left", "generated_token_ids": [3, 4]},
                ]
            }
        ),
        encoding="utf-8",
    )
    (work_dir / "trtfb_predictions.json").write_text(
        json.dumps(
            {
                "responses": [
                    {"sample_id": "a", "output_text": "same", "generated_token_ids": [1, 2]},
                    {"sample_id": "b", "output_text": "right", "generated_token_ids": [3, 5]},
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "continuation.json"

    rc = task_eval.cmd_compare_continuation(
        argparse.Namespace(
            work_dir=str(work_dir),
            hf_predictions="",
            trtfb_predictions="",
            model="",
            trust_remote_code=False,
            local_files_only=False,
            output=str(output),
        )
    )

    summary = json.loads(output.read_text(encoding="utf-8"))
    assert rc == 0
    assert summary["comparison_granularity"] == "generated_token_ids"
    assert summary["exact_match_rate"] == 0.5
    assert summary["token_prefix_agreement"] == 0.75
    assert summary["samples"][1]["first_divergence"] == 1


def test_convert_trtfb_uses_generated_text_field(tmp_path: Path) -> None:
    raw = tmp_path / "trtfb_raw.jsonl"
    raw.write_text(
        json.dumps(
            {
                "sample_id": "mmlu_000000",
                "gold_answer": "B",
                "pred_answer": "",
                "text": "Answer: B",
                "generated_tokens": 1,
                "generated_token_ids": [42],
                "wall_ms": 3.5,
            }
        )
        + "\n",
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


def test_score_predictions_accepts_answer_aliases() -> None:
    answers = {"requests": [{"answer": "enabled", "answer_aliases": ["enabled", "on"]}]}
    predictions = {"responses": [{"sample_id": "ocrbench_v2_000000", "output_text": "on"}]}

    score = task_eval.score_predictions(predictions, answers)

    assert score["overall_accuracy"] == 1.0
    assert score["samples"][0]["answer_aliases"] == ["on"]


def test_tts_intelligibility_scores_asr_and_waveform_health(tmp_path: Path) -> None:
    reference_wav = tmp_path / "reference.wav"
    generated_wav = tmp_path / "generated.wav"
    _write_pcm_wav(reference_wav)
    _write_pcm_wav(generated_wav, seconds=1.1)
    answers = {
        "scoring": {
            "max_wer": 0.25,
            "max_ned": 0.20,
            "min_rms": 0.001,
            "min_duration_ratio": 0.5,
            "max_duration_ratio": 2.0,
        },
        "requests": [
            {
                "id": "seedtts-1",
                "reference": "The test sentence.",
                "answer": "The test sentence.",
                "reference_wav": str(reference_wav),
                "subject": "en",
            }
        ],
    }
    predictions = {
        "responses": [
            {
                "sample_id": "seedtts-1",
                "output_text": "the test sentence",
                "wav_path": str(generated_wav),
            }
        ]
    }

    score = task_eval.score_predictions(predictions, answers, scorer="tts_intelligibility")

    assert score["overall_accuracy"] == 1.0
    assert score["mean_wer"] == 0.0
    assert score["samples"][0]["wav_exists"] is True
    assert 1.09 < score["samples"][0]["duration_ratio"] < 1.11


def test_tts_intelligibility_fails_wrong_or_missing_audio(tmp_path: Path) -> None:
    reference_wav = tmp_path / "reference.wav"
    _write_pcm_wav(reference_wav)
    answers = {
        "scoring": {"max_wer": 0.25, "max_ned": 0.20},
        "requests": [
            {
                "reference": "The test sentence.",
                "answer": "The test sentence.",
                "reference_wav": str(reference_wav),
            }
        ],
    }
    predictions = {
        "responses": [
            {
                "sample_id": "seedtts-1",
                "output_text": "completely different words",
                "wav_path": str(tmp_path / "missing.wav"),
            }
        ]
    }

    score = task_eval.score_predictions(predictions, answers, scorer="tts_intelligibility")

    assert score["overall_accuracy"] == 0.0
    assert score["samples"][0]["correct"] is False
    assert score["samples"][0]["wer"] > 0.25


def test_tts_disagreement_reports_full_normalized_transcripts(tmp_path: Path) -> None:
    reference_wav = tmp_path / "reference.wav"
    hf_wav = tmp_path / "hf.wav"
    trtfb_wav = tmp_path / "trtfb.wav"
    for wav_path in (reference_wav, hf_wav, trtfb_wav):
        _write_pcm_wav(wav_path)
    answers = {
        "requests": [
            {
                "answer": "I'm never more aware of a room's acoustics.",
                "reference_wav": str(reference_wav),
            }
        ],
    }
    hf = {
        "responses": [
            {
                "sample_id": "seedtts-1",
                "output_text": "I'm never more aware of a room's acoustics.",
                "wav_path": str(hf_wav),
            }
        ]
    }
    trtfb = {
        "responses": [
            {
                "sample_id": "seedtts-1",
                "output_text": "I am never more aware of other rooms.",
                "wav_path": str(trtfb_wav),
            }
        ]
    }

    summary = task_eval.compare_prediction_sets(hf, trtfb, answers, scorer="tts_intelligibility")

    assert summary["disagreements"][0]["hf_prediction"] == (
        "i m never more aware of a room s acoustics"
    )
    assert summary["disagreements"][0]["trtfb_prediction"] == (
        "i am never more aware of other rooms"
    )


def test_run_tts_trtfb_generates_audio_and_batches_asr(tmp_path: Path, monkeypatch) -> None:
    dataset = tmp_path / "SeedTTS_en_meta" / "seedtts_en_meta.json"
    dataset.parent.mkdir()
    _write_seedtts(dataset)
    suite = task_eval.suite_by_id(task_eval.load_suites(), "seedtts_en_tts_intelligibility")
    work_dir = tmp_path / "work"
    task_eval.prepare_seedtts_dataset(
        dataset_path=dataset,
        work_dir=work_dir,
        suite=suite,
        limit=1,
        task_eval_config={
            "family": "bark",
            "model_max_new_tokens": 12,
            "runtime_config": {"audio_magpie": {"seed": 42}},
        },
    )
    commands: list[list[str]] = []

    class Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, **_kwargs):
        commands.append(cmd)
        output = Path(cmd[cmd.index("--output") + 1])
        _write_pcm_wav(output)
        return Result()

    monkeypatch.setattr(task_eval.subprocess, "run", fake_run)
    monkeypatch.setattr(
        task_eval,
        "_transcribe_audio_files",
        lambda paths, **_kwargs: ["The test sentence." for _path in paths],
    )
    args = argparse.Namespace(
        work_dir=str(work_dir),
        raw_output="",
        predictions="",
        log="",
        max_new_tokens=None,
        bundle="model.trtfb",
        trtmc_binary="build/trtmc",
        hf_python="",
        backend_dir="",
        config="",
        set=[],
        cuda_visible_devices="",
    )

    task_eval.run_tts_trtfb(args)

    assert commands[0][:3] == ["build/trtmc", "generate-audio", "model.trtfb"]
    assert commands[0][commands[0].index("--max-new-tokens") + 1] == "12"
    assert "audio_magpie.seed=42" in commands[0]
    assert "audio_bark.seed=42" in commands[0]
    predictions = json.loads((work_dir / "trtfb_predictions.json").read_text(encoding="utf-8"))
    assert predictions["responses"][0]["output_text"] == "The test sentence."
    assert Path(predictions["responses"][0]["wav_path"]).is_file()


def test_ocrbench_v2_scores_short_vqa_with_contains() -> None:
    answers = {
        "requests": [
            {
                "answer": "San Francisco",
                "subject": "APP agent en",
                "ocrbench_type": "APP agent en",
                "ocrbench_answers": ["San Francisco"],
            }
        ]
    }
    predictions = {
        "responses": [
            {
                "sample_id": "ocrbench_v2_000009",
                "output_text": "San Francisco, CA",
            }
        ]
    }

    score = task_eval.score_predictions(predictions, answers, scorer="ocrbench_v2")

    assert score["overall_accuracy"] == 1.0
    assert score["samples"][0]["score"] == 1.0
    assert score["samples"][0]["metric"] == "vqa"
    assert score["ocrbench_v2"]["language_scores"]["en"]["overall_accuracy"] == 1.0


def test_ocrbench_v2_scores_counting_regression() -> None:
    answers = {
        "requests": [
            {
                "answer": "10",
                "subject": "text counting en",
                "ocrbench_type": "text counting en",
                "ocrbench_eval": "regression",
                "ocrbench_answers": ["10"],
            }
        ]
    }
    predictions = {
        "responses": [
            {
                "sample_id": "ocrbench_v2_008200",
                "output_text": "There are 9 words.",
            }
        ]
    }

    score = task_eval.score_predictions(predictions, answers, scorer="ocrbench_v2")

    assert score["overall_accuracy"] == 0.9
    assert score["samples"][0]["metric"] == "counting"


def test_ocrbench_v2_scores_text_grounding_iou_from_answer_coords() -> None:
    answers = {
        "requests": [
            {
                "answer": "0",
                "subject": "text grounding en",
                "ocrbench_type": "text grounding en",
                "ocrbench_answers": ["0", "0", "100", "100"],
            }
        ]
    }
    predictions = {
        "responses": [
            {
                "sample_id": "ocrbench_v2_008400",
                "output_text": "(0, 0, 50, 100)",
            }
        ]
    }

    score = task_eval.score_predictions(predictions, answers, scorer="ocrbench_v2")

    assert score["overall_accuracy"] == 0.5
    assert score["samples"][0]["metric"] == "bbox_iou"


def test_ocrbench_v2_scores_key_information_f1() -> None:
    answers = {
        "requests": [
            {
                "answer": "{'name': ['Ada'], 'total': ['42']}",
                "subject": "key information extraction en",
                "ocrbench_type": "key information extraction en",
                "ocrbench_answers": ["{'name': ['Ada'], 'total': ['42']}"],
            }
        ]
    }
    predictions = {
        "responses": [
            {
                "sample_id": "ocrbench_v2_000900",
                "output_text": "{'name': 'Ada', 'total': '41'}",
            }
        ]
    }

    score = task_eval.score_predictions(predictions, answers, scorer="ocrbench_v2")

    assert score["overall_accuracy"] == 0.5
    assert score["samples"][0]["metric"] == "key_value_f1"


def test_ocrbench_v2_agreement_uses_correctness_not_text_match() -> None:
    answers = {
        "requests": [
            {
                "answer": "alpha",
                "subject": "APP agent en",
                "ocrbench_type": "APP agent en",
                "ocrbench_answers": ["alpha"],
            },
            {
                "answer": "Facebook",
                "subject": "APP agent en",
                "ocrbench_type": "APP agent en",
                "ocrbench_answers": ["Facebook"],
            },
        ]
    }
    hf = {
        "responses": [
            {"sample_id": "both_wrong", "output_text": "zzz"},
            {"sample_id": "hf_correct", "output_text": "Facebook"},
        ]
    }
    trtfb = {
        "responses": [
            {"sample_id": "both_wrong", "output_text": "yyy"},
            {"sample_id": "hf_correct", "output_text": "Instagram"},
        ]
    }

    summary = task_eval.compare_prediction_sets(hf, trtfb, answers, scorer="ocrbench_v2")

    assert summary["prediction_agreement_rate"] == 0.5
    assert summary["agreement_count"] == 1
    assert summary["buckets"]["both_wrong"] == 1
    assert summary["buckets"]["hf_correct_trtfb_wrong"] == 1
    assert len(summary["disagreements"]) == 1
    assert summary["disagreements"][0]["sample_id"] == "hf_correct"
    assert summary["disagreements"][0]["hf_correct"] is True
    assert summary["disagreements"][0]["trtfb_correct"] is False


def test_asr_transcript_scorer_reports_wer_cer_and_exact_rate() -> None:
    answers = {
        "requests": [
            {"answer": "Hello, world!", "subject": "test-clean"},
            {"answer": "The quick brown fox", "subject": "test-clean"},
        ]
    }
    predictions = {
        "responses": [
            {"sample_id": "a", "output_text": "hello world"},
            {"sample_id": "b", "output_text": "the quick brown box"},
        ]
    }

    score = task_eval.score_predictions(predictions, answers, scorer="asr_transcript")

    assert score["overall_accuracy"] == 0.5
    assert score["exact_match_rate"] == 0.5
    assert score["correct"] == 1
    assert score["samples"][0]["normalized_answer"] == "hello world"
    assert score["samples"][0]["exact_match"] is True
    assert score["samples"][1]["word_error_rate"] == 0.25
    assert score["samples"][1]["correct"] is False


def test_asr_transcript_scorer_marks_high_wer_wrong_and_skips_errors() -> None:
    answers = {
        "requests": [
            {"answer": "alpha beta gamma", "subject": "test-clean"},
            {"answer": "delta epsilon", "subject": "test-clean"},
        ]
    }
    predictions = {
        "responses": [
            {"sample_id": "a", "output_text": "wrong words here"},
            {"sample_id": "b", "output_text": task_eval.ERROR_OUTPUT_TEXT},
        ]
    }

    score = task_eval.score_predictions(predictions, answers, scorer="asr_transcript")

    assert score["overall_accuracy"] == 0.0
    assert score["valid_count"] == 1
    assert score["skipped_count"] == 1
    assert score["samples"][0]["word_error_rate"] == 1.0
    assert score["samples"][1]["skipped"] is True


def test_asr_transcript_agreement_uses_correctness_thresholds() -> None:
    answers = {
        "requests": [
            {"answer": "alpha beta", "subject": "test-clean"},
            {"answer": "gamma delta", "subject": "test-clean"},
        ]
    }
    hf = {
        "responses": [
            {"sample_id": "same_correctness", "output_text": "alpha beta"},
            {"sample_id": "hf_correct", "output_text": "gamma delta"},
        ]
    }
    trtfb = {
        "responses": [
            {"sample_id": "same_correctness", "output_text": "alpha, beta."},
            {"sample_id": "hf_correct", "output_text": "totally wrong"},
        ]
    }

    summary = task_eval.compare_prediction_sets(hf, trtfb, answers, scorer="asr_transcript")

    assert summary["prediction_agreement_rate"] == 0.5
    assert summary["agreement_count"] == 1
    assert summary["buckets"]["both_correct"] == 1
    assert summary["buckets"]["hf_correct_trtfb_wrong"] == 1
    assert summary["disagreements"][0]["hf_prediction"] == "gamma delta"
    assert summary["disagreements"][0]["trtfb_prediction"] == "totally wrong"


def test_selected_models_for_suite_accepts_manifest_name() -> None:
    suite = task_eval.suite_by_id(task_eval.load_suites(), "mmlu_five_shot_mcq")
    models = [
        {
            "name": "decoder-chat",
            "hf_id": "example-org/decoder-chat",
            "bundle": "decoder-chat.trtfb",
            "runtime_strategy": "decoder_family_decoder_kv_cache",
            "task_strategy": "text_generation_causal",
            "reference_family": "chat_instruct_template",
            "user_contract": "chat_response",
            "family": "decoder_family",
            "ci_tier": "default",
            "requires_multi_device": False,
            "manifest": "tests/e2e/models/decoder_family/decoder-chat.json",
            "skip": "",
        }
    ]

    selected = task_eval.selected_models_for_suite(
        suite,
        models,
        selectors=["decoder-chat"],
        single_device_only=True,
    )

    assert [model["name"] for model in selected] == ["decoder-chat"]


def test_seedtts_default_selection_uses_canonical_single_device_models() -> None:
    suite = task_eval.suite_by_id(task_eval.load_suites(), "seedtts_en_tts_intelligibility")

    selected = task_eval.selected_models_for_suite(
        suite,
        task_eval.load_manifest_records(),
        single_device_only=True,
    )

    assert {model["name"] for model in selected} == {
        "bark-large",
        "bark-small",
        "magpie-tts-357m",
    }


def test_seedtts_plan_marks_only_default_models_selected() -> None:
    suite = task_eval.suite_by_id(task_eval.load_suites(), "seedtts_en_tts_intelligibility")
    rows = task_eval.build_plan(
        [suite],
        task_eval.load_manifest_records(),
        suite_id=suite["id"],
        include_non_matching=True,
    )
    selected = {row["model"] for row in rows if row["selected"]}

    assert selected == {"bark-large", "bark-small", "magpie-tts-357m"}


def test_waives_exclude_default_selection_but_explicit_model_can_debug(tmp_path: Path) -> None:
    suite = {
        "id": "mmlu_five_shot_mcq",
        "selectors": {
            "task_strategies": ["text_generation_causal"],
            "runtime_strategies": ["decoder_family_decoder_kv_cache"],
            "user_contracts": ["chat_response"],
        },
    }
    models = [
        {
            "name": "decoder-waived",
            "hf_id": "example-org/decoder-waived",
            "bundle": "decoder-waived.trtfb",
            "runtime_strategy": "decoder_family_decoder_kv_cache",
            "task_strategy": "text_generation_causal",
            "reference_family": "chat_instruct_template",
            "user_contract": "chat_response",
            "family": "decoder_family",
            "ci_tier": "default",
            "requires_multi_device": False,
            "manifest": "tests/e2e/models/decoder-waived.json",
            "skip": "",
        },
        {
            "name": "decoder-active",
            "hf_id": "example-org/decoder-active",
            "bundle": "decoder-active.trtfb",
            "runtime_strategy": "decoder_family_decoder_kv_cache",
            "task_strategy": "text_generation_causal",
            "reference_family": "chat_instruct_template",
            "user_contract": "chat_response",
            "family": "decoder_family",
            "ci_tier": "default",
            "requires_multi_device": False,
            "manifest": "tests/e2e/models/decoder-active.json",
            "skip": "",
        },
    ]
    waives_path = tmp_path / "waives.txt"
    waives_path.write_text(
        "decoder-waived  SKIP  (reference dependency unavailable)\n",
        encoding="utf-8",
    )
    waives = task_eval.load_waives(waives_path)

    selected = task_eval.selected_models_for_suite(suite, models, waives=waives)
    explicit = task_eval.selected_models_for_suite(
        suite,
        models,
        selectors=["decoder-waived"],
        waives=waives,
    )
    rows = task_eval.build_plan([suite], models, include_non_matching=True, waives=waives)
    decoder_family_row = next(row for row in rows if row["model"] == "decoder-waived")

    assert [model["name"] for model in selected] == ["decoder-active"]
    assert [model["name"] for model in explicit] == ["decoder-waived"]
    assert decoder_family_row["selected"] is False
    assert "waived SKIP" in decoder_family_row["reason"]


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
    assert ["--method", "trt"] == cmd[cmd.index("--method") : cmd.index("--method") + 2]
    assert ["--tp-size", "2"] == cmd[cmd.index("--tp-size") : cmd.index("--tp-size") + 2]
    assert ["--precision", "bf16"] == cmd[cmd.index("--precision") : cmd.index("--precision") + 2]
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


def test_run_hf_reference_subprocess_uses_hf_python(tmp_path: Path, monkeypatch) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    captured: dict[str, list[str]] = {}

    class Result:
        returncode = 0

    def fake_run(cmd, **_kwargs):
        captured["cmd"] = cmd
        return Result()

    monkeypatch.setattr(task_eval.subprocess, "run", fake_run)

    args = argparse.Namespace(
        hf_python="/opt/deepseek-hf/bin/python3",
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
        seed=None,
    )
    model = {"hf_id": "org/model", "trust_remote_code": False}

    task_eval.run_hf_reference_subprocess(args, model, work_dir)

    assert captured["cmd"][0] == "/opt/deepseek-hf/bin/python3"
    assert captured["cmd"][1:3] == [str(Path(task_eval.__file__).resolve()), "run-hf"]


def test_run_hf_reference_subprocess_passes_asr_family_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    captured: dict[str, list[str]] = {}

    class Result:
        returncode = 0

    def fake_run(cmd, **_kwargs):
        captured["cmd"] = cmd
        return Result()

    monkeypatch.setattr(task_eval.subprocess, "run", fake_run)
    args = argparse.Namespace(
        hf_python="",
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
        seed=None,
    )
    model = {
        "hf_id": "nvidia/canary-1b-v2",
        "family": "canary",
        "reference_family": "asr_canary",
        "trust_remote_code": False,
    }

    task_eval.run_hf_reference_subprocess(args, model, work_dir)

    assert captured["cmd"][captured["cmd"].index("--family") + 1] == "canary"
    assert captured["cmd"][captured["cmd"].index("--reference-family") + 1] == "asr_canary"


def test_asr_reference_detection_identifies_canary() -> None:
    assert task_eval._is_canary_asr_reference(
        argparse.Namespace(
            model="nvidia/canary-1b-v2",
            family="",
            reference_family="",
        )
    )
    assert task_eval._is_canary_asr_reference(
        argparse.Namespace(
            model="nvidia/other",
            family="canary",
            reference_family="",
        )
    )
    assert task_eval._is_canary_asr_reference(
        argparse.Namespace(
            model="nvidia/other",
            family="",
            reference_family="asr_canary",
        )
    )


def test_nemo_asr_reference_detection_identifies_streaming() -> None:
    assert task_eval._is_nemo_asr_reference(
        argparse.Namespace(
            model="nvidia/nemotron-speech-streaming-en-0.6b",
            family="",
            reference_family="",
        )
    )
    assert task_eval._is_nemo_asr_reference(
        argparse.Namespace(
            model="nvidia/other",
            family="nemotron_speech_streaming",
            reference_family="",
        )
    )
    assert task_eval._is_nemo_asr_reference(
        argparse.Namespace(
            model="nvidia/canary-1b-v2",
            family="canary",
            reference_family="asr_canary",
        )
    )


def test_run_hf_reference_dispatches_asr_workdir(tmp_path: Path, monkeypatch) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "manifest.json").write_text(
        json.dumps({"dataset_kind": "asr_chat_json"}),
        encoding="utf-8",
    )
    calls: list[str] = []

    def fake_asr(_args):
        calls.append("asr")

    monkeypatch.setattr(task_eval, "run_asr_hf_reference", fake_asr)

    task_eval.run_hf_reference(argparse.Namespace(work_dir=str(work_dir)))

    assert calls == ["asr"]


def test_run_asr_trtfb_invokes_transcribe_per_audio(tmp_path: Path, monkeypatch) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "manifest.json").write_text(
        json.dumps({"dataset_kind": "asr_chat_json", "generation": {"max_new_tokens": 32}}),
        encoding="utf-8",
    )
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake")
    (work_dir / "prompts.jsonl").write_text(
        json.dumps({"sample_id": "asr_000000", "audio": str(audio_path)}) + "\n",
        encoding="utf-8",
    )
    commands: list[list[str]] = []

    class Result:
        returncode = 0
        stdout = "Hello world\n"
        stderr = "tokens: 1 2 3\n"

    def fake_run(cmd, **_kwargs):
        commands.append(cmd)
        return Result()

    monkeypatch.setattr(task_eval.subprocess, "run", fake_run)
    args = argparse.Namespace(
        work_dir=str(work_dir),
        bundle="bundle.trtfb",
        trtmc_binary="build/trtmc",
        raw_output="",
        predictions="",
        log="",
        max_new_tokens=None,
        cuda_visible_devices="",
        hf_python="",
    )

    task_eval.run_asr_trtfb(args)

    assert commands == [
        [
            "build/trtmc",
            "transcribe",
            "bundle.trtfb",
            "--audio",
            str(audio_path),
            "--max-new-tokens",
            "32",
        ]
    ]
    predictions = json.loads((work_dir / "trtfb_predictions.json").read_text(encoding="utf-8"))
    assert predictions["responses"][0]["output_text"] == "Hello world"
    assert predictions["responses"][0]["generated_token_ids"] == [1, 2, 3]


def test_load_vlm_model_falls_back_between_auto_classes() -> None:
    calls: list[str] = []

    class UnsupportedAutoModel:
        __name__ = "UnsupportedAutoModel"

        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            calls.append("unsupported")
            raise ValueError("Unrecognized configuration class")

    class SupportedAutoModel:
        __name__ = "SupportedAutoModel"

        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            calls.append("supported")
            return SupportedAutoModel()

        def eval(self):
            calls.append("eval")
            return self

    class Transformers:
        AutoModelForImageTextToText = UnsupportedAutoModel
        AutoModel = SupportedAutoModel

    model = task_eval._load_vlm_model(Transformers, "org/model", {})

    assert isinstance(model, SupportedAutoModel)
    assert calls == ["unsupported", "supported", "eval"]


def test_vlm_chat_text_falls_back_when_chat_template_missing() -> None:
    class Processor:
        def apply_chat_template(self, *_args, **_kwargs):
            raise ValueError("tokenizer.chat_template is not set")

    request = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Extract text."}],
            }
        ]
    }

    assert (
        task_eval._vlm_chat_text(
            Processor(),
            request,
            "Extract text.",
            "deepseek-ai/DeepSeek-OCR-2",
        )
        == "Extract text."
    )


def test_run_deepseek_ocr_hf_reference_writes_predictions(tmp_path: Path) -> None:
    calls: list[dict[str, str]] = []

    class Model:
        def infer(self, _tokenizer, **kwargs):
            calls.append(kwargs)
            return "enabled"

    class Tokenizer:
        def __call__(self, text, **_kwargs):
            assert text == "enabled"
            return argparse.Namespace(input_ids=[1, 2])

    task_eval._run_deepseek_ocr_hf_reference(
        model=Model(),
        tokenizer=Tokenizer(),
        answers={"requests": [{"answer": "enabled"}]},
        prompt_rows=[
            {
                "sample_id": "ocrbench_v2_000000",
                "prompt": "What is shown?",
                "images": ["/tmp/image.jpg"],
            }
        ],
        work_dir=tmp_path,
    )

    payload = json.loads((tmp_path / "hf_predictions.json").read_text(encoding="utf-8"))

    assert calls[0]["prompt"] == "<image>\nWhat is shown?"
    assert calls[0]["image_file"] == "/tmp/image.jpg"
    assert calls[0]["eval_mode"] is True
    assert payload["responses"][0]["output_text"] == "enabled"
    assert payload["responses"][0]["generated_token_ids"] == [1, 2]


def test_eval_one_model_reuses_hf_builds_bundle_and_reruns_trtfb(
    tmp_path: Path, monkeypatch
) -> None:
    dataset = tmp_path / "mmlu.json"
    _write_mmlu(dataset)
    suite = task_eval.suite_by_id(task_eval.load_suites(), "mmlu_five_shot_mcq")
    model = {
        "name": "decoder-small",
        "hf_id": "example-org/decoder-small",
        "bundle": "decoder-small.trtfb",
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
        json.dumps(
            {
                "responses": [
                    {"sample_id": "mmlu_000000", "output_text": "B"},
                    {"sample_id": "mmlu_000001", "output_text": "A"},
                ]
            }
        ),
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
            json.dumps(
                {
                    "responses": [
                        {"sample_id": "mmlu_000000", "output_text": "B"},
                        {"sample_id": "mmlu_000001", "output_text": "B"},
                    ]
                }
            ),
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
        model=["decoder-small"],
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


def test_eval_one_model_uses_vlm_prepare_outputs_for_vlm_suite(tmp_path: Path, monkeypatch) -> None:
    dataset_dir = tmp_path / "MMMU_Pro_vision"
    dataset_dir.mkdir()
    dataset = dataset_dir / "mmmu_pro_vision_dataset.json"
    _write_vlm_mmmu_pro_vision(dataset)
    suite = task_eval.suite_by_id(task_eval.load_suites(), "vlm_mmmu_pro_vision_mcq")
    model = {
        "name": "vl-primary",
        "hf_id": "example-org/vl-primary",
        "bundle": "vl-primary.trtfb",
        "max_cache_length": 512,
        "precision": "fp32",
        "trust_remote_code": False,
        "build_args": {},
        "quantization": {},
        "task_eval": {
            "vlm_fallback_prompt_template": "<image>{prompt}",
        },
    }
    calls: list[str] = []

    def fake_run_hf(_args, _model, work_dir):
        calls.append("hf")
        prompts = task_eval.load_jsonl(work_dir / "prompts.jsonl")
        manifest = json.loads(Path(work_dir, "manifest.json").read_text(encoding="utf-8"))
        assert prompts[0]["images"] == [str(dataset_dir / "images" / "sample.jpg")]
        assert manifest["task_eval"] == {
            "vlm_fallback_prompt_template": "<image>{prompt}",
        }
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
        model=["vl-primary"],
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


def test_eval_one_model_skips_prompt_length_check_for_asr_suite(
    tmp_path: Path, monkeypatch
) -> None:
    dataset_dir = tmp_path / "librispeech_clean_test"
    dataset_dir.mkdir()
    dataset = dataset_dir / "librispeech_clean_test.json"
    _write_asr_librispeech(dataset)
    suite = task_eval.suite_by_id(task_eval.load_suites(), "librispeech_clean_asr")
    model = {
        "name": "whisper-tiny-fp16",
        "hf_id": "openai/whisper-tiny",
        "bundle": "whisper-tiny-fp16.trtfb",
        "max_cache_length": 64,
        "precision": "fp16",
        "trust_remote_code": False,
        "build_args": {},
        "quantization": {},
        "family": "whisper",
        "reference_family": "asr_whisper",
        "task_eval": {},
    }
    calls: list[str] = []

    def fake_prompt_length(**_kwargs):
        raise AssertionError("ASR suite should not run text prompt length validation")

    def fake_run_hf(_args, _model, work_dir):
        calls.append("hf")
        Path(work_dir, "hf_predictions.json").write_text(
            json.dumps(
                {"responses": [{"sample_id": "clean_000000", "output_text": "The quick brown fox"}]}
            ),
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
        assert prompts[0]["audio"].endswith("clean_000000.wav")
        Path(args.work_dir, "trtfb_predictions.json").write_text(
            json.dumps(
                {"responses": [{"sample_id": "clean_000000", "output_text": "the quick brown fox"}]}
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(task_eval, "max_prompt_token_length", fake_prompt_length)
    monkeypatch.setattr(task_eval, "run_hf_reference_subprocess", fake_run_hf)
    monkeypatch.setattr(task_eval, "ensure_bundle", fake_ensure_bundle)
    monkeypatch.setattr(task_eval, "run_trtfb", fake_run_trtfb)

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
        model=["whisper-tiny-fp16"],
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
    assert result["mode"] == "asr_transcript"
    assert result["max_prompt_tokens"] is None
    assert result["hf_accuracy"] == 1.0
    assert result["trtfb_accuracy"] == 1.0
    assert result["prediction_agreement_rate"] == 1.0


def test_eval_one_model_runs_hf_for_golden_snapshot_vlm_model(tmp_path: Path, monkeypatch) -> None:
    dataset_dir = tmp_path / "OCRBench_v2" / "unified"
    dataset_dir.mkdir(parents=True)
    dataset = dataset_dir / "dataset.json"
    _write_ocrbench_unified(dataset)
    suite = task_eval.suite_by_id(task_eval.load_suites(), "ocrbench_v2_unified")
    model = {
        "name": "deepseek-ocr-l0",
        "hf_id": "deepseek-ai/DeepSeek-OCR-2",
        "bundle": "deepseek-ocr-l0.trtfb",
        "max_cache_length": 4096,
        "precision": "fp32",
        "trust_remote_code": True,
        "reference_backend": "golden_snapshot",
        "build_args": {},
        "quantization": {},
    }
    calls: list[str] = []

    def fake_run_hf(_args, _model, work_dir):
        calls.append("hf")
        Path(work_dir, "hf_predictions.json").write_text(
            json.dumps({"responses": [{"sample_id": "ocrbench_v2_000000", "output_text": "on"}]}),
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
        assert prompts[0]["images"] == [str(dataset_dir / "images" / "ocrbench_v2_000000.jpg")]
        Path(args.work_dir, "trtfb_predictions.json").write_text(
            json.dumps({"responses": [{"sample_id": "ocrbench_v2_000000", "output_text": "on"}]}),
            encoding="utf-8",
        )

    monkeypatch.setattr(task_eval, "run_hf_reference_subprocess", fake_run_hf)
    monkeypatch.setattr(task_eval, "ensure_bundle", fake_ensure_bundle)
    monkeypatch.setattr(task_eval, "run_trtfb", fake_run_trtfb)

    args = argparse.Namespace(
        work_root=str(tmp_path / "work"),
        dataset=str(dataset),
        limit=1,
        subject="",
        sample_seed=None,
        force_hf=False,
        force_build=False,
        build_max_cache_length=None,
        skip_prompt_length_check=True,
        bundle="",
        model=["deepseek-ocr-l0"],
        engine_dir="",
        trtmc_binary="build/trtmc",
        extra_build_arg=[],
        hf_dtype="auto",
        hf_device="cuda",
        hf_device_map="",
        hf_attn_impl="",
        trust_remote_code=True,
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
    assert result["mode"] == "ocrbench_v2"
    assert result["hf_reference_status"] == "ran"
    assert result["hf_accuracy"] == 1.0
    assert result["prediction_agreement_rate"] == 1.0
    assert result["trtfb_accuracy"] == 1.0
    assert (tmp_path / "work" / suite["id"] / model["name"] / "hf_predictions.json").is_file()


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


def test_eval_stops_after_oom_when_gpu_cleanup_is_not_confirmed(
    tmp_path: Path, monkeypatch
) -> None:
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
