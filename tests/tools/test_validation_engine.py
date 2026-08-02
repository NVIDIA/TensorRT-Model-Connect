# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import hashlib
import io
import json
import struct
import sys
import types
import wave
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image

from tools import prepare_media_validation_datasets as prepare_media
from tools.validation import engine as validation_engine


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


def _write_stsbenchmark(path: Path) -> None:
    rows = [
        {
            "split": "test",
            "genre": "main-captions",
            "dataset": "MSRvid",
            "sid": "0001",
            "score": 5.0,
            "sentence1": "A plane is taking off.",
            "sentence2": "An airplane is taking off.",
        },
        {
            "split": "test",
            "genre": "main-news",
            "dataset": "headlines",
            "sid": "0002",
            "score": 0.0,
            "sentence1": "Stocks rose on Monday.",
            "sentence2": "A dog sleeps by the fire.",
        },
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _write_time_series_csv(path: Path, row_count: int = 40) -> None:
    lines = ["date,A,B"]
    for index in range(row_count):
        lines.append(f"2026-01-{index + 1:02d},{index},{100 + index}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_default_suites_include_etth1_time_series_parity() -> None:
    suite = validation_engine.suite_by_id(validation_engine.load_suites(), "etth1_time_series_parity")

    assert suite["dataset"]["kind"] == "time_series_csv"
    assert suite["dataset"]["source_revision"] == (
        "1d16c8f4f943005d613b5bc962e9eeb06058cf07"
    )
    assert suite["dataset"]["sha256"] == (
        "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066"
    )
    assert suite["scoring"]["scorer"] == "time_series_parity"
    assert suite["gates"]["min_sample_agreement_rate"] == 1.0
    assert suite["default_model_names"] == [
        "chronos-bolt-tiny-official",
        "patchtsmixer-granite-official",
        "patchtst-etth1-regression-distribution",
        "patchtst-granite-official",
        "timesfm-2.0-500m-official",
    ]
    models = validation_engine.load_manifest_records()
    plan = validation_engine.build_plan([suite], models)
    assert {row["model"] for row in plan if row["selected"]} == set(suite["default_model_names"])
    assert suite["ci"]["eligible"] is True
    assert suite["ci"]["lane"] == "nightly"
    assert suite["ci"]["limit"] == 10
    assert suite["ci"]["sample_seed"] == 20260715
    expected_gates = {
        "chronos-bolt-tiny-official": (1.0e-06, 8.0e-06),
        "patchtsmixer-granite-official": (5.0e-04, 2.5e-02),
        "patchtst-etth1-regression-distribution": (1.0e-03, 1.0e-03),
        "patchtst-granite-official": (1.5e-03, 3.5e-02),
        "timesfm-2.0-500m-official": (4.0e-03, 7.0e-03),
    }
    for model_name, (max_relative_l2, max_absolute_error) in expected_gates.items():
        profile = suite["model_profiles"][model_name]
        assert profile["gates"]["max_relative_l2"] == max_relative_l2
        assert profile["gates"]["max_absolute_error"] == max_absolute_error
        assert suite["model_overrides"]["by_model"][model_name]["time_series"]["stride"] == 24


def test_prepare_time_series_csv_dataset_uses_time_major_windows(tmp_path: Path) -> None:
    dataset_path = tmp_path / "series.csv"
    _write_time_series_csv(dataset_path)
    work_dir = tmp_path / "work"
    suite = {
        "id": "time_series_test",
        "dataset": {"kind": "time_series_csv", "name": "test series"},
        "scoring": {"scorer": "time_series_parity"},
    }

    validation_engine.prepare_time_series_csv_dataset(
        dataset_path=dataset_path,
        work_dir=work_dir,
        suite=suite,
        limit=2,
        validation_config={
            "time_series": {
                "input_columns": ["A", "B"],
                "target_columns": ["B"],
                "input_key": "branch_input",
                "context_length": 3,
                "prediction_length": 2,
                "stride": 2,
                "test_fraction": 0.5,
                "frequency": 0,
            }
        },
    )

    prompts = validation_engine.load_jsonl(work_dir / "prompts.jsonl")
    answers = json.loads((work_dir / "answers.json").read_text(encoding="utf-8"))
    manifest = json.loads((work_dir / "manifest.json").read_text(encoding="utf-8"))
    assert len(prompts) == len(answers["requests"]) == 2
    assert prompts[0]["dataset_index"] == 20
    assert prompts[0]["context_index"] == 17
    assert prompts[0]["inputs"] == {
        "branch_input": [17.0, 117.0, 18.0, 118.0, 19.0, 119.0],
        "trunk_input": [0],
    }
    assert prompts[0]["target_values"] == [120.0, 121.0]
    assert manifest["time_series"]["context_length"] == 3


def test_time_series_case_replaces_manifest_probe_inputs() -> None:
    template = SimpleNamespace(name="template", inputs={"field_input": [999], "stale": True})

    case = validation_engine._time_series_case_for_request(
        template,
        {
            "sample_id": "etth1_000001",
            "inputs": {"branch_input": [1.0, 2.0], "trunk_input": [0]},
        },
        0,
    )

    assert case.name == "etth1_000001"
    assert case.inputs == {"branch_input": [1.0, 2.0], "trunk_input": [0]}
    assert template.inputs == {"field_input": [999], "stale": True}


def test_time_series_trtfb_reuses_model_runner_and_writes_run_log(
    tmp_path: Path, monkeypatch
) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "manifest.json").write_text(
        json.dumps({"dataset_kind": "time_series_csv"}), encoding="utf-8"
    )
    (work_dir / "prompts.jsonl").write_text(
        json.dumps(
            {
                "sample_id": "etth1_011520",
                "inputs": {"field_input": [1.0, 2.0, 3.0]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    template = SimpleNamespace(
        name="template",
        inputs={"field_input": [999.0]},
        stages=[],
        bundle="template.trtfb",
    )
    captured_inputs: list[dict] = []

    class FakeRunner:
        def run_stage(self, case, stage, context):
            captured_inputs.append(dict(case.inputs))
            return SimpleNamespace(
                data={"output_field": [1.5, 2.5], "output_dim": 2},
                metadata={"command": ["trtmc", "solve"], "returncode": 0},
                timing_s=0.01,
            )

    monkeypatch.setattr(
        validation_engine,
        "_load_time_series_validation_plugins",
        lambda _work_dir: (template, object(), FakeRunner()),
    )
    validation_engine.run_time_series_trtfb(
        SimpleNamespace(
            work_dir=str(work_dir),
            raw_output="",
            predictions="",
            log="",
            bundle=str(tmp_path / "model.trtfb"),
            trtmc_binary="trtmc",
            hf_python="python",
            model_plugin_dir="",
        )
    )

    predictions = json.loads(
        (work_dir / "trtfb_predictions.json").read_text(encoding="utf-8")
    )
    log_row = validation_engine.load_jsonl(work_dir / "trtfb_run.log")[0]
    assert captured_inputs == [{"field_input": [1.0, 2.0, 3.0]}]
    assert predictions["responses"][0]["output_values"] == [1.5, 2.5]
    assert log_row["sample_id"] == "etth1_011520"
    assert log_row["command"] == ["trtmc", "solve"]


def test_time_series_parity_requires_every_sample_to_pass() -> None:
    hf = {
        "responses": [
            {"sample_id": "a", "output_values": [1.0, 2.0]},
            {"sample_id": "b", "output_values": [3.0, 4.0]},
        ]
    }
    trtfb = {
        "responses": [
            {"sample_id": "a", "output_values": [1.0, 2.0001]},
            {"sample_id": "b", "output_values": [3.0, 4.1]},
        ]
    }

    summary = validation_engine.compare_time_series_prediction_sets(
        hf,
        trtfb,
        gates={
            "max_relative_l2": 1e-3,
            "max_absolute_error": 1e-3,
            "min_sample_agreement_rate": 1.0,
        },
    )

    assert summary["status"] == "failed"
    assert summary["passed_count"] == 1
    assert summary["sample_agreement_rate"] == 0.5
    assert summary["cases"][0]["passed"] is True
    assert summary["cases"][1]["passed"] is False


def test_default_suites_include_ocrbench_v2_unified() -> None:
    suites = validation_engine.load_suites()
    suite = validation_engine.suite_by_id(suites, "ocrbench_v2_unified")

    assert suite["dataset"]["kind"] == "vlm_unified_json"
    assert suite["scoring"]["scorer"] == "ocrbench_v2"
    assert suite["selectors"]["runtime_strategies"] == ["deepseek_ocr_vision_language"]
    assert suite["selectors"]["families"] == ["deepseek_ocr"]


def test_default_suites_include_librispeech_clean_asr() -> None:
    suites = validation_engine.load_suites()
    suite = validation_engine.suite_by_id(suites, "librispeech_clean_asr")

    assert suite["dataset"]["kind"] == "asr_chat_json"
    assert suite["scoring"]["scorer"] == "asr_transcript"
    assert suite["selectors"]["runtime_strategies"] == [
        "whisper_speech_to_text",
        "canary_speech_to_text",
    ]
    assert suite["selectors"]["families"] == ["whisper", "canary"]


def test_default_suites_do_not_split_librispeech_asr_by_family() -> None:
    suite_ids = {suite["id"] for suite in validation_engine.load_suites()}

    assert "librispeech_clean_asr" in suite_ids
    assert "librispeech_clean_asr_whisper" not in suite_ids
    assert "librispeech_clean_asr_canary" not in suite_ids


def test_default_suites_include_seedtts_tts_intelligibility() -> None:
    suite = validation_engine.suite_by_id(validation_engine.load_suites(), "seedtts_en_tts_intelligibility")

    assert suite["dataset"]["kind"] == "seedtts_json"
    assert suite["scoring"]["scorer"] == "tts_intelligibility"
    assert suite["default_model_names"] == [
        "bark-large",
        "bark-small",
        "magpie-tts-357m",
    ]


def test_default_suites_include_librispeech_clean_asr_streaming() -> None:
    suites = validation_engine.load_suites()
    suite = validation_engine.suite_by_id(suites, "librispeech_clean_asr_streaming")

    assert suite["dataset"]["kind"] == "asr_chat_json"
    assert suite["scoring"]["scorer"] == "asr_transcript"
    assert suite["default_model_names"] == [
        "nemotron-3.5-asr-streaming-0.6b",
        "nemotron-speech-streaming-en-0.6b",
    ]
    assert suite["selectors"]["runtime_strategies"] == [
        "nemotron_speech_streaming_speech_to_text_rnnt"
    ]
    assert suite["selectors"]["families"] == ["nemotron_speech_streaming"]
    model = next(
        model
        for model in validation_engine.load_manifest_records()
        if model["name"] == "nemotron-3.5-asr-streaming-0.6b"
    )
    resolved = validation_engine.resolve_suite_for_model(suite, model)
    assert resolved["generation"]["language"] == "en-US"
    assert resolved["generation"]["streaming"] == {
        "enabled": True,
        "chunk_ms": 1120,
        "att_context_size": [56, 13],
    }
    non_streaming = validation_engine.suite_by_id(suites, "librispeech_clean_asr")
    assert "nemotron_speech_streaming" in non_streaming["selectors"]["exclude_families"]


def test_default_suites_include_text_generation_gap_models() -> None:
    suites = validation_engine.load_suites()
    expected = {
        "humaneval_code_continuation_parity": ["codegen-350m", "starcoder2-3b"],
        "wikitext103_distilgpt2_continuation_parity": ["distilgpt2"],
        "newstest2019_en_ru_marian_translation_parity": ["marian-en-ru"],
        "wmt14_en_de_t5_translation_parity": ["t5-small"],
        "flores200_en_fr_riva_translation_parity": ["riva-translate-4b"],
    }

    for suite_id, model_names in expected.items():
        suite = validation_engine.suite_by_id(suites, suite_id)
        assert suite["dataset"]["kind"] == "text_generation_json"
        assert suite["scoring"]["scorer"] == "continuation"
        assert suite["default_model_names"] == model_names
        assert suite["gates"] == {}

    for suite_id in (
        "newstest2019_en_ru_marian_translation_parity",
        "wmt14_en_de_t5_translation_parity",
        "flores200_en_fr_riva_translation_parity",
    ):
        assert validation_engine.suite_by_id(suites, suite_id)["scoring"]["task_metric"] == (
            "sacrebleu"
        )


def test_default_suites_include_one_dpg_bench_diffusion_image_suite() -> None:
    suites = validation_engine.load_suites()
    suite = validation_engine.suite_by_id(suites, "dpg_bench_diffusion_image")

    assert suite["dataset"]["kind"] == "diffusion_prompt_json"
    assert suite["selectors"]["task_strategies"] == ["diffusion_media_generation"]
    assert len(suite["selectors"]["families"]) == 4
    assert {"flux", "pixart", "z_image"} < set(suite["selectors"]["families"])
    assert len(suite["selectors"]["runtime_strategies"]) == 4
    assert {
        "diffusion_flux", "diffusion_pixart", "diffusion_zimage"
    } < set(suite["selectors"]["runtime_strategies"])
    expected_non_neutral_models = {
        "flux-schnell-l0",
        "flux-2-dev-l0",
        "flux-2-dev-fp8-l0",
        "pixart-sigma-1024",
        "z-image-turbo",
    }
    assert len(suite["default_model_names"]) == 7
    assert expected_non_neutral_models < set(suite["default_model_names"])
    assert len(suite["selectors"]["exclude_model_names"]) == 1
    assert suite["selectors"]["exclude_model_names"][0].endswith("image-edit-2511")
    assert suite["scoring"]["scorer"] == "diffusion_image_clip_parity"
    assert suite["gates"]["min_trt_hf_image_clip_cosine"] == 0.85
    assert suite["ci"] == {
        "eligible": False,
        "lane": "local_only",
        "notes": "Local-only until the DPG-Bench scorecard is visually calibrated.\n",
    }


def test_default_suites_include_media_generation_gap_models() -> None:
    suites = validation_engine.load_suites()

    video = validation_engine.suite_by_id(suites, "vbench_t2v_diffusion_video")
    assert video["default_model_names"] == [
        "ltx-video-l0",
        "wan21-t2v-1.3b-l0",
        "wan21-t2v-1.3b",
    ]
    assert video["dataset"]["default_path"] == (
        "/mnt/data/VBench/vbench_t2v_task_eval.json"
    )
    assert video["generation"]["use_shared_initial_latents"] is True
    assert video["gates"]["min_trt_hf_image_clip_cosine"] == 0.85
    assert video["gates"]["require_matching_initial_latents"] == 1
    models = {model["name"]: model for model in validation_engine.load_manifest_records()}
    ltx = validation_engine.resolve_suite_for_model(video, models["ltx-video-l0"])
    wan = validation_engine.resolve_suite_for_model(video, models["wan21-t2v-1.3b-l0"])
    assert ltx["generation"]["text_max_length"] == 128
    assert wan["generation"]["text_max_length"] == 226

    image_edit = validation_engine.suite_by_id(suites, "gedit_bench_image_edit")
    assert image_edit["default_model_names"] == image_edit["selectors"]["model_names"]
    assert len(image_edit["default_model_names"]) == 1
    assert image_edit["dataset"]["asset_fields"] == ["image"]
    assert image_edit["gates"]["require_matching_initial_latents"] == 1

    world_model = validation_engine.suite_by_id(
        suites, "sana_wm_benchmark_diffusion_video"
    )
    assert world_model["default_model_names"] == ["sana-wm-bidirectional"]
    assert world_model["dataset"]["asset_fields"] == [
        "image",
        "prompt_file",
        "camera_intrinsics_file",
    ]

    models = validation_engine.load_manifest_records()
    for suite in (video, image_edit, world_model):
        selected = validation_engine.selected_models_for_suite(
            suite, models, single_device_only=True
        )
        assert [model["name"] for model in selected] == suite["default_model_names"]


def test_default_suites_include_model_aligned_vision_tasks() -> None:
    suites = validation_engine.load_suites()

    classification = validation_engine.suite_by_id(suites, "imagenette_image_classification")
    assert classification["dataset"]["kind"] == "image_classification_json"
    assert classification["scoring"]["task_metric"] == "top1_accuracy"
    assert classification["default_model_names"] == [
        "timm-vit-base-p16-224-augreg-in21k-ft-in1k"
    ]

    semantic = validation_engine.suite_by_id(suites, "ade20k_semantic_segmentation")
    assert semantic["dataset"]["kind"] == "semantic_segmentation_json"
    assert semantic["dataset"]["num_classes"] == 150
    assert semantic["scoring"]["task_metric"] == "mean_iou"
    assert semantic["default_model_names"] == ["segformer-b0-ade"]

    prompted = validation_engine.suite_by_id(suites, "coco2017_prompted_segmentation")
    assert prompted["dataset"]["kind"] == "prompted_segmentation_json"
    assert prompted["default_model_names"] == ["sam-vit-base", "sam3"]
    assert prompted["model_overrides"]["by_family"]["sam"]["prompt_mode"] == "point"
    assert prompted["model_overrides"]["by_family"]["sam3"]["prompt_mode"] == "text"


def test_default_suites_include_scifact_reranking_parity() -> None:
    suite = validation_engine.suite_by_id(validation_engine.load_suites(), "beir_scifact_reranking")
    selected = validation_engine.selected_models_for_suite(
        suite, validation_engine.load_manifest_records(), single_device_only=True
    )

    assert suite["dataset"]["kind"] == "reranking_json"
    assert suite["scoring"]["scorer"] == "reranking_parity"
    assert [model["name"] for model in selected] == ["nemotron-rerank-vl-1b-v2"]
    assert suite["gates"]["min_sample_pass_rate"] == 1.0


def test_prepare_reranking_dataset_preserves_query_documents_and_gold(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "scifact.json"
    dataset.write_text(
        json.dumps(
            {
                "dataset": "BEIR SciFact test",
                "requests": [
                    {
                        "id": "scifact-1",
                        "subset": "test",
                        "query": "Does the evidence support the claim?",
                        "documents": ["relevant evidence", "distractor evidence"],
                        "relevant_document_indices": [0],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    suite = validation_engine.suite_by_id(validation_engine.load_suites(), "beir_scifact_reranking")

    outputs = validation_engine.prepare_reranking_dataset(
        dataset_path=dataset,
        work_dir=tmp_path / "work",
        suite=suite,
        limit=1,
    )

    prompts = validation_engine.load_jsonl(outputs["prompts"])
    answers = json.loads(outputs["answers"].read_text(encoding="utf-8"))
    assert prompts[0]["query"] == "Does the evidence support the claim?"
    assert prompts[0]["documents"] == ["relevant evidence", "distractor evidence"]
    assert answers["requests"][0]["relevant_document_indices"] == [0]


def test_reranking_parity_records_each_low_agreement_sample() -> None:
    from tests.e2e.models.eagle_vlm.e2e_plugins.comparators.reranking import (
        RerankingComparator,
    )

    answers = {
        "requests": [
            {"sample_id": "agree", "relevant_document_indices": [0]},
            {"sample_id": "disagree", "relevant_document_indices": [0]},
        ]
    }
    hf = {
        "responses": [
            {"sample_id": "agree", "scores": [0.9, 0.2, 0.1]},
            {"sample_id": "disagree", "scores": [0.9, 0.2, 0.1]},
        ]
    }
    trtfb = {
        "responses": [
            {"sample_id": "agree", "scores": [0.8, 0.3, 0.1]},
            {"sample_id": "disagree", "scores": [0.1, 0.2, 0.9]},
        ]
    }

    summary = validation_engine.compare_reranking_prediction_sets(
        hf,
        trtfb,
        answers,
        gates={
            "pairwise_ordering_agreement": 1.0,
            "kendall_tau": 1.0,
            "spearman_rho": 1.0,
            "score_correlation": 0.95,
            "min_sample_pass_rate": 1.0,
        },
        comparator=RerankingComparator(),
    )

    assert summary["status"] == "failed"
    assert summary["sample_pass_rate"] == 0.5
    assert len(summary["cases"]) == 2
    disagreement = next(
        case for case in summary["cases"] if case["sample_id"] == "disagree"
    )
    assert disagreement["passed"] is False
    assert disagreement["metrics"]["pairwise_ordering_agreement"]["value"] < 1.0


def test_prepare_vision_datasets_resolves_model_specific_assets(tmp_path: Path) -> None:
    image = tmp_path / "image.jpg"
    mask = tmp_path / "mask.png"
    category_mask = tmp_path / "category.png"
    for path in (image, mask, category_mask):
        path.write_bytes(b"fixture")

    cases = [
        (
            "imagenette_image_classification",
            {
                "id": "class-1",
                "image": image.name,
                "label": 217,
                "label_name": "English springer",
                "synset": "n02102040",
            },
            validation_engine.prepare_image_classification_dataset,
        ),
        (
            "ade20k_semantic_segmentation",
            {"id": "seg-1", "image": image.name, "mask": mask.name, "subset": "validation"},
            validation_engine.prepare_semantic_segmentation_dataset,
        ),
        (
            "coco2017_prompted_segmentation",
            {
                "id": "prompt-1",
                "image": image.name,
                "instance_mask": mask.name,
                "category_mask": category_mask.name,
                "point_x": 0.25,
                "point_y": 0.75,
                "text_prompt": "dog",
                "category": "dog",
            },
            validation_engine.prepare_prompted_segmentation_dataset,
        ),
    ]
    suites = validation_engine.load_suites()
    for suite_id, request, prepare in cases:
        dataset = tmp_path / f"{suite_id}.json"
        dataset.write_text(
            json.dumps({"dataset": suite_id, "requests": [request]}), encoding="utf-8"
        )
        outputs = prepare(
            dataset_path=dataset,
            work_dir=tmp_path / f"work-{suite_id}",
            suite=validation_engine.suite_by_id(suites, suite_id),
        )
        rows = validation_engine.load_jsonl(outputs["prompts"])
        assert len(rows) == 1
        assert rows[0]["image"] == str(image.resolve())
        manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
        assert manifest["request_count"] == 1


def test_image_classification_parity_separates_accuracy_and_agreement() -> None:
    answers = {
        "requests": [
            {"sample_id": "a", "label": 1, "label_name": "one"},
            {"sample_id": "b", "label": 2, "label_name": "two"},
        ]
    }
    hf = {"responses": [{"sample_id": "a", "top_class": 1}, {"sample_id": "b", "top_class": 0}]}
    trtfb = {
        "responses": [
            {"sample_id": "a", "top_class": 1},
            {"sample_id": "b", "top_class": 0},
        ]
    }

    summary = validation_engine.compare_image_classification_prediction_sets(
        hf, trtfb, answers, gates={"min_top1_agreement": 1.0}
    )

    assert summary["status"] == "passed"
    assert summary["hf_top1_accuracy"] == 0.5
    assert summary["trtfb_top1_accuracy"] == 0.5
    assert summary["top1_agreement"] == 1.0


def test_image_classification_runner_forwards_model_plugin_dir(monkeypatch) -> None:
    from types import SimpleNamespace

    from tests.e2e.models.timm_vit.e2e_plugins.runners import image_classification
    from tests.e2e_harness.contracts import RunContext

    case = validation_engine.load_manifest(
        Path(
            "tests/e2e/models/timm_vit/manifests/"
            "timm-vit-base-p16-224-augreg-in21k-ft-in1k.json"
        )
    )
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout='{"top_class": 1}', stderr="")

    monkeypatch.setattr(image_classification.subprocess, "run", fake_run)
    context = RunContext(
        case=case,
        binary_path="/tmp/trtmc",
        engine_dir="/tmp/engines",
        model_plugin_dir="/tmp/plugins/timm_vit",
    )

    image_classification.ImageClassificationRunner().run_stage(
        case, case.stages[0], context
    )

    assert commands[0][-2:] == ["--model-plugin-dir", "/tmp/plugins/timm_vit"]


def test_semantic_segmentation_parity_reports_dataset_miou(tmp_path: Path) -> None:
    import numpy as np

    ground = np.array([[0, 0], [1, 1]], dtype=np.uint8)
    hf_map = ground.copy()
    trtfb_map = ground.copy()
    paths = {}
    for name, values in (("ground", ground), ("hf", hf_map), ("trtfb", trtfb_map)):
        path = tmp_path / f"{name}.npy"
        np.save(path, values)
        paths[name] = path
    answers = {"requests": [{"sample_id": "seg", "mask": str(paths["ground"])}]}
    hf = {"responses": [{"sample_id": "seg", "class_map_path": str(paths["hf"])}]}
    trtfb = {
        "responses": [{"sample_id": "seg", "class_map_path": str(paths["trtfb"])}]
    }

    summary = validation_engine.compare_semantic_segmentation_prediction_sets(
        hf,
        trtfb,
        answers,
        gates={},
        num_classes=2,
        ignore_index=255,
    )

    assert summary["status"] == "passed"
    assert summary["hf_mean_iou"] == 1.0
    assert summary["trtfb_mean_iou"] == 1.0
    assert summary["backend_pixel_agreement"] == 1.0


def test_semantic_segmentation_uses_raw_hf_map_for_backend_parity(
    tmp_path: Path,
) -> None:
    import numpy as np

    ground = np.array([[0, 0], [1, 1]], dtype=np.uint8)
    hf_postprocessed = ground.copy()
    hf_raw = np.array([[0]], dtype=np.uint8)
    trtfb_raw = hf_raw.copy()
    paths = {}
    for name, values in (
        ("ground", ground),
        ("hf", hf_postprocessed),
        ("hf_raw", hf_raw),
        ("trtfb", trtfb_raw),
    ):
        path = tmp_path / f"{name}.npy"
        np.save(path, values)
        paths[name] = path
    answers = {"requests": [{"sample_id": "seg", "mask": str(paths["ground"])}]}
    hf = {
        "responses": [
            {
                "sample_id": "seg",
                "class_map_path": str(paths["hf"]),
                "raw_class_map_path": str(paths["hf_raw"]),
            }
        ]
    }
    trtfb = {
        "responses": [{"sample_id": "seg", "class_map_path": str(paths["trtfb"])}]
    }

    summary = validation_engine.compare_semantic_segmentation_prediction_sets(
        hf,
        trtfb,
        answers,
        gates={"max_mean_iou_drop_from_hf": 1.0},
        num_classes=2,
        ignore_index=255,
    )

    assert summary["hf_mean_iou"] == 1.0
    assert summary["backend_pixel_agreement"] == 1.0
    assert summary["backend_mean_iou"] == 1.0


def test_prompted_segmentation_uses_family_prompt_semantics(tmp_path: Path) -> None:
    import numpy as np

    ground = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    masks = np.stack([ground, np.zeros_like(ground)])
    ground_path = tmp_path / "ground.npy"
    hf_path = tmp_path / "hf.npy"
    trtfb_path = tmp_path / "trtfb.npy"
    np.save(ground_path, ground)
    np.save(hf_path, masks)
    np.save(trtfb_path, masks)
    answers = {
        "requests": [
            {
                "sample_id": "prompt",
                "instance_mask": str(ground_path),
                "category_mask": str(ground_path),
                "text_prompt": "object",
            }
        ]
    }
    hf = {
        "responses": [
            {"sample_id": "prompt", "masks_path": str(hf_path), "mask_scores": [0.9, 0.1]}
        ]
    }
    trtfb = {
        "responses": [
            {
                "sample_id": "prompt",
                "masks_path": str(trtfb_path),
                "mask_scores": [0.9, 0.1],
            }
        ]
    }

    point = validation_engine.compare_prompted_segmentation_prediction_sets(
        hf,
        trtfb,
        answers,
        gates={},
        prompt_mode="point",
        ground_truth_mask_field="instance_mask",
    )
    text = validation_engine.compare_prompted_segmentation_prediction_sets(
        hf,
        trtfb,
        answers,
        gates={},
        prompt_mode="text",
        ground_truth_mask_field="category_mask",
    )

    assert point["status"] == "passed"
    assert text["status"] == "passed"
    assert point["mean_backend_mask_iou"] == 1.0
    assert text["mean_backend_mask_iou"] == 1.0


def test_prompted_segmentation_empty_prediction_is_a_comparison_failure(
    tmp_path: Path,
) -> None:
    ground = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    ground_path = tmp_path / "ground.npy"
    hf_path = tmp_path / "hf.npy"
    np.save(ground_path, ground)
    np.save(hf_path, ground[None, ...])
    answers = {
        "requests": [
            {
                "sample_id": "prompt",
                "category_mask": str(ground_path),
                "text_prompt": "object",
            }
        ]
    }
    hf = {
        "responses": [
            {
                "sample_id": "prompt",
                "masks_path": str(hf_path),
                "mask_scores": [0.9],
            }
        ]
    }
    trtfb = {
        "responses": [
            {
                "sample_id": "prompt",
                "masks_path": "",
                "mask_scores": [],
                "num_masks": 0,
                "empty_prediction": True,
                "returncode": 1,
            }
        ]
    }

    summary = validation_engine.compare_prompted_segmentation_prediction_sets(
        hf,
        trtfb,
        answers,
        gates={},
        prompt_mode="text",
        ground_truth_mask_field="category_mask",
    )

    assert summary["status"] == "failed"
    assert summary["valid_count"] == 1
    assert summary["mean_backend_mask_iou"] == 0.0
    assert summary["cases"][0]["trtfb_empty_prediction"] is True


def test_vision_response_preserves_prompted_segmentation_empty_prediction(
    tmp_path: Path,
) -> None:
    response = validation_engine._vision_response(
        case=SimpleNamespace(name="prompt"),
        source="trtfb",
        output=SimpleNamespace(
            data={"masks": [], "num_masks": 0},
            metadata={
                "returncode": 1,
                "stderr": "Error: prompted segmentation produced no masks",
            },
            timing_s=0.1,
        ),
        dataset_kind="prompted_segmentation_json",
        prompt_row={"image": "image.jpg", "text_prompt": "train"},
        artifact_dir=tmp_path,
    )

    assert response["returncode"] == 1
    assert response["num_masks"] == 0
    assert response["empty_prediction"] is True


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (
            {
                "mode": "image_classification_parity",
                "hf_top1_accuracy": 1.0,
                "trtfb_top1_accuracy": 1.0,
                "top1_agreement": 1.0,
            },
            "hf_top1=1.0000",
        ),
        (
            {
                "mode": "semantic_segmentation_parity",
                "hf_mean_iou": 0.5,
                "trtfb_mean_iou": 0.5,
                "backend_mean_iou": 1.0,
            },
            "backend_miou=1.0000",
        ),
        (
            {
                "mode": "prompted_segmentation_parity",
                "mean_backend_mask_iou": 1.0,
                "hf_mean_ground_truth_iou": 0.8,
                "trtfb_mean_ground_truth_iou": 0.8,
            },
            "backend_mask_iou=1.0000",
        ),
    ],
)
def test_vision_result_lines_use_task_specific_metrics(result, expected) -> None:
    result.update({"hf_reused": False, "bundle_built": False, "status": "passed"})

    line = validation_engine._format_result_line({"name": "vision-model"}, result)

    assert expected in line


def test_default_suites_include_encoder_embedding_parity() -> None:
    suite = validation_engine.suite_by_id(
        validation_engine.load_suites(), "stsbenchmark_encoder_embedding_parity"
    )
    models = validation_engine.load_manifest_records()

    selected = validation_engine.selected_models_for_suite(
        suite, models, single_device_only=True
    )

    assert suite["dataset"]["kind"] == "sts_pair_jsonl"
    assert suite["scoring"]["scorer"] == "encoder_embedding_parity"
    assert suite["selectors"]["task_strategies"] == [
        "encoder_only_nlp",
        "embedding",
    ]
    assert suite["selectors"]["user_contracts"] == [
        "representation_parity",
        "embedding_vector",
    ]
    assert len(suite["default_model_names"]) == 19
    assert len(selected) == 19
    assert {model["name"] for model in selected} == set(suite["default_model_names"])
    models_by_name = {model["name"]: model for model in models}
    assert validation_engine.resolve_suite_for_model(
        suite, models_by_name["bert-base-uncased"]
    )["gates"]["min_vector_cosine"] == 0.999
    assert validation_engine.resolve_suite_for_model(
        suite, models_by_name["convbert-base"]
    )["gates"]["min_vector_cosine"] == 0.95
    assert validation_engine.resolve_suite_for_model(
        suite, models_by_name["fnet-base"]
    )["gates"]["min_vector_cosine"] == 0.6
    assert validation_engine.resolve_suite_for_model(
        suite, models_by_name["fnet-base"]
    )["gates"]["max_pair_cosine_abs_delta"] == 0.1


def test_phi4_multimodal_workload_reserves_dynamic_hd_cache() -> None:
    suite = validation_engine.suite_by_id(
        validation_engine.load_suites(),
        "mmmu_pro_vision_plugin_parity",
    )
    model = next(
        model
        for model in validation_engine.load_manifest_records()
        if model["name"] == "phi4-multimodal"
    )

    assert model["max_cache_length"] == 768
    assert suite["build"]["min_max_cache_length"] == 1024
    assert validation_engine.requested_build_max_cache_length(suite, model) == 1024


def test_prepare_stsbenchmark_expands_each_pair_to_shared_sentence_inputs(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "stsbenchmark_test.jsonl"
    _write_stsbenchmark(dataset)
    suite = validation_engine.suite_by_id(
        validation_engine.load_suites(), "stsbenchmark_encoder_embedding_parity"
    )

    outputs = validation_engine.prepare_sts_pair_dataset(
        dataset_path=dataset,
        work_dir=tmp_path / "work",
        suite=suite,
        limit=1,
        subject="main-captions",
        sample_seed=None,
    )

    prompts = validation_engine.load_jsonl(outputs["prompts"])
    answers = json.loads(outputs["answers"].read_text(encoding="utf-8"))
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert [row["prompt"] for row in prompts] == [
        "A plane is taking off.",
        "An airplane is taking off.",
    ]
    assert [row["pair_side"] for row in prompts] == ["sentence1", "sentence2"]
    assert len(answers["requests"]) == 2
    assert manifest["pair_count"] == 1
    assert manifest["request_count"] == 2


def test_compare_encoder_embedding_predictions_gates_vector_and_pair_parity() -> None:
    hf = {
        "responses": [
            {"sample_id": "pair-a", "pair_id": "pair", "pair_side": "sentence1", "score": 2.5, "vector": [1.0, 0.0]},
            {"sample_id": "pair-b", "pair_id": "pair", "pair_side": "sentence2", "score": 2.5, "vector": [0.0, 1.0]},
        ]
    }
    trt = {
        "responses": [
            {"sample_id": "pair-a", "pair_id": "pair", "pair_side": "sentence1", "score": 2.5, "vector": [0.999, 0.001]},
            {"sample_id": "pair-b", "pair_id": "pair", "pair_side": "sentence2", "score": 2.5, "vector": [0.001, 0.999]},
        ]
    }

    summary = validation_engine.compare_encoder_embedding_prediction_sets(
        hf,
        trt,
        gates={
            "min_vector_cosine": 0.99,
            "min_vector_pass_rate": 1.0,
            "max_pair_cosine_abs_delta": 0.01,
        },
    )

    assert summary["status"] == "passed"
    assert summary["vector_pass_rate"] == 1.0
    assert summary["min_vector_cosine"] > 0.99
    assert summary["max_pair_cosine_abs_delta"] < 0.01


def test_encoder_reference_uses_dpr_context_classes() -> None:
    assert validation_engine.encoder_reference_class_names("dpr_context_embed") == (
        "DPRContextEncoder",
        "DPRContextEncoderTokenizerFast",
    )
    assert validation_engine.encoder_reference_class_names("encoder_base_features") == (
        "AutoModel",
        "AutoTokenizer",
    )


def test_partiprompts_defaults_to_canonical_models_across_image_families() -> None:
    suite = validation_engine.suite_by_id(
        validation_engine.load_suites(), "dpg_bench_diffusion_image"
    )

    selected = validation_engine.selected_models_for_suite(
        suite, validation_engine.load_manifest_records(), single_device_only=True
    )

    selected_names = [model["name"] for model in selected]
    expected_non_neutral_models = {
        "flux-2-dev-fp8-l0",
        "flux-2-dev-l0",
        "flux-schnell-l0",
        "pixart-sigma-1024",
        "z-image-turbo",
    }
    assert len(selected_names) == 7
    assert expected_non_neutral_models < set(selected_names)


def test_explicit_partiprompts_model_can_select_compatible_non_default() -> None:
    suite = validation_engine.suite_by_id(
        validation_engine.load_suites(), "dpg_bench_diffusion_image"
    )

    selected = validation_engine.selected_models_for_suite(
        suite,
        validation_engine.load_manifest_records(),
        selectors=["flux-2-dev-l0"],
        single_device_only=True,
    )

    assert [model["name"] for model in selected] == ["flux-2-dev-l0"]


def test_partiprompts_uses_model_manifest_generation_and_profile_gates() -> None:
    suite = validation_engine.suite_by_id(
        validation_engine.load_suites(), "dpg_bench_diffusion_image"
    )
    models = {model["name"]: model for model in validation_engine.load_manifest_records()}

    pixart_suite = validation_engine.resolve_suite_for_model(
        suite, models["pixart-sigma-1024"]
    )
    assert pixart_suite["generation"] == {
        "seed": 42,
        "use_shared_initial_latents": True,
        "image_height": 1024,
        "image_width": 1024,
        "video_num_frames": 1,
        "num_inference_steps": 20,
    }
    assert pixart_suite["gates"]["min_trt_hf_image_clip_cosine"] == 0.85
    assert pixart_suite["gates"]["ssim"] == 0.75
    assert pixart_suite["gates"]["psnr"] == 10.0
    assert pixart_suite["gates"]["require_matching_initial_latents"] == 1
    assert "max_prompt_clipscore_drop" not in pixart_suite["gates"]
    assert "min_hf_prompt_clipscore" not in pixart_suite["gates"]

    flux_suite = validation_engine.resolve_suite_for_model(
        suite, models["flux-schnell-l0"]
    )
    assert flux_suite["generation"]["image_height"] == 384
    assert flux_suite["generation"]["image_width"] == 384
    assert flux_suite["generation"]["num_inference_steps"] == 20
    assert flux_suite["gates"]["psnr"] == 5.0
    assert flux_suite["gates"]["ssim"] == 0.1

    non_default_flux_suite = validation_engine.resolve_suite_for_model(
        suite, models["flux-2-dev"]
    )
    assert non_default_flux_suite["generation"]["image_height"] == 1024
    assert non_default_flux_suite["generation"]["num_inference_steps"] == 28
    assert non_default_flux_suite["gates"]["psnr"] == 5.0

    z_image_suite = validation_engine.resolve_suite_for_model(
        suite, models["z-image-turbo"]
    )
    assert z_image_suite["generation"]["image_height"] == 1024
    assert z_image_suite["generation"]["num_inference_steps"] == 9


def test_partiprompts_has_no_family_specific_suite_ids() -> None:
    suite_ids = {suite["id"] for suite in validation_engine.load_suites()}

    assert "dpg_bench_diffusion_image" in suite_ids
    assert "partiprompts_pixart_diffusion_image" not in suite_ids
    assert "partiprompts_flux_diffusion_image" not in suite_ids


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

    suites = validation_engine.load_suites(custom)

    assert [suite["id"] for suite in suites] == ["custom_only"]


def test_plan_selects_chat_text_generation_manifests() -> None:
    suites = validation_engine.load_suites()
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

    rows = validation_engine.build_plan(
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
        "\n".join(
            (
                'test_manifests = ["manifests/example-decoder.json"]',
                "",
                "[model_reference_cache]",
                'repository = "https://example.invalid/reference.git"',
                'revision = "0123456789abcdef0123456789abcdef01234567"',
                'relative_path = "example_decoder/reference/source-0123456789ab"',
                'entrypoint = "reference.py"',
                "",
            )
        ),
        encoding="utf-8",
    )
    (manifest_dir / "example-decoder.json").write_text(
        json.dumps(
            {
                "name": "example-decoder",
                "hf_id": "example-org/example-decoder",
                "hf_revision": "0123456789abcdef",
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

    records = validation_engine.load_manifest_records(tmp_path)

    assert [record["name"] for record in records] == ["example-decoder"]
    assert records[0]["manifest"].endswith("example_decoder/manifests/example-decoder.json")
    assert records[0]["hf_revision"] == "0123456789abcdef"
    assert records[0]["task_eval"] == {
        "vlm_fallback_prompt_template": "<image>{prompt}",
    }
    assert records[0]["model_reference_cache"] == {
        "repository": "https://example.invalid/reference.git",
        "revision": "0123456789abcdef0123456789abcdef01234567",
        "relative_path": "example_decoder/reference/source-0123456789ab",
        "entrypoint": "reference.py",
    }


def test_default_model_names_match_selected_plan_models() -> None:
    suites = validation_engine.load_suites()
    models = validation_engine.load_manifest_records()

    for suite in suites:
        rows = validation_engine.build_plan(suites, models, suite_id=suite["id"])
        selected_names = {row["model"] for row in rows if row["selected"]}

        assert selected_names == set(suite["default_model_names"]), suite["id"]


def test_plan_selects_vlm_mmmu_pro_vision_models() -> None:
    suites = validation_engine.load_suites()
    suite = dict(validation_engine.suite_by_id(suites, "vlm_mmmu_pro_vision_mcq"))
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

    rows = validation_engine.build_plan([suite], models)

    selected = {row["model"]: row for row in rows}
    assert "vl-primary" in selected
    assert selected["vl-primary"]["runtime_strategy"] == "vision_family_vision_language"
    assert "vl-secondary" in selected
    assert "vl-excluded" not in selected
    assert "text-decoder" not in selected


def test_plan_selects_ocrbench_v2_unified_models() -> None:
    suites = validation_engine.load_suites()
    models = validation_engine.load_manifest_records()

    rows = validation_engine.build_plan(suites, models, suite_id="ocrbench_v2_unified")

    selected = {row["model"]: row for row in rows}
    model_by_name = {model["name"]: model for model in models}
    assert set(selected) == {"deepseek-ocr"}
    assert model_by_name["deepseek-ocr"]["reference_backend"] == "hf_transformers"
    assert "qwen25vl-3b" not in selected
    assert "internvl3-2b" not in selected
    assert "locateanything-3b" not in selected


def test_plan_selects_librispeech_asr_models() -> None:
    suites = validation_engine.load_suites()
    models = validation_engine.load_manifest_records()

    rows = validation_engine.build_plan(suites, models, suite_id="librispeech_clean_asr")

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
    suites = validation_engine.load_suites()
    models = validation_engine.load_manifest_records()

    rows = validation_engine.build_plan(suites, models, suite_id="librispeech_clean_asr_streaming")

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
    suite = validation_engine.suite_by_id(validation_engine.load_suites(), "mmlu_five_shot_mcq")

    outputs = validation_engine.prepare_mmlu_dataset(
        dataset_path=dataset,
        work_dir=tmp_path / "work",
        suite=suite,
        limit=1,
    )

    answers = json.loads(outputs["answers"].read_text(encoding="utf-8"))
    prompts = validation_engine.load_jsonl(outputs["prompts"])
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


def test_prepare_text_generation_json_preserves_dataset_sample_id(tmp_path: Path) -> None:
    dataset = tmp_path / "humaneval.json"
    dataset.write_text(
        json.dumps(
            {
                "dataset": "OpenAI HumanEval",
                "requests": [
                    {
                        "id": "HumanEval/0",
                        "prompt": "def add(a, b):\n",
                        "answer": "    return a + b\n",
                        "subject": "python",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    suite = validation_engine.suite_by_id(
        validation_engine.load_suites(), "humaneval_code_continuation_parity"
    )

    outputs = validation_engine.prepare_task_dataset(
        dataset_path=dataset,
        work_dir=tmp_path / "work",
        suite=suite,
        limit=10,
    )

    prompts = validation_engine.load_jsonl(outputs["prompts"])
    answers = json.loads(outputs["answers"].read_text(encoding="utf-8"))
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert answers["requests"][0]["sample_id"] == "HumanEval/0"
    assert prompts[0]["sample_id"] == "HumanEval/0"
    assert prompts[0]["prompt"] == "def add(a, b):\n"
    assert manifest["dataset_kind"] == "text_generation_json"
    assert manifest["limit"] == 10


@pytest.mark.parametrize(
    ("is_encoder_decoder", "expected_model_class"),
    [(False, "causal"), (True, "seq2seq")],
)
def test_load_hf_text_generation_model_selects_configured_auto_class(
    is_encoder_decoder: bool, expected_model_class: str
) -> None:
    calls: list[tuple[str, str, dict]] = []

    class AutoConfig:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls.append(("config", model_id, kwargs))
            return SimpleNamespace(is_encoder_decoder=is_encoder_decoder)

    class Model:
        def __init__(self, kind):
            self.kind = kind

        def eval(self):
            calls.append(("eval", self.kind, {}))
            return self

    def auto_model(kind):
        return SimpleNamespace(
            from_pretrained=lambda model_id, **kwargs: (
                calls.append((kind, model_id, kwargs)) or Model(kind)
            )
        )

    transformers = SimpleNamespace(
        AutoConfig=AutoConfig,
        AutoModelForCausalLM=auto_model("causal"),
        AutoModelForSeq2SeqLM=auto_model("seq2seq"),
    )

    model, detected_seq2seq = validation_engine.load_hf_text_generation_model(
        transformers,
        "example/model",
        model_kwargs={"torch_dtype": "auto"},
        trust_remote_code=True,
        local_files_only=True,
    )

    assert model.kind == expected_model_class
    assert detected_seq2seq is is_encoder_decoder
    assert calls[0] == (
        "config",
        "example/model",
        {"trust_remote_code": True, "local_files_only": True},
    )
    assert calls[1] == (expected_model_class, "example/model", {"torch_dtype": "auto"})


def test_prepare_diffusion_prompts_writes_stable_prompt_rows(tmp_path: Path) -> None:
    dataset = tmp_path / "PartiPrompts.tsv"
    dataset.write_text(
        "Prompt\tCategory\tChallenge\tNote\n"
        "a red cube\tSimple Detail\tBasic\tcolor binding\n"
        "a horse riding an astronaut\tImagination\tComplex\trole reversal\n",
        encoding="utf-8",
    )
    suite = {
        "id": "dpg_bench_diffusion_image",
        "dataset": {"kind": "diffusion_prompt_tsv"},
        "generation": {"seed": 42, "image_height": 384, "image_width": 384},
    }

    outputs = validation_engine.prepare_diffusion_prompt_dataset(
        dataset_path=dataset,
        work_dir=tmp_path / "work",
        suite=suite,
        limit=1,
    )

    answers = json.loads(outputs["answers"].read_text(encoding="utf-8"))
    prompts = validation_engine.load_jsonl(outputs["prompts"])
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))

    assert answers["requests"] == [{
        "sample_id": "partiprompts_000000",
        "dataset_index": 0,
        "prompt": "a red cube",
        "category": "Simple Detail",
        "challenge": "Basic",
        "note": "color binding",
    }]
    assert prompts == [{
        "sample_id": "partiprompts_000000",
        "dataset_index": 0,
        "eval_index": 0,
        "prompt": "a red cube",
        "category": "Simple Detail",
        "challenge": "Basic",
    }]
    assert manifest["dataset_kind"] == "diffusion_prompt_tsv"
    assert manifest["request_count"] == 1
    assert manifest["generation"]["seed"] == 42


def test_prepare_task_dataset_dispatches_diffusion_prompt_json(tmp_path: Path) -> None:
    dataset = tmp_path / "dpg_bench.json"
    dataset.write_text(json.dumps({
        "dataset": "DPG-Bench",
        "version": "test",
        "requests": [{
            "sample_id": "dpg_bench_000001",
            "dataset_index": 1,
            "prompt": "a red cube above a blue sphere",
            "category": "entity,relation",
            "challenge": "dense_prompt_following",
            "questions": [{"question": "Is the red cube above the blue sphere?"}],
        }],
    }), encoding="utf-8")
    suite = validation_engine.suite_by_id(
        validation_engine.load_suites(), "dpg_bench_diffusion_image"
    )

    outputs = validation_engine.prepare_task_dataset(
        dataset_path=dataset,
        work_dir=tmp_path / "work",
        suite=suite,
    )

    prompt = validation_engine.load_jsonl(outputs["prompts"])[0]
    assert prompt["prompt"] == "a red cube above a blue sphere"
    assert prompt["questions"] == [{"question": "Is the red cube above the blue sphere?"}]
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert manifest["dataset_kind"] == "diffusion_prompt_json"
    assert manifest["dataset_name"] == "DPG-Bench"


def test_prepare_diffusion_json_resolves_declared_sample_assets(
    tmp_path: Path,
) -> None:
    condition = tmp_path / "images" / "condition.png"
    condition.parent.mkdir()
    condition.write_bytes(b"condition")
    dataset = tmp_path / "gedit.json"
    dataset.write_text(
        json.dumps(
            {
                "dataset": "GEdit-Bench",
                "requests": [
                    {
                        "sample_id": "gedit_000000",
                        "prompt": "turn the object blue",
                        "image": "images/condition.png",
                        "category": "color",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    suite = {
        "id": "gedit_bench_image_edit",
        "dataset": {
            "kind": "diffusion_prompt_json",
            "asset_fields": ["image"],
        },
    }

    outputs = validation_engine.prepare_diffusion_prompt_json_dataset(
        dataset_path=dataset,
        work_dir=tmp_path / "work",
        suite=suite,
    )

    answers = json.loads(outputs["answers"].read_text(encoding="utf-8"))
    prompts = validation_engine.load_jsonl(outputs["prompts"])
    assert answers["requests"][0]["image"] == str(condition.resolve())
    assert prompts[0]["image"] == str(condition.resolve())


def test_prepare_model_plugin_dataset_resolves_nested_input_assets(
    tmp_path: Path,
) -> None:
    image = tmp_path / "images" / "input.jpg"
    image.parent.mkdir()
    image.write_bytes(b"image")
    dataset = tmp_path / "model_plugin.json"
    dataset.write_text(
        json.dumps(
            {
                "dataset": "fixed plugin inputs",
                "requests": [
                    {
                        "sample_id": "sample-1",
                        "testcase": "model-case",
                        "stage": "full_generation",
                        "inputs": {
                            "prompt": "describe",
                            "image": "images/input.jpg",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    suite = {
        "id": "model_plugin_suite",
        "dataset": {
            "kind": "model_plugin_json",
            "input_asset_fields": ["image"],
        },
    }

    outputs = validation_engine.prepare_model_plugin_dataset(
        dataset_path=dataset,
        work_dir=tmp_path / "work",
        suite=suite,
    )

    prompts = validation_engine.load_jsonl(outputs["prompts"])
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert prompts[0]["inputs"]["image"] == str(image.resolve())
    assert prompts[0]["image"] == str(image.resolve())
    assert prompts[0]["eval_index"] == 0
    assert manifest["dataset_kind"] == "model_plugin_json"
    assert manifest["request_count"] == 1


def test_compare_model_plugin_prediction_sets_uses_model_comparator(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from tests.e2e_harness.contracts import (
        CompareResult,
        MetricResult,
        StageOutput,
        StageSpec,
        StageStatus,
    )
    from tools.validation.model_plugin_contract import serialize_stage_output

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_kind": "model_plugin_json",
                "task_eval": {"model_manifest": "model.json"},
            }
        ),
        encoding="utf-8",
    )
    case = SimpleNamespace(
        task_strategy="custom_strategy",
        family="custom_family",
        comparison_profile="default",
        threshold_overrides={"score": 0.9},
        metadata={
            "model_test_dir": "tests/e2e/models/custom",
            "validation_manifest_case_name": "custom-case",
        },
    )
    stage = StageSpec(name="full_generation", required=True)
    monkeypatch.setattr(
        validation_engine,
        "select_case",
        lambda *_args, **_kwargs: (case, stage),
    )
    monkeypatch.setattr(
        validation_engine,
        "activate_model_plugins",
        lambda _path: None,
    )

    class Comparator:
        def compare(self, trt, ref, threshold, selected_stage):
            assert trt.text == ref.text == "same"
            assert threshold.metrics["score"] == 0.9
            assert selected_stage.name == "full_generation"
            return CompareResult(
                stage_name=selected_stage.name,
                status=StageStatus.PASSED.value,
                metrics={
                    "score": MetricResult(
                        value=1.0,
                        threshold=0.9,
                        operator=">=",
                        passed=True,
                    )
                },
            )

    monkeypatch.setattr(
        validation_engine,
        "get_comparator",
        lambda _strategy: Comparator(),
    )
    output = StageOutput(stage_name="full_generation", text="same")
    serialized = serialize_stage_output(
        output,
        artifact_dir=tmp_path / "artifacts",
        sample_id="sample-1",
    )
    response = {
        "sample_id": "sample-1",
        "testcase": "custom-case",
        "stage": "full_generation",
        "stage_output": serialized,
    }

    summary = validation_engine.compare_model_plugin_prediction_sets(
        {"responses": [response]},
        {"responses": [response]},
        {
            "requests": [
                {
                    "sample_id": "sample-1",
                    "testcase": "custom-case",
                    "stage": "full_generation",
                    "inputs": {},
                }
            ]
        },
        work_dir=work_dir,
        gates={"min_sample_pass_rate": 1.0},
    )

    assert summary["status"] == "passed"
    assert summary["sample_pass_rate"] == 1.0
    assert summary["metrics"]["score"]["mean"] == 1.0
    assert summary["cases"][0]["passed"] is True


def test_compare_model_plugin_marks_native_returncode_as_execution_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from tests.e2e_harness.contracts import StageOutput, StageSpec
    from tools.validation.model_plugin_contract import serialize_stage_output

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_kind": "model_plugin_json",
                "task_eval": {"model_manifest": "model.json"},
            }
        ),
        encoding="utf-8",
    )
    case = SimpleNamespace(
        task_strategy="custom_strategy",
        family="custom_family",
        comparison_profile="default",
        threshold_overrides={},
        metadata={
            "model_test_dir": "tests/e2e/models/custom",
            "validation_manifest_case_name": "custom-case",
        },
    )
    monkeypatch.setattr(
        validation_engine,
        "select_case",
        lambda *_args, **_kwargs: (
            case,
            StageSpec(name="full_generation", required=True),
        ),
    )
    monkeypatch.setattr(
        validation_engine,
        "activate_model_plugins",
        lambda _path: None,
    )
    monkeypatch.setattr(
        validation_engine,
        "get_comparator",
        lambda _strategy: SimpleNamespace(
            compare=lambda *_args, **_kwargs: pytest.fail(
                "comparator must not run after backend failure"
            )
        ),
    )

    hf_output = serialize_stage_output(
        StageOutput(stage_name="full_generation", text="A"),
        artifact_dir=tmp_path / "hf",
        sample_id="sample-1",
    )
    trt_output = serialize_stage_output(
        StageOutput(
            stage_name="full_generation",
            metadata={"returncode": 1, "stderr": "preprocessing failed"},
        ),
        artifact_dir=tmp_path / "trt",
        sample_id="sample-1",
    )
    request = {
        "sample_id": "sample-1",
        "testcase": "custom-case",
        "stage": "full_generation",
        "inputs": {},
    }

    summary = validation_engine.compare_model_plugin_prediction_sets(
        {
            "responses": [
                {
                    **request,
                    "stage_output": hf_output,
                }
            ]
        },
        {
            "responses": [
                {
                    **request,
                    "stage_output": trt_output,
                }
            ]
        },
        {"requests": [request]},
        work_dir=work_dir,
        gates={"min_sample_pass_rate": 1.0},
    )

    assert summary["status"] == "failed"
    assert summary["valid_count"] == 0
    assert summary["skipped_count"] == 1
    assert summary["cases"][0]["status"] == "error"
    assert summary["execution_errors"][0]["failures"][0] == {
        "backend": "trtmc",
        "returncode": 1,
        "stderr": "preprocessing failed",
    }


def test_prepare_mmlu_applies_gpt_oss_family_override(tmp_path: Path) -> None:
    dataset = tmp_path / "mmlu.json"
    _write_mmlu(dataset)
    suite = validation_engine.suite_by_id(validation_engine.load_suites(), "mmlu_five_shot_mcq")
    model = {"name": "gpt-oss-20b", "family": "gpt_oss", "task_eval": {}}
    config = validation_engine.effective_validation_config(suite, model)

    outputs = validation_engine.prepare_mmlu_dataset(
        dataset_path=dataset,
        work_dir=tmp_path / "work",
        suite=suite,
        limit=1,
        validation_config=config,
    )

    prompt = validation_engine.load_jsonl(outputs["prompts"])[0]["prompt"]
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert prompt == (
        "<|start|>system<|message|>You are a helpful assistant. "
        "Answer with only the option letter.<|end|>"
        "<|start|>user<|message|>Question one\nA. a\nB. b\nAnswer:<|end|>"
        "<|start|>assistant<|channel|>final<|message|>"
    )
    assert manifest["generation"]["max_new_tokens"] == 8
    assert manifest["generation"]["apply_chat_template"] is False
    assert manifest["task_eval"]["answer_parser"] == "gpt_oss_harmony_final_mcq"


def test_non_gpt_oss_mmlu_model_keeps_suite_defaults() -> None:
    suite = validation_engine.suite_by_id(validation_engine.load_suites(), "mmlu_five_shot_mcq")
    model = {"name": "tinyllama-1.1b", "family": "llama", "task_eval": {}}

    assert validation_engine.effective_validation_config(suite, model) == {}


@pytest.mark.parametrize(
    ("model_name", "prompt_token_limit"),
    [
        ("bart-base", 958),
        ("gpt2-125m", 960),
        ("gpt-neo-125m", 1984),
        ("opt-125m", 1984),
    ],
)
def test_continuation_suite_limits_prompts_to_model_context(
    model_name: str, prompt_token_limit: int
) -> None:
    suite = validation_engine.suite_by_id(validation_engine.load_suites(), "mmlu_continuation_parity")
    config = validation_engine.effective_validation_config(
        suite,
        {"name": model_name, "family": "", "task_eval": {}},
    )
    assert config["prompt_token_limit"] == prompt_token_limit
    assert config["prompt_truncation_side"] == "left"


@pytest.mark.parametrize(
    "model_name", ["falcon-rw-1b", "granite-3.1-2b", "olmo2-1b"]
)
def test_continuation_suite_uses_aligned_fp32_comparison_for_sensitive_models(
    model_name: str,
) -> None:
    suite = validation_engine.suite_by_id(
        validation_engine.load_suites(),
        "mmlu_continuation_parity",
    )
    config = validation_engine.effective_validation_config(
        suite,
        {"name": model_name, "family": "", "task_eval": {}},
    )

    assert config["comparison_precision"] == "fp32"


def test_mmlu_suite_disables_hf_cache_for_internlm() -> None:
    suite = validation_engine.suite_by_id(validation_engine.load_suites(), "mmlu_five_shot_mcq")
    config = validation_engine.effective_validation_config(
        suite,
        {"name": "internlm2-1.8b", "family": "internlm", "task_eval": {}},
    )
    assert config["hf_use_cache"] is False


def test_truncate_prompt_rows_preserves_suffix_and_records_provenance() -> None:
    class Tokenizer:
        def __call__(self, text, *, add_special_tokens=False):
            assert add_special_tokens is False
            return argparse.Namespace(input_ids=text.split())

        def decode(self, token_ids, **kwargs):
            assert kwargs == {
                "skip_special_tokens": False,
                "clean_up_tokenization_spaces": False,
            }
            return " ".join(token_ids)

    rows = [
        {"sample_id": "long", "prompt": "one two three four five"},
        {"sample_id": "short", "prompt": "six seven"},
    ]
    summary = validation_engine.truncate_prompt_rows(
        rows,
        tokenizer=Tokenizer(),
        token_limit=3,
        truncation_side="left",
    )
    assert rows == [
        {
            "sample_id": "long",
            "prompt": "three four five",
            "prompt_tokens_before": 5,
            "prompt_tokens_after": 3,
            "prompt_truncated": True,
        },
        {
            "sample_id": "short",
            "prompt": "six seven",
            "prompt_tokens_before": 2,
            "prompt_tokens_after": 2,
            "prompt_truncated": False,
        },
    ]
    assert summary == {
        "token_limit": 3,
        "truncation_side": "left",
        "prompt_count": 2,
        "truncated_count": 1,
        "max_tokens_before": 5,
        "max_tokens_after": 3,
    }


def test_hf_generation_overrides_reads_explicit_cache_setting(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "manifest.json").write_text(
        json.dumps({"task_eval": {"hf_use_cache": False}}),
        encoding="utf-8",
    )
    assert validation_engine.hf_generation_overrides(work_dir) == {"use_cache": False}


def test_model_reference_python_resolves_family_profile(monkeypatch) -> None:
    calls: list[tuple[object, ...]] = []

    def fake_normalize(raw, **kwargs):
        calls.append(("normalize", raw, kwargs))
        return {"build": "base", "runtime": "base", "reference": "deepseek_ocr"}

    def fake_resolve(profile, base_python):
        calls.append(("resolve", profile, base_python))
        return "/profiles/deepseek-ocr/bin/python"

    monkeypatch.setattr(validation_engine, "normalize_execution_profiles", fake_normalize)
    monkeypatch.setattr(validation_engine, "resolve_profile_python", fake_resolve)

    resolved = validation_engine.model_reference_python(
        {
            "family": "deepseek_ocr",
            "runtime_strategy": "deepseek_ocr_vision_language",
            "reference_backend": "hf_transformers",
            "execution_profiles": {"reference": "deepseek_ocr"},
        },
        "/opt/venv/bin/python3",
    )

    assert resolved == "/profiles/deepseek-ocr/bin/python"
    assert calls == [
        (
            "normalize",
            {"reference": "deepseek_ocr"},
            {
                "family": "deepseek_ocr",
                "runtime_strategy": "deepseek_ocr_vision_language",
                "reference_backend": "hf_transformers",
            },
        ),
        ("resolve", "deepseek_ocr", "/opt/venv/bin/python3"),
    ]


def test_prepare_seedtts_writes_resolved_audio_and_scoring_contract(tmp_path: Path) -> None:
    dataset = tmp_path / "SeedTTS_en_meta" / "seedtts_en_meta.json"
    dataset.parent.mkdir()
    _write_seedtts(dataset)
    suite = validation_engine.suite_by_id(validation_engine.load_suites(), "seedtts_en_tts_intelligibility")

    outputs = validation_engine.prepare_seedtts_dataset(
        dataset_path=dataset,
        work_dir=tmp_path / "work",
        suite=suite,
        limit=1,
    )

    answers = json.loads(outputs["answers"].read_text(encoding="utf-8"))
    prompts = validation_engine.load_jsonl(outputs["prompts"])
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
    suite = validation_engine.suite_by_id(validation_engine.load_suites(), "vlm_mmmu_pro_vision_mcq")

    outputs = validation_engine.prepare_vlm_chat_dataset(
        dataset_path=dataset,
        work_dir=tmp_path / "work",
        suite=suite,
        limit=1,
    )

    answers = json.loads(outputs["answers"].read_text(encoding="utf-8"))
    prompts = validation_engine.load_jsonl(outputs["prompts"])
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
    suite = validation_engine.suite_by_id(validation_engine.load_suites(), "ocrbench_v2_unified")

    outputs = validation_engine.prepare_vlm_unified_dataset(
        dataset_path=dataset,
        work_dir=tmp_path / "work",
        suite=suite,
        limit=1,
    )

    answers = json.loads(outputs["answers"].read_text(encoding="utf-8"))
    prompts = validation_engine.load_jsonl(outputs["prompts"])
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
    suite = validation_engine.suite_by_id(validation_engine.load_suites(), "ocrbench_v2_unified")

    try:
        validation_engine.prepare_vlm_unified_dataset(
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
    suite = validation_engine.suite_by_id(validation_engine.load_suites(), "librispeech_clean_asr")

    outputs = validation_engine.prepare_asr_chat_dataset(
        dataset_path=dataset,
        work_dir=tmp_path / "work",
        suite=suite,
        limit=1,
    )

    answers = json.loads(outputs["answers"].read_text(encoding="utf-8"))
    prompts = validation_engine.load_jsonl(outputs["prompts"])
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
    suite = validation_engine.suite_by_id(validation_engine.load_suites(), "librispeech_clean_asr")

    try:
        validation_engine.prepare_asr_chat_dataset(
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

    monkeypatch.setattr(validation_engine, "_resize_image_to_square", fake_resize)
    suite = validation_engine.suite_by_id(validation_engine.load_suites(), "vlm_mmmu_pro_vision_fixed_mcq")

    outputs = validation_engine.prepare_vlm_chat_dataset(
        dataset_path=dataset,
        work_dir=tmp_path / "work",
        suite=suite,
        limit=1,
    )

    answers = json.loads(outputs["answers"].read_text(encoding="utf-8"))
    prompts = validation_engine.load_jsonl(outputs["prompts"])
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

    rendered = validation_engine._vlm_chat_text(
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

    rendered = validation_engine._vlm_chat_text(
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

    rc = validation_engine.cmd_prepare(
        argparse.Namespace(
            suites=str(validation_engine.DEFAULT_SUITES),
            suite="vlm_mmmu_pro_vision_mcq",
            dataset=str(dataset),
            work_dir=str(work_dir),
            limit=1,
            subject="",
            sample_seed=None,
        )
    )

    assert rc == 0
    assert validation_engine.load_jsonl(work_dir / "prompts.jsonl")[0]["images"] == [
        str(dataset_dir / "images" / "sample.jpg")
    ]


def test_continuation_parity_reports_divergence_severity() -> None:
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

    summary = validation_engine.compare_continuation_sets(hf, trtfb, tokenize=lambda s: s.split())

    assert summary["count"] == 2
    assert summary["divergence_metric_scope"] == "divergent_samples_only"
    assert (
        summary["normalization_denominator"]
        == "max_hf_trtfb_generated_length"
    )
    assert summary["exact_match_rate"] == 0.5  # "a" exact, "b" not
    assert summary["samples"][0]["first_divergence"] == 3  # all 3 tokens match
    assert summary["samples"][1]["first_divergence"] == 1  # diverge at token index 1
    # matched prefixes 3 + 1 = 4, ref token counts 3 + 2 = 5
    assert abs(summary["token_prefix_agreement"] - 4 / 5) < 1e-9
    assert summary["divergent_count"] == 1
    assert summary["divergence_rate"] == 0.5
    assert summary["mean_divergent_first_divergence"] == 1.0
    assert summary["mean_divergent_prefix_ratio"] == 0.5
    assert summary["min_divergent_prefix_ratio"] == 0.5
    assert summary["mean_divergent_severity"] == 0.5
    assert summary["max_divergent_severity"] == 0.5
    assert summary["samples"][0]["diverged"] is False
    assert summary["samples"][0]["normalized_first_divergence"] == 1.0
    assert summary["samples"][0]["divergence_severity"] == 0.0
    assert summary["samples"][1]["diverged"] is True
    assert summary["samples"][1]["normalized_first_divergence"] == 0.5
    assert summary["samples"][1]["divergence_severity"] == 0.5


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

    summary = validation_engine.compare_continuation_sets(hf, trtfb, require_token_ids=True)

    assert summary["comparison_granularity"] == "generated_token_ids"
    assert summary["exact_match_rate"] == 0.5
    assert summary["token_id_exact_match_rate"] == 0.5
    assert summary["text_exact_match_rate"] == 1.0
    assert summary["samples"][1]["first_divergence"] == 2
    assert summary["samples"][1]["hf_token_at_divergence"] == 3
    assert summary["samples"][1]["trtfb_token_at_divergence"] == 4
    assert summary["samples"][1]["normalized_first_divergence"] == 2 / 3
    assert summary["samples"][1]["divergence_severity"] == 1 / 3
    assert summary["mean_divergent_prefix_ratio"] == 2 / 3
    assert summary["mean_divergent_severity"] == 1 / 3


def test_continuation_parity_reports_no_divergence_without_empty_means() -> None:
    predictions = {
        "responses": [
            {"sample_id": "a", "output_text": "same", "generated_token_ids": [1, 2]},
            {"sample_id": "b", "output_text": "", "generated_token_ids": []},
        ]
    }

    summary = validation_engine.compare_continuation_sets(
        predictions, predictions, require_token_ids=True
    )

    assert summary["divergent_count"] == 0
    assert summary["divergence_rate"] == 0.0
    assert summary["mean_divergent_first_divergence"] is None
    assert summary["mean_divergent_prefix_ratio"] is None
    assert summary["min_divergent_prefix_ratio"] is None
    assert summary["mean_divergent_severity"] == 0.0
    assert summary["max_divergent_severity"] == 0.0
    assert summary["samples"][1]["normalized_first_divergence"] == 1.0
    assert summary["samples"][1]["divergence_severity"] == 0.0


def test_continuation_parity_requires_token_ids_when_requested() -> None:
    hf = {"responses": [{"sample_id": "a", "output_text": "x"}]}
    trtfb = {"responses": [{"sample_id": "a", "output_text": "x"}]}

    try:
        validation_engine.compare_continuation_sets(hf, trtfb, require_token_ids=True)
    except ValueError as exc:
        assert "generated_token_ids" in str(exc)
    else:
        raise AssertionError("expected missing token-id validation failure")


def test_continuation_suite_accepts_one_divergent_sample_in_ten() -> None:
    hf = {
        "responses": [
            {
                "sample_id": f"sample-{index}",
                "output_text": "same",
                "generated_token_ids": [1, 2],
            }
            for index in range(10)
        ]
    }
    trtfb = json.loads(json.dumps(hf))
    trtfb["responses"][-1]["output_text"] = "different"
    trtfb["responses"][-1]["generated_token_ids"] = [9, 9]
    summary = validation_engine.compare_continuation_sets(
        hf, trtfb, require_token_ids=True
    )
    result = {
        "exact_match_rate": summary["exact_match_rate"],
        "token_prefix_agreement": summary["token_prefix_agreement"],
    }
    suite = validation_engine.suite_by_id(
        validation_engine.load_suites(), "mmlu_continuation_parity"
    )

    validation_engine.apply_metric_gates(result, suite["gates"])

    assert result["exact_match_rate"] == 0.9
    assert result["token_prefix_agreement"] == 0.9
    assert result["status"] == "passed"


def test_validation_suites_keep_continuation_and_drop_trace_cloze() -> None:
    suites = validation_engine.load_suites()
    ids = {suite["id"] for suite in suites}
    continuation = validation_engine.suite_by_id(suites, "mmlu_continuation_parity")

    assert "mmlu_continuation_parity" in ids
    assert "mmlu_trace_cloze" not in ids
    assert continuation["dataset"]["kind"] == "mmlu_five_shot_json"
    assert continuation["scoring"]["scorer"] == "continuation"
    assert continuation["user_contract"] == "continuation_parity"
    assert continuation["gates"] == {"min_exact_match_rate": 0.9}


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

    rc = validation_engine.cmd_compare_continuation(
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
    assert summary["divergence_rate"] == 0.5
    assert summary["mean_divergent_first_divergence"] == 1.0
    assert summary["mean_divergent_prefix_ratio"] == 0.5
    assert summary["mean_divergent_severity"] == 0.5
    assert summary["samples"][1]["first_divergence"] == 1


def test_continuation_summary_markdown_prioritizes_divergence_severity(
    tmp_path: Path,
) -> None:
    hf = {
        "responses": [
            {
                "sample_id": "exact",
                "output_text": "same",
                "generated_token_ids": [1, 2],
            },
            {
                "sample_id": "diverged",
                "output_text": "left",
                "generated_token_ids": [3, 4],
            },
        ]
    }
    trtfb = {
        "responses": [
            {
                "sample_id": "exact",
                "output_text": "same",
                "generated_token_ids": [1, 2],
            },
            {
                "sample_id": "diverged",
                "output_text": "right",
                "generated_token_ids": [3, 5],
            },
        ]
    }
    summary = validation_engine.compare_continuation_sets(hf, trtfb)
    output = tmp_path / "summary.md"

    validation_engine.write_continuation_summary_markdown(summary, output)

    markdown = output.read_text(encoding="utf-8")
    assert markdown.startswith("# Continuation Divergence Summary\n")
    assert "| divergence_metric_scope | divergent_samples_only |" in markdown
    assert (
        "| normalization_denominator | max_hf_trtfb_generated_length |"
        in markdown
    )
    assert "| divergence_rate | 0.5000 |" in markdown
    assert "| mean_divergent_prefix_ratio | 0.5000 |" in markdown
    assert "| mean_divergent_severity | 0.5000 |" in markdown
    assert "## Compatibility Diagnostics" in markdown
    assert "| diverged | 1 | 2 | 2 | 0.5000 | 0.5000 | 4 | 5 |" in markdown
    assert "| exact |" not in markdown


def test_continuation_result_line_prioritizes_divergence_severity() -> None:
    line = validation_engine._format_result_line(
        {"name": "example"},
        {
            "mode": "continuation",
            "comparison_granularity": "generated_token_ids",
            "divergent_count": 2,
            "divergence_rate": 0.2,
            "mean_divergent_first_divergence": 8.0,
            "mean_divergent_prefix_ratio": 0.125,
            "min_divergent_prefix_ratio": 0.109375,
            "mean_divergent_severity": 0.875,
            "max_divergent_severity": 0.890625,
            "hf_reused": True,
            "bundle_built": False,
        },
    )

    assert "divergent_count=2" in line
    assert "divergence_rate=0.2000" in line
    assert "mean_divergent_prefix_ratio=0.1250" in line
    assert "min_divergent_prefix_ratio=0.1094" in line
    assert "mean_divergent_severity=0.8750" in line
    assert "max_divergent_severity=0.8906" in line
    assert "exact=" not in line
    assert "token_agreement=" not in line


def test_dataset_benchmark_serializes_generated_token_ids() -> None:
    source = (
        validation_engine.REPO_ROOT / "examples" / "trtmc_dataset_benchmark.cpp"
    ).read_text(encoding="utf-8")

    assert '\\"generated_token_ids\\":[' in source


def test_dataset_benchmark_accepts_model_plugin_directory() -> None:
    source = (
        validation_engine.REPO_ROOT / "examples" / "trtmc_dataset_benchmark.cpp"
    ).read_text(encoding="utf-8")

    assert 'arg == "--model-plugin-dir"' in source
    assert "load_options.model_plugin_search_paths.emplace_back" in source
    assert "result.token_ids[token_idx]" in source


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

    validation_engine.convert_trtfb_jsonl_to_predictions(raw, predictions)

    payload = json.loads(predictions.read_text(encoding="utf-8"))
    assert payload["responses"][0]["output_text"] == "Answer: B"
    assert payload["responses"][0]["generated_token_ids"] == [42]
    assert payload["responses"][0]["source"] == "trtfb"


def test_convert_trtfb_replaces_invalid_utf8_in_generated_text(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "trtfb_raw.jsonl"
    raw.write_bytes(
        b'{"sample_id":"sample-0","text":"bad \x88 text",'
        b'"generated_token_ids":[7]}\n'
    )
    predictions = tmp_path / "predictions.json"

    validation_engine.convert_trtfb_jsonl_to_predictions(raw, predictions)

    payload = json.loads(predictions.read_text(encoding="utf-8"))
    assert payload["responses"][0]["output_text"] == "bad \ufffd text"
    assert payload["responses"][0]["generated_token_ids"] == [7]


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

    hf_score = validation_engine.score_predictions(hf, answers)
    summary = validation_engine.compare_prediction_sets(hf, trtfb, answers)

    assert hf_score["overall_accuracy"] == 1.0
    assert summary["hf"]["overall_accuracy"] == 1.0
    assert summary["trtfb"]["overall_accuracy"] == 0.5
    assert summary["accuracy_delta_trtfb_minus_hf"] == -0.5
    assert summary["prediction_agreement_rate"] == 0.5
    assert summary["buckets"]["hf_correct_trtfb_wrong"] == 1


def test_mcq_exact_reference_logit_ties_are_equivalent() -> None:
    answers = {"requests": [{"answer": "B", "subject": "test"}]}
    hf = {
        "responses": [
            {
                "sample_id": "tie",
                "output_text": "B",
                "generated_token_ids": [365],
                "generated_token_max_score_ids": [[365, 384]],
            }
        ]
    }
    trtfb = {
        "responses": [
            {
                "sample_id": "tie",
                "output_text": "D",
                "generated_token_ids": [384],
            }
        ]
    }

    summary = validation_engine.compare_prediction_sets(
        hf, trtfb, answers, scorer="mcq"
    )
    gate = validation_engine.prediction_agreement_gate_result(
        summary,
        {
            "max_accuracy_drop_from_hf": 0.01,
            "min_prediction_agreement": 0.98,
        },
    )

    assert summary["hf"]["overall_accuracy"] == 1.0
    assert summary["trtfb"]["overall_accuracy"] == 0.0
    assert summary["prediction_agreement_rate"] == 1.0
    assert summary["reference_tie_equivalent_count"] == 1
    assert summary["tie_adjusted_accuracy_delta_trtfb_minus_hf"] == 0.0
    assert gate["status"] == "passed"
    assert gate["accuracy_drop_from_hf"] == 0.0


def test_gpt_oss_harmony_parser_rejects_control_only_predictions() -> None:
    parser = "gpt_oss_harmony_final_mcq"

    assert validation_engine.parse_model_prediction(
        "<|channel|>final<|message|> B", answer_parser=parser
    ) == "B"
    assert validation_engine.parse_model_prediction(" B", answer_parser=parser) == "B"
    assert validation_engine.parse_model_prediction("<|channel|>", answer_parser=parser) == ""


def test_required_valid_prediction_does_not_agree_on_empty_outputs() -> None:
    answers = {"requests": [{"answer": "A", "subject": "test"}]}
    hf = {"responses": [{"sample_id": "one", "output_text": "<|channel|>"}]}
    trtfb = {"responses": [{"sample_id": "one", "output_text": "\n\n"}]}

    summary = validation_engine.compare_prediction_sets(
        hf,
        trtfb,
        answers,
        answer_parser="gpt_oss_harmony_final_mcq",
        require_valid_prediction=True,
    )

    assert summary["prediction_agreement_rate"] == 0.0
    assert summary["hf"]["valid_prediction_count"] == 0
    assert summary["trtfb"]["valid_prediction_count"] == 0
    assert summary["disagreements"][0]["hf_prediction"] == ""
    assert summary["disagreements"][0]["trtfb_prediction"] == ""


def test_score_predictions_parses_vlm_a_to_j_choices() -> None:
    answers = {"requests": [{"answer": "J", "subject": "History"}]}
    predictions = {"responses": [{"sample_id": "test_case_1", "output_text": "Answer: J"}]}

    score = validation_engine.score_predictions(predictions, answers)

    assert score["overall_accuracy"] == 1.0
    assert score["samples"][0]["parsed_prediction"] == "J"


def test_score_predictions_accepts_answer_aliases() -> None:
    answers = {"requests": [{"answer": "enabled", "answer_aliases": ["enabled", "on"]}]}
    predictions = {"responses": [{"sample_id": "ocrbench_v2_000000", "output_text": "on"}]}

    score = validation_engine.score_predictions(predictions, answers)

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

    score = validation_engine.score_predictions(predictions, answers, scorer="tts_intelligibility")

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

    score = validation_engine.score_predictions(predictions, answers, scorer="tts_intelligibility")

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

    summary = validation_engine.compare_prediction_sets(hf, trtfb, answers, scorer="tts_intelligibility")

    assert summary["disagreements"][0]["hf_prediction"] == (
        "i m never more aware of a room s acoustics"
    )
    assert summary["disagreements"][0]["trtfb_prediction"] == (
        "i am never more aware of other rooms"
    )


def test_tts_intelligibility_gate_rejects_reference_pass_rate_drop() -> None:
    result = validation_engine.tts_intelligibility_gate_result(
        {
            "hf": {"overall_accuracy": 1.0},
            "trtfb": {"overall_accuracy": 1.0 / 3.0},
            "correctness_agreement_rate": 1.0 / 3.0,
        },
        {
            "max_pass_rate_drop_from_hf": 0.05,
            "min_correctness_agreement": 0.95,
        },
    )

    assert result["status"] == "failed"
    assert result["pass_rate_drop_from_hf"] == pytest.approx(2.0 / 3.0)
    assert result["correctness_agreement_rate"] == pytest.approx(1.0 / 3.0)


def test_tts_intelligibility_gate_accepts_matching_correctness() -> None:
    result = validation_engine.tts_intelligibility_gate_result(
        {
            "hf": {"overall_accuracy": 1.0},
            "trtfb": {"overall_accuracy": 1.0},
            "correctness_agreement_rate": 1.0,
        },
        {
            "max_pass_rate_drop_from_hf": 0.05,
            "min_correctness_agreement": 0.95,
        },
    )

    assert result["status"] == "passed"


def test_run_tts_trtfb_generates_audio_and_batches_asr(tmp_path: Path, monkeypatch) -> None:
    dataset = tmp_path / "SeedTTS_en_meta" / "seedtts_en_meta.json"
    dataset.parent.mkdir()
    _write_seedtts(dataset)
    suite = validation_engine.suite_by_id(validation_engine.load_suites(), "seedtts_en_tts_intelligibility")
    work_dir = tmp_path / "work"
    validation_engine.prepare_seedtts_dataset(
        dataset_path=dataset,
        work_dir=work_dir,
        suite=suite,
        limit=1,
        validation_config={
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

    monkeypatch.setattr(validation_engine.subprocess, "run", fake_run)
    monkeypatch.setattr(
        validation_engine,
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

    validation_engine.run_tts_trtfb(args)

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

    score = validation_engine.score_predictions(predictions, answers, scorer="ocrbench_v2")

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

    score = validation_engine.score_predictions(predictions, answers, scorer="ocrbench_v2")

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

    score = validation_engine.score_predictions(predictions, answers, scorer="ocrbench_v2")

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

    score = validation_engine.score_predictions(predictions, answers, scorer="ocrbench_v2")

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

    summary = validation_engine.compare_prediction_sets(hf, trtfb, answers, scorer="ocrbench_v2")

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

    score = validation_engine.score_predictions(predictions, answers, scorer="asr_transcript")

    assert score["overall_accuracy"] == 0.5
    assert score["exact_match_rate"] == 0.5
    assert score["correct"] == 1
    assert score["samples"][0]["normalized_answer"] == "hello world"
    assert score["samples"][0]["exact_match"] is True
    assert score["samples"][1]["word_error_rate"] == 0.25
    assert score["samples"][1]["correct"] is False


def test_asr_transcript_scorer_strips_nemotron_language_tag() -> None:
    answers = {"requests": [{"answer": "The examination resulted in no discovery"}]}
    predictions = {
        "responses": [
            {
                "sample_id": "tagged",
                "output_text": "The examination resulted in no discovery. <en-US>",
            }
        ]
    }

    score = validation_engine.score_predictions(predictions, answers, scorer="asr_transcript")

    assert score["overall_accuracy"] == 1.0
    assert score["samples"][0]["normalized_prediction"] == (
        "the examination resulted in no discovery"
    )


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
            {"sample_id": "b", "output_text": validation_engine.ERROR_OUTPUT_TEXT},
        ]
    }

    score = validation_engine.score_predictions(predictions, answers, scorer="asr_transcript")

    assert score["overall_accuracy"] == 0.0
    assert score["valid_count"] == 1
    assert score["skipped_count"] == 1
    assert score["samples"][0]["word_error_rate"] == 1.0
    assert score["samples"][1]["skipped"] is True


def test_asr_transcript_agreement_uses_direct_transcript_similarity() -> None:
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

    summary = validation_engine.compare_prediction_sets(hf, trtfb, answers, scorer="asr_transcript")

    expected_similarity = (
        1.0
        + 1.0
        - validation_engine._normalized_edit_distance("gamma delta", "totally wrong")
    ) / 2.0
    assert summary["prediction_agreement_rate"] == pytest.approx(expected_similarity)
    assert summary["normalized_transcript_exact_agreement_rate"] == 0.5
    assert summary["correctness_agreement_rate"] == 0.5
    assert summary["agreement_count"] == 1
    assert summary["buckets"]["both_correct"] == 1
    assert summary["buckets"]["hf_correct_trtfb_wrong"] == 1
    assert summary["disagreements"][0]["hf_prediction"] == "gamma delta"
    assert summary["disagreements"][0]["trtfb_prediction"] == "totally wrong"
    assert summary["asr_parity_samples"][0]["transcript_similarity"] == 1.0


def test_selected_models_for_suite_accepts_manifest_name() -> None:
    suite = validation_engine.suite_by_id(validation_engine.load_suites(), "mmlu_five_shot_mcq")
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

    selected = validation_engine.selected_models_for_suite(
        suite,
        models,
        selectors=["decoder-chat"],
        single_device_only=True,
    )

    assert [model["name"] for model in selected] == ["decoder-chat"]


def test_seedtts_default_selection_uses_canonical_single_device_models() -> None:
    suite = validation_engine.suite_by_id(validation_engine.load_suites(), "seedtts_en_tts_intelligibility")

    selected = validation_engine.selected_models_for_suite(
        suite,
        validation_engine.load_manifest_records(),
        single_device_only=True,
    )

    assert {model["name"] for model in selected} == {
        "bark-large",
        "bark-small",
        "magpie-tts-357m",
    }


def test_seedtts_plan_marks_only_default_models_selected() -> None:
    suite = validation_engine.suite_by_id(validation_engine.load_suites(), "seedtts_en_tts_intelligibility")
    rows = validation_engine.build_plan(
        [suite],
        validation_engine.load_manifest_records(),
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
    waives = validation_engine.load_waives(waives_path)

    selected = validation_engine.selected_models_for_suite(suite, models, waives=waives)
    explicit = validation_engine.selected_models_for_suite(
        suite,
        models,
        selectors=["decoder-waived"],
        waives=waives,
    )
    rows = validation_engine.build_plan([suite], models, include_non_matching=True, waives=waives)
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

    cmd = validation_engine.build_bundle_command(
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


def test_build_bundle_command_passes_manifest_hf_revision(tmp_path: Path) -> None:
    model = {
        "name": "case",
        "hf_id": "org/model",
        "hf_revision": "0123456789abcdef",
        "max_cache_length": 256,
        "precision": "fp32",
    }

    cmd = validation_engine.build_bundle_command(
        model,
        trtmc_binary="build/trtmc",
        bundle_path=tmp_path / "case.trtfb",
    )

    assert cmd[cmd.index("--model-revision") : cmd.index("--model-revision") + 2] == [
        "--model-revision",
        "0123456789abcdef",
    ]


def test_build_bundle_command_passes_manifest_fp32_layers(tmp_path: Path) -> None:
    """Reduced-precision selectors must reach the build, matching the E2E harness."""
    model = {
        "name": "case",
        "hf_id": "org/model",
        "max_cache_length": 256,
        "precision": "fp16",
        "fp32_layers": [2, 3, 4, 7, 8],
    }

    cmd = validation_engine.build_bundle_command(
        model,
        trtmc_binary="build/trtmc",
        bundle_path=tmp_path / "case.trtfb",
    )

    idx = cmd.index("--fp32-layers")
    assert cmd[idx : idx + 2] == ["--fp32-layers", "2,3,4,7,8"]


def test_build_bundle_command_omits_fp32_layers_when_absent(tmp_path: Path) -> None:
    model = {
        "name": "case",
        "hf_id": "org/model",
        "max_cache_length": 256,
        "precision": "fp16",
    }

    cmd = validation_engine.build_bundle_command(
        model,
        trtmc_binary="build/trtmc",
        bundle_path=tmp_path / "case.trtfb",
    )

    assert "--fp32-layers" not in cmd


def test_ensure_bundle_replaces_existing_file_before_build(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle = tmp_path / "shared" / "model.trtfb"
    bundle.parent.mkdir()
    bundle.write_bytes(b"old")

    class Result:
        returncode = 0

    def fake_run(command, **_kwargs):
        output = Path(command[command.index("-o") + 1])
        assert not output.exists()
        output.write_bytes(b"new")
        return Result()

    monkeypatch.setattr(validation_engine.subprocess, "run", fake_run)

    result, built = validation_engine.ensure_bundle(
        {
            "name": "model",
            "hf_id": "org/model",
            "precision": "fp32",
        },
        bundle_path=bundle,
        trtmc_binary="trtmc",
        force_build=True,
        replace_existing=True,
    )

    assert result == bundle
    assert built is True
    assert bundle.read_bytes() == b"new"


def test_ensure_bundle_applies_selected_cuda_device_to_build(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle = tmp_path / "shared" / "model.trtfb"
    captured: dict[str, str] = {}

    class Result:
        returncode = 0

    def fake_run(command, **kwargs):
        captured["cuda_visible_devices"] = kwargs["env"]["CUDA_VISIBLE_DEVICES"]
        Path(command[command.index("-o") + 1]).write_bytes(b"bundle")
        return Result()

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "parent-device")
    monkeypatch.setattr(validation_engine.subprocess, "run", fake_run)

    validation_engine.ensure_bundle(
        {
            "name": "model",
            "hf_id": "org/model",
            "precision": "fp32",
        },
        bundle_path=bundle,
        trtmc_binary="trtmc",
        cuda_visible_devices="selected-device",
    )

    assert captured["cuda_visible_devices"] == "selected-device"


def test_ensure_bundle_removes_partial_replacement_after_failed_build(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle = tmp_path / "shared" / "model.trtfb"
    bundle.parent.mkdir()
    bundle.write_bytes(b"old")

    class Result:
        returncode = 1

    def fake_run(command, **_kwargs):
        Path(command[command.index("-o") + 1]).write_bytes(b"partial")
        return Result()

    monkeypatch.setattr(validation_engine.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="Bundle build failed"):
        validation_engine.ensure_bundle(
            {
                "name": "model",
                "hf_id": "org/model",
                "precision": "fp32",
            },
            bundle_path=bundle,
            trtmc_binary="trtmc",
            force_build=True,
            replace_existing=True,
        )

    assert not bundle.exists()


def test_ensure_bundle_replaces_dangling_shared_bundle_symlink(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle = tmp_path / "shared" / "model.trtfb"
    bundle.parent.mkdir()
    bundle.symlink_to(tmp_path / "missing.trtfb")

    class Result:
        returncode = 0

    def fake_run(command, **_kwargs):
        output = Path(command[command.index("-o") + 1])
        assert not output.is_symlink()
        output.write_bytes(b"new")
        return Result()

    monkeypatch.setattr(validation_engine.subprocess, "run", fake_run)

    validation_engine.ensure_bundle(
        {
            "name": "model",
            "hf_id": "org/model",
            "precision": "fp32",
        },
        bundle_path=bundle,
        trtmc_binary="trtmc",
        replace_existing=True,
    )

    assert not bundle.is_symlink()
    assert bundle.read_bytes() == b"new"


def test_ensure_bundle_replaces_incompatible_tensorrt_abi(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle = tmp_path / "shared" / "model.trtfb"
    bundle.parent.mkdir()
    bundle.write_bytes(b"old")
    commands: list[list[str]] = []

    class Result:
        returncode = 0
        stdout = "TRT ABI:            11.0\nMax cache length:   256\n"

    def fake_run(command, **_kwargs):
        commands.append(command)
        if command[1] == "inspect":
            return Result()
        output = Path(command[command.index("-o") + 1])
        assert not output.exists()
        output.write_bytes(b"new")
        return Result()

    monkeypatch.setattr(validation_engine.subprocess, "run", fake_run)
    monkeypatch.setattr(validation_engine, "runtime_tensorrt_abi", lambda: "11.1")

    _, built = validation_engine.ensure_bundle(
        {
            "name": "model",
            "hf_id": "org/model",
            "precision": "fp32",
        },
        bundle_path=bundle,
        trtmc_binary="trtmc",
        max_cache_length=256,
        replace_existing=True,
    )

    assert built is True
    assert bundle.read_bytes() == b"new"
    assert [command[1] for command in commands] == ["inspect", "build"]


def test_ensure_bundle_replaces_mismatched_precision(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle = tmp_path / "shared" / "model.trtfb"
    bundle.parent.mkdir()
    bundle.write_bytes(b"fp16")
    commands: list[list[str]] = []

    class Result:
        returncode = 0
        stdout = (
            "TRT ABI:            11.1\n"
            "Max cache length:   256\n"
            "Precision:          fp16\n"
        )

    def fake_run(command, **_kwargs):
        commands.append(command)
        if command[1] == "inspect":
            return Result()
        output = Path(command[command.index("-o") + 1])
        output.write_bytes(b"fp32")
        return Result()

    monkeypatch.setattr(validation_engine.subprocess, "run", fake_run)
    monkeypatch.setattr(validation_engine, "runtime_tensorrt_abi", lambda: "11.1")

    _, built = validation_engine.ensure_bundle(
        {
            "name": "model",
            "hf_id": "org/model",
            "precision": "fp32",
        },
        bundle_path=bundle,
        trtmc_binary="trtmc",
        max_cache_length=256,
        replace_existing=True,
    )

    assert built is True
    assert bundle.read_bytes() == b"fp32"
    assert [command[1] for command in commands] == ["inspect", "build"]


def test_bundle_reuse_rejects_unrecognized_precision(monkeypatch) -> None:
    monkeypatch.setattr(validation_engine, "runtime_tensorrt_abi", lambda: "11.1")

    assert not validation_engine._bundle_can_be_reused(
        {
            "TRT ABI": "11.1",
            "Max cache length": "256",
            "Precision": "unknown",
        },
        max_cache_length=256,
        expected_precision="fp32",
        allow_unknown=False,
    )


def test_suite_build_cache_minimum_overrides_manifest_cache() -> None:
    suite = {"build": {"min_max_cache_length": 1024}}
    model = {"max_cache_length": 256}

    assert validation_engine.requested_build_max_cache_length(suite, model) == 1024
    assert validation_engine.requested_build_max_cache_length(suite, model, prompt_max_tokens=2048) == 2048
    assert validation_engine.requested_build_max_cache_length(suite, model, 512) == 512


def test_continuation_reserves_generation_cache_headroom() -> None:
    assert (
        validation_engine.generation_cache_headroom(
            scorer="continuation",
            validation_config={},
            generation={"max_new_tokens": 64},
            max_new_tokens=None,
        )
        == 64
    )
    assert (
        validation_engine.generation_cache_headroom(
            scorer="continuation",
            validation_config={},
            generation={"max_new_tokens": 64},
            max_new_tokens=8,
        )
        == 8
    )


def test_non_continuation_reserves_generation_headroom_by_default() -> None:
    generation = {"max_new_tokens": 8}

    assert (
        validation_engine.generation_cache_headroom(
            scorer="mcq",
            validation_config={},
            generation=generation,
            max_new_tokens=None,
        )
        == 8
    )
    assert (
        validation_engine.generation_cache_headroom(
            scorer="mcq",
            validation_config={"build_generation_headroom": False},
            generation=generation,
            max_new_tokens=None,
        )
        == 0
    )


def test_eval_continuation_builds_for_prompt_and_generated_tokens(
    tmp_path: Path, monkeypatch
) -> None:
    dataset = tmp_path / "mmlu.json"
    _write_mmlu(dataset)
    suite = validation_engine.suite_by_id(
        validation_engine.load_suites(), "mmlu_continuation_parity"
    )
    suite["model_overrides"]["by_model"]["decoder-small"] = {
        "comparison_precision": "fp32"
    }
    model = {
        "name": "decoder-small",
        "hf_id": "example-org/decoder-small",
        "hf_revision": "0123456789abcdef",
        "bundle": "decoder-small.trtfb",
        "max_cache_length": 256,
        "precision": "fp16",
        "trust_remote_code": False,
        "build_args": {},
        "quantization": {},
    }
    responses = [
        {
            "sample_id": "mmlu_000000",
            "output_text": " A",
            "generated_token_ids": [1],
        },
        {
            "sample_id": "mmlu_000001",
            "output_text": " B",
            "generated_token_ids": [2],
        },
    ]

    def fake_run_hf(_args, _model, work_dir):
        assert _model["precision"] == "fp32"
        validation_engine.write_predictions(work_dir / "hf_predictions.json", responses)

    def fake_ensure_bundle(*_args, **kwargs):
        assert _args[0]["precision"] == "fp32"
        assert kwargs["max_cache_length"] == 445
        bundle = kwargs["bundle_path"]
        bundle.parent.mkdir(parents=True, exist_ok=True)
        bundle.write_bytes(b"bundle")
        return bundle, True

    def fake_run_trtfb(args):
        validation_engine.write_predictions(
            Path(args.work_dir) / "trtfb_predictions.json", responses
        )

    def fake_max_prompt_token_length(**kwargs):
        assert kwargs["model_revision"] == "0123456789abcdef"
        return 381

    monkeypatch.setattr(
        validation_engine,
        "max_prompt_token_length",
        fake_max_prompt_token_length,
    )
    monkeypatch.setattr(
        validation_engine, "run_hf_reference_subprocess", fake_run_hf
    )
    monkeypatch.setattr(validation_engine, "ensure_bundle", fake_ensure_bundle)
    monkeypatch.setattr(validation_engine, "run_trtfb", fake_run_trtfb)
    monkeypatch.setattr(
        validation_engine,
        "validate_text_input_token_contract",
        lambda **_kwargs: None,
    )
    args = validation_engine.build_arg_parser().parse_args(
        [
            "eval",
            "--dataset",
            str(dataset),
            "--work-root",
            str(tmp_path / "work"),
            "--engine-dir",
            str(tmp_path / "engines"),
            "--model",
            model["name"],
            "--local-files-only",
        ]
    )

    result = validation_engine.eval_one_model(suite=suite, model=model, args=args)

    assert result["build_max_cache_length"] == 445
    assert result["generation_cache_headroom"] == 64
    assert result["status"] == "passed"


def test_prompt_length_validation_rejects_over_cache(tmp_path: Path, monkeypatch) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "prompts.jsonl").write_text(
        json.dumps({"prompt": "long prompt"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(validation_engine, "max_prompt_token_length", lambda **_kwargs: 513)

    try:
        validation_engine.validate_prompt_lengths_for_cache(
            model={"name": "case", "hf_id": "org/model"},
            work_dir=work_dir,
            max_cache_length=512,
        )
    except RuntimeError as exc:
        assert "max_prompt_tokens=513" in str(exc)
    else:
        raise AssertionError("expected prompt length validation failure")


def test_max_prompt_token_length_uses_pinned_model_revision(
    tmp_path: Path,
    monkeypatch,
) -> None:
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps({"prompt": "one two"}) + "\n", encoding="utf-8")
    captured: dict[str, Any] = {}

    class Tokenizer:
        def __call__(self, text, *, add_special_tokens=False):
            assert add_special_tokens is False
            return argparse.Namespace(input_ids=text.split())

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            captured["model_id"] = model_id
            captured.update(kwargs)
            return Tokenizer()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=AutoTokenizer),
    )

    length = validation_engine.max_prompt_token_length(
        model_id="org/model",
        model_revision="0123456789abcdef",
        prompts_path=prompts,
        local_files_only=True,
    )

    assert length == 2
    assert captured == {
        "model_id": "org/model",
        "revision": "0123456789abcdef",
        "local_files_only": True,
        "trust_remote_code": False,
    }


def test_run_hf_reference_subprocess_uses_hf_python(tmp_path: Path, monkeypatch) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "manifest.json").write_text(
        json.dumps(
            {
                "task_eval": {
                    "hf_experts_implementation": "batched_mm",
                }
            }
        ),
        encoding="utf-8",
    )
    captured: dict[str, list[str]] = {}

    class Result:
        returncode = 0

    def fake_run(cmd, **_kwargs):
        captured["cmd"] = cmd
        return Result()

    monkeypatch.setattr(validation_engine.subprocess, "run", fake_run)

    args = argparse.Namespace(
        hf_python="/opt/deepseek-hf/bin/python3",
        reference_cache_dir=str(tmp_path / "references"),
        reference_cache_identity="org/model/reference-contract-v1",
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
        "hf_id": "org/model",
        "hf_revision": "0123456789abcdef",
        "trust_remote_code": False,
    }

    validation_engine.run_hf_reference_subprocess(args, model, work_dir)

    assert captured["cmd"][0] == "/opt/deepseek-hf/bin/python3"
    assert captured["cmd"][1:3] == [str(validation_engine.REFERENCE_RUNNER), "run"]
    assert captured["cmd"][captured["cmd"].index("--model-revision") + 1] == (
        "0123456789abcdef"
    )
    assert captured["cmd"][captured["cmd"].index("--cache-dir") + 1] == str(
        tmp_path / "references"
    )
    assert captured["cmd"][
        captured["cmd"].index("--reference-cache-identity") + 1
    ] == "org/model/reference-contract-v1"
    assert captured["cmd"][
        captured["cmd"].index("--experts-implementation") + 1
    ] == "batched_mm"


@pytest.mark.parametrize(
    ("precision", "expected"),
    [
        ("fp16", "float16"),
        ("bf16", "bfloat16"),
        ("fp32", "float32"),
    ],
)
def test_text_reference_auto_dtype_follows_engine_precision(
    tmp_path: Path,
    precision: str,
    expected: str,
) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "manifest.json").write_text(
        json.dumps({"dataset_kind": "mmlu_five_shot_json"}),
        encoding="utf-8",
    )

    dtype = validation_engine.resolve_hf_reference_dtype(
        argparse.Namespace(hf_dtype="auto"),
        {"precision": precision},
        work_dir,
    )

    assert dtype == expected


@pytest.mark.parametrize(
    "dataset_kind",
    [
        "asr_chat_json",
        "seedtts_json",
        "vlm_chat_json",
        "vlm_unified_json",
    ],
)
def test_native_multimodal_reference_dtype_follows_engine_precision(
    tmp_path: Path,
    dataset_kind: str,
) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "manifest.json").write_text(
        json.dumps({"dataset_kind": dataset_kind}),
        encoding="utf-8",
    )

    dtype = validation_engine.resolve_hf_reference_dtype(
        argparse.Namespace(hf_dtype="auto"),
        {"precision": "fp32"},
        work_dir,
    )

    assert dtype == "float32"


def test_explicit_reference_dtype_must_match_engine_precision(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "manifest.json").write_text(
        json.dumps({"dataset_kind": "mmlu_five_shot_json"}),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="reference precision bf16 does not match TRTMC base precision fp16",
    ):
        validation_engine.resolve_hf_reference_dtype(
            argparse.Namespace(hf_dtype="bfloat16"),
            {"precision": "fp16"},
            work_dir,
        )


def test_declared_native_reference_dtype_exception_is_recorded(
    tmp_path: Path,
) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_kind": "mmlu_five_shot_json",
                "task_eval": {
                    "reference_precision": "bf16",
                    "allow_reference_precision_mismatch": True,
                },
            }
        ),
        encoding="utf-8",
    )

    contract = validation_engine.resolve_reference_precision_contract(
        argparse.Namespace(hf_dtype="auto"),
        {"precision": "fp16"},
        work_dir,
    )

    assert contract == {
        "trtmc_base_precision": "fp16",
        "trtmc_quantization": "none",
        "reference_precision": "bf16",
        "reference_dtype": "bfloat16",
        "comparison": "reference_defined",
    }


def test_comparison_precision_overrides_both_candidate_and_reference(
    tmp_path: Path,
) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "manifest.json").write_text(
        json.dumps({"dataset_kind": "mmlu_five_shot_json"}),
        encoding="utf-8",
    )
    original = {
        "name": "sensitive-model",
        "precision": "fp16",
        "quantization": {},
    }

    model = validation_engine.apply_comparison_precision(
        original,
        {"comparison_precision": "fp32"},
    )
    contract = validation_engine.resolve_reference_precision_contract(
        argparse.Namespace(hf_dtype="auto"),
        model,
        work_dir,
    )

    assert original["precision"] == "fp16"
    assert model["precision"] == "fp32"
    assert contract["trtmc_base_precision"] == "fp32"
    assert contract["reference_precision"] == "fp32"
    assert contract["comparison"] == "aligned"


@pytest.mark.parametrize("model_name", ["fnet-base", "xlnet-base"])
def test_encoder_validation_compares_candidate_and_reference_in_fp32(
    model_name: str,
) -> None:
    suite = validation_engine.suite_by_id(
        validation_engine.load_suites(),
        "stsbenchmark_encoder_embedding_parity",
    )
    model = next(
        model
        for model in validation_engine.load_manifest_records()
        if model["name"] == model_name
    )

    validation_config = validation_engine.effective_validation_config(
        suite,
        model,
    )
    resolved_model = validation_engine.apply_comparison_precision(
        model,
        validation_config,
    )

    assert validation_config["comparison_precision"] == "fp32"
    assert model["precision"] == "fp16"
    assert resolved_model["precision"] == "fp32"


def test_segformer_validation_compares_candidate_and_reference_in_fp16() -> None:
    suite = validation_engine.suite_by_id(
        validation_engine.load_suites(),
        "ade20k_semantic_segmentation",
    )
    model = next(
        model
        for model in validation_engine.load_manifest_records()
        if model["name"] == "segformer-b0-ade"
    )

    validation_config = validation_engine.effective_validation_config(
        suite,
        model,
    )

    assert model["precision"] == "fp16"
    assert validation_config["reference_precision"] == "fp16"


@pytest.mark.parametrize(
    ("suite_id", "model_name"),
    [
        ("mmlu_five_shot_mcq", "gpt-oss-20b"),
        ("ocrbench_v2_unified", "deepseek-ocr"),
    ],
)
def test_official_bf16_reference_exceptions_are_explicit(
    suite_id: str,
    model_name: str,
) -> None:
    suite = validation_engine.suite_by_id(
        validation_engine.load_suites(),
        suite_id,
    )
    model = next(
        model
        for model in validation_engine.load_manifest_records()
        if model["name"] == model_name
    )

    validation_config = validation_engine.effective_validation_config(
        suite,
        model,
    )

    assert validation_config["reference_precision"] == "bf16"
    assert validation_config["allow_reference_precision_mismatch"] is True


def test_comparison_precision_rejects_quantized_models() -> None:
    with pytest.raises(
        ValueError,
        match="FP8 quantization.*may only override unquantized base precision",
    ):
        validation_engine.apply_comparison_precision(
            {
                "name": "quantized-model",
                "precision": "fp16",
                "quantization": {"format": "fp8"},
            },
            {"comparison_precision": "fp32"},
        )


@pytest.mark.parametrize("quantization_format", ["fp8", "nvfp4", "mxfp8"])
def test_quantized_reference_requires_explicit_base_precision(
    tmp_path: Path,
    quantization_format: str,
) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "manifest.json").write_text(
        json.dumps({"dataset_kind": "mmlu_five_shot_json"}),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=rf"{quantization_format.upper()}.*requires task_eval.reference_precision",
    ):
        validation_engine.resolve_hf_reference_dtype(
            argparse.Namespace(hf_dtype="auto"),
            {
                "name": "quantized-model",
                "precision": "bf16",
                "quantization": {"format": quantization_format},
            },
            work_dir,
        )


def test_quantized_reference_contract_records_candidate_and_base_precision(
    tmp_path: Path,
) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_kind": "mmlu_five_shot_json",
                "task_eval": {"reference_precision": "bf16"},
            }
        ),
        encoding="utf-8",
    )
    model = {
        "name": "quantized-model",
        "precision": "bf16",
        "quantization": {"format": "fp8"},
    }

    contract = validation_engine.resolve_reference_precision_contract(
        argparse.Namespace(hf_dtype="auto"),
        model,
        work_dir,
    )

    assert contract == {
        "trtmc_base_precision": "bf16",
        "trtmc_quantization": "fp8",
        "reference_precision": "bf16",
        "reference_dtype": "bfloat16",
        "comparison": "quantized_vs_unquantized_reference",
    }
    assert (
        validation_engine.resolve_hf_reference_dtype(
            argparse.Namespace(hf_dtype="auto"),
            model,
            work_dir,
        )
        == "bfloat16"
    )


def test_legacy_fp8_scale_model_is_treated_as_quantized(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_kind": "diffusion_prompt_json",
                "task_eval": {"reference_precision": "bf16"},
            }
        ),
        encoding="utf-8",
    )

    contract = validation_engine.resolve_reference_precision_contract(
        argparse.Namespace(hf_dtype="auto"),
        {
            "name": "flux-2-dev-fp8",
            "precision": "fp16",
            "fp8_scales": "scales.json",
        },
        work_dir,
    )

    assert contract["trtmc_quantization"] == "fp8"
    assert contract["reference_precision"] == "bf16"
    assert contract["comparison"] == "quantized_vs_unquantized_reference"


def test_non_transformers_reference_keeps_auto_dtype(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "manifest.json").write_text(
        json.dumps({"dataset_kind": "diffusion_prompt_json"}),
        encoding="utf-8",
    )

    dtype = validation_engine.resolve_hf_reference_dtype(
        argparse.Namespace(hf_dtype="auto"),
        {"precision": "fp16"},
        work_dir,
    )

    assert dtype == "auto"


def test_non_transformers_reference_can_declare_pipeline_dtype(
    tmp_path: Path,
) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_kind": "diffusion_prompt_json",
                "task_eval": {"reference_precision": "bf16"},
            }
        ),
        encoding="utf-8",
    )

    contract = validation_engine.resolve_reference_precision_contract(
        argparse.Namespace(hf_dtype="auto"),
        {"precision": "fp16"},
        work_dir,
    )

    assert contract == {
        "trtmc_base_precision": "fp16",
        "trtmc_quantization": "none",
        "reference_precision": "bf16",
        "reference_dtype": "bfloat16",
        "comparison": "reference_defined",
    }


def test_text_input_contract_reports_first_token_mismatch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "prompts.jsonl").write_text(
        json.dumps({"sample_id": "mmlu_000007", "prompt": "Question"}) + "\n",
        encoding="utf-8",
    )

    class HfTokenizer:
        def __call__(self, text):
            assert text == "Question"
            return SimpleNamespace(input_ids=[10, 11, 12])

    class BundleTokenizer:
        def encode(self, text, *, add_special_tokens):
            assert text == "Question"
            assert add_special_tokens is False
            return SimpleNamespace(ids=[10, 11, 12])

    monkeypatch.setattr(
        validation_engine,
        "_load_text_input_contract",
        lambda **_kwargs: (
            HfTokenizer(),
            BundleTokenizer(),
            {
                "tokenizer_special_prefix_ids": [2],
                "tokenizer_special_suffix_ids": [],
            },
        ),
    )

    with pytest.raises(
        RuntimeError,
        match=r"mmlu_000007.*first_difference=0.*HF=\[10, 11, 12\].*TRTMC=\[2, 10, 11, 12\]",
    ):
        validation_engine.validate_text_input_token_contract(
            model={"name": "opt-125m", "hf_id": "facebook/opt-125m"},
            work_dir=work_dir,
            bundle_path=tmp_path / "opt-125m.trtfb",
            local_files_only=True,
            trust_remote_code=False,
        )

    artifact = json.loads(
        (work_dir / "input_token_contract.json").read_text(encoding="utf-8")
    )
    assert artifact["status"] == "mismatch"
    assert artifact["samples"][0]["hf_token_count"] == 3
    assert artifact["samples"][0]["trtmc_token_count"] == 4
    assert artifact["samples"][0]["first_difference"] == 0


def test_text_input_contract_records_aligned_token_digests(
    tmp_path: Path,
    monkeypatch,
) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "prompts.jsonl").write_text(
        json.dumps({"sample_id": "sample-1", "prompt": "Question"}) + "\n",
        encoding="utf-8",
    )

    class HfTokenizer:
        def __call__(self, _text):
            return SimpleNamespace(input_ids=[2, 10, 11])

    class BundleTokenizer:
        def encode(self, _text, *, add_special_tokens):
            assert add_special_tokens is False
            return SimpleNamespace(ids=[10, 11])

    monkeypatch.setattr(
        validation_engine,
        "_load_text_input_contract",
        lambda **_kwargs: (
            HfTokenizer(),
            BundleTokenizer(),
            {
                "tokenizer_special_prefix_ids": [2],
                "tokenizer_special_suffix_ids": [],
            },
        ),
    )

    validation_engine.validate_text_input_token_contract(
        model={"name": "decoder", "hf_id": "org/decoder"},
        work_dir=work_dir,
        bundle_path=tmp_path / "decoder.trtfb",
        local_files_only=True,
        trust_remote_code=False,
    )

    artifact = json.loads(
        (work_dir / "input_token_contract.json").read_text(encoding="utf-8")
    )
    assert artifact["status"] == "aligned"
    assert artifact["samples"] == [
        {
            "sample_id": "sample-1",
            "hf_token_count": 3,
            "trtmc_token_count": 3,
            "hf_token_sha256": artifact["samples"][0]["hf_token_sha256"],
            "trtmc_token_sha256": artifact["samples"][0]["trtmc_token_sha256"],
        }
    ]
    assert artifact["samples"][0]["hf_token_sha256"] == (
        artifact["samples"][0]["trtmc_token_sha256"]
    )


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

    monkeypatch.setattr(validation_engine.subprocess, "run", fake_run)
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

    validation_engine.run_hf_reference_subprocess(args, model, work_dir)

    assert captured["cmd"][captured["cmd"].index("--family") + 1] == "canary"
    assert captured["cmd"][captured["cmd"].index("--reference-family") + 1] == "asr_canary"


def test_asr_reference_detection_identifies_canary() -> None:
    assert validation_engine._is_canary_asr_reference(
        argparse.Namespace(
            model="nvidia/canary-1b-v2",
            family="",
            reference_family="",
        )
    )
    assert validation_engine._is_canary_asr_reference(
        argparse.Namespace(
            model="nvidia/other",
            family="canary",
            reference_family="",
        )
    )
    assert validation_engine._is_canary_asr_reference(
        argparse.Namespace(
            model="nvidia/other",
            family="",
            reference_family="asr_canary",
        )
    )


def test_nemo_asr_reference_detection_identifies_streaming() -> None:
    assert validation_engine._is_nemo_asr_reference(
        argparse.Namespace(
            model="nvidia/nemotron-speech-streaming-en-0.6b",
            family="",
            reference_family="",
        )
    )
    assert validation_engine._is_nemo_asr_reference(
        argparse.Namespace(
            model="nvidia/other",
            family="nemotron_speech_streaming",
            reference_family="",
        )
    )
    assert validation_engine._is_nemo_asr_reference(
        argparse.Namespace(
            model="nvidia/canary-1b-v2",
            family="canary",
            reference_family="asr_canary",
        )
    )


def test_nemotron_35_runtime_flags_enable_language_and_streaming() -> None:
    flags = validation_engine._asr_runtime_flags(
        {"language": "en-US"},
        {
            "streaming": {
                "enabled": True,
                "chunk_ms": 1120,
                "att_context_size": [56, 13],
            }
        },
    )

    assert flags == [
        "--language",
        "en-US",
        "--stream",
        "--chunk-ms",
        "1120",
        "--att-context-size",
        "56,13",
    ]


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

    monkeypatch.setattr(validation_engine, "run_asr_hf_reference", fake_asr)

    validation_engine.run_hf_reference(argparse.Namespace(work_dir=str(work_dir)))

    assert calls == ["asr"]


def test_run_hf_reference_dispatches_diffusion_prompt_workdir(
    tmp_path: Path, monkeypatch
) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "manifest.json").write_text(
        json.dumps({"dataset_kind": "diffusion_prompt_tsv"}),
        encoding="utf-8",
    )
    calls: list[str] = []

    def fake_diffusion(_args):
        calls.append("diffusion")

    monkeypatch.setattr(validation_engine, "run_diffusion_hf_reference", fake_diffusion)

    validation_engine.run_hf_reference(argparse.Namespace(work_dir=str(work_dir)))

    assert calls == ["diffusion"]


def test_run_diffusion_hf_reference_writes_image_artifact_predictions(
    tmp_path: Path, monkeypatch
) -> None:
    from tests.e2e_harness.contracts import E2ECase, StageOutput

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "manifest.json").write_text(json.dumps({
        "dataset_kind": "diffusion_prompt_tsv",
        "generation": {
            "seed": 42,
            "image_height": 384,
            "image_width": 384,
            "num_inference_steps": 20,
        },
        "task_eval": {"model_manifest": "tests/e2e/models/flux/manifests/flux-schnell-l0.json"},
    }), encoding="utf-8")
    (work_dir / "prompts.jsonl").write_text(
        json.dumps({"sample_id": "partiprompts_000000", "prompt": "a red cube"}) + "\n"
        + json.dumps({"sample_id": "partiprompts_000001", "prompt": "a blue sphere"}) + "\n",
        encoding="utf-8",
    )
    seen: list[tuple[str, int, int]] = []

    class FakeReference:
        def run_stage(self, case, _stage, _ctx):
            seen.append((case.inputs["prompt"], case.inputs["seed"], case.inputs["image_height"]))
            frames_dir = work_dir / "fake_frames" / case.name
            frames_dir.mkdir(parents=True)
            (frames_dir / "frame_0000.png").write_bytes(b"png")
            return StageOutput(
                stage_name="end_to_end",
                data={
                    "returncode": 0,
                    "num_frames": 1,
                    "frames_dir": str(frames_dir),
                    "frame_stats": {"mean": 0.5, "std": 0.2},
                },
                timing_s=1.25,
            )

    template = E2ECase(
        name="flux-schnell-l0",
        hf_id="black-forest-labs/FLUX.1-schnell",
        family="flux",
        runtime_strategy="diffusion_flux",
        task_strategy="diffusion_media_generation",
        reference_backend="hf_diffusers",
        bundle="flux-schnell-l0.trtfb",
        inputs={},
    )
    monkeypatch.setattr(
        validation_engine,
        "_load_diffusion_validation_plugins",
        lambda _work_dir: (template, FakeReference(), object()),
        raising=False,
    )

    validation_engine.run_diffusion_hf_reference(argparse.Namespace(
        work_dir=str(work_dir),
        predictions="hf_predictions.json",
        raw_output="hf_raw.jsonl",
        dtype="float16",
    ))

    predictions = json.loads((work_dir / "hf_predictions.json").read_text(encoding="utf-8"))
    assert seen == [("a red cube", 42, 384), ("a blue sphere", 43, 384)]
    assert template.metadata["reference_precision"] == "fp16"
    assert predictions["responses"][0]["sample_id"] == "partiprompts_000000"
    assert predictions["responses"][0]["num_frames"] == 1
    assert predictions["responses"][0]["source"] == "hf"


def test_captured_utf8_subprocess_replaces_invalid_native_output() -> None:
    result = validation_engine._run_captured_utf8_subprocess(
        [
            sys.executable,
            "-c",
            (
                "import os; "
                "os.write(1, b'answer:\\x93A'); "
                "os.write(2, b'warning:\\xff')"
            ),
        ]
    )

    assert result.returncode == 0
    assert result.stdout == "answer:\ufffdA"
    assert result.stderr == "warning:\ufffd"


def test_run_trtfb_dispatches_diffusion_prompt_workdir(
    tmp_path: Path, monkeypatch
) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "manifest.json").write_text(
        json.dumps({"dataset_kind": "diffusion_prompt_tsv"}),
        encoding="utf-8",
    )
    calls: list[str] = []

    def fake_diffusion(_args):
        calls.append("diffusion")

    monkeypatch.setattr(
        validation_engine, "run_diffusion_trtfb", fake_diffusion, raising=False
    )

    validation_engine.run_trtfb(argparse.Namespace(work_dir=str(work_dir)))

    assert calls == ["diffusion"]


def test_dataset_benchmark_reproduction_is_direct_and_uses_single_input(
    tmp_path: Path,
) -> None:
    command = [
        "/workspace/build/trtmc_dataset_benchmark",
        "/runs/engines/model.trtfb",
        "/runs/work/prompts.jsonl",
        "/runs/work/trtfb_raw.jsonl",
        "--max-new-tokens",
        "8",
    ]

    validation_engine._write_dataset_benchmark_reproduction(tmp_path, command)

    payload = json.loads(
        (tmp_path / "trtfb_repro.json").read_text(encoding="utf-8")
    )
    assert payload["backend"] == "trtmc_dataset_benchmark"
    assert payload["command"][0] == "/workspace/build/trtmc_dataset_benchmark"
    assert payload["command"][2] == "{input_jsonl}"
    assert payload["command"][3] == "{trtmc_raw_jsonl}"
    assert "validation_engine.py" not in " ".join(payload["command"])


def test_dataset_benchmark_reproduction_preserves_per_sample_seed(
    tmp_path: Path,
) -> None:
    command = [
        "/workspace/build/trtmc_dataset_benchmark",
        "/runs/engines/model.trtfb",
        "/runs/work/prompts.jsonl",
        "/runs/work/trtfb_raw.jsonl",
        "--seed",
        "42",
    ]

    validation_engine._write_dataset_benchmark_reproduction(tmp_path, command)

    payload = json.loads(
        (tmp_path / "trtfb_repro.json").read_text(encoding="utf-8")
    )
    assert payload["base_seed"] == 42
    assert payload["command"][payload["command"].index("--seed") + 1] == (
        "{sample_seed}"
    )


def test_native_trtmc_command_recorder_extracts_nested_model_command(
    tmp_path: Path,
) -> None:
    validation_engine._reset_native_trtmc_commands(tmp_path)
    output = SimpleNamespace(
        metadata={
            "cpp": {
                "command": [
                    "/workspace/build/trtmc",
                    "run",
                    "/runs/engines/model.trtfb",
                    "--prompt",
                    "hello",
                ]
            }
        }
    )

    validation_engine._record_output_native_command(
        tmp_path,
        "sample-7",
        output,
    )

    row = json.loads(
        (tmp_path / "trtfb_native_commands.jsonl").read_text(encoding="utf-8")
    )
    assert row["sample_id"] == "sample-7"
    assert row["command"][0:2] == ["/workspace/build/trtmc", "run"]
    assert "validation_engine.py" not in " ".join(row["command"])


def test_run_diffusion_trtfb_writes_image_artifact_predictions(
    tmp_path: Path, monkeypatch
) -> None:
    from tests.e2e_harness.contracts import E2ECase, StageOutput

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "manifest.json").write_text(json.dumps({
        "dataset_kind": "diffusion_prompt_tsv",
        "generation": {"seed": 7, "num_inference_steps": 20},
        "task_eval": {"model_manifest": "tests/e2e/models/flux/manifests/flux-schnell-l0.json"},
    }), encoding="utf-8")
    (work_dir / "prompts.jsonl").write_text(
        json.dumps({"sample_id": "partiprompts_000000", "prompt": "a red cube"}) + "\n",
        encoding="utf-8",
    )
    seen: list[tuple[str, str, str]] = []

    class FakeRunner:
        def run_stage(self, case, _stage, ctx):
            seen.append((case.inputs["prompt"], ctx.binary_path, case.bundle))
            frames_dir = work_dir / "fake_trt_frames" / case.name
            frames_dir.mkdir(parents=True)
            (frames_dir / "frame_0000.png").write_bytes(b"png")
            return StageOutput(
                stage_name="end_to_end",
                data={
                    "returncode": 0,
                    "num_frames": 1,
                    "frames_dir": str(frames_dir),
                    "frame_stats": {"mean": 0.5, "std": 0.2},
                    "prompt": case.inputs["prompt"],
                },
                metadata={
                    "command": [
                        "build/trtmc",
                        "generate-video",
                        str(tmp_path / "bundles" / "flux-schnell-l0.trtfb"),
                        "--prompt",
                        case.inputs["prompt"],
                    ]
                },
                timing_s=0.5,
            )

    template = E2ECase(
        name="flux-schnell-l0",
        hf_id="black-forest-labs/FLUX.1-schnell",
        family="flux",
        runtime_strategy="diffusion_flux",
        task_strategy="diffusion_media_generation",
        reference_backend="hf_diffusers",
        bundle="flux-schnell-l0.trtfb",
        inputs={},
    )
    monkeypatch.setattr(
        validation_engine,
        "_load_diffusion_validation_plugins",
        lambda _work_dir: (template, object(), FakeRunner()),
    )

    validation_engine.run_diffusion_trtfb(argparse.Namespace(
        work_dir=str(work_dir),
        bundle=str(tmp_path / "bundles" / "flux-schnell-l0.trtfb"),
        trtmc_binary="build/trtmc",
        hf_python="/opt/venv/bin/python",
        predictions="trtfb_predictions.json",
        raw_output="trtfb_raw.jsonl",
    ))

    predictions = json.loads((work_dir / "trtfb_predictions.json").read_text(encoding="utf-8"))
    assert seen == [("a red cube", "build/trtmc", "flux-schnell-l0.trtfb")]
    assert predictions["responses"][0]["source"] == "trtfb"
    assert predictions["responses"][0]["num_frames"] == 1
    assert (work_dir / "trtfb_run.log").read_text(encoding="utf-8") == (
        "$ build/trtmc generate-video "
        f"{tmp_path / 'bundles' / 'flux-schnell-l0.trtfb'} "
        "--prompt 'a red cube'\n"
    )


def test_compare_diffusion_image_predictions_aggregates_model_comparator_metrics(
    tmp_path: Path, monkeypatch
) -> None:
    from tests.e2e_harness.contracts import (
        CompareResult,
        MetricResult,
        StageStatus,
    )

    class FakeComparator:
        def compare(self, trt, ref, threshold, stage):
            assert trt.data["prompt"] == "a red cube"
            assert ref.data["frames_dir"] == "/hf/frames"
            assert threshold.metrics["max_prompt_clipscore_drop"] == 3.0
            assert stage.name == "end_to_end"
            return CompareResult(
                stage_name="end_to_end",
                status=StageStatus.PASSED.value,
                metrics={
                    "prompt_clipscore_delta": MetricResult(
                        value=-0.5, threshold=-3.0, operator=">=", passed=True
                    ),
                    "trt_prompt_clipscore": MetricResult(
                        value=24.0, threshold=None, operator="info", passed=True
                    ),
                    "hf_prompt_clipscore": MetricResult(
                        value=24.5, threshold=20.0, operator=">=", passed=True
                    ),
                    "trt_hf_image_clip_cosine": MetricResult(
                        value=0.9, threshold=None, operator=">=", passed=True
                    ),
                },
            )

    monkeypatch.setattr(
        validation_engine,
        "_load_diffusion_validation_comparator",
        lambda _work_dir: FakeComparator(),
        raising=False,
    )
    hf = {"responses": [{
        "sample_id": "partiprompts_000000",
        "returncode": 0,
        "num_frames": 1,
        "frames_dir": "/hf/frames",
        "frame_stats": {"mean": 0.5, "std": 0.2},
        "prompt": "a red cube",
    }]}
    trt = {"responses": [{
        "sample_id": "partiprompts_000000",
        "returncode": 0,
        "num_frames": 1,
        "frames_dir": "/trt/frames",
        "frame_stats": {"mean": 0.5, "std": 0.2},
        "prompt": "a red cube",
    }]}
    answers = {"requests": [{
        "sample_id": "partiprompts_000000",
        "category": "Simple Detail",
        "challenge": "Basic",
        "prompt": "a red cube",
        "questions": [{"question": "Is there a red cube?"}],
    }]}

    summary = validation_engine.compare_diffusion_image_predictions(
        hf,
        trt,
        answers,
        work_dir=tmp_path,
        gates={"max_prompt_clipscore_drop": 3.0},
    )

    assert summary["overall_pass_rate"] == 1.0
    assert summary["passed_count"] == 1
    assert summary["metrics"]["prompt_clipscore_delta"]["mean"] == -0.5
    assert summary["samples"][0]["category"] == "Simple Detail"
    assert summary["samples"][0]["prompt"] == "a red cube"
    review = Path(summary["visual_review"])
    assert review.is_file()
    review_html = review.read_text(encoding="utf-8")
    assert "Is there a red cube?" in review_html
    assert "HF image missing" in review_html
    assert "TRTMC image missing" in review_html


def test_diffusion_response_preserves_initial_latent_identity() -> None:
    from tests.e2e_harness.contracts import StageOutput

    response = validation_engine._diffusion_response(
        "sample-1",
        "hf",
        StageOutput(
            stage_name="end_to_end",
            data={
                "returncode": 0,
                "num_frames": 1,
                "frames_dir": "/frames",
                "initial_latents_sha256": "abc123",
            },
        ),
    )

    assert response["initial_latents_sha256"] == "abc123"


def test_diffusion_response_preserves_model_owned_native_acceptance() -> None:
    from tests.e2e_harness.contracts import StageOutput

    policy = {
        "kind": "native_visual_semantic_acceptance",
        "reference_role": "diagnostic",
    }
    response = validation_engine._diffusion_response(
        "sample-1",
        "hf",
        StageOutput(
            stage_name="end_to_end",
            data={
                "returncode": 0,
                "num_frames": 1,
                "frames_dir": "/frames",
                "native_acceptance": policy,
            },
        ),
    )

    assert response["native_acceptance"] == policy
    assert response["native_acceptance"] is not policy


def test_diffusion_comparison_restores_frame_paths_and_native_acceptance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from tests.e2e_harness.contracts import (
        CompareResult,
        MetricResult,
        StageStatus,
    )

    hf_frames = tmp_path / "hf"
    trt_frames = tmp_path / "trt"
    hf_frames.mkdir()
    trt_frames.mkdir()
    for root in (hf_frames, trt_frames):
        (root / "frame_0000.png").write_bytes(b"png")
        (root / "frame_0001.png").write_bytes(b"png")
    policy = {"kind": "native_visual_semantic_acceptance"}

    class Comparator:
        def compare(self, trt, ref, _threshold, _stage):
            assert trt.data["frame_paths"] == [
                str(trt_frames / "frame_0000.png"),
                str(trt_frames / "frame_0001.png"),
            ]
            assert ref.data["frame_paths"] == [
                str(hf_frames / "frame_0000.png"),
                str(hf_frames / "frame_0001.png"),
            ]
            assert ref.data["native_acceptance"] == policy
            return CompareResult(
                stage_name="end_to_end",
                status=StageStatus.PASSED.value,
                metrics={
                    "prompt_clipscore_delta": MetricResult(
                        value=0.0, threshold=None, operator="info", passed=True
                    ),
                    "trt_prompt_clipscore": MetricResult(
                        value=30.0, threshold=None, operator="info", passed=True
                    ),
                    "hf_prompt_clipscore": MetricResult(
                        value=30.0, threshold=None, operator="info", passed=True
                    ),
                    "trt_hf_image_clip_cosine": MetricResult(
                        value=1.0, threshold=None, operator=">=", passed=True
                    ),
                },
            )

    monkeypatch.setattr(
        validation_engine,
        "_load_diffusion_validation_comparator",
        lambda _work_dir: Comparator(),
    )
    monkeypatch.setattr(
        validation_engine,
        "_model_owned_diffusion_native_acceptance",
        lambda _work_dir: policy,
    )
    base = {
        "sample_id": "wan22_000000",
        "returncode": 0,
        "num_frames": 2,
        "frame_stats": {"mean": 0.5, "std": 0.2},
        "prompt": "cats boxing",
        "seed": 42,
    }

    summary = validation_engine.compare_diffusion_image_predictions(
        {
            "responses": [
                {
                    **base,
                    "frames_dir": str(hf_frames),
                }
            ]
        },
        {"responses": [{**base, "frames_dir": str(trt_frames)}]},
        {"requests": [{"sample_id": "wan22_000000", "prompt": "cats boxing"}]},
        work_dir=tmp_path,
        gates={},
    )

    assert summary["overall_pass_rate"] == 1.0


def test_diffusion_sample_inputs_and_response_record_shared_conditions(
    tmp_path: Path,
) -> None:
    from tests.e2e_harness.contracts import E2ECase, StageOutput

    condition = tmp_path / "condition.png"
    condition.write_bytes(b"condition-image")
    template = E2ECase(
        name="image-edit-model",
        hf_id="example/image-edit-model",
        family="image_edit_family",
        runtime_strategy="diffusion_image_edit",
        task_strategy="diffusion_media_generation",
        inputs={"image": "/old/image.png", "action": "w-320"},
    )
    case = validation_engine._diffusion_case_for_prompt(
        template,
        {
            "sample_id": "gedit_000000",
            "prompt": "turn it blue",
            "image": str(condition),
            "action": "w-160,d-160",
        },
        {"seed": 42},
        3,
    )

    assert case.inputs["image"] == str(condition)
    assert case.inputs["action"] == "w-160,d-160"
    assert case.inputs["seed"] == 45
    response = validation_engine._diffusion_response(
        case.name,
        "hf",
        StageOutput(
            stage_name="end_to_end",
            data={"returncode": 0, "num_frames": 1},
        ),
        case=case,
    )
    assert response["seed"] == 45
    assert response["action"] == "w-160,d-160"
    assert response["condition_image_sha256"] == validation_engine._sha256_file(condition)


def test_diffusion_parity_rejects_mismatched_shared_sample_inputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    base = {
        "sample_id": "sample-1",
        "returncode": 0,
        "num_frames": 1,
        "frames_dir": "/frames",
        "prompt": "a moving object",
        "seed": 42,
    }
    monkeypatch.setattr(
        validation_engine,
        "_load_diffusion_validation_comparator",
        lambda _work_dir: object(),
    )

    with pytest.raises(ValueError, match="shared input mismatch.*seed"):
        validation_engine.compare_diffusion_image_predictions(
            {"responses": [base]},
            {"responses": [{**base, "seed": 43}]},
            {"requests": [{"sample_id": "sample-1", "prompt": "a moving object"}]},
            work_dir=tmp_path,
            gates={},
        )


def test_diffusion_parity_rejects_dataset_condition_digest_mismatch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    row = {
        "sample_id": "sample-1",
        "returncode": 0,
        "num_frames": 1,
        "frames_dir": "/frames",
        "prompt": "turn it blue",
        "condition_image_sha256": "actual-digest",
    }
    monkeypatch.setattr(
        validation_engine,
        "_load_diffusion_validation_comparator",
        lambda _work_dir: object(),
    )

    with pytest.raises(ValueError, match="dataset input mismatch.*condition_image"):
        validation_engine.compare_diffusion_image_predictions(
            {"responses": [row]},
            {"responses": [row]},
            {
                "requests": [
                    {
                        "sample_id": "sample-1",
                        "prompt": "turn it blue",
                        "condition_image_sha256": "declared-digest",
                    }
                ]
            },
            work_dir=tmp_path,
            gates={},
        )


def test_diffusion_parity_rejects_mismatched_initial_latents(
    tmp_path: Path, monkeypatch
) -> None:
    from tests.e2e_harness.contracts import CompareResult, MetricResult, StageStatus

    class Comparator:
        def compare(self, *_args):
            return CompareResult(
                stage_name="end_to_end",
                status=StageStatus.PASSED.value,
                metrics={
                    "prompt_clipscore_delta": MetricResult(
                        value=0.0, threshold=None, operator="info", passed=True
                    ),
                    "trt_prompt_clipscore": MetricResult(
                        value=30.0, threshold=None, operator="info", passed=True
                    ),
                    "hf_prompt_clipscore": MetricResult(
                        value=30.0, threshold=None, operator="info", passed=True
                    ),
                    "trt_hf_image_clip_cosine": MetricResult(
                        value=1.0, threshold=0.9, operator=">=", passed=True
                    ),
                },
            )

    monkeypatch.setattr(
        validation_engine,
        "_load_diffusion_validation_comparator",
        lambda _work_dir: Comparator(),
    )
    base = {
        "sample_id": "partiprompts_000000",
        "returncode": 0,
        "num_frames": 1,
        "frames_dir": "/frames",
        "frame_stats": {"mean": 0.5, "std": 0.2},
        "prompt": "a red cube",
    }

    with pytest.raises(ValueError, match="initial latent"):
        validation_engine.compare_diffusion_image_predictions(
            {"responses": [{**base, "initial_latents_sha256": "hf-hash"}]},
            {"responses": [{**base, "initial_latents_sha256": "trt-hash"}]},
            {"requests": [{"sample_id": "partiprompts_000000"}]},
            work_dir=tmp_path,
            gates={"min_trt_hf_image_clip_cosine": 0.9},
        )


def test_diffusion_parity_requires_declared_initial_latent_identity(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        validation_engine,
        "_load_diffusion_validation_comparator",
        lambda _work_dir: object(),
    )
    row = {
        "sample_id": "partiprompts_000000",
        "returncode": 0,
        "num_frames": 1,
        "frames_dir": "/frames",
        "frame_stats": {"mean": 0.5, "std": 0.2},
        "prompt": "a red cube",
    }

    with pytest.raises(ValueError, match="requires matching initial latents"):
        validation_engine.compare_diffusion_image_predictions(
            {"responses": [row]},
            {"responses": [row]},
            {"requests": [{"sample_id": "partiprompts_000000"}]},
            work_dir=tmp_path,
            gates={"require_matching_initial_latents": 1},
        )


def test_compare_diffusion_image_predictions_requires_clip_metrics(
    tmp_path: Path, monkeypatch
) -> None:
    from tests.e2e_harness.contracts import CompareResult, StageStatus

    class ComparatorWithoutClip:
        def compare(self, *_args):
            return CompareResult(
                stage_name="end_to_end",
                status=StageStatus.PASSED.value,
                metrics={},
            )

    monkeypatch.setattr(
        validation_engine,
        "_load_diffusion_validation_comparator",
        lambda _work_dir: ComparatorWithoutClip(),
    )
    monkeypatch.setattr(
        validation_engine,
        "_compute_validation_clip_metrics",
        lambda *_args: None,
        raising=False,
    )
    row = {
        "sample_id": "partiprompts_000000",
        "returncode": 0,
        "num_frames": 1,
        "frames_dir": "/frames",
        "frame_stats": {"mean": 0.5, "std": 0.2},
        "prompt": "a red cube",
    }

    try:
        validation_engine.compare_diffusion_image_predictions(
            {"responses": [row]},
            {"responses": [row]},
            {"requests": [{"sample_id": "partiprompts_000000"}]},
            work_dir=tmp_path,
            gates={"max_prompt_clipscore_drop": 3.0},
        )
    except RuntimeError as exc:
        assert "required CLIP metrics" in str(exc)
    else:
        raise AssertionError("expected missing CLIP metric failure")


def test_compare_diffusion_image_predictions_adds_generic_clip_metrics(
    tmp_path: Path, monkeypatch
) -> None:
    from types import SimpleNamespace
    from tests.e2e_harness.contracts import CompareResult, StageStatus

    class PixArtComparator:
        def compare(self, *_args):
            return CompareResult(
                stage_name="end_to_end",
                status=StageStatus.PASSED.value,
                metrics={},
            )

    monkeypatch.setattr(
        validation_engine,
        "_load_diffusion_validation_comparator",
        lambda _work_dir: PixArtComparator(),
    )
    monkeypatch.setattr(
        validation_engine,
        "_compute_validation_clip_metrics",
        lambda trt_dir, ref_dir, prompt: SimpleNamespace(
            trt_prompt_clipscore=24.0,
            hf_prompt_clipscore=25.0,
            prompt_clipscore_delta=-1.0,
            trt_hf_image_clip_cosine=0.8,
            prompt_truncated=False,
        ),
        raising=False,
    )
    row = {
        "sample_id": "partiprompts_000000",
        "returncode": 0,
        "num_frames": 1,
        "frames_dir": "/frames",
        "frame_stats": {"mean": 0.5, "std": 0.2},
        "prompt": "a red cube",
    }

    summary = validation_engine.compare_diffusion_image_predictions(
        {"responses": [{**row, "frames_dir": "/hf"}]},
        {"responses": [{**row, "frames_dir": "/trt"}]},
        {"requests": [{"sample_id": "partiprompts_000000"}]},
        work_dir=tmp_path,
        gates={
            "max_prompt_clipscore_drop": 3.0,
            "min_hf_prompt_clipscore": 20.0,
            "min_trt_hf_image_clip_cosine": 0.0,
        },
    )

    assert summary["metrics"]["prompt_clipscore_delta"]["mean"] == -1.0
    assert summary["overall_pass_rate"] == 1.0


def test_diffusion_parity_gates_image_to_image_not_prompt_alignment(
    tmp_path: Path, monkeypatch
) -> None:
    from types import SimpleNamespace
    from tests.e2e_harness.contracts import CompareResult, StageStatus

    class PixArtComparator:
        def compare(self, *_args):
            return CompareResult(
                stage_name="end_to_end",
                status=StageStatus.PASSED.value,
                metrics={},
            )

    monkeypatch.setattr(
        validation_engine,
        "_load_diffusion_validation_comparator",
        lambda _work_dir: PixArtComparator(),
    )
    monkeypatch.setattr(
        validation_engine,
        "_compute_validation_clip_metrics",
        lambda *_args: SimpleNamespace(
            trt_prompt_clipscore=31.0,
            hf_prompt_clipscore=32.0,
            prompt_clipscore_delta=-1.0,
            trt_hf_image_clip_cosine=0.70,
            prompt_truncated=False,
        ),
    )
    row = {
        "sample_id": "partiprompts_000000",
        "returncode": 0,
        "num_frames": 1,
        "frames_dir": "/frames",
        "frame_stats": {"mean": 0.5, "std": 0.2},
        "prompt": "a red cube",
    }

    summary = validation_engine.compare_diffusion_image_predictions(
        {"responses": [{**row, "frames_dir": "/hf"}]},
        {"responses": [{**row, "frames_dir": "/trt"}]},
        {"requests": [{"sample_id": "partiprompts_000000"}]},
        work_dir=tmp_path,
        gates={"min_trt_hf_image_clip_cosine": 0.90},
    )

    metrics = summary["samples"][0]["metrics"]
    assert summary["overall_pass_rate"] == 0.0
    assert summary["samples"][0]["message"].startswith("FAIL:")
    assert metrics["trt_hf_image_clip_cosine"]["passed"] is False
    assert metrics["prompt_clipscore_delta"]["threshold"] is None
    assert metrics["hf_prompt_clipscore"]["threshold"] is None


def test_eval_one_model_passes_model_manifest_to_diffusion_prepare(
    tmp_path: Path, monkeypatch
) -> None:
    suite = validation_engine.suite_by_id(
        validation_engine.load_suites(), "dpg_bench_diffusion_image"
    )
    model = {
        "name": "flux-schnell-l0",
        "manifest": "tests/e2e/models/flux/manifests/flux-schnell-l0.json",
        "hf_id": "black-forest-labs/FLUX.1-schnell",
        "bundle": "flux-schnell-l0.trtfb",
        "family": "flux",
        "task_eval": {},
    }
    captured: dict = {}

    class Prepared(Exception):
        pass

    def fake_prepare(**kwargs):
        captured.update(kwargs["validation_config"])
        raise Prepared

    monkeypatch.setattr(validation_engine, "prepare_task_dataset", fake_prepare)

    try:
        validation_engine.eval_one_model(
            suite=suite,
            model=model,
            args=argparse.Namespace(
                work_root=str(tmp_path / "work"),
                dataset=str(tmp_path / "PartiPrompts.tsv"),
                limit=1,
                subject="",
                sample_seed=None,
            ),
        )
    except Prepared:
        pass
    else:
        raise AssertionError("expected prepare sentinel")

    assert captured["model_manifest"] == model["manifest"]
    assert captured["family"] == "flux"


def test_flux_validation_build_command_preserves_diffusion_shape(tmp_path: Path) -> None:
    model = next(
        model
        for model in validation_engine.load_manifest_records()
        if model["name"] == "flux-schnell-l0"
    )

    command = validation_engine.build_bundle_command(
        model,
        trtmc_binary="build/trtmc",
        bundle_path=tmp_path / "flux-schnell-l0.trtfb",
    )

    assert command[command.index("--image-height") + 1] == "384"
    assert command[command.index("--image-width") + 1] == "384"
    assert command[command.index("--video-num-frames") + 1] == "1"
    assert command[command.index("--num-inference-steps") + 1] == "20"


def test_flux_fp8_build_command_resolves_model_owned_scales(tmp_path: Path) -> None:
    model = next(
        model
        for model in validation_engine.load_manifest_records()
        if model["name"] == "flux-2-dev-fp8"
    )

    command = validation_engine.build_bundle_command(
        model,
        trtmc_binary="build/trtmc",
        bundle_path=tmp_path / "flux-2-dev-fp8.trtfb",
    )

    scales = Path(command[command.index("--fp8-scales") + 1])
    assert scales == (
        validation_engine.REPO_ROOT
        / "tests/e2e/models/flux/data/flux2-fp8-scales.json"
    )
    assert scales.is_file()


def test_eval_one_model_diffusion_uses_clip_parity_summary(
    tmp_path: Path, monkeypatch
) -> None:
    dataset = tmp_path / "dpg_bench.json"
    dataset.write_text(json.dumps({"dataset": "DPG-Bench", "requests": [{
        "sample_id": "dpg_bench_000000",
        "prompt": "a red cube above a blue sphere",
        "category": "entity,relation",
        "questions": [{"question": "Is the red cube above the blue sphere?"}],
    }]}), encoding="utf-8")
    suite = validation_engine.suite_by_id(
        validation_engine.load_suites(), "dpg_bench_diffusion_image"
    )
    model = next(
        model
        for model in validation_engine.load_manifest_records()
        if model["name"] == "flux-schnell-l0"
    )

    def fake_hf(_args, _model, work_dir):
        validation_engine.write_predictions(work_dir / "hf_predictions.json", [{
            "sample_id": "dpg_bench_000000", "returncode": 0, "num_frames": 1
        }])

    def fake_bundle(*_args, **kwargs):
        return kwargs["bundle_path"], True

    def fake_trt(args):
        validation_engine.write_predictions(Path(args.work_dir) / "trtfb_predictions.json", [{
            "sample_id": "dpg_bench_000000", "returncode": 0, "num_frames": 1
        }])

    def fake_compare(_hf, _trt, _answers, *, work_dir, gates):
        assert work_dir.name == "flux-schnell-l0"
        assert "max_prompt_clipscore_drop" not in gates
        assert "min_hf_prompt_clipscore" not in gates
        assert gates["psnr"] == 5.0
        assert gates["ssim"] == 0.1
        return {
            "mode": "diffusion_image_clip_parity",
            "overall_pass_rate": 1.0,
            "passed_count": 1,
            "valid_count": 1,
            "skipped_count": 0,
            "total_count": 1,
            "metrics": {},
            "samples": [],
        }

    monkeypatch.setattr(validation_engine, "run_hf_reference_subprocess", fake_hf)
    monkeypatch.setattr(validation_engine, "ensure_bundle", fake_bundle)
    monkeypatch.setattr(validation_engine, "run_trtfb", fake_trt)
    monkeypatch.setattr(validation_engine, "compare_diffusion_image_predictions", fake_compare)
    monkeypatch.setattr(
        validation_engine,
        "max_prompt_token_length",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("not for diffusion")),
    )
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
        model=["flux-schnell-l0"],
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
        seed=None,
        benchmark_binary="build/trtmc_dataset_benchmark",
        hf_python="",
        backend_dir="",
        kv_cache_size="",
        config="",
        set=[],
        cuda_visible_devices="",
        chat_template=False,
    )

    result = validation_engine.eval_one_model(suite=suite, model=model, args=args)

    assert result["mode"] == "diffusion_image_clip_parity"
    assert result["overall_pass_rate"] == 1.0
    assert result["bundle_built"] is True


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

    monkeypatch.setattr(validation_engine.subprocess, "run", fake_run)
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

    validation_engine.run_asr_trtfb(args)

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

    model = validation_engine._load_vlm_model(Transformers, "org/model", {})

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
        validation_engine._vlm_chat_text(
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

    validation_engine._run_deepseek_ocr_hf_reference(
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


def test_eval_one_model_reuses_cached_hf_builds_bundle_and_reruns_trtfb(
    tmp_path: Path, monkeypatch
) -> None:
    dataset = tmp_path / "mmlu.json"
    _write_mmlu(dataset)
    suite = validation_engine.suite_by_id(validation_engine.load_suites(), "mmlu_five_shot_mcq")
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
    validation_engine.prepare_mmlu_dataset(
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

    def fake_run_hf(_args, _model, reference_work_dir):
        calls.append("hf-cache")
        (reference_work_dir / "hf_cache.json").write_text(
            json.dumps({"status": "reused", "key": "abc123"}),
            encoding="utf-8",
        )

    monkeypatch.setattr(validation_engine, "max_prompt_token_length", lambda **_kwargs: 405)

    def fake_ensure_bundle(*_args, **kwargs):
        calls.append("build")
        assert kwargs["max_cache_length"] == 406
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

    monkeypatch.setattr(validation_engine, "run_hf_reference_subprocess", fake_run_hf)
    monkeypatch.setattr(validation_engine, "ensure_bundle", fake_ensure_bundle)
    monkeypatch.setattr(validation_engine, "run_trtfb", fake_run_trtfb)

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

    result = validation_engine.eval_one_model(suite=suite, model=model, args=args)

    assert calls == ["hf-cache", "build", "trtfb-seed=123"]
    assert result["hf_reused"] is True
    assert result["hf_cache_key"] == "abc123"
    assert result["bundle_built"] is True
    assert result["max_prompt_tokens"] == 405
    assert result["generation_cache_headroom"] == 1
    assert result["build_max_cache_length"] == 406
    assert result["trtfb_accuracy"] == 0.5
    assert result["prediction_agreement_rate"] == 0.5
    assert result["status"] == "failed"
    assert result["error_type"] == "BenchmarkGateError"
    assert result["gate_failures"] == [
        {
            "gate": "max_accuracy_drop_from_hf",
            "metric": "accuracy_drop_from_hf",
            "actual": 0.5,
            "required": 0.01,
        },
        {
            "gate": "min_prediction_agreement",
            "metric": "prediction_agreement_rate",
            "actual": 0.5,
            "required": 0.98,
        }
    ]
    assert (work_dir / "summary.json").is_file()


def test_eval_one_model_uses_vlm_prepare_outputs_for_vlm_suite(tmp_path: Path, monkeypatch) -> None:
    dataset_dir = tmp_path / "MMMU_Pro_vision"
    dataset_dir.mkdir()
    dataset = dataset_dir / "mmmu_pro_vision_dataset.json"
    _write_vlm_mmmu_pro_vision(dataset)
    suite = validation_engine.suite_by_id(validation_engine.load_suites(), "vlm_mmmu_pro_vision_mcq")
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
        prompts = validation_engine.load_jsonl(work_dir / "prompts.jsonl")
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
        prompts = validation_engine.load_jsonl(Path(args.work_dir) / "prompts.jsonl")
        assert prompts[0]["images"] == [str(dataset_dir / "images" / "sample.jpg")]
        Path(args.work_dir, "trtfb_predictions.json").write_text(
            json.dumps({"responses": [{"sample_id": "test_case_1", "output_text": "Answer: J"}]}),
            encoding="utf-8",
        )

    monkeypatch.setattr(validation_engine, "run_hf_reference_subprocess", fake_run_hf)
    monkeypatch.setattr(validation_engine, "ensure_bundle", fake_ensure_bundle)
    monkeypatch.setattr(validation_engine, "run_trtfb", fake_run_trtfb)
    monkeypatch.setattr(validation_engine, "max_prompt_token_length", lambda **_kwargs: 128)

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

    result = validation_engine.eval_one_model(suite=suite, model=model, args=args)

    assert calls == ["hf", "build", "trtfb"]
    assert result["trtfb_accuracy"] == 1.0
    assert result["prediction_agreement_rate"] == 1.0
    assert result["status"] == "passed"
    assert result["gate_failures"] == []


def test_prediction_agreement_gate_reports_accuracy_drop() -> None:
    result = validation_engine.prediction_agreement_gate_result(
        {
            "hf": {"overall_accuracy": 0.8},
            "trtfb": {"overall_accuracy": 0.6},
            "prediction_agreement_rate": 1.0,
        },
        {
            "max_accuracy_drop_from_hf": 0.02,
            "min_prediction_agreement": 0.95,
        },
    )

    assert result["status"] == "failed"
    assert result["error_type"] == "BenchmarkGateError"
    assert result["gate_failures"] == [
        {
            "gate": "max_accuracy_drop_from_hf",
            "metric": "accuracy_drop_from_hf",
            "actual": pytest.approx(0.2),
            "required": 0.02,
        }
    ]


def test_eval_one_model_skips_prompt_length_check_for_asr_suite(
    tmp_path: Path, monkeypatch
) -> None:
    dataset_dir = tmp_path / "librispeech_clean_test"
    dataset_dir.mkdir()
    dataset = dataset_dir / "librispeech_clean_test.json"
    _write_asr_librispeech(dataset)
    suite = validation_engine.suite_by_id(validation_engine.load_suites(), "librispeech_clean_asr")
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
        prompts = validation_engine.load_jsonl(Path(args.work_dir) / "prompts.jsonl")
        assert prompts[0]["audio"].endswith("clean_000000.wav")
        Path(args.work_dir, "trtfb_predictions.json").write_text(
            json.dumps(
                {"responses": [{"sample_id": "clean_000000", "output_text": "the quick brown fox"}]}
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(validation_engine, "max_prompt_token_length", fake_prompt_length)
    monkeypatch.setattr(validation_engine, "run_hf_reference_subprocess", fake_run_hf)
    monkeypatch.setattr(validation_engine, "ensure_bundle", fake_ensure_bundle)
    monkeypatch.setattr(validation_engine, "run_trtfb", fake_run_trtfb)

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

    result = validation_engine.eval_one_model(suite=suite, model=model, args=args)

    assert calls == ["hf", "build", "trtfb"]
    assert result["mode"] == "asr_transcript"
    assert result["max_prompt_tokens"] is None
    assert result["hf_accuracy"] == 1.0
    assert result["trtfb_accuracy"] == 1.0
    assert result["prediction_agreement_rate"] == 1.0
    assert result["status"] == "passed"
    assert result["gates"] == {
        "max_accuracy_drop_from_hf": 0.05,
        "min_prediction_agreement": 0.9,
    }


def test_eval_one_model_runs_hf_for_golden_snapshot_vlm_model(tmp_path: Path, monkeypatch) -> None:
    dataset_dir = tmp_path / "OCRBench_v2" / "unified"
    dataset_dir.mkdir(parents=True)
    dataset = dataset_dir / "dataset.json"
    _write_ocrbench_unified(dataset)
    suite = validation_engine.suite_by_id(validation_engine.load_suites(), "ocrbench_v2_unified")
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
        prompts = validation_engine.load_jsonl(Path(args.work_dir) / "prompts.jsonl")
        assert prompts[0]["images"] == [str(dataset_dir / "images" / "ocrbench_v2_000000.jpg")]
        Path(args.work_dir, "trtfb_predictions.json").write_text(
            json.dumps({"responses": [{"sample_id": "ocrbench_v2_000000", "output_text": "on"}]}),
            encoding="utf-8",
        )

    monkeypatch.setattr(validation_engine, "run_hf_reference_subprocess", fake_run_hf)
    monkeypatch.setattr(validation_engine, "ensure_bundle", fake_ensure_bundle)
    monkeypatch.setattr(validation_engine, "run_trtfb", fake_run_trtfb)

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

    result = validation_engine.eval_one_model(suite=suite, model=model, args=args)

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

    monkeypatch.setattr(validation_engine, "load_suites", lambda *_args, **_kwargs: [suite])
    monkeypatch.setattr(validation_engine, "load_manifest_records", lambda *_args, **_kwargs: models)
    monkeypatch.setattr(
        validation_engine,
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

    monkeypatch.setattr(validation_engine, "eval_one_model", fake_eval_one_model)

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

    assert validation_engine.cmd_eval(args) == 0

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


def test_eval_preserves_failed_diffusion_gate_status(tmp_path: Path, monkeypatch) -> None:
    suite = {
        "id": "dpg_bench_diffusion_image",
        "dataset": {"kind": "diffusion_prompt_tsv"},
    }
    model = {"name": "pixart", "hf_id": "org/pixart", "bundle": "pixart.trtfb"}
    monkeypatch.setattr(validation_engine, "load_suites", lambda *_args, **_kwargs: [suite])
    monkeypatch.setattr(validation_engine, "load_manifest_records", lambda *_args, **_kwargs: [model])
    monkeypatch.setattr(
        validation_engine,
        "selected_models_for_suite",
        lambda *_args, **_kwargs: [model],
    )
    monkeypatch.setattr(
        validation_engine,
        "eval_one_model",
        lambda **_kwargs: {
            "suite": suite["id"],
            "model": model["name"],
            "mode": "diffusion_image_clip_parity",
            "overall_pass_rate": 0.0,
            "passed_count": 0,
            "valid_count": 10,
            "skipped_count": 0,
            "hf_reused": False,
            "bundle_built": False,
            "status": "failed",
        },
    )
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

    assert validation_engine.cmd_eval(args) == 0

    summary = json.loads(
        (tmp_path / "work" / suite["id"] / "eval_summary.json").read_text()
    )
    assert summary["passed_count"] == 0
    assert summary["failed_count"] == 1
    assert summary["results"][0]["status"] == "failed"


def test_eval_parser_accepts_explicit_model_plugin_dir() -> None:
    args = validation_engine.build_arg_parser().parse_args([
        "eval",
        "--suite",
        "dpg_bench_diffusion_image",
        "--work-root",
        "/work",
        "--model-plugin-dir",
        "/runtime/models/pixart",
    ])

    assert args.model_plugin_dir == "/runtime/models/pixart"


def test_eval_accepts_reranking_dataset_kind(tmp_path: Path, monkeypatch) -> None:
    suite = {"id": "beir_scifact_reranking", "dataset": {"kind": "reranking_json"}}
    model = {
        "name": "reranker",
        "hf_id": "org/reranker",
        "bundle": "reranker.trtfb",
    }
    monkeypatch.setattr(validation_engine, "load_suites", lambda *_args, **_kwargs: [suite])
    monkeypatch.setattr(validation_engine, "load_manifest_records", lambda *_args, **_kwargs: [model])
    monkeypatch.setattr(
        validation_engine,
        "selected_models_for_suite",
        lambda *_args, **_kwargs: [model],
    )
    monkeypatch.setattr(
        validation_engine,
        "eval_one_model",
        lambda **_kwargs: {
            "suite": suite["id"],
            "model": model["name"],
            "mode": "reranking_parity",
            "sample_pass_rate": 1.0,
            "mean_pairwise_ordering_agreement": 1.0,
            "min_pairwise_ordering_agreement": 1.0,
            "hf_reused": False,
            "bundle_built": False,
        },
    )
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

    assert validation_engine.cmd_eval(args) == 0


def test_eval_accepts_model_plugin_dataset_kind(tmp_path: Path, monkeypatch) -> None:
    suite = {
        "id": "mmmu_pro_vision_plugin_parity",
        "dataset": {"kind": "model_plugin_json"},
    }
    model = {
        "name": "lance",
        "hf_id": "org/lance",
        "bundle": "lance.trtfb",
    }
    monkeypatch.setattr(validation_engine, "load_suites", lambda *_args, **_kwargs: [suite])
    monkeypatch.setattr(validation_engine, "load_manifest_records", lambda *_args, **_kwargs: [model])
    monkeypatch.setattr(
        validation_engine,
        "selected_models_for_suite",
        lambda *_args, **_kwargs: [model],
    )
    monkeypatch.setattr(
        validation_engine,
        "eval_one_model",
        lambda **_kwargs: {
            "suite": suite["id"],
            "model": model["name"],
            "mode": "model_plugin_parity",
            "sample_pass_rate": 1.0,
            "passed_count": 1,
            "valid_count": 1,
            "hf_reused": False,
            "bundle_built": False,
        },
    )
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

    assert validation_engine.cmd_eval(args) == 0


def test_eval_stops_after_oom_when_gpu_cleanup_is_not_confirmed(
    tmp_path: Path, monkeypatch
) -> None:
    suite = {"id": "mmlu_five_shot_mcq", "dataset": {"kind": "mmlu_five_shot_json"}}
    models = [
        {"name": "oom", "hf_id": "org/oom", "bundle": "oom.trtfb"},
        {"name": "next", "hf_id": "org/next", "bundle": "next.trtfb"},
    ]

    monkeypatch.setattr(validation_engine, "load_suites", lambda *_args, **_kwargs: [suite])
    monkeypatch.setattr(validation_engine, "load_manifest_records", lambda *_args, **_kwargs: models)
    monkeypatch.setattr(
        validation_engine,
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

    monkeypatch.setattr(validation_engine, "run_eval_model_worker", fake_run_worker)

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

    assert validation_engine.cmd_eval(args) == 0

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


def _write_conditional_text_jsonl(path: Path) -> None:
    rows = [
        {"id": "sample-1", "input": "Quelle eins", "output": "Reference one", "subset": "test"},
        {"id": "sample-2", "input": "Quelle zwei", "output": "Reference two", "subset": "test"},
    ]
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_unconditional_text_requests(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "dataset": "ELF OpenWebText generation",
                "requests": [
                    {"id": "owt-0", "seed": 42},
                    {"id": "owt-1", "seed": 43},
                ],
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("suite_id", "dataset_kind", "model_name", "task_metric"),
    [
        ("wmt14_de_en_elf_diffusion_text", "conditional_text_jsonl", "elf-b-de-en-l0", "sacrebleu"),
        ("xsum_elf_diffusion_text", "conditional_text_jsonl", "elf-b-xsum-l0", "rouge"),
        (
            "openwebtext_elf_diffusion_text",
            "unconditional_text_json",
            "elf-b-owt-l0",
            "unconditional_text_quality",
        ),
    ],
)
def test_default_suites_include_elf_diffusion_text_tasks(
    suite_id: str, dataset_kind: str, model_name: str, task_metric: str
) -> None:
    suites = validation_engine.load_suites()
    suite = validation_engine.suite_by_id(suites, suite_id)
    models = validation_engine.load_manifest_records()
    selected = validation_engine.selected_models_for_suite(suite, models)

    assert suite["dataset"]["kind"] == dataset_kind
    assert suite["selectors"]["model_names"] == [model_name]
    assert suite["selectors"]["task_strategies"] == ["diffusion_text_generation"]
    assert [model["name"] for model in selected] == [model_name]
    assert suite["reference"]["mode"] == "hf_elf_torch"
    assert suite["reference"]["checkpoint"].endswith("-torch")
    assert suite["scoring"]["scorer"] == "diffusion_text_parity"
    assert suite["scoring"]["task_metric"] == task_metric
    assert "parity" not in suite


def test_prepare_conditional_text_dataset_preserves_sources_and_gold(tmp_path: Path) -> None:
    dataset = tmp_path / "conditional.jsonl"
    _write_conditional_text_jsonl(dataset)
    suite = validation_engine.suite_by_id(validation_engine.load_suites(), "wmt14_de_en_elf_diffusion_text")

    outputs = validation_engine.prepare_conditional_text_jsonl_dataset(
        dataset_path=dataset,
        work_dir=tmp_path / "work",
        suite=suite,
        limit=1,
    )

    answers = json.loads(outputs["answers"].read_text(encoding="utf-8"))
    prompts = validation_engine.load_jsonl(outputs["prompts"])
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert answers["requests"][0]["answer"] == "Reference one"
    assert prompts[0]["source_text"] == "Quelle eins"
    assert prompts[0]["sample_id"] == "sample-1"
    assert manifest["reference_mode"] == "hf_elf_torch"
    assert manifest["generation"]["num_sampling_steps"] == 64


def test_prepare_unconditional_text_dataset_preserves_request_seeds(tmp_path: Path) -> None:
    dataset = tmp_path / "owt.json"
    _write_unconditional_text_requests(dataset)
    suite = validation_engine.suite_by_id(validation_engine.load_suites(), "openwebtext_elf_diffusion_text")

    outputs = validation_engine.prepare_unconditional_text_dataset(
        dataset_path=dataset,
        work_dir=tmp_path / "work",
        suite=suite,
    )

    prompts = validation_engine.load_jsonl(outputs["prompts"])
    assert [(row["sample_id"], row["seed"], row["prompt"]) for row in prompts] == [
        ("owt-0", 42, ""),
        ("owt-1", 43, ""),
    ]


def test_diffusion_text_runner_replaces_e2e_replay_with_hf_shared_inputs(
    tmp_path: Path, monkeypatch
) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_kind": "conditional_text_jsonl",
                "generation": {"num_sampling_steps": 64, "seed": 42},
                "task_eval": {"model_manifest": "unused.json"},
            }
        ),
        encoding="utf-8",
    )
    (work_dir / "prompts.jsonl").write_text(
        json.dumps(
            {
                "sample_id": "sample-1",
                "dataset_index": 7,
                "source_text": "Der echte Eingabetext.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    shared_dir = work_dir / "hf_shared_inputs" / "sample-1"
    shared_dir.mkdir(parents=True)
    (shared_dir / "initial_latents.f32").write_bytes(b"latents")
    (shared_dir / "sampling_steps.f32").write_bytes(b"steps")
    template = SimpleNamespace(
        name="template",
        bundle="template.trtfb",
        inputs={
            "elf_replay_artifact": "fixed-replay.json",
            "condition_latents_raw": "fixed-condition.f32",
            "initial_latents_raw": "fixed-initial.f32",
            "expected_generated_samples": [{"generated": "fixed"}],
        },
    )
    captured_inputs: list[dict] = []

    class FakeRunner:
        def run_stage(self, case, stage, context):
            captured_inputs.append(dict(case.inputs))
            return SimpleNamespace(
                data={"generated_samples": [{"generated": "Real output", "token_ids": [1, 2]}]},
                text="Real output",
                timing_s=0.01,
            )

    monkeypatch.setattr(
        validation_engine,
        "_load_diffusion_text_validation_runner",
        lambda _work_dir: (template, FakeRunner()),
    )

    validation_engine.run_diffusion_text_trtfb(
        SimpleNamespace(
            work_dir=str(work_dir),
            raw_output="",
            predictions="",
            bundle=str(tmp_path / "elf.trtfb"),
            trtmc_binary="trtmc",
            hf_python="python",
            model_plugin_dir="",
        )
    )

    assert captured_inputs[0]["source_text"] == "Der echte Eingabetext."
    assert captured_inputs[0]["seed"] == 49
    assert "elf_replay_artifact" not in captured_inputs[0]
    assert "condition_latents_raw" not in captured_inputs[0]
    assert "expected_generated_samples" not in captured_inputs[0]
    assert captured_inputs[0]["initial_latents_raw"] == str(shared_dir / "initial_latents.f32")
    assert captured_inputs[0]["sampling_steps_raw"] == str(shared_dir / "sampling_steps.f32")


def test_diffusion_text_scores_gold_and_unconditional_quality(monkeypatch) -> None:
    predictions = {
        "responses": [
            {"sample_id": "a", "output_text": "the cat sat", "generated_token_ids": [1, 2, 3]},
            {"sample_id": "b", "output_text": "a cat sat", "generated_token_ids": [1, 2, 2]},
        ]
    }
    answers = {
        "requests": [
            {"sample_id": "a", "answer": "the cat sat"},
            {"sample_id": "b", "answer": "the cat slept"},
        ]
    }
    fake_bleu = SimpleNamespace(
        score=31.5, bp=0.9, sys_len=6, ref_len=6, precisions=[50.0, 25.0, 0.0, 0.0]
    )
    monkeypatch.setitem(
        sys.modules,
        "sacrebleu",
        SimpleNamespace(corpus_bleu=lambda hypotheses, references: fake_bleu),
    )

    class FakeRougeScorer:
        def __init__(self, metrics, use_stemmer):
            assert metrics == ["rouge1", "rouge2", "rougeL"]
            assert use_stemmer is True

        def score(self, reference, prediction):
            return {
                "rouge1": SimpleNamespace(fmeasure=0.75),
                "rouge2": SimpleNamespace(fmeasure=0.50),
                "rougeL": SimpleNamespace(fmeasure=0.625),
            }

    monkeypatch.setitem(
        sys.modules,
        "rouge_score",
        SimpleNamespace(rouge_scorer=SimpleNamespace(RougeScorer=FakeRougeScorer)),
    )

    bleu = validation_engine.score_sacrebleu_predictions(predictions, answers)
    rouge = validation_engine.score_rouge_predictions(predictions, answers)
    quality = validation_engine.score_unconditional_text_predictions(
        predictions,
        {"requests": [{"sample_id": "a"}, {"sample_id": "b"}]},
        generation_ppl=24.1,
        unigram_entropy=5.15,
    )

    assert bleu["corpus_bleu"] == 31.5
    assert bleu["non_empty_rate"] == 1.0
    assert rouge["rouge1"] > rouge["rouge2"]
    assert rouge["rouge_l"] == 0.625
    assert quality["generation_ppl"] == 24.1
    assert quality["unigram_entropy"] == 5.15
    assert quality["distinct_1"] == 0.5


def test_continuation_translation_reports_bleu_without_gating_task_quality(
    monkeypatch,
) -> None:
    answers = {
        "requests": [
            {"sample_id": "translation_0", "answer": "Bonjour.", "subject": "en-fr"}
        ]
    }
    hf = {
        "responses": [
            {
                "sample_id": "translation_0",
                "output_text": "Bonjour.",
                "generated_token_ids": [1, 2],
            }
        ]
    }
    trtfb = {
        "responses": [
            {
                "sample_id": "translation_0",
                "output_text": "Salut.",
                "generated_token_ids": [3, 4],
            }
        ]
    }
    scores = iter(
        [
            SimpleNamespace(
                score=42.0, bp=1.0, sys_len=2, ref_len=2, precisions=[100.0]
            ),
            SimpleNamespace(
                score=37.5, bp=1.0, sys_len=2, ref_len=2, precisions=[75.0]
            ),
        ]
    )
    monkeypatch.setitem(
        sys.modules,
        "sacrebleu",
        SimpleNamespace(corpus_bleu=lambda _hypotheses, _references: next(scores)),
    )

    diagnostics = validation_engine.continuation_task_quality_diagnostics(
        "sacrebleu", hf, trtfb, answers
    )

    assert diagnostics["hf_corpus_bleu"] == 42.0
    assert diagnostics["trtfb_corpus_bleu"] == 37.5
    assert diagnostics["corpus_bleu_abs_delta"] == 4.5
    assert validation_engine.continuation_task_quality_diagnostics("", hf, trtfb, answers) == {}


def test_diffusion_text_hf_parity_uses_token_agreement_only() -> None:
    result = validation_engine.compare_diffusion_text_prediction_sets(
        {
            "responses": [
                {
                    "sample_id": "sample",
                    "output_text": "Change Good",
                    "generated_token_ids": [483, 1804],
                    "shared_sampling_inputs": {"initial_latents": "/tmp/shared.f32"},
                }
            ]
        },
        {
            "responses": [
                {
                    "sample_id": "sample",
                    "output_text": "change  good",
                    "generated_token_ids": [5968, 1804],
                    "shared_sampling_inputs": {"initial_latents": "/tmp/shared.f32"},
                }
            ]
        },
    )

    assert result["token_agreement_rate"] == 0.5
    assert result["shared_sampling_inputs_match_rate"] == 1.0
    assert "normalized_text_exact_match_rate" not in result
    assert "text_ned" not in result["samples"][0]


def test_diffusion_text_shared_inputs_match_through_reference_cache_symlink(
    tmp_path: Path,
) -> None:
    cached_dir = tmp_path / "cache" / "hf_shared_inputs" / "sample"
    cached_dir.mkdir(parents=True)
    cached_input = cached_dir / "initial_latents.f32"
    cached_input.write_bytes(b"latents")
    materialized_root = tmp_path / "work" / "hf_shared_inputs"
    materialized_root.parent.mkdir()
    materialized_root.symlink_to(cached_dir.parent, target_is_directory=True)
    materialized_input = materialized_root / "sample" / "initial_latents.f32"

    result = validation_engine.compare_diffusion_text_prediction_sets(
        {
            "responses": [
                {
                    "sample_id": "sample",
                    "generated_token_ids": [1],
                    "shared_sampling_inputs": {
                        "initial_latents": str(cached_input),
                    },
                }
            ]
        },
        {
            "responses": [
                {
                    "sample_id": "sample",
                    "generated_token_ids": [1],
                    "shared_sampling_inputs": {
                        "initial_latents": str(materialized_input),
                    },
                }
            ]
        },
    )

    assert result["shared_sampling_inputs_match_rate"] == 1.0


def test_diffusion_text_hf_parity_requires_token_ids() -> None:
    with pytest.raises(ValueError, match="must contain token_ids or generated_token_ids"):
        validation_engine.compare_diffusion_text_prediction_sets(
            {
                "responses": [
                    {
                        "sample_id": "sample",
                        "output_text": "HF output",
                        "shared_sampling_inputs": {"initial_latents": "/tmp/shared.f32"},
                    }
                ]
            },
            {
                "responses": [
                    {
                        "sample_id": "sample",
                        "output_text": "TRT output",
                        "generated_token_ids": [1, 2],
                        "shared_sampling_inputs": {"initial_latents": "/tmp/shared.f32"},
                    }
                ]
            },
        )


@pytest.mark.parametrize(
    ("task_metric", "diagnostics", "expected"),
    [
        (
            "sacrebleu",
            {"hf_corpus_bleu": 26.4, "trtfb_corpus_bleu": 26.1},
            {"corpus_bleu_abs_delta": pytest.approx(0.3)},
        ),
        (
            "rouge",
            {
                "hf_rouge1": 0.36,
                "trtfb_rouge1": 0.355,
                "hf_rouge2": 0.122,
                "trtfb_rouge2": 0.120,
                "hf_rouge_l": 0.278,
                "trtfb_rouge_l": 0.281,
            },
            {
                "rouge1_abs_delta": pytest.approx(0.005),
                "rouge2_abs_delta": pytest.approx(0.002),
                "rouge_l_abs_delta": pytest.approx(0.003),
            },
        ),
        (
            "unconditional_text_quality",
            {
                "hf_generation_ppl": 24.2,
                "trtfb_generation_ppl": 23.9,
                "hf_unigram_entropy": 5.12,
                "trtfb_unigram_entropy": 5.10,
            },
            {
                "generation_ppl_abs_delta": pytest.approx(0.3),
                "unigram_entropy_abs_delta": pytest.approx(0.02),
            },
        ),
    ],
)
def test_diffusion_text_task_metric_deltas(
    task_metric: str,
    diagnostics: dict[str, float],
    expected: dict[str, float],
) -> None:
    assert validation_engine.diffusion_text_task_metric_deltas(task_metric, diagnostics) == expected


def test_metric_gates_fail_on_missing_or_out_of_range_metrics() -> None:
    result = {"corpus_bleu": 19.0, "non_empty_rate": 1.0}

    validation_engine.apply_metric_gates(
        result,
        {"min_corpus_bleu": 20.0, "min_non_empty_rate": 0.99, "max_generation_ppl": 40.0},
    )

    assert result["status"] == "failed"
    assert [failure["gate"] for failure in result["gate_failures"]] == [
        "min_corpus_bleu",
        "max_generation_ppl",
    ]


def _ci_suite(*, digest: str = "a" * 64) -> dict:
    return {
        "id": "ci_suite",
        "default_model_names": ["chronos", "timesfm"],
        "dataset": {
            "default_path": "/missing/ETTh1.csv",
            "source": "https://example.com/ETTh1.csv",
            "sha256": digest,
        },
        "ci": {
            "eligible": True,
            "lane": "nightly",
            "limit": 10,
            "sample_seed": 20260715,
        },
    }


def test_real_etth1_suite_is_nightly_ci_eligible() -> None:
    suite = validation_engine.suite_by_id(validation_engine.load_suites(), "etth1_time_series_parity")

    ci = validation_engine.validate_ci_suite(suite, "nightly")

    assert ci["limit"] == 10
    assert ci["sample_seed"] == 20260715
    assert len(suite["default_model_names"]) == 5


def test_validate_ci_suite_rejects_wrong_lane() -> None:
    with pytest.raises(ValueError, match="belongs to lane"):
        validation_engine.validate_ci_suite(_ci_suite(), "premerge")


def test_ensure_ci_dataset_downloads_and_verifies_pinned_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b"date,OT\n2026-01-01,1.0\n"
    digest = hashlib.sha256(content).hexdigest()
    monkeypatch.setattr(
        validation_engine.urllib.request,
        "urlopen",
        lambda _source, timeout: io.BytesIO(content),
    )

    dataset = validation_engine.ensure_ci_dataset(
        _ci_suite(digest=digest), explicit_path=None, cache_root=tmp_path / "cache"
    )

    assert dataset.read_bytes() == content
    assert validation_engine._sha256_file(dataset) == digest


def test_ensure_ci_dataset_rejects_explicit_checksum_mismatch(tmp_path: Path) -> None:
    dataset = tmp_path / "ETTh1.csv"
    dataset.write_text("wrong", encoding="utf-8")

    with pytest.raises(ValueError, match="wrong checksum"):
        validation_engine.ensure_ci_dataset(
            _ci_suite(), explicit_path=dataset, cache_root=tmp_path / "cache"
        )


def test_configure_ci_eval_uses_suite_models_limit_seed_and_dataset(tmp_path: Path) -> None:
    dataset = tmp_path / "ETTh1.csv"
    dataset.write_text("date,OT\n2026-01-01,1.0\n", encoding="utf-8")
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    args = argparse.Namespace(
        ci_lane="nightly",
        model=[],
        limit=0,
        sample_seed=None,
        single_device_only=False,
        local_files_only=False,
        waive_platform="",
        engine_dir=str(tmp_path / "engines"),
        dataset=str(dataset),
        dataset_cache_root=str(tmp_path / "cache"),
    )

    expected_models = validation_engine.configure_ci_eval(args, _ci_suite(digest=digest))

    assert expected_models == ["chronos", "timesfm"]
    assert args.model == ["chronos", "timesfm"]
    assert args.limit == 10
    assert args.sample_seed == 20260715
    assert args.single_device_only is True
    assert args.local_files_only is True
    assert args.waive_platform == "GB300"


def test_configure_ci_eval_allows_a_fail_closed_model_subset(tmp_path: Path) -> None:
    dataset = tmp_path / "ETTh1.csv"
    dataset.write_text("date,OT\n2026-01-01,1.0\n", encoding="utf-8")
    args = argparse.Namespace(
        ci_lane="nightly",
        model=["timesfm"],
        limit=0,
        sample_seed=None,
        single_device_only=False,
        local_files_only=False,
        waive_platform="",
        engine_dir=str(tmp_path / "engines"),
        dataset=str(dataset),
        dataset_cache_root=str(tmp_path / "cache"),
    )

    expected_models = validation_engine.configure_ci_eval(
        args,
        _ci_suite(digest=hashlib.sha256(dataset.read_bytes()).hexdigest()),
    )

    assert expected_models == ["timesfm"]
    assert args.model == ["timesfm"]


def test_validate_eval_summary_fails_closed_on_failed_model() -> None:
    passed, results = validation_engine.validate_eval_summary(
        {
            "results": [
                {"model": "chronos", "status": "passed"},
                {"model": "timesfm", "status": "failed"},
            ]
        },
        ["chronos", "timesfm"],
    )

    assert passed is False
    assert len(results) == 2


def test_public_ci_artifacts_omit_private_runner_paths(tmp_path: Path) -> None:
    work_root = tmp_path / "private-work"
    numeric = work_root / "ci_suite" / "chronos" / "summary.json"
    numeric.parent.mkdir(parents=True)
    numeric.write_text(
        '{"status":"passed","cases":[],"private_path":"/private/numeric"}\n',
        encoding="utf-8",
    )
    results = [
        {
            "suite": "ci_suite",
            "model": "chronos",
            "status": "passed",
            "sample_agreement_rate": 1.0,
            "work_dir": "/private/runner/work",
            "bundle": "/private/runner/engine.trtfb",
        },
        {"model": "timesfm", "status": "failed", "error": "/private/error"},
    ]
    artifact_dir = tmp_path / "public"

    validation_engine.write_public_ci_artifacts(
        suite=_ci_suite(),
        expected_models=["chronos", "timesfm"],
        results=results,
        work_root=work_root,
        artifact_dir=artifact_dir,
    )

    public = (artifact_dir / "eval_summary.json").read_text(encoding="utf-8")
    assert "/private" not in public
    assert "work_dir" not in public
    assert "bundle" not in public
    numeric_public = artifact_dir / "models" / "chronos" / "summary.json"
    assert "/private" not in numeric_public.read_text(encoding="utf-8")


def test_prepare_vbench_selects_ten_unique_review_dimensions(tmp_path: Path) -> None:
    source = tmp_path / "VBench_full_info.json"
    source.write_text(
        json.dumps(
            [
                {
                    "prompt_en": f"official prompt {index}",
                    "dimension": [dimension],
                }
                for index, dimension in enumerate(prepare_media.VBENCH_DIMENSIONS)
            ]
        ),
        encoding="utf-8",
    )

    output = prepare_media.prepare_vbench(source, tmp_path / "out")
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["request_count"] == 10
    assert [row["category"] for row in payload["requests"]] == list(
        prepare_media.VBENCH_DIMENSIONS
    )
    assert len({row["prompt"] for row in payload["requests"]}) == 10
    assert payload["source_info_sha256"]
    assert payload["license"] == "Apache-2.0"


def test_prepare_gedit_writes_task_diverse_static_condition_images(tmp_path: Path) -> None:
    rows = [
        {
            "key": f"sample/{index}",
            "instruction": f"edit instruction {index}",
            "instruction_language": "en",
            "task_type": f"task_{index}",
            "input_image": Image.new("RGB", (40 + index, 30), (index, 20, 30)),
            "Intersection_exist": index % 2 == 0,
        }
        for index in range(10)
    ]

    output = prepare_media.prepare_gedit_rows(rows, tmp_path / "GEdit-Bench")
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["request_count"] == 10
    assert len({row["category"] for row in payload["requests"]}) == 10
    first = payload["requests"][0]
    condition = output.parent / first["image"]
    assert Image.open(condition).size == (1024, 1024)
    assert first["condition_image_sha256"]
    assert payload["license"] == "MIT"
    assert payload["source_revision"] == prepare_media.GEDIT_REVISION


def test_prepare_gedit_loads_local_hf_arrow_checkout(
    tmp_path: Path, monkeypatch: Any
) -> None:
    source = tmp_path / "gedit-source"
    source.mkdir()
    arrow = source / "data-00000-of-00001.arrow"
    arrow.touch()
    rows = [
        {
            "key": f"sample-{index}",
            "instruction": f"edit instruction {index}",
            "instruction_language": "en",
            "task_type": f"task_{index}",
            "input_image": Image.new("RGB", (8, 8)),
        }
        for index in range(10)
    ]
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def fake_load_dataset(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        calls.append((args, kwargs))
        return rows

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = fake_load_dataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    output = prepare_media.prepare_gedit(str(source), tmp_path / "out")

    assert output.is_file()
    assert calls == [
        (("arrow",), {"data_files": [str(arrow.resolve())], "split": "train"})
    ]


def test_prepare_gedit_streams_local_arrow_without_datasets(
    tmp_path: Path, monkeypatch: Any
) -> None:
    pa = pytest.importorskip("pyarrow")
    source = tmp_path / "gedit-source"
    source.mkdir()
    arrow = source / "data-00000-of-00001.arrow"
    rows = []
    for index in range(10):
        encoded = BytesIO()
        Image.new("RGB", (8, 8), (index, 20, 30)).save(encoded, format="PNG")
        rows.append(
            {
                "key": f"sample-{index}",
                "instruction": f"edit instruction {index}",
                "instruction_language": "en",
                "task_type": f"task_{index}",
                "input_image": {"bytes": encoded.getvalue(), "path": None},
            }
        )
    table = pa.Table.from_pylist(rows)
    with pa.OSFile(str(arrow), "wb") as sink:
        with pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
    monkeypatch.setitem(sys.modules, "datasets", None)

    output = prepare_media.prepare_gedit(str(source), tmp_path / "out")
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["request_count"] == 10
    assert len(list((output.parent / "images").glob("*.png"))) == 10


def _write_sana_split(root: Path, split: str, color_offset: int) -> None:
    manifest = root / split / "sanawm_export_v2" / "run_manifest.jsonl"
    manifest.parent.mkdir(parents=True)
    rows = []
    categories = ("game_style", "indoor", "outdoor_city", "outdoor_nature")
    for index in range(12):
        category = categories[index % len(categories)]
        scene_id = f"{category}_{index // len(categories) + 1:03d}"
        image = root / "images" / f"{scene_id}.png"
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 6), (index, color_offset, 20)).save(image)
        camera = root / split / "sanawm_export_v2" / f"{scene_id}.npz"
        intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 961, axis=0)
        intrinsics[:, 0, 0] = 800 + index
        intrinsics[:, 1, 1] = 810 + index
        intrinsics[:, 0, 2] = 640
        intrinsics[:, 1, 2] = 352
        np.savez(
            camera,
            c2w=np.zeros((961, 4, 4), dtype=np.float32),
            intrinsics=intrinsics,
        )
        rows.append(
            {
                "id": scene_id,
                "image_path": f"images/{scene_id}.png",
                "camera_path": f"{split}/sanawm_export_v2/{scene_id}.npz",
                "prompt": f"official scene prompt {scene_id}",
            }
        )
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_prepare_sana_wm_uses_official_scene_assets_with_supported_actions(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    for offset, split in enumerate(prepare_media.SANA_WM_SPLITS):
        _write_sana_split(source, split, offset * 5)

    output = prepare_media.prepare_sana_wm(source, tmp_path / "out")
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["request_count"] == 10
    assert len({row["source_scene_id"] for row in payload["requests"]}) == 10
    assert "not arbitrary official c2w files" in payload["control_limitation"]
    assert payload["license"] == "CC-BY-4.0"
    assert payload["source_revision"] == prepare_media.SANA_WM_REVISION
    assert set(payload["source_manifest_sha256"]) == set(prepare_media.SANA_WM_SPLITS)
    first = payload["requests"][0]
    assert (output.parent / first["image"]).is_file()
    assert (output.parent / first["prompt_file"]).is_file()
    intrinsics = np.load(output.parent / first["camera_intrinsics_file"])
    assert intrinsics.shape == (3, 3)
    for action in (row["action"] for row in payload["requests"]):
        assert sum(int(segment.rsplit("-", 1)[1]) for segment in action.split(",")) == 320
