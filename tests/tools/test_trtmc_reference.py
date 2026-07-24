# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
import stat
import sys
from types import SimpleNamespace

from tools.reference import (
    elf_prepared,
    plugin_reference,
    speech,
    transformers_encoder,
    transformers_text,
    transformers_vlm,
)
from tools import trtmc_reference


def test_tts_transcriber_passes_local_files_only_once(monkeypatch) -> None:
    load_calls: list[tuple[str, str, dict[str, object]]] = []
    pipeline_call: dict[str, object] = {}
    loaded_model = object()
    loaded_processor = SimpleNamespace(
        tokenizer=object(),
        feature_extractor=SimpleNamespace(sampling_rate=16000),
    )

    class FakeTranscriber:
        feature_extractor = SimpleNamespace(sampling_rate=16000)

        def __call__(self, _waveforms, **_kwargs):
            return [{"text": "hello"}]

    class FakeModelLoader:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs):
            load_calls.append(("model", model_id, kwargs))
            return loaded_model

    class FakeProcessorLoader:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs):
            load_calls.append(("processor", model_id, kwargs))
            return loaded_processor

    def fake_pipeline(task: str, **kwargs):
        assert task == "automatic-speech-recognition"
        pipeline_call.update(kwargs)
        return FakeTranscriber()

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModelForSpeechSeq2Seq=FakeModelLoader,
            AutoProcessor=FakeProcessorLoader,
            pipeline=fake_pipeline,
        ),
    )
    monkeypatch.setattr(
        speech,
        "_read_wav_float32",
        lambda _path: ([0.0], 16000),
    )
    monkeypatch.setattr(
        speech,
        "_resample_audio",
        lambda audio, _source_rate, _target_rate: audio,
    )

    result = speech._transcribe_tts(
        SimpleNamespace(device="cuda", local_files_only=True),
        [Path("sample.wav")],
        "openai/whisper-tiny",
    )

    assert result == ["hello"]
    assert load_calls == [
        ("model", "openai/whisper-tiny", {"local_files_only": True}),
        ("processor", "openai/whisper-tiny", {"local_files_only": True}),
    ]
    assert pipeline_call == {
        "model": loaded_model,
        "tokenizer": loaded_processor.tokenizer,
        "feature_extractor": loaded_processor.feature_extractor,
        "device": -1,
    }


def _prepare_work(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "answers.json").write_text(
        json.dumps({"requests": [{"sample_id": "one", "answer": "A"}]}),
        encoding="utf-8",
    )
    (path / "prompts.jsonl").write_text(
        json.dumps({"sample_id": "one", "prompt": "question"}) + "\n",
        encoding="utf-8",
    )
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_kind": "mmlu_json",
                "files": {
                    "answers": str(path / "answers.json"),
                    "prompts": str(path / "prompts.jsonl"),
                },
            }
        ),
        encoding="utf-8",
    )


def _args(work_dir: Path, cache_dir: Path, *extra: str):
    return trtmc_reference.build_parser().parse_args(
        [
            "run",
            "--model",
            "org/model",
            "--family",
            "family",
            "--reference-family",
            "causal",
            "--work-dir",
            str(work_dir),
            "--cache-dir",
            str(cache_dir),
            *extra,
        ]
    )


def test_reference_cache_reuses_same_settings_across_work_directories(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cache_dir = tmp_path / "cache"
    first = tmp_path / "first"
    second = tmp_path / "second"
    _prepare_work(first)
    _prepare_work(second)
    calls: list[Path] = []

    def fake_reference(args) -> None:
        work_dir = Path(args.work_dir)
        calls.append(work_dir)
        artifact = work_dir / "hf_artifacts" / "one.bin"
        artifact.parent.mkdir()
        artifact.write_bytes(b"reference")
        (work_dir / "hf_predictions.json").write_text(
            json.dumps(
                {
                    "responses": [
                        {
                            "sample_id": "one",
                            "output_text": "A",
                            "artifact": str(artifact),
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        (work_dir / "hf_raw.jsonl").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(
        trtmc_reference.task_eval,
        "run_hf_reference",
        fake_reference,
    )

    assert trtmc_reference.run_reference(_args(first, cache_dir)) == "generated"
    assert trtmc_reference.run_reference(_args(second, cache_dir)) == "reused"

    assert calls == [first]
    assert (first / "hf_predictions.json").is_symlink()
    assert (second / "hf_predictions.json").is_symlink()
    assert not (second / "hf_predictions.json").readlink().is_absolute()
    payload = json.loads(
        (second / "hf_predictions.json").read_text(encoding="utf-8")
    )
    assert Path(payload["responses"][0]["artifact"]).read_bytes() == b"reference"
    assert json.loads((second / "hf_cache.json").read_text(encoding="utf-8"))[
        "status"
    ] == "reused"
    entries = [path for path in cache_dir.iterdir() if not path.name.startswith(".")]
    assert len(entries) == 1
    assert stat.S_IMODE(entries[0].stat().st_mode) & 0o055 == 0o055


def test_reference_cache_key_changes_with_inference_setting(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cache_dir = tmp_path / "cache"
    first = tmp_path / "first"
    second = tmp_path / "second"
    _prepare_work(first)
    _prepare_work(second)
    calls = 0

    def fake_reference(args) -> None:
        nonlocal calls
        calls += 1
        Path(args.work_dir, "hf_predictions.json").write_text(
            json.dumps(
                {"responses": [{"sample_id": "one", "output_text": "A"}]}
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(
        trtmc_reference.task_eval,
        "run_hf_reference",
        fake_reference,
    )

    trtmc_reference.run_reference(_args(first, cache_dir, "--seed", "1"))
    trtmc_reference.run_reference(_args(second, cache_dir, "--seed", "2"))

    assert calls == 2
    assert len([path for path in cache_dir.iterdir() if not path.name.startswith(".")]) == 2


def test_reference_cache_can_adopt_an_existing_result(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cache_dir = tmp_path / "cache"
    work_dir = tmp_path / "existing"
    _prepare_work(work_dir)
    (work_dir / "hf_predictions.json").write_text(
        json.dumps({"responses": [{"sample_id": "one", "output_text": "A"}]}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        trtmc_reference.task_eval,
        "run_hf_reference",
        lambda _args: (_ for _ in ()).throw(AssertionError("must not infer")),
    )

    status = trtmc_reference.run_reference(
        _args(work_dir, cache_dir, "--adopt-existing")
    )

    assert status == "adopted"
    assert (work_dir / "hf_predictions.json").is_symlink()


def test_causal_reference_uses_native_transformers_entrypoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cache_dir = tmp_path / "cache"
    work_dir = tmp_path / "native"
    _prepare_work(work_dir)
    arguments = _args(work_dir, cache_dir)
    arguments.reference_family = "causal_base_continuation"
    manifest_path = work_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dataset_kind"] = "mmlu_five_shot_json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        assert kwargs["stderr"] == trtmc_reference.subprocess.STDOUT
        predictions = Path(command[command.index("--predictions") + 1])
        raw_output = Path(command[command.index("--raw-output") + 1])
        metadata = Path(command[command.index("--repro-metadata") + 1])
        predictions.write_text(
            json.dumps(
                {"responses": [{"sample_id": "one", "output_text": "A"}]}
            ),
            encoding="utf-8",
        )
        raw_output.write_text("{}\n", encoding="utf-8")
        metadata.write_text(
            json.dumps(
                {
                    "command": [
                        command[0],
                        command[1],
                        "--sample-id",
                        "{sample_id}",
                    ]
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        trtmc_reference.task_eval,
        "run_hf_reference",
        lambda _args: (_ for _ in ()).throw(AssertionError("wrapper was used")),
    )
    monkeypatch.setattr(trtmc_reference.subprocess, "run", fake_run)

    assert trtmc_reference.run_reference(arguments) == "generated"

    command = captured["command"]
    assert command[1].endswith("tools/reference/transformers_text.py")
    assert "task_eval.py" not in " ".join(command)
    assert (work_dir / "hf_native_run.log").is_symlink()
    assert (work_dir / "hf_native_repro.json").is_symlink()


def test_transformers_reference_metadata_is_direct_and_sample_selectable(
    tmp_path: Path,
) -> None:
    metadata = tmp_path / "reproduction.json"
    arguments = transformers_text.build_parser().parse_args(
        [
            "--model",
            "org/model",
            "--prompts",
            str(tmp_path / "prompts.jsonl"),
            "--answers",
            str(tmp_path / "answers.json"),
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--predictions",
            str(tmp_path / "predictions.json"),
            "--raw-output",
            str(tmp_path / "raw.jsonl"),
            "--repro-metadata",
            str(metadata),
            "--local-files-only",
        ]
    )

    transformers_text._write_reproduction_metadata(arguments)

    payload = json.loads(metadata.read_text(encoding="utf-8"))
    command = payload["command"]
    assert command[1].endswith("tools/reference/transformers_text.py")
    assert command[command.index("--sample-id") + 1] == "{sample_id}"
    assert command[command.index("--prompts") + 1] == "{work_dir}/prompts.jsonl"
    assert "task_eval.py" not in " ".join(command)


def test_encoder_reference_metadata_is_direct_and_sample_selectable(
    tmp_path: Path,
) -> None:
    metadata = tmp_path / "reproduction.json"
    arguments = transformers_encoder.build_parser().parse_args(
        [
            "--model",
            "org/model",
            "--reference-family",
            "encoder_base_features",
            "--prompts",
            str(tmp_path / "prompts.jsonl"),
            "--answers",
            str(tmp_path / "answers.json"),
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--predictions",
            str(tmp_path / "predictions.json"),
            "--raw-output",
            str(tmp_path / "raw.jsonl"),
            "--repro-metadata",
            str(metadata),
            "--local-files-only",
        ]
    )

    transformers_encoder._write_reproduction_metadata(arguments)

    payload = json.loads(metadata.read_text(encoding="utf-8"))
    command = payload["command"]
    assert command[1].endswith("tools/reference/transformers_encoder.py")
    assert command[command.index("--sample-id") + 1] == "{sample_id}"
    assert command[command.index("--prompts") + 1] == "{work_dir}/prompts.jsonl"
    assert "task_eval.py" not in " ".join(command)


def test_native_reference_runner_uses_prepared_dataset_kind(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    _prepare_work(work_dir)
    arguments = _args(work_dir, tmp_path / "cache")
    manifest_path = work_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    manifest["dataset_kind"] = "text_generation_json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert trtmc_reference._native_reference_runner(arguments).name == (
        "transformers_text.py"
    )

    manifest["dataset_kind"] = "sts_pair_jsonl"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert trtmc_reference._native_reference_runner(arguments).name == (
        "transformers_encoder.py"
    )

    manifest["dataset_kind"] = "diffusion_prompt_json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert trtmc_reference._native_reference_runner(arguments).name == (
        "plugin_reference.py"
    )

    manifest["dataset_kind"] = "vlm_chat_json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert trtmc_reference._native_reference_runner(arguments).name == (
        "transformers_vlm.py"
    )

    manifest["dataset_kind"] = "asr_chat_json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert trtmc_reference._native_reference_runner(arguments).name == "speech.py"

    manifest["dataset_kind"] = "conditional_text_jsonl"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert trtmc_reference._native_reference_runner(arguments).name == (
        "elf_prepared.py"
    )


def test_plugin_reference_metadata_uses_recorded_upstream_command(
    tmp_path: Path,
) -> None:
    metadata = tmp_path / "reproduction.json"
    arguments = plugin_reference.build_parser().parse_args(
        [
            "--model",
            "org/model",
            "--reference-family",
            "diffusers_image_gen",
            "--prompts",
            str(tmp_path / "prompts.jsonl"),
            "--answers",
            str(tmp_path / "answers.json"),
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--predictions",
            str(tmp_path / "predictions.json"),
            "--raw-output",
            str(tmp_path / "raw.jsonl"),
            "--repro-metadata",
            str(metadata),
            "--local-files-only",
        ]
    )

    plugin_reference._write_reproduction_metadata(arguments)

    payload = json.loads(metadata.read_text(encoding="utf-8"))
    assert payload["command_source"] == "hf_native_commands.jsonl"
    assert "command" not in payload


def test_plugin_reference_records_nested_upstream_command(tmp_path: Path) -> None:
    command_path = tmp_path / "hf_native_commands.jsonl"
    command_path.write_text("", encoding="utf-8")
    output = SimpleNamespace(
        metadata={
            "backend": {
                "command": [
                    "/profiles/diffusers/bin/python",
                    "/workspace/model/reference.py",
                    "--prompt",
                    "a red cube",
                ]
            }
        }
    )

    plugin_reference._record_native_command(
        command_path,
        "sample-3",
        output,
    )

    row = json.loads(command_path.read_text(encoding="utf-8"))
    assert row["sample_id"] == "sample-3"
    assert row["command"][1] == "/workspace/model/reference.py"
    assert "plugin_reference.py" not in " ".join(row["command"])


def test_plugin_reference_preserves_model_owned_subprocess_command() -> None:
    namespace: dict[str, object] = {}

    def fake_run_reference_subprocess(*, command, **_kwargs):
        return SimpleNamespace(
            metadata={"returncode": 0},
            command_was_run=list(command),
        )

    namespace["run_reference_subprocess"] = fake_run_reference_subprocess
    exec(
        "def run_stage(case, stage, context):\n"
        "    return run_reference_subprocess(\n"
        "        command=[context.hf_python, '-c', case.script],\n"
        "    )\n",
        namespace,
    )
    original = namespace["run_reference_subprocess"]
    reference = SimpleNamespace(run_stage=namespace["run_stage"])
    case = SimpleNamespace(script="print('native HF')")
    context = SimpleNamespace(hf_python="/profiles/reference/bin/python")

    output = plugin_reference._run_reference_stage(
        reference,
        case,
        SimpleNamespace(name="full_inference"),
        context,
    )

    assert output.metadata["command"] == [
        "/profiles/reference/bin/python",
        "-c",
        "print('native HF')",
    ]
    assert output.command_was_run == output.metadata["command"]
    assert namespace["run_reference_subprocess"] is original


def test_plugin_reference_preserves_direct_subprocess_command() -> None:
    namespace: dict[str, object] = {}

    def fake_subprocess_run(command, **_kwargs):
        return SimpleNamespace(
            returncode=0,
            command_was_run=list(command),
        )

    subprocess_module = SimpleNamespace(run=fake_subprocess_run)
    namespace["subprocess"] = subprocess_module
    exec(
        "def run_stage(case, stage, context):\n"
        "    completed = subprocess.run(\n"
        "        [context.hf_python, '-c', case.script],\n"
        "        capture_output=True,\n"
        "    )\n"
        "    return type('Output', (), {\n"
        "        'metadata': {'returncode': completed.returncode},\n"
        "        'command_was_run': completed.command_was_run,\n"
        "    })()\n",
        namespace,
    )
    reference = SimpleNamespace(run_stage=namespace["run_stage"])
    case = SimpleNamespace(script="print('direct HF')")
    context = SimpleNamespace(hf_python="/profiles/reference/bin/python")

    output = plugin_reference._run_reference_stage(
        reference,
        case,
        SimpleNamespace(name="end_to_end"),
        context,
    )

    assert output.metadata["command"] == [
        "/profiles/reference/bin/python",
        "-c",
        "print('direct HF')",
    ]
    assert output.command_was_run == output.metadata["command"]
    assert subprocess_module.run is fake_subprocess_run


def test_vlm_reference_metadata_is_direct_and_sample_selectable(
    tmp_path: Path,
) -> None:
    metadata = tmp_path / "reproduction.json"
    arguments = transformers_vlm.build_parser().parse_args(
        [
            "--model",
            "org/model",
            "--reference-family",
            "vl_instruct_qa",
            "--prompts",
            str(tmp_path / "prompts.jsonl"),
            "--answers",
            str(tmp_path / "answers.json"),
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--predictions",
            str(tmp_path / "predictions.json"),
            "--raw-output",
            str(tmp_path / "raw.jsonl"),
            "--repro-metadata",
            str(metadata),
            "--local-files-only",
        ]
    )

    transformers_vlm._write_reproduction_metadata(arguments)

    payload = json.loads(metadata.read_text(encoding="utf-8"))
    command = payload["command"]
    assert command[1].endswith("tools/reference/transformers_vlm.py")
    assert command[command.index("--sample-id") + 1] == "{sample_id}"
    assert command[command.index("--answers") + 1] == "{work_dir}/answers.json"
    assert "task_eval.py" not in " ".join(command)


def test_speech_reference_metadata_is_direct_and_sample_selectable(
    tmp_path: Path,
) -> None:
    metadata = tmp_path / "reproduction.json"
    arguments = speech.build_parser().parse_args(
        [
            "--model",
            "openai/whisper-tiny",
            "--family",
            "whisper",
            "--reference-family",
            "asr_whisper",
            "--prompts",
            str(tmp_path / "prompts.jsonl"),
            "--answers",
            str(tmp_path / "answers.json"),
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--predictions",
            str(tmp_path / "predictions.json"),
            "--raw-output",
            str(tmp_path / "raw.jsonl"),
            "--repro-metadata",
            str(metadata),
            "--local-files-only",
        ]
    )

    speech._write_reproduction_metadata(arguments)

    payload = json.loads(metadata.read_text(encoding="utf-8"))
    command = payload["command"]
    assert command[1].endswith("tools/reference/speech.py")
    assert command[command.index("--sample-id") + 1] == "{sample_id}"
    assert command[command.index("--manifest") + 1] == "{work_dir}/manifest.json"
    assert "task_eval.py" not in " ".join(command)


def test_elf_metadata_points_to_official_reference_entrypoint(tmp_path: Path) -> None:
    repo = tmp_path / "elf"
    config = repo / "src/configs/model.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("{}\n", encoding="utf-8")
    metadata = tmp_path / "reproduction.json"
    arguments = elf_prepared.build_parser().parse_args(
        [
            "--model",
            "org/elf",
            "--elf-reference-repo",
            str(repo),
            "--prompts",
            str(tmp_path / "prompts.jsonl"),
            "--answers",
            str(tmp_path / "answers.json"),
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--predictions",
            str(tmp_path / "predictions.json"),
            "--raw-output",
            str(tmp_path / "raw.jsonl"),
            "--repro-metadata",
            str(metadata),
        ]
    )
    manifest = {
        "task_eval": {
            "reference": {
                "config": "src/configs/model.yaml",
                "checkpoint": "org/elf",
            }
        },
        "generation": {"generation_mode": "conditional"},
    }

    elf_prepared._write_reproduction_metadata(arguments, manifest)

    payload = json.loads(metadata.read_text(encoding="utf-8"))
    command = payload["command"]
    assert command[1].endswith("tools/elf_hf_reference.py")
    assert command[command.index("--dataset") + 1] == "{reference_input_jsonl}"
    assert command[command.index("--seed") + 1] == "{reference_sample_seed}"
    assert payload["base_seed"] == 42
    assert "elf_prepared.py" not in " ".join(command)
    assert "task_eval.py" not in " ".join(command)


def test_speech_runner_dispatches_asr_without_task_eval(
    tmp_path: Path,
    monkeypatch,
) -> None:
    prompts = tmp_path / "prompts.jsonl"
    answers = tmp_path / "answers.json"
    manifest = tmp_path / "manifest.json"
    predictions = tmp_path / "predictions.json"
    raw_output = tmp_path / "raw.jsonl"
    metadata = tmp_path / "reproduction.json"
    prompts.write_text(
        json.dumps({"sample_id": "asr-1", "audio": "/data/one.wav"}) + "\n",
        encoding="utf-8",
    )
    answers.write_text(
        json.dumps({"requests": [{"sample_id": "asr-1"}]}),
        encoding="utf-8",
    )
    manifest.write_text(
        json.dumps({"dataset_kind": "asr_chat_json", "generation": {}}),
        encoding="utf-8",
    )
    calls = []

    def fake_whisper(arguments, selected, generation):
        calls.append((arguments.model, selected, generation))
        return [
            {
                "sample_id": "asr-1",
                "output_text": "hello",
                "source": "hf",
            }
        ]

    monkeypatch.setattr(speech, "_run_whisper_asr", fake_whisper)
    arguments = speech.build_parser().parse_args(
        [
            "--model",
            "openai/whisper-tiny",
            "--family",
            "whisper",
            "--prompts",
            str(prompts),
            "--answers",
            str(answers),
            "--manifest",
            str(manifest),
            "--predictions",
            str(predictions),
            "--raw-output",
            str(raw_output),
            "--repro-metadata",
            str(metadata),
        ]
    )

    speech.run(arguments)

    assert calls[0][0] == "openai/whisper-tiny"
    assert calls[0][1][0]["sample_id"] == "asr-1"
    assert json.loads(predictions.read_text(encoding="utf-8"))["responses"][0][
        "output_text"
    ] == "hello"
    assert "task_eval.py" not in metadata.read_text(encoding="utf-8")


def test_plugin_reference_records_actual_command_during_inference(
    tmp_path: Path,
    monkeypatch,
) -> None:
    prompts = tmp_path / "prompts.jsonl"
    answers = tmp_path / "answers.json"
    manifest = tmp_path / "manifest.json"
    predictions = tmp_path / "hf_predictions.json"
    raw_output = tmp_path / "hf_raw.jsonl"
    metadata = tmp_path / "hf_native_repro.json"
    prompts.write_text(
        json.dumps(
            {
                "sample_id": "series-1",
                "inputs": {"past_values": [1.0, 2.0]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    answers.write_text(json.dumps({"requests": []}), encoding="utf-8")
    manifest.write_text(
        json.dumps(
            {
                "dataset_kind": "time_series_csv",
                "task_eval": {"model_manifest": "unused.json"},
            }
        ),
        encoding="utf-8",
    )
    template = SimpleNamespace(
        name="template",
        inputs={},
        stages=[SimpleNamespace(name="full_inference")],
    )

    class FakeReference:
        def run_stage(self, case, stage, context):
            assert case.name == "series-1"
            assert stage.name == "full_inference"
            assert context.artifacts_dir.endswith("hf_artifacts")
            return SimpleNamespace(
                data={
                    "output_field": [3.0, 4.0],
                    "output_shape": [1, 2],
                },
                metadata={
                    "command": [
                        "/profiles/timeseries/bin/python",
                        "/workspace/model/reference.py",
                        "--input",
                        "series-1.csv",
                    ],
                    "returncode": 0,
                },
                timing_s=0.01,
            )

    monkeypatch.setattr(
        plugin_reference,
        "_load_reference_plugin",
        lambda _manifest: (template, FakeReference()),
    )
    arguments = plugin_reference.build_parser().parse_args(
        [
            "--model",
            "org/model",
            "--prompts",
            str(prompts),
            "--answers",
            str(answers),
            "--manifest",
            str(manifest),
            "--predictions",
            str(predictions),
            "--raw-output",
            str(raw_output),
            "--repro-metadata",
            str(metadata),
        ]
    )

    plugin_reference.run(arguments)

    response = json.loads(predictions.read_text(encoding="utf-8"))["responses"][0]
    assert response["output_values"] == [3.0, 4.0]
    command = json.loads(
        (tmp_path / "hf_native_commands.jsonl").read_text(encoding="utf-8")
    )["command"]
    assert command[1] == "/workspace/model/reference.py"
    assert "plugin_reference.py" not in " ".join(command)


def test_elf_adapter_preserves_original_sample_seed_and_answer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    reference_repo = tmp_path / "elf"
    config = reference_repo / "src/configs/model.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("{}\n", encoding="utf-8")
    prompts = tmp_path / "prompts.jsonl"
    answers = tmp_path / "answers.json"
    manifest = tmp_path / "manifest.json"
    predictions = tmp_path / "hf_predictions.json"
    raw_output = tmp_path / "hf_raw.jsonl"
    metadata = tmp_path / "hf_native_repro.json"
    prompts.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "sample_id": "elf-0",
                        "source_text": "zero",
                    }
                ),
                json.dumps(
                    {
                        "sample_id": "elf-1",
                        "source_text": "one",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    answers.write_text(
        json.dumps(
            {
                "requests": [
                    {"sample_id": "elf-0", "answer": "ZERO"},
                    {"sample_id": "elf-1", "answer": "ONE"},
                ]
            }
        ),
        encoding="utf-8",
    )
    manifest.write_text(
        json.dumps(
            {
                "dataset_kind": "conditional_text_jsonl",
                "generation": {
                    "generation_mode": "conditional",
                    "sampling_method": "ode",
                    "seed": 42,
                },
                "task_eval": {
                    "reference": {
                        "config": "src/configs/model.yaml",
                        "checkpoint": "org/elf",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    def fake_run(command, **_kwargs):
        dataset = Path(command[command.index("--dataset") + 1])
        row = json.loads(dataset.read_text(encoding="utf-8"))
        assert row == {"id": "elf-1", "input": "one", "output": "ONE"}
        assert command[command.index("--seed") + 1] == "43"
        Path(command[command.index("--output") + 1]).write_text(
            json.dumps(
                {
                    "responses": [
                        {"sample_id": "elf-1", "output_text": "generated"}
                    ]
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(elf_prepared.subprocess, "run", fake_run)
    arguments = elf_prepared.build_parser().parse_args(
        [
            "--model",
            "org/elf",
            "--elf-reference-repo",
            str(reference_repo),
            "--prompts",
            str(prompts),
            "--answers",
            str(answers),
            "--manifest",
            str(manifest),
            "--predictions",
            str(predictions),
            "--raw-output",
            str(raw_output),
            "--repro-metadata",
            str(metadata),
            "--sample-id",
            "elf-1",
        ]
    )

    elf_prepared.run(arguments)

    assert json.loads(predictions.read_text(encoding="utf-8"))["responses"][0][
        "sample_id"
    ] == "elf-1"
    repro = json.loads(metadata.read_text(encoding="utf-8"))
    assert repro["command"][1].endswith("tools/elf_hf_reference.py")
