# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
import stat
import sys
from types import ModuleType, SimpleNamespace

import pytest

from tools.reference import (
    elf_prepared,
    plugin_reference,
    speech,
    transformers_encoder,
    transformers_text,
    transformers_vlm,
)
from tools import elf_hf_reference, trtmc_reference


def test_elf_reference_uses_canonical_t5_cache_id() -> None:
    assert elf_hf_reference._canonical_hf_model_id("t5-small") == "google-t5/t5-small"
    assert elf_hf_reference._canonical_hf_model_id("org/custom") == "org/custom"


def test_elf_reference_ignores_training_only_optimizer_state() -> None:
    optimizer = elf_hf_reference._InferenceOptimizer()
    assert optimizer.load_state_dict({"state": {"unused": True}}) is None


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


def _prepare_work(path: Path, *, model_manifest: str = "") -> None:
    path.mkdir(parents=True)
    (path / "answers.json").write_text(
        json.dumps({"requests": [{"sample_id": "one", "answer": "A"}]}),
        encoding="utf-8",
    )
    (path / "prompts.jsonl").write_text(
        json.dumps({"sample_id": "one", "prompt": "question"}) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "dataset_kind": "mmlu_json",
        "files": {
            "answers": str(path / "answers.json"),
            "prompts": str(path / "prompts.jsonl"),
        },
    }
    if model_manifest:
        manifest["task_eval"] = {"model_manifest": model_manifest}
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


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


def test_prepared_dataset_without_native_runner_fails_closed(
    tmp_path: Path,
) -> None:
    work_dir = tmp_path / "unsupported"
    _prepare_work(work_dir)

    with pytest.raises(
        trtmc_reference.ReferenceError,
        match="dataset kind 'mmlu_json'",
    ):
        trtmc_reference._run_reference_inference(
            _args(work_dir, tmp_path / "cache")
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
        trtmc_reference,
        "_run_reference_inference",
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


def test_reference_cache_identity_shares_equivalent_trtmc_variants(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cache_dir = tmp_path / "cache"
    first = tmp_path / "first"
    second = tmp_path / "second"
    model_dir = tmp_path / "model"
    manifest_dir = model_dir / "manifests"
    plugins_dir = model_dir / "e2e_plugins"
    manifest_dir.mkdir(parents=True)
    plugins_dir.mkdir()
    fp16_manifest = manifest_dir / "model-fp16.json"
    fp8_manifest = manifest_dir / "model-fp8.json"
    fp16_manifest.write_text('{"precision": "fp16"}', encoding="utf-8")
    fp8_manifest.write_text('{"precision": "fp8"}', encoding="utf-8")
    (plugins_dir / "reference.py").write_text("REFERENCE = 1\n", encoding="utf-8")
    _prepare_work(first, model_manifest=str(fp16_manifest))
    _prepare_work(second, model_manifest=str(fp8_manifest))
    for work_dir in (first, second):
        manifest_path = work_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["dataset_kind"] = "diffusion_prompt_json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
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
        trtmc_reference,
        "_run_reference_inference",
        fake_reference,
    )

    identity = "org/model/reference-contract-v1"
    first_args = _args(
        first,
        cache_dir,
        "--reference-cache-identity",
        identity,
    )
    second_args = _args(
        second,
        cache_dir,
        "--reference-cache-identity",
        identity,
    )

    assert trtmc_reference.run_reference(first_args) == "generated"
    assert trtmc_reference.run_reference(second_args) == "reused"
    assert calls == 1
    entries = [
        path for path in cache_dir.iterdir() if not path.name.startswith(".")
    ]
    assert len(entries) == 1


def test_reference_cache_identity_ignores_native_runner_variant_metadata(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    first = tmp_path / "first"
    second = tmp_path / "second"
    _prepare_work(first, model_manifest="manifests/model-fp16.json")
    _prepare_work(second, model_manifest="manifests/model-fp8.json")
    for work_dir in (first, second):
        manifest_path = work_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["dataset_kind"] = "mmlu_five_shot_json"
        manifest["generation"] = {"max_new_tokens": 1, "do_sample": False}
        manifest["task_eval"].update(
            {
                "family": "qwen",
                "model_max_new_tokens": 10,
                "task_strategy": "text_generation_causal",
            }
        )
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    second_manifest_path = second / "manifest.json"
    second_manifest = json.loads(second_manifest_path.read_text(encoding="utf-8"))
    second_manifest["task_eval"].update(
        {
            "model_max_new_tokens": 20,
            "reference_backend": "hf_transformers",
            "reference_family": "chat_qwen3_posttrained",
            "user_contract": "chat_response",
        }
    )
    second_manifest_path.write_text(
        json.dumps(second_manifest),
        encoding="utf-8",
    )
    identity = "qwen3-0.6b-mmlu-five-shot-v1"

    first_key, _ = trtmc_reference.reference_key(
        _args(first, cache_dir, "--reference-cache-identity", identity)
    )
    second_key, _ = trtmc_reference.reference_key(
        _args(second, cache_dir, "--reference-cache-identity", identity)
    )

    assert first_key == second_key


def test_reference_cache_identity_keeps_effective_generation_separate(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    first = tmp_path / "first"
    second = tmp_path / "second"
    _prepare_work(first, model_manifest="manifests/model-fp16.json")
    _prepare_work(second, model_manifest="manifests/model-fp8.json")
    for work_dir, max_new_tokens in ((first, 1), (second, 2)):
        manifest_path = work_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["dataset_kind"] = "mmlu_five_shot_json"
        manifest["generation"] = {"max_new_tokens": max_new_tokens}
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    identity = "org/model/reference-contract-v1"

    first_key, _ = trtmc_reference.reference_key(
        _args(first, cache_dir, "--reference-cache-identity", identity)
    )
    second_key, _ = trtmc_reference.reference_key(
        _args(second, cache_dir, "--reference-cache-identity", identity)
    )

    assert first_key != second_key


def test_reference_cache_keeps_variant_manifests_separate_without_identity(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    first = tmp_path / "first"
    second = tmp_path / "second"
    _prepare_work(first, model_manifest="manifests/model-fp16.json")
    _prepare_work(second, model_manifest="manifests/model-fp8.json")

    first_key, _ = trtmc_reference.reference_key(_args(first, cache_dir))
    second_key, _ = trtmc_reference.reference_key(_args(second, cache_dir))

    assert first_key != second_key


def test_plugin_reference_cache_key_tracks_model_owned_implementation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cache_dir = tmp_path / "cache"
    model_dir = tmp_path / "z_image"
    manifest_path = model_dir / "manifests" / "z-image-turbo.json"
    reference_path = (
        model_dir / "e2e_plugins" / "references" / "hf_diffusers.py"
    )
    manifest_path.parent.mkdir(parents=True)
    reference_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "name": "z-image-turbo",
                "testcases": [
                    {
                        "name": "z-image-turbo",
                        "reference_precision": "fp16",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    reference_path.write_text("REFERENCE_DTYPE = 'fp16'\n", encoding="utf-8")

    def prepare_work(name: str) -> Path:
        work_dir = tmp_path / name
        _prepare_work(work_dir, model_manifest=str(manifest_path))
        work_manifest_path = work_dir / "manifest.json"
        work_manifest = json.loads(
            work_manifest_path.read_text(encoding="utf-8")
        )
        work_manifest["dataset_kind"] = "diffusion_prompt_json"
        work_manifest_path.write_text(
            json.dumps(work_manifest),
            encoding="utf-8",
        )
        return work_dir

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
        trtmc_reference,
        "_run_reference_inference",
        fake_reference,
    )

    first = prepare_work("first")
    assert trtmc_reference.run_reference(_args(first, cache_dir)) == "generated"
    model_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_manifest["testcases"][0]["reference_precision"] = "bf16"
    manifest_path.write_text(json.dumps(model_manifest), encoding="utf-8")
    second = prepare_work("second")
    assert trtmc_reference.run_reference(_args(second, cache_dir)) == "generated"
    reference_path.write_text("REFERENCE_DTYPE = 'bf16'\n", encoding="utf-8")
    third = prepare_work("third")
    assert trtmc_reference.run_reference(_args(third, cache_dir)) == "generated"
    fourth = prepare_work("fourth")
    assert trtmc_reference.run_reference(_args(fourth, cache_dir)) == "reused"

    assert calls == 3
    assert len(list(cache_dir.iterdir())) == 3


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
        trtmc_reference,
        "_run_reference_inference",
        fake_reference,
    )

    trtmc_reference.run_reference(_args(first, cache_dir, "--seed", "1"))
    trtmc_reference.run_reference(_args(second, cache_dir, "--seed", "2"))

    assert calls == 2
    assert len([path for path in cache_dir.iterdir() if not path.name.startswith(".")]) == 2


def test_reference_cache_key_changes_with_experts_implementation(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    first = tmp_path / "first"
    second = tmp_path / "second"
    _prepare_work(first)
    _prepare_work(second)

    first_key, _ = trtmc_reference.reference_key(
        _args(first, cache_dir, "--experts-implementation", "batched_mm")
    )
    second_key, _ = trtmc_reference.reference_key(
        _args(second, cache_dir, "--experts-implementation", "eager")
    )

    assert first_key != second_key


def test_reference_cache_key_changes_with_model_revision(
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
        trtmc_reference,
        "_run_reference_inference",
        fake_reference,
    )

    trtmc_reference.run_reference(
        _args(first, cache_dir, "--model-revision", "revision-one")
    )
    trtmc_reference.run_reference(
        _args(second, cache_dir, "--model-revision", "revision-two")
    )

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
        trtmc_reference,
        "_run_reference_inference",
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
    arguments = _args(
        work_dir,
        cache_dir,
        "--model-revision",
        "0123456789abcdef",
        "--experts-implementation",
        "batched_mm",
    )
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

    monkeypatch.setattr(trtmc_reference.subprocess, "run", fake_run)

    assert trtmc_reference.run_reference(arguments) == "generated"

    command = captured["command"]
    assert command[1].endswith("tools/reference/transformers_text.py")
    assert command[command.index("--model-revision") + 1] == "0123456789abcdef"
    assert command[command.index("--experts-implementation") + 1] == "batched_mm"
    assert "validation/engine.py" not in " ".join(command)
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
            "--experts-implementation",
            "batched_mm",
            "--local-files-only",
        ]
    )

    transformers_text._write_reproduction_metadata(arguments)

    payload = json.loads(metadata.read_text(encoding="utf-8"))
    command = payload["command"]
    assert command[1].endswith("tools/reference/transformers_text.py")
    assert command[command.index("--sample-id") + 1] == "{sample_id}"
    assert command[command.index("--prompts") + 1] == "{work_dir}/prompts.jsonl"
    assert command[command.index("--experts-implementation") + 1] == "batched_mm"
    assert "validation/engine.py" not in " ".join(command)


def test_transformers_text_forwards_experts_implementation(monkeypatch) -> None:
    arguments = transformers_text.build_parser().parse_args(
        [
            "--model",
            "org/model",
            "--prompts",
            "/tmp/prompts.jsonl",
            "--answers",
            "/tmp/answers.json",
            "--manifest",
            "/tmp/manifest.json",
            "--predictions",
            "/tmp/predictions.json",
            "--raw-output",
            "/tmp/raw.jsonl",
            "--dtype",
            "float16",
            "--experts-implementation",
            "batched_mm",
        ]
    )
    captured: dict[str, object] = {}
    tokenizer = SimpleNamespace(pad_token_id=0)
    model = SimpleNamespace(device="cuda", to=lambda *args, **kwargs: None)
    transformers_module = SimpleNamespace(
        logging=SimpleNamespace(set_verbosity_error=lambda: None),
        AutoTokenizer=SimpleNamespace(
            from_pretrained=lambda _model_id, **_kwargs: tokenizer
        ),
    )

    def fake_load_model(
        _transformers_module,
        _model_id,
        *,
        model_kwargs,
        **_kwargs,
    ):
        captured.update(model_kwargs)
        return model, False

    monkeypatch.setattr(transformers_text, "_load_model", fake_load_model)

    transformers_text._load_runtime(
        arguments,
        SimpleNamespace(float16=object(), device=lambda name: name),
        transformers_module,
    )

    assert captured["experts_implementation"] == "batched_mm"


def test_transformers_text_moves_all_model_state_to_requested_dtype(
    monkeypatch,
) -> None:
    arguments = transformers_text.build_parser().parse_args(
        [
            "--model",
            "org/model",
            "--prompts",
            "/tmp/prompts.jsonl",
            "--answers",
            "/tmp/answers.json",
            "--manifest",
            "/tmp/manifest.json",
            "--predictions",
            "/tmp/predictions.json",
            "--raw-output",
            "/tmp/raw.jsonl",
            "--dtype",
            "bfloat16",
        ]
    )
    captured: dict[str, object] = {}
    tokenizer = SimpleNamespace(pad_token_id=0)

    class FakeModel:
        def to(self, *args: object, **kwargs: object) -> None:
            captured["to_args"] = args
            captured["to_kwargs"] = kwargs

    model = FakeModel()
    transformers_module = SimpleNamespace(
        logging=SimpleNamespace(set_verbosity_error=lambda: None),
        AutoTokenizer=SimpleNamespace(
            from_pretrained=lambda _model_id, **_kwargs: tokenizer
        ),
    )

    def fake_load_model(
        _transformers_module,
        _model_id,
        *,
        model_kwargs,
        **_kwargs,
    ):
        captured["model_kwargs"] = model_kwargs
        return model, False

    monkeypatch.setattr(transformers_text, "_load_model", fake_load_model)
    torch_module = SimpleNamespace(
        bfloat16="bf16",
        device=lambda name: f"device:{name}",
    )

    transformers_text._load_runtime(
        arguments,
        torch_module,
        transformers_module,
    )

    assert captured["model_kwargs"]["torch_dtype"] == "bf16"
    assert captured["to_args"] == ()
    assert captured["to_kwargs"] == {
        "device": "device:cuda",
        "dtype": "bf16",
    }


def test_transformers_vlm_moves_all_model_state_to_requested_dtype(
    monkeypatch,
) -> None:
    arguments = transformers_vlm.build_parser().parse_args(
        [
            "--model",
            "org/model",
            "--prompts",
            "/tmp/prompts.jsonl",
            "--answers",
            "/tmp/answers.json",
            "--manifest",
            "/tmp/manifest.json",
            "--predictions",
            "/tmp/predictions.json",
            "--raw-output",
            "/tmp/raw.jsonl",
            "--dtype",
            "bfloat16",
        ]
    )
    captured: dict[str, object] = {}
    processor = object()

    class FakeProcessor:
        @staticmethod
        def from_pretrained(_model: str, **_kwargs: object) -> object:
            return processor

    class FakeModel:
        def to(self, *args: object, **kwargs: object) -> None:
            captured["to_args"] = args
            captured["to_kwargs"] = kwargs

    model = FakeModel()
    transformers_module = SimpleNamespace(
        logging=SimpleNamespace(set_verbosity_error=lambda: None),
    )

    def fake_load_model(
        _transformers_module,
        _model_id,
        model_kwargs,
    ):
        captured["model_kwargs"] = model_kwargs
        return model

    monkeypatch.setattr(transformers_vlm, "_load_model", fake_load_model)
    torch_module = SimpleNamespace(
        bfloat16="bf16",
        device=lambda name: f"device:{name}",
    )

    loaded_processor, loaded_model, device = transformers_vlm._load_runtime(
        arguments,
        torch_module,
        transformers_module,
        FakeProcessor,
    )

    assert loaded_processor is processor
    assert loaded_model is model
    assert device == "device:cuda"
    assert captured["model_kwargs"]["torch_dtype"] == "bf16"
    assert captured["to_args"] == ()
    assert captured["to_kwargs"] == {
        "device": "device:cuda",
        "dtype": "bf16",
    }


def test_transformers_vlm_dispatches_locateanything_to_manual_runtime(
    monkeypatch,
) -> None:
    arguments = transformers_vlm.build_parser().parse_args(
        [
            "--model",
            "nvidia/LocateAnything-3B",
            "--prompts",
            "/tmp/prompts.jsonl",
            "--answers",
            "/tmp/answers.json",
            "--manifest",
            "/tmp/manifest.json",
            "--predictions",
            "/tmp/predictions.json",
            "--raw-output",
            "/tmp/raw.jsonl",
        ]
    )
    expected = (object(), object(), "cuda")
    captured = {}

    def fake_load(args, torch_module, transformers_module):
        captured.update(
            args=args,
            torch_module=torch_module,
            transformers_module=transformers_module,
        )
        return expected

    monkeypatch.setattr(
        transformers_vlm, "_load_locateanything_runtime", fake_load
    )
    result = transformers_vlm._load_runtime(
        arguments,
        "torch-module",
        "transformers-module",
        object(),
    )

    assert result == expected
    assert captured["args"] is arguments


def test_locateanything_reference_forces_slow_ar_generation(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class Tensor:
        ndim = 2
        shape = (1, 3)

        def to(self, *args, **kwargs):
            return self

    class Context:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    class Torch:
        int32 = "int32"
        cuda = SimpleNamespace(is_available=lambda: False)
        from_numpy = staticmethod(lambda _value: Tensor())
        inference_mode = staticmethod(Context)

    class Tokenizer:
        def __call__(self, _text, return_tensors=None):
            assert return_tensors == "pt"
            return {"input_ids": Tensor(), "attention_mask": Tensor()}

        def encode(self, _text, add_special_tokens=False):
            return [1, 2]

    class Model:
        def generate(self, **kwargs):
            captured.update(kwargs)
            return ["<ref>vehicle</ref><box><300><200><800><700></box>"]

    import numpy as np
    from tensorrt_model_connect.families.locateanything import vl_debug_runner

    monkeypatch.setattr(
        vl_debug_runner,
        "preprocess_image_inputs_for_trt",
        lambda *_args, **_kwargs: {
            "pixel_values": np.zeros((1,), dtype=np.float32),
            "image_grid_hws": np.zeros((1, 2), dtype=np.int32),
        },
    )

    result = transformers_vlm._locateanything_response(
        torch_module=Torch(),
        tokenizer=Tokenizer(),
        model=Model(),
        device="cuda",
        prompt_row={
            "sample_id": "one",
            "prompt": "Point to: white vehicle.",
            "images": ["/tmp/image.jpg"],
        },
        source_index=0,
        settings={"seed": -1, "max_new_tokens": 64},
    )

    assert captured["generation_mode"] == "slow"
    assert captured["do_sample"] is False
    assert result["output_text"].startswith("<ref>vehicle</ref><box>")


def test_transformers_text_reference_accepts_float32_dtype() -> None:
    arguments = transformers_text.build_parser().parse_args(
        [
            "--model",
            "org/model",
            "--prompts",
            "/tmp/prompts.jsonl",
            "--answers",
            "/tmp/answers.json",
            "--manifest",
            "/tmp/manifest.json",
            "--predictions",
            "/tmp/predictions.json",
            "--raw-output",
            "/tmp/raw.jsonl",
            "--dtype",
            "float32",
        ]
    )
    torch_module = SimpleNamespace(float32=object())

    assert transformers_text._model_dtype(torch_module, arguments.dtype) is (
        torch_module.float32
    )


def test_reference_entrypoint_accepts_float32_dtype() -> None:
    arguments = trtmc_reference.build_parser().parse_args(
        [
            "run",
            "--model",
            "org/model",
            "--work-dir",
            "/tmp/reference-work",
            "--dtype",
            "float32",
        ]
    )

    assert arguments.dtype == "float32"


def test_remaining_native_reference_runners_accept_float32_dtype() -> None:
    common = [
        "--model",
        "org/model",
        "--prompts",
        "/tmp/prompts.jsonl",
        "--answers",
        "/tmp/answers.json",
        "--manifest",
        "/tmp/manifest.json",
        "--predictions",
        "/tmp/predictions.json",
        "--raw-output",
        "/tmp/raw.jsonl",
        "--dtype",
        "float32",
    ]

    for module in (plugin_reference, transformers_vlm, speech):
        assert module.build_parser().parse_args(common).dtype == "float32"

    torch_module = SimpleNamespace(float32=object())
    assert (
        transformers_vlm._model_dtype(torch_module, "float32")
        is torch_module.float32
    )
    assert speech._model_dtype(torch_module, "float32") is torch_module.float32


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
    assert "validation/engine.py" not in " ".join(command)


def test_encoder_reference_moves_all_model_parameters_to_requested_dtype() -> None:
    captured: dict[str, object] = {}
    tokenizer = object()

    class FakeTokenizer:
        @staticmethod
        def from_pretrained(_model: str, **_kwargs: object) -> object:
            return tokenizer

    class FakeModel:
        def eval(self) -> "FakeModel":
            return self

        def to(self, *args: object, **kwargs: object) -> None:
            captured["to_args"] = args
            captured["to_kwargs"] = kwargs

    model = FakeModel()

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(_model: str, **kwargs: object) -> FakeModel:
            captured["model_kwargs"] = kwargs
            return model

    transformers = SimpleNamespace(
        AutoModel=FakeAutoModel,
        AutoTokenizer=FakeTokenizer,
        logging=SimpleNamespace(set_verbosity_error=lambda: None),
    )
    torch = SimpleNamespace(
        float16="fp16",
        bfloat16="bf16",
        float32="fp32",
        device=lambda name: f"device:{name}",
    )
    arguments = SimpleNamespace(
        reference_family="encoder_base_features",
        trust_remote_code=False,
        local_files_only=True,
        model_revision="",
        model="microsoft/deberta-base",
        dtype="float16",
        device_map="",
        device="cuda",
    )

    loaded_tokenizer, loaded_model, device = transformers_encoder._load_runtime(
        arguments,
        torch,
        transformers,
    )

    assert loaded_tokenizer is tokenizer
    assert loaded_model is model
    assert device == "device:cuda"
    assert captured["model_kwargs"] == {
        "torch_dtype": "fp16",
        "trust_remote_code": False,
        "local_files_only": True,
    }
    assert captured["to_args"] == ()
    assert captured["to_kwargs"] == {
        "device": "device:cuda",
        "dtype": "fp16",
    }


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

    manifest["dataset_kind"] = "model_plugin_json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert trtmc_reference._native_reference_runner(arguments).name == (
        "plugin_reference.py"
    )

    manifest["dataset_kind"] = "vlm_chat_json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert trtmc_reference._native_reference_runner(arguments).name == (
        "transformers_vlm.py"
    )

    manifest["dataset_kind"] = "vlm_grounding_json"
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


def test_plugin_reference_applies_validation_reference_precision(
    monkeypatch,
    tmp_path: Path,
) -> None:
    case = SimpleNamespace(
        metadata={"model_test_dir": ""},
        reference_backend="hf_diffusers",
    )
    reference = object()
    monkeypatch.setattr(
        plugin_reference,
        "_model_manifest_path",
        lambda _manifest: tmp_path / "model.json",
    )
    monkeypatch.setattr(plugin_reference, "load_manifest", lambda _path: case)
    monkeypatch.setattr(
        plugin_reference,
        "activate_model_plugins",
        lambda _path: None,
    )
    monkeypatch.setattr(
        plugin_reference,
        "get_reference",
        lambda _name: reference,
    )

    loaded_case, loaded_reference = plugin_reference._load_reference_plugin(
        {
            "task_eval": {
                "model_manifest": "model.json",
                "reference_precision": "bf16",
            }
        }
    )

    assert loaded_case.metadata["reference_precision"] == "bf16"
    assert loaded_reference is reference


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
    assert "validation/engine.py" not in " ".join(command)


def test_speech_reference_metadata_is_direct_and_sample_selectable(
    tmp_path: Path,
) -> None:
    metadata = tmp_path / "reproduction.json"
    arguments = speech.build_parser().parse_args(
        [
            "--model",
            "openai/whisper-tiny",
            "--model-revision",
            "0123456789abcdef",
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
    assert command[command.index("--model-revision") + 1] == "0123456789abcdef"
    assert command[command.index("--sample-id") + 1] == "{sample_id}"
    assert command[command.index("--manifest") + 1] == "{work_dir}/manifest.json"
    assert "validation/engine.py" not in " ".join(command)


def test_magpie_reference_restores_exact_pinned_archive(
    tmp_path: Path,
    monkeypatch,
) -> None:
    archive = tmp_path / "magpie_tts_multilingual_357m.nemo"
    archive.write_bytes(b"checkpoint")
    speaker_checkpoint = tmp_path / "speaker.bin"
    speaker_checkpoint.write_bytes(b"speaker")
    download_calls: list[dict[str, object]] = []
    opened_paths: list[object] = []
    restore_kwargs: dict[str, object] = {}

    def fake_hf_hub_download(**kwargs):
        download_calls.append(kwargs)
        if kwargs["repo_id"] == speech.MAGPIE_SPEAKER_ENCODER_REPO:
            return str(speaker_checkpoint)
        return str(archive)

    def fake_fsspec_open(path, *_args, **_kwargs):
        opened_paths.append(path)
        return object()

    class FakeModel:
        @classmethod
        def restore_from(cls, **kwargs):
            import fsspec

            fsspec.open(speech.MAGPIE_SPEAKER_ENCODER_URL)
            restore_kwargs.update(kwargs)
            return cls()

        def eval(self):
            return self

        def to(self, device):
            restore_kwargs["device"] = device
            return self

    huggingface_hub = ModuleType("huggingface_hub")
    huggingface_hub.hf_hub_download = fake_hf_hub_download
    fsspec = ModuleType("fsspec")
    fsspec.open = fake_fsspec_open
    nemo = ModuleType("nemo")
    nemo.__path__ = []
    collections = ModuleType("nemo.collections")
    collections.__path__ = []
    tts = ModuleType("nemo.collections.tts")
    tts.__path__ = []
    models = ModuleType("nemo.collections.tts.models")
    models.MagpieTTSModel = FakeModel
    monkeypatch.setitem(sys.modules, "huggingface_hub", huggingface_hub)
    monkeypatch.setitem(sys.modules, "fsspec", fsspec)
    monkeypatch.setitem(sys.modules, "nemo", nemo)
    monkeypatch.setitem(sys.modules, "nemo.collections", collections)
    monkeypatch.setitem(sys.modules, "nemo.collections.tts", tts)
    monkeypatch.setitem(sys.modules, "nemo.collections.tts.models", models)

    arguments = SimpleNamespace(
        model="nvidia/magpie_tts_multilingual_357m",
        model_revision="34d7e40da85cabc97f92198889b65cea27bc7fd1",
        local_files_only=True,
        device="cuda",
    )
    processor, model = speech._load_tts_runtime(
        arguments,
        SimpleNamespace(device=lambda name: f"device:{name}"),
    )

    assert processor is None
    assert isinstance(model, FakeModel)
    assert download_calls == [
        {
            "repo_id": speech.MAGPIE_SPEAKER_ENCODER_REPO,
            "filename": speech.MAGPIE_SPEAKER_ENCODER_FILENAME,
            "local_files_only": True,
        },
        {
            "repo_id": "nvidia/magpie_tts_multilingual_357m",
            "filename": "magpie_tts_multilingual_357m.nemo",
            "local_files_only": True,
            "revision": "34d7e40da85cabc97f92198889b65cea27bc7fd1",
        },
    ]
    assert opened_paths == [str(speaker_checkpoint)]
    assert restore_kwargs == {
        "restore_path": str(archive),
        "device": "device:cuda",
    }


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
    assert "validation/engine.py" not in " ".join(command)


def test_speech_runner_dispatches_asr_without_engine_wrapper(
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
    assert "validation/engine.py" not in metadata.read_text(encoding="utf-8")


def test_nemotron35_asr_restores_nemo_archive_and_uses_language_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    archive = tmp_path / "nemotron.nemo"
    archive.write_bytes(b"archive")
    transcribe_calls: list[tuple[str, int]] = []

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def empty_cache() -> None:
            return None

    fake_torch = SimpleNamespace(cuda=FakeCuda())

    class FakeModel:
        def transcribe(self, manifest_path, *, batch_size, verbose):
            rows = [
                json.loads(line)
                for line in Path(manifest_path)
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            expected_audio = (
                tmp_path
                / "predictions"
                / "hf_canary_audio"
                / "asr-1.wav"
            )
            assert rows == [
                {
                    "audio_filepath": str(expected_audio),
                    "duration": 0.5,
                    "text": "",
                    "lang": "en-US",
                }
            ]
            transcribe_calls.append((manifest_path, batch_size))
            assert verbose is False
            return [SimpleNamespace(text="CONCORD RETURNED")]

    monkeypatch.setattr(
        speech,
        "_load_nemotron35_model",
        lambda _arguments, resolved_archive: (
            fake_torch,
            FakeModel(),
        ),
    )
    monkeypatch.setattr(
        speech,
        "_resolve_nemotron35_archive",
        lambda _arguments: archive,
    )
    monkeypatch.setattr(
        speech,
        "_audio_for_prompt",
        lambda _prompt, _rate: ([0.0] * 8000, Path("source.wav")),
    )

    def write_fake_wav(path, _audio, _rate):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"wav")

    monkeypatch.setattr(speech, "_write_wav_pcm16", write_fake_wav)
    arguments = SimpleNamespace(
        model="nvidia/nemotron-3.5-asr-streaming-0.6b",
        device="cuda",
        dtype="auto",
        local_files_only=True,
        predictions=tmp_path / "predictions" / "hf_predictions.json",
    )

    responses = speech._run_nemotron35_asr(
        arguments,
        [{"sample_id": "asr-1", "language": "en-US"}],
        {"sample_rate": 16000},
    )

    assert len(transcribe_calls) == 1
    assert responses[0]["output_text"] == "CONCORD RETURNED"
    assert responses[0]["generated_token_ids"] is None


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


def test_model_plugin_reference_runs_manifest_owned_official_reference(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from tests.e2e.models.lance.e2e_plugins.references import lance_official
    from tests.e2e_harness.contracts import StageOutput

    monkeypatch.setattr(
        lance_official.LanceOfficialReference,
        "run_stage",
        lambda _self, _case, stage, _ctx: StageOutput(
            stage_name=stage.name,
            data={"text": "White"},
            text="White",
            metadata={
                "command": [
                    "/profiles/lance/bin/python",
                    "/references/Lance/inference_lance.py",
                ]
            },
        ),
    )
    manifest_path = (
        trtmc_reference.REPO_ROOT
        / "tests/e2e/models/lance/manifests/lance-3b-x2t-image.json"
    )
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        json.dumps(
            {
                "sample_id": "lance-sample",
                "testcase": "lance-3b-x2t-image",
                "stage": "full_generation",
                "inputs": {
                    "prompt": "What color is the vehicle?",
                    "image": str(tmp_path / "vehicle.png"),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    answers = tmp_path / "answers.json"
    answers.write_text(
        json.dumps({"requests": [{"sample_id": "lance-sample"}]}),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset_kind": "model_plugin_json",
                "task_eval": {"model_manifest": str(manifest_path)},
            }
        ),
        encoding="utf-8",
    )
    predictions = tmp_path / "hf_predictions.json"
    arguments = plugin_reference.build_parser().parse_args(
        [
            "--model",
            "bytedance-research/Lance",
            "--prompts",
            str(prompts),
            "--answers",
            str(answers),
            "--manifest",
            str(manifest),
            "--predictions",
            str(predictions),
            "--raw-output",
            str(tmp_path / "hf_raw.jsonl"),
            "--repro-metadata",
            str(tmp_path / "hf_native_repro.json"),
            "--dtype",
            "bfloat16",
        ]
    )

    plugin_reference.run(arguments)

    row = json.loads(predictions.read_text(encoding="utf-8"))["responses"][0]
    assert row["sample_id"] == "lance-sample"
    assert row["testcase"] == "lance-3b-x2t-image"
    assert row["stage"] == "full_generation"
    assert row["output_text"] == "White"
    assert row["stage_output"]["text"] == "White"


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
