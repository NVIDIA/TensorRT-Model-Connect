# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime, and official-reference E2E for nemotron_speech_streaming."""

from __future__ import annotations
import io
import json
import os
import re
import shutil
import subprocess
import tarfile
from pathlib import Path
import pytest
from tensorrt_model_connect import BuildRequest, build

FAMILY = "nemotron_speech_streaming"
TASKS = frozenset({"transcription_streaming"})
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"
THRESHOLD_ROOT = TEST_ROOT / "thresholds"
_NEMOTRON35_OPTIONAL_CTC_STATE_KEYS = frozenset(
    {
        "ctc_decoder.decoder_layers.0.bias",
        "ctc_decoder.decoder_layers.0.weight",
    }
)


def _case_index() -> dict[str, tuple[Path, dict, dict]]:
    result = {}
    for path in sorted(MANIFEST_ROOT.glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["family"] == FAMILY
        assert manifest["task"] in TASKS
        for case in manifest["testcases"]:
            name = str(case["name"])
            assert name not in result
            result[name] = (path, manifest, case)
    return result


CASES = _case_index()


def _selected_cases(config) -> tuple[list[str], bool]:
    model_filters = set()
    for raw in config.getoption("--e2e-model") or []:
        model_filters.update((item.strip() for item in str(raw).split(",") if item.strip()))
    models_file = config.getoption("--e2e-models-file")
    if models_file:
        model_filters.update(
            (
                line.strip()
                for line in Path(models_file).read_text(encoding="utf-8").splitlines()
                if line.strip() and (not line.lstrip().startswith("#"))
            )
        )
    testcase_filters = set()
    for raw in config.getoption("--e2e-testcase") or []:
        testcase_filters.update((item.strip() for item in str(raw).split(",") if item.strip()))
    if not model_filters and (not testcase_filters):
        return (sorted(CASES), False)
    selected = []
    for name, (_, manifest, _) in CASES.items():
        model_match = (
            not model_filters
            or FAMILY in model_filters
            or name in model_filters
            or (manifest["name"] in model_filters)
        )
        testcase_match = not testcase_filters or name in testcase_filters
        if model_match and testcase_match:
            selected.append(name)
    return (sorted(selected), True)


def pytest_generate_tests(metafunc) -> None:
    if "case_name" in metafunc.fixturenames:
        names, enabled = _selected_cases(metafunc.config)
        parameters = names
        if not enabled:
            parameters = [
                pytest.param(
                    name,
                    marks=pytest.mark.skip(
                        reason="direct E2E requires one of the three explicit E2E selectors"
                    ),
                )
                for name in names
            ]
        metafunc.parametrize("case_name", parameters, ids=names)


def _required_path(value: str | None, label: str) -> Path:
    assert value, f"selected {FAMILY} E2E requires {label}"
    path = Path(value)
    assert path.exists(), f"selected {FAMILY} E2E {label} does not exist: {path}"
    return path


def _model_dir(manifest: dict) -> Path:
    explicit = os.environ.get(f"TRTMC_{FAMILY.upper()}_MODEL_DIR")
    if explicit:
        return _required_path(explicit, f"TRTMC_{FAMILY.upper()}_MODEL_DIR")
    from huggingface_hub import snapshot_download

    try:
        snapshot = snapshot_download(
            repo_id=manifest["hf_id"], revision=manifest.get("hf_revision"), local_files_only=True
        )
    except Exception as error:
        raise AssertionError(
            f"selected {FAMILY} E2E requires the exact cached checkpoint {manifest['hf_id']}"
        ) from error
    return Path(snapshot)


def _runtime(manifest: dict) -> tuple[Path, Path]:
    binary = _required_path(os.environ.get("TRTMC_BINARY"), "TRTMC_BINARY")
    runtime_root = _required_path(os.environ.get("TRTMC_RUNTIME_ROOT"), "TRTMC_RUNTIME_ROOT")
    assert (runtime_root / "libtrtmc_backend_trt.so").is_file()
    assert (runtime_root / f"libtrtmc_model_{FAMILY}.so").is_file()
    import torch

    required_gpus = int(manifest["tensor_parallel_size"])
    assert torch.cuda.is_available(), f"selected {FAMILY} E2E requires CUDA"
    assert torch.cuda.device_count() >= required_gpus, (
        f"selected {FAMILY} E2E requires {required_gpus} GPUs, found {torch.cuda.device_count()}"
    )
    return (binary, runtime_root)


def _build(model_dir: Path, bundle: Path, manifest: dict) -> None:
    build(
        BuildRequest(
            model_dir=model_dir,
            output_path=bundle,
            family=FAMILY,
            task=manifest["task"],
            precision=manifest["precision"],
            max_sequence_length=manifest.get("max_sequence_length"),
            image_height=manifest.get("image_height"),
            image_width=manifest.get("image_width"),
            video_num_frames=manifest.get("video_num_frames"),
            max_batch_size=int(manifest.get("max_batch_size", 1)),
            tensor_parallel_size=int(manifest["tensor_parallel_size"]),
            quantization=manifest.get("quantization"),
            fp32_layers=tuple((int(layer) for layer in manifest.get("fp32_layers", ()))),
        )
    )


def _run_json(
    binary: Path,
    runtime_root: Path,
    bundle: Path,
    manifest: dict,
    case: dict,
    command: str,
    *arguments: str,
) -> dict:
    invocation = [
        str(binary),
        command,
        str(bundle),
        "--runtime-root",
        str(runtime_root),
        *arguments,
    ]
    if int(manifest["tensor_parallel_size"]) > 1:
        mpirun = shutil.which("mpirun")
        assert mpirun, "selected multi-GPU E2E requires mpirun"
        invocation = [
            mpirun,
            "--tag-output",
            "-x",
            "LD_LIBRARY_PATH",
            "-x",
            "TRTMC_NCCL_RENDEZVOUS",
            "-np",
            str(manifest["tensor_parallel_size"]),
            *invocation,
        ]
    env = os.environ.copy()
    env["TRTMC_NCCL_RENDEZVOUS"] = str(bundle.with_suffix(".nccl-rendezvous"))
    env["LD_LIBRARY_PATH"] = ":".join(
        (value for value in (str(runtime_root), env.get("LD_LIBRARY_PATH", "")) if value)
    )
    completed = subprocess.run(
        invocation,
        check=True,
        capture_output=True,
        text=True,
        env=env,
        timeout=int(case.get("runtime_timeout_s", 3600)),
    )
    payloads = []
    for line in completed.stdout.splitlines():
        start = line.find("{")
        if start >= 0:
            try:
                payloads.append(json.loads(line[start:]))
            except json.JSONDecodeError:
                pass
    assert payloads, f"native {command} returned no JSON: {completed.stdout[-1000:]}"
    assert all((payload == payloads[0] for payload in payloads))
    return payloads[0]


def _thresholds(case_name: str) -> dict:
    path = THRESHOLD_ROOT / f"{case_name}.json"
    assert path.is_file(), f"selected {FAMILY} E2E requires exact thresholds: {path}"
    return json.loads(path.read_text(encoding="utf-8"))["threshold_overrides"]


def _asset(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = TEST_ROOT / path
    assert path.is_file(), f"selected {FAMILY} E2E asset does not exist: {path}"
    return path


def _distance(left: list[str], right: list[str]) -> int:
    a = left
    b = right
    previous = list(range(len(b) + 1))
    for index, char_a in enumerate(a, start=1):
        current = [index]
        for offset, char_b in enumerate(b, start=1):
            current.append(
                min(
                    current[-1] + 1, previous[offset] + 1, previous[offset - 1] + (char_a != char_b)
                )
            )
        previous = current
    return previous[-1]


def _normalized_text_edit_distance(reference: str, hypothesis: str) -> float:
    reference_text = " ".join(reference.casefold().split())
    hypothesis_text = " ".join(hypothesis.casefold().split())
    return _distance(list(reference_text), list(hypothesis_text)) / max(
        len(reference_text), len(hypothesis_text), 1
    )


def _word_error_rate(reference: str, hypothesis: str) -> float:
    reference_words = reference.casefold().split()
    hypothesis_words = hypothesis.casefold().split()
    if not reference_words:
        return 0.0 if not hypothesis_words else 1.0
    return _distance(reference_words, hypothesis_words) / len(reference_words)


def _character_error_rate(reference: str, hypothesis: str) -> float:
    reference_characters = list(reference.casefold())
    hypothesis_characters = list(hypothesis.casefold())
    if not reference_characters:
        return 0.0 if not hypothesis_characters else 1.0
    return _distance(reference_characters, hypothesis_characters) / len(reference_characters)


def _no_speech_state(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return "empty"
    if re.fullmatch(r"\[?\s*blank[\s_-]*audio\s*\]?", stripped, flags=re.IGNORECASE):
        return "blank_audio_token"
    return "speech"


def test_error_rates_keep_word_and_character_semantics() -> None:
    assert _word_error_rate("one two", "one three") == 0.5
    assert _character_error_rate("ab", "ac") == 0.5


def test_tokenizer_bundle_does_not_mutate_model_dir(tmp_path: Path) -> None:
    pytest.importorskip("tensorrt")
    sentencepiece = pytest.importorskip("sentencepiece")
    from families.nemotron_speech_streaming import model as family_model

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    tokenizer = io.BytesIO()
    sentencepiece.SentencePieceTrainer.Train(
        sentence_iterator=iter(("hello world", "hello tokenizer")),
        model_writer=tokenizer,
        vocab_size=32,
        hard_vocab_limit=False,
        minloglevel=2,
    )
    archive = model_dir / "checkpoint.nemo"
    with tarfile.open(archive, "w") as nemo:
        member = tarfile.TarInfo("artifacts/tokenizer.model")
        member.size = len(tokenizer.getvalue())
        nemo.addfile(member, io.BytesIO(tokenizer.getvalue()))

    original_files = set(model_dir.iterdir())
    model_dir.chmod(0o555)
    try:
        runtime, artifacts = family_model._tokenizer_bundle_artifacts(model_dir)
    finally:
        model_dir.chmod(0o755)

    assert runtime["tokenizer_add_special_tokens"] is False
    assert set(artifacts) == {
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model",
    }
    assert set(model_dir.iterdir()) == original_files


def _native(
    binary: Path,
    runtime_root: Path,
    bundle: Path,
    model_dir: Path,
    manifest: dict,
    case: dict,
    tmp_path: Path,
):
    manifest["task"]
    import soundfile as sf

    audio = _asset(case["test_input_audio"])
    sample_rate = int(sf.info(audio).samplerate)
    chunk_samples = sample_rate * int(case["chunk_ms"]) // 1000
    arguments = ["--input", str(audio), "--chunk-samples", str(chunk_samples)]
    for field, option in (
        ("max_new_tokens", "--max-new-tokens"),
        ("att_context_left", "--att-context-left"),
        ("att_context_right", "--att-context-right"),
        ("language", "--language"),
    ):
        if field in case:
            arguments.extend((option, str(case[field])))
    payload = _run_json(
        binary,
        runtime_root,
        bundle,
        manifest,
        case,
        "transcribe-streaming",
        *arguments,
    )
    payload["text"] = str(payload["final"]["text"])
    return payload


def test_nemotron35_streaming_forwards_its_checkpoint_contract(monkeypatch, tmp_path: Path) -> None:
    import sys
    from types import ModuleType, SimpleNamespace

    captured = {}

    def fake_run_json(*args):
        captured["arguments"] = args[6:]
        return {"final": {"text": "transcript"}}

    monkeypatch.setattr(
        "families.nemotron_speech_streaming.tests.test_e2e._run_json", fake_run_json
    )
    soundfile = ModuleType("soundfile")
    soundfile.info = lambda _path: SimpleNamespace(samplerate=16000)
    monkeypatch.setitem(sys.modules, "soundfile", soundfile)
    _, manifest, case = CASES["nemotron-3.5-asr-streaming-0.6b"]

    result = _native(
        Path("/trtmc"),
        Path("/runtime"),
        tmp_path / "model.bundle",
        tmp_path,
        manifest,
        case,
        tmp_path,
    )

    assert result["text"] == "transcript"
    arguments = captured["arguments"]
    for option, value in (
        ("--max-new-tokens", "80"),
        ("--att-context-left", "56"),
        ("--att-context-right", "13"),
        ("--language", "en-US"),
    ):
        assert arguments[arguments.index(option) + 1] == value


def _load_nemotron35_reference(archive: Path, reference_precision: str):
    import torch
    from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModelWithPrompt
    from nemo.core.connectors.save_restore_connector import SaveRestoreConnector

    class Nemotron35SaveRestoreConnector(SaveRestoreConnector):
        def load_instance_with_state_dict(self, instance, state_dict, strict) -> None:
            del strict
            incompatible = instance.load_state_dict(state_dict, strict=False)
            missing = frozenset(incompatible.missing_keys)
            unexpected = frozenset(incompatible.unexpected_keys)
            if missing not in (frozenset(), _NEMOTRON35_OPTIONAL_CTC_STATE_KEYS) or unexpected:
                raise RuntimeError(
                    "Nemotron 3.5 ASR archive state_dict mismatch: "
                    f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
                )
            instance._set_model_restore_state(is_being_restored=False)

    device = torch.device("cuda")
    model = EncDecHybridRNNTCTCBPEModelWithPrompt.restore_from(
        str(archive),
        map_location=device,
        strict=False,
        save_restore_connector=Nemotron35SaveRestoreConnector(),
    )
    model.eval()
    reference_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
    }[reference_precision]
    model.to(device=device, dtype=reference_dtype)
    return torch, model


def _extend_nemotron35_prompt(torch_module, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
    prompt = kwargs.get("prompt")
    if prompt is None or prompt.shape[1] == 0:
        return args, kwargs
    updated = dict(kwargs)
    updated["prompt"] = torch_module.cat((prompt, prompt[:, -1:, :]), dim=1)
    return args, updated


def _official_reference(model_dir: Path, manifest: dict, case: dict, tmp_path: Path):
    manifest["task"]
    from math import gcd

    import numpy as np
    import soundfile as sf
    import torch
    from nemo.collections.asr.models import ASRModel
    from scipy.signal import resample_poly

    archives = sorted(model_dir.glob("*.nemo"))
    if not archives:
        raise FileNotFoundError(f"Nemotron speech NeMo archive is missing under {model_dir}")

    audio, sample_rate = sf.read(
        _asset(case["test_input_audio"]),
        dtype="float32",
        always_2d=True,
    )
    audio = np.asarray(audio, dtype=np.float32).mean(axis=1)
    target_rate = 16000
    if sample_rate != target_rate:
        divisor = gcd(int(sample_rate), target_rate)
        audio = resample_poly(
            audio,
            target_rate // divisor,
            int(sample_rate) // divisor,
        ).astype(np.float32)
    reference_audio = tmp_path / "nemotron-reference.wav"
    sf.write(reference_audio, audio, target_rate, subtype="PCM_16")

    is_nemotron35 = "nemotron-3.5" in str(manifest["name"]).lower()
    reference_precision = str(case["reference_precision"])
    if is_nemotron35:
        torch_module, model = _load_nemotron35_reference(archives[0], reference_precision)
    else:
        torch_module = torch
        reference_dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp32": torch.float32,
        }[reference_precision]
        device = torch.device("cuda")
        model = ASRModel.restore_from(
            restore_path=str(archives[0]),
            map_location=device,
        )
        model.eval()
        model.to(device=device, dtype=reference_dtype)

    record = {
        "audio_filepath": str(reference_audio),
        "duration": len(audio) / target_rate,
        "text": "",
    }
    if is_nemotron35:
        language = str(case.get("language") or "")
        if not language:
            raise ValueError("Nemotron 3.5 ASR reference requires testcase language")
        record["lang"] = language
    manifest_path = tmp_path / "nemotron-reference.jsonl"
    manifest_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

    original_forward = model.forward
    if is_nemotron35:

        def forward_with_extended_prompt(*args, **kwargs):
            args, kwargs = _extend_nemotron35_prompt(torch_module, args, kwargs)
            return original_forward(*args, **kwargs)

        model.forward = forward_with_extended_prompt
    try:
        options = {"batch_size": 1}
        if is_nemotron35:
            options["verbose"] = False
        transcriptions = model.transcribe(str(manifest_path), **options)
    finally:
        model.forward = original_forward

    value = transcriptions[0] if isinstance(transcriptions, tuple) else transcriptions
    value = value[0] if isinstance(value, list) else value
    return {"text": str(value.text if hasattr(value, "text") else value)}


def _assert_parity(actual, expected, manifest: dict, case: dict, thresholds: dict) -> None:
    manifest["task"]
    reference = str(expected["text"])
    hypothesis = str(actual["text"])
    if case.get("name") == "nemotron-3.5-asr-streaming-0.6b":
        if reference and hypothesis:
            assert _word_error_rate(reference, hypothesis) <= float(thresholds.get("wer", 0.1))
            assert _character_error_rate(reference, hypothesis) <= float(
                thresholds.get("cer", 0.05)
            )
        return

    assert reference.strip(), "official ASR reference produced an empty transcript"
    assert _no_speech_state(hypothesis) == _no_speech_state(reference)
    ned_threshold = thresholds.get(
        "contract_ned_threshold", thresholds.get("normalized_text_edit_distance", 0.1)
    )
    wer_threshold = thresholds.get("contract_wer_threshold", thresholds.get("wer", 0.1))
    cer_threshold = thresholds.get("contract_cer_threshold", thresholds.get("cer", 0.1))
    assert _normalized_text_edit_distance(reference, hypothesis) <= float(ned_threshold)
    assert _word_error_rate(reference, hypothesis) <= float(wer_threshold)
    assert _character_error_rate(reference, hypothesis) <= float(cer_threshold)


def test_contract_rejects_empty_reference_and_no_speech_state_mismatch() -> None:
    manifest = {"task": "transcription_streaming"}
    thresholds = {"contract_ned_threshold": 0.1}
    with pytest.raises(AssertionError, match="empty transcript"):
        _assert_parity({"text": ""}, {"text": ""}, manifest, {}, thresholds)
    with pytest.raises(AssertionError):
        _assert_parity({"text": "[blank audio]"}, {"text": "hello"}, manifest, {}, thresholds)
    with pytest.raises(AssertionError):
        _assert_parity(
            {"text": "ab"},
            {"text": "ac"},
            manifest,
            {},
            {"contract_ned_threshold": 1.0, "wer": 1.0, "cer": 0.0},
        )


def test_nemotron35_uses_only_the_active_speech_comparator_metrics() -> None:
    _assert_parity(
        {"text": "[blank audio]"},
        {"text": "hello"},
        {"task": "transcription_streaming"},
        {"name": "nemotron-3.5-asr-streaming-0.6b"},
        {"wer": 2.0, "cer": 3.0},
    )


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    _, manifest, case = CASES[case_name]
    model_dir = _model_dir(manifest)
    binary, runtime_root = _runtime(manifest)
    bundle = tmp_path / manifest["bundle"]
    _build(model_dir, bundle, manifest)
    actual = _native(binary, runtime_root, bundle, model_dir, manifest, case, tmp_path)
    expected = _official_reference(model_dir, manifest, case, tmp_path)
    _assert_parity(actual, expected, manifest, case, _thresholds(case_name))
