# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime, and official-reference E2E for canary."""

from __future__ import annotations
import io
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path
import pytest
from tensorrt_model_connect import BuildRequest, build

FAMILY = "canary"
TASKS = frozenset({"transcription"})
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"
THRESHOLD_ROOT = TEST_ROOT / "thresholds"


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


def test_runtime_tokenizer_document_uses_array_vocab(monkeypatch, tmp_path: Path) -> None:
    import sys
    from types import SimpleNamespace

    class Processor:
        def unk_id(self):
            return 0

        def GetPieceSize(self):
            return 2

        def IdToPiece(self, index):
            return ("<unk>", "▁hello")[index]

    monkeypatch.setitem(
        sys.modules,
        "sentencepiece",
        SimpleNamespace(SentencePieceProcessor=lambda **kwargs: Processor()),
    )
    from families.canary.model import _runtime_tokenizer_document

    document = _runtime_tokenizer_document(tmp_path / "tokenizer.model")
    assert document["model"] == {
        "type": "Unigram",
        "unk_id": 0,
        "vocab": [["<unk>", 0.0], ["▁hello", 0.0]],
    }


def test_build_extracts_tokenizer_without_mutating_read_only_snapshot(
    monkeypatch, tmp_path: Path
) -> None:
    from types import SimpleNamespace

    from families.canary import model

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text('{"model_type":"canary"}', encoding="utf-8")
    snapshot_tokenizer = {"version": "1.0", "source": "checkpoint"}
    (snapshot / "tokenizer.json").write_text(json.dumps(snapshot_tokenizer), encoding="utf-8")
    snapshot_tokenizer_config = {
        "tokenizer_class": "CanaryTokenizer",
        "model_max_length": 4096,
    }
    (snapshot / "tokenizer_config.json").write_text(
        json.dumps(snapshot_tokenizer_config), encoding="utf-8"
    )
    tokenizer_bytes = b"canary sentencepiece"
    with tarfile.open(snapshot / "canary.nemo", "w") as archive:
        member = tarfile.TarInfo("artifacts/tokenizer_spe_bpe_v1024.model")
        member.size = len(tokenizer_bytes)
        archive.addfile(member, io.BytesIO(tokenizer_bytes))

    tokenizer_document = {
        "version": "1.0",
        "model": {"type": "Unigram", "unk_id": 0, "vocab": [["<unk>", 0.0]]},
    }
    tokenizer_dirs = []

    def tokenizer_json(path: Path) -> dict[str, object]:
        assert path.read_bytes() == tokenizer_bytes
        return tokenizer_document

    def load_weights(self, model_dir: str, config, tokenizer_dir: Path):
        assert Path(model_dir) == snapshot
        assert tokenizer_dir != snapshot
        tokenizer_dirs.append(tokenizer_dir)
        extracted = model._extract_tokenizer_from_nemo(
            model_dir,
            tokenizer_dir,
            {"tokenizer": {"model_path": "nemo:tokenizer_spe_bpe_v1024.model"}},
        )
        assert extracted == tokenizer_dir / "tokenizer.model"
        config.hidden_size = 16
        config.vocab_size = 32
        config.num_hidden_layers = 2
        config.num_attention_heads = 2
        return {}

    def tokenizer_contract(path: Path) -> dict[str, object]:
        assert path != snapshot
        assert (path / "tokenizer.model").read_bytes() == tokenizer_bytes
        assert json.loads((path / "tokenizer.json").read_text(encoding="utf-8")) == (
            snapshot_tokenizer
        )
        return {
            "tokenizer_add_special_tokens": False,
            "tokenizer_prefix_ids": [],
            "tokenizer_suffix_ids": [],
        }

    monkeypatch.setattr(model, "_runtime_tokenizer_document", tokenizer_json)
    monkeypatch.setattr(model, "_tokenizer_runtime_contract", tokenizer_contract)
    monkeypatch.setattr(model._CanaryModel, "load_weights", load_weights)
    monkeypatch.setattr(model._CanaryModel, "build_engine", lambda *args, **kwargs: b"decoder")
    monkeypatch.setattr(
        model._CanaryModel, "build_vision_engine", lambda *args, **kwargs: b"encoder"
    )
    monkeypatch.setattr(
        model._CanaryModel,
        "build_extra_engines",
        lambda *args, **kwargs: {"mel_filterbank": b"mel"},
    )

    snapshot.chmod(0o555)
    before = {path.name: path.read_bytes() for path in snapshot.iterdir()}
    sections = {}
    writer = SimpleNamespace(
        set_header=lambda **value: None,
        add_bytes=lambda name, value: sections.__setitem__(name, value),
        add_json=lambda name, value: sections.__setitem__(name, value),
    )
    try:
        model.build(
            BuildRequest(
                model_dir=snapshot,
                output_path=tmp_path / "canary.bundle",
                family=FAMILY,
                task="transcription",
                precision="fp16",
            ),
            writer,
        )
        assert {path.name: path.read_bytes() for path in snapshot.iterdir()} == before
    finally:
        snapshot.chmod(0o755)

    assert tokenizer_dirs and not tokenizer_dirs[0].exists()
    assert sections["tokenizer.model"] == tokenizer_bytes
    assert sections["tokenizer.json"] == tokenizer_document
    assert json.loads(sections["tokenizer_config.json"]) == (snapshot_tokenizer_config)


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


def test_word_error_rate_keeps_word_semantics() -> None:
    assert _word_error_rate("one two", "one three") == 0.5


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
    audio = _asset(case["test_input_audio"])
    return _run_json(
        binary,
        runtime_root,
        bundle,
        manifest,
        case,
        "transcribe",
        "--input",
        str(audio),
        "--max-output-tokens",
        str(int(case["max_new_tokens"])),
    )


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
        raise FileNotFoundError(f"Canary NeMo archive is missing under {model_dir}")

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
    reference_audio = tmp_path / "canary-reference.wav"
    sf.write(reference_audio, audio, target_rate, subtype="PCM_16")

    device = torch.device("cuda")
    model = ASRModel.restore_from(
        restore_path=str(archives[0]),
        map_location=device,
    )
    reference_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
    }[case["reference_precision"]]
    model.to(device=device, dtype=reference_dtype)
    model.eval()
    transcriptions = model.transcribe([str(reference_audio)], batch_size=1)
    value = transcriptions[0] if isinstance(transcriptions, tuple) else transcriptions
    value = value[0] if isinstance(value, list) else value
    return {"text": str(value.text if hasattr(value, "text") else value)}


def _assert_parity(actual, expected, manifest: dict, case: dict, thresholds: dict) -> None:
    manifest["task"]
    reference = str(expected["text"])
    hypothesis = str(actual["text"])
    assert reference.strip(), "official ASR reference produced an empty transcript"
    ned_threshold = thresholds.get(
        "contract_ned_threshold", thresholds.get("normalized_text_edit_distance", 0.1)
    )
    wer_threshold = thresholds.get("contract_wer_threshold", thresholds.get("wer", 0.1))
    assert _normalized_text_edit_distance(reference, hypothesis) <= float(ned_threshold)
    assert _word_error_rate(reference, hypothesis) <= float(wer_threshold)


def test_contract_rejects_empty_reference() -> None:
    manifest = {"task": "transcription"}
    thresholds = {"contract_ned_threshold": 0.1}
    with pytest.raises(AssertionError, match="empty transcript"):
        _assert_parity({"text": ""}, {"text": ""}, manifest, {}, thresholds)


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    _, manifest, case = CASES[case_name]
    model_dir = _model_dir(manifest)
    binary, runtime_root = _runtime(manifest)
    bundle = tmp_path.parent / manifest["bundle"]
    if not bundle.is_file():
        _build(model_dir, bundle, manifest)
    actual = _native(binary, runtime_root, bundle, model_dir, manifest, case, tmp_path)
    expected = _official_reference(model_dir, manifest, case, tmp_path)
    _assert_parity(actual, expected, manifest, case, _thresholds(case_name))
