# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native CLI, C and C++ SDK, and official-reference parity for Granite Speech 5 CTC.

Every case reads a 16 kHz mono WAV of the family's test data (optionally its first ``samples``
values), written again as float32 so every reader sees the same samples. The oracle is the official
GraniteSpeech5ForCTC in FP32 on the GPU (TF32 off), used as the model card does: the processor's
features, ``generate`` (greedy CTC) and ``batch_decode(skip_special_tokens=True)``. Its frame ids
are collapsed as ParakeetTokenizer collapses them (repeats, then the blank).

FP32 bundles must reproduce the collapsed ids and the transcript exactly. BF16 bundles are held to
the FP32 reference within an edit-distance bound on the ids and on the transcript
(``BF16_MAX_TOKEN_ERROR``, ``BF16_MAX_CHARACTER_ERROR``).
"""

from __future__ import annotations

import json
import os
import subprocess
from fnmatch import fnmatch
from itertools import groupby
from pathlib import Path

import numpy as np
import pytest

from tensorrt_model_connect import BuildRequest, build
from tools.e2e_evidence import evidence_stage, mark_loaded, record_evidence

FAMILY = "granite_speech5"
TASKS = frozenset({"speech_transcription"})
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"
SAMPLE_RATE = 16000
BLANK_ID = 0
# The one case of a testcase-only selection that names none of this family's cases: another
# collected family may own it, or it is a typo.
UNMATCHED = "unmatched-testcase-selection"
# BF16 against the FP32 reference: edit distance over the reference length, ids and characters.
# Measured on one B300 on the three clips of the manifests: 0.0 for the BF16 and FP32 bundles and for the
# official model in BF16. BF16 bundles built broken on purpose score 0.20-0.90 on the ids and 0.10-0.66 on
# the characters (relative positions read with the wrong sign) or 0.079-0.70 and 0.036-0.50 (subsampling
# blocks without their pooled residual). 0.03 still admits two flipped near-tie frames of the 89-token
# LibriSpeech transcript (its smallest FP32 top-2 margin is 0.048). An engine without self-conditioning keeps
# these transcripts (it moves 5-7 % of the frame alignments only); test_graph.py checks that part.
BF16_MAX_TOKEN_ERROR = 0.03
BF16_MAX_CHARACTER_ERROR = 0.03


def _manifest_value(container: dict, key: str, kinds: tuple[type, ...]):
    """A manifest value of its own JSON type, never coerced (a boolean is not a number)."""
    value = container[key]
    if isinstance(value, bool) or not isinstance(value, kinds):
        names = " or ".join(kind.__name__ for kind in kinds)
        raise TypeError(f"{FAMILY} manifest {key} must be a JSON {names}, not {value!r}")
    return value


def _case_index() -> dict[str, tuple[Path, dict, dict]]:
    result = {}
    for path in sorted(MANIFEST_ROOT.glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["family"] == FAMILY
        assert manifest["task"] in TASKS
        for case in manifest["testcases"]:
            name = _manifest_value(case, "name", (str,))
            assert name not in result
            result[name] = (path, manifest, case)
    return result


CASES = _case_index()


def _filters(config) -> tuple[set[str], set[str]]:
    model_filters = set()
    for raw in config.getoption("--e2e-model") or []:
        model_filters.update(item.strip() for item in str(raw).split(",") if item.strip())
    models_file = config.getoption("--e2e-models-file")
    if models_file:
        model_filters.update(
            line.strip()
            for line in Path(models_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    testcase_filters = set()
    for raw in config.getoption("--e2e-testcase") or []:
        testcase_filters.update(item.strip() for item in str(raw).split(",") if item.strip())
    return model_filters, testcase_filters


def _selected_cases(config) -> tuple[list[str], bool]:
    model_filters, testcase_filters = _filters(config)
    if not model_filters and not testcase_filters:
        return sorted(CASES), False
    selected = []
    for name, (_, manifest, _) in CASES.items():
        model_match = (not model_filters or FAMILY in model_filters or name in model_filters
                       or manifest["name"] in model_filters)
        if model_match and (not testcase_filters or name in testcase_filters):
            selected.append(name)
    named = FAMILY in model_filters or any(
        name in model_filters or manifest["name"] in model_filters for name, (_, manifest, _) in CASES.items())
    if named and not selected:
        # Pytest skips an empty parametrization, so a selection that names this family but
        # matches none of its cases would pass without testing.
        raise pytest.UsageError(
            f"the E2E selection names {FAMILY}, but --e2e-testcase {', '.join(sorted(testcase_filters))} "
            f"matches none of its cases: {', '.join(sorted(CASES))}")
    return sorted(selected), True


def _resolve_unmatched(request) -> None:
    """Skip when another collected family's E2E owns the selected testcase; fail otherwise."""
    _, wanted = _filters(request.config)
    owned = set()
    for item in request.session.items:
        params = getattr(getattr(item, "callspec", None), "params", {})
        name = getattr(item, "originalname", None) or item.name
        if fnmatch(name, "test_*e2e*") and params.get("case_name") not in (None, UNMATCHED):
            owned.add(params["case_name"])
    if wanted & owned:
        pytest.skip(f"another collected family owns --e2e-testcase {', '.join(sorted(wanted))}")
    pytest.fail(f"--e2e-testcase {', '.join(sorted(wanted))} matches no collected E2E case; "
                f"{FAMILY} has {', '.join(sorted(CASES))}")


def pytest_generate_tests(metafunc) -> None:
    if "case_name" in metafunc.fixturenames:
        names, enabled = _selected_cases(metafunc.config)
        model_filters, testcase_filters = _filters(metafunc.config)
        parameters = names
        if enabled and not names and testcase_filters and not model_filters:
            # Pytest would skip the empty parametrization; whether the testcase is a typo is known
            # only once the whole session is collected.
            parameters = names = [UNMATCHED]
        if not enabled:
            parameters = [
                pytest.param(name, marks=pytest.mark.skip(
                    reason="direct E2E requires one of the three explicit E2E selectors"))
                for name in names
            ]
        metafunc.parametrize("case_name", parameters, ids=names)


def _required_path(value: str | None, label: str) -> Path:
    assert value, f"selected {FAMILY} E2E requires {label}"
    path = Path(value)
    assert path.exists(), f"selected {FAMILY} E2E {label} does not exist: {path}"
    return path


def _model_dir(manifest: dict) -> Path:
    explicit = os.environ.get("TRTMC_GRANITE_SPEECH5_MODEL_DIR")
    if explicit:
        return _required_path(explicit, "TRTMC_GRANITE_SPEECH5_MODEL_DIR")
    from huggingface_hub import snapshot_download

    try:
        snapshot = snapshot_download(repo_id=manifest["hf_id"], revision=manifest["hf_revision"],
                                     local_files_only=True)
    except Exception as error:  # noqa: BLE001 - every cache miss is the same failure
        raise AssertionError(f"selected {FAMILY} E2E requires the exact cached checkpoint {manifest['hf_id']}") \
            from error
    return Path(snapshot)


def _runtime() -> tuple[Path, Path]:
    binary = _required_path(os.environ.get("TRTMC_BINARY"), "TRTMC_BINARY")
    runtime_root = _required_path(os.environ.get("TRTMC_RUNTIME_ROOT"), "TRTMC_RUNTIME_ROOT")
    assert (runtime_root / "libtrtmc_backend_trt.so").is_file()
    assert (runtime_root / f"libtrtmc_model_{FAMILY}.so").is_file()
    import torch

    assert torch.cuda.is_available(), f"selected {FAMILY} E2E requires CUDA"
    return binary, runtime_root


def _sdk_consumers() -> dict[str, Path]:
    native = _required_path(os.environ.get("TRTMC_NATIVE_BUILD_DIR"), "TRTMC_NATIVE_BUILD_DIR")
    targets = [f"test_{FAMILY}_sdk_c", f"test_{FAMILY}_sdk_cpp"]
    subprocess.run(["cmake", "--build", str(native), "--parallel", "8", "--target", *targets],
                   check=True, timeout=900)
    consumers = {target.rsplit("_", 1)[1]: native / target for target in targets}
    for path in consumers.values():
        assert path.is_file(), f"native build did not produce {path.name}"
    return consumers


def _write_wav(path: Path, samples: np.ndarray, rate: int = SAMPLE_RATE) -> Path:
    import soundfile

    path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(str(path), samples, rate, subtype="FLOAT")
    return path


def _case_audio(case: dict, directory: Path) -> tuple[Path, np.ndarray]:
    """The case's samples, written as a float32 mono WAV that the CLI and the reference both read."""
    import soundfile

    source = TEST_ROOT / _manifest_value(case, "test_input_audio", (str,))
    assert source.is_file(), f"selected {FAMILY} E2E input does not exist: {source}"
    audio, rate = soundfile.read(str(source), dtype="float32", always_2d=True)
    assert rate == SAMPLE_RATE and audio.shape[1] == 1, f"{source.name} is not 16 kHz mono"
    samples = np.ascontiguousarray(audio[:, 0])
    if "samples" in case:
        count = _manifest_value(case, "samples", (int,))
        assert 0 < count <= samples.shape[0], f"{case['name']} asks for {count} of {samples.shape[0]} samples"
        samples = np.ascontiguousarray(samples[:count])
    return _write_wav(directory / f"{case['name']}.wav", samples), samples


def _json_line(stdout: str, label: str) -> dict:
    payloads = [json.loads(line[line.find("{"):]) for line in stdout.splitlines() if line.strip().startswith("{")]
    assert len(payloads) == 1, f"{label} printed {len(payloads)} JSON lines: {stdout[-2000:]}"
    return payloads[0]


def _environment(runtime_root: Path) -> dict:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(v for v in (str(runtime_root), env.get("LD_LIBRARY_PATH", "")) if v)
    return env


def _native_cli(binary: Path, runtime_root: Path, bundle: Path, wav: Path) -> dict:
    completed = subprocess.run(
        [str(binary), "transcribe", str(bundle), "--runtime-root", str(runtime_root), "--input", str(wav)],
        check=True, capture_output=True, text=True, env=_environment(runtime_root), timeout=1800)
    return _json_line(completed.stdout, "trtmc transcribe")


def _native_sdk(consumer: Path, runtime_root: Path, bundle: Path, samples: np.ndarray, directory: Path) -> dict:
    pcm = directory / "sdk-input.f32"
    samples.astype("<f4").tofile(pcm)
    completed = subprocess.run(
        [str(consumer), str(bundle), str(runtime_root), str(pcm), str(SAMPLE_RATE), "1"],
        check=True, capture_output=True, text=True, env=_environment(runtime_root), timeout=1800)
    return _json_line(completed.stdout, consumer.name)


def _official(model_dir: Path, samples: np.ndarray, dtype) -> dict:
    """The model card's usage of GraniteSpeech5ForCTC on the GPU, TF32 off and restored afterwards."""
    import torch
    from transformers import AutoModelForCTC, AutoProcessor

    saved = torch.get_float32_matmul_precision(), torch.backends.cudnn.allow_tf32
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    try:
        processor = AutoProcessor.from_pretrained(str(model_dir))
        model = AutoModelForCTC.from_pretrained(str(model_dir), dtype=dtype).to("cuda").eval()
        inputs = processor(samples, sampling_rate=SAMPLE_RATE, return_tensors="pt").to("cuda")
        with torch.no_grad():
            frames = model.generate(input_features=inputs["input_features"].to(dtype),
                                    attention_mask=inputs["attention_mask"])
        text = processor.batch_decode(frames, skip_special_tokens=True)[0]
        frame_ids = frames[0].tolist()
        del model
        torch.cuda.empty_cache()
    finally:
        torch.set_float32_matmul_precision(saved[0])
        torch.backends.cudnn.allow_tf32 = saved[1]
    ids = [token for token, _ in groupby(frame_ids) if token != BLANK_ID]
    return {"text": text, "token_ids": ids, "frames": len(frame_ids)}


def _edit_distance(first, second) -> int:
    previous = list(range(len(second) + 1))
    for index, left in enumerate(first, 1):
        current = [index]
        for column, right in enumerate(second, 1):
            current.append(min(previous[column] + 1, current[column - 1] + 1, previous[column - 1] + (left != right)))
        previous = current
    return previous[-1]


def _error_rate(hypothesis, reference) -> float:
    return _edit_distance(hypothesis, reference) / max(len(reference), 1)


@pytest.fixture(scope="module")
def bundles(tmp_path_factory) -> dict:
    """Bundles built in this session, one per manifest: its cases share the build."""
    return {"root": tmp_path_factory.mktemp("bundles")}


def _bundle(bundles: dict, model_dir: Path, manifest: dict) -> Path:
    bundle = bundles["root"] / _manifest_value(manifest, "bundle", (str,))
    if manifest["name"] not in bundles:
        with evidence_stage("build"):
            build(BuildRequest(model_dir=model_dir, output_path=bundle, family=FAMILY, task=manifest["task"],
                               precision=_manifest_value(manifest, "precision", (str,)),
                               tensor_parallel_size=_manifest_value(manifest, "tensor_parallel_size", (int,))))
        bundles[manifest["name"]] = bundle
    assert bundle.is_file(), f"{FAMILY} bundle was not produced: {bundle}"
    return bundle


def _versions() -> dict:
    """The environment the evidence was produced in."""
    import tensorrt
    import torch
    import torchaudio
    import transformers

    def command(*arguments):
        try:
            completed = subprocess.run(arguments, capture_output=True, text=True, cwd=TEST_ROOT, timeout=60)
        except (OSError, subprocess.TimeoutExpired) as error:
            return f"unavailable ({type(error).__name__})"
        return completed.stdout.strip() or completed.stderr.strip()

    return {
        "head": command("git", "rev-parse", "HEAD"),
        "gpu": torch.cuda.get_device_name(0),
        "driver": command("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"),
        "tensorrt": tensorrt.__version__,
        "torch": torch.__version__,
        "torchaudio": torchaudio.__version__,
        "transformers": transformers.__version__,
    }


def _refused(binary: Path, runtime_root: Path, bundle: Path, wav: Path, options: list[str], message: str) -> None:
    completed = subprocess.run(
        [str(binary), "transcribe", str(bundle), "--runtime-root", str(runtime_root), "--input", str(wav), *options],
        capture_output=True, text=True, env=_environment(runtime_root), timeout=600)
    assert completed.returncode != 0, f"accepted a request it must refuse: {options} {wav.name}"
    assert message in completed.stderr, f"{options} {wav.name}: {completed.stderr[-1000:]}"


def _check_contract(binary: Path, runtime_root: Path, bundle: Path, samples: np.ndarray, mono: dict,
                    directory: Path) -> None:
    """Stereo is averaged like the reference; every request outside the contract fails with its reason."""
    stereo = _native_cli(binary, runtime_root, bundle,
                         _write_wav(directory / "stereo.wav", np.stack([samples, samples], axis=1)))
    assert (stereo["text"], stereo["token_ids"]) == (mono["text"], mono["token_ids"]), \
        "a stereo clip of two equal channels must transcribe as the mono clip"
    tiny = samples[:1119]
    longest = np.resize(samples, 480001)
    broken = samples.copy()
    broken[len(broken) // 2] = np.nan
    _refused(binary, runtime_root, bundle, _write_wav(directory / "rate.wav", samples, 48000), [],
             "requires 16000 Hz audio, not 48000 Hz")
    _refused(binary, runtime_root, bundle, _write_wav(directory / "short.wav", tiny), [],
             "accepts 1120 to 480000 samples, not 1119")
    _refused(binary, runtime_root, bundle, _write_wav(directory / "long.wav", longest), [],
             "accepts 1120 to 480000 samples, not 480001")
    _refused(binary, runtime_root, bundle, _write_wav(directory / "nan.wav", broken), [],
             "non-finite audio sample")
    _refused(binary, runtime_root, bundle, directory / "stereo.wav", ["--source-language", "en"],
             "omit source_language")
    _refused(binary, runtime_root, bundle, directory / "stereo.wav", ["--set", "beam_size=2"],
             "does not declare config 'beam_size'")


def _check(name: str, label: str, actual, operator: str, expected) -> dict:
    passed = {"==": actual == expected, "<=": actual <= expected}[operator]
    return {"name": name, "label": label, "scope": "independent_reference", "actual": actual,
            "operator": operator, "expected": expected, "passed": passed}


def _compare(native: dict, reference: dict, precision: str, directory: Path, label: str) -> list[dict]:
    """FP32: ids and transcript identical; BF16: both within their edit-distance bound.

    The ids of both sides are kept as .npy files, the comparison is recorded with every check.
    """
    native_ids, reference_ids = [int(value) for value in native["token_ids"]], reference["token_ids"]
    native_path, reference_path = directory / "native_token_ids.npy", directory / "reference_token_ids.npy"
    np.save(native_path, np.asarray(native_ids, dtype=np.int32))
    np.save(reference_path, np.asarray(reference_ids, dtype=np.int32))
    if precision == "fp32":
        checks = [
            _check("token_edit_distance", "Token edit distance", _edit_distance(native_ids, reference_ids), "==", 0),
            _check("character_edit_distance", "Transcript edit distance",
                   _edit_distance(native["text"], reference["text"]), "==", 0),
        ]
    else:
        checks = [
            _check("token_error_rate", "Token edit distance / reference tokens",
                   _error_rate(native_ids, reference_ids), "<=", BF16_MAX_TOKEN_ERROR),
            _check("character_error_rate", "Transcript edit distance / reference characters",
                   _error_rate(native["text"], reference["text"]), "<=", BF16_MAX_CHARACTER_ERROR),
        ]
    record_evidence("reference_comparison", {
        "label": label, "scope": "independent_reference", "enforced": True,
        "native": native_path, "reference": reference_path, "checks": checks,
    })
    return checks


def test_e2e(tmp_path: Path, bundles: dict, case_name: str, request) -> None:
    if case_name == UNMATCHED:
        _resolve_unmatched(request)
    manifest_path, manifest, case = CASES[case_name]
    assert manifest_path.is_file()
    binary, runtime_root = _runtime()
    consumers = _sdk_consumers()
    model_dir = _model_dir(manifest)
    wav, samples = _case_audio(case, tmp_path / "audio")
    record_evidence("inputs", {"audio": wav, "sample_rate": SAMPLE_RATE, "samples": int(samples.shape[0]),
                               "source": case["test_input_audio"], "manifest": manifest["name"]})
    bundle = _bundle(bundles, model_dir, manifest)
    versions = record_evidence("environment", _versions())
    print(f"{FAMILY} environment: {json.dumps(versions)}")
    with evidence_stage("native"):
        native = _native_cli(binary, runtime_root, bundle, wav)
        sdk = {language: _native_sdk(consumer, runtime_root, bundle, samples, tmp_path)
               for language, consumer in consumers.items()}
    mark_loaded()
    for language, result in sdk.items():
        assert result == {"text": native["text"], "token_ids": native["token_ids"]}, f"{language} SDK differs from the CLI"
    checked = bundles.setdefault("contract checked", set())
    if manifest["name"] not in checked:
        with evidence_stage("contract"):
            _check_contract(binary, runtime_root, bundle, samples, native, tmp_path / "contract")
        checked.add(manifest["name"])
    with evidence_stage("reference"):
        import torch

        reference = _official(model_dir, samples, torch.float32)
    native_text, native_ids = native["text"], [int(value) for value in native["token_ids"]]
    reference_text, reference_ids = reference["text"], reference["token_ids"]
    record_evidence("native", {"text": native_text, "token_ids": native_ids})
    record_evidence("reference", {"reference_text": reference_text, "reference_ids": reference_ids,
                                  "frames": reference["frames"], "precision": "fp32"})
    assert reference_text.strip(), "the reference transcript is empty, so the comparison proves nothing"
    precision = manifest["precision"]
    if precision != "fp32":
        record_evidence("thresholds", {"max_token_error_rate": BF16_MAX_TOKEN_ERROR,
                                       "max_character_error_rate": BF16_MAX_CHARACTER_ERROR})
    checks = _compare(native, reference, precision, tmp_path, f"{case_name} against the FP32 reference")
    print(f"{FAMILY} {case_name}: " + ", ".join(f"{check['name']} {check['actual']}" for check in checks))
    failed = [check for check in checks if not check["passed"]]
    assert not failed, f"{case_name}: {failed}; native {native_text!r}, reference {reference_text!r}"
    if precision == "fp32":
        assert native_ids == reference_ids
        assert native_text == reference_text
    print(f"{FAMILY} parity: {len(reference_ids)} ids over {reference['frames']} frames, transcript {reference_text!r}")


def test_manifest_cases_read_family_audio_only():
    for _, manifest, case in CASES.values():
        assert manifest["hf_revision"] and manifest["precision"] in {"bf16", "fp32"}
        assert (TEST_ROOT / case["test_input_audio"]).is_file()
        assert _manifest_value(manifest, "tensor_parallel_size", (int,)) == 1
        if "samples" in case:
            assert _manifest_value(case, "samples", (int,)) >= 1120
