# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native CLI, C and C++ SDK, and official-reference parity for Granite Speech.

Inputs are files of the pinned checkpoint snapshot (optionally a leading slice of one, written as
float32 WAV so every reader sees the same samples). The oracle is the official
GraniteSpeechForConditionalGeneration in FP32, prompted and decoded as the model card does
(greedy, the case's max_new_tokens): the generated ids, EOS included, and the decoded transcript
must be identical. The bundle's prompt ids are checked against the official processor's.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tensorrt_model_connect import BuildRequest, build
from tools.e2e_evidence import evidence_stage, record_evidence

FAMILY = "granite_speech"
TASKS = frozenset({"speech_transcription"})
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"
SAMPLE_RATE = 16000


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
    if not model_filters and not testcase_filters:
        return sorted(CASES), False
    selected = []
    for name, (_, manifest, _) in CASES.items():
        model_match = (not model_filters or FAMILY in model_filters or name in model_filters
                       or manifest["name"] in model_filters)
        testcase_match = not testcase_filters or name in testcase_filters
        if model_match and testcase_match:
            selected.append(name)
    return sorted(selected), True


def pytest_generate_tests(metafunc) -> None:
    if "case_name" in metafunc.fixturenames:
        names, enabled = _selected_cases(metafunc.config)
        parameters = names
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
    explicit = os.environ.get("TRTMC_GRANITE_SPEECH_MODEL_DIR")
    if explicit:
        return _required_path(explicit, "TRTMC_GRANITE_SPEECH_MODEL_DIR")
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


def _case_audio(model_dir: Path, case: dict, directory: Path) -> tuple[Path, np.ndarray]:
    """The case's samples, written as a float32 mono WAV the CLI and the reference both read."""
    import soundfile

    audio, rate = soundfile.read(str(model_dir / case["checkpoint_audio"]), dtype="float32", always_2d=True)
    assert rate == SAMPLE_RATE and audio.shape[1] == 1, "the checkpoint audio is 16 kHz mono"
    samples = np.ascontiguousarray(audio[:, 0])
    if "end_seconds" in case:
        samples = samples[: int(round(float(case["end_seconds"]) * SAMPLE_RATE))]
    if "repeat" in case:  # a longer clip: more encoder blocks and a longer prompt
        samples = np.tile(samples, int(case["repeat"]))
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{case['name']}.wav"
    soundfile.write(str(path), samples, SAMPLE_RATE, subtype="FLOAT")
    return path, samples


def _json_line(stdout: str, label: str) -> dict:
    payloads = [json.loads(line[line.find("{"):]) for line in stdout.splitlines() if line.strip().startswith("{")]
    assert len(payloads) == 1, f"{label} printed {len(payloads)} JSON lines: {stdout[-2000:]}"
    return payloads[0]


def _environment(runtime_root: Path) -> dict:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(v for v in (str(runtime_root), env.get("LD_LIBRARY_PATH", "")) if v)
    return env


def _native_cli(binary: Path, runtime_root: Path, bundle: Path, wav: Path, case: dict) -> dict:
    completed = subprocess.run(
        [str(binary), "transcribe", str(bundle), "--runtime-root", str(runtime_root), "--input", str(wav),
         "--set", f"max_new_tokens={int(case['max_new_tokens'])}"],
        check=True, capture_output=True, text=True, env=_environment(runtime_root), timeout=1800)
    return _json_line(completed.stdout, "trtmc transcribe")


def _native_sdk(consumer: Path, runtime_root: Path, bundle: Path, samples: np.ndarray, case: dict,
                directory: Path) -> dict:
    pcm = directory / f"{case['name']}.f32"
    samples.astype("<f4").tofile(pcm)
    completed = subprocess.run(
        [str(consumer), str(bundle), str(runtime_root), str(pcm), str(SAMPLE_RATE), str(int(case["max_new_tokens"]))],
        check=True, capture_output=True, text=True, env=_environment(runtime_root), timeout=1800)
    return _json_line(completed.stdout, consumer.name)


def _official_reference(model_dir: Path, samples: np.ndarray, case: dict) -> dict:
    """The model card's usage in FP32 on the GPU, TF32 off, features computed on the CPU."""
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    from families.granite_speech.prompt import AUDIO_TOKEN, INSTRUCTION, prompt_ids

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    processor = AutoProcessor.from_pretrained(str(model_dir))
    tokenizer = processor.tokenizer
    chat = [{"role": "user", "content": AUDIO_TOKEN + INSTRUCTION}]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = processor(prompt, torch.from_numpy(samples)[None], device="cpu", return_tensors="pt")
    ids = inputs["input_ids"][0].tolist()
    audio_id = tokenizer.convert_tokens_to_ids(AUDIO_TOKEN)
    _, prefix, suffix = prompt_ids(model_dir, audio_id)
    assert ids == prefix + [audio_id] * ids.count(audio_id) + suffix, "bundle prompt ids differ from the processor's"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(str(model_dir), dtype=torch.float32).to("cuda").eval()
    assert torch.get_float32_matmul_precision() == "highest"
    with torch.no_grad():
        out = model.generate(**inputs.to("cuda"), max_new_tokens=int(case["max_new_tokens"]),
                             do_sample=False, num_beams=1)
    new = out[0, len(ids):].tolist()
    text = tokenizer.batch_decode(out[:, len(ids):], add_special_tokens=False, skip_special_tokens=True)[0]
    del model
    torch.cuda.empty_cache()
    return {"text": text, "token_ids": new, "prompt_tokens": len(ids), "audio_tokens": ids.count(audio_id)}


def _assert_parity(native: dict, reference: dict, label: str) -> None:
    assert reference["text"].strip(), "the reference transcript is empty, so the gate proves nothing"
    ids = [int(value) for value in native["token_ids"]]
    expected = reference["token_ids"]
    first = next((i for i, (a, b) in enumerate(zip(ids, expected)) if a != b), min(len(ids), len(expected)))
    assert ids == expected, f"{label}: ids differ from position {first} ({len(ids)} native, {len(expected)} reference)"
    assert native["text"] == reference["text"], f"{label}: {native['text']!r} != {reference['text']!r}"


@pytest.fixture(scope="module")
def bundles(tmp_path_factory) -> dict:
    """Bundles built in this session, one per manifest: its cases share the build."""
    return {"root": tmp_path_factory.mktemp("bundles")}


def _bundle(bundles: dict, model_dir: Path, manifest: dict) -> Path:
    bundle = bundles["root"] / manifest["bundle"]
    if manifest["name"] not in bundles:
        with evidence_stage("build"):
            build(BuildRequest(model_dir=model_dir, output_path=bundle, family=FAMILY, task=manifest["task"],
                               precision=manifest["precision"],
                               max_sequence_length=manifest.get("max_sequence_length"),
                               tensor_parallel_size=int(manifest["tensor_parallel_size"])))
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
        "cuda_torch": torch.version.cuda,
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


def _assert_refusals(binary: Path, runtime_root: Path, bundle: Path, manifest: dict, samples: np.ndarray,
                     directory: Path) -> None:
    """Each request outside the family's contract fails with its reason instead of transcribing."""
    import soundfile

    directory.mkdir(parents=True, exist_ok=True)
    clips = {
        "rate": (samples, 48000), "stereo": (np.stack([samples, samples], axis=1), SAMPLE_RATE),
        "short": (samples[:200], SAMPLE_RATE), "valid": (samples, SAMPLE_RATE),
    }
    paths = {}
    for name, (data, rate) in clips.items():
        paths[name] = directory / f"refusal-{name}.wav"
        soundfile.write(str(paths[name]), data, rate, subtype="FLOAT")
    capacity = int(manifest.get("max_sequence_length") or 4096)
    _refused(binary, runtime_root, bundle, paths["rate"], [], "requires 16000 Hz audio, not 48000 Hz")
    _refused(binary, runtime_root, bundle, paths["stereo"], [], "requires mono audio, not 2 channels")
    _refused(binary, runtime_root, bundle, paths["short"], [], "samples, not 200")
    _refused(binary, runtime_root, bundle, paths["valid"], ["--source-language", "en"], "omit source_language")
    _refused(binary, runtime_root, bundle, paths["valid"], ["--set", "max_new_tokens=0"], "max_new_tokens must be")
    _refused(binary, runtime_root, bundle, paths["valid"], ["--set", f"max_new_tokens={capacity}"],
             f"exceeds the bundle's capacity of {capacity}")


def test_e2e(tmp_path: Path, bundles: dict, case_name: str) -> None:
    manifest_path, manifest, case = CASES[case_name]
    assert manifest_path.is_file()
    binary, runtime_root = _runtime()
    consumers = _sdk_consumers()
    model_dir = _model_dir(manifest)
    record_evidence("inputs", {"manifest": manifest, "case": case})
    wav, samples = _case_audio(model_dir, case, tmp_path / "audio")
    bundle = _bundle(bundles, model_dir, manifest)
    versions = record_evidence("environment", _versions())
    print(f"{FAMILY} environment: {json.dumps(versions)}")
    with evidence_stage("native"):
        native = {"cli": _native_cli(binary, runtime_root, bundle, wav, case)}
        for language, consumer in consumers.items():
            native[f"sdk_{language}"] = _native_sdk(consumer, runtime_root, bundle, samples, case, tmp_path)
    refused = bundles.setdefault("refusals checked", set())
    if manifest["name"] not in refused:
        with evidence_stage("refusals"):
            _assert_refusals(binary, runtime_root, bundle, manifest, samples, tmp_path / "refusals")
        refused.add(manifest["name"])
    with evidence_stage("reference"):
        reference = _official_reference(model_dir, samples, case)
    record_evidence("results", {"native": native, "reference": reference})
    for label, result in native.items():
        _assert_parity(result, reference, label)
    print(f"{FAMILY} parity: {len(reference['token_ids'])} ids over {reference['audio_tokens']} audio tokens, "
          f"transcript {reference['text']!r}")


def test_manifest_cases_read_checkpoint_audio_only():
    for _, manifest, case in CASES.values():
        assert manifest["hf_revision"] and case["checkpoint_audio"].endswith(".wav")
        assert int(case["max_new_tokens"]) > 0 and int(manifest.get("max_sequence_length") or 4096) > 0
        assert float(case.get("end_seconds", 1.0)) > 0 and int(case.get("repeat", 1)) >= 1
