# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native CLI and official-reference E2E for vibevoice.

Each case builds the bundle from the original checkpoint, synthesizes speech
with the native ``generate-audio`` command and runs Transformers'
``VibeVoiceForConditionalGeneration.generate`` on the same tensors with the
same noise: the family's documented noise stream (seeded by ``--seed``) is
injected into the reference's diffusion and voice-prompt sampling. The oracle
compares the two syntheses frame by frame and transcribes the native audio
with the declared Whisper checkpoint; the family's C and C++ SDK consumers
then synthesize from the same bundle.
"""

from __future__ import annotations

import json
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from tools.e2e_evidence import evidence_stage, record_evidence

FAMILY = "vibevoice"
TASKS = frozenset({"text_to_speech"})
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"
THRESHOLD_ROOT = TEST_ROOT / "thresholds"
ASR_REPOSITORY = "openai/whisper-large-v3-turbo"
TOKENIZER_REPOSITORY = "Qwen/Qwen2.5-1.5B"
DEFAULT_THRESHOLDS = {
    # float32: the early frames follow the reference to float32 rounding. The
    # autoregressive sampler amplifies the remaining 1e-7 differences, so the
    # syntheses then part gradually (measured: at least 84 dB per frame over the
    # first 12 frames, falling to 57 dB by the 16th frame, a loud one). Every one
    # of the first frames must match.
    "fp32_matched_frames": 12,
    "fp32_min_frame_snr_db": 60.0,
    # bfloat16: rounding moves the sampler onto another trajectory after a few
    # frames, in the official pipeline too (its bfloat16 and float32 runs part
    # after 4-13 frames, and their lengths differ by up to 12 %). Measured: the
    # first 4 frames agree with the official bfloat16 synthesis at 26-33 dB. The
    # native output also may not stray from the float32 reference much further
    # than the official bfloat16 synthesis does.
    "bf16_prefix_frames": 4,
    "bf16_min_prefix_snr_db": 20.0,
    "bf16_max_drift_excess_db": 6.0,
    "bf16_max_frame_count_ratio": 1.25,
    "max_asr_edit_distance": 0.2,
    "min_rms": 0.005,
}


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
        model_match = (
            not model_filters
            or FAMILY in model_filters
            or name in model_filters
            or manifest["name"] in model_filters
        )
        if model_match and (not testcase_filters or name in testcase_filters):
            selected.append(name)
    return sorted(selected), True


def pytest_generate_tests(metafunc) -> None:
    if "case_name" in metafunc.fixturenames:
        names, enabled = _selected_cases(metafunc.config)
        parameters = (
            names
            if enabled
            else [
                pytest.param(
                    name,
                    marks=pytest.mark.skip(
                        reason="direct E2E requires one of the three explicit E2E selectors"
                    ),
                )
                for name in names
            ]
        )
        metafunc.parametrize("case_name", parameters, ids=names)


def _required_path(value: str | None, label: str) -> Path:
    assert value, f"selected {FAMILY} E2E requires {label}"
    path = Path(value)
    assert path.exists(), f"selected {FAMILY} E2E {label} does not exist: {path}"
    return path


# The files each input needs; an offline snapshot is complete only for what it lists.
_ALLOW = {
    "checkpoint": ("*.json", "*.safetensors"),
    TOKENIZER_REPOSITORY: ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"),
    ASR_REPOSITORY: ("*.json", "*.safetensors", "*.txt"),
}


def _snapshot(repo_id: str, revision: str | None, allow: tuple[str, ...]) -> Path:
    from huggingface_hub import snapshot_download

    try:
        return Path(
            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                local_files_only=True,
                allow_patterns=list(allow),
            )
        )
    except Exception as error:  # noqa: BLE001 - surfaced as the missing input
        raise AssertionError(
            f"selected {FAMILY} E2E requires the cached snapshot {repo_id}"
        ) from error


def _model_dir(manifest: dict) -> Path:
    explicit = os.environ.get(f"TRTMC_{FAMILY.upper()}_MODEL_DIR")
    if explicit:
        return _required_path(explicit, f"TRTMC_{FAMILY.upper()}_MODEL_DIR")
    return _snapshot(manifest["hf_id"], manifest.get("hf_revision"), _ALLOW["checkpoint"])


def _dependency(manifest: dict, repo_id: str) -> Path:
    matches = [item for item in manifest["hf_dependencies"] if item.get("repo_id") == repo_id]
    assert len(matches) == 1, f"selected {FAMILY} E2E requires one {repo_id} dependency"
    return _snapshot(repo_id, matches[0].get("revision"), _ALLOW[repo_id])


def _runtime() -> tuple[Path, Path]:
    binary = _required_path(os.environ.get("TRTMC_BINARY"), "TRTMC_BINARY")
    runtime_root = _required_path(os.environ.get("TRTMC_RUNTIME_ROOT"), "TRTMC_RUNTIME_ROOT")
    assert (runtime_root / "libtrtmc_backend_trt.so").is_file()
    assert (runtime_root / f"libtrtmc_model_{FAMILY}.so").is_file()
    import torch

    assert torch.cuda.is_available(), f"selected {FAMILY} E2E requires CUDA"
    return binary, runtime_root


def _build(model_dir: Path, bundle: Path, manifest: dict) -> None:
    from tensorrt_model_connect import BuildRequest, build

    build(
        BuildRequest(
            model_dir=model_dir,
            output_path=bundle,
            family=FAMILY,
            task=manifest["task"],
            precision=manifest["precision"],
            max_sequence_length=manifest.get("max_sequence_length"),
            tensor_parallel_size=int(manifest["tensor_parallel_size"]),
        )
    )


def _thresholds(case_name: str) -> dict:
    values = dict(DEFAULT_THRESHOLDS)
    path = THRESHOLD_ROOT / f"{case_name}.json"
    if path.is_file():
        values.update(json.loads(path.read_text(encoding="utf-8"))["threshold_overrides"])
    return values


def _generate(
    binary: Path,
    runtime_root: Path,
    bundle: Path,
    text: str,
    output: Path,
    *,
    seed: int,
    max_new_tokens: int,
    voice_prompts: list[str],
) -> dict:
    command = [
        str(binary),
        "generate-audio",
        str(bundle),
        "--runtime-root",
        str(runtime_root),
        "--prompt",
        text,
        "--output",
        str(output),
        "--seed",
        str(seed),
        "--max-new-tokens",
        str(max_new_tokens),
    ]
    if voice_prompts:
        command += ["--set", "voice_prompts=" + json.dumps(voice_prompts)]
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = ":".join(
        v for v in (str(runtime_root), env.get("LD_LIBRARY_PATH", "")) if v
    )
    completed = subprocess.run(
        command, check=True, capture_output=True, text=True, env=env, timeout=3600
    )
    record_evidence("commands", {"argv": command})
    record_evidence("native", {"stdout": completed.stdout, "stderr": completed.stderr[-4000:]})
    payload = [
        json.loads(line[line.find("{") :]) for line in completed.stdout.splitlines() if "{" in line
    ][-1]
    assert output.is_file()
    return payload


def _read_audio(path: Path) -> np.ndarray:
    import soundfile as sf

    samples, rate = sf.read(path, dtype="float32", always_2d=False)
    assert rate == 24000 and np.asarray(samples).ndim == 1
    return np.asarray(samples, dtype=np.float32)


def _turns(text: str) -> list[tuple[int, str]]:
    from families.vibevoice.tests.host_fixtures import _turns as parse

    return [(int(speaker), body) for speaker, body in parse(text)]


@contextmanager
def _injected_noise(seed: int, voice_frames: list[int], latent_size: int):
    """Feed the reference the family's noise stream in the family's draw order."""
    import torch

    from families.vibevoice.tests.reference import NoiseStream

    stream = NoiseStream(seed)
    voice_scales, voice_noise = [], []
    for frames in voice_frames:
        voice_scales.append(stream.normal())
        voice_noise.append(stream.normals(frames * latent_size).reshape(frames, latent_size))
    original_randn, original_randn_like = torch.randn, torch.randn_like
    calls = {"frames": 0}

    def randn(*size, **kwargs):
        shape = (
            tuple(size[0]) if len(size) == 1 and isinstance(size[0], (tuple, list)) else tuple(size)
        )
        if shape == (len(voice_frames),) and voice_frames:
            return torch.tensor(voice_scales, dtype=torch.float32).to(
                device=kwargs.get("device"), dtype=kwargs.get("dtype") or torch.float32
            )
        if len(shape) == 2 and shape[0] == 2 and shape[1] == latent_size:
            calls["frames"] += 1
            values = np.zeros(shape, dtype=np.float32)
            values[0] = stream.normals(latent_size)
            return torch.from_numpy(values)
        raise AssertionError(f"unexpected reference randn{shape}")

    def randn_like(tensor, **kwargs):
        batch, frames, width = tensor.shape
        values = np.zeros((batch, frames, width), dtype=np.float32)
        for index, noise in enumerate(voice_noise):
            values[index, : noise.shape[0]] = noise
        return torch.from_numpy(values).to(
            device=tensor.device, dtype=kwargs.get("dtype") or tensor.dtype
        )

    torch.randn, torch.randn_like = randn, randn_like
    try:
        yield calls
    finally:
        torch.randn, torch.randn_like = original_randn, original_randn_like


def _reference(
    model_dir: Path,
    tokenizer_dir: Path,
    precision: str,
    text: str,
    *,
    seed: int,
    max_new_tokens: int,
    voice_prompts: list[Path],
) -> np.ndarray:
    import torch

    from families.vibevoice.tests import reference

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[precision]
    model = reference.load_reference(model_dir, dtype, "cuda")
    processor = reference.reference_processor(tokenizer_dir, 3200)
    turns = _turns(text)
    voices = {index: _read_audio(path) for index, path in enumerate(voice_prompts)}
    conversation = []
    seen = set()
    for speaker, body in turns:
        content = [{"type": "text", "text": body}]
        if speaker in voices and speaker not in seen:
            content.append({"type": "audio", "audio": voices[speaker]})
            seen.add(speaker)
        conversation.append({"role": str(speaker), "content": content})
    inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        return_dict=True,
        add_generation_prompt=True,
        return_tensors="pt",
        processor_kwargs={"sampling_rate": 24000},
    )
    inputs = inputs.to("cuda")
    if "input_values" in inputs:
        inputs["input_values"] = inputs["input_values"].to(dtype)
    frames = [int(np.ceil(len(voices[index]) / 3200)) for index in sorted(voices)]
    try:
        with torch.no_grad(), _injected_noise(seed, frames, model.config.audio_config.hidden_size):
            audio = model.generate(**inputs, max_new_tokens=max_new_tokens)
    finally:
        del model
        torch.cuda.empty_cache()
    assert audio[0] is not None, "reference generated no speech"
    return audio[0].float().cpu().numpy().reshape(-1)


def _normalized(text: str) -> str:
    kept = "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in text)
    return " ".join(kept.split())


def _edit_distance(actual: str, expected: str) -> float:
    actual, expected = _normalized(actual).split(), _normalized(expected).split()
    previous = list(range(len(expected) + 1))
    for row, word in enumerate(actual, start=1):
        current = [row]
        for column, target in enumerate(expected, start=1):
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + (word != target),
                )
            )
        previous = current
    return previous[-1] / max(len(expected), 1)


def _transcribe(audio: np.ndarray, manifest: dict) -> str:
    import torch
    from scipy.signal import resample_poly
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    model_dir = _dependency(manifest, ASR_REPOSITORY)
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = (
        AutoModelForSpeechSeq2Seq.from_pretrained(
            model_dir, local_files_only=True, torch_dtype=torch.float16
        )
        .to("cuda")
        .eval()
    )
    try:
        resampled = resample_poly(audio, 2, 3).astype(np.float32)  # 24 kHz -> 16 kHz for Whisper
        features = processor(
            resampled, sampling_rate=16000, return_tensors="pt", return_attention_mask=True
        )
        features = {
            key: (
                value.to("cuda", torch.float16) if value.is_floating_point() else value.to("cuda")
            )
            for key, value in features.items()
        }
        with torch.inference_mode():
            generated = model.generate(
                **features, language="en", task="transcribe", max_new_tokens=440
            )
        return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
    finally:
        del model
        torch.cuda.empty_cache()


def _snr_db(actual: np.ndarray, expected: np.ndarray) -> float:
    noise = float(np.sum((actual.astype(np.float64) - expected) ** 2))
    signal = float(np.sum(expected.astype(np.float64) ** 2))
    return float("inf") if noise == 0.0 else 10.0 * np.log10(max(signal, 1e-30) / noise)


def _frame_snr_db(actual: np.ndarray, expected: np.ndarray, frames: int) -> list[float]:
    return [
        _snr_db(
            actual[index * 3200 : (index + 1) * 3200], expected[index * 3200 : (index + 1) * 3200]
        )
        for index in range(min(frames, actual.size // 3200, expected.size // 3200))
    ]


def _assert_contract(
    native: np.ndarray,
    references: dict[str, np.ndarray],
    text: str,
    manifest: dict,
    thresholds: dict,
    *,
    truncated: bool = False,
) -> dict:
    assert native.size > 0 and native.size % 3200 == 0 and np.isfinite(native).all()
    assert float(np.sqrt(np.mean(native**2))) >= thresholds["min_rms"]
    expected = references[manifest["precision"]]
    report = {"native_frames": native.size // 3200, "reference_frames": expected.size // 3200}
    if manifest["precision"] == "fp32":
        assert native.size == expected.size, f"frame counts differ: {report}"
        frames = min(int(thresholds["fp32_matched_frames"]), report["native_frames"])
        report["frame_snr_db"] = _frame_snr_db(native, expected, frames)
        report["frame_rms"] = [
            float(np.sqrt(np.mean(expected[index * 3200 : (index + 1) * 3200] ** 2)))
            for index in range(frames)
        ]
        report["prefix_snr_db"] = _snr_db(native[: frames * 3200], expected[: frames * 3200])
        assert len(report["frame_snr_db"]) == frames, report
        assert min(report["frame_snr_db"]) >= thresholds["fp32_min_frame_snr_db"], report
    else:
        frames = int(thresholds["bf16_prefix_frames"])
        prefix = min(native.size, expected.size, references["fp32"].size, frames * 3200)
        report["frame_count_ratio"] = max(native.size, expected.size) / min(
            native.size, expected.size
        )
        report["prefix_snr_db"] = _snr_db(native[:prefix], expected[:prefix])
        report["native_vs_fp32_prefix_snr_db"] = _snr_db(
            native[:prefix], references["fp32"][:prefix]
        )
        report["official_bf16_vs_fp32_prefix_snr_db"] = _snr_db(
            expected[:prefix], references["fp32"][:prefix]
        )
        assert report["frame_count_ratio"] <= thresholds["bf16_max_frame_count_ratio"], report
        assert report["prefix_snr_db"] >= thresholds["bf16_min_prefix_snr_db"], report
        excess = (
            report["official_bf16_vs_fp32_prefix_snr_db"] - report["native_vs_fp32_prefix_snr_db"]
        )
        assert excess <= thresholds["bf16_max_drift_excess_db"], report
    spoken = " ".join(body for _, body in _turns(text))
    if truncated:
        # max_new_tokens ends the speech before the script does (a long prompt that
        # needs chunked prefill): the native speech must say what the reference's says.
        assert report["native_frames"] == report["reference_frames"], report
        report["reference_transcript"] = _transcribe(expected, manifest)
        spoken = report["reference_transcript"]
    report["transcript"] = _transcribe(native, manifest)
    report["asr_edit_distance"] = _edit_distance(report["transcript"], spoken)
    assert report["asr_edit_distance"] <= thresholds["max_asr_edit_distance"], report
    return report


def _assert_sdk_consumers(runtime_root: Path, bundle: Path) -> None:
    native_build = _required_path(
        os.environ.get("TRTMC_NATIVE_BUILD_DIR"), "TRTMC_NATIVE_BUILD_DIR"
    )
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = ":".join(
        v for v in (str(runtime_root), env.get("LD_LIBRARY_PATH", "")) if v
    )
    for language in ("c", "cpp"):
        consumer = native_build / f"test_{FAMILY}_sdk_{language}"
        assert consumer.is_file(), f"build the family-owned SDK consumer: {consumer.name}"
        completed = subprocess.run(
            [str(consumer), str(bundle), str(runtime_root)],
            check=True,
            capture_output=True,
            text=True,
            env=env,
            timeout=1800,
        )
        record_evidence(f"sdk_{language}", {"stdout": completed.stdout})
        assert "all checks passed" in completed.stdout, completed.stdout


def test_noise_injection_follows_the_family_draw_order() -> None:
    torch = pytest.importorskip("torch")
    from families.vibevoice.tests.reference import NoiseStream

    expected = NoiseStream(5)
    scale = expected.normal()
    voice = expected.normals(3 * 4).reshape(3, 4)
    frame = expected.normals(4)
    with _injected_noise(5, [3], 4):
        assert float(torch.randn(1)[0]) == scale
        assert np.array_equal(torch.randn_like(torch.zeros(1, 5, 4))[0, :3].numpy(), voice)
        assert np.array_equal(torch.randn(2, 4)[0].numpy(), frame)


def test_word_edit_distance_is_case_and_punctuation_blind() -> None:
    assert _edit_distance("Hello, WORLD!", "hello world") == 0.0
    assert _edit_distance("hello there world", "hello world") == 0.5


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    _, manifest, case = CASES[case_name]
    record_evidence("inputs", {"manifest": manifest, "case": case})
    model_dir = _model_dir(manifest)
    tokenizer_dir = _dependency(manifest, TOKENIZER_REPOSITORY)
    record_evidence(
        "checkpoint",
        {
            "model_dir": str(model_dir),
            "hf_id": manifest.get("hf_id"),
            "hf_revision": manifest.get("hf_revision"),
        },
    )
    binary, runtime_root = _runtime()
    bundle = tmp_path.parent / manifest["bundle"]
    if not bundle.is_file():
        with evidence_stage("build"):
            _build(model_dir, bundle, manifest)
    voice_prompts: list[Path] = []
    if "voice_text" in case:
        # The voice prompt is speech the family synthesized itself, so the case
        # needs no third-party recording; both sides then clone that voice.
        voice = tmp_path / "voice.wav"
        with evidence_stage("native"):
            _generate(
                binary,
                runtime_root,
                bundle,
                case["voice_text"],
                voice,
                seed=int(case["voice_seed"]),
                max_new_tokens=int(case["max_new_tokens"]),
                voice_prompts=[],
            )
        voice_prompts.append(voice)
    output = tmp_path / "native.wav"
    with evidence_stage("native"):
        payload = _generate(
            binary,
            runtime_root,
            bundle,
            case["text"],
            output,
            seed=int(case["seed"]),
            max_new_tokens=int(case["max_new_tokens"]),
            voice_prompts=[str(path) for path in voice_prompts],
        )
    record_evidence("native", payload)
    native = _read_audio(output)
    with evidence_stage("reference"):
        # A bfloat16 case also runs the float32 reference: it bounds how far the
        # native output may drift compared with the official bfloat16 synthesis.
        precisions = ["fp32"] if manifest["precision"] == "fp32" else ["bf16", "fp32"]
        references = {
            precision: _reference(
                model_dir,
                tokenizer_dir,
                precision,
                case["text"],
                seed=int(case["seed"]),
                max_new_tokens=int(case["max_new_tokens"]),
                voice_prompts=voice_prompts,
            )
            for precision in precisions
        }
    with evidence_stage("compare"):
        report = _assert_contract(
            native,
            references,
            case["text"],
            manifest,
            record_evidence("thresholds", _thresholds(case_name)),
            truncated=bool(case.get("truncated", False)),
        )
    record_evidence("compare", report)
    with evidence_stage("sdk"):
        _assert_sdk_consumers(runtime_root, bundle)
