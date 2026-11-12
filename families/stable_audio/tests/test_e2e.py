# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime and official-reference E2E for stable_audio.

The official ``StableAudioPipeline`` draws its SDE noise from a torchsde Brownian tree,
which no native runtime can reproduce. The native runtime draws its noise from the
family's documented counter-based generator (``runtime/noise.h``). This test redraws
the same values with the Python mirror (``family_noise.py``), hands the initial latents
to the official pipeline and replaces its Brownian sampler with a replay of the per-step
noise, so both sides walk the same trajectory and their audio is compared directly:

* FP32 bundles against the official FP32 pipeline (TF32 off on both sides), within a
  relative L2 error of 1e-2;
* FP16 bundles against the official FP32 pipeline, no further from it than the official
  FP16 pipeline driven by the same noise (calibrated in the same run), and never beyond 0.5.

A whole trajectory amplifies precision noise about a hundredfold (guidance multiplies the
branch difference every step), so every case also checks each engine on the recorded
inputs of the official FP32 run ("teacher forced"):

* conditioning: the bundle's conditioning engine on the official token ids, attention
  masks and seconds against the official encoder / global states of every encoded branch,
  within 1e-4 (the engine is FP32 in both precisions);
* denoiser: every recorded denoiser call, at 1e-4 per call for FP32, and for FP16 at 1.25
  times the error of the official FP16 denoiser on the same inputs;
* decoder: the official final latents, at 1e-4 for FP32, and for FP16 at 1.25 times the
  error of the official FP16 decoder on the same latents.

Every case also requires the prompt token ids to equal the official tokenizer's.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

from families.stable_audio.tests.family_noise import initial_latents, step_noise
from tensorrt_model_connect import BuildRequest, build
from tools.e2e_evidence import evidence_stage, record_evidence

FAMILY = "stable_audio"
TASKS = frozenset({"text_to_audio"})
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"
THRESHOLD_ROOT = TEST_ROOT / "thresholds"
T5_REPOSITORY = "google-t5/t5-base"
_INPUT_IDS = re.compile(r"\[trtmc\] stable_audio: input_ids=\[([0-9,]*)\]")
# Defaults when a case has no threshold sidecar. Calibrated on the full-size random-weight
# checkpoint (hammer case, 100 guided steps): the official FP32 pipeline itself moves by
# 8.3e-4 when only its attention kernel changes (math vs. default SDPA), the native FP32
# bundle by 6.1e-4 / 8.7e-4, while the closest negative control (99 steps) is at 0.45; per
# call the FP32 bundle is within 8.6e-6, a one-step timestep shift moves it by up to 8.8e-4.
# The FP16 bounds hold the bundle to the official FP16 pipeline on the same noise / inputs.
_FP32_MAX_RELATIVE_L2 = 1e-2
_FP16_DRIFT_FACTOR = 1.0
_FP16_DRIFT_FLOOR = 1e-2
_FP32_TEACHER_FORCED_MAX = 1e-4
_FP16_TEACHER_FORCED_FACTOR = 1.25
_FP16_TEACHER_FORCED_FLOOR = 1e-3
_FP16_DRIFT_CAP = 0.5
_CONDITIONER_MAX = 1e-4
_FP32_DECODER_MAX = 1e-4
_FP16_DECODER_FACTOR = 1.25
_FP16_DECODER_FLOOR = 1e-3
# The default of both the family's ``num_inference_steps`` field and the diffusers pipeline.
_DEFAULT_STEPS = 100


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
        if model_match and (not testcase_filters or name in testcase_filters):
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


def _t5_revision(manifest: dict) -> str:
    matches = [item for item in manifest["hf_dependencies"] if item["repo_id"] == T5_REPOSITORY]
    assert len(matches) == 1, f"{FAMILY} random checkpoints need one pinned {T5_REPOSITORY}"
    return str(matches[0]["revision"])


def _model_dir(manifest: dict, scratch: Path) -> Path:
    if "hf_id" in manifest:
        from huggingface_hub import snapshot_download

        try:
            return Path(snapshot_download(
                repo_id=manifest["hf_id"], revision=manifest["hf_revision"], local_files_only=True))
        except Exception as error:
            raise AssertionError(
                f"selected {FAMILY} E2E requires the exact cached checkpoint {manifest['hf_id']}") from error
    # A random-weight checkpoint of the published architecture, generated deterministically.
    spec = manifest["random_checkpoint"]
    root = Path(os.environ.get("TRTMC_STABLE_AUDIO_RANDOM_ROOT") or scratch)
    from families.stable_audio.tests.random_checkpoint import checkpoint_name, is_current, write_random_checkpoint

    size, seed, revision = str(spec["size"]), int(spec["seed"]), _t5_revision(manifest)
    target = root / checkpoint_name(size, seed)
    if not is_current(target, size=size, seed=seed, t5_revision=revision):
        write_random_checkpoint(target, size=size, seed=seed, t5_revision=revision)
    return target


def _runtime() -> tuple[Path, Path]:
    binary = _required_path(os.environ.get("TRTMC_BINARY"), "TRTMC_BINARY")
    runtime_root = _required_path(os.environ.get("TRTMC_RUNTIME_ROOT"), "TRTMC_RUNTIME_ROOT")
    assert (runtime_root / "libtrtmc_backend_trt.so").is_file()
    assert (runtime_root / f"libtrtmc_model_{FAMILY}.so").is_file()
    import torch

    assert torch.cuda.is_available(), f"selected {FAMILY} E2E requires CUDA"
    return binary, runtime_root


def _build(model_dir: Path, bundle: Path, manifest: dict) -> None:
    build(BuildRequest(
        model_dir=model_dir,
        output_path=bundle,
        family=FAMILY,
        task=manifest["task"],
        precision=manifest["precision"],
        tensor_parallel_size=int(manifest["tensor_parallel_size"]),
    ))


def read_float_wav(path: Path) -> tuple[np.ndarray, int]:
    """Read the CLI's IEEE-float WAV (format 3, 32 bit) as [frames, channels]."""
    data = path.read_bytes()
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE"
    offset, fmt, samples = 12, None, None
    while offset + 8 <= len(data):
        chunk, size = data[offset:offset + 4], int.from_bytes(data[offset + 4:offset + 8], "little")
        body = data[offset + 8:offset + 8 + size]
        if chunk == b"fmt ":
            fmt = (int.from_bytes(body[0:2], "little"), int.from_bytes(body[2:4], "little"),
                   int.from_bytes(body[4:8], "little"), int.from_bytes(body[14:16], "little"))
        elif chunk == b"data":
            samples = np.frombuffer(body, dtype="<f4")
        offset += 8 + size + (size & 1)
    assert fmt is not None and samples is not None, f"{path} is not a complete WAV file"
    audio_format, channels, sample_rate, bits = fmt
    assert (audio_format, bits) == (3, 32), f"{path}: expected 32-bit float PCM, got {fmt}"
    return samples.reshape(-1, channels).astype(np.float32), sample_rate


def _config(case: dict) -> dict:
    """The case's Task config fields (the benchmark passes the same object to the Task)."""
    return dict(case.get("config", {}))


def _steps(case: dict) -> int:
    return int(_config(case).get("num_inference_steps", _DEFAULT_STEPS))


def _case_arguments(case: dict) -> list[str]:
    """CLI arguments of one case: prompt, seed and the case's config fields through ``--set``."""
    arguments = ["--prompt", str(case["prompt"]), "--seed", str(int(case["seed"]))]
    for key, value in _config(case).items():
        arguments.extend(("--set", f"{key}={value}"))
    return arguments


def _native(binary: Path, runtime_root: Path, bundle: Path, case: dict, tmp_path: Path) -> dict:
    output = tmp_path / "native.wav"
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = ":".join(
        value for value in (str(runtime_root), environment.get("LD_LIBRARY_PATH", "")) if value)
    environment["TRTMC_STABLE_AUDIO_DEBUG_TOKENS"] = "1"
    command = [str(binary), "generate-audio", str(bundle), "--runtime-root", str(runtime_root),
               *_case_arguments(case), "--output", str(output)]
    completed = subprocess.run(command, check=True, capture_output=True, text=True, env=environment,
                               timeout=int(case.get("runtime_timeout_s", 1800)))
    record_evidence("commands", {"argv": command})
    record_evidence("native_log", {"stdout": completed.stdout, "stderr": completed.stderr})
    payloads = [json.loads(line[line.find("{"):]) for line in completed.stdout.splitlines() if "{" in line]
    assert payloads, f"native generate-audio returned no JSON: {completed.stdout[-1000:]}"
    audio, sample_rate = read_float_wav(output)
    token_ids = [[int(value) for value in match.split(",") if value]
                 for match in _INPUT_IDS.findall(completed.stderr)]
    return {"audio": audio, "sample_rate": sample_rate, "token_ids": token_ids, "payload": payloads[-1]}


class _ReplayNoise:
    """Stands in for diffusers' BrownianTreeNoiseSampler: returns the native step noise."""

    def __init__(self, like, seed: int, channels: int, frames: int) -> None:
        self.like, self.seed, self.channels, self.frames = like, seed, channels, frames
        self.calls = 0

    def __call__(self, sigma, sigma_next):
        import torch

        noise = step_noise(self.seed, self.calls, self.channels, self.frames)
        self.calls += 1
        return torch.from_numpy(noise).to(device=self.like.device, dtype=self.like.dtype)


def _official_reference(model_dir: Path, case: dict, precision: str, recording: dict | None = None) -> dict:
    """Run the official pipeline with the native noise.

    The FP32 run records the inputs and outputs of its text encoder, duration projection,
    every denoiser call and the decoder; given that ``recording``, the FP16 run also
    evaluates its own denoiser and decoder on the recorded inputs.
    """
    import torch
    from diffusers import StableAudioPipeline
    from diffusers.schedulers import scheduling_cosine_dpmsolver_multistep as cosine_scheduler

    from families.stable_audio.tests.reference_t5 import restore_t5_layer_norm

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dtype = {"fp32": torch.float32, "fp16": torch.float16}[precision]
    pipeline = StableAudioPipeline.from_pretrained(model_dir, torch_dtype=dtype, local_files_only=True).to("cuda")
    record_evidence(f"reference_{precision}_apex_norms_replaced", restore_t5_layer_norm(pipeline.text_encoder))
    channels = int(pipeline.transformer.config.in_channels)
    frames = int(pipeline.transformer.config.sample_size)
    seed = int(case["seed"])
    latents = torch.from_numpy(initial_latents(seed, channels, frames)).to("cuda", dtype)
    samplers = []

    def replay(x, sigma_min, sigma_max, seed=None, **kwargs):
        del sigma_min, sigma_max, seed, kwargs
        samplers.append(_ReplayNoise(x, int(case["seed"]), channels, frames))
        return samplers[-1]

    config = _config(case)
    arguments = {key: config[key] for key in ("negative_prompt", "audio_start_in_s", "audio_end_in_s",
                                               "guidance_scale") if key in config}
    recorded = []
    texts, durations, decoded = [], [], []

    def record_text(module, args, kwargs, output):
        del module, output
        input_ids = args[0] if args else kwargs["input_ids"]
        texts.append({"input_ids": input_ids.cpu().numpy().astype(np.int32),
                      "attention_mask": kwargs["attention_mask"].float().cpu().numpy()})

    def record_duration(module, args, kwargs, output):
        del module, args, output
        if kwargs.get("start_seconds") is not None:
            durations.append({"seconds_start": kwargs["start_seconds"].float().cpu().numpy(),
                              "seconds_end": kwargs["end_seconds"].float().cpu().numpy()})

    def record_decoder(module, args, output):
        del module
        decoded.append({"latents": args[0].float().cpu().numpy(), "audio": output.float().cpu().numpy()})

    def record(module, args, kwargs, output):
        del module
        recorded.append({
            "hidden_states": args[0].float().cpu().numpy(),
            "timestep": args[1].float().cpu().numpy(),
            "encoder_hidden_states": kwargs["encoder_hidden_states"].float().cpu().numpy(),
            "global_hidden_states": kwargs["global_hidden_states"].float().cpu().numpy(),
            "rotary_embedding": tuple(value.cpu() for value in kwargs["rotary_embedding"]),
            "sample": output[0].float().cpu().numpy(),
        })

    hooks = []
    if precision == "fp32":
        hooks = [pipeline.transformer.register_forward_hook(record, with_kwargs=True),
                 pipeline.text_encoder.register_forward_hook(record_text, with_kwargs=True),
                 pipeline.projection_model.register_forward_hook(record_duration, with_kwargs=True),
                 pipeline.vae.decoder.register_forward_hook(record_decoder)]
    original = cosine_scheduler.BrownianTreeNoiseSampler
    cosine_scheduler.BrownianTreeNoiseSampler = replay
    try:
        with torch.inference_mode():
            audio = pipeline(str(case["prompt"]), num_inference_steps=_steps(case),
                             latents=latents, output_type="np", **arguments).audios
    finally:
        cosine_scheduler.BrownianTreeNoiseSampler = original
        for hook in hooks:
            hook.remove()
    assert len(samplers) == 1 and samplers[0].calls == _steps(case)
    if precision == "fp32":
        assert len(durations) == 1 and len(decoded) == 1 and 1 <= len(texts) <= 2, (
            len(texts), len(durations), len(decoded))
    teacher_forced = []
    for call in (recording or {}).get("denoiser_calls", []):
        with torch.inference_mode():
            output = pipeline.transformer(
                torch.from_numpy(call["hidden_states"]).to("cuda", dtype),
                torch.from_numpy(call["timestep"]).to("cuda"),
                encoder_hidden_states=torch.from_numpy(call["encoder_hidden_states"]).to("cuda", dtype),
                global_hidden_states=torch.from_numpy(call["global_hidden_states"]).to("cuda", dtype),
                rotary_embedding=tuple(value.to("cuda") for value in call["rotary_embedding"]),
                return_dict=False)[0]
        teacher_forced.append(output.float().cpu().numpy())
    decoder_forced = None
    if recording is not None:
        with torch.inference_mode():
            decoder_forced = pipeline.vae.decoder(
                torch.from_numpy(recording["decoder"]["latents"]).to("cuda", dtype)).float().cpu().numpy()
    token_ids = []
    prompts = [str(case["prompt"])]
    if "negative_prompt" in config and float(config.get("guidance_scale", 7.0)) > 1.0:
        prompts.append(str(config["negative_prompt"]))
    for text in prompts:
        encoded = pipeline.tokenizer(text, padding="max_length", max_length=pipeline.tokenizer.model_max_length,
                                     truncation=True)
        token_ids.append([int(i) for i, keep in zip(encoded["input_ids"], encoded["attention_mask"]) if keep])
    sample_rate = int(pipeline.vae.config.sampling_rate)
    del pipeline
    torch.cuda.empty_cache()
    return {"audio": np.asarray(audio[0], dtype=np.float32).T.copy(), "sample_rate": sample_rate,
            "token_ids": token_ids, "denoiser_calls": recorded, "teacher_forced": teacher_forced,
            "texts": texts, "duration": durations[0] if durations else None,
            "decoder": decoded[0] if decoded else None, "decoder_forced": decoder_forced}


def _bundle_stages(bundle: Path, reference: dict) -> dict:
    """The bundle's three engines on the recorded inputs of the official FP32 run."""
    from families.stable_audio.tests.trt_runner import EngineRunner, bundle_section

    conditioner = EngineRunner(bundle_section(bundle, "conditioner.plan"))
    conditioned = [conditioner({**text, **reference["duration"]}, ["encoder_hidden_states", "global_hidden_states"])
                   for text in reference["texts"]]
    del conditioner
    denoiser = EngineRunner(bundle_section(bundle, "transformer.plan"))
    names = ("hidden_states", "timestep", "encoder_hidden_states", "global_hidden_states")
    denoised = [denoiser({name: call[name] for name in names}, ["sample"])["sample"]
                for call in reference["denoiser_calls"]]
    del denoiser
    decoder = EngineRunner(bundle_section(bundle, "vae_decoder.plan"))
    decoded = decoder({"latents": reference["decoder"]["latents"]}, ["audio"])["audio"]
    return {"conditioned": conditioned, "teacher_forced": denoised, "decoded": decoded}


def _conditioning_errors(conditioned: list, reference: dict) -> list:
    """Per encoded branch: the engine's states against the official first denoiser input.

    The official pipeline encodes the prompt, then the negative prompt, and stacks the
    guidance batch as [negative or zeros, prompt]; without guidance the batch is [prompt].
    """
    first = reference["denoiser_calls"][0]
    errors = []
    for index, output in enumerate(conditioned):
        row = -1 if index == 0 else 0
        errors.append(max(relative_l2(output["encoder_hidden_states"][0], first["encoder_hidden_states"][row]),
                          relative_l2(output["global_hidden_states"][0], first["global_hidden_states"][row])))
    return errors


def _thresholds(case_name: str) -> dict:
    path = THRESHOLD_ROOT / f"{case_name}.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))["threshold_overrides"]


def relative_l2(actual: np.ndarray, expected: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    return float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-30))


def _assert_contract(native: dict, references: dict, case: dict, precision: str, thresholds: dict) -> dict:
    reference = references["fp32"]
    assert native["token_ids"] == reference["token_ids"], "native prompt tokens differ from the official tokenizer"
    assert native["sample_rate"] == reference["sample_rate"]
    assert native["audio"].shape == reference["audio"].shape, (native["audio"].shape, reference["audio"].shape)
    assert np.isfinite(native["audio"]).all()
    distance = relative_l2(native["audio"], reference["audio"])
    summary = {"relative_l2_vs_official_fp32": distance,
               "snr_db_vs_official_fp32": -20.0 * np.log10(max(distance, 1e-30)),
               "frames": int(native["audio"].shape[0]),
               "reference_rms": float(np.sqrt(np.mean(np.square(reference["audio"], dtype=np.float64))))}
    # Teacher forced: every denoiser call of the official FP32 trajectory, same inputs.
    calls = reference["denoiser_calls"]
    assert calls and len(native["teacher_forced"]) == len(calls) == _steps(case)
    steps = [relative_l2(output, call["sample"]) for output, call in zip(native["teacher_forced"], calls)]
    summary["teacher_forced_max_relative_l2"] = max(steps)
    summary["teacher_forced_median_relative_l2"] = float(np.median(steps))
    branches = _conditioning_errors(native["conditioned"], reference)
    summary["conditioner_branches"] = len(branches)
    summary["conditioner_max_relative_l2"] = max(branches)
    summary["conditioner_bound"] = float(thresholds.get("conditioner_max", _CONDITIONER_MAX))
    summary["decoder_relative_l2"] = relative_l2(native["decoded"], reference["decoder"]["audio"])
    if precision == "fp32":
        bound = float(thresholds.get("fp32_max_relative_l2", _FP32_MAX_RELATIVE_L2))
        step_bound = float(thresholds.get("fp32_teacher_forced_max", _FP32_TEACHER_FORCED_MAX))
        decoder_bound = float(thresholds.get("fp32_decoder_max", _FP32_DECODER_MAX))
    else:
        official = relative_l2(references["fp16"]["audio"], reference["audio"])
        summary["official_fp16_relative_l2_vs_fp32"] = official
        bound = min(max(float(thresholds.get("fp16_drift_factor", _FP16_DRIFT_FACTOR)) * official,
                        float(thresholds.get("fp16_drift_floor", _FP16_DRIFT_FLOOR))),
                    float(thresholds.get("fp16_drift_cap", _FP16_DRIFT_CAP)))
        official_steps = [relative_l2(output, call["sample"])
                          for output, call in zip(references["fp16"]["teacher_forced"], calls)]
        summary["official_fp16_teacher_forced_max_relative_l2"] = max(official_steps)
        step_bound = max(float(thresholds.get("fp16_teacher_forced_factor", _FP16_TEACHER_FORCED_FACTOR))
                         * max(official_steps),
                         float(thresholds.get("fp16_teacher_forced_floor", _FP16_TEACHER_FORCED_FLOOR)))
        official_decoder = relative_l2(references["fp16"]["decoder_forced"], reference["decoder"]["audio"])
        summary["official_fp16_decoder_relative_l2"] = official_decoder
        decoder_bound = max(float(thresholds.get("fp16_decoder_factor", _FP16_DECODER_FACTOR)) * official_decoder,
                            float(thresholds.get("fp16_decoder_floor", _FP16_DECODER_FLOOR)))
    summary["bound"] = bound
    summary["teacher_forced_bound"] = step_bound
    summary["decoder_bound"] = decoder_bound
    record_evidence("comparison", summary)
    assert distance <= bound, summary
    assert summary["teacher_forced_max_relative_l2"] <= step_bound, summary
    assert summary["conditioner_max_relative_l2"] <= summary["conditioner_bound"], summary
    assert summary["decoder_relative_l2"] <= decoder_bound, summary
    return summary


def _assert_sdk_consumers(runtime_root: Path, bundle: Path) -> None:
    native_build = _required_path(os.environ.get("TRTMC_NATIVE_BUILD_DIR"), "TRTMC_NATIVE_BUILD_DIR")
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = ":".join(
        value for value in (str(runtime_root), environment.get("LD_LIBRARY_PATH", "")) if value)
    for language in ("c", "cpp"):
        consumer = native_build / f"test_{FAMILY}_sdk_{language}"
        assert consumer.is_file(), f"missing family SDK consumer: {consumer.name}"
        completed = subprocess.run([str(consumer), str(bundle), str(runtime_root)], check=True,
                                   capture_output=True, text=True, env=environment, timeout=900)
        assert "all tasks passed" in completed.stdout, completed.stdout
        record_evidence(f"sdk_{language}", {"stdout": completed.stdout})


def test_read_float_wav_round_trip(tmp_path: Path) -> None:
    frames = np.arange(12, dtype=np.float32).reshape(6, 2) / 7.0
    body = frames.tobytes()
    header = (b"RIFF" + (36 + len(body)).to_bytes(4, "little") + b"WAVEfmt "
              + (16).to_bytes(4, "little") + (3).to_bytes(2, "little") + (2).to_bytes(2, "little")
              + (44100).to_bytes(4, "little") + (44100 * 8).to_bytes(4, "little")
              + (8).to_bytes(2, "little") + (32).to_bytes(2, "little")
              + b"data" + len(body).to_bytes(4, "little"))
    path = tmp_path / "x.wav"
    path.write_bytes(header + body)
    audio, rate = read_float_wav(path)
    assert rate == 44100 and np.array_equal(audio, frames)
    assert relative_l2(audio, audio) == 0.0


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    _, manifest, case = CASES[case_name]
    record_evidence("inputs", {"manifest": manifest, "case": case})
    model_dir = _model_dir(manifest, tmp_path.parent)
    record_evidence("checkpoint", {"model_dir": str(model_dir), "hf_id": manifest.get("hf_id"),
                                   "hf_revision": manifest.get("hf_revision"),
                                   "random_checkpoint": manifest.get("random_checkpoint")})
    binary, runtime_root = _runtime()
    bundle = tmp_path.parent / manifest["bundle"]
    if not bundle.is_file():
        with evidence_stage("build"):
            _build(model_dir, bundle, manifest)
    with evidence_stage("native"):
        native = _native(binary, runtime_root, bundle, case, tmp_path)
    precision = str(manifest["precision"])
    with evidence_stage("reference"):
        references = {"fp32": _official_reference(model_dir, case, "fp32")}
        if precision == "fp16":
            references["fp16"] = _official_reference(model_dir, case, "fp16", references["fp32"])
    with evidence_stage("teacher_forced"):
        native.update(_bundle_stages(bundle, references["fp32"]))
    with evidence_stage("compare"):
        _assert_contract(native, references, case, precision, record_evidence("thresholds", _thresholds(case_name)))
    with evidence_stage("sdk"):
        _assert_sdk_consumers(runtime_root, bundle)
