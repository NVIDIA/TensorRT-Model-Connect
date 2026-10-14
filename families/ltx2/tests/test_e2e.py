# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime and diffusers-reference E2E for ltx2 (LTX-2.5 distilled).

The native run and the diffusers ``LTX2Pipeline`` start from the same seeded noise: the
native runtime reads it through ``TRTMC_LTX2_INITIAL_LATENTS`` (packed, normalized video
then audio noise) and the reference receives the equivalent unpacked, denormalized
``latents``/``audio_latents``, which its ``prepare_latents`` maps back to the same noise.
Both run the checkpoint's 8-step distilled schedule without guidance.

Video frames are compared by PSNR. The released pipeline runs its vocoder in bf16, which
alone moves the waveform by a log-spectrum L1 of about 1.2 relative to an fp32 vocoder,
so the soundtrack is compared by the correlation of the log spectrograms instead of
sample-wise.
"""

from __future__ import annotations

import gc
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tensorrt_model_connect import BuildRequest, build
from tools.e2e_evidence import evidence_stage, record_evidence

FAMILY = "ltx2"
TASKS = frozenset({"text_to_audio_video"})
TEST_ROOT = Path(__file__).resolve().parent
REPO = TEST_ROOT.parents[2]
MANIFEST_ROOT = TEST_ROOT / "manifests"
THRESHOLD_ROOT = TEST_ROOT / "thresholds"

# diffusers ``pipelines/ltx2/utils.py`` DISTILLED_SIGMA_VALUES (also baked into the bundle).
DISTILLED_SIGMAS = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875]
FRAME_RATE = 24.0
LATENT_CHANNELS = 128
AUDIO_LATENT_CHANNELS = 8
AUDIO_LATENT_MEL_BINS = 16


def _case_index() -> dict[str, tuple[Path, dict, dict]]:
    result = {}
    for path in sorted(MANIFEST_ROOT.glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["family"] == FAMILY
        assert manifest["task"] in TASKS
        assert manifest["tensor_parallel_size"] == 1
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
            repo_id=manifest["hf_id"],
            revision=manifest.get("hf_revision"),
            local_files_only=True,
        )
    except Exception as error:
        raise AssertionError(
            f"selected {FAMILY} E2E requires the exact cached checkpoint {manifest['hf_id']}"
        ) from error
    return Path(snapshot)


def _parallel_size(manifest: dict) -> int:
    return int(manifest.get("context_parallel_size", 1))


def _backend() -> str:
    """The build backend: ``trt`` when TensorRT is installed, else ``trt_rtx`` (as conftest binds)."""
    import importlib.util

    choice = os.environ.get("TRTMC_LTX2_TEST_BACKEND", "").strip()
    if choice:
        return choice
    if sys.modules.get("tensorrt_rtx") is not None:  # conftest bound TensorRT-RTX
        return "trt_rtx"
    return "trt" if importlib.util.find_spec("tensorrt") is not None else "trt_rtx"


def _library(runtime_root: Path, name: str) -> Path:
    return runtime_root / (f"{name}.dll" if sys.platform == "win32" else f"lib{name}.so")


def _runtime(manifest: dict) -> tuple[Path, Path]:
    binary = _required_path(os.environ.get("TRTMC_BINARY"), "TRTMC_BINARY")
    runtime_root = _required_path(os.environ.get("TRTMC_RUNTIME_ROOT"), "TRTMC_RUNTIME_ROOT")
    assert _library(runtime_root, f"trtmc_backend_{_backend()}").is_file()
    assert _library(runtime_root, f"trtmc_model_{FAMILY}").is_file()
    import torch

    required_gpus = _parallel_size(manifest)
    assert torch.cuda.is_available(), f"selected {FAMILY} E2E requires CUDA"
    assert torch.cuda.device_count() >= required_gpus, (
        f"selected {FAMILY} E2E requires {required_gpus} GPUs, found {torch.cuda.device_count()}"
    )
    return binary, runtime_root


def _build(model_dir: Path, bundle: Path, manifest: dict) -> None:
    build(
        BuildRequest(
            model_dir=model_dir,
            output_path=bundle,
            family=FAMILY,
            task=manifest["task"],
            precision=manifest["precision"],
            backend=_backend(),
            image_height=manifest.get("image_height"),
            image_width=manifest.get("image_width"),
            video_num_frames=manifest.get("video_num_frames"),
            tensor_parallel_size=int(manifest["tensor_parallel_size"]),
            context_parallel_size=_parallel_size(manifest),
        )
    )


def _select_json_payload(stdout: str, parallel_size: int) -> dict:
    """The output rank's JSON line (rank 0 under ``mpirun --tag-output`` or ``launch_ranks``)."""
    payloads = []
    for line in stdout.splitlines():
        if parallel_size > 1 and not line.startswith("[1,0]<stdout>:"):
            continue
        start = line.find("{")
        if start >= 0:
            try:
                payloads.append(json.loads(line[start:]))
            except json.JSONDecodeError:
                pass
    payloads = [payload for payload in payloads if not payload.get("worker")]
    assert len(payloads) == 1, f"expected one output-rank JSON payload: {stdout[-2000:]}"
    return payloads[0]


def test_json_selection_uses_only_the_output_rank() -> None:
    stdout = "\n".join(
        (
            '[1,1]<stdout>:{"worker": true}',
            '[1,0]<stdout>:{"output": "frames", "audio": "frames/audio.wav"}',
        )
    )

    assert _select_json_payload(stdout, 2) == {"output": "frames", "audio": "frames/audio.wav"}


def _run_json(
    binary: Path,
    runtime_root: Path,
    bundle: Path,
    manifest: dict,
    case: dict,
    noise_path: Path,
    *arguments: str,
) -> dict:
    invocation = [
        str(binary),
        "generate-video",
        str(bundle),
        "--runtime-root",
        str(runtime_root),
        *arguments,
    ]
    parallel_size = _parallel_size(manifest)
    env = os.environ.copy()
    env["TRTMC_LTX2_INITIAL_LATENTS"] = str(noise_path)
    if parallel_size > 1 and shutil.which("mpirun"):
        rendezvous = bundle.with_suffix(".nccl-rendezvous")
        # A file left by an interrupted launch would hand its stale id to the ranks.
        rendezvous.unlink(missing_ok=True)
        env["TRTMC_NCCL_RENDEZVOUS"] = str(rendezvous)
        invocation = [
            shutil.which("mpirun"),
            "--tag-output",
            "-x",
            "LD_LIBRARY_PATH",
            "-x",
            "TRTMC_NCCL_RENDEZVOUS",
            "-x",
            "TRTMC_LTX2_INITIAL_LATENTS",
            "-np",
            str(parallel_size),
            *invocation,
        ]
    elif parallel_size > 1:
        # Hosts without OpenMPI (native Windows): the repository's local rank launcher
        # provides the same rank environment and a fresh rendezvous file per launch.
        invocation = [
            sys.executable,
            str(REPO / "tools" / "launch_ranks.py"),
            "-n",
            str(parallel_size),
            "--",
            *invocation,
        ]
    if sys.platform == "win32":
        env["PATH"] = os.pathsep.join(value for value in (str(runtime_root), env.get("PATH", "")) if value)
    else:
        env["LD_LIBRARY_PATH"] = ":".join(
            value for value in (str(runtime_root), env.get("LD_LIBRARY_PATH", "")) if value
        )
    completed = subprocess.run(
        invocation,
        capture_output=True,
        text=True,
        env=env,
        timeout=int(case.get("runtime_timeout_s", 3600)),
    )
    record_evidence("commands", {"argv": getattr(completed, "args", None)})
    record_evidence("native", {"stdout": completed.stdout[-20000:], "stderr": completed.stderr[-20000:]})
    assert completed.returncode == 0, (
        f"native generate-video failed ({completed.returncode}): {completed.stderr[-4000:]}"
    )
    return _select_json_payload(completed.stdout, parallel_size)


def _thresholds(case_name: str) -> dict:
    path = THRESHOLD_ROOT / f"{case_name}.json"
    assert path.is_file(), f"selected {FAMILY} E2E requires exact thresholds: {path}"
    return json.loads(path.read_text(encoding="utf-8"))["threshold_overrides"]


def _case_text(case: dict) -> str:
    value = str(case.get("test_prompt") or "")
    assert value, f"selected {FAMILY} E2E requires a direct prompt"
    return value


def _latent_layout(manifest: dict) -> tuple[int, int, int, int]:
    frames = int(manifest["video_num_frames"])
    height = int(manifest["image_height"])
    width = int(manifest["image_width"])
    assert (frames - 1) % 8 == 0 and height % 32 == 0 and width % 32 == 0
    # LTX2Pipeline: round(duration * sampling_rate / hop_length / temporal_compression).
    audio_frames = round(frames / FRAME_RATE * 16000 / 160 / 4)
    return (frames - 1) // 8 + 1, height // 32, width // 32, audio_frames


def _initial_noise(manifest: dict, case: dict) -> tuple[np.ndarray, np.ndarray]:
    """Packed, normalized noise: video ``[S, 128]`` (tokens f, h, w) and audio ``[Sa, 128]``."""
    latent_frames, latent_height, latent_width, audio_frames = _latent_layout(manifest)
    rng = np.random.default_rng(int(case["seed"]))
    video = rng.standard_normal((latent_frames * latent_height * latent_width, LATENT_CHANNELS), dtype=np.float32)
    audio = rng.standard_normal((audio_frames, AUDIO_LATENT_CHANNELS * AUDIO_LATENT_MEL_BINS), dtype=np.float32)
    return video, audio


def _native(binary, runtime_root, bundle, manifest, case, tmp_path, noise) -> dict:
    noise_path = tmp_path / "initial-noise.f32"
    np.concatenate([noise[0].ravel(), noise[1].ravel()]).astype(np.float32).tofile(noise_path)
    record_evidence("inputs", {"raw_file": noise_path})
    output = tmp_path / "native-frames"
    payload = _run_json(
        binary, runtime_root, bundle, manifest, case, noise_path,
        "--prompt", _case_text(case), "--output", str(output), "--set", f"seed={int(case['seed'])}",
    )
    payload["artifact"] = str(output)
    return payload


def _official_reference(model_dir: Path, manifest: dict, case: dict, noise) -> dict:
    import torch
    from diffusers import LTX2Pipeline

    # The prompt enhancer (with its processor) and the duration head are optional components the
    # distilled text-to-audio-video call does not use; the bundle does not contain them either.
    pipeline = LTX2Pipeline.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, local_files_only=True,
        processor=None, prompt_enhancer=None, duration_head=None,
    ).to("cuda")
    latent_frames, latent_height, latent_width, audio_frames = _latent_layout(manifest)
    # Denormalize so that the pipeline's prepare_latents normalizes back to the same noise.
    vae, audio_vae = pipeline.vae, pipeline.audio_vae
    video = torch.from_numpy(noise[0]).reshape(1, latent_frames, latent_height, latent_width, LATENT_CHANNELS)
    video = video.permute(0, 4, 1, 2, 3).to("cuda", torch.float32)
    mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(video)
    std = vae.latents_std.view(1, -1, 1, 1, 1).to(video)
    video = video * std / vae.config.scaling_factor + mean
    audio = torch.from_numpy(noise[1]).reshape(1, audio_frames, -1).to("cuda", torch.float32)
    audio = audio * audio_vae.latents_std.to(audio) + audio_vae.latents_mean.to(audio)
    audio = audio.unflatten(2, (AUDIO_LATENT_CHANNELS, AUDIO_LATENT_MEL_BINS)).transpose(1, 2)
    frames, waveform = pipeline(
        prompt=_case_text(case),
        width=int(manifest["image_width"]),
        height=int(manifest["image_height"]),
        num_frames=int(manifest["video_num_frames"]),
        frame_rate=FRAME_RATE,
        sigmas=DISTILLED_SIGMAS,
        guidance_scale=1.0,
        audio_guidance_scale=1.0,
        stg_scale=0.0,
        audio_stg_scale=0.0,
        modality_scale=1.0,
        audio_modality_scale=1.0,
        guidance_rescale=0.0,
        audio_guidance_rescale=0.0,
        spatio_temporal_guidance_blocks=None,
        use_cross_timestep=True,
        enable_prompt_enhancement=False,
        latents=video,
        audio_latents=audio,
        generator=torch.Generator("cuda").manual_seed(int(case["seed"])),
        output_type="np",
        return_dict=False,
    )
    result = {
        "frames": (np.clip(frames[0], 0, 1) * 255).round().astype(np.uint8),
        "audio": waveform[0].float().cpu().numpy(),
        "audio_sample_rate": int(pipeline.vocoder.config.output_sampling_rate),
    }
    # Release the reference before the next case's native run needs the GPU memory.
    del pipeline, frames, waveform
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    import struct

    data = path.read_bytes()
    position, layout, samples = 12, None, b""
    while position + 8 <= len(data):
        chunk, size = data[position:position + 4], struct.unpack("<I", data[position + 4:position + 8])[0]
        body = data[position + 8:position + 8 + size]
        if chunk == b"fmt ":
            layout = struct.unpack("<HHIIHH", body[:16])
        elif chunk == b"data":
            samples = body
        position += 8 + size + (size & 1)
    assert layout is not None, f"{path} has no fmt chunk"
    tag, channels, rate, _, _, bits = layout
    if tag == 3 and bits == 32:
        values = np.frombuffer(samples, "<f4")
    else:
        assert tag == 1 and bits == 16, f"unsupported WAV layout {layout}"
        values = np.frombuffer(samples, "<i2").astype(np.float32) / 32768.0
    return values.reshape(-1, channels).T, rate


def _log_spectrogram(wave: np.ndarray) -> np.ndarray:
    size, hop = 2048, 512
    windows = np.lib.stride_tricks.sliding_window_view(wave, size, axis=-1)[..., ::hop, :]
    return np.log(np.abs(np.fft.rfft(windows * np.hanning(size), axis=-1)) + 1e-5)


def _metrics(actual: dict, expected: dict) -> dict:
    from PIL import Image

    paths = sorted(Path(actual["artifact"]).glob("frame-*.png"))
    frames = np.stack([np.asarray(Image.open(path).convert("RGB")) for path in paths]).astype(np.float64)
    reference = expected["frames"].astype(np.float64)
    assert frames.shape == reference.shape, (frames.shape, reference.shape)
    mse = ((frames - reference) ** 2).mean(axis=(1, 2, 3))
    psnr = 10.0 * np.log10(255.0**2 / np.maximum(mse, 1e-12))
    wave, rate = _read_wav(Path(actual["audio"]))
    reference_wave = expected["audio"]
    assert rate == expected["audio_sample_rate"]
    assert wave.shape == reference_wave.shape, (wave.shape, reference_wave.shape)
    spectrogram = _log_spectrogram(wave).ravel()
    reference_spectrogram = _log_spectrogram(reference_wave).ravel()
    return {
        "frames": int(frames.shape[0]),
        "frame_psnr_mean_db": float(psnr.mean()),
        "frame_psnr_min_db": float(psnr.min()),
        "audio_log_spectrogram_corr": float(np.corrcoef(spectrogram, reference_spectrogram)[0, 1]),
        "audio_rms_ratio": float(np.sqrt((wave**2).mean()) / max(np.sqrt((reference_wave**2).mean()), 1e-12)),
    }


def _assert_contract(metrics: dict, manifest: dict, thresholds: dict) -> None:
    assert metrics["frames"] == int(manifest["video_num_frames"])
    assert metrics["frame_psnr_mean_db"] >= float(thresholds["min_frame_psnr_mean_db"])
    assert metrics["frame_psnr_min_db"] >= float(thresholds["min_frame_psnr_min_db"])
    assert metrics["audio_log_spectrogram_corr"] >= float(thresholds["min_audio_log_spectrogram_corr"])
    low, high = (float(value) for value in thresholds["audio_rms_ratio_range"])
    assert low <= metrics["audio_rms_ratio"] <= high


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    _, manifest, case = CASES[case_name]
    record_evidence("inputs", {"manifest": manifest, "case": case})
    model_dir = _model_dir(manifest)
    record_evidence(
        "checkpoint",
        {"model_dir": str(model_dir), "hf_id": manifest.get("hf_id"), "hf_revision": manifest.get("hf_revision")},
    )
    binary, runtime_root = _runtime(manifest)
    bundle = tmp_path / manifest["bundle"]
    with evidence_stage("build"):
        _build(model_dir, bundle, manifest)
    noise = _initial_noise(manifest, case)
    with evidence_stage("native"):
        actual = _native(binary, runtime_root, bundle, manifest, case, tmp_path, noise)
    record_evidence("native", {"payload": {key: actual[key] for key in actual if key != "frames"}})
    with evidence_stage("reference"):
        expected = _official_reference(model_dir, manifest, case, noise)
    with evidence_stage("compare"):
        metrics = record_evidence("metrics", _metrics(actual, expected))
        print(f"[ltx2-e2e] {case_name} {json.dumps(metrics)}")
        _assert_contract(metrics, manifest, record_evidence("thresholds", _thresholds(case_name)))
