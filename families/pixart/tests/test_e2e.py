# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime, and official-reference E2E for pixart."""

from __future__ import annotations

from tools.e2e_evidence import evidence_stage, record_evidence
from families.pixart.tests.reporting import (
    native_snapshot, record_native_preview, record_report_views, reference_snapshot,
)
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType
import pytest
import numpy as np
from tensorrt_model_connect import BuildRequest, build

FAMILY = "pixart"
TASKS = frozenset({"image_generation"})
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"
THRESHOLD_ROOT = TEST_ROOT / "thresholds"
_MPI_RANK_ZERO = re.compile(r"^\[[^,]+,0\]<stdout>:(.*)$")


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
            repo_id=manifest["hf_id"],
            revision=manifest.get("hf_revision"),
            local_files_only=True,
            allow_patterns=["model_index.json"],
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
    tp_size = int(manifest["tensor_parallel_size"])
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(
        (value for value in (str(runtime_root), env.get("LD_LIBRARY_PATH", "")) if value)
    )
    invocation = [
        str(binary),
        command,
        str(bundle),
        "--runtime-root",
        str(runtime_root),
        *arguments,
    ]
    if tp_size > 1:
        mpirun = shutil.which("mpirun")
        assert mpirun, "selected multi-GPU E2E requires mpirun"
        env["TRTMC_NCCL_RENDEZVOUS"] = str(bundle.with_suffix(".nccl-rendezvous"))
        prefix = [mpirun, "--tag-output", "-np", str(tp_size)]
        for name in ("LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "TRTMC_NCCL_RENDEZVOUS"):
            if name in env:
                prefix.extend(["-x", name])
        invocation = [*prefix, *invocation]
    completed = subprocess.run(
        invocation,
        check=True,
        capture_output=True,
        text=True,
        env=env,
        timeout=int(case.get("runtime_timeout_s", 3600)),
    )
    record_evidence("commands", {"argv": getattr(completed, "args", None)})
    record_evidence("native", {"stdout": getattr(completed, "stdout", None), "stderr": getattr(completed, "stderr", None)})
    payloads = []
    for line in completed.stdout.splitlines():
        if tp_size > 1:
            match = _MPI_RANK_ZERO.fullmatch(line)
            candidate = match.group(1) if match else ""
        else:
            start = line.find("{")
            candidate = line[start:] if start >= 0 else ""
        if candidate.lstrip().startswith("{"):
            try:
                payloads.append(json.loads(candidate))
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
    record_evidence("inputs", {"asset": path})
    return path


def _case_text(case: dict) -> str:
    inputs = case.get("inputs") or {}
    value = str(case.get("prompt") or case.get("test_prompt") or inputs.get("prompt") or "")
    assert value, f"selected {FAMILY} E2E requires a direct prompt"
    return value


def _cast_floating(value, dtype):
    import torch

    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype) if value.is_floating_point() else value
    if isinstance(value, tuple):
        return tuple(_cast_floating(item, dtype) for item in value)
    if isinstance(value, list):
        return [_cast_floating(item, dtype) for item in value]
    if isinstance(value, dict):
        return {key: _cast_floating(item, dtype) for key, item in value.items()}
    return value


def _reference_pipeline(model_dir: Path):
    import torch
    from diffusers import PixArtSigmaPipeline
    from transformers import T5EncoderModel

    text_encoder = T5EncoderModel.from_pretrained(
        model_dir,
        subfolder="text_encoder",
        torch_dtype=torch.float32,
        local_files_only=True,
    )
    pipeline = PixArtSigmaPipeline.from_pretrained(
        model_dir,
        text_encoder=text_encoder,
        torch_dtype=torch.float16,
        local_files_only=True,
    )

    def fp16_transformer_inputs(_module, args, kwargs):
        return _cast_floating(args, torch.float16), _cast_floating(kwargs, torch.float16)

    def fp32_transformer_output(_module, _args, output):
        return _cast_floating(output, torch.float32)

    pipeline.transformer.register_forward_pre_hook(fp16_transformer_inputs, with_kwargs=True)
    pipeline.transformer.register_forward_hook(fp32_transformer_output)
    return pipeline.to("cuda")


def _initial_latents(manifest: dict, case: dict) -> np.ndarray:
    height = int(manifest["image_height"])
    width = int(manifest["image_width"])
    assert height % 8 == 0 and width % 8 == 0
    shape = (1, 4, height // 8, width // 8)
    return np.random.default_rng(int(case["seed"])).standard_normal(shape, dtype=np.float32)


def _native(
    binary: Path,
    runtime_root: Path,
    bundle: Path,
    model_dir: Path,
    manifest: dict,
    case: dict,
    tmp_path: Path,
    initial_latents: np.ndarray,
):
    manifest["task"]
    frames = int(manifest["video_num_frames"]) if "video_num_frames" in manifest else 1
    is_video = frames > 1 or "video" in str(case.get("test_type", ""))
    command = "generate-video" if is_video else "generate-image"
    output = tmp_path / ("native-frames" if is_video else "native.png")
    arguments = [
        "--prompt",
        _case_text(case),
        "--output",
        str(output),
        "--height",
        str(int(manifest["image_height"])),
        "--width",
        str(int(manifest["image_width"])),
        "--num-steps",
        str(int(case["num_inference_steps"])),
        "--seed",
        str(int(case["seed"])),
    ]
    if case.get("negative_prompt"):
        arguments.extend(("--negative-prompt", str(case["negative_prompt"])))
    for key, option in (("guidance_scale", "--guidance-scale"), ("cfg_scale", "--cfg-scale")):
        if key in case:
            arguments.extend((option, str(float(case[key]))))
    latents_path = tmp_path / "initial-latents.raw"
    np.ascontiguousarray(initial_latents, dtype=np.float32).tofile(latents_path)
    record_evidence("inputs", {"raw_file": latents_path})
    arguments.extend(("--initial-latents-raw", str(latents_path)))
    payload = _run_json(binary, runtime_root, bundle, manifest, case, command, *arguments)
    payload["artifact"] = str(output)
    record_native_preview(output)
    return payload


def _official_reference(
    model_dir: Path,
    manifest: dict,
    case: dict,
    tmp_path: Path,
    initial_latents: np.ndarray,
):
    task = manifest["task"]
    import torch

    pipeline = _reference_pipeline(model_dir)
    prompts = _case_text(case)
    generator = torch.Generator(device="cuda").manual_seed(int(case["seed"]))
    kwargs = {
        "prompt": prompts,
        "height": int(manifest["image_height"]),
        "width": int(manifest["image_width"]),
        "num_inference_steps": int(case["num_inference_steps"]),
        "generator": generator,
        "latents": torch.from_numpy(initial_latents.copy()).to("cuda"),
    }
    if int(manifest.get("video_num_frames", 1)) > 1:
        kwargs["num_frames"] = int(manifest["video_num_frames"])
    if case.get("negative_prompt"):
        kwargs["negative_prompt"] = case["negative_prompt"]
    if "guidance_scale" in case:
        kwargs["guidance_scale"] = float(case["guidance_scale"])
    output = pipeline(**kwargs)
    if task == "world_model_generation" or int(manifest.get("video_num_frames", 1)) > 1:
        images = output.frames[0]
    else:
        images = output.images
    return {
        "images": [np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0 for image in images]
    }


def _write_semantic_artifacts(case: dict, actual_images: list, expected_images: list) -> None:
    if not case.get("semantic_assessment"):
        return
    output_root = os.environ.get("TRTMC_E2E_ARTIFACT_DIR")
    if not output_root:
        return
    from PIL import Image

    output = Path(output_root) / str(case["name"])
    output.mkdir(parents=True, exist_ok=False)
    paired_count = min(len(actual_images), len(expected_images))
    assert paired_count > 0
    sample_count = int(case.get("semantic_samples", 1))
    assert 1 <= sample_count <= min(6, paired_count)
    if sample_count == 1:
        sample_indices = [(paired_count - 1) // 2]
    else:
        sample_indices = [
            round(index * (paired_count - 1) / (sample_count - 1)) for index in range(sample_count)
        ]
    for source_index in sample_indices:
        actual = actual_images[source_index]
        expected = expected_images[source_index]
        for label, image in (("trt", actual), ("reference", expected)):
            pixels = np.rint(np.clip(np.asarray(image), 0.0, 1.0) * 255.0).astype(np.uint8)
            Image.fromarray(pixels).save(output / f"{label}-{source_index:03d}.png")
    (output / "metadata.json").write_text(
        json.dumps(
            {
                "family": "pixart",
                "case": case["name"],
                "frame_count": len(actual_images),
                "prompt": _case_text(case),
                "sample_count": sample_count,
                "sampled_frame_indices": sample_indices,
                "schema_version": 1,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _psnr(actual: np.ndarray, expected: np.ndarray) -> float:
    mse = float(np.mean((actual.astype(np.float64) - expected.astype(np.float64)) ** 2))
    return 100.0 if mse < 1.0e-12 else float(10.0 * np.log10(1.0 / mse))


def _ssim(actual: np.ndarray, expected: np.ndarray) -> float:
    left = actual.astype(np.float64)
    right = expected.astype(np.float64)
    left_mean = float(left.mean())
    right_mean = float(right.mean())
    left_variance = float(left.var())
    right_variance = float(right.var())
    covariance = float(np.mean((left - left_mean) * (right - right_mean)))
    c1 = 0.01**2
    c2 = 0.03**2
    return float(
        ((2.0 * left_mean * right_mean + c1) * (2.0 * covariance + c2))
        / ((left_mean**2 + right_mean**2 + c1) * (left_variance + right_variance + c2))
    )


def _assert_contract(actual, expected, manifest: dict, case: dict, thresholds: dict) -> None:
    del manifest
    from PIL import Image

    artifact = Path(actual["artifact"])
    if artifact.is_dir():
        actual_paths = sorted(artifact.glob("*.png"))
    else:
        actual_paths = [artifact]
    assert actual_paths
    actual_images = [
        np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        for path in actual_paths
    ]
    expected_images = expected["images"]
    assert len(actual_images) == len(expected_images) and actual_images
    expected_images = [np.asarray(image, dtype=np.float32) for image in expected_images]
    assert all(
        actual.shape == reference.shape
        for actual, reference in zip(actual_images, expected_images, strict=True)
    )
    pixels = np.asarray(actual_images, dtype=np.float32)
    reference_pixels = np.asarray(expected_images, dtype=np.float32)
    for values in (pixels, reference_pixels):
        assert float(values.mean()) >= float(thresholds["min_pixel_mean"])
        assert float(values.mean()) <= float(thresholds["max_pixel_mean"])
        assert float(values.std()) >= float(thresholds["min_pixel_std"])
    reference_std = float(reference_pixels.std())
    if reference_std >= float(thresholds["reference_min_pixel_std_for_ratio"]):
        assert float(pixels.std()) / reference_std >= float(thresholds["min_reference_std_ratio"])
    psnr = np.mean(
        [
            _psnr(actual, reference)
            for actual, reference in zip(actual_images, expected_images, strict=True)
        ]
    )
    ssim = np.mean(
        [
            _ssim(actual, reference)
            for actual, reference in zip(actual_images, expected_images, strict=True)
        ]
    )
    assert float(psnr) >= float(thresholds["psnr"])
    assert float(ssim) >= float(thresholds["ssim"])
    _write_semantic_artifacts(case, actual_images, expected_images)


def test_reference_pipeline_restores_component_dtypes_and_transformer_hooks(monkeypatch) -> None:
    import torch

    captured: dict[str, object] = {}

    class FakeTransformer:
        def register_forward_pre_hook(self, hook, *, with_kwargs):
            captured["pre_hook"] = hook
            captured["with_kwargs"] = with_kwargs

        def register_forward_hook(self, hook):
            captured["output_hook"] = hook

    class FakePipeline:
        transformer = FakeTransformer()

        def to(self, device):
            captured["device"] = device
            return self

    text_encoder = object()
    pipeline = FakePipeline()

    def fake_t5(model_dir, **kwargs):
        captured["t5"] = (model_dir, kwargs)
        return text_encoder

    def fake_pipeline(model_dir, **kwargs):
        captured["pipeline"] = (model_dir, kwargs)
        return pipeline

    class FakeT5EncoderModel:
        from_pretrained = staticmethod(fake_t5)

    class FakePixArtSigmaPipeline:
        from_pretrained = staticmethod(fake_pipeline)

    transformers = ModuleType("transformers")
    transformers.T5EncoderModel = FakeT5EncoderModel
    diffusers = ModuleType("diffusers")
    diffusers.PixArtSigmaPipeline = FakePixArtSigmaPipeline
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)

    assert _reference_pipeline(Path("/model")) is pipeline
    assert captured["t5"] == (
        Path("/model"),
        {
            "subfolder": "text_encoder",
            "torch_dtype": torch.float32,
            "local_files_only": True,
        },
    )
    assert captured["pipeline"] == (
        Path("/model"),
        {
            "text_encoder": text_encoder,
            "torch_dtype": torch.float16,
            "local_files_only": True,
        },
    )
    assert captured["device"] == "cuda"
    assert captured["with_kwargs"] is True

    floating = torch.ones(1)
    integer = torch.ones(1, dtype=torch.int64)
    args, kwargs = captured["pre_hook"](None, (floating,), {"mask": integer})
    assert args[0].dtype == torch.float16
    assert kwargs["mask"] is integer
    assert captured["output_hook"](None, (), floating).dtype == torch.float32


def test_tp_launcher_exports_rendezvous_and_selects_rank_zero(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        stdout = "\n".join(('[1,1]<stdout>:{"rank":1}', '[1,0]<stdout>:{"rank":0}'))
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/mpirun")
    monkeypatch.setattr(subprocess, "run", fake_run)
    bundle = tmp_path / "pixart.bundle"

    payload = _run_json(
        Path("/trtmc"),
        Path("/runtime"),
        bundle,
        {"tensor_parallel_size": 4},
        {},
        "generate-image",
    )

    assert payload == {"rank": 0}
    assert captured["env"]["TRTMC_NCCL_RENDEZVOUS"] == str(bundle.with_suffix(".nccl-rendezvous"))
    command = captured["command"]
    for name in ("LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "TRTMC_NCCL_RENDEZVOUS"):
        assert command[command.index(name) - 1] == "-x"


def test_semantic_artifacts_are_paired(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TRTMC_E2E_ARTIFACT_DIR", str(tmp_path))
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("a test image\n", encoding="utf-8")
    case = {
        "name": "semantic-probe",
        "prompt": "a test image",
        "prompt_file": str(prompt_file),
        "semantic_assessment": True,
    }
    _write_semantic_artifacts(
        case,
        [np.zeros((2, 2, 3), dtype=np.float32)],
        [np.ones((2, 2, 3), dtype=np.float32)],
    )
    output = tmp_path / "semantic-probe"
    assert sorted(path.name for path in output.iterdir()) == [
        "metadata.json",
        "reference-000.png",
        "trt-000.png",
    ]
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata == {
        "case": "semantic-probe",
        "family": "pixart",
        "frame_count": 1,
        "prompt": "a test image",
        "sample_count": 1,
        "sampled_frame_indices": [0],
        "schema_version": 1,
    }


@pytest.mark.parametrize("threshold_case", sorted(CASES))
def test_reference_metrics_reject_rearranged_pixels(
    threshold_case: str, tmp_path: Path
) -> None:
    from PIL import Image

    actual_pixels = np.array(
        [[[0, 0, 0], [255, 255, 255]], [[255, 255, 255], [0, 0, 0]]], dtype=np.uint8
    )
    reference = (
        np.array([[[255, 255, 255], [0, 0, 0]], [[0, 0, 0], [255, 255, 255]]], dtype=np.float32)
        / 255.0
    )
    actual_path = tmp_path / "actual.png"
    Image.fromarray(actual_pixels).save(actual_path)
    with pytest.raises(AssertionError):
        _assert_contract(
            {"artifact": str(actual_path)},
            {"images": [reference]},
            {},
            {"name": "adversarial"},
            _thresholds(threshold_case),
        )


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    _, manifest, case = CASES[case_name]
    record_evidence("inputs", {"manifest": manifest, "case": CASES[case_name][-1]})
    model_dir = _model_dir(manifest)
    record_evidence("checkpoint", {"model_dir": str(model_dir), "hf_id": manifest.get("hf_id"), "hf_revision": manifest.get("hf_revision")})
    binary, runtime_root = _runtime(manifest)
    bundle = tmp_path / manifest["bundle"]
    with evidence_stage("build"):
        _build(model_dir, bundle, manifest)
    initial_latents = _initial_latents(manifest, case)
    record_evidence("inputs", {"initial_latents": None if initial_latents is None else {"shape": list(initial_latents.shape), "dtype": str(initial_latents.dtype), "note": "The native stage records the supplied raw latent file within the evidence bound."}})
    with evidence_stage("native"):
        actual = _native(
            binary,
            runtime_root,
            bundle,
            model_dir,
            manifest,
            case,
            tmp_path,
            initial_latents,
        )
    record_evidence("native", native_snapshot(actual))
    with evidence_stage("reference"):
        expected = _official_reference(model_dir, manifest, case, tmp_path, initial_latents)
    record_report_views(actual, expected, tmp_path / "paired-report-views")
    record_evidence("reference", reference_snapshot(expected))
    with evidence_stage("compare"):
        _assert_contract(actual, expected, manifest, case, record_evidence("thresholds", _thresholds(case_name)))
