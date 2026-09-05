# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime, and official-reference E2E for z_image."""

from __future__ import annotations
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

FAMILY = "z_image"
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
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))["threshold_overrides"]


def _asset(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = TEST_ROOT / path
    assert path.is_file(), f"selected {FAMILY} E2E asset does not exist: {path}"
    return path


def _case_text(case: dict) -> str:
    inputs = case.get("inputs") or {}
    value = str(case.get("prompt") or case.get("test_prompt") or inputs.get("prompt") or "")
    assert value, f"selected {FAMILY} E2E requires a direct prompt"
    return value


def _reference_pipeline(model_dir: Path):
    import torch
    from diffusers import DiffusionPipeline

    return DiffusionPipeline.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, local_files_only=True
    ).to("cuda")


def _initial_latents(manifest: dict, case: dict) -> np.ndarray:
    height = int(manifest["image_height"])
    width = int(manifest["image_width"])
    assert height > 0 and width > 0 and height % 8 == 0 and width % 8 == 0
    shape = (1, 16, height // 8, width // 8)
    return np.random.default_rng(int(case["seed"])).standard_normal(shape, dtype=np.float32)


def _require_finite_latents(_pipeline, step, _timestep, callback_kwargs):
    import torch

    if not torch.isfinite(callback_kwargs["latents"]).all():
        raise RuntimeError(f"Z-Image HF reference produced non-finite latents at step {step}")
    return callback_kwargs


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
    initial_latents_path = tmp_path / "initial-latents.f32"
    initial_latents.tofile(initial_latents_path)
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
        "--initial-latents-raw",
        str(initial_latents_path),
    ]
    if case.get("negative_prompt"):
        arguments.extend(("--negative-prompt", str(case["negative_prompt"])))
    for key, option in (("guidance_scale", "--guidance-scale"), ("cfg_scale", "--cfg-scale")):
        if key in case:
            arguments.extend((option, str(float(case[key]))))
    payload = _run_json(binary, runtime_root, bundle, manifest, case, command, *arguments)
    payload["artifact"] = str(output)
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
    reference_latents = torch.from_numpy(initial_latents.copy()).to(
        device="cuda", dtype=torch.bfloat16
    )
    kwargs = {
        "prompt": prompts,
        "height": int(manifest["image_height"]),
        "width": int(manifest["image_width"]),
        "num_inference_steps": int(case["num_inference_steps"]),
        "generator": generator,
        "latents": reference_latents,
        "callback_on_step_end": _require_finite_latents,
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
                "family": "z_image",
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
    assert expected_images
    pixels = np.asarray(actual_images)
    assert float(pixels.mean()) >= float(thresholds.get("min_pixel_mean", 0.15))
    assert float(pixels.mean()) <= float(thresholds.get("max_pixel_mean", 0.85))
    assert float(pixels.std()) >= float(thresholds.get("min_pixel_std", 0.05))
    _write_semantic_artifacts(case, actual_images, expected_images)


def test_initial_latents_match_the_family_reference_rng() -> None:
    manifest = {"image_height": 16, "image_width": 24}
    case = {"seed": 43}

    actual = _initial_latents(manifest, case)

    assert actual.shape == (1, 16, 2, 3)
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(
        actual,
        np.random.default_rng(43).standard_normal((1, 16, 2, 3), dtype=np.float32),
    )


def test_native_consumes_the_exact_raw_initial_latents(monkeypatch, tmp_path: Path) -> None:
    initial_latents = np.random.default_rng(7).standard_normal((1, 16, 2, 2), dtype=np.float32)
    captured: dict[str, tuple[str, ...]] = {}

    def fake_run_json(_binary, _runtime, _bundle, _manifest, _case, _command, *arguments):
        captured["arguments"] = arguments
        return {}

    monkeypatch.setattr("families.z_image.tests.test_e2e._run_json", fake_run_json)
    _native(
        Path("/trtmc"),
        Path("/runtime"),
        tmp_path / "model.bundle",
        Path("/model"),
        {
            "task": "image_generation",
            "image_height": 16,
            "image_width": 16,
            "tensor_parallel_size": 1,
        },
        {"prompt": "cat", "num_inference_steps": 1, "seed": 7},
        tmp_path,
        initial_latents,
    )

    arguments = captured["arguments"]
    latent_path = Path(arguments[arguments.index("--initial-latents-raw") + 1])
    np.testing.assert_array_equal(
        np.fromfile(latent_path, dtype=np.float32), initial_latents.ravel()
    )


def test_hf_reference_keeps_bf16_paired_latents_and_finite_callback(
    monkeypatch, tmp_path: Path
) -> None:
    import torch
    from PIL import Image

    captured: dict[str, object] = {}

    class FakeLatents:
        def to(self, **kwargs):
            captured["latent_to"] = kwargs
            return self

    class FakeGenerator:
        def __init__(self, *, device):
            captured["generator_device"] = device

        def manual_seed(self, seed):
            captured["seed"] = seed
            return self

    class FakePipeline:
        def to(self, device):
            captured["pipeline_device"] = device
            return self

        def __call__(self, **kwargs):
            captured["generation"] = kwargs
            return type("Output", (), {"images": [Image.new("RGB", (2, 2))]})()

    pipeline = FakePipeline()

    def fake_from_pretrained(model_dir, **kwargs):
        captured["load"] = (model_dir, kwargs)
        return pipeline

    def fake_from_numpy(values):
        captured["latent_values"] = values.copy()
        return FakeLatents()

    class FakeDiffusionPipeline:
        from_pretrained = staticmethod(fake_from_pretrained)

    diffusers = ModuleType("diffusers")
    diffusers.DiffusionPipeline = FakeDiffusionPipeline
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    monkeypatch.setattr(torch, "from_numpy", fake_from_numpy)
    monkeypatch.setattr(torch, "Generator", FakeGenerator)
    initial_latents = np.random.default_rng(9).standard_normal((1, 16, 2, 2), dtype=np.float32)

    _official_reference(
        Path("/model"),
        {"task": "image_generation", "image_height": 16, "image_width": 16},
        {"prompt": "cat", "num_inference_steps": 1, "seed": 9},
        tmp_path,
        initial_latents,
    )

    assert captured["load"] == (
        Path("/model"),
        {"torch_dtype": torch.bfloat16, "local_files_only": True},
    )
    assert captured["pipeline_device"] == "cuda"
    np.testing.assert_array_equal(captured["latent_values"], initial_latents)
    assert captured["latent_to"] == {"device": "cuda", "dtype": torch.bfloat16}
    generation = captured["generation"]
    assert isinstance(generation, dict)
    assert isinstance(generation["latents"], FakeLatents)
    assert generation["callback_on_step_end"] is _require_finite_latents

    finite = {"latents": torch.ones(1)}
    assert _require_finite_latents(None, 0, None, finite) is finite
    with pytest.raises(RuntimeError, match="non-finite latents at step 1"):
        _require_finite_latents(None, 1, None, {"latents": torch.tensor([float("nan")])})


def test_tp_launcher_exports_rendezvous_and_selects_rank_zero(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        stdout = "\n".join(('[1,1]<stdout>:{"rank":1}', '[1,0]<stdout>:{"rank":0}'))
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/mpirun")
    monkeypatch.setattr(subprocess, "run", fake_run)
    bundle = tmp_path / "z-image.bundle"

    payload = _run_json(
        Path("/trtmc"),
        Path("/runtime"),
        bundle,
        {"tensor_parallel_size": 2},
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
        "family": "z_image",
        "frame_count": 1,
        "prompt": "a test image",
        "sample_count": 1,
        "sampled_frame_indices": [0],
        "schema_version": 1,
    }


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    _, manifest, case = CASES[case_name]
    model_dir = _model_dir(manifest)
    binary, runtime_root = _runtime(manifest)
    bundle = tmp_path / manifest["bundle"]
    _build(model_dir, bundle, manifest)
    initial_latents = _initial_latents(manifest, case)
    actual = _native(
        binary, runtime_root, bundle, model_dir, manifest, case, tmp_path, initial_latents
    )
    expected = _official_reference(model_dir, manifest, case, tmp_path, initial_latents)
    _assert_contract(actual, expected, manifest, case, _thresholds(case_name))
