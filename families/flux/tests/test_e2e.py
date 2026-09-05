# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime, and official-reference E2E for flux."""

from __future__ import annotations
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
import pytest
import numpy as np
from tensorrt_model_connect import BuildRequest, build

FAMILY = "flux"
TASKS = frozenset({"image_generation", "image_generation_batch"})
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"
THRESHOLD_ROOT = TEST_ROOT / "thresholds"
MPIRUN_TAG = re.compile(r"^\[(?:\d+,)?(?P<rank>\d+)\]<stdout>:")


def _parallel_size(manifest: dict) -> int:
    tensor_parallel_size = int(manifest.get("tensor_parallel_size", 1))
    context_parallel_size = int(manifest.get("context_parallel_size", 1))
    assert tensor_parallel_size == 1 or context_parallel_size == 1
    return max(tensor_parallel_size, context_parallel_size)


def _select_json_payload(stdout: str, parallel_size: int) -> dict:
    payloads = []
    rank0_payloads = []
    for line in stdout.splitlines():
        start = line.find("{")
        if start < 0:
            continue
        try:
            payload = json.loads(line[start:])
        except json.JSONDecodeError:
            continue
        payloads.append(payload)
        if parallel_size == 1:
            rank0_payloads.append(payload)
            continue
        match = MPIRUN_TAG.match(line)
        if match is not None and int(match.group("rank")) == 0:
            rank0_payloads.append(payload)
    assert payloads, f"native runtime returned no JSON: {stdout[-1000:]}"
    assert rank0_payloads, (
        f"native distributed runtime returned no tagged rank 0 JSON: {stdout[-1000:]}"
    )
    assert all((payload == rank0_payloads[0] for payload in rank0_payloads))
    return rank0_payloads[0]


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

    required_gpus = _parallel_size(manifest)
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
            tensor_parallel_size=int(manifest.get("tensor_parallel_size", 1)),
            context_parallel_size=int(manifest.get("context_parallel_size", 1)),
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
    parallel_size = _parallel_size(manifest)
    env = os.environ.copy()
    if parallel_size > 1:
        mpirun = shutil.which("mpirun")
        assert mpirun, "selected multi-GPU E2E requires mpirun"
        env["TRTMC_NCCL_RENDEZVOUS"] = str(bundle.with_suffix(".nccl-rendezvous"))
        invocation = [
            mpirun,
            "--tag-output",
            "-x",
            "LD_LIBRARY_PATH",
            "-x",
            "TRTMC_NCCL_RENDEZVOUS",
            "-np",
            str(parallel_size),
            *invocation,
        ]
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
    return _select_json_payload(completed.stdout, parallel_size)


def test_mpirun_json_selection_uses_only_canonical_rank_zero() -> None:
    stdout = "\n".join(
        (
            '[1,2]<stdout>:{"worker": true}',
            '[1,0]<stdout>:{"output": "image.png", "worker": false}',
            '[1,1]<stdout>:{"worker": true}',
            '[1,3]<stdout>:{"worker": true}',
        )
    )

    assert _select_json_payload(stdout, 4) == {"output": "image.png", "worker": False}


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


def _torch_dtype(precision: str):
    import torch

    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
    }[precision]


def _is_flux2(manifest: dict) -> bool:
    return Path(str(manifest["hf_id"])).name.startswith("FLUX.2")


def _initial_latents(manifest: dict, case: dict) -> np.ndarray:
    height = int(manifest["image_height"])
    width = int(manifest["image_width"])
    if _is_flux2(manifest):
        assert height % 16 == 0 and width % 16 == 0
        shape = (1, 128, height // 16, width // 16)
    else:
        assert height % 8 == 0 and width % 8 == 0
        shape = (1, 16, height // 8, width // 8)
    return np.random.default_rng(int(case["seed"])).standard_normal(shape, dtype=np.float32)


def _native(
    binary: Path,
    runtime_root: Path,
    bundle: Path,
    model_dir: Path,
    manifest: dict,
    case: dict,
    tmp_path: Path,
    initial_latents: np.ndarray | None,
):
    task = manifest["task"]
    if task == "image_generation":
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
        assert initial_latents is not None
        latents_path = tmp_path / "initial-latents.raw"
        np.ascontiguousarray(initial_latents).tofile(latents_path)
        arguments.extend(("--initial-latents-raw", str(latents_path)))
        payload = _run_json(binary, runtime_root, bundle, manifest, case, command, *arguments)
        payload["artifact"] = str(output)
        return payload
    if task == "image_generation_batch":
        inputs = case["inputs"]
        prompts = [str(prompt) for prompt in inputs["batch_prompts"]]
        seeds = [int(seed) for seed in inputs["batch_seeds"]]
        assert len(prompts) == len(seeds) == int(inputs["expected_batch_size"])
        prompts_path = tmp_path / "prompts.txt"
        prompts_path.write_text("\n".join(prompts) + "\n", encoding="utf-8")
        output = tmp_path / "native-images"
        arguments = [
            "--prompts",
            str(prompts_path),
            "--seeds",
            ",".join((str(seed) for seed in seeds)),
            "--output",
            str(output),
            "--height",
            str(int(manifest["image_height"])),
            "--width",
            str(int(manifest["image_width"])),
            "--num-steps",
            str(int(case["num_inference_steps"])),
        ]
        if case.get("negative_prompt"):
            arguments.extend(("--negative-prompt", str(case["negative_prompt"])))
        for key, option in (("guidance_scale", "--guidance-scale"), ("cfg_scale", "--cfg-scale")):
            if key in case:
                arguments.extend((option, str(float(case[key]))))
        payload = _run_json(
            binary, runtime_root, bundle, manifest, case, "generate-image-batch", *arguments
        )
        assert len(payload["outputs"]) == int(inputs["expected_batch_size"])
        assert [Path(item["output"]).name for item in payload["outputs"]] == [
            f"{index}.png" for index in range(int(inputs["expected_batch_size"]))
        ]
        payload["artifact"] = str(output)
        return payload
    raise AssertionError(f"{FAMILY} has no direct native test for task={task}")


def _official_reference(
    model_dir: Path,
    manifest: dict,
    case: dict,
    tmp_path: Path,
    initial_latents: np.ndarray | None,
):
    del tmp_path
    task = manifest["task"]
    import torch
    from diffusers import Flux2Pipeline, FluxPipeline

    dtype = _torch_dtype(case["reference_precision"])
    pipeline_class = Flux2Pipeline if _is_flux2(manifest) else FluxPipeline
    load_options = {"torch_dtype": dtype, "local_files_only": True}
    if _is_flux2(manifest):
        load_options["low_cpu_mem_usage"] = True
    pipeline = pipeline_class.from_pretrained(model_dir, **load_options)
    pipeline.enable_sequential_cpu_offload()
    if task == "image_generation_batch":
        prompts = [str(prompt) for prompt in case["inputs"]["batch_prompts"]]
        generator = [
            torch.Generator(device="cuda").manual_seed(int(seed))
            for seed in case["inputs"]["batch_seeds"]
        ]
    else:
        prompts = _case_text(case)
        assert initial_latents is not None
        unpacked = torch.from_numpy(initial_latents.copy()).to(device="cuda", dtype=dtype)
        if _is_flux2(manifest):
            reference_latents = unpacked
        else:
            reference_latents = pipeline._pack_latents(
                unpacked,
                initial_latents.shape[0],
                initial_latents.shape[1],
                initial_latents.shape[2],
                initial_latents.shape[3],
            )
    kwargs = {
        "prompt": prompts,
        "height": int(manifest["image_height"]),
        "width": int(manifest["image_width"]),
        "num_inference_steps": int(case["num_inference_steps"]),
    }
    if task == "image_generation_batch":
        kwargs["generator"] = generator
    else:
        kwargs["latents"] = reference_latents
    if int(manifest.get("video_num_frames", 1)) > 1:
        kwargs["num_frames"] = int(manifest["video_num_frames"])
    if case.get("negative_prompt"):
        kwargs["negative_prompt"] = case["negative_prompt"]
    if "guidance_scale" in case:
        kwargs["guidance_scale"] = float(case["guidance_scale"])
    elif _is_flux2(manifest):
        kwargs["guidance_scale"] = 3.5
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
                "family": "flux",
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
    expected_count = int(case.get("inputs", {}).get("expected_batch_size", 1))
    if manifest["task"] == "image_generation_batch":
        assert len(actual_images) == len(expected_images) == expected_count
        pairwise_mae = min(
            float(np.mean(np.abs(actual_images[left] - actual_images[right])))
            for left in range(len(actual_images))
            for right in range(left + 1, len(actual_images))
        )
        assert pairwise_mae >= float(thresholds.get("batch_min_pairwise_pixel_mae", 0.001))
    else:
        assert len(actual_images) >= 1

    pixels = np.asarray(actual_images)
    assert float(pixels.mean()) >= float(thresholds.get("min_pixel_mean", 0.15))
    assert float(pixels.mean()) <= float(thresholds.get("max_pixel_mean", 0.85))
    assert float(pixels.std()) >= float(thresholds.get("min_pixel_std", 0.05))
    _write_semantic_artifacts(case, actual_images, expected_images)


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
        "family": "flux",
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
    initial_latents = (
        None if manifest["task"] == "image_generation_batch" else _initial_latents(manifest, case)
    )
    actual = _native(
        binary, runtime_root, bundle, model_dir, manifest, case, tmp_path, initial_latents
    )
    expected = _official_reference(model_dir, manifest, case, tmp_path, initial_latents)
    _assert_contract(actual, expected, manifest, case, _thresholds(case_name))
