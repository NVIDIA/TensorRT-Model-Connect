# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime, and declared-reference E2E for deepseek_ocr."""

from __future__ import annotations
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
import numpy as np
import pytest
from tensorrt_model_connect import BuildRequest, build

FAMILY = "deepseek_ocr"
TASKS = frozenset({"vision_language_generation"})
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"
THRESHOLD_ROOT = TEST_ROOT / "thresholds"

_REQUIRED_OCR_SUBSTRINGS = {
    "deepseek-ocr-l0": (
        "Architecture:",
        "Attention: Standard Q/K/V/O",
        "RoPE: Standard rotary position embeddings",
        "Layer 0: Dense SwiGLU MLP",
        "Layers 1-11: MoE",
        "Norm: RMS",
    ),
    "deepseek-ocr": (
        "Architecture",
        "Attention:Standard Q/K/V/O",
        "RoPE:Standard rotary position embeddings",
        "Layer 0:Dense SwiGLU MLP",
        "Layers 1-11:MoE",
        "Vision: SAM ViT-B + Qwen2 encoder",
    ),
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
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(
        (value for value in (str(runtime_root), env.get("LD_LIBRARY_PATH", "")) if value)
    )
    if int(manifest["tensor_parallel_size"]) > 1:
        mpirun = shutil.which("mpirun")
        assert mpirun, "selected multi-GPU E2E requires mpirun"
        env["TRTMC_NCCL_RENDEZVOUS"] = str(bundle.with_suffix(".nccl-rendezvous"))
        prefix = [mpirun, "--tag-output", "-np", str(manifest["tensor_parallel_size"])]
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


def _edit_distance(left: str, right: str) -> float:
    a = " ".join(left.lower().split())
    b = " ".join(right.lower().split())
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
    return previous[-1] / max(len(a), len(b), 1)


def _canonical_ocr_text(value: str) -> str:
    normalized = " ".join(value.lower().split())
    return re.sub(r"\s*:\s*", ":", normalized)


def _assert_required_ocr_substrings(
    case_name: str,
    actual: str,
    expected: str,
    required_substrings=(),
) -> None:
    required = tuple(required_substrings) or _REQUIRED_OCR_SUBSTRINGS.get(case_name, ())
    actual_text = _canonical_ocr_text(actual)
    expected_text = _canonical_ocr_text(expected)
    missing_from_reference = [
        value for value in required if _canonical_ocr_text(value) not in expected_text
    ]
    assert not missing_from_reference, f"official OCR reference missing: {missing_from_reference}"
    missing = [value for value in required if _canonical_ocr_text(value) not in actual_text]
    assert not missing, f"native OCR output missing: {missing}"


def _torch_dtype(precision: str):
    import torch

    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
    }[precision]


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
    image = _asset(case["test_image"])
    return _run_json(
        binary,
        runtime_root,
        bundle,
        manifest,
        case,
        "run",
        "--prompt",
        _case_text(case),
        "--image",
        str(image),
        "--max-new-tokens",
        str(int(case["max_new_tokens"])),
        "--temperature",
        "1",
        "--top-k",
        "1",
    )


def _official_reference(model_dir: Path, bundle: Path, manifest: dict, case: dict, tmp_path: Path):
    manifest["task"]
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModel

    image = Image.open(_asset(case["test_image"])).convert("RGB")
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model = (
        AutoModel.from_pretrained(
            model_dir, trust_remote_code=True, torch_dtype=_torch_dtype(case["reference_precision"])
        )
        .to("cuda")
        .eval()
    )
    encoded = processor(text=_case_text(case), images=image, return_tensors="pt")
    encoded = {
        key: value.to("cuda") if hasattr(value, "to") else value for key, value in encoded.items()
    }
    with torch.no_grad():
        generated = model.generate(
            **encoded, max_new_tokens=int(case["max_new_tokens"]), do_sample=False
        )
    input_length = encoded.get("input_ids", torch.empty(1, 0)).shape[-1]
    ids = generated[0]
    if ids.numel() > input_length:
        ids = ids[input_length:]
    return {
        "token_ids": ids.cpu().tolist(),
        "text": processor.decode(ids, skip_special_tokens=True),
    }


def _golden_reference(case: dict) -> dict:
    metadata = case.get("metadata")
    assert isinstance(metadata, dict), f"{case['name']} golden reference requires metadata"
    value = metadata.get("golden_snapshot_path")
    assert isinstance(value, str) and value, (
        f"{case['name']} golden reference requires metadata.golden_snapshot_path"
    )
    path = _asset(value)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict), f"{case['name']} golden reference must be an object: {path}"
    assert isinstance(payload.get("text"), str) and payload["text"].strip(), (
        f"{case['name']} golden reference requires non-empty text: {path}"
    )
    required = payload.get("required_substrings")
    assert (
        isinstance(required, list)
        and required
        and all(isinstance(value, str) and value for value in required)
    ), f"{case['name']} golden reference requires OCR substrings: {path}"
    return payload


def _reference(model_dir: Path, bundle: Path, manifest: dict, case: dict, tmp_path: Path):
    backend = case.get("reference_backend")
    if backend == "golden_snapshot":
        return _golden_reference(case)
    if backend == "hf_transformers":
        return _official_reference(model_dir, bundle, manifest, case, tmp_path)
    raise AssertionError(f"unsupported {FAMILY} reference backend: {backend!r}")


def _assert_parity(actual, expected, manifest: dict, case: dict, thresholds: dict) -> None:
    del manifest
    actual_text = str(actual["text"])
    expected_text = str(expected["text"])
    assert actual_text
    assert _edit_distance(actual_text, expected_text) <= float(
        thresholds.get("contract_ned_threshold", thresholds["normalized_text_edit_distance"])
    )
    _assert_required_ocr_substrings(
        str(case["name"]),
        actual_text,
        expected_text,
        expected.get("required_substrings", ()),
    )


def _assert_native_vision_health(features) -> None:
    values = np.asarray(features)
    assert values.size > 0
    assert np.isfinite(values).all()
    assert np.any(values != 0)


def test_native_vision_health_rejects_invalid_output() -> None:
    _assert_native_vision_health(np.asarray([1.0], dtype=np.float32))
    for invalid in ([], [0.0], [np.nan]):
        with pytest.raises(AssertionError):
            _assert_native_vision_health(invalid)


def test_required_ocr_substrings_preserve_content_but_ignore_colon_spacing() -> None:
    required = "\n".join(_REQUIRED_OCR_SUBSTRINGS["deepseek-ocr-l0"])
    compact = required.replace(": ", ":")
    _assert_required_ocr_substrings("deepseek-ocr-l0", compact, required)
    with pytest.raises(AssertionError, match="native OCR output missing"):
        _assert_required_ocr_substrings(
            "deepseek-ocr-l0", compact.replace("Norm:RMS", "Norm:LayerNorm"), required
        )


def test_reference_routes_restore_the_checked_in_l0_golden() -> None:
    expected_backends = {
        "deepseek-ocr-l0": "golden_snapshot",
        "deepseek-ocr-l0-tp2": "golden_snapshot",
        "deepseek-ocr": "hf_transformers",
    }
    assert {name: case["reference_backend"] for name, (_, _, case) in CASES.items()} == (
        expected_backends
    )
    for name in ("deepseek-ocr-l0", "deepseek-ocr-l0-tp2"):
        _, manifest, case = CASES[name]
        reference = _reference(Path("unused"), Path("unused"), manifest, case, Path("unused"))
        assert reference["text"]
        assert "vision_features" not in reference
        _assert_parity(
            {"text": reference["text"]},
            reference,
            manifest,
            case,
            _thresholds(name),
        )


def test_golden_reference_fails_closed_for_missing_or_invalid_data(tmp_path: Path) -> None:
    case = {
        "name": "broken-golden",
        "metadata": {"golden_snapshot_path": str(tmp_path / "missing.json")},
    }
    with pytest.raises(AssertionError, match="does not exist"):
        _golden_reference(case)

    invalid = tmp_path / "invalid.json"
    invalid.write_text("not JSON", encoding="utf-8")
    case["metadata"]["golden_snapshot_path"] = str(invalid)
    with pytest.raises(json.JSONDecodeError):
        _golden_reference(case)

    invalid.write_text("{}", encoding="utf-8")
    with pytest.raises(AssertionError, match="requires non-empty text"):
        _golden_reference(case)

    invalid.write_text('{"text": "present"}', encoding="utf-8")
    with pytest.raises(AssertionError, match="requires OCR substrings"):
        _golden_reference(case)


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    _, manifest, case = CASES[case_name]
    model_dir = _model_dir(manifest)
    binary, runtime_root = _runtime(manifest)
    bundle = tmp_path / manifest["bundle"]
    _build(model_dir, bundle, manifest)
    actual = _native(binary, runtime_root, bundle, model_dir, manifest, case, tmp_path)
    expected = _reference(model_dir, bundle, manifest, case, tmp_path)
    from PIL import Image

    from families.deepseek_ocr.tests.vision_oracle import native_vision_features

    image = Image.open(_asset(case["test_image"])).convert("RGB")
    _assert_native_vision_health(native_vision_features(bundle, image))
    _assert_parity(actual, expected, manifest, case, _thresholds(case_name))
