# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime, and official-reference E2E for qwen_vl."""

from __future__ import annotations
import json
import os
import re
import shutil
import string
import subprocess
from pathlib import Path
import numpy as np
import pytest
from tensorrt_model_connect import BuildRequest, build

FAMILY = "qwen_vl"
TASKS = frozenset({"vision_language_generation"})
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


def _canonical_vl_answers(left: str, right: str) -> tuple[str, str]:
    def normalize(value: str) -> str:
        return " ".join(value.casefold().split()).strip(string.punctuation + string.whitespace)

    left = normalize(left)
    right = normalize(right)
    left_words = re.findall(r"\b\w+\b", left)
    right_words = re.findall(r"\b\w+\b", right)
    if len(left_words) == 1 and len(right_words) > 1 and left_words[0] in right_words:
        return left_words[0], left_words[0]
    if len(right_words) == 1 and len(left_words) > 1 and right_words[0] in left_words:
        return right_words[0], right_words[0]
    return left, right


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


def _official_reference(model_dir: Path, manifest: dict, case: dict, tmp_path: Path):
    manifest["task"]
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText

    image_path = _asset(case["test_image"])
    image = Image.open(image_path).convert("RGB")
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    prompt = processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": _case_text(case)},
                ],
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(prompt, str) or not prompt.strip():
        raise RuntimeError("Qwen-VL processor chat template produced an empty prompt")
    if "<|image_pad|>" not in prompt:
        raise RuntimeError("Qwen-VL processor chat template produced no image placeholder")
    model = (
        AutoModelForImageTextToText.from_pretrained(
            model_dir, trust_remote_code=True, torch_dtype=_torch_dtype(case["reference_precision"])
        )
        .to("cuda")
        .eval()
    )
    encoded = processor(text=prompt, images=image, return_tensors="pt")
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


def _assert_parity(actual, expected, manifest: dict, case: dict, thresholds: dict) -> None:
    del manifest, case
    actual_text = str(actual["text"])
    expected_text = str(expected["text"])
    assert actual_text
    canonical_actual, canonical_expected = _canonical_vl_answers(actual_text, expected_text)
    assert _edit_distance(canonical_actual, canonical_expected) <= float(
        thresholds.get("contract_ned_threshold", 0.15)
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


def test_canonical_vl_contract_aligns_an_embedded_single_word_answer() -> None:
    assert _canonical_vl_answers("Paris.", "The answer is Paris") == ("paris", "paris")


def test_official_reference_uses_processor_multimodal_chat_template(
    monkeypatch, tmp_path: Path
) -> None:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    calls = {}

    class InputIds:
        shape = (1, 2)

        def to(self, device):
            calls["input_device"] = device
            return self

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            calls["messages"] = messages
            calls["chat_kwargs"] = kwargs
            return "<|vision_start|><|image_pad|><|vision_end|>Describe.<|im_start|>assistant"

        def __call__(self, **kwargs):
            calls["processor_kwargs"] = kwargs
            return {"input_ids": InputIds()}

        def decode(self, ids, **kwargs):
            calls["decode_kwargs"] = kwargs
            return "answer"

    class Model:
        def to(self, device):
            calls["model_device"] = device
            return self

        def eval(self):
            return self

        def generate(self, **kwargs):
            calls["generate_kwargs"] = kwargs
            return torch.tensor([[10, 11, 12]])

    processor = Processor()
    monkeypatch.setattr(AutoProcessor, "from_pretrained", lambda *args, **kwargs: processor)
    monkeypatch.setattr(
        AutoModelForImageTextToText, "from_pretrained", lambda *args, **kwargs: Model()
    )
    case = {
        "test_image": "data/test_img.jpeg",
        "prompt": "Describe.",
        "max_new_tokens": 3,
        "reference_precision": "fp32",
    }

    result = _official_reference(
        tmp_path,
        {"task": "vision_language_generation", "image_height": 448},
        case,
        tmp_path,
    )

    assert calls["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(TEST_ROOT / "data/test_img.jpeg")},
                {"type": "text", "text": "Describe."},
            ],
        }
    ]
    assert calls["chat_kwargs"] == {"tokenize": False, "add_generation_prompt": True}
    assert calls["processor_kwargs"]["text"].startswith("<|vision_start|><|image_pad|>")
    assert result == {"token_ids": [12], "text": "answer"}


@pytest.mark.parametrize("rendered", ["", "Describe this image."])
def test_official_reference_rejects_invalid_processor_prompt(
    monkeypatch, tmp_path: Path, rendered: str
) -> None:
    from transformers import AutoProcessor

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            return rendered

    monkeypatch.setattr(AutoProcessor, "from_pretrained", lambda *args, **kwargs: Processor())
    expected = "empty prompt" if not rendered else "no image placeholder"
    with pytest.raises(RuntimeError, match=expected):
        _official_reference(
            tmp_path,
            {"task": "vision_language_generation"},
            {
                "test_image": "data/test_img.jpeg",
                "prompt": "Describe this image.",
                "max_new_tokens": 3,
                "reference_precision": "fp32",
            },
            tmp_path,
        )


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    _, manifest, case = CASES[case_name]
    model_dir = _model_dir(manifest)
    binary, runtime_root = _runtime(manifest)
    bundle = tmp_path / manifest["bundle"]
    _build(model_dir, bundle, manifest)
    actual = _native(binary, runtime_root, bundle, model_dir, manifest, case, tmp_path)
    from families.qwen_vl.tests.vision_oracle import native_vision_features

    _assert_native_vision_health(native_vision_features(bundle, _asset(case["test_image"])))
    expected = _official_reference(model_dir, manifest, case, tmp_path)
    _assert_parity(actual, expected, manifest, case, _thresholds(case_name))
