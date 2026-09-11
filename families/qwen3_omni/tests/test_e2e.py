# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native text Task API, and official-reference E2E for Qwen3-Omni."""

from __future__ import annotations

from tools.e2e_evidence import evidence_stage, record_evidence

import json
import os
import subprocess
from pathlib import Path

import pytest

from tensorrt_model_connect import BuildRequest, build


FAMILY = "qwen3_omni"
TASKS = frozenset({"text_generation"})
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"
SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)
TEXT_CHAT_TEMPLATE = """{%- for message in messages %}
{{- '<|im_start|>' + message.role + '\n' }}
{%- if message.content is string %}
{{- message.content }}
{%- else %}
{%- for item in message.content %}
{%- if item.type == 'text' %}{{- item.text }}{%- endif %}
{%- endfor %}
{%- endif %}
{{- '<|im_end|>\n' }}
{%- endfor %}
{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}{%- endif %}"""


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
    if "case_name" not in metafunc.fixturenames:
        return
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


def _required_path(raw: str | None, label: str) -> Path:
    assert raw, f"selected {FAMILY} E2E requires {label}"
    path = Path(raw)
    assert path.exists(), f"selected {FAMILY} E2E {label} does not exist: {path}"
    return path


def _model_dir(manifest: dict) -> Path:
    explicit = os.environ.get("TRTMC_QWEN3_OMNI_MODEL_DIR")
    if explicit:
        return _required_path(explicit, "TRTMC_QWEN3_OMNI_MODEL_DIR")
    from huggingface_hub import snapshot_download

    try:
        return Path(
            snapshot_download(
                repo_id=manifest["hf_id"],
                revision=manifest["hf_revision"],
                local_files_only=True,
            )
        )
    except Exception as error:
        raise AssertionError(
            f"selected {FAMILY} E2E requires exact cached checkpoint {manifest['hf_id']}"
        ) from error


def _runtime_paths() -> tuple[Path, Path]:
    binary = _required_path(os.environ.get("TRTMC_BINARY"), "TRTMC_BINARY")
    runtime_root = _required_path(os.environ.get("TRTMC_RUNTIME_ROOT"), "TRTMC_RUNTIME_ROOT")
    assert (runtime_root / "libtrtmc_backend_trt.so").is_file()
    assert (runtime_root / "libtrtmc_model_qwen3_omni.so").is_file()
    import torch

    assert torch.cuda.is_available(), f"selected {FAMILY} E2E requires CUDA"
    return binary, runtime_root


def _build(model_dir: Path, bundle: Path, manifest: dict) -> None:
    build(
        BuildRequest(
            model_dir=model_dir,
            output_path=bundle,
            family=FAMILY,
            task=manifest["task"],
            precision=manifest["precision"],
            max_sequence_length=int(manifest["max_sequence_length"]),
            tensor_parallel_size=int(manifest["tensor_parallel_size"]),
        )
    )


def _native_text(binary: Path, runtime_root: Path, bundle: Path, case: dict) -> dict:
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = ":".join(
        value for value in (str(runtime_root), environment.get("LD_LIBRARY_PATH", "")) if value
    )
    completed = subprocess.run(
        [
            str(binary),
            "run",
            str(bundle),
            "--runtime-root",
            str(runtime_root),
            "--prompt",
            str(case["prompt"]),
            "--max-new-tokens",
            str(int(case["max_new_tokens"])),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
        timeout=int(case["runtime_timeout_s"]),
    )
    record_evidence("commands", {"argv": getattr(completed, "args", None)})
    record_evidence("native", {"stdout": getattr(completed, "stdout", None), "stderr": getattr(completed, "stderr", None)})
    return json.loads(completed.stdout)


def _official_reference(model_dir: Path, manifest: dict, case: dict) -> str:
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

    processor = Qwen3OmniMoeProcessor.from_pretrained(
        model_dir, revision=manifest["hf_revision"], local_files_only=True
    )
    model = (
        Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_dir,
            revision=manifest["hf_revision"],
            local_files_only=True,
            dtype=torch.bfloat16,
            enable_audio_output=False,
        )
        .to("cuda")
        .eval()
    )
    conversation = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                }
            ],
        },
        {"role": "user", "content": [{"type": "text", "text": str(case["prompt"])}]},
    ]
    inputs = processor.apply_chat_template(
        conversation,
        chat_template=TEXT_CHAT_TEMPLATE,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    ).to(model.device)
    # Keep the correctness oracle independent of fused SDPA backend selection.
    # The math backend retains FP32 attention intermediates for BF16 inputs.
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        text_ids = model.generate(
            **inputs,
            thinker_max_new_tokens=int(case["max_new_tokens"]),
            thinker_do_sample=False,
            return_audio=False,
        )
    generated = text_ids[:, inputs["input_ids"].shape[1] :]
    text = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
    assert text
    return text


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    _, manifest, case = CASES[case_name]
    record_evidence("inputs", {"manifest": manifest, "case": CASES[case_name][-1]})
    record_evidence("thresholds", {"reference_equals_expected": True, "native_equals_reference": True})
    model_dir = _model_dir(manifest)
    record_evidence("checkpoint", {"model_dir": str(model_dir), "hf_id": manifest.get("hf_id"), "hf_revision": manifest.get("hf_revision")})
    binary, runtime_root = _runtime_paths()
    bundle = tmp_path / manifest["bundle"]
    with evidence_stage("build"):
        _build(model_dir, bundle, manifest)
    with evidence_stage("native"):
        native_text = _native_text(binary, runtime_root, bundle, case)
    record_evidence("native", native_text)
    with evidence_stage("reference"):
        reference_text = _official_reference(model_dir, manifest, case)
    record_evidence("reference", {"text": reference_text})
    expected_text = str(case["expected_continuation_text"]).strip()
    record_evidence("inputs", {"expected_text": expected_text})
    with evidence_stage("compare"):
        assert reference_text == expected_text
    with evidence_stage("compare"):
        assert native_text["text"] == reference_text
