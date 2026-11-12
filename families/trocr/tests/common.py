# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Case selection, checkpoint and runtime lookup, and the official reference for TrOCR tests."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

FAMILY = "trocr"
TASKS = frozenset({"images_text_to_text"})
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"
# The files the family and the reference read; the repository also carries a
# duplicate pytorch_model.bin that neither needs.
CHECKPOINT_FILES = (
    "config.json", "generation_config.json", "preprocessor_config.json", "model.safetensors",
    "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt",
)


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


def _filters(config, option: str) -> set[str]:
    values = set()
    for raw in config.getoption(option) or []:
        values.update(item.strip() for item in str(raw).split(",") if item.strip())
    return values


def selected_cases(config) -> tuple[list[str], bool]:
    """The cases the E2E selectors name, and whether any selector was given."""
    model_filters = _filters(config, "--e2e-model")
    models_file = config.getoption("--e2e-models-file")
    if models_file:
        model_filters.update(
            line.strip() for line in Path(models_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#"))
    testcase_filters = _filters(config, "--e2e-testcase")
    if not model_filters and not testcase_filters:
        return sorted(CASES), False
    selected = []
    for name, (_, manifest, _) in CASES.items():
        model_match = (not model_filters or FAMILY in model_filters or name in model_filters
                       or manifest["name"] in model_filters)
        if model_match and (not testcase_filters or name in testcase_filters):
            selected.append(name)
    return sorted(selected), True


def parametrize_cases(metafunc, fixture: str, keep=lambda manifest: True) -> None:
    names, enabled = selected_cases(metafunc.config)
    names = [name for name in names if keep(CASES[name][1])]
    parameters = names if enabled else [
        pytest.param(name, marks=pytest.mark.skip(
            reason="direct E2E requires one of the three explicit E2E selectors"))
        for name in names
    ]
    metafunc.parametrize(fixture, parameters, ids=names)


def required_path(value: str | None, label: str) -> Path:
    assert value, f"selected {FAMILY} E2E requires {label}"
    path = Path(value)
    assert path.exists(), f"selected {FAMILY} E2E {label} does not exist: {path}"
    return path


def model_dir(hf_id: str, revision: str | None) -> Path:
    explicit = os.environ.get(f"TRTMC_{FAMILY.upper()}_MODEL_DIR")
    if explicit:
        return required_path(explicit, f"TRTMC_{FAMILY.upper()}_MODEL_DIR")
    from huggingface_hub import snapshot_download

    try:
        snapshot = snapshot_download(repo_id=hf_id, revision=revision,
                                     allow_patterns=list(CHECKPOINT_FILES), local_files_only=True)
    except Exception as error:
        raise AssertionError(
            f"selected {FAMILY} E2E requires the exact cached checkpoint {hf_id}") from error
    return Path(snapshot)


def asset(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = TEST_ROOT / path
    assert path.is_file(), f"selected {FAMILY} E2E asset does not exist: {path}"
    return path


def strict_fp32_torch():
    """torch with TF32 off: the NGC images turn it on, and it is worth about 1e-3."""
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.get_float32_matmul_precision() == "highest"
    assert torch.cuda.is_available(), f"selected {FAMILY} E2E requires CUDA"
    return torch


class Reference:
    """The checkpoint run by Transformers in float32, with its own processor and tokenizer.

    The image processor is the Pillow one (``ViTImageProcessor``): the family
    reproduces Pillow's bilinear filter, and the torchvision processor resizes
    differently.
    """

    def __init__(self, checkpoint: Path) -> None:
        torch = strict_fp32_torch()
        from transformers import AutoTokenizer, ViTImageProcessor, VisionEncoderDecoderModel

        self.torch = torch
        self.model = VisionEncoderDecoderModel.from_pretrained(
            str(checkpoint), dtype=torch.float32).to("cuda").eval()
        self.processor = ViTImageProcessor.from_pretrained(str(checkpoint))
        self.tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))

    def pixel_values(self, image: Path):
        from PIL import Image

        rgb = Image.open(image).convert("RGB")
        return self.processor(images=rgb, return_tensors="pt").pixel_values.to("cuda")

    def generate(self, image: Path, max_new_tokens: int) -> list[int]:
        """Greedy token ids after the decoder start token, the end token included."""
        with self.torch.no_grad():
            sequence = self.model.generate(self.pixel_values(image), max_new_tokens=max_new_tokens,
                                           do_sample=False, num_beams=1)[0].tolist()
        assert sequence[0] == self.model.generation_config.decoder_start_token_id
        return sequence[1:]

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
