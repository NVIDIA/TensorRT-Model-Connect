# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The committed host fixtures are what their reference sources produce today."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parent / "data" / "host_fixtures.json"


def _committed() -> dict:
    return json.loads(DATA.read_text(encoding="utf-8"))


def test_noise_timesteps_and_scripts_are_current():
    pytest.importorskip("tensorrt")
    from families.vibevoice.tests import host_fixtures, reference

    committed = _committed()
    stream = reference.NoiseStream(committed["noise"]["seed"])
    assert [float(v) for v in stream.normals(len(committed["noise"]["values"]))] == committed[
        "noise"
    ]["values"]
    for steps, values in committed["timesteps"].items():
        assert host_fixtures._linspace(int(steps)).tolist() == values
    for name, entry in committed["scripts"].items():
        assert entry["text"] == host_fixtures.SCRIPTS[name]
        assert host_fixtures._turns(entry["text"]) == entry["turns"]


def test_train_sigmas_are_the_builder_schedule():
    pytest.importorskip("tensorrt")
    from families.vibevoice.model import train_sigmas

    assert [float(v) for v in train_sigmas(1000)] == _committed()["train_sigmas"]


def test_scheduler_cases_match_diffusers():
    torch = pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    pytest.importorskip("tensorrt")
    from families.vibevoice.tests import host_fixtures

    for case in _committed()["schedulers"]:
        if case["precision"] == "bf16" and not torch.cuda.is_available():
            continue  # generated where the reference samples (CUDA); see host_fixtures.py
        seed = {(10, "fp32"): 0, (10, "bf16"): 1, (7, "fp32"): 2, (25, "bf16"): 3}[
            (case["steps"], case["precision"])
        ]
        assert host_fixtures._scheduler_case(case["steps"], case["precision"], seed) == case


def test_prompts_match_the_reference_template(tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("tensorrt")
    tokenizer_dir = _cached_tokenizer()
    from families.vibevoice.tests import host_fixtures

    fresh = host_fixtures.build_host_fixtures(tokenizer_dir)["scripts"]
    for name, entry in _committed()["scripts"].items():
        assert fresh[name]["prompt"] == entry["prompt"]
        assert fresh[name]["ids"] == entry["ids"]


def _cached_tokenizer() -> Path:
    try:
        from huggingface_hub import snapshot_download

        from families.vibevoice import config

        return Path(
            snapshot_download(
                config.TOKENIZER_REPOSITORY,
                revision=config.TOKENIZER_REVISION,
                allow_patterns=list(config.TOKENIZER_FILES),
                local_files_only=True,
            )
        )
    except Exception:  # noqa: BLE001 - the tokenizer is an optional cached input here
        pytest.skip("the Qwen2.5 tokenizer snapshot is not cached")
