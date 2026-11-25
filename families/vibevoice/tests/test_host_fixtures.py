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

    import numpy as np

    for case in _committed()["schedulers"]:
        if case["precision"] == "bf16" and not torch.cuda.is_available():
            continue  # generated where the reference samples (CUDA); see host_fixtures.py
        seed = {(10, "fp32"): 0, (10, "bf16"): 1, (7, "fp32"): 2, (25, "bf16"): 3}[
            (case["steps"], case["precision"])
        ]
        fresh = host_fixtures._scheduler_case(case["steps"], case["precision"], seed)
        for key in ("steps", "precision", "initial", "velocities", "timesteps"):
            assert fresh[key] == case[key], key
        # torch's float32 arithmetic differs between builds: torch 2.12 against the 2.15
        # that wrote the fixtures moves the float32 outputs by up to 4.8e-7 (both CPU,
        # 136 values). The bound is the one tests/cpp/test_host.cpp holds the native
        # solver to (relative to max(1, |value|)); a flipped bfloat16 rounding moves one
        # value by one bf16 step. A NaN fails the comparison.
        fresh_outputs = np.asarray(fresh["outputs"], dtype=np.float64)
        outputs = np.asarray(case["outputs"], dtype=np.float64)
        assert fresh_outputs.shape == outputs.shape
        worst = float(np.max(np.abs(fresh_outputs - outputs) / np.maximum(1.0, np.abs(outputs))))
        assert worst <= (8e-3 if case["precision"] == "bf16" else 1e-5), worst


def test_prompts_match_the_reference_template(tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("tensorrt")
    tokenizer_dir = _cached_tokenizer()
    from families.vibevoice.tests import host_fixtures

    fresh = host_fixtures.script_fixtures(tokenizer_dir)
    committed = _committed()
    for name, entry in committed["scripts"].items():
        assert fresh["scripts"][name]["prompt"] == entry["prompt"]
        assert fresh["scripts"][name]["ids"] == entry["ids"]
    assert fresh["tokenizer_probes"] == committed["tokenizer_probes"]
    assert set(committed["tokenizer_probes"]) == set(host_fixtures.TOKENIZER_PROBES)


def test_script_fixtures_need_no_scheduler(monkeypatch):
    """The prompt cases are rebuilt on CPU hosts too: they must not touch the
    scheduler cases, whose bfloat16 half is generated on CUDA."""
    pytest.importorskip("tensorrt")
    from families.vibevoice.tests import host_fixtures

    def no_scheduler(*args, **kwargs):
        raise AssertionError("script fixtures built a scheduler case")

    monkeypatch.setattr(host_fixtures, "_scheduler_case", no_scheduler)
    scripts = host_fixtures.script_fixtures(None)["scripts"]
    assert set(scripts) == set(host_fixtures.SCRIPTS)


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
