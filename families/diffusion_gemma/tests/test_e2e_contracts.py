# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qualification must run the requested pinned model and require the complete answer."""
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import venv

import pytest

from families.diffusion_gemma.tests.test_e2e import (
    MANIFEST, ROOT, _checkpoint, _path, normalized_answer, pytest_generate_tests,
)


@pytest.mark.parametrize("answer", ["Paris", "  PARIS.  ", '"Paris"', "**Paris**", "“Paris”"])
def test_complete_answer_accepts_harmless_formatting(answer):
    assert normalized_answer(answer) == normalized_answer("Paris")


@pytest.mark.parametrize("answer", ["Paris is not the answer; Berlin is.", "Paris, France", "It is Paris", "Paris\nBerlin", "", "Berlin"])
def test_complete_answer_rejects_extra_or_conflicting_text(answer):
    assert normalized_answer(answer) != normalized_answer("Paris")


def test_wrong_local_checkpoint_revision_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("TRTMC_DIFFUSION_GEMMA_MODEL_DIR", str(tmp_path))
    with pytest.raises(AssertionError, match="snapshot directory for revision"):
        _checkpoint(MANIFEST)


def test_local_checkpoint_alias_must_resolve_to_pinned_revision(tmp_path, monkeypatch):
    snapshot = tmp_path / MANIFEST["hf_revision"]
    snapshot.mkdir()
    alias = tmp_path / "checkpoint"
    alias.symlink_to(snapshot, target_is_directory=True)
    monkeypatch.setenv("TRTMC_DIFFUSION_GEMMA_MODEL_DIR", str(alias))
    assert _checkpoint(MANIFEST) == snapshot


def test_reference_python_keeps_its_virtual_environment(tmp_path, monkeypatch):
    environment = tmp_path / "reference"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
    monkeypatch.setenv("TRTMC_DIFFUSION_GEMMA_REFERENCE_PYTHON", str(environment / "bin/python"))
    prefix = subprocess.check_output([str(_path("TRTMC_DIFFUSION_GEMMA_REFERENCE_PYTHON")),
        "-c", "import sys; print(sys.prefix)"], text=True).strip()
    assert Path(prefix) == environment


def test_unknown_explicit_testcase_fails_collection():
    result = subprocess.run([sys.executable, "-m", "pytest", str(ROOT / "test_e2e.py"),
        "--collect-only", "--e2e-testcase", "diffusiongemma-misspelled", "-q"],
        env=os.environ.copy(), capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "No diffusion_gemma E2E testcase matches" in result.stdout + result.stderr


def test_other_family_selection_remains_skippable():
    values = {"--e2e-model": ["another_family"], "--e2e-models-file": None, "--e2e-testcase": []}
    config = SimpleNamespace(args=["families"], getoption=lambda key: values[key])
    selected = []
    metafunc = SimpleNamespace(fixturenames=["case_name"], config=config,
        parametrize=lambda name, cases: selected.extend(cases))
    pytest_generate_tests(metafunc)
    assert selected == []
