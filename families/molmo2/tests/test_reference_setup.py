# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Find the prepared reference interpreter and reject missing setup before build."""

import json
from pathlib import Path
import subprocess
import venv

import pytest

from families.molmo2.tests import test_e2e
from tools.ci.process import CiError


@pytest.fixture
def prepared_reference(tmp_path, monkeypatch):
    directory = tmp_path / "reference/molmo2"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(directory)
    (directory / "ready.json").write_text(
        json.dumps({"requirements": (test_e2e.ROOT / "requirements.txt").read_text()})
    )
    monkeypatch.delenv("TRTMC_MOLMO2_REFERENCE_PYTHON", raising=False)
    monkeypatch.setenv("TRTMC_CI_STATE_DIR", str(tmp_path))
    return directory / "bin/python"


def test_default_reference_uses_prepared_virtual_environment(prepared_reference):
    python = test_e2e._reference_python()
    assert python == prepared_reference
    prefix = subprocess.check_output(
        [str(python), "-c", "import sys; print(sys.prefix)"], text=True
    )
    assert Path(prefix.strip()) == prepared_reference.parent.parent


@pytest.mark.parametrize("value", ["", "missing-python", "directory", "not-executable"])
def test_invalid_reference_override_does_not_use_prepared_default(
    prepared_reference, monkeypatch, value
):
    if value in {"directory", "not-executable"}:
        path = prepared_reference.parent / value
        if value == "directory":
            path.mkdir()
        else:
            path.touch()
        value = str(path)
    monkeypatch.setenv("TRTMC_MOLMO2_REFERENCE_PYTHON", value)
    with pytest.raises(AssertionError, match="TRTMC_MOLMO2_REFERENCE_PYTHON"):
        test_e2e._reference_python()


def test_missing_reference_fails_before_engine_build(tmp_path, tmp_path_factory, monkeypatch):
    (tmp_path / "config.json").write_text("{}")
    binary = tmp_path / "trtmc"
    binary.touch()
    monkeypatch.setenv("TRTMC_MOLMO2_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("TRTMC_RUNTIME_ROOT", str(tmp_path))
    monkeypatch.setenv("TRTMC_BINARY", str(binary))
    monkeypatch.setenv("TRTMC_CI_STATE_DIR", str(tmp_path / "unprepared"))
    for name in (
        "TRTMC_MOLMO2_REFERENCE_PYTHON",
        "TRTMC_MOLMO2_REFERENCE_RESULTS",
        "TRTMC_MOLMO2_BUNDLE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        test_e2e, "build", lambda *_args: pytest.fail("engine build started before reference setup")
    )
    with pytest.raises(CiError, match="not prepared"):
        test_e2e.artifacts.__wrapped__(tmp_path_factory)()
