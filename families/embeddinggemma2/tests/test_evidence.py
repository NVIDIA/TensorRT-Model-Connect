# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the selected family E2E's integration with the evidence collector."""

import json
import os
from pathlib import Path
import shutil

import pytest

pytest_plugins = ("pytester",)


@pytest.mark.parametrize("explicit_reference", [False, True])
def test_selected_cases_retain_native_and_reference_evidence(
    pytester, monkeypatch, explicit_reference
):
    source = Path(__file__).parent
    target = pytester.path / "families/embeddinggemma2/tests"
    target.mkdir(parents=True)
    shutil.copy(source / "test_e2e.py", target / "test_e2e.py")
    shutil.copytree(source / "manifests", target / "manifests")
    repository = source.parents[2]
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(map(str, (
        repository, repository / "core/builder", repository / "apps/benchmark",
    ))))
    monkeypatch.setenv("TRTMC_E2E_ARTIFACT_DIR", str(pytester.path / "captured"))
    for name in (
        "TRTMC_RUNTIME_ROOT", "TRTMC_NATIVE_BUILD_DIR",
        "TRTMC_EMBEDDINGGEMMA2_MODEL_DIR",
    ):
        monkeypatch.setenv(name, str(pytester.path))
    if explicit_reference:
        interpreter = pytester.path / "reference-python"
        interpreter.touch()
        monkeypatch.setenv("TRTMC_EMBEDDINGGEMMA2_REFERENCE_PYTHON", str(interpreter))
    else:
        monkeypatch.delenv("TRTMC_EMBEDDINGGEMMA2_REFERENCE_PYTHON", raising=False)
    pytester.makeconftest('''
import json
import os
import sys
from types import SimpleNamespace
import pytest

pytest_plugins = ("tools.e2e_evidence",)

def pytest_addoption(parser):
    for name in ("--e2e-model", "--e2e-testcase"):
        parser.addoption(name, action="append", default=[])
    parser.addoption("--e2e-models-file", default=None)

@pytest.fixture(autouse=True)
def native_and_reference_fixture(request, monkeypatch):
    module = request.module
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True)))
    monkeypatch.setattr(module, "build", lambda request: None)
    def run(command, **kwargs):
        vector = [1.0] + [0.0] * 767
        if len(command) > 1 and str(command[1]).endswith("reference.py"):
            assert str(command[0]) == os.environ.get(
                "TRTMC_EMBEDDINGGEMMA2_REFERENCE_PYTHON", sys.executable)
            payload = {"embedding": vector, "token_ids": [1, 2]}
        else:
            payload = vector
        return SimpleNamespace(stdout=json.dumps(payload))
    monkeypatch.setattr(module, "subprocess", SimpleNamespace(run=run))
''')
    result = pytester.runpytest_subprocess(str(target / "test_e2e.py"), "--e2e-model", "embeddinggemma2", "-q")
    result.assert_outcomes(passed=6)
    receipts = sorted((pytester.path / "captured/evidence/embeddinggemma2").glob("*/evidence.json"))
    assert len(receipts) == 6
    for receipt in receipts:
        evidence = json.loads(receipt.read_text(encoding="utf-8"))
        assert evidence["status"] == "passed"
        assert evidence["native"]["interface"] == "text_to_embedding"
        assert evidence["reference"]["hf_revision"] == "914f7f89142e33e77833254d9c9b90c3cef7303b"
        assert evidence["thresholds"]["cosine_similarity"] == 0.99
