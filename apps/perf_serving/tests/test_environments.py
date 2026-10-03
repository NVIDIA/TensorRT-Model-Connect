# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from pathlib import Path

import pytest

from trtmc_perf_serving import cli, environments


def test_reference_env_without_requirements_is_the_serving_interpreter(capsys, tmp_path):
    assert cli.main(["reference-env", "--root", str(tmp_path)]) == 0
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed == {"python": sys.executable, "requirements": None}


def test_environments_are_keyed_by_requirements_and_reused(tmp_path, monkeypatch):
    requirements = tmp_path / "fam" / "requirements.txt"
    requirements.parent.mkdir()
    requirements.write_text("# nothing to install\n")
    commands = []

    def run(command, log, timeout):
        commands.append(command[:4])
        if command[1:3] == ["-m", "venv"]:
            environment = Path(command[-1])
            (environment / "bin").mkdir(parents=True)
            (environment / "bin/python").write_text("")
            (environment / "lib/python3/site-packages").mkdir(parents=True)

    monkeypatch.setattr(environments, "_run", run)
    monkeypatch.setattr(environments, "_freeze", lambda python: "numpy==1\n")
    first = environments.reference_python(requirements, tmp_path / "envs")
    assert first.parent.parent.name.startswith("fam-") and (first.parent.parent / ".requirements.sha256").is_file()
    pth = first.parent.parent / "lib/python3/site-packages/trtmc-serving-environment.pth"
    assert pth.is_file() and pth.read_text().strip()  # the serving interpreter's packages stay visible
    assert environments.reference_python(requirements, tmp_path / "envs") == first and len(commands) == 2  # reused
    requirements.write_text("numpy\n")
    assert environments.reference_python(requirements, tmp_path / "envs") != first  # new requirements, new environment
    with pytest.raises(RuntimeError, match="not found"):
        environments.reference_python(tmp_path / "missing.txt", tmp_path / "envs")
    current = environments.reference_python(requirements, tmp_path / "envs")
    monkeypatch.setattr(environments, "_freeze", lambda python: "numpy==2\n")  # changed under us: a fresh one
    fresh = environments.reference_python(requirements, tmp_path / "envs")
    assert fresh != current and fresh.parent.parent.name.endswith("-r1") and current.is_file()  # the old one is kept
    assert environments.reference_python(requirements, tmp_path / "envs") == fresh  # and the fresh one reused
    (fresh.parent.parent / ".freeze").unlink()  # no freeze record: not trusted either
    assert environments.reference_python(requirements, tmp_path / "envs").parent.parent.name.endswith("-r2")
    other = tmp_path / "other" / "requirements.txt"
    other.parent.mkdir()
    other.write_text("# a directory left by an unfinished creation\n")
    stale = tmp_path / "envs" / f"other-{environments._digest(other.resolve(), True)[:12]}"
    (stale / "keep.txt").parent.mkdir(parents=True)
    (stale / "keep.txt").write_text("x")
    created = environments.reference_python(other, tmp_path / "envs")
    assert created.parent.parent.name.endswith("-r1") and (stale / "keep.txt").read_text() == "x"  # never overlaid


def test_the_serving_cli_has_no_script_backend():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["serve", "--profile", "p", "--backend", "script", "--records", "r", "--scratch", "s"])


def test_family_adapters_load_from_their_file(tmp_path):
    from trtmc_perf_serving.backends.reference import ReferenceBackend
    from trtmc_perf_serving.backends.reference.common import ReferenceSpec

    adapter = tmp_path / "fam" / "native_reference.py"
    adapter.parent.mkdir()
    adapter.write_text("class Adapter:\n    def __init__(self, spec):\n        self.spec = spec\n")
    backend = ReferenceBackend(ReferenceSpec(operation="world_model", model="m", adapter=str(adapter)))
    assert backend.describe()["adapter"] == "Adapter"
    with pytest.raises(Exception, match="not found"):
        ReferenceBackend(ReferenceSpec(operation="world_model", model="m", adapter=str(tmp_path / "missing.py")))
