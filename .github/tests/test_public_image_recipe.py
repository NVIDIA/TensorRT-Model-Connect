# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise public image input staging and isolated environment lock semantics."""

import importlib.util
import json
import os
import subprocess
import sys
import venv
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RECORDER = ROOT / "requirements/image-environment.py"
HELPER = ROOT / "families/nemotron_h/ci/build-dependencies.sh"
SPEC = importlib.util.spec_from_file_location("image_environment", RECORDER)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_actual_venv_capture_ignores_inherited_system_distributions(tmp_path):
    venv.EnvBuilder(with_pip=False, system_site_packages=True).create(tmp_path / "venv")
    python = tmp_path / "venv/bin/python"
    purelib = Path(
        subprocess.check_output(
            [str(python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True
        ).strip()
    )
    local = purelib / "recipe_local-1.2.3.dist-info"
    local.mkdir()
    (local / "METADATA").write_text("Name: recipe-local\nVersion: 1.2.3\n")
    program = (
        "import importlib.util; s=importlib.util.spec_from_file_location('e', "
        + repr(str(RECORDER))
        + "); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        "import json; print(json.dumps(m.venv_packages()))"
    )
    observed = json.loads(subprocess.check_output([str(python), "-c", program], text=True))
    assert observed == ["recipe-local==1.2.3"]


@pytest.mark.parametrize(
    "body",
    ["# capture required\n", "x>=1\n", "x @ https://private.invalid/x.whl\n", "x==1\nx==2\n"],
)
def test_lock_rejects_empty_ranges_urls_and_conflicting_versions(tmp_path, body):
    lock = tmp_path / "lock"
    lock.write_text(body)
    with pytest.raises(ValueError):
        MODULE.read_lock(lock)


def test_captured_package_name_order_round_trips_with_hyphenated_children(tmp_path):
    lock = tmp_path / "environment.lock"
    captured = ["torch==2.12.0+cu130", "torch-c-dlpack-ext==0.1.5", "torchaudio==2.11.0+cu130"]
    lock.write_text("\n".join(captured) + "\n")
    assert MODULE.read_lock(lock) == captured


def test_actual_captured_public_locks_use_the_recorders_order():
    for lock, receipt in (
        (
            ROOT / "requirements/community-gpu-linux-amd64.lock",
            ROOT / "requirements/community-gpu-linux-amd64.json",
        ),
        (
            ROOT / "families/nemotron_h/ci/environment-linux-amd64.lock",
            ROOT / "families/nemotron_h/ci/environment-linux-amd64.json",
        ),
    ):
        assert MODULE.read_lock(lock) == json.loads(receipt.read_text())["venv_packages"]


def test_verify_rejects_apt_or_abi_drift(tmp_path, monkeypatch):
    lock, receipt = tmp_path / "lock", tmp_path / "receipt"
    lock.write_text("x==1\n")
    observed = {
        "venv_packages": ["x==1"],
        "apt_packages": ["compiler==1"],
        "abi": {"torch": "2.12"},
    }
    receipt.write_text(json.dumps(observed))
    monkeypatch.setattr(MODULE, "environment", lambda snapshot: observed)
    monkeypatch.setattr(
        sys, "argv", [str(RECORDER), "verify", "--lock", str(lock), "--receipt", str(receipt)]
    )
    MODULE.main()
    for key in ("apt_packages", "abi"):
        changed = {**observed, key: []}
        receipt.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="differs"):
            MODULE.main()


def fake_transport(tmp_path):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    (binaries / "uname").write_text("#!/bin/sh\necho x86_64\n")
    (binaries / "uname").chmod(0o755)
    docker = binaries / "docker"
    docker.write_text(
        "#!/usr/bin/env python3\n"
        + """import json, os, pathlib, sys
args = sys.argv[1:]
record = {"args": args}
if args[0] == "build" and any(a.startswith("BASE_IMAGE=") for a in args):
    context = pathlib.Path(args[-1])
    record["context_files"] = sorted(p.name for p in context.iterdir())
    record["requirements"] = (context / "requirements.txt").read_text()
    record["embedded_manifest"] = (context / "build-inputs.json").read_text()
with open(os.environ["RECIPE_CALLS"], "a") as stream: stream.write(json.dumps(record) + "\\n")
if args[:2] == ["image", "inspect"]: print("sha256:" + "a" * 64)
if args[0] == "run" and "capture" in args:
    mount = args[args.index("--volume") + 1].rsplit(":", 1)[0]
    for flag in ["--lock", "--receipt"]:
        target = pathlib.Path(mount) / pathlib.Path(args[args.index(flag) + 1]).name
        target.write_text("x==1\\n" if flag == "--lock" else "{}\\n")
"""
    )
    docker.chmod(0o755)
    return {
        **os.environ,
        "PATH": str(binaries) + ":" + os.environ["PATH"],
        "RECIPE_CALLS": str(tmp_path / "calls"),
    }


def test_public_bootstrap_stages_only_consumed_inputs_and_never_pushes(tmp_path):
    env = fake_transport(tmp_path)
    output = tmp_path / "output"
    subprocess.run(
        [
            "bash",
            str(HELPER),
            "--bootstrap",
            "--max-jobs",
            "2",
            "--output",
            str(output),
            "--image",
            "recipe-test:local",
            "--oci-source",
            "https://github.com/example/image-builder",
        ],
        env=env,
        check=True,
    )
    calls = [json.loads(line) for line in (tmp_path / "calls").read_text().splitlines()]
    builds = [call for call in calls if call["args"][0] == "build"]
    assert len(builds) == 2
    assert all(
        "org.opencontainers.image.source=https://github.com/example/image-builder" in call["args"]
        for call in builds
    )
    assert builds[0]["args"][-1] == str(ROOT / "requirements")
    assert builds[1]["context_files"] == [
        "Dockerfile",
        "build-inputs.json",
        "constraints-linux-amd64.txt",
        "environment-linux-amd64.json",
        "environment-linux-amd64.lock",
        "requirements.txt",
    ]
    assert builds[1]["requirements"] == (ROOT / "families/nemotron_h/requirements.txt").read_text()
    frozen_tag = "trtmc-nemotron-h-base:" + "a" * 64
    assert {"args": ["tag", "sha256:" + "a" * 64, frozen_tag]} in calls
    assert "BASE_IMAGE=" + frozen_tag in builds[1]["args"]
    assert "MAX_JOBS=2" in builds[1]["args"]
    assert not any(
        "--gpus" in c["args"] or c["args"][0] in {"push", "login", "commit"} for c in calls
    )
    record = json.loads((output / "build-inputs.json").read_text())
    assert (output / "build-inputs.json").read_text() == builds[1]["embedded_manifest"]
    assert len(record["inputs"]) == 11 and "build-inputs.json" not in record["inputs"]
    assert (
        "COPY build-inputs.json /opt/trtmc-ci/build-inputs.json"
        in (ROOT / "families/nemotron_h/ci/Dockerfile.dependencies").read_text()
    )
    assert (
        record["mode"] == "bootstrap"
        and not record["native_byok_passed"]
        and not record["family_e2e_passed"]
    )
    assert "requirements/image-environment.py" in record["inputs"]
    assert "families/nemotron_h/ci/constraints-linux-amd64.txt" in record["inputs"]
    assert "https://github.com/example/image-builder" not in json.dumps(record)


def test_locked_default_fails_before_docker_until_full_locks_are_captured(tmp_path):
    env = fake_transport(tmp_path)
    # Exercise an uncaptured checkout explicitly; the shipped checkout now has
    # real locks and must not remain broken for this negative control to pass.
    checkout = tmp_path / "uncaptured"
    helper = checkout / "families/nemotron_h/ci/build-dependencies.sh"
    helper.parent.mkdir(parents=True)
    helper.write_text(HELPER.read_text())
    requirements = checkout / "requirements"
    requirements.mkdir()
    (requirements / "image-environment.py").write_text(RECORDER.read_text())
    (requirements / "community-gpu-linux-amd64.lock").write_text("# capture required\n")
    result = subprocess.run(
        ["bash", str(helper), "--output", str(tmp_path / "output")],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "captured lock" in result.stderr
    assert not (tmp_path / "calls").exists()
