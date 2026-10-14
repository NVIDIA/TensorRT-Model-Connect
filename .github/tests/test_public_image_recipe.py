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


def test_verify_rejects_package_apt_or_abi_drift(tmp_path, monkeypatch):
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
    for key in ("venv_packages", "apt_packages", "abi"):
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
if args[0] == "run" and "verify" in args:
    context = pathlib.Path(args[args.index("--volume") + 1].rsplit(":", 2)[0])
    record["verification_files"] = sorted(p.name for p in context.iterdir())
    record["verification_inputs"] = {p.name: p.read_text() for p in context.iterdir()}
with open(os.environ["RECIPE_CALLS"], "a") as stream: stream.write(json.dumps(record) + "\\n")
if args[:2] == ["image", "inspect"]:
    if os.environ.get("RECIPE_INSPECT_FAIL"): sys.exit(1)
    identity = os.environ.get("RECIPE_BASE_ID", "sha256:" + "a" * 64)
    if "{{.Os}}" in args[args.index("--format") + 1]:
        identity += " " + os.environ.get("RECIPE_BASE_OS", "linux")
        identity += " " + os.environ.get("RECIPE_BASE_ARCH", "amd64")
    print(identity)
if args[0] == "run" and "verify" in args and os.environ.get("RECIPE_VERIFY_FAIL"):
    sys.exit(1)
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
            "--base-image",
            "recipe-base:local",
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
    assert builds[0]["args"][builds[0]["args"].index("--tag") + 1] == "recipe-base:local"
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


@pytest.mark.parametrize("bootstrap", [False, True])
def test_existing_base_is_verified_before_only_the_family_layer_is_built(tmp_path, bootstrap):
    env = fake_transport(tmp_path)
    output = tmp_path / "output"
    image_id = "sha256:" + "a" * 64
    args = [
        "bash",
        str(HELPER),
        "--from-base",
        image_id,
        "--max-jobs",
        "2",
        "--output",
        str(output),
    ]
    if bootstrap:
        args.append("--bootstrap")
    subprocess.run(args, env=env, check=True)
    calls = [json.loads(line) for line in (tmp_path / "calls").read_text().splitlines()]
    builds = [call for call in calls if call["args"][0] == "build"]
    assert len(builds) == 1 and "BASE_IMAGE=trtmc-nemotron-h-base:" + "a" * 64 in builds[0]["args"]
    assert str(ROOT / "Dockerfile.dev.x86-gpu") not in builds[0]["args"]
    assert "MAX_JOBS=2" in builds[0]["args"]
    verification = next(call for call in calls if "verification_files" in call)
    assert calls.index(verification) < calls.index(builds[0])
    assert verification["verification_files"] == [
        "environment.json",
        "environment.lock",
        "image-environment.py",
    ]
    assert verification["verification_inputs"] == {
        "environment.json": (ROOT / "requirements/community-gpu-linux-amd64.json").read_text(),
        "environment.lock": (ROOT / "requirements/community-gpu-linux-amd64.lock").read_text(),
        "image-environment.py": RECORDER.read_text(),
    }
    assert "--network" in verification["args"] and "none" in verification["args"]
    assert verification["args"][verification["args"].index("--volume") + 1].endswith(":ro")
    assert (
        verification["args"][verification["args"].index("--entrypoint") + 1]
        == "/opt/venv/bin/python"
    )
    assert image_id in verification["args"]
    assert builds[0]["context_files"] == [
        "Dockerfile",
        "build-inputs.json",
        "constraints-linux-amd64.txt",
        "environment-linux-amd64.json",
        "environment-linux-amd64.lock",
        "requirements.txt",
    ]
    assert not any(
        call["args"][0] in {"login", "pull", "push"} or "--gpus" in call["args"] for call in calls
    )
    record = json.loads((output / "build-inputs.json").read_text())
    assert len(record["inputs"]) == 11
    assert record["native_byok_passed"] is False and record["family_e2e_passed"] is False


@pytest.mark.parametrize(
    "reference",
    [
        "",
        "base:latest",
        "example.invalid/image@sha256:" + "a" * 64,
        "sha256:" + "A" * 64,
        "sha256:abc",
    ],
)
def test_existing_base_refuses_mutable_or_registry_references_before_docker(tmp_path, reference):
    env = fake_transport(tmp_path)
    result = subprocess.run(
        ["bash", str(HELPER), "--from-base", reference, "--output", str(tmp_path / "output")],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "immutable local image ID" in result.stderr
    assert not (tmp_path / "calls").exists()


@pytest.mark.parametrize(
    "setting,value",
    [
        ("RECIPE_BASE_ID", "sha256:" + "b" * 64),
        ("RECIPE_BASE_OS", "windows"),
        ("RECIPE_BASE_ARCH", "arm64"),
        ("RECIPE_INSPECT_FAIL", "1"),
        ("RECIPE_VERIFY_FAIL", "1"),
    ],
)
def test_wrong_or_drifted_existing_base_stops_before_any_family_build(tmp_path, setting, value):
    env = {**fake_transport(tmp_path), setting: value}
    result = subprocess.run(
        [
            "bash",
            str(HELPER),
            "--from-base",
            "sha256:" + "a" * 64,
            "--output",
            str(tmp_path / "output"),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    calls = [json.loads(line) for line in (tmp_path / "calls").read_text().splitlines()]
    assert not any(call["args"][0] in {"build", "tag", "pull", "login", "push"} for call in calls)
