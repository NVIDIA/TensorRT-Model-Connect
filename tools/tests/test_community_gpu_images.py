# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
from types import SimpleNamespace

import pytest

from tools import community_gpu_images as images

BASE = "sha256:" + "a" * 64
CHILD = "sha256:" + "b" * 64
PREFIX = "ghcr.io/example/private-ci"
INPUTS = {name: hashlib.sha256(name.encode()).hexdigest() for name in images.BASE_INPUTS}


def base_manifest():
    return {
        "schema_version": 1,
        "kind": "community-base",
        "platform": "linux/amd64",
        "registry_visibility": "private",
        "image": PREFIX + "/base@sha256:" + "c" * 64,
        "environment_source_sha": "d" * 40,
        "inputs": INPUTS,
        "input_key": images.base_key(INPUTS),
        "cpu_environment_verified": True,
        "local_image_id": BASE,
    }


def image(identity=BASE, layers=None, labels=None):
    return {
        "Id": identity,
        "Architecture": "amd64",
        "Os": "linux",
        "RootFS": {"Layers": layers if layers is not None else ["shared-layer"]},
        "Config": {"Labels": labels or {}},
    }


def repository(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    for family in ("first", "second"):
        owner = root / "families" / family
        owner.mkdir(parents=True)
        (owner / "requirements.txt").write_text(f"{family}-dependency==1.0\n")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    return root


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("kind", "family"),
        ("platform", "linux/arm64"),
        ("registry_visibility", "public"),
        ("cpu_environment_verified", False),
        ("image", PREFIX + "/base:latest"),
        ("input_key", "0" * 64),
        ("environment_source_sha", "main"),
        ("local_image_id", "latest"),
    ],
)
def test_rejects_unverified_or_mutable_base(field, value):
    base = copy.deepcopy(base_manifest())
    base[field] = value
    with pytest.raises(images.ImagePreparationError):
        images.validate_base(base, PREFIX, INPUTS)


def test_base_identity_requires_the_exact_trusted_shared_inputs():
    base = base_manifest()
    assert images.validate_base(base, PREFIX, INPUTS) is base
    changed = {**INPUTS, "requirements/community-ci.txt": "0" * 64}
    with pytest.raises(images.ImagePreparationError, match="trusted CI"):
        images.validate_base(base, PREFIX, changed)
    with pytest.raises(images.ImagePreparationError):
        images.base_key({**INPUTS, "families/first/requirements.txt": "f" * 64})


def test_family_build_uses_one_frozen_shared_base_and_only_selected_declarations(
    tmp_path, monkeypatch
):
    root = repository(tmp_path)
    calls, contexts = [], []
    prepared = image(CHILD, ["shared-layer", "family-layer"])
    base = image()
    base["Config"]["Entrypoint"] = prepared["Config"]["Entrypoint"] = ["/vendor-init"]

    def inspect(reference, **_):
        return base if reference == BASE else prepared

    def docker(args, **kwargs):
        calls.append((args, kwargs))
        if args[0] == "build":
            context = Path(args[-1])
            contexts.append(
                {
                    str(p.relative_to(context)): p.read_text()
                    for p in context.rglob("*")
                    if p.is_file()
                }
            )
            label = args[args.index("--label") + 1]
            name, value = label.split("=", 1)
            prepared["Config"]["Labels"][name] = value
        return ""

    monkeypatch.setattr(images, "inspect", inspect)
    monkeypatch.setattr(images, "docker", docker)
    assert images.ensure_family_image(root, "first", BASE, time.monotonic() + 3600) == CHILD
    builds = [args for args, _ in calls if args[0] == "build"]
    assert len(builds) == 1
    assert "BASE_IMAGE=trtmc-community-base:" + "a" * 64 in builds[0]
    assert contexts[0]["requirements.txt"] == "first-dependency==1.0\n"
    assert "second-dependency" not in json.dumps(contexts)
    assert all("login" not in args and "pull" not in args for args, _ in calls)
    assert "--no-build-isolation" in contexts[0]["Dockerfile.dependencies"]
    assert calls[-1][0] == [
        "run",
        "--rm",
        "--network",
        "none",
        "--entrypoint",
        "/opt/venv/bin/python",
        CHILD,
        "-m",
        "pip",
        "check",
    ]


def test_owner_without_extra_dependencies_uses_shared_image_directly(tmp_path, monkeypatch):
    owner = tmp_path / "families" / "plain"
    owner.mkdir(parents=True)
    monkeypatch.setattr(images, "inspect", lambda *_, **__: image())
    monkeypatch.setattr(images, "docker", lambda *_, **__: pytest.fail("unnecessary build"))
    assert images.ensure_family_image(tmp_path, "plain", BASE, time.monotonic() + 60) == BASE


@pytest.mark.parametrize("parent,child", [([], []), (["shared"], ["other"]), (["shared"], [])])
def test_family_recipe_cannot_replace_the_shared_base(tmp_path, monkeypatch, parent, child):
    root = repository(tmp_path)
    prepared = image(CHILD, child)
    monkeypatch.setattr(
        images,
        "inspect",
        lambda reference, **_: image(BASE, parent) if reference == BASE else prepared,
    )

    def docker(args, **kwargs):
        if args[0] == "build":
            name, value = args[args.index("--label") + 1].split("=", 1)
            prepared["Config"]["Labels"][name] = value
        assert args[0] != "run"
        return ""

    monkeypatch.setattr(images, "docker", docker)
    with pytest.raises(images.ImagePreparationError, match="shared base layers"):
        images.ensure_family_image(root, "first", BASE, time.monotonic() + 60)


def test_generated_recipe_is_part_of_preparation_identity(tmp_path, monkeypatch):
    root = repository(tmp_path)
    prepared = image(CHILD)
    keys = []
    monkeypatch.setattr(
        images, "inspect", lambda reference, **_: image() if reference == BASE else prepared
    )

    def docker(args, **kwargs):
        if args[0] == "build":
            receipt = json.loads((Path(args[-1]) / "build-inputs.json").read_text())
            assert "Dockerfile.dependencies" in receipt["inputs"]
            keys.append(receipt["input_key"])
            name, value = args[args.index("--label") + 1].split("=", 1)
            prepared["Config"]["Labels"][name] = value
        return ""

    monkeypatch.setattr(images, "docker", docker)
    original = images._default_recipe
    images.ensure_family_image(root, "first", BASE, time.monotonic() + 60)
    monkeypatch.setattr(images, "_default_recipe", lambda files: original(files) + b"RUN true\n")
    images.ensure_family_image(root, "first", BASE, time.monotonic() + 60)
    assert keys[0] != keys[1]


def test_family_dependency_layer_cannot_intercept_the_test_command(tmp_path, monkeypatch):
    root = repository(tmp_path)
    base = image()
    base["Config"]["Entrypoint"] = ["/opt/nvidia/nvidia_entrypoint.sh"]
    prepared = image(CHILD)
    prepared["Config"]["Entrypoint"] = ["python"]
    monkeypatch.setattr(
        images, "inspect", lambda reference, **_: base if reference == BASE else prepared
    )

    def docker(args, **kwargs):
        if args[0] == "build":
            name, value = args[args.index("--label") + 1].split("=", 1)
            prepared["Config"]["Labels"][name] = value
        assert args[0] != "run"
        return ""

    monkeypatch.setattr(images, "docker", docker)
    with pytest.raises(images.ImagePreparationError, match="cannot replace the base entrypoint"):
        images.ensure_family_image(root, "first", BASE, time.monotonic() + 60)


def test_incomplete_dependency_declaration_fails_before_docker(tmp_path, monkeypatch):
    ci = tmp_path / "families" / "first" / "ci"
    ci.mkdir(parents=True)
    (ci / "constraints-linux-amd64.txt").write_text("example==1.0\n")
    monkeypatch.setattr(images, "inspect", lambda *_, **__: pytest.fail("image inspection"))
    with pytest.raises(images.ImagePreparationError, match="requirements file or recipe"):
        images.ensure_family_image(tmp_path, "first", BASE, time.monotonic() + 60)


def test_family_parent_symlink_is_rejected_even_without_requirements(tmp_path):
    external = tmp_path / "external" / "first"
    external.mkdir(parents=True)
    root = tmp_path / "repo"
    root.mkdir()
    (root / "families").symlink_to(external.parent, target_is_directory=True)
    with pytest.raises(images.ImagePreparationError, match="owner is unavailable"):
        images.ensure_family_image(root, "first", BASE, time.monotonic() + 60)


def test_undeclared_owner_and_context_symlink_cannot_read_host_files(tmp_path):
    root = repository(tmp_path)
    owner = root / "families" / "first"
    (owner / "requirements.txt").unlink()
    secret = tmp_path / "host-secret"
    secret.write_text("must not enter context")
    (owner / "requirements.txt").symlink_to(secret)
    with pytest.raises(images.ImagePreparationError):
        images._family_inputs(root, "first")
    with pytest.raises(images.ImagePreparationError):
        images._family_inputs(root, "../first")


def test_pull_erases_auth_before_any_image_execution(tmp_path, monkeypatch):
    base = base_manifest()
    token = tmp_path / "token"
    token.write_text("read-secret")
    events, auth_paths = [], []
    labels = {
        images.BASE_KEY_LABEL: base["input_key"],
        images.BASE_KIND_LABEL: "community-base",
        images.BASE_INPUTS_LABEL: json.dumps(INPUTS),
    }
    pulled = image(labels=labels)
    pulled["RepoDigests"] = [base["image"]]
    monkeypatch.setattr(
        images, "require_private_package", lambda *args: events.append("private-check")
    )
    monkeypatch.setattr(images, "inspect", lambda *args, **kwargs: pulled)

    def docker(args, **kwargs):
        events.append(args[0])
        if kwargs.get("config"):
            auth_paths.append(Path(kwargs["config"]))
            assert auth_paths[-1].exists()
        if args[0] == "login":
            assert kwargs["stdin"] == "read-secret" and kwargs["private"] is True
        if args[0] == "tag":
            assert all(not path.exists() for path in auth_paths)
        assert args[0] not in {"run", "build"}
        return ""

    monkeypatch.setattr(images, "docker", docker)
    assert (
        images.pull_shared_base(
            {"base": base, "registry_prefix": PREFIX}, token, "reader", "local-base"
        )
        == BASE
    )
    assert events == ["private-check", "login", "pull", "tag"]
    assert not token.exists()


def test_auth_failure_removes_token_and_does_not_execute_image(tmp_path, monkeypatch):
    token = tmp_path / "token"
    token.write_text("read-secret")

    def reject(*_):
        raise images.ImagePreparationError("access denied")

    monkeypatch.setattr(images, "require_private_package", reject)
    monkeypatch.setattr(images, "docker", lambda *_, **__: pytest.fail("image execution"))
    with pytest.raises(images.ImagePreparationError, match="access denied"):
        images.pull_shared_base(
            {"base": base_manifest(), "registry_prefix": PREFIX}, token, "reader", "local-base"
        )
    assert not token.exists()


def _reader_response(monkeypatch, raw, *, status=200, error=None):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, amount):
            assert amount == images.MAX_FILE + 1
            return raw[:amount]

    def open_request(request, *, timeout):
        assert request.full_url == "https://api.github.com/user"
        assert request.get_header("Authorization") == "Bearer dedicated-test-reader-token"
        assert timeout == 30
        if error is not None:
            raise error
        response = Response()
        response.status = status
        return response

    def opener(handler):
        assert isinstance(handler, images.NoRedirect)
        assert handler.redirect_request(None, None, 302, "", {}, "https://other.invalid/") is None
        return SimpleNamespace(open=open_request)

    monkeypatch.setattr(images.urllib.request, "build_opener", opener)


def test_registry_reader_username_is_derived_from_token_without_manual_config(monkeypatch, capsys):
    monkeypatch.delenv("TRTMC_COMMUNITY_REGISTRY_USERNAME", raising=False)
    monkeypatch.delenv("REGISTRY_USERNAME", raising=False)
    _reader_response(monkeypatch, b'{"login":"reader-account","other":"must-not-log"}')
    assert images.registry_reader_login("dedicated-test-reader-token\n") == "reader-account"
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize(
    "raw,status",
    [
        (b"{}", 401),
        (b"{}", 403),
        (b"{}", 302),
        (b"{}", 500),
        (b"not JSON", 200),
        (b"[]", 200),
        (b'{"login":null}', 200),
        (b'{"login":"reader\\nmalformed"}', 200),
        (b'{"login":"reader\'; command"}', 200),
        (b'{"login":"one","login":"two"}', 200),
        (b"x" * (images.MAX_FILE + 1), 200),
    ],
)
def test_reader_identity_errors_fail_without_echoing_response(monkeypatch, capsys, raw, status):
    _reader_response(monkeypatch, raw, status=status)
    with pytest.raises(images.ImagePreparationError) as error:
        images.registry_reader_login("dedicated-test-reader-token")
    assert "dedicated-test-reader-token" not in str(error.value)
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize(
    "error",
    [
        TimeoutError("dedicated-test-reader-token"),
        OSError("dedicated-test-reader-token"),
        urllib.error.HTTPError(
            "https://api.github.com/user", 403, "dedicated-test-reader-token", {}, None
        ),
    ],
)
def test_reader_identity_transport_errors_are_redacted(monkeypatch, error):
    _reader_response(monkeypatch, b"", error=error)
    with pytest.raises(images.ImagePreparationError, match="could not be identified") as caught:
        images.registry_reader_login("dedicated-test-reader-token")
    assert "dedicated-test-reader-token" not in str(caught.value)


@pytest.mark.parametrize("token", [None, "", " \n", "contains whitespace"])
def test_missing_or_invalid_reader_token_never_sends_an_auth_request(monkeypatch, token):
    monkeypatch.setattr(
        images.urllib.request, "build_opener", lambda *_: pytest.fail("HTTP request")
    )
    with pytest.raises(images.ImagePreparationError, match="missing or invalid"):
        images.registry_reader_login(token)


def test_real_process_output_is_bounded_during_read(monkeypatch):
    monkeypatch.setattr(images, "MAX_FILE", 1024)
    with pytest.raises(images.ImagePreparationError, match="size limit"):
        images._captured(
            [sys.executable, "-c", "import os; os.write(1,b'x'*5000)"],
            dict(os.environ),
            None,
            2,
            True,
        )


def test_real_process_timeout_does_not_wait_forever():
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        images._captured(
            [sys.executable, "-c", "import time; time.sleep(30)"], dict(os.environ), None, 0.1, True
        )
    assert time.monotonic() - started < 2


def test_docker_receives_no_host_credentials(tmp_path, monkeypatch):
    captured = {}
    for key in ("GH_TOKEN", "GITHUB_TOKEN", "HF_TOKEN", "BREV_API_KEY", "DOCKER_AUTH_CONFIG"):
        monkeypatch.setenv(key, "secret")

    def run(command, **kwargs):
        captured.update(kwargs)
        assert Path(kwargs["env"]["DOCKER_CONFIG"]).is_dir()
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(images.subprocess, "run", run)
    images.docker(["build", str(tmp_path)])
    assert not any(
        key.endswith("TOKEN") or "AUTH" in key or key == "BREV_API_KEY" for key in captured["env"]
    )
