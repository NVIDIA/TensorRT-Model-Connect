# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for keeping trusted-container credentials out of Docker argv."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tools.ci.__main__ import CiCommand
from tools.ci.container import CiContainer
from tools.ci.process import CiError


class _Commands:
    def __init__(
        self,
        *,
        fail_one_shot: bool = False,
        fail_remove: bool = False,
        interrupt_signal: int | None = None,
    ) -> None:
        self.calls: list[list[str]] = []
        self.fail_one_shot = fail_one_shot
        self.fail_remove = fail_remove
        self.interrupt_signal = interrupt_signal
        self.container_present = False
        self.secret_source: Path | None = None
        self.secret_existed_during_cleanup = False
        self.secret_existed_during_verification = False

    def run(self, command, *, check=True, **_kwargs):
        rendered = [str(item) for item in command]
        self.calls.append(rendered)
        if rendered[:3] == ["docker", "run", "--rm"]:
            mount = next((item for item in rendered if "dst=/run/secrets/hf-token" in item), None)
            if mount is not None:
                self.secret_source = Path(mount.split("src=", 1)[1].split(",dst=", 1)[0])
                assert self.secret_source.is_file()
                assert self.secret_source.stat().st_mode & 0o777 == 0o600
            if self.interrupt_signal is not None:
                self.container_present = True
                signal.raise_signal(self.interrupt_signal)
            if self.fail_one_shot:
                self.container_present = True
                raise CiError("simulated one-shot failure")
            self.container_present = False
        if rendered[:2] == ["docker", "ps"]:
            if self.secret_source is not None:
                self.secret_existed_during_verification = self.secret_source.is_file()
            stdout = "abc123\n" if self.container_present else ""
            return subprocess.CompletedProcess(rendered, 0, stdout, "")
        if rendered[:3] == ["docker", "rm", "-f"] and self.secret_source is not None:
            self.secret_existed_during_cleanup = self.secret_source.is_file()
        if rendered[:3] == ["docker", "rm", "-f"]:
            if self.fail_remove:
                return subprocess.CompletedProcess(rendered, 1, "", "simulated remove failure")
            self.container_present = False
        returncode = 1 if rendered[:2] == ["docker", "exec"] else 0
        result = subprocess.CompletedProcess(rendered, returncode, "", "")
        if check and returncode:
            raise CiError("simulated command failure")
        return result


def _container(tmp_path: Path, **extra: str) -> CiContainer:
    workspace = tmp_path / "workspace"
    runner_temp = tmp_path / "runner-temp"
    workspace.mkdir()
    runner_temp.mkdir()
    test_name = "".join(
        character if character.isalnum() or character in "_-" else "-"
        for character in tmp_path.name
    )
    environment = {
        "GITHUB_WORKSPACE": str(workspace),
        "RUNNER_TEMP": str(runner_temp),
        "TRTMC_CI_IMAGE": "ci-image",
        "TRTMC_CI_CONTAINER_NAME": f"trtmc-test-{os.getpid()}-{test_name}",
        **extra,
    }
    return CiContainer(environment)


def test_one_shot_hf_token_uses_an_ephemeral_read_only_file_mount(tmp_path: Path) -> None:
    token = "hf_test_secret_value"
    container = _container(tmp_path, HF_TOKEN=token, HUGGING_FACE_HUB_TOKEN=token)
    commands = _Commands()
    container.commands = commands

    container.run_once(["python", "-V"])

    docker_run = next(call for call in commands.calls if call[:3] == ["docker", "run", "--rm"])
    rendered = " ".join(docker_run)
    assert token not in rendered
    assert "HF_TOKEN=" not in rendered
    assert "HUGGING_FACE_HUB_TOKEN=" not in rendered
    assert "HF_TOKEN_PATH=/run/secrets/hf-token" in docker_run
    assert "-d" not in docker_run
    assert "sleep" not in docker_run
    assert ["--user", f"{os.getuid()}:{os.getgid()}"] == docker_run[
        docker_run.index("--user") : docker_run.index("--user") + 2
    ]
    assert docker_run[-3:] == ["ci-image", "python", "-V"]
    assert commands.secret_source is not None
    assert commands.secret_existed_during_verification
    assert not commands.secret_source.exists()
    assert not commands.secret_source.parent.exists()
    assert commands.calls[-1][:2] == ["docker", "ps"]


def test_one_shot_failure_removes_container_before_secret(tmp_path: Path) -> None:
    container = _container(tmp_path, HF_TOKEN="hf_failure_secret")
    commands = _Commands(fail_one_shot=True)
    container.commands = commands

    with pytest.raises(CiError, match="simulated one-shot failure"):
        container.run_once(["false"])

    assert commands.secret_source is not None
    assert commands.secret_existed_during_cleanup
    assert not commands.secret_source.exists()
    assert any(call[:3] == ["docker", "rm", "-f"] for call in commands.calls)


@pytest.mark.parametrize("interrupt_signal", [signal.SIGINT, signal.SIGTERM])
def test_one_shot_interrupt_removes_exact_container_before_secret(
    tmp_path: Path, interrupt_signal: int
) -> None:
    container = _container(tmp_path, HF_TOKEN="hf_interrupted_secret")
    commands = _Commands(interrupt_signal=interrupt_signal)
    container.commands = commands

    with pytest.raises(CiError, match=signal.Signals(interrupt_signal).name):
        container.run_once(["long-running-command"])

    assert commands.secret_source is not None
    assert commands.secret_existed_during_cleanup
    assert not commands.secret_source.exists()
    assert ["docker", "rm", "-f", container.config.name] in commands.calls
    assert commands.container_present is False


def test_cli_sigterm_cleans_a_real_blocking_docker_child_promptly(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker_log = tmp_path / "docker.log"
    run_started = tmp_path / "run-started"
    container_state = tmp_path / "container-state"
    raw_token_leaked = tmp_path / "raw-token-leaked"
    docker = fake_bin / "docker"
    docker.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'printf \'%s\\n\' "$*" >> "$DOCKER_LOG"\n'
        'case "${1:-} ${2:-}" in\n'
        "  'ps --all')\n"
        "    if [ -f \"$CONTAINER_STATE\" ]; then printf 'abc123\\n'; fi ;;\n"
        "  'image inspect') exit 0 ;;\n"
        "  'run --rm')\n"
        '    if [ -n "${HF_TOKEN:-}" ] || [ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]; then\n'
        '      touch "$RAW_TOKEN_LEAKED"\n'
        "    fi\n"
        '    touch "$CONTAINER_STATE" "$RUN_STARTED"\n'
        "    trap '' INT TERM\n"
        '    while [ -f "$CONTAINER_STATE" ]; do sleep 0.05; done ;;\n'
        "  'rm -f') rm -f \"$CONTAINER_STATE\" ;;\n"
        "  *) exit 2 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runner_temp = tmp_path / "runner-temp"
    runner_temp.mkdir()
    container_name = "signal-container"
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "DOCKER_LOG": str(docker_log),
            "RUN_STARTED": str(run_started),
            "CONTAINER_STATE": str(container_state),
            "RAW_TOKEN_LEAKED": str(raw_token_leaked),
            "GITHUB_WORKSPACE": str(workspace),
            "RUNNER_TEMP": str(runner_temp),
            "TRTMC_CI_IMAGE": "ci-image",
            "TRTMC_CI_CONTAINER_NAME": container_name,
            "HF_TOKEN": "hf_must_only_exist_in_the_mode_0600_file",
        }
    )
    process = subprocess.Popen(
        [sys.executable, "-m", "tools.ci", "container", "run", "--", "wait-forever"],
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not run_started.is_file():
            assert process.poll() is None
            time.sleep(0.02)
        assert run_started.is_file()
        started = time.monotonic()
        os.kill(process.pid, signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=5)
        elapsed = time.monotonic() - started
    finally:
        if process.poll() is None:
            container_state.unlink(missing_ok=True)
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate(timeout=5)

    assert process.returncode == 1, stdout + stderr
    assert elapsed < 2
    assert "interrupted by SIGTERM" in stderr
    assert not container_state.exists()
    assert not (runner_temp / f"trtmc-hf-secret-{container_name}").exists()
    assert not (Path("/tmp") / f"trtmc-hf-secret-{container_name}").exists()
    assert not raw_token_leaked.exists()
    assert f"rm -f {container_name}" in docker_log.read_text(encoding="utf-8")


def test_docker_remove_failure_retains_the_token_and_fails_closed(tmp_path: Path) -> None:
    container = _container(tmp_path, HF_TOKEN="hf_retained_until_verified")
    commands = _Commands(fail_one_shot=True, fail_remove=True)
    container.commands = commands

    with pytest.raises(CiError, match="Could not remove exact CI container"):
        container.run_once(["false"])

    assert commands.secret_source is not None
    assert commands.secret_source.read_text(encoding="utf-8") == "hf_retained_until_verified"
    assert commands.container_present

    commands.fail_remove = False
    container.cleanup()
    assert not commands.secret_source.exists()
    assert not commands.container_present


def test_always_cleanup_removes_deterministic_current_run_residue(tmp_path: Path) -> None:
    container = _container(
        tmp_path,
        GITHUB_RUN_ID="4242",
        GITHUB_RUN_ATTEMPT="3",
        TRTMC_CI_CONTAINER_NAME="trtmc-nightly-cache-4242-3",
    )
    secret_directory = container._secret_directory()
    secret_directory.mkdir(mode=0o700)
    secret_directory.chmod(0o700)
    token_file = secret_directory / "token"
    token_file.write_text("hard-cancel-residue", encoding="utf-8")
    token_file.chmod(0o600)
    commands = _Commands()
    commands.container_present = True
    commands.secret_source = token_file
    container.commands = commands

    container.cleanup()

    assert ["docker", "rm", "-f", "trtmc-nightly-cache-4242-3"] in commands.calls
    assert commands.secret_existed_during_cleanup
    assert not secret_directory.exists()


def test_secret_cleanup_rejects_a_symlink_without_touching_its_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    container = _container(tmp_path)
    secret_root = tmp_path / "secret-root"
    secret_root.mkdir()
    monkeypatch.setattr("tools.ci.container._HOST_SECRET_ROOT", secret_root)
    victim = tmp_path / "victim"
    victim.mkdir()
    victim_token = victim / "token"
    victim_token.write_text("do-not-delete", encoding="utf-8")
    container._secret_directory().symlink_to(victim, target_is_directory=True)
    container.commands = _Commands()

    with pytest.raises(CiError, match="not a safe directory"):
        container.cleanup()

    assert victim_token.read_text(encoding="utf-8") == "do-not-delete"


def test_secret_path_rejects_a_symlinked_host_secret_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    real_root = tmp_path / "real-root"
    real_root.mkdir()
    linked_root = tmp_path / "linked-root"
    linked_root.symlink_to(real_root, target_is_directory=True)
    monkeypatch.setattr("tools.ci.container._HOST_SECRET_ROOT", linked_root)
    container = CiContainer(
        {
            "GITHUB_WORKSPACE": str(workspace),
            "TRTMC_CI_IMAGE": "ci-image",
        }
    )
    container.commands = _Commands()

    with pytest.raises(CiError, match="Host CI secret root must be an absolute, real directory"):
        container.cleanup()


def test_realistic_runner_temp_secret_has_no_shared_users_alias(tmp_path: Path) -> None:
    container = _container(
        tmp_path,
        RUNNER_TEMP="/workspace/users/yifeif/actions-runner/_work/_temp",
        HF_TOKEN="hf_no_shared_mount_alias",
        TRTMC_CONTAINER_OPTIONS="-v /workspace/users/yifeif:/workspace/users/yifeif",
    )
    commands = _Commands()
    container.commands = commands

    container.run_once(["python", "-V"])

    docker_run = next(call for call in commands.calls if call[:3] == ["docker", "run", "--rm"])
    assert "/workspace/users/yifeif:/workspace/users/yifeif" in docker_run
    assert commands.secret_source is not None
    assert Path("/workspace/users/yifeif") not in commands.secret_source.parents
    assert container.config.workspace not in commands.secret_source.parents
    assert commands.secret_source.parent.parent == Path("/tmp")
    assert not commands.secret_source.exists()


@pytest.mark.parametrize(
    "options",
    [
        "-v /tmp:/host-tmp",
        "--volume=/tmp:/host-tmp:ro",
        "--mount type=bind,src=/tmp,dst=/host-tmp,readonly",
        "--mount=type=bind,source=/tmp,destination=/host-tmp,readonly",
    ],
)
def test_secret_fails_closed_when_an_ordinary_bind_covers_it(tmp_path: Path, options: str) -> None:
    container = _container(
        tmp_path,
        HF_TOKEN="hf_must_not_have_a_mount_alias",
        TRTMC_CONTAINER_OPTIONS=options,
    )
    commands = _Commands()
    container.commands = commands

    with pytest.raises(CiError, match="would be exposed by container bind source: /tmp"):
        container.run_once(["python", "-V"])

    assert not any(call[:2] == ["docker", "run"] for call in commands.calls)
    assert not container._secret_directory().exists()


def test_secret_fails_closed_for_inherited_container_volumes(tmp_path: Path) -> None:
    container = _container(
        tmp_path,
        HF_TOKEN="hf_must_not_inherit_an_unknown_bind",
        TRTMC_CONTAINER_OPTIONS="--volumes-from another-container:ro",
    )
    commands = _Commands()
    container.commands = commands

    with pytest.raises(CiError, match="cannot inherit unverified container volumes"):
        container.run_once(["python", "-V"])

    assert not any(call[:2] == ["docker", "run"] for call in commands.calls)
    assert not container._secret_directory().exists()


def test_no_token_one_shot_keeps_the_plain_cli_boundary(tmp_path: Path) -> None:
    container = _container(tmp_path)
    commands = _Commands()
    container.commands = commands

    container.run_once(["python", "-V"])

    docker_run = next(call for call in commands.calls if call[:3] == ["docker", "run", "--rm"])
    assert docker_run[:7] == [
        "docker",
        "run",
        "--rm",
        "--name",
        container.config.name,
        "--user",
        f"{os.getuid()}:{os.getgid()}",
    ]
    assert "HF_TOKEN_PATH=/run/secrets/hf-token" not in docker_run
    assert not any("/run/secrets" in item for item in docker_run)
    assert not container._secret_directory().exists()


def test_raw_hf_tokens_are_not_inherited_by_host_subprocesses(tmp_path: Path) -> None:
    container = _container(
        tmp_path,
        HF_TOKEN="hf_primary",
        HUGGING_FACE_HUB_TOKEN="hf_primary",
    )

    assert container.env["HF_TOKEN"] == "hf_primary"
    assert container.env["HUGGING_FACE_HUB_TOKEN"] == "hf_primary"
    assert "HF_TOKEN" not in container.commands.env
    assert "HUGGING_FACE_HUB_TOKEN" not in container.commands.env


def test_unsafe_container_name_is_rejected_before_cleanup(tmp_path: Path) -> None:
    with pytest.raises(CiError, match="CI container name is unsafe"):
        _container(tmp_path, TRTMC_CI_CONTAINER_NAME="../../not-owned")


def test_conflicting_hf_token_values_fail_before_docker_run(tmp_path: Path) -> None:
    container = _container(
        tmp_path,
        HF_TOKEN="first",
        HUGGING_FACE_HUB_TOKEN="second",
    )
    commands = _Commands()
    container.commands = commands

    with pytest.raises(CiError, match="values disagree"):
        container.run_once(["python", "-V"])

    assert not any(call[:2] == ["docker", "run"] for call in commands.calls)


@pytest.mark.parametrize("token_name", ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"])
def test_long_lived_container_rejects_hf_secret(tmp_path: Path, token_name: str) -> None:
    container = _container(
        tmp_path,
        **{token_name: "must-not-cross-boundary"},
    )
    commands = _Commands()
    container.commands = commands

    with pytest.raises(CiError, match="Long-lived CI containers must not receive"):
        container.start()

    assert commands.calls == []


def test_container_run_cli_dispatches_the_bounded_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    captured: list[list[str]] = []
    monkeypatch.setenv("GITHUB_WORKSPACE", str(workspace))
    monkeypatch.setenv("TRTMC_CI_IMAGE", "ci-image")
    monkeypatch.setattr(
        CiContainer,
        "run_once",
        lambda _self, command: captured.append(list(command)),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["python3", "container", "run", "--", "python", "-V"],
    )

    assert CiCommand().run() == 0
    assert captured == [["python", "-V"]]


def test_container_cleanup_cli_dispatches_exact_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    called: list[bool] = []
    monkeypatch.setenv("GITHUB_WORKSPACE", str(workspace))
    monkeypatch.setattr(CiContainer, "cleanup", lambda _self: called.append(True))
    monkeypatch.setattr(sys, "argv", ["python3", "container", "cleanup"])

    assert CiCommand().run() == 0
    assert called == [True]
