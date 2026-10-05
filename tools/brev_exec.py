# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reserve a ready Brev GPU VM or preserve one remote application result."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import secrets
import shlex
import signal
import subprocess
import sys
import time
from collections.abc import Iterable, Sequence
from pathlib import Path

from tools.ci.process import CiError


def remote_wrapper(command: Sequence[str], result_file: Path, marker: str) -> str:
    """Cache the application result remotely while returning success to Brev."""
    if not command:
        raise ValueError("remote application command must not be empty")
    if not result_file.is_absolute():
        raise ValueError("remote result file must be absolute")
    if not marker or "\n" in marker:
        raise ValueError("remote result marker must be one non-empty line")

    rendered = shlex.join(command)
    quoted_result = shlex.quote(str(result_file))
    quoted_marker = shlex.quote(marker)
    return "\n".join(
        (
            "set -u",
            f"result_file={quoted_result}",
            'if [ -s "$result_file" ]; then',
            '  status="$(cat -- "$result_file")"',
            "else",
            "  set +e",
            f"  {rendered}",
            "  status=$?",
            "  set -e",
            '  temporary="${result_file}.tmp.$$"',
            '  printf \'%s\\n\' "$status" > "$temporary"',
            '  mv -- "$temporary" "$result_file"',
            "fi",
            "case \"$status\" in ''|*[!0-9]*) status=255 ;; esac",
            f"printf '%s%s\\n' {quoted_marker} \"$status\"",
            "exit 0",
        )
    )


def parse_remote_status(lines: Iterable[str], marker: str) -> int:
    """Extract one consistent cached application result from Brev output."""
    statuses: list[int] = []
    for raw in lines:
        line = raw.strip()
        if not line.startswith(marker):
            continue
        value = line.removeprefix(marker)
        if not value.isdigit() or not 0 <= int(value) <= 255:
            raise CiError(f"invalid remote application result: {value!r}")
        statuses.append(int(value))
    if not statuses:
        raise CiError("Brev completed without a remote application result")
    if len(set(statuses)) != 1:
        raise CiError(f"Brev returned inconsistent application results: {statuses}")
    return statuses[0]


def execute(instance: str, command: Sequence[str], log: Path, result_file: Path) -> int:
    """Stream a Brev execution and return the cached application exit code."""
    marker = f"__TRTMC_REMOTE_EXIT_{secrets.token_hex(16)}__="
    wrapper = remote_wrapper(command, result_file, marker)
    log.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    with log.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(
            ["brev", "exec", instance, wrapper],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if process.stdout is None:
            raise CiError("Brev output stream was not created")
        for line in process.stdout:
            lines.append(line)
            output.write(line)
            output.flush()
            print(line, end="", flush=True)
        return_code = process.wait()
    if return_code != 0:
        raise CiError(f"Brev transport failed after retry handling (exit {return_code})")
    return parse_remote_status(lines, marker)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--result-file", required=True, type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def execution_main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and preserve the remote application's real conclusion."""
    arguments = _parser().parse_args(argv)
    command = arguments.command
    if command and command[0] == "--":
        command = command[1:]
    try:
        return execute(arguments.instance, command, arguments.log, arguments.result_file)
    except (CiError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


DEFAULT_PROBE_IMAGE = (
    "nvidia/cuda:13.0.2-base-ubuntu24.04"
    "@sha256:2ab6381d970b211fb93853796dc6707eb8a72575a375c422b17cf4d8b2641701"
)
POLL_INTERVAL = 20.0
CLI_TIMEOUT = 30.0
PROBE_TIMEOUT = 180.0
CLEANUP_TIMEOUT = 60.0
ATTEMPT_TIMEOUT = 600.0
DIAGNOSTIC_INTERVAL = 120.0

_STATE_VALUES = frozenset(
    {
        "RUNNING",
        "STARTING",
        "STOPPING",
        "STOPPED",
        "DEPLOYING",
        "DELETING",
        "DELETED",
        "TERMINATED",
        "FAILURE",
        "FAILED",
        "UNHEALTHY",
        "UNAVAILABLE",
        "HEALTHY",
        "PENDING",
        "BUILDING",
        "COMPLETED",
        "CREATE_FAILED",
        "READY",
        "NOT READY",
        "",
    }
)
_UNIT_VALUES = frozenset(
    {
        "loaded",
        "not-found",
        "masked",
        "error",
        "bad-setting",
        "merged",
        "stub",
        "active",
        "inactive",
        "failed",
        "activating",
        "deactivating",
        "reloading",
        "maintenance",
        "refreshing",
        "running",
        "exited",
        "dead",
        "start",
        "start-pre",
        "start-post",
        "stop",
        "stop-post",
        "auto-restart",
        "waiting",
        "listening",
        "success",
        "exit-code",
        "signal",
        "timeout",
        "core-dump",
        "watchdog",
        "start-limit-hit",
        "resources",
        "protocol",
        "oom-kill",
        "exec-condition",
        "assert",
        "dependency",
        "canceled",
    }
)


class ProvisionError(RuntimeError):
    """Provisioning did not establish the complete readiness contract."""


class InventoryError(ProvisionError):
    """Inventory visibility is unknown; retry without replacing the allocation."""


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ProvisionError("provisioning deadline expired")
    return remaining


def _log(start: float, message: str) -> None:
    print(f"[brev provision +{time.monotonic() - start:.1f}s] {message}", file=sys.stderr)


def _run(
    command: Sequence[str], deadline: float, cap: float = CLI_TIMEOUT
) -> subprocess.CompletedProcess[str]:
    """Bound the CLI and its SSH subprocesses by the same absolute deadline."""
    timeout = min(cap, _remaining(deadline))
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
        # Brev's internal retries can leave ssh children running after the CLI
        # is killed. Terminate the complete process group, including those children.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        # An escaped daemon may retain the pipes after the CLI group is gone.
        # Drain only within the remaining deadline; do not wait for that daemon.
        try:
            process.communicate(timeout=max(0.0, min(1.0, deadline - time.monotonic())))
        except subprocess.TimeoutExpired:
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
            try:
                process.wait(timeout=max(0.0, min(1.0, deadline - time.monotonic())))
            except subprocess.TimeoutExpired:
                pass
        if isinstance(error, KeyboardInterrupt):
            raise
        raise ProvisionError(f"brev {command[1]} exceeded its bounded wait") from None
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def _pause(deadline: float) -> None:
    time.sleep(min(POLL_INTERVAL, _remaining(deadline)))


def _instance(name: str, deadline: float) -> dict[str, str] | None:
    _remaining(deadline)
    try:
        result = _run(["brev", "ls", "--json"], deadline)
    except (ProvisionError, OSError):
        raise InventoryError("Brev inventory query unavailable") from None
    if result.returncode:
        raise InventoryError(f"Brev inventory failed (exit {result.returncode})")
    try:
        document = json.loads(result.stdout)
    except (ValueError, TypeError):
        raise InventoryError("Brev inventory did not return valid JSON") from None
    if not isinstance(document, dict) or "workspaces" not in document:
        raise InventoryError("Brev inventory must contain the workspaces collection")
    records = document["workspaces"]
    if records is None:
        records = []
    if not isinstance(records, list) or any(
        not isinstance(row, dict) or not isinstance(row.get("name"), str) for row in records
    ):
        raise InventoryError("Brev workspaces must be an array of named instance records or null")
    matching = [row for row in records if row.get("name") == name]
    if len(matching) > 1:
        raise ProvisionError("Brev inventory returned duplicate exact instance names")
    if not matching:
        return None
    row = matching[0]
    for key in ("id", "status", "build_status", "shell_status", "health_status"):
        if not isinstance(row.get(key), str):
            raise InventoryError(f"Brev inventory omitted a string {key}")
    if not row["id"]:
        raise InventoryError("Brev inventory omitted the instance ID")
    return row


def _inventory(name: str, deadline: float, start: float) -> dict[str, str] | None:
    while True:
        try:
            return _instance(name, deadline)
        except InventoryError as error:
            _log(start, f"{name}: {error}; retrying inventory on the same allocation")
            _pause(deadline)


def _state(row: dict[str, str]) -> str:
    fields = {
        key: row[key] if row[key] in _STATE_VALUES else "UNKNOWN"
        for key in ("status", "build_status", "shell_status", "health_status")
    }
    for key in ("id", "instance_type"):
        value = row.get(key, "")
        fields[key] = (
            value
            if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", value)
            else "UNKNOWN"
        )
    return json.dumps(fields)


def _ready(row: dict[str, str]) -> bool:
    if row["status"] in {"FAILURE", "FAILED", "DELETED", "TERMINATED"}:
        raise ProvisionError(f"Brev instance entered terminal state {row['status']}")
    if row["build_status"] in {"CREATE_FAILED", "FAILED", "FAILURE"}:
        raise ProvisionError("Brev environment setup failed")
    if row["status"] == "UNHEALTHY" or row["health_status"] in {"UNHEALTHY", "UNAVAILABLE"}:
        return False
    return (
        row["status"] == "RUNNING"
        and row["build_status"] == "COMPLETED"
        and row["shell_status"] == "READY"
    )


def _probe(image: str, marker: str) -> str:
    # This is repeatable infrastructure validation. Model build/tests never run
    # here, so a provision retry cannot hide an application failure.
    return "\n".join(
        (
            "set -eu",
            "phase() {",
            '  key="$1"; shift',
            '  if "$@" >/dev/null 2>&1; then code=0; else code=$?; fi',
            '  printf \'TRTMC_PHASE_%s=%s\\n\' "$key" "$code"',
            '  return "$code"',
            "}",
            "if command -v cloud-init >/dev/null 2>&1; then",
            "  phase cloud_init sudo -n cloud-init status --wait",
            "else",
            "  printf '%s\\n' 'TRTMC_PHASE_cloud_init=not_installed'",
            "fi",
            "phase docker sudo -n docker info",
            "phase host_gpu nvidia-smi --query-gpu=uuid --format=csv,noheader",
            f"phase container_gpu sudo -n docker run --rm --gpus all {shlex.quote(image)} "
            "nvidia-smi --query-gpu=uuid --format=csv,noheader",
            f"printf '%s\\n' {shlex.quote(marker)}",
        )
    )


def _diagnostics() -> str:
    """Read fixed infrastructure facts without exposing logs, user data or environment."""
    return "\n".join(
        (
            "set -u",
            "if command -v cloud-init >/dev/null 2>&1; then",
            "  if output=$(sudo -n cloud-init status 2>/dev/null); then code=0; else code=$?; fi",
            "  printf 'TRTMC_DIAG_CLOUD_EXIT=%s\\n' \"$code\"",
            "  printf '%s\\n' \"$output\" | awk '/^status: (not run|disabled|running|done|error)$/ "
            '{value=substr($0,9); gsub(/ /,"_",value); print "TRTMC_DIAG_CLOUD_STATE=" value}\'',
            "else printf '%s\\n' 'TRTMC_DIAG_CLOUD_STATE=not_installed'; fi",
            "for unit in docker cloud-init cloud-final jupyter cloudflared; do",
            '  sudo -n systemctl show "$unit.service" --no-pager '
            "--property=LoadState,ActiveState,SubState,Result,NRestarts 2>/dev/null | "
            'awk -v unit="$unit" -F= \'{print "TRTMC_DIAG_UNIT_" unit "_" $1 "=" $2}\'',
            "done",
            "if sudo -n test -r /var/log/brev-workspace.log 2>/dev/null; then",
            "  printf '%s\\n' 'TRTMC_DIAG_SETUP_READABLE=1'",
            '  sudo -n awk \'$0 == "------ Setup Begin ------" {begin++} '
            '$0 == "------ Setup End ------" {end++} '
            '$0 == "------ Success ------" {success++} '
            '$0 == "------ Failure ------" {failure++} '
            'END {printf "TRTMC_DIAG_SETUP_BEGIN=%d\\nTRTMC_DIAG_SETUP_END=%d\\n'
            'TRTMC_DIAG_SETUP_SUCCESS=%d\\nTRTMC_DIAG_SETUP_FAILURE=%d\\n", '
            "begin,end,success,failure}' /var/log/brev-workspace.log 2>/dev/null",
            "else printf '%s\\n' 'TRTMC_DIAG_SETUP_READABLE=0'; fi",
            "if nvidia-smi --query-gpu=uuid --format=csv,noheader >/dev/null 2>&1; "
            "then code=0; else code=$?; fi",
            "printf 'TRTMC_DIAG_HOST_GPU_EXIT=%s\\n' \"$code\"",
            "exit 0",
        )
    )


def _safe_lines(output: str) -> list[str]:
    result = []
    for line in output.splitlines():
        if re.fullmatch(
            r"TRTMC_PHASE_(cloud_init|docker|host_gpu|container_gpu)=(not_installed|[0-9]{1,3})",
            line,
        ):
            result.append(line)
        elif re.fullmatch(
            r"TRTMC_DIAG_(CLOUD_EXIT|HOST_GPU_EXIT|SETUP_(READABLE|BEGIN|END|SUCCESS|FAILURE))=[0-9]{1,9}",
            line,
        ):
            result.append(line)
        elif re.fullmatch(
            r"TRTMC_DIAG_CLOUD_STATE=(not_installed|not_run|disabled|running|done|error)", line
        ):
            result.append(line)
        else:
            match = re.fullmatch(
                r"TRTMC_DIAG_UNIT_(docker|cloud-init|cloud-final|jupyter|cloudflared)_"
                r"(LoadState|ActiveState|SubState|Result|NRestarts)=(.*)",
                line,
            )
            if match and (
                match[3] in _UNIT_VALUES
                or (match[2] == "NRestarts" and re.fullmatch(r"[0-9]{1,9}", match[3]))
            ):
                result.append(line)
    return result


def _diagnose(name: str, deadline: float, start: float) -> None:
    try:
        result = _run(["brev", "exec", name, _diagnostics()], deadline, CLI_TIMEOUT)
    except (ProvisionError, OSError):
        _log(start, f"{name}: bootstrap diagnostic unavailable within its bounded wait")
        return
    _log(start, f"{name}: bootstrap diagnostic CLI exit {result.returncode}")
    for line in _safe_lines(result.stdout):
        _log(start, f"{name}: {line}")


def _wait_ready(name: str, image: str, deadline: float, start: float) -> None:
    identity: str | None = None
    previous_state = ""
    next_diagnostic = start
    marker = f"TRTMC_GPU_READY_{secrets.token_hex(16)}"
    while True:
        _remaining(deadline)
        row = _inventory(name, deadline, start)
        if row is None:
            _log(start, f"{name}: not yet visible in inventory")
            _pause(deadline)
            continue
        if identity is not None and row["id"] != identity:
            raise ProvisionError("the instance ID changed while waiting for readiness")
        identity = row["id"]
        state = _state(row)
        if state != previous_state:
            _log(start, f"{name}: {state}")
            previous_state = state
        if _ready(row):
            try:
                result = _run(
                    ["brev", "exec", name, _probe(image, marker)], deadline, PROBE_TIMEOUT
                )
            except ProvisionError:
                _log(start, f"{name}: readiness probe exceeded its bounded wait")
            else:
                for line in _safe_lines(result.stdout):
                    _log(start, f"{name}: {line}")
                _log(start, f"{name}: readiness probe CLI exit {result.returncode}")
                if result.returncode == 0 and marker in result.stdout.splitlines():
                    # Bind the remote receipt to an instance that remains ready,
                    # rather than trusting an earlier inventory snapshot.
                    current = _inventory(name, deadline, start)
                    if current is None or current["id"] != identity:
                        raise ProvisionError(
                            "the instance disappeared or was replaced during its readiness probe"
                        )
                    if _ready(current):
                        _log(
                            start,
                            f"{name}: verified SSH, bootstrap, Docker, host GPU and container GPU",
                        )
                        return
                _log(
                    start, f"{name}: readiness probe has not established the complete GPU contract"
                )
        elif (
            row["status"] == "RUNNING"
            and row["build_status"] == "BUILDING"
            and time.monotonic() >= next_diagnostic
        ):
            _diagnose(name, deadline, start)
            next_diagnostic = time.monotonic() + DIAGNOSTIC_INTERVAL
        _pause(deadline)


def _publish_name(name: str) -> None:
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as stream:
            stream.write(f"instance_name={name}\n")
            stream.flush()


def _cleanup(name: str, deadline: float, start: float, uncertain_create: bool = False) -> None:
    """Do not replace an instance while its deletion remains unconfirmed."""
    allocation_seen = False
    delete_pending = True
    delete_attempted = False
    previous_state = ""
    identity: str | None = None
    while True:
        try:
            row = _instance(name, deadline)
            if row is None and delete_attempted and (not uncertain_create or allocation_seen):
                _log(start, f"{name}: deletion confirmed by exact name")
                return
            if row is not None:
                if identity is not None and row["id"] != identity:
                    break
                identity = row["id"]
                state = _state(row)
                if state != previous_state:
                    _log(start, f"{name}: cleanup inventory {state}")
                    previous_state = state
                # Also delete a late allocation from a timed-out create.
                delete_pending = delete_pending or not allocation_seen
                allocation_seen = True
        except InventoryError:
            _log(start, f"{name}: cleanup inventory unavailable; absence remains unconfirmed")
        except ProvisionError:
            break
        if delete_pending:
            delete_attempted = True
            try:
                result = _run(["brev", "delete", name], deadline)
                delete_pending = result.returncode != 0
                _log(start, f"{name}: delete requested (exit {result.returncode})")
                if not delete_pending:
                    # Confirm immediately when a slow delete leaves only a few
                    # seconds of the cleanup budget; later polling stays paced.
                    continue
            except (ProvisionError, OSError):
                _log(start, f"{name}: delete request unavailable; retrying within cleanup budget")
        try:
            _pause(deadline)
        except ProvisionError:
            break
    raise ProvisionError(f"cleanup of {name} is unconfirmed; refusing replacement") from None


def provision(
    instance: str,
    gpu: str,
    *,
    timeout: float = 1200,
    provider: str = "",
    fallback_provider: str = "aws",
    attempts: int = 3,
    probe_image: str = DEFAULT_PROBE_IMAGE,
) -> str:
    """Return one verified instance name, or fail within the overall deadline."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", instance):
        raise ValueError(
            "instance name must contain only letters, numbers, underscores, dots and hyphens"
        )
    if not math.isfinite(timeout) or timeout <= 0 or attempts < 1:
        raise ValueError("timeout must be finite and positive and attempts must be at least one")
    if not gpu or not probe_image:
        raise ValueError("GPU and probe image must not be empty")
    start = time.monotonic()
    overall_deadline = start + timeout
    active_deadline = overall_deadline - min(CLEANUP_TIMEOUT, timeout / 5)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        _remaining(active_deadline)
        # AWS can need more than five minutes just to establish SSH. Do not
        # shorten a healthy replacement's wait by dividing time among attempts
        # that may never be needed. Every attempt still shares the total budget.
        attempt_deadline = min(active_deadline, time.monotonic() + ATTEMPT_TIMEOUT)
        name = instance if attempt == 1 else f"{instance}-r{attempt}"
        if _inventory(name, attempt_deadline, start) is not None:
            raise ProvisionError(
                f"instance {name} already exists; refusing to reuse an earlier allocation"
            )
        chosen_provider = provider if attempt == 1 else fallback_provider
        # CI does not need Jupyter. Request the official CLI opt-out; the pinned
        # CLI omits false from JSON, so this is not proof of server-side disablement.
        command = ["brev", "create", name, "-g", gpu, "--detached", "--jupyter=false"]
        if chosen_provider:
            command.extend(("--provider", chosen_provider))
        # Persist the name before even a timed-out create can allocate remotely.
        _publish_name(name)
        _log(start, f"{name}: creating GPU {gpu} (attempt {attempt}/{attempts})")
        create_accepted = False
        try:
            result = _run(command, attempt_deadline, PROBE_TIMEOUT)
            if result.returncode:
                raise ProvisionError(f"Brev create failed (exit {result.returncode})")
            create_accepted = True
            _wait_ready(name, probe_image, attempt_deadline, start)
            return name
        except KeyboardInterrupt:
            _log(start, f"{name}: interrupted; cleaning up without replacement")
            try:
                _cleanup(
                    name,
                    min(overall_deadline, time.monotonic() + CLEANUP_TIMEOUT),
                    start,
                    uncertain_create=not create_accepted,
                )
            except (ProvisionError, OSError):
                _log(start, f"{name}: cleanup remains unconfirmed after interruption")
            raise
        except (ProvisionError, OSError) as error:
            last_error = error
            _log(
                start,
                f"{name}: provision/readiness failed: {error}; cleaning up before any replacement",
            )
            _cleanup(
                name,
                min(overall_deadline, time.monotonic() + CLEANUP_TIMEOUT),
                start,
                uncertain_create=not create_accepted,
            )
    raise ProvisionError(f"GPU provisioning failed after {attempts} attempts: {last_error}")


def provision_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Block until a newly reserved Brev GPU VM is usable."
    )
    parser.add_argument("--instance", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--provider", default="")
    parser.add_argument("--fallback-provider", default="aws")
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--probe-image", default=DEFAULT_PROBE_IMAGE)
    arguments = parser.parse_args(argv)
    try:
        print(provision(**vars(arguments)))
        return 0
    except (ProvisionError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("ERROR: GPU provisioning interrupted", file=sys.stderr)
        return 130


def main(argv: Sequence[str] | None = None) -> int:
    """Keep remote execution compatible and expose blocking GPU reservation."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "provision":
        return provision_main(arguments[1:])
    return execution_main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
