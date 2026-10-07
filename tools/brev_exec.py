# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reserve a ready Brev GPU VM or preserve one remote application result."""

from __future__ import annotations

import argparse
import base64
import hashlib
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


CHUNK_BYTES = 65536
RECONNECT_TIMEOUT = 300.0

_WORKER = r"""
import fcntl, json, os, pathlib, signal, subprocess, sys
p = json.loads(sys.argv[1])
root = pathlib.Path(p['directory'])
result = pathlib.Path(p['result'])
with (root / 'lock').open('a') as lock:
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(0)
    if result.exists() or (root / 'started').exists():
        raise SystemExit(0)
    (root / 'started').write_text('started\n')
    status = 255
    with (root / 'output.log').open('wb') as output:
        try:
            process = subprocess.Popen(p['command'], stdin=subprocess.DEVNULL,
                stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                status = process.wait(timeout=p['timeout'])
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        output.write(b'Remote process did not exit after kill; VM cleanup is required.\n')
                status = 124
        except OSError:
            output.write(b'Unable to start the remote application.\n')
            status = 127
    if status < 0:
        status = min(255, 128 - status)
    temporary = result.with_name(result.name + '.tmp')
    temporary.write_text(str(status) + '\n')
    temporary.replace(result)
"""


def _payload(command: Sequence[str], result: Path, timeout: float) -> dict:
    serialized = json.dumps(list(command), separators=(",", ":"))
    return {
        "command": list(command),
        "result": str(result),
        "directory": str(result) + ".task",
        "fingerprint": hashlib.sha256(serialized.encode()).hexdigest(),
        "timeout": timeout,
    }


def _script(program: str, payload: dict) -> str:
    encoded = base64.b64encode(json.dumps(payload).encode()).decode()
    prefix = f"import base64, json\np = json.loads(base64.b64decode({encoded!r}))\n"
    return "python3 -c " + shlex.quote(prefix + program)


def launch_script(payload: dict, marker: str) -> str:
    """Concurrent launches share a fingerprint and a host-side execution lock."""
    program = r"""
import fcntl, os, pathlib, subprocess, sys
os.umask(0o077)
root = pathlib.Path(p['directory'])
root.mkdir(parents=True, exist_ok=True)
with (root / 'launch.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    fingerprint = root / 'fingerprint'
    conflict = (fingerprint.exists() and fingerprint.read_text() != p['fingerprint'])
    conflict = conflict or (pathlib.Path(p['result']).exists() and not fingerprint.exists())
    if conflict:
        state = 'conflict'
    else:
        if not fingerprint.exists():
            fingerprint.write_text(p['fingerprint'])
        if not (root / 'started').exists() and not pathlib.Path(p['result']).exists():
            worker = root / 'worker.py'
            if not worker.exists():
                temporary = root / 'worker.tmp'
                temporary.write_text(p['worker'])
                temporary.replace(worker)
            subprocess.Popen([sys.executable, str(worker), json.dumps(p)],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, start_new_session=True, close_fds=True)
        state = 'accepted'
print(p['marker'] + json.dumps({'state': state}))
"""
    return _script(program, {**payload, "worker": _WORKER, "marker": marker})


def poll_script(payload: dict, marker: str, offset: int) -> str:
    """Transfer bounded log chunks and report completion independently of SSH."""
    program = r"""
import fcntl, pathlib
root = pathlib.Path(p['directory'])
result = pathlib.Path(p['result'])
fingerprint = root / 'fingerprint'
record = {'state': 'missing', 'offset': p['offset'], 'data': '', 'eof': True}
def completed():
    value = result.read_text().strip()
    if not value.isdecimal() or not 0 <= int(value) <= 255:
        record['state'] = 'invalid'
    else:
        record.update(state='complete', exit_code=int(value))
if fingerprint.exists():
    if fingerprint.read_text() != p['fingerprint']:
        record['state'] = 'conflict'
    else:
        record['state'] = 'pending'
        if result.exists():
            completed()
        else:
            with (root / 'lock').open('a') as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    if result.exists():
                        completed()
                    elif (root / 'started').exists():
                        record['state'] = 'lost'
                except BlockingIOError:
                    record['state'] = 'running'
        log = root / 'output.log'
        if log.exists():
            with log.open('rb') as stream:
                stream.seek(p['offset'])
                data = stream.read(p['chunk_bytes'])
                record.update(data=base64.b64encode(data).decode(),
                    offset=stream.tell(), eof=not bool(stream.read(1)))
print(p['marker'] + json.dumps(record))
"""
    return _script(
        program, {**payload, "marker": marker, "offset": offset, "chunk_bytes": CHUNK_BYTES}
    )


def _record(output: str, marker: str) -> dict:
    records = [line.removeprefix(marker) for line in output.splitlines() if line.startswith(marker)]
    if len(records) != 1:
        raise CiError("SSH did not return exactly one durable task receipt")
    try:
        result = json.loads(records[0])
    except ValueError as error:
        raise CiError("SSH returned an invalid durable task receipt") from error
    if not isinstance(result, dict):
        raise CiError("SSH returned a non-object durable task receipt")
    return result


def _ssh(instance: str, script: str, deadline: float) -> subprocess.CompletedProcess[str]:
    argv = [
        "ssh",
        "-F",
        str(Path.home() / ".brev/ssh_config"),
        "-o",
        "BatchMode=yes",
        "-o",
        "ControlMaster=no",
        "-o",
        "ControlPath=none",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ServerAliveInterval=20",
        "-o",
        "ServerAliveCountMax=3",
        "-T",
        instance,
        script,
    ]
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise CiError("Remote task observation deadline expired")
    process = subprocess.Popen(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=min(45, remaining))
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.communicate(timeout=1)
        except subprocess.TimeoutExpired:
            for pipe in (process.stdout, process.stderr):
                if pipe is not None:
                    pipe.close()
        raise
    return subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)


def execute(
    instance: str,
    command: Sequence[str],
    log: Path,
    result_file: Path,
    timeout: float = 2700,
    poll_interval: float = 20,
) -> int:
    """Start once, reconnect to the same task, and return its persisted exit code."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", instance):
        raise ValueError("invalid Brev instance name")
    if not command or not all(isinstance(value, str) and value for value in command):
        raise ValueError("remote application command must not be empty")
    if not result_file.is_absolute() or not result_file.name:
        raise ValueError("remote result file must be an absolute file path")
    if (
        not math.isfinite(timeout)
        or timeout <= 0
        or not math.isfinite(poll_interval)
        or poll_interval <= 0
    ):
        raise ValueError("task and polling timeouts must be finite and positive")
    if not (Path.home() / ".brev/ssh_config").is_file():
        raise CiError("Brev SSH configuration is missing; run brev refresh first")
    payload = _payload(command, result_file, timeout)
    marker = f"__TRTMC_TASK_{secrets.token_hex(16)}__="
    deadline = time.monotonic() + timeout + 60
    outage = None
    launched = False
    offset = 0
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("wb") as output:
        while time.monotonic() < deadline:
            script = (
                poll_script(payload, marker, offset) if launched else launch_script(payload, marker)
            )
            try:
                response = _ssh(instance, script, deadline)
            except (OSError, subprocess.TimeoutExpired):
                response = None
            if response is None or response.returncode:
                outage = time.monotonic() if outage is None else outage
                if time.monotonic() - outage >= RECONNECT_TIMEOUT:
                    raise CiError("SSH remained unavailable; the existing task was not restarted")
                print("Waiting to reconnect to the existing remote GPU task", flush=True)
            else:
                outage = None
                record = _record(response.stdout, marker)
                state = record.get("state")
                if not launched:
                    if state != "accepted":
                        raise CiError("Remote result path belongs to a different task")
                    launched = True
                    # A delayed acknowledgment must not shorten the worker's
                    # own execution budget; launch reconnects are bounded above.
                    deadline = time.monotonic() + timeout + 60
                    continue
                if state not in {"pending", "running", "complete"}:
                    raise CiError(f"Remote GPU task has no trustworthy result ({state})")
                try:
                    chunk = base64.b64decode(record["data"], validate=True)
                    next_offset = record["offset"]
                    if type(next_offset) is not int or next_offset != offset + len(chunk):
                        raise ValueError("invalid log offset")
                    if not isinstance(record["eof"], bool) or len(chunk) > CHUNK_BYTES:
                        raise ValueError("invalid log chunk")
                except (KeyError, TypeError, ValueError) as error:
                    raise CiError("Remote GPU task returned malformed log data") from error
                output.write(chunk)
                output.flush()
                print(chunk.decode("utf-8", errors="replace"), end="", flush=True)
                offset = next_offset
                if state == "complete":
                    code = record.get("exit_code")
                    if type(code) is not int or not 0 <= code <= 255:
                        raise CiError("Remote GPU task returned an invalid application exit code")
                    if record["eof"]:
                        return code
                if not record["eof"]:
                    continue
            time.sleep(min(poll_interval, max(0, deadline - time.monotonic())))
    raise CiError("Remote GPU task exceeded its observation budget; refusing to rerun it")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--result-file", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=2700)
    parser.add_argument("--poll-interval", type=float, default=20)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def execution_main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and preserve the remote application's real conclusion."""
    arguments = _parser().parse_args(argv)
    command = arguments.command
    if command and command[0] == "--":
        command = command[1:]
    try:
        return execute(
            arguments.instance,
            command,
            arguments.log,
            arguments.result_file,
            arguments.timeout,
            arguments.poll_interval,
        )
    except (CiError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


DEFAULT_PROBE_IMAGE = (
    "nvidia/cuda:13.0.2-base-ubuntu24.04"
    "@sha256:2ab6381d970b211fb93853796dc6707eb8a72575a375c422b17cf4d8b2641701"
)
DEFAULT_INSTANCE_TYPE = "g6.4xlarge"
NEBIUS_INSTANCE_TYPE = "gpu-l40s-a.1gpu-16vcpu-64gb"
POLL_INTERVAL = 60.0
CLI_TIMEOUT = 30.0
PROBE_TIMEOUT = 180.0
CLEANUP_TIMEOUT = 900.0

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
    command: Sequence[str],
    deadline: float,
    cap: float = CLI_TIMEOUT,
    *,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Bound the CLI and its SSH subprocesses by the same absolute deadline."""
    timeout = min(cap, _remaining(deadline))
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(input_text, timeout=timeout)
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


def _json(text: str):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def invalid(_value):
        raise ValueError("invalid JSON constant")

    return json.loads(text, object_pairs_hook=unique, parse_constant=invalid)


def _instance(
    name: str, deadline: float, identity: str | None = None, instance_type: str = ""
) -> dict[str, str] | None:
    _remaining(deadline)
    try:
        result = _run(["brev", "ls", "--json"], deadline)
    except (ProvisionError, OSError):
        raise InventoryError("Brev inventory query unavailable") from None
    if result.returncode:
        raise InventoryError(f"Brev inventory failed (exit {result.returncode})")
    try:
        document = _json(result.stdout)
    except (ValueError, TypeError):
        raise InventoryError("Brev inventory did not return valid JSON") from None
    if not isinstance(document, dict) or "workspaces" not in document:
        raise InventoryError("Brev inventory must contain the workspaces collection")
    records = document["workspaces"]
    if records is None:
        records = []
    if not isinstance(records, list) or any(
        not isinstance(row, dict)
        or not isinstance(row.get("name"), str)
        or not isinstance(row.get("id"), str)
        or not row["id"]
        for row in records
    ):
        raise InventoryError("Brev workspaces must be an array of named instance records or null")
    matching = [row for row in records if row.get("name") == name]
    if len(matching) > 1 or len({row["id"] for row in records}) != len(records):
        raise ProvisionError("Brev inventory returned ambiguous instance identities")
    if identity is not None:
        identified = [row for row in records if row["id"] == identity]
        if (matching and matching[0]["id"] != identity) or (
            identified and identified[0]["name"] != name
        ):
            raise ProvisionError("Brev instance name or ID changed; refusing to use or delete it")
    if not matching:
        return None
    row = matching[0]
    for key in ("id", "status", "build_status", "shell_status", "health_status", "instance_type"):
        if not isinstance(row.get(key), str):
            raise InventoryError(f"Brev inventory omitted a string {key}")
    if not row["id"]:
        raise InventoryError("Brev inventory omitted the instance ID")
    if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", row["id"]):
        raise InventoryError("Brev inventory returned an invalid instance ID")
    if instance_type:
        if not row["instance_type"]:
            raise InventoryError("Brev inventory has not identified the instance type")
        if row["instance_type"] != instance_type:
            raise ProvisionError("Brev instance type changed; refusing to use or delete it")
    return row


def _inventory(
    name: str, deadline: float, start: float, identity: str | None = None, instance_type: str = ""
) -> dict[str, str] | None:
    while True:
        try:
            return _instance(name, deadline, identity, instance_type)
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
    # These are the pinned CLI's failure enums. Unknown status/build values
    # remain pending; transient health never starts another allocation.
    if row["status"] == "FAILURE":
        raise ProvisionError(f"Brev instance entered terminal state {row['status']}")
    if row["build_status"] == "CREATE_FAILED":
        raise ProvisionError("Brev environment setup failed")
    if row["status"] == "UNHEALTHY" or row["health_status"] in {"UNHEALTHY", "UNAVAILABLE"}:
        return False
    return (
        row["status"] == "RUNNING"
        and row["build_status"] == "COMPLETED"
        and row["shell_status"] == "READY"
    )


def _probe(image: str, marker: str, min_free_disk_gb: float = 200) -> str:
    # These checks run only after native metadata READY. Repeated checks reuse
    # this VM; model build/tests never run inside provisioning.
    return "\n".join(
        (
            "set -eu",
            "phase() {",
            '  key="$1"; shift',
            '  if "$@" >/dev/null 2>&1; then code=0; else code=$?; fi',
            '  printf \'TRTMC_PHASE_%s=%s\\n\' "$key" "$code"',
            '  return "$code"',
            "}",
            "phase sudo sudo -n true",
            "if command -v cloud-init >/dev/null 2>&1; then",
            "  phase cloud_init sudo -n cloud-init status || true",
            "else",
            "  printf '%s\\n' 'TRTMC_PHASE_cloud_init=not_installed'",
            "fi",
            "phase docker sudo -n docker info",
            "free=$(python3 -c 'import os; print(min((s := os.statvfs(p)).f_bavail * s.f_frsize "
            'for p in ("/", "/tmp")))\')',
            "printf 'TRTMC_DISK_AVAILABLE_BYTES=%s\\n' \"$free\"",
            f'phase disk_headroom test "$free" -ge {int(min_free_disk_gb * 1024**3)}',
            "phase host_gpu nvidia-smi --query-gpu=uuid --format=csv,noheader",
            f"phase container_gpu sudo -n docker run --rm --gpus all {shlex.quote(image)} "
            "nvidia-smi --query-gpu=uuid --format=csv,noheader",
            f"printf '%s\\n' {shlex.quote(marker)}",
        )
    )


def _safe_lines(output: str) -> list[str]:
    result = []
    for line in output.splitlines():
        if re.fullmatch(
            r"TRTMC_PHASE_(sudo|cloud_init|disk_headroom|docker|host_gpu|container_gpu)=(not_installed|[0-9]{1,3})",
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
                r"TRTMC_DIAG_UNIT_(docker|cloud-init|cloud-final|instance-oneshot|nvidia-cdi-refresh)_"
                r"(LoadState|ActiveState|SubState|Result|NRestarts)=(.*)",
                line,
            )
            if match and (
                match[3] in _UNIT_VALUES
                or (match[2] == "NRestarts" and re.fullmatch(r"[0-9]{1,9}", match[3]))
            ):
                result.append(line)
        if re.fullmatch(r"TRTMC_DISK_AVAILABLE_BYTES=[0-9]{1,18}", line):
            result.append(line)
    return result


def _save_lease(path: Path, lease: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(lease, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_lease(path: Path | None, name: str) -> dict | None:
    if path is None or not path.exists():
        return None
    try:
        lease = _json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise ProvisionError("the persisted Brev lease is unreadable or invalid") from None
    if (
        not isinstance(lease, dict)
        or type(lease.get("schema_version")) is not int
        or lease["schema_version"] != 1
        or lease.get("name") != name
    ):
        raise ProvisionError("the persisted Brev lease does not match the requested instance")
    identity, sku = lease.get("instance_id"), lease.get("sku")
    if "create_started" in lease and type(lease["create_started"]) is not bool:
        raise ProvisionError("the persisted Brev lease has an invalid creation state")
    for key in (
        "create_accepted",
        "delete_accepted",
        "delete_attempted",
        "delete_outcome_unknown",
        "owned_instance_observed",
        "ownership_blocked",
        "cleanup_confirmed",
    ):
        if key in lease and type(lease[key]) is not bool:
            raise ProvisionError("the persisted Brev lease has an invalid ownership state")
    if identity is not None and not sku:
        raise ProvisionError("the persisted Brev lease omitted the recorded instance type")
    if (
        (
            identity is not None
            and (
                not isinstance(identity, str)
                or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", identity)
            )
        )
        or (
            sku is not None
            and (not isinstance(sku, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", sku))
        )
        or type(lease.get("allocation_pending", False)) is not bool
    ):
        raise ProvisionError("the persisted Brev lease contains an invalid identity")
    org = lease.get("organization_id")
    if org is not None and (
        not isinstance(org, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", org)
    ):
        raise ProvisionError("the persisted Brev lease contains an invalid organization")
    run, attempt = lease.get("run_id"), lease.get("run_attempt")
    if run is not None or attempt is not None:
        if (
            not str(run).isdigit()
            or not str(attempt).isdigit()
            or int(str(run)) < 1
            or int(str(attempt)) < 1
            or name != f"trtmc-gpu-ci-{run}-{attempt}"
        ):
            raise ProvisionError("the persisted Brev lease has a different original run attempt")
    return lease


def _cleanup_credentials() -> tuple[str, str]:
    """Use v0.6.335 credential precedence without exposing credentials."""
    path = Path.home() / ".brev" / "credentials.json"
    saved = {}
    key = os.environ.get("BREV_API_KEY", "").strip()
    if path.exists():
        try:
            saved = _json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            if not key:
                raise InventoryError("Brev cleanup credentials are unavailable") from None
            saved = {}
        if not isinstance(saved, dict):
            if not key:
                raise InventoryError("Brev cleanup credentials are unavailable")
            saved = {}
    if not key:
        value = saved.get("api_key", "")
        if not isinstance(value, str):
            raise InventoryError("Brev cleanup credentials are unavailable")
        key = value.strip()
    token = key or saved.get("access_token", "")
    if (
        not isinstance(token, str)
        or not token
        or any(character.isspace() for character in token)
        or (key and not key.startswith("bak-"))
    ):
        raise InventoryError("Brev cleanup credentials are unavailable")
    org = saved.get("api_key_org_id", "") if key else ""
    if not isinstance(org, str):
        raise InventoryError("Brev cleanup credential scope is unavailable")
    return token, org.strip()


def _cleanup_http(method: str, identity: str) -> tuple[int, dict | None]:
    """Perform one official request; the parent enforces its absolute timeout."""
    import urllib.error
    import urllib.parse
    import urllib.request

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, request, file, code, message, headers, url):
            return None

    token, credential_org = _cleanup_credentials()
    base = os.environ.get(
        "BREV_API_URL", "https://brevapi.us-west-2-prod.control-plane.brev.dev"
    ).rstrip("/")
    parsed = urllib.parse.urlsplit(base)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "brevapi.us-west-2-prod.control-plane.brev.dev"
        or parsed.port not in (None, 443)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path
        or parsed.query
        or parsed.fragment
    ):
        raise InventoryError("Brev cleanup requires the official HTTPS control-plane origin")
    if method in {"AUTH", "LIST"}:
        org = identity if method == "LIST" else identity or credential_org
        if not org or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", org):
            raise InventoryError("Brev cleanup credential organization is unavailable")
        endpoint = f"api/organizations/{urllib.parse.quote(org, safe='')}"
        if method == "LIST":
            endpoint += "/workspaces"
        verb = "GET"
    elif method in {"GET", "DELETE"} and re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", identity):
        endpoint = f"api/workspaces/{urllib.parse.quote(identity, safe='')}"
        verb = method
    else:
        raise InventoryError("invalid Brev cleanup request")
    request = urllib.request.Request(
        f"{base}/{endpoint}",
        method=verb,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(NoRedirect())
    try:
        with opener.open(request, timeout=CLI_TIMEOUT) as response:
            status = response.status
            headers = getattr(response, "headers", {})
            if method == "LIST" and (
                re.search(r'rel\s*=\s*["\']?next', headers.get("Link", ""), re.I)
                or headers.get("X-Next-Page")
                or headers.get("X-Next-Cursor")
                or headers.get("Content-Range")
            ):
                raise InventoryError("Brev cleanup inventory is incomplete")
            raw = response.read(1024 * 1024 + 1)
    except urllib.error.HTTPError as error:
        return error.code, None
    except (OSError, ValueError):
        raise InventoryError("Brev cleanup request is unavailable") from None
    if len(raw) > 1024 * 1024:
        raise InventoryError("Brev cleanup response is invalid")
    try:
        body = _json(raw.decode("utf-8"))
    except (UnicodeError, ValueError):
        raise InventoryError("Brev cleanup response is invalid") from None
    if method == "LIST":
        if not isinstance(body, list) or any(not isinstance(item, dict) for item in body):
            raise InventoryError("Brev cleanup inventory is invalid")
        return status, {
            "items": [
                {field: item.get(field) for field in ("id", "name", "organizationId")}
                for item in body
            ]
        }
    if not isinstance(body, dict):
        raise InventoryError("Brev cleanup response is invalid")
    if method == "AUTH" and body.get("id") != org:
        raise InventoryError("Brev cleanup authenticated organization is invalid")
    fields = (
        ("id",) if method == "AUTH" else ("id", "name", "organizationId", "instanceType", "status")
    )
    return status, {field: body.get(field) for field in fields}


def _cleanup_request(method: str, identity: str, deadline: float | None) -> tuple[int, dict | None]:
    # Read the credential inside the child: neither argv nor captured output contains it.
    script = (
        "import json,sys\n"
        "from tools.brev_exec import _cleanup_http, InventoryError\n"
        "try:\n"
        " result = _cleanup_http(sys.argv[1], sys.argv[2])\n"
        "except (InventoryError, OSError, ValueError):\n"
        " result = (0, None)\n"
        "print(json.dumps(result))\n"
    )
    request_deadline = deadline if deadline is not None else time.monotonic() + CLI_TIMEOUT
    try:
        result = _run([sys.executable, "-c", script, method, identity], request_deadline)
        status, body = _json(result.stdout)
    except (ProvisionError, OSError, ValueError, TypeError):
        raise InventoryError("Brev cleanup request is unavailable") from None
    if result.returncode or type(status) is not int or not 100 <= status <= 599:
        raise InventoryError("Brev cleanup request is unavailable")
    if body is not None and not isinstance(body, dict):
        raise InventoryError("Brev cleanup response is invalid")
    return status, body


def _allocation_organization(deadline: float) -> str:
    """Bind the original CI credential's canonical organization before requesting a VM."""
    token, org = _cleanup_credentials()
    if not token.startswith("bak-") or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", org):
        raise ProvisionError(
            "allocation requires a logged-in API key with its original organization"
        )
    status, body = _cleanup_request("AUTH", org, deadline)
    if status != 200 or not isinstance(body, dict) or body.get("id") != org:
        raise ProvisionError("the original allocation organization could not be authenticated")
    return org


def _cleanup_workspace(body: dict | None, lease: dict) -> None:
    if isinstance(body, dict):
        for field, expected in (
            ("id", lease["instance_id"]),
            ("name", lease["name"]),
            ("instanceType", lease["sku"]),
            ("organizationId", lease.get("organization_id")),
        ):
            value = body.get(field)
            if expected is not None and isinstance(value, str) and value and value != expected:
                raise ProvisionError("Brev cleanup instance identity changed; refusing deletion")
    if not isinstance(body, dict) or any(
        not isinstance(body.get(field), str) or not body[field]
        for field in ("id", "name", "organizationId", "instanceType")
    ):
        raise InventoryError("Brev cleanup response omitted its instance identity")
    if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", body["organizationId"]):
        raise InventoryError("Brev cleanup response omitted its organization identity")
    lease["organization_id"] = body["organizationId"]


def _cleanup_list_absent(body: dict | None, lease: dict) -> bool:
    if not isinstance(body, dict) or set(body) != {"items"} or not isinstance(body["items"], list):
        raise InventoryError("Brev cleanup inventory is invalid or incomplete")
    items = body["items"]
    identities = set()
    for item in items:
        if (
            not isinstance(item, dict)
            or any(
                not isinstance(item.get(field), str) or not item[field]
                for field in ("id", "name", "organizationId")
            )
            or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", item["id"])
        ):
            raise InventoryError("Brev cleanup inventory omitted an instance identity")
        if item["id"] in identities:
            raise InventoryError("Brev cleanup inventory duplicated an instance identity")
        identities.add(item["id"])
        if item["organizationId"] != lease["organization_id"]:
            raise ProvisionError(
                "Brev cleanup inventory changed organization; refusing confirmation"
            )
        if (item["id"] == lease["instance_id"] and item["name"] != lease["name"]) or (
            item["name"] == lease["name"] and item["id"] != lease["instance_id"]
        ):
            raise ProvisionError(
                "Brev cleanup inventory changed instance identity; refusing confirmation"
            )
    return not any(
        item["id"] == lease["instance_id"] or item["name"] == lease["name"] for item in items
    )


def _select_instance(instance_type: str, disk_gb: int, provider: str, deadline: float) -> dict:
    result = _run(["brev", "search", "--min-disk", str(disk_gb), "--json"], deadline)
    if result.returncode:
        raise ProvisionError(f"Brev catalog search failed (exit {result.returncode})")
    try:
        items = _json(result.stdout)
    except ValueError:
        raise ProvisionError("Brev catalog search returned invalid JSON") from None
    if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
        raise ProvisionError("Brev catalog search must return instance records")
    matching = [item for item in items if item.get("type") == instance_type]
    if len(matching) != 1:
        raise ProvisionError("Brev catalog must contain exactly one requested instance type")
    candidate = matching[0]
    low, high = candidate.get("disk_min_gb"), candidate.get("disk_max_gb")
    if (
        type(low) not in (int, float)
        or type(high) not in (int, float)
        or not math.isfinite(low)
        or not math.isfinite(high)
        or not low <= disk_gb <= high
    ):
        raise ProvisionError("the selected Brev type cannot provide the requested disk")
    if (
        not isinstance(candidate.get("provider"), str)
        or candidate["provider"] not in {"aws", "nebius"}
        or (provider and candidate["provider"] != provider)
    ):
        raise ProvisionError(
            "the requested type and provider do not match the qualification profile"
        )
    return {"type": instance_type, "target_disk_gb": disk_gb, "provider": candidate["provider"]}


def _probe_receipt(output: str, marker: str, min_free_disk_gb: float) -> dict | None:
    pairs = re.findall(r"^TRTMC_PHASE_([a-z_]+)=([0-9]+|not_installed)$", output, re.M)
    if len(pairs) != len({key for key, _value in pairs}):
        return None
    phases = dict(pairs)
    required = {"sudo", "docker", "disk_headroom", "host_gpu", "container_gpu"}
    disks = re.findall(r"^TRTMC_DISK_AVAILABLE_BYTES=([0-9]{1,18})$", output, re.M)
    if (
        output.splitlines().count(marker) != 1
        or any(phases.get(key) != "0" for key in required)
        or len(disks) != 1
    ):
        return None
    available = int(disks[0])
    if available < int(min_free_disk_gb * 1024**3):
        return None
    return {"phases": phases, "disk_available_bytes": available}


def _wait_ready(
    name: str,
    image: str,
    deadline: float,
    start: float,
    *,
    lease: dict,
    lease_file: Path,
    min_free_disk_gb: float,
) -> None:
    previous_state = ""
    refreshed = False
    marker = f"TRTMC_GPU_READY_{secrets.token_hex(16)}"
    while True:
        row = _inventory(name, deadline, start, lease["instance_id"], lease["sku"])
        if row is None:
            _log(start, f"{name}: not yet visible in inventory")
            _pause(deadline)
            continue
        if lease["instance_id"] is None:
            lease.update(instance_id=row["id"], allocation_pending=False, phase="provisioning")
            _save_lease(lease_file, lease)
        state = _state(row)
        if state != previous_state:
            _log(start, f"{name}: {state}")
            previous_state = state
        ready = _ready(row)
        if not ready:
            _log(start, f"{name}: waiting for Brev metadata READY on the same instance")
        if ready and "metadata_ready_elapsed_seconds" not in lease:
            lease["metadata_ready_elapsed_seconds"] = round(time.monotonic() - start, 1)
            _save_lease(lease_file, lease)
        # Leave platform bootstrap alone until the native inventory says READY.
        if ready and not refreshed:
            try:
                result = _run(["brev", "refresh"], deadline)
                refreshed = result.returncode == 0
            except (ProvisionError, OSError):
                _log(start, f"{name}: SSH configuration refresh unavailable; retrying the same VM")
        if ready and refreshed:
            try:
                result = _run(
                    ["brev", "exec", name, _probe(image, marker, min_free_disk_gb)],
                    deadline,
                    PROBE_TIMEOUT,
                )
            except (ProvisionError, OSError):
                _log(start, f"{name}: readiness probe unavailable within its bounded wait")
            else:
                for line in _safe_lines(result.stdout):
                    _log(start, f"{name}: {line}")
                receipt = _probe_receipt(result.stdout, marker, min_free_disk_gb)
                if result.returncode == 0 and receipt is not None:
                    current = _inventory(name, deadline, start, lease["instance_id"], lease["sku"])
                    if current is None:
                        raise ProvisionError("the instance disappeared during its readiness probe")
                    if _ready(current):
                        _remaining(deadline)
                        lease.update(
                            phase="ready",
                            ready_elapsed_seconds=round(time.monotonic() - start, 1),
                            **receipt,
                        )
                        _save_lease(lease_file, lease)
                        _log(
                            start,
                            f"{name}: verified SSH, Docker, host GPU, container GPU and disk headroom",
                        )
                        return
                _log(start, f"{name}: readiness probe has not established the GPU contract")
        _pause(deadline)


def _publish_name(name: str, instance_type: str = "", organization_id: str = "") -> None:
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as stream:
            stream.write(f"instance_name={name}\n")
            if organization_id:
                stream.write(f"organization_id={organization_id}\n")
            if instance_type:
                stream.write(f"instance_type={instance_type}\nallocation_requested=true\n")
            stream.flush()


def _validate_name(name: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,62}", name):
        raise ValueError(
            "instance name must contain only letters, numbers, underscores, dots and hyphens"
        )


def _cleanup(
    name: str,
    deadline: float | None,
    start: float,
    uncertain_create: bool = False,
    *,
    lease: dict | None = None,
    lease_file: Path | None = None,
) -> None:
    del uncertain_create
    accepted = bool(lease and lease.get("delete_accepted"))
    absent = 0
    identity_blocked = bool(lease and lease.get("ownership_blocked"))
    while True:
        if deadline is not None:
            _remaining(deadline)
        delay = POLL_INTERVAL
        try:
            if lease is None or lease.get("create_started") is not True:
                # An arbitrary same-name instance is not evidence of our ownership.
                lease = _load_lease(lease_file, name)
                if lease is None or lease.get("create_started") is not True:
                    raise InventoryError("cleanup is waiting for the original owned-creation lease")
            identity, sku = lease.get("instance_id"), lease.get("sku")
            if not sku:
                raise ProvisionError("the owned Brev lease omitted its exact instance type")
            if identity is None:
                org = lease.get("organization_id")
                if not org:
                    raise InventoryError("the original discovery organization remains unknown")
                if org:
                    try:
                        auth_status, auth_body = _cleanup_request("AUTH", org, deadline)
                    except InventoryError:
                        _log(start, f"{name}: cleanup AUTH HTTP 0")
                        raise
                    _log(start, f"{name}: cleanup AUTH HTTP {auth_status}")
                    if auth_status != 200 or not isinstance(auth_body, dict):
                        raise InventoryError("the original discovery organization is unavailable")
                    if auth_body.get("id") != org:
                        raise ProvisionError(
                            "Brev cleanup organization changed; refusing discovery"
                        )
                request_deadline = (
                    deadline if deadline is not None else time.monotonic() + CLI_TIMEOUT
                )
                row = _instance(name, request_deadline, instance_type=sku)
                if row is None:
                    # A killed create request may still materialize later. Never call this deleted.
                    raise InventoryError("cleanup is reconciling the original uncertain allocation")
                identity = row["id"]
                lease.update(instance_id=identity, allocation_pending=False, phase="cleanup")
                if lease_file is not None:
                    _save_lease(lease_file, lease)
            try:
                status, body = _cleanup_request("GET", identity, deadline)
            except InventoryError:
                status, body = 0, None
            _log(start, f"{name}: cleanup GET HTTP {status}")
            if status == 200:
                try:
                    _cleanup_workspace(body, lease)
                except InventoryError:
                    status, body = 0, None
                else:
                    lease["owned_instance_observed"] = True
                    lease["ownership_blocked"] = False
                    identity_blocked = False
                    absent = 0
                    if lease_file is not None:
                        _save_lease(lease_file, lease)
            if status not in (200, 404):
                _log(
                    start, f"{name}: direct instance read unavailable; requesting owned-ID deletion"
                )
            if identity_blocked:
                raise ProvisionError("cleanup is blocked until the original identity is verified")
            # Prior validated inventory recorded this owned ID. An unavailable fresh GET
            # must not strand cleanup when DELETE succeeded but its acknowledgement was lost.
            owned_lease = bool(
                lease.get("create_started") is True
                and lease.get("instance_id")
                and lease.get("sku")
                and lease.get("allocation_pending") is False
            )
            unknown_response = bool(
                lease.get("organization_id")
                and (lease.get("owned_instance_observed") or owned_lease)
                and lease.get("delete_attempted")
                and lease.get("delete_outcome_unknown")
            )
            # A valid owned lease is sufficient to request deletion even when fresh reads fail.
            if not accepted and (status == 200 or absent == 0):
                lease.update(delete_attempted=True, delete_outcome_unknown=True)
                if lease_file is not None:
                    _save_lease(lease_file, lease)
                try:
                    deleted_status, deleted_body = _cleanup_request("DELETE", identity, deadline)
                except InventoryError:
                    deleted_status, deleted_body = 0, None
                    _log(start, f"{name}: cleanup DELETE HTTP 0")
                else:
                    _log(start, f"{name}: cleanup DELETE HTTP {deleted_status}")
                lease["delete_outcome_unknown"] = (
                    deleted_status in (0, 200, 202, 404) or 500 <= deleted_status <= 599
                )
                if deleted_status in (200, 202):
                    try:
                        _cleanup_workspace(deleted_body, lease)
                    except InventoryError:
                        pass
                    else:
                        accepted = True
                        lease.update(
                            delete_accepted=True, delete_outcome_unknown=False, phase="deleting"
                        )
                        _log(start, f"{name}: owned-ID delete accepted; awaiting verified deletion")
                if lease_file is not None:
                    _save_lease(lease_file, lease)
            unknown_response = bool(
                lease.get("organization_id")
                and (lease.get("owned_instance_observed") or owned_lease)
                and lease.get("delete_attempted")
                and lease.get("delete_outcome_unknown")
            )
            if status != 200 and (accepted or unknown_response):
                org = lease.get("organization_id", "")
                if not org:
                    raise InventoryError("the original allocation organization remains unknown")
                try:
                    auth_status, auth_body = _cleanup_request("AUTH", org, deadline)
                except InventoryError:
                    _log(start, f"{name}: cleanup AUTH HTTP 0")
                    raise
                _log(start, f"{name}: cleanup AUTH HTTP {auth_status}")
                if (
                    auth_status != 200
                    or not isinstance(auth_body, dict)
                    or not isinstance(auth_body.get("id"), str)
                    or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", auth_body["id"])
                ):
                    raise InventoryError("cleanup absence has not been authenticated")
                if org and auth_body["id"] != org:
                    raise ProvisionError("Brev cleanup organization changed; refusing confirmation")
                try:
                    list_status, list_body = _cleanup_request("LIST", org, deadline)
                except InventoryError:
                    _log(start, f"{name}: cleanup LIST HTTP 0")
                    raise
                _log(start, f"{name}: cleanup LIST HTTP {list_status}")
                if list_status != 200:
                    raise InventoryError("the original organization inventory is unavailable")
                is_absent = _cleanup_list_absent(list_body, lease)
                absent = absent + 1 if is_absent else 0
                if not is_absent and not accepted:
                    lease["delete_outcome_unknown"] = False
                    if lease_file is not None:
                        _save_lease(lease_file, lease)
                if absent >= 2:
                    lease.update(phase="deleted", allocation_pending=False, cleanup_confirmed=True)
                    if lease_file is not None:
                        _save_lease(lease_file, lease)
                    _log(
                        start,
                        f"{name}: deletion confirmed by two authenticated organization inventories",
                    )
                    return
                if is_absent:
                    delay = 2.0
            else:
                absent = 0
        except (InventoryError, OSError):
            absent = 0
            _log(start, f"{name}: cleanup remains unconfirmed; retrying the same owned allocation")
        except ProvisionError:
            absent = 0
            identity_blocked = True
            if lease is not None:
                lease["ownership_blocked"] = True
                if lease_file is not None:
                    _save_lease(lease_file, lease)
            if deadline is not None:
                raise
            _log(start, f"{name}: ownership evidence is inconsistent; deletion remains blocked")
        time.sleep(delay if deadline is None else min(delay, _remaining(deadline)))


def cleanup(
    instance: str,
    *,
    lease_file: Path | None = None,
    timeout: float = CLEANUP_TIMEOUT,
    until_deleted: bool = False,
) -> str:
    _validate_name(instance)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("cleanup timeout must be finite and positive")
    start = time.monotonic()
    path = Path(lease_file) if lease_file is not None else None
    while True:
        try:
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
            lease = _load_lease(path, instance)
            break
        except (ProvisionError, OSError):
            if not until_deleted:
                raise
            _log(
                start,
                f"{instance}: original lease unavailable or invalid; deletion remains blocked",
            )
            time.sleep(POLL_INTERVAL)
    if lease is not None and lease.get("create_started") is False:
        if (
            lease.get("instance_id") is not None
            or lease.get("allocation_pending")
            or lease.get("create_accepted")
            or lease.get("delete_accepted")
            or lease.get("delete_attempted")
        ):
            raise ProvisionError("a pre-creation lease contains an inconsistent allocation")
        lease.update(phase="unallocated", cleanup_confirmed=True)
        if path is not None:
            _save_lease(path, lease)
        _log(start, f"{instance}: persisted lease confirms no allocation was requested")
        return instance
    if lease is None and path is not None:
        lease = {
            "schema_version": 1,
            "name": instance,
            "instance_id": None,
            "sku": None,
            "allocation_pending": True,
            "phase": "cleanup_unknown",
        }
        _save_lease(path, lease)
    try:
        _cleanup(
            instance,
            None if until_deleted else start + timeout,
            start,
            lease=lease,
            lease_file=path,
        )
    except (ProvisionError, OSError):
        if lease is not None and path is not None:
            lease.update(phase="cleanup_failed", cleanup_confirmed=False)
            _save_lease(path, lease)
        raise ProvisionError(f"cleanup of {instance} remains unconfirmed") from None
    return instance


def provision(
    instance: str,
    gpu: str = "",
    *,
    timeout: float = 2700,
    provider: str = "",
    instance_type: str = "",
    disk_gb: int = 500,
    min_free_disk_gb: float = 200,
    lease_file: Path | None = None,
    fallback_provider: str = "aws",
    attempts: int = 1,
    probe_image: str = DEFAULT_PROBE_IMAGE,
) -> str:
    """Create exactly one fixed-profile VM; leave its persisted lease for cleanup."""
    _validate_name(instance)
    if not math.isfinite(timeout) or timeout <= 0 or attempts != 1:
        raise ValueError(
            "timeout must be finite and positive; exactly one allocation attempt is supported"
        )
    if (
        type(disk_gb) is not int
        or disk_gb < 500
        or not math.isfinite(min_free_disk_gb)
        or min_free_disk_gb < 200
    ):
        raise ValueError(
            "the qualification profile requires a 500 GiB disk and 200 GiB free headroom"
        )
    if provider not in {"", "aws", "nebius"} or not probe_image or lease_file is None:
        raise ValueError("a supported provider, probe image and persistent lease file are required")
    instance_type = instance_type or (
        NEBIUS_INSTANCE_TYPE if provider == "nebius" else DEFAULT_INSTANCE_TYPE
    )
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", instance_type):
        raise ValueError("one valid exact instance type is required")
    # Legacy GPU/fallback flags remain parseable; the exact SKU is authoritative.
    del gpu, fallback_provider
    path = Path(lease_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ProvisionError("a persisted lease already exists; refusing another allocation")
    start = time.monotonic()
    deadline = start + timeout
    lease = {
        "schema_version": 1,
        "name": instance,
        "sku": instance_type,
        "provider": provider,
        "requested_disk_gb": disk_gb,
        "instance_id": None,
        "phase": "selecting",
        "allocation_pending": False,
        "create_started": False,
        "create_accepted": False,
        "stock_bootstrap": True,
    }
    with path.open("x", encoding="utf-8") as output:
        output.write(json.dumps(lease, indent=2) + "\n")
        output.flush()
    try:
        lease["organization_id"] = _allocation_organization(deadline)
        _save_lease(path, lease)
        if _inventory(instance, deadline, start) is not None:
            raise ProvisionError("instance name already exists; refusing to reuse or delete it")
        candidate = _select_instance(instance_type, disk_gb, provider, deadline)
        lease.update(
            provider=candidate["provider"],
            phase="creating",
            allocation_pending=True,
            create_started=True,
        )
        _save_lease(path, lease)
        _publish_name(instance, instance_type, lease["organization_id"])
        _log(start, f"{instance}: creating exact type {instance_type} with {disk_gb} GiB disk")
        # v0.6.335 reads JSON stdin automatically. Adding --type would prepend a
        # second specification with its default 120 GiB disk. Keep default Jupyter.
        result = _run(
            ["brev", "create", instance, "--detached", "--mode", "vm"],
            deadline,
            PROBE_TIMEOUT,
            input_text=json.dumps([candidate]),
        )
        if result.returncode:
            raise ProvisionError(f"Brev create failed (exit {result.returncode})")
        lease.update(create_accepted=True, phase="provisioning")
        _save_lease(path, lease)
        _wait_ready(
            instance,
            probe_image,
            deadline,
            start,
            lease=lease,
            lease_file=path,
            min_free_disk_gb=min_free_disk_gb,
        )
        return instance
    except (ProvisionError, OSError, KeyboardInterrupt) as error:
        if lease["create_started"] and lease["instance_id"] is None:
            try:
                row = _instance(instance, deadline, instance_type=instance_type)
                if row is not None:
                    lease.update(instance_id=row["id"], allocation_pending=False)
            except (ProvisionError, OSError):
                pass
        lease["phase"] = (
            "interrupted" if isinstance(error, KeyboardInterrupt) else "provision_failed"
        )
        _save_lease(path, lease)
        _log(
            start,
            f"{instance}: provisioning failed; persisted lease retained for confirmed cleanup",
        )
        raise


def provision_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create one exact Brev VM and verify GPU readiness."
    )
    parser.add_argument("--instance", required=True)
    parser.add_argument("--instance-type", "--type", default="")
    parser.add_argument("--disk-gb", "--disk", type=int, default=500)
    parser.add_argument("--min-free-disk-gb", type=float, default=200)
    parser.add_argument("--lease-file", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=2700)
    parser.add_argument("--provider", default="")
    parser.add_argument(
        "--gpu", default="", help="Legacy option; exact instance type governs selection"
    )
    parser.add_argument(
        "--fallback-provider", default="aws", help="Legacy option; allocation is never retried"
    )
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--probe-image", default=DEFAULT_PROBE_IMAGE)
    arguments = parser.parse_args(argv)
    try:
        print(provision(**vars(arguments)))
        return 0
    except (ProvisionError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 75
    except KeyboardInterrupt:
        print(
            "ERROR: GPU provisioning interrupted; cleanup must consume the persisted lease",
            file=sys.stderr,
        )
        return 130


def cleanup_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Delete one owned Brev VM and confirm authenticated absence."
    )
    parser.add_argument("--instance", required=True)
    parser.add_argument("--lease-file", type=Path)
    parser.add_argument("--timeout", type=float, default=CLEANUP_TIMEOUT)
    parser.add_argument(
        "--until-deleted",
        action="store_true",
        help="Retry until deletion is verified, without an overall software deadline.",
    )
    arguments = parser.parse_args(argv)
    try:
        print(cleanup(**vars(arguments)))
        return 0
    except (ProvisionError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 75
    except KeyboardInterrupt:
        print("ERROR: GPU cleanup interrupted; deletion remains unconfirmed", file=sys.stderr)
        return 130


def main(argv: Sequence[str] | None = None) -> int:
    """Keep remote execution compatible and expose provision/confirmed cleanup."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "provision":
        return provision_main(arguments[1:])
    if arguments and arguments[0] == "cleanup":
        return cleanup_main(arguments[1:])
    return execution_main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
