# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import json
import re
import sys
import time
from pathlib import Path

import pytest

from tools import launch_ranks

TAGGED = re.compile(r"^\[1,(\d+)\]<(stdout|stderr)>:(.*)$")


def _child(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def _run(command, world_size, **kwargs):
    out, err = io.StringIO(), io.StringIO()
    status = launch_ranks.launch(command, world_size, stdout=out, stderr=err, **kwargs)
    return status, out.getvalue(), err.getvalue()


def _rank_lines(text: str, stream: str) -> dict[int, list[str]]:
    lines: dict[int, list[str]] = {}
    for line in text.splitlines():
        match = TAGGED.fullmatch(line)
        if match and match.group(2) == stream:
            lines.setdefault(int(match.group(1)), []).append(match.group(3))
    return lines


def test_rank_environments_follow_the_openmpi_contract(tmp_path: Path) -> None:
    base = {"KEEP": "1", "OMPI_COMM_WORLD_RANK": "stale"}
    rendezvous = tmp_path / "id.bin"
    envs = launch_ranks.rank_environments(
        base, 2, rendezvous, gpus=["3", "5"], nccl_library="/opt/nccl/libnccl.so.2"
    )
    assert base == {"KEEP": "1", "OMPI_COMM_WORLD_RANK": "stale"}
    assert [env["OMPI_COMM_WORLD_RANK"] for env in envs] == ["0", "1"]
    assert [env["OMPI_COMM_WORLD_LOCAL_RANK"] for env in envs] == ["0", "1"]
    for env in envs:
        assert env["OMPI_COMM_WORLD_SIZE"] == "2"
        assert env["OMPI_COMM_WORLD_LOCAL_SIZE"] == "2"
        # Every rank sees every selected GPU and picks one by local rank, as under mpirun.
        assert env["CUDA_VISIBLE_DEVICES"] == "3,5"
        assert env["TRTMC_NCCL_RENDEZVOUS"] == str(rendezvous)
        assert env["TRTMC_NCCL_LIBRARY"] == "/opt/nccl/libnccl.so.2"
        assert env["KEEP"] == "1"
    assert set(launch_ranks.RANK_ENV_NAMES) <= set(envs[0])


def test_rank_environments_keep_visibility_and_nccl_when_not_requested(tmp_path: Path) -> None:
    base = {"CUDA_VISIBLE_DEVICES": "7", "TRTMC_NCCL_LIBRARY": "custom"}
    (env,) = launch_ranks.rank_environments(base, 1, tmp_path / "id.bin")
    assert env["CUDA_VISIBLE_DEVICES"] == "7"
    assert env["TRTMC_NCCL_LIBRARY"] == "custom"


def test_library_dirs_prepend_to_the_platform_search_path(tmp_path: Path) -> None:
    (linux,) = launch_ranks.rank_environments(
        {"LD_LIBRARY_PATH": "/usr/lib"},
        1,
        tmp_path / "id",
        library_dirs=["/a", "/b"],
        platform="linux",
    )
    assert linux["LD_LIBRARY_PATH"] == "/a:/b:/usr/lib"
    # Windows environment names are case-insensitive: extend the existing "Path" entry.
    (windows,) = launch_ranks.rank_environments(
        {"Path": r"C:\Windows"},
        1,
        tmp_path / "id",
        library_dirs=[r"C:\nccl\bin"],
        platform="win32",
    )
    assert windows == {**windows, "Path": r"C:\nccl\bin;C:\Windows"}
    assert "PATH" not in windows
    (empty,) = launch_ranks.rank_environments(
        {}, 1, tmp_path / "id", library_dirs=[r"C:\x"], platform="win32"
    )
    assert empty["PATH"] == r"C:\x"


@pytest.mark.parametrize(
    ("value", "world_size", "expected"),
    [(None, 2, None), ("", 2, None), ("0,1", 2, ["0", "1"]), (" 1 , 0 ,2", 2, ["1", "0", "2"])],
)
def test_parse_gpus_accepts_valid_lists(value, world_size, expected) -> None:
    assert launch_ranks.parse_gpus(value, world_size) == expected


@pytest.mark.parametrize("value", ["0", "0,,1", "0,0", ","])
def test_parse_gpus_rejects_invalid_lists(value) -> None:
    with pytest.raises(ValueError):
        launch_ranks.parse_gpus(value, 2)


def test_each_launch_gets_a_fresh_rendezvous_path(tmp_path: Path) -> None:
    first = launch_ranks.new_rendezvous_path(tmp_path)
    second = launch_ranks.new_rendezvous_path(tmp_path)
    assert first != second
    assert first.parent == tmp_path and not first.exists()
    assert first.name.startswith("trtmc_nccl_") and first.suffix == ".bin"


def test_library_path_variable() -> None:
    assert launch_ranks.library_path_variable("win32") == "PATH"
    assert launch_ranks.library_path_variable("linux") == "LD_LIBRARY_PATH"


def test_launch_tags_output_and_passes_rank_environment(tmp_path: Path) -> None:
    code = (
        "import json, os, sys\n"
        "keys = ['OMPI_COMM_WORLD_SIZE', 'OMPI_COMM_WORLD_RANK', 'OMPI_COMM_WORLD_LOCAL_RANK',"
        " 'CUDA_VISIBLE_DEVICES', 'TRTMC_NCCL_RENDEZVOUS']\n"
        "print(json.dumps({k: os.environ.get(k) for k in keys}))\n"
        "print('rank-stderr', os.environ['OMPI_COMM_WORLD_RANK'], file=sys.stderr)\n"
    )
    status, out, err = _run(_child(code), 3, gpus=["0", "1", "2"], rendezvous_dir=tmp_path)
    assert status == 0, err
    stdout_lines = _rank_lines(out, "stdout")
    assert sorted(stdout_lines) == [0, 1, 2]
    payloads = {rank: json.loads(lines[0]) for rank, lines in stdout_lines.items()}
    for rank, payload in payloads.items():
        assert payload["OMPI_COMM_WORLD_RANK"] == str(rank)
        assert payload["OMPI_COMM_WORLD_LOCAL_RANK"] == str(rank)
        assert payload["OMPI_COMM_WORLD_SIZE"] == "3"
        assert payload["CUDA_VISIBLE_DEVICES"] == "0,1,2"
    rendezvous = {payload["TRTMC_NCCL_RENDEZVOUS"] for payload in payloads.values()}
    assert len(rendezvous) == 1
    (path,) = rendezvous
    assert Path(path).parent == tmp_path
    assert not Path(path).exists()
    assert _rank_lines(err, "stderr") == {
        0: ["rank-stderr 0"],
        1: ["rank-stderr 1"],
        2: ["rank-stderr 2"],
    }


def test_launch_supports_the_file_rendezvous_contract(tmp_path: Path) -> None:
    # Mirrors the family runtimes: rank 0 writes the 128-byte id to "<path>.tmp"
    # and renames it; other ranks poll for the file and read it.
    code = (
        "import os, sys, time\n"
        "path = os.environ['TRTMC_NCCL_RENDEZVOUS']\n"
        "rank = int(os.environ['OMPI_COMM_WORLD_RANK'])\n"
        "if rank == 0:\n"
        "    time.sleep(0.3)\n"
        "    data = os.urandom(128)\n"
        "    open(path + '.tmp', 'wb').write(data)\n"
        "    os.replace(path + '.tmp', path)\n"
        "else:\n"
        "    deadline = time.time() + 30\n"
        "    while not os.path.exists(path):\n"
        "        assert time.time() < deadline\n"
        "        time.sleep(0.02)\n"
        "    data = open(path, 'rb').read()\n"
        "print(data.hex())\n"
    )
    status, out, err = _run(_child(code), 2, rendezvous_dir=tmp_path)
    assert status == 0, err
    lines = _rank_lines(out, "stdout")
    assert len(lines[0][0]) == 256
    assert lines[0] == lines[1]
    assert list(tmp_path.iterdir()) == []


def test_a_failing_rank_terminates_the_others_and_sets_the_status(tmp_path: Path) -> None:
    code = (
        "import os, sys, time\n"
        "if os.environ['OMPI_COMM_WORLD_RANK'] == '1':\n"
        "    print('boom', file=sys.stderr)\n"
        "    sys.exit(3)\n"
        "time.sleep(120)\n"
    )
    start = time.monotonic()
    status, _out, err = _run(_child(code), 2, rendezvous_dir=tmp_path, grace_s=5)
    assert status == 3
    assert time.monotonic() - start < 60
    assert "[1,1]<stderr>:boom" in err
    assert "rank 1 exited with status 3" in err


def test_timeout_terminates_every_rank(tmp_path: Path) -> None:
    status, _out, err = _run(
        _child("import time; time.sleep(120)"),
        2,
        rendezvous_dir=tmp_path,
        timeout_s=1,
        grace_s=5,
    )
    assert status == 124
    assert "timed out" in err


def test_untagged_output(tmp_path: Path) -> None:
    status, out, _err = _run(_child("print('plain')"), 1, rendezvous_dir=tmp_path, tagged=False)
    assert status == 0
    assert out == "plain\n"


def test_cli_runs_the_command_after_double_dash(tmp_path: Path, capsys) -> None:
    status = launch_ranks.main(
        [
            "-n",
            "2",
            "--rendezvous-dir",
            str(tmp_path),
            "--",
            sys.executable,
            "-c",
            "import os; print(os.environ['OMPI_COMM_WORLD_RANK'])",
        ]
    )
    assert status == 0
    out = capsys.readouterr().out
    assert sorted(out.splitlines()) == ["[1,0]<stdout>:0", "[1,1]<stdout>:1"]


def test_cli_rejects_too_few_gpus(capsys) -> None:
    with pytest.raises(SystemExit) as error:
        launch_ranks.main(["-n", "2", "--gpus", "0", "--", sys.executable, "-c", "pass"])
    assert error.value.code == 2
    assert "1 device(s) for 2 ranks" in capsys.readouterr().err
