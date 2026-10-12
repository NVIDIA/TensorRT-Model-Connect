# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

from qualification_tests.benchmark_qualification.performance import reference_protocol


def _command(tmp_path: Path, *, script: Path | None, adapter: str | None) -> list[str]:
    return reference_protocol.command(
        python=Path("/reference/bin/python"),
        generic_runner=Path("/repo/generic_reference.py"),
        script=script,
        family="example",
        operation="generate",
        manifest=tmp_path / "model.json",
        selected_task="text_generation" if script else None,
        testcase_name="smoke",
        adapter=adapter,
        adapter_options={"option": 1},
        timing_contract={"timing_scope": "task-model-call-wall"},
        padding="longest",
        model="owner/model",
        revision="a" * 40,
        request={"prompt": "hello"},
        precision="fp32",
        mode="hf-eager",
        warmup=0,
        iterations=1,
        case_name="example",
        output=tmp_path / "reference.json",
        trust_remote_code=False,
        local_files_only=True,
    )


def test_generic_reference_command_has_no_family_dispatch(tmp_path: Path) -> None:
    command = _command(tmp_path, script=None, adapter="hf-transformers-vision")

    assert command[:2] == ["/reference/bin/python", "/repo/generic_reference.py"]
    assert command[command.index("--adapter") + 1] == "hf-transformers-vision"
    assert "--family" not in command
    assert command[command.index("--testcase-name") + 1] == "smoke"
    assert json.loads(command[command.index("--request-json") + 1]) == {"prompt": "hello"}


def test_family_reference_command_names_only_its_owner(tmp_path: Path) -> None:
    script = tmp_path / "families/example/tests/benchmark/reference.py"
    command = _command(tmp_path, script=script, adapter=None)

    assert command[:2] == ["/reference/bin/python", str(script)]
    assert command[command.index("--family") + 1] == "example"
    assert command[command.index("--selected-task") + 1] == "text_generation"
    assert "--adapter" not in command
    assert "--testcase-name" not in command


def test_reference_command_rejects_ambiguous_implementation(tmp_path: Path) -> None:
    try:
        _command(tmp_path, script=None, adapter=None)
    except ValueError as error:
        assert "exactly one" in str(error)
    else:
        raise AssertionError("ambiguous reference must fail")
