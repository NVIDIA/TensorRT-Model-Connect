# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repro commands must survive a copy into a shell and back out again."""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

import pytest

from tests.e2e_harness import orchestrator
from tests.e2e_harness.contracts import E2ECase, RunContext, StageSpec

AWKWARD_PROMPTS = (
    pytest.param("a prompt with spaces", id="spaces"),
    pytest.param("", id="empty"),
    pytest.param("it's a prompt", id="single-quote"),
    pytest.param('say "hello"', id="double-quote"),
    pytest.param("cost is $PATH and ${HOME}", id="dollar"),
    pytest.param("back\\slash", id="backslash"),
    pytest.param("first line\nsecond line", id="newline"),
    pytest.param("semi; colon && amp | pipe", id="shell-operators"),
    pytest.param("glob* and ?mark and [bracket]", id="glob"),
    pytest.param("tab\there", id="tab"),
    pytest.param("trailing space ", id="trailing-space"),
    pytest.param("#hash comment", id="hash"),
)


def _make_case(name: str = "repro-unit") -> E2ECase:
    return E2ECase(
        name=name,
        hf_id="hf/unit-model",
        family="unit",
        runtime_strategy="unit_runtime",
        task_strategy="unit_task",
        reference_backend="unit_ref",
        bundle=f"{name}.bundle",
        preflight=[],
        stages=[StageSpec(name="generate")],
        determinism={},
    )


def _make_ctx(
    tmp_path: Path,
    case: E2ECase,
    *,
    engine_dir_name: str = "engines",
    binary_path: str = "/tmp/trtmc",
    hf_python: str | None = None,
) -> RunContext:
    engine_dir = tmp_path / engine_dir_name
    engine_dir.mkdir()
    return RunContext(
        case=case,
        artifacts_dir=str(tmp_path / "artifacts"),
        binary_path=binary_path,
        hf_python=hf_python or sys.executable,
        engine_dir=str(engine_dir),
    )


@pytest.mark.parametrize("prompt", AWKWARD_PROMPTS)
def test_trt_inference_prompt_survives_a_round_trip(
    tmp_path: Path,
    prompt: str,
) -> None:
    case = _make_case()
    case.inputs["prompt"] = prompt
    ctx = _make_ctx(tmp_path, case)

    repro = orchestrator._build_repro_commands(case, ctx, "/tmp/unit.bundle", {})
    argv = shlex.split(repro["trt_inference"])

    assert argv[argv.index("--prompt") + 1] == prompt


@pytest.mark.parametrize("prompt", AWKWARD_PROMPTS)
def test_rendered_commands_are_a_single_shell_word_per_token(
    tmp_path: Path,
    prompt: str,
) -> None:
    """A prompt must not split into extra argv entries however it is spelled."""
    case = _make_case()
    case.inputs["prompt"] = prompt
    ctx = _make_ctx(tmp_path, case)

    repro = orchestrator._build_repro_commands(case, ctx, "/tmp/unit.bundle", {})
    argv = shlex.split(repro["trt_inference"])

    assert argv.count("--prompt") == 1
    assert argv[0] == ctx.binary_path
    assert argv[1] == "run"
    assert argv[2] == "/tmp/unit.bundle"


def test_prompts_file_path_with_spaces_round_trips(tmp_path: Path) -> None:
    case = _make_case()
    prompt_file = tmp_path / "a dir with spaces" / "prompts.jsonl"
    case.inputs["prompt_file"] = str(prompt_file)
    ctx = _make_ctx(tmp_path, case)

    repro = orchestrator._build_repro_commands(case, ctx, "/tmp/unit.bundle", {})
    argv = shlex.split(repro["trt_inference"])

    assert argv[argv.index("--prompts-file") + 1] == str(prompt_file)


def test_rerun_test_node_id_round_trips(tmp_path: Path) -> None:
    """The pytest node id carries brackets, which a shell would glob."""
    case = _make_case("case-with-brackets")
    ctx = _make_ctx(tmp_path, case)

    repro = orchestrator._build_repro_commands(case, ctx, "/tmp/unit.bundle", {})
    argv = shlex.split(repro["rerun_test"])

    assert argv[0] == "pytest"
    assert argv[1] == f"tests/test_e2e.py::test_e2e[{case.name}]"


def test_rerun_rebuild_keeps_the_rerun_argv_and_adds_the_flag(
    tmp_path: Path,
) -> None:
    case = _make_case()
    ctx = _make_ctx(tmp_path, case)

    repro = orchestrator._build_repro_commands(case, ctx, "/tmp/unit.bundle", {})

    rerun = shlex.split(repro["rerun_test"])
    rebuild = shlex.split(repro["rerun_test_rebuild"])

    assert rebuild == rerun + ["--rebuild-engines"]


def test_build_bundle_round_trips(tmp_path: Path) -> None:
    case = _make_case()
    ctx = _make_ctx(tmp_path, case)

    repro = orchestrator._build_repro_commands(case, ctx, "/tmp/unit.bundle", {})
    argv = shlex.split(repro["build_bundle"])

    assert argv[0] == sys.executable
    assert "build" in argv
    assert case.hf_id in argv


def test_no_command_carries_pre_quoted_tokens(tmp_path: Path) -> None:
    """Rendering happens once, so no token should arrive already quoted."""
    case = _make_case()
    case.inputs["prompt"] = "plain"
    ctx = _make_ctx(tmp_path, case)

    repro = orchestrator._build_repro_commands(case, ctx, "/tmp/unit.bundle", {})

    for command in repro.values():
        for token in shlex.split(command):
            assert not (token.startswith("'") and token.endswith("'"))
            assert not (token.startswith('"') and token.endswith('"'))


# The tokens below were never quoted at all before rendering moved to
# shlex.join: they were dropped into a " ".join with whatever they contained.
# A path with a space in it therefore came back as two argv entries.


def test_build_bundle_survives_an_engine_dir_with_spaces(tmp_path: Path) -> None:
    case = _make_case()
    ctx = _make_ctx(tmp_path, case, engine_dir_name="engine dir")

    # With no bundle path resolved yet, the target falls back to the engine dir.
    repro = orchestrator._build_repro_commands(case, ctx, None, {})
    argv = shlex.split(repro["build_bundle"])

    expected = str(Path(ctx.engine_dir) / case.bundle)
    assert argv[argv.index("-o") + 1] == expected


def test_rerun_test_survives_an_engine_dir_with_spaces(tmp_path: Path) -> None:
    case = _make_case()
    ctx = _make_ctx(tmp_path, case, engine_dir_name="engine dir")

    repro = orchestrator._build_repro_commands(case, ctx, "/tmp/unit.bundle", {})
    argv = shlex.split(repro["rerun_test"])

    assert argv[argv.index("--engine-dir") + 1] == ctx.engine_dir


def test_rerun_test_survives_a_binary_path_with_spaces(tmp_path: Path) -> None:
    case = _make_case()
    ctx = _make_ctx(tmp_path, case, binary_path="/opt/my tools/trtmc")

    repro = orchestrator._build_repro_commands(case, ctx, "/tmp/unit.bundle", {})
    argv = shlex.split(repro["rerun_test"])

    assert argv[argv.index("--trtmc-binary") + 1] == "/opt/my tools/trtmc"


def test_rerun_test_survives_an_hf_python_with_spaces(tmp_path: Path) -> None:
    case = _make_case()
    ctx = _make_ctx(tmp_path, case, hf_python="/opt/py envs/bin/python")

    repro = orchestrator._build_repro_commands(case, ctx, "/tmp/unit.bundle", {})
    argv = shlex.split(repro["rerun_test"])

    assert argv[argv.index("--hf-python") + 1] == "/opt/py envs/bin/python"


def test_trt_inference_survives_a_binary_path_with_spaces(tmp_path: Path) -> None:
    case = _make_case()
    case.inputs["prompt"] = "hello"
    ctx = _make_ctx(tmp_path, case, binary_path="/opt/my tools/trtmc")

    repro = orchestrator._build_repro_commands(case, ctx, "/tmp/unit.bundle", {})
    argv = shlex.split(repro["trt_inference"])

    assert argv[0] == "/opt/my tools/trtmc"


def test_bundle_path_with_spaces_round_trips(tmp_path: Path) -> None:
    case = _make_case()
    case.inputs["prompt"] = "hello"
    ctx = _make_ctx(tmp_path, case)

    repro = orchestrator._build_repro_commands(
        case, ctx, "/tmp/bundle dir/unit.bundle", {}
    )
    argv = shlex.split(repro["trt_inference"])

    assert argv[2] == "/tmp/bundle dir/unit.bundle"
