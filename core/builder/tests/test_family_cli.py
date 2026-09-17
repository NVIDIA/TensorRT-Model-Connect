# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from tensorrt_model_connect import family_cli


@pytest.fixture
def declared(tmp_path, monkeypatch):
    command = {
        "name": "execute", "executor": "python", "handler": "cli:execute",
        "arguments": [
            {"name": "source", "type": "path"},
            {"name": "count", "flags": ["-n", "--count"], "type": "int", "default": 2},
            {"name": "value", "flags": ["--value"], "type": "float"},
            {"name": "active", "flags": ["--active"], "type": "bool"},
            {"name": "verbose", "flags": ["-v", "--verbose"], "type": "bool", "action": "store_true"},
            {"name": "layers", "flags": ["--layer"], "type": "int", "action": "append", "default": []},
        ],
    }
    document = {"version": 1, "commands": [command]}
    path = tmp_path / "alpha" / "cli.json"
    path.parent.mkdir()
    path.write_text(json.dumps(document))
    monkeypatch.setattr(family_cli, "_root", lambda: tmp_path)
    calls = []
    monkeypatch.setitem(sys.modules, "families.alpha.cli", SimpleNamespace(execute=lambda **values: calls.append(values) or 7))
    return document, path, calls


def test_help_and_discovery_do_not_import_handlers(declared, monkeypatch, capsys):
    monkeypatch.setattr(family_cli.importlib, "import_module", lambda name: pytest.fail(f"imported {name}"))
    assert list(family_cli.discover()) == ["alpha"]
    assert family_cli.load_family_cli("absent") is None
    with pytest.raises(SystemExit) as error:
        family_cli.main(["alpha", "execute", "--help"])
    assert error.value.code == 0
    assert "--layer" in capsys.readouterr().out


def test_selected_handler_receives_declared_typed_values(declared):
    _, _, calls = declared
    assert family_cli.main(["alpha", "execute", "input", "--count", "0", "--active", "false", "--layer", "1", "--layer", "2"]) == 7
    assert calls == [{"source": Path("input"), "count": 0, "active": False, "layers": [1, 2]}]
    assert "value" not in calls[0]
    assert "verbose" not in calls[0]


def test_new_owner_command_and_option_need_only_owner_files(declared, monkeypatch):
    _, path, original_calls = declared
    owner = path.parent.parent / "beta"
    owner.mkdir()
    command = {
        "name": "compose", "executor": "python", "handler": "actions:combine",
        "arguments": [
            {"name": "policy", "flags": ["--owner-policy"], "type": "string",
             "choices": ["exact", "fast"], "default": "exact"},
        ],
    }
    (owner / "cli.json").write_text(json.dumps({"version": 1, "commands": [command]}))
    calls = []
    monkeypatch.setitem(sys.modules, "families.beta.actions", SimpleNamespace(
        combine=lambda **values: calls.append(values) or 19,
    ))

    assert family_cli.main(["beta", "compose"]) == 19
    assert family_cli.main(["beta", "compose", "--owner-policy", "fast"]) == 19
    assert calls == [{"policy": "exact"}, {"policy": "fast"}]
    assert original_calls == []
    with pytest.raises(SystemExit):
        family_cli.main(["beta", "compose", "--count", "1"])
    assert len(calls) == 2
    assert family_cli.main(["alpha", "execute", "input"]) == 7
    assert original_calls[-1]["count"] == 2


@pytest.mark.parametrize("arguments", [
    ["--missing", "1"], ["--cou", "1"], ["--count", "1", "--count", "2"],
    ["--verbose", "--verbose"], ["--active", "yes"],
    ["-n3"], ["-vv"], ["--count", "1_0"], ["--count", "１２"],
])
def test_invalid_arguments_never_invoke_owner(declared, arguments):
    with pytest.raises(SystemExit):
        family_cli.main(["alpha", "execute", "input", *arguments])
    assert declared[2] == []


@pytest.mark.parametrize("arguments", [
    ["--count", str(1 << 63)], ["--count", str(-(1 << 63) - 1)], ["--value", "nan"], ["--value", "inf"],
])
def test_numeric_boundary_errors_never_invoke_owner(declared, arguments):
    with pytest.raises((ValueError, SystemExit)):
        family_cli.main(["alpha", "execute", "input", *arguments])
    assert declared[2] == []


def test_serialization_round_trip_preserves_false_zero_repeat_and_leading_dash(declared):
    document, _, calls = declared
    values = {"source": Path("--input"), "count": 0, "active": False, "layers": [0, 2], "value": 0.0}
    argv = family_cli.serialize_arguments(document["commands"][0], values)
    assert family_cli.main(["alpha", "execute", *argv]) == 7
    assert calls[-1] == values
    with pytest.raises(ValueError, match="unknown"):
        family_cli.serialize_arguments(document["commands"][0], {**values, "typo": 2})
    with pytest.raises(ValueError, match="required"):
        family_cli.serialize_arguments(document["commands"][0], {})
    with pytest.raises(ValueError):
        family_cli.serialize_arguments(document["commands"][0], {**values, "count": True})


@pytest.mark.parametrize("mutation", [
    lambda d: d.update(version=True), lambda d: d.update(unknown=1),
    lambda d: d["commands"].append(deepcopy(d["commands"][0])),
    lambda d: d["commands"][0].update(handler="../other:run"),
    lambda d: d["commands"][0].update(executor=[]),
    lambda d: d["commands"][0]["arguments"][0].pop("type"),
    lambda d: d["commands"][0]["arguments"][1].update(default=None),
    lambda d: d["commands"][0]["arguments"][1].update(default=1 << 63),
    lambda d: d["commands"][0]["arguments"][1].update(choices=[1, 1]),
    lambda d: d["commands"][0]["arguments"][1].update(flags=["--help"]),
    lambda d: d["commands"][0]["arguments"][1].update(flags=["--count", "--count"]),
    lambda d: d["commands"][0]["arguments"][1].update(unknown=1),
])
def test_malformed_declarations_fail_closed(declared, mutation):
    document, path, calls = declared
    mutation(document)
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="invalid family CLI declaration"):
        family_cli.discover()
    assert calls == []


def test_missing_or_failing_handler_never_falls_back(declared, monkeypatch):
    monkeypatch.setitem(sys.modules, "families.alpha.cli", SimpleNamespace())
    with pytest.raises(ValueError, match="does not provide handler"):
        family_cli.main(["alpha", "execute", "input"])
    error = RuntimeError("owner failed")
    def fail(**values):
        raise error
    monkeypatch.setitem(sys.modules, "families.alpha.cli", SimpleNamespace(execute=fail))
    with pytest.raises(RuntimeError) as caught:
        family_cli.main(["alpha", "execute", "input"])
    assert caught.value is error


def test_native_dispatch_preserves_argv_without_injected_options(declared, monkeypatch):
    document, path, calls = declared
    document["commands"][0].update(executor="native", handler="execute")
    path.write_text(json.dumps(document))
    class Replaced(BaseException):
        pass
    executions = []
    def execute(path, argv):
        executions.append((path, argv))
        raise Replaced
    monkeypatch.setattr(family_cli.os, "execv", execute)
    arguments = ["alpha", "execute", "input", "--count", "0"]
    with pytest.raises(Replaced):
        family_cli.main(arguments)
    assert executions[0][1] == [executions[0][0], *arguments]
    assert calls == []


def test_append_defaults_round_trip_without_repeating_the_prefix(declared):
    document, path, calls = declared
    command = document["commands"][0]
    command["arguments"] = [{"name": "tags", "flags": ["--tag"], "type": "string", "action": "append", "default": ["base"]}]
    path.write_text(json.dumps(document))
    assert family_cli.main(["alpha", "execute", "--tag", "one"]) == 7
    assert calls[-1] == {"tags": ["base", "one"]}
    argv = family_cli.serialize_arguments(command, calls[-1])
    assert argv == ["--tag=one"]
    assert family_cli.main(["alpha", "execute", *argv]) == 7
    assert calls[-1] == {"tags": ["base", "one"]}
    with pytest.raises(ValueError, match="append default"):
        family_cli.serialize_arguments(command, {"tags": ["replacement"]})


def test_optional_numeric_positional_can_be_omitted_or_interspersed_with_options(declared):
    document, path, calls = declared
    command = document["commands"][0]
    command["arguments"] = [{"name": "first", "type": "string"}, {"name": "count", "type": "int", "required": False}, {"name": "flag", "flags": ["--flag"], "type": "string"}]
    path.write_text(json.dumps(document))
    for values in ({"first": "--help"}, {"count": 0, "first": "value"}):
        argv = family_cli.serialize_arguments(command, values)
        assert family_cli.main(["alpha", "execute", *argv]) == 7
        assert calls[-1] == values
    assert family_cli.main(["alpha", "execute", "value", "--flag", "present", "4"]) == 7
    assert calls[-1] == {"first": "value", "flag": "present", "count": 4}
    command["arguments"] = [command["arguments"][1], command["arguments"][2], command["arguments"][0]]
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="must precede"):
        family_cli.load_family_cli("alpha")


def test_serializer_rejects_unrepresentable_positional_gap_and_false(declared):
    document, _, _ = declared
    command = document["commands"][0]
    command["arguments"] = [{"name": "first", "type": "string", "required": False}, {"name": "second", "type": "string", "required": False}]
    with pytest.raises(ValueError, match="positional"):
        family_cli.serialize_arguments(command, {"second": "value"})
    command["arguments"] = [{"name": "active", "type": "bool", "flags": ["--active"], "action": "store_true"}]
    with pytest.raises(ValueError, match="cannot represent false"):
        family_cli.serialize_arguments(command, {"active": False})


def test_unrelated_malformed_owner_does_not_block_selected_or_legacy_routes(declared, monkeypatch):
    from tensorrt_model_connect import __main__ as launcher
    from tensorrt_model_connect import build_cli

    _, path, _ = declared
    broken = path.parent.parent / "broken"
    broken.mkdir()
    (broken / "cli.json").write_text('{"version":false}')
    assert family_cli.main(["alpha", "execute", "input"]) == 7
    assert launcher.main(["alpha", "execute", "input"]) == 7
    monkeypatch.setattr(build_cli, "main", lambda argv: 9)
    assert launcher.main(["build", "model", "-o", "out"]) == 9
    class Replaced(BaseException):
        pass
    def execute(path, argv):
        assert argv == [path, "version"]
        raise Replaced
    monkeypatch.setattr(launcher.os, "execv", execute)
    with pytest.raises(Replaced):
        launcher.main(["version"])
    with pytest.raises(ValueError, match="invalid family CLI declaration"):
        launcher.main(["broken", "--help"])
    with pytest.raises(ValueError, match="invalid family CLI declaration"):
        launcher.main(["--help"])


def test_help_reports_declared_defaults_and_preserves_literal_terminator(declared, capsys):
    document, path, calls = declared
    command = document["commands"][0]
    command["arguments"].append({"name": "optional", "type": "int", "required": False})
    path.write_text(json.dumps(document))
    with pytest.raises(SystemExit) as error:
        family_cli.main(["alpha", "execute", "--typo", "--help"])
    assert error.value.code == 0
    text = capsys.readouterr().out
    assert "default: 2" in text and "Value of type int" in text
    assert "object at" not in text and "default: None" not in text
    assert family_cli.main(["alpha", "execute", "--", "--help"]) == 7
    assert calls[-1]["source"] == Path("--help")
    assert "optional" not in calls[-1]


@pytest.mark.parametrize("arguments", [["--value", "--active"], ["--value", "--typo"]])
def test_missing_option_value_never_swallows_a_following_option(declared, arguments):
    with pytest.raises(SystemExit):
        family_cli.main(["alpha", "execute", "input", *arguments])
    assert declared[2] == []


def test_leading_dash_values_respect_declared_numeric_type(declared):
    document, path, calls = declared
    document["commands"][0]["arguments"].append({"name": "label", "flags": ["--label"], "type": "string"})
    path.write_text(json.dumps(document))
    assert family_cli.main(["alpha", "execute", "input", "--count", "-2", "--value", "-1e2", "--label=-1"]) == 7
    assert calls[-1]["count"] == -2 and calls[-1]["value"] == -100.0 and calls[-1]["label"] == "-1"
    for arguments in (["--label", "-1"], ["--count", "-2x"]):
        with pytest.raises(SystemExit):
            family_cli.main(["alpha", "execute", "input", *arguments])


def test_path_defaults_and_choices_keep_spelling_until_owner_invocation(declared):
    document, path, calls = declared
    command = document["commands"][0]
    command["arguments"] = [
        {"name": "seed", "flags": ["--seed"], "type": "path", "choices": ["./seed"], "default": "./seed"},
        {"name": "paths", "flags": ["--path"], "type": "path", "action": "append", "choices": ["./seed", "link/../target"], "default": ["./seed"]},
    ]
    path.write_text(json.dumps(document))
    assert family_cli.main(["alpha", "execute"]) == 7
    assert calls[-1] == {"seed": Path("seed"), "paths": [Path("seed")]}
    values = {"seed": Path("./seed"), "paths": [Path("./seed"), Path("link/../target")]}
    argv = family_cli.serialize_arguments(command, values)
    assert argv == ["--seed=./seed", "--path=link/../target"]
    assert family_cli.main(["alpha", "execute", *argv]) == 7
    assert calls[-1] == values
    with pytest.raises(ValueError):
        family_cli.serialize_arguments(command, {"seed": "seed"})
    with pytest.raises(ValueError):
        family_cli.serialize_arguments(command, {"paths": [Path("seed"), Path("target")]})
