# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The family's command declaration (cli.json), without a model or a GPU."""

from __future__ import annotations

import pytest

from tensorrt_model_connect import family_cli

FAMILY = "nemotron3_diarization"
HEAVY = {"torch", "tensorrt", "transformers", "huggingface_hub"}


@pytest.fixture
def no_model_imports(monkeypatch):
    """Fail any import of the family's modules or of a heavy dependency."""
    original = family_cli.importlib.import_module

    def guarded(name, *args, **kwargs):
        assert not name.startswith(f"families.{FAMILY}.") and name not in HEAVY, name
        return original(name, *args, **kwargs)

    monkeypatch.setattr(family_cli.importlib, "import_module", guarded)


@pytest.fixture
def executed(monkeypatch):
    """Record the native hand-off instead of replacing the test process."""
    calls = []

    def execv(path, argv):
        calls.append(argv[1:])
        raise SystemExit(0)

    monkeypatch.setattr(family_cli.os, "execv", execv)
    return calls


def test_declares_one_native_diarize_command():
    (command,) = family_cli.load_family_cli(FAMILY)["commands"]
    assert (command["name"], command["executor"], command["handler"]) == (
        "diarize", "native", "diarize")
    arguments = {argument["name"]: argument for argument in command["arguments"]}
    assert set(arguments) == {"bundle", "audio", "runtime_root", "threshold"}
    assert arguments["audio"]["required"] is True
    # Without a declared default, an absent threshold keeps the family's own.
    assert arguments["threshold"]["type"] == "float" and "default" not in arguments["threshold"]


def test_help_loads_no_model(no_model_imports, capsys):
    with pytest.raises(SystemExit) as caught:
        family_cli.main([FAMILY, "diarize", "--help"])
    assert caught.value.code == 0
    help_text = capsys.readouterr().out
    assert "--audio" in help_text and "--threshold" in help_text


def test_a_valid_call_reaches_the_native_handler(no_model_imports, executed):
    argv = [FAMILY, "diarize", "model.bundle", "--audio", "speech.wav", "--threshold", "0.3"]
    with pytest.raises(SystemExit) as caught:
        family_cli.main(argv)
    assert caught.value.code == 0 and executed == [argv]


@pytest.mark.parametrize("arguments", [
    ["model.bundle"],
    ["model.bundle", "--audio", "speech.wav", "--threshold", "nan"],
    ["model.bundle", "--audio", "speech.wav", "--threshold", "0.3", "--threshold", "0.4"],
    ["model.bundle", "--audio", "speech.wav", "--set", "threshold=0.3"],
])
def test_invalid_calls_are_refused_before_the_native_handler(
        no_model_imports, executed, arguments):
    with pytest.raises(SystemExit) as caught:
        family_cli.main([FAMILY, "diarize", *arguments])
    assert caught.value.code == 2 and executed == []
