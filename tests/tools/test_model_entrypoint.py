# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed tests for model-owned tool entrypoint discovery."""

from pathlib import Path

import pytest

from tools import model_entrypoint


def _owner(tmp_path: Path, descriptor: str) -> Path:
    owner = tmp_path / "demo"
    owner.mkdir()
    (owner / "MODEL.toml").write_text(descriptor, encoding="utf-8")
    return owner


def test_owner_without_declaration_uses_generic_path(tmp_path, monkeypatch) -> None:
    _owner(tmp_path, 'id = "demo"\n')
    monkeypatch.setattr(model_entrypoint, "MODELS_ROOT", tmp_path)

    assert model_entrypoint.load_model_entrypoint("demo", "reference_entrypoint") is None


def test_unknown_owner_fails_closed(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(model_entrypoint, "MODELS_ROOT", tmp_path)

    with pytest.raises(ValueError, match="unknown model owner"):
        model_entrypoint.load_model_entrypoint("unknown", "reference_entrypoint")


def test_missing_declared_entrypoint_fails_closed(tmp_path, monkeypatch) -> None:
    _owner(
        tmp_path,
        'id = "demo"\nreference_entrypoint = "tools/reference.py|run"\n',
    )
    monkeypatch.setattr(model_entrypoint, "MODELS_ROOT", tmp_path)

    with pytest.raises(FileNotFoundError, match="not a regular file"):
        model_entrypoint.load_model_entrypoint("demo", "reference_entrypoint")
