# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Importing a stablelm submodule first must not go through the plugin.

The family package resolves unknown attributes by importing ``.plugin``. That
hook also runs for ``from . import <submodule>``, so a submodule that imports a
sibling during its own import used to be answered by loading the plugin, which
imports that submodule back while it is still half-built.

Each case runs in a fresh interpreter because the failure only appears when the
submodule is the first thing to touch the package; once anything else has
imported the plugin, the attribute is already set and the hook never runs.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest


_PACKAGE = "tensorrt_model_connect.families.stablelm"


def _run(statement: str) -> subprocess.CompletedProcess:
    """Run one statement in a fresh interpreter that sees the same packages.

    The subprocess inherits this interpreter's search path so it imports the
    checkout under test rather than whatever happens to be installed.
    """
    return subprocess.run(
        [sys.executable, "-c", statement],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)},
    )


@pytest.mark.parametrize(
    "submodule",
    ["default_decoder", "standard_decoder_builder", "default_dual_profile_decoder"],
)
def test_submodule_imports_first_without_the_plugin(submodule: str) -> None:
    result = _run(
        f"import importlib; importlib.import_module('{_PACKAGE}.{submodule}')")
    assert result.returncode == 0, (
        f"importing {submodule} before anything else touches the package failed:\n"
        f"{result.stderr}"
    )


def test_plugin_attribute_is_the_plugin_not_the_module() -> None:
    """The submodule shortcut must not shadow the ``plugin`` attribute.

    ``stablelm.plugin`` is both a module name and the public FamilyPlugin
    instance; the package deliberately exposes the instance.
    """
    result = _run(
        f"import importlib, types;"
        f" m = importlib.import_module('{_PACKAGE}');"
        f" assert not isinstance(m.plugin, types.ModuleType), type(m.plugin)")
    assert result.returncode == 0, result.stderr
