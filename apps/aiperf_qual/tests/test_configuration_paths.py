# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The application reads a family only through the files its configuration names, so each must exist: a family that
renames or removes its reference adapter, inputs or requirements breaks the profiles that name them."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

from trtmc_aiperf_qual.config import CONFIG_ROOT

REPOSITORY = Path(__file__).resolve().parents[3]
ROOTS = ("apps/", "core/", "examples/", "families/")


def _named_paths(value: Any) -> Iterator[str]:
    if isinstance(value, dict):
        for item in value.values():
            yield from _named_paths(item)
    elif isinstance(value, list):
        for item in value:
            yield from _named_paths(item)
    elif isinstance(value, str) and value.startswith(ROOTS):
        yield value


def test_every_repository_path_the_configuration_names_exists():
    named = {
        (path.relative_to(CONFIG_ROOT).as_posix(), value)
        for path in sorted(CONFIG_ROOT.rglob("*.yaml"))
        for value in _named_paths(yaml.safe_load(path.read_text(encoding="utf-8")))
    }
    # Adapters, inputs, preparation scripts and their inputs, requirements, and validation commands.
    assert len(named) > 100
    assert not sorted(item for item in named if not (REPOSITORY / item[1]).exists())
