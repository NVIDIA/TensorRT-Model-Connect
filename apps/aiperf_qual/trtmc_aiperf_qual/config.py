# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model, suite, and environment configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

CONFIG_ROOT = Path(__file__).resolve().parent.parent / "config"


class ConfigError(ValueError):
    pass


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"missing configuration file: {path}")
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ConfigError(f"{path} must contain a mapping")
    return value


def require(mapping: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"{where}: missing required field {key!r}")
    return mapping[key]


@dataclass(frozen=True)
class Environment:
    values: Mapping[str, Any]

    def path(self, key: str) -> Path:
        return Path(require(self.values, key, "environment"))

    def __getitem__(self, key: str) -> Any:
        return require(self.values, key, "environment")


def load_environment(path: Path) -> Environment:
    return Environment(_load(path))


def load_suite(name: str, root: Path = CONFIG_ROOT) -> dict[str, Any]:
    suite = _load(root / "suites" / f"{name}.yaml")
    for key in ("suite", "version", "source", "selection"):
        require(suite, key, f"suite {name}")
    return suite
