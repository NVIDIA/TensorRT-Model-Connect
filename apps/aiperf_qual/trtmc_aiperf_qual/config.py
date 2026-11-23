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

    def media_root(self) -> Path | None:
        """The media store of this machine, shared by every model, side and run: ``media_root``, else a folder of
        the datasets cache. Each input image, clip and frame is kept once, named by its digest; requests,
        AIPerf inputs and raw exports carry the digest only."""
        if self.values.get("media_root"):
            return Path(self.values["media_root"])
        cache = self.values.get("hf_datasets_cache")
        return Path(cache) / "trtmc-media" if cache else None


def load_environment(path: Path, root: Path = CONFIG_ROOT) -> Environment:
    values = _load(path)
    if "ledger" in values:  # a machine's file names no run's records
        raise ConfigError(f"{path} names a ledger; pass a run's ledger with run-all --ledger")
    if values.get("platform") is not None:
        values = with_platform(values, root)
    return Environment(values)


PLATFORM_KEYS = {"models", "deadlines"}


def platforms(root: Path = CONFIG_ROOT) -> list[str]:
    """The platforms with defaults (``config/platforms/<name>.yaml``)."""
    return sorted(path.stem for path in (root / "platforms").glob("*.yaml"))


def with_platform(values: Mapping[str, Any], root: Path = CONFIG_ROOT) -> dict[str, Any]:
    """The environment with its platform's defaults: the models the platform cannot run follow the machine's own
    exclusions (their reasons name the platform), and the machine's deadlines override the platform's."""
    name = values["platform"]
    known = platforms(root)
    if not isinstance(name, str) or name not in known:
        raise ConfigError(f"unknown platform {name!r}; config/platforms has: {', '.join(known) or 'none'}")
    where = f"platform {name}"
    platform = _load(root / "platforms" / f"{name}.yaml")
    if set(platform) - PLATFORM_KEYS:
        raise ConfigError(f"{where}: unknown keys {', '.join(sorted(set(platform) - PLATFORM_KEYS))} "
                          f"(allowed: {', '.join(sorted(PLATFORM_KEYS))})")
    defaults = platform.get("models") or {}
    if not isinstance(defaults, Mapping) or set(defaults) - {"exclude"}:
        raise ConfigError(f"{where}: models may only hold exclude")
    rules = defaults.get("exclude") or []
    if not isinstance(rules, list) or not all(
            isinstance(rule, Mapping) and set(rule) == {"profile", "reason"} and isinstance(rule["profile"], str)
            and rule["profile"] and isinstance(rule["reason"], str) and rule["reason"].strip() for rule in rules):
        raise ConfigError(f"{where}: models.exclude must be a list of {{profile, reason}}")
    deadlines = platform.get("deadlines") or {}
    if not isinstance(deadlines, Mapping) or not all(
            isinstance(seconds, int) and not isinstance(seconds, bool) and seconds > 0 for seconds in deadlines.values()):
        raise ConfigError(f"{where}: deadlines must map profiles to positive seconds")
    machine = values.get("models") or {}
    own = (machine.get("exclude") or []) if isinstance(machine, Mapping) else None
    own_deadlines = values.get("deadlines") or {}
    if not isinstance(own, list) or not isinstance(own_deadlines, Mapping):
        raise ConfigError("environment: models.exclude must be a list and deadlines a mapping")
    exclude = [*own, *({"profile": rule["profile"], "reason": f"{where}: {rule['reason'].strip()}"} for rule in rules)]
    return {**values, "models": {**machine, "exclude": exclude}, "deadlines": {**deadlines, **own_deadlines}}


def load_suite(name: str, root: Path = CONFIG_ROOT) -> dict[str, Any]:
    suite = _load(root / "suites" / f"{name}.yaml")
    for key in ("suite", "version", "source", "selection"):
        require(suite, key, f"suite {name}")
    return suite
