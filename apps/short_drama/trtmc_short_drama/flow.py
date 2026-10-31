# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The flow: ordered steps, the model of each and its parameters, saved as numbered versions.

Steps are plain functions. Core steps live in `steps/`; a plugin is one Python file in the
plugins directory that defines `STEP = Step(...)`. Reloading picks up new files, and deleting a
file removes its step: a flow version that names a missing step skips it and says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
import json
import logging
from pathlib import Path
import re
import threading
from typing import Any, Callable

from .store import Project, read_json, write_json

log = logging.getLogger(__name__)

FLOW_NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,39}")


@dataclass
class Context:
    """What a step may touch while it runs."""

    project: Project
    params: dict[str, Any]
    model: str | None
    episode: int | None
    services: Any
    progress: Callable[[str], None] = print
    cancelled: threading.Event = field(default_factory=threading.Event)

    def check(self) -> None:
        if self.cancelled.is_set():
            raise InterruptedError("cancelled")


@dataclass(frozen=True)
class Step:
    name: str
    run: Callable[[Context], None]
    # Model role the step loads ("llm", "image", "video", "asr") or None.
    role: str | None = None
    # "project" steps run once; "episode" steps run per episode.
    scope: str = "project"
    description: str = ""
    source: str = "core"
    # An action, not a stage of the flow: it runs with the model of the named flow step.
    model_from: str | None = None


class Steps:
    """Core steps plus the plugins found in a directory."""

    def __init__(self, core: list[Step], plugins: Path | None):
        self.core = {step.name: step for step in core}
        self.plugin_dir = plugins
        self.plugins: dict[str, Step] = {}
        self.errors: dict[str, str] = {}
        self.reload()

    def reload(self) -> None:
        plugins, errors = {}, {}
        for path in sorted(self.plugin_dir.glob("*.py")) if self.plugin_dir else []:
            try:
                spec = importlib.util.spec_from_file_location(f"short_drama_plugin_{path.stem}", path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                step = module.STEP
                if not isinstance(step, Step) or step.name in self.core:
                    raise TypeError("STEP must be a Step whose name differs from the core steps")
                plugins[step.name] = Step(step.name, step.run, step.role, step.scope,
                                          step.description, source=path.name, model_from=step.model_from)
            except Exception as error:  # a broken plugin must not take the app down
                errors[path.name] = f"{type(error).__name__}: {error}"
                log.warning("plugin %s not loaded: %s", path.name, error)
        self.plugins, self.errors = plugins, errors

    def get(self, name: str) -> Step | None:
        return self.core.get(name) or self.plugins.get(name)

    def describe(self) -> list[dict[str, Any]]:
        return [{"name": s.name, "role": s.role, "scope": s.scope, "description": s.description,
                 "source": s.source, "model_from": s.model_from}
                for s in [*self.core.values(), *self.plugins.values()]]


def default_flow(models: dict[str, str]) -> dict[str, Any]:
    """The built-in first version; `models` maps a role to the model name serving it."""

    return {"steps": [
        {"step": "plan", "model": models.get("llm"),
         "params": {"temperature": 0.7, "shot_seconds": 8, "max_tokens": 12000}},
        {"step": "stills", "model": models.get("image"), "params": {"seed": 11}},
        {"step": "shots", "model": models.get("video"), "params": {"seed": 7, "previous_last_frame": True}},
        {"step": "subtitles", "model": models.get("asr"), "params": {}},
        {"step": "assemble", "model": None,
         "params": {"font": "Noto Sans CJK SC", "font_size": 12, "crf": 18}},
    ]}


class Flows:
    """Flow versions under `root/<name>/v<N>.json`; versions are never overwritten."""

    def __init__(self, root: Path, models: dict[str, str]):
        self.root = root
        if not self.versions("default"):
            self.save("default", default_flow(models), note="built-in")

    def names(self) -> list[str]:
        return sorted(path.name for path in self.root.iterdir() if path.is_dir()) if self.root.is_dir() else []

    def versions(self, name: str) -> list[int]:
        directory = self.root / name
        if not directory.is_dir():
            return []
        return sorted(int(path.stem[1:]) for path in directory.glob("v*.json") if path.stem[1:].isdigit())

    def load(self, reference: str) -> tuple[str, dict[str, Any]]:
        """Resolve `name` (latest) or `name@N` to its canonical reference and content."""

        name, _, number = reference.partition("@")
        if not FLOW_NAME.fullmatch(name) or not self.versions(name):
            raise KeyError(f"unknown flow {reference!r}")
        version = int(number) if number else self.versions(name)[-1]
        flow = read_json(self.root / name / f"v{version}.json")
        if flow is None:
            raise KeyError(f"unknown flow {reference!r}")
        return f"{name}@{version}", flow

    def save(self, name: str, flow: dict[str, Any], *, note: str = "") -> str:
        if not FLOW_NAME.fullmatch(name):
            raise ValueError("flow names use lowercase letters, digits, '-' and '_'")
        validate(flow)
        version = (self.versions(name) or [0])[-1] + 1
        write_json(self.root / name / f"v{version}.json", {**flow, "note": note})
        return f"{name}@{version}"


def validate(flow: Any) -> None:
    if not isinstance(flow, dict) or not isinstance(flow.get("steps"), list) or not flow["steps"]:
        raise ValueError("a flow is an object with a non-empty 'steps' list")
    for entry in flow["steps"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("step"), str):
            raise ValueError("each step needs a 'step' name")
        if not isinstance(entry.get("params", {}), dict):
            raise ValueError(f"params of {entry['step']} must be an object")
        if entry.get("model") is not None and not isinstance(entry["model"], str):
            raise ValueError(f"model of {entry['step']} must be a name or null")
    json.dumps(flow)
