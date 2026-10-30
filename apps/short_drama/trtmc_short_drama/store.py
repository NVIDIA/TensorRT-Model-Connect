# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Projects on disk: the novel, the editable plan, reference stills, shots and episodes."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import time
from typing import Any

PROJECT_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")
SNAPSHOT = re.compile(r"\d{8}-\d{6}-\d{3}\.json")
KEEP_SNAPSHOTS = 50


def write_json(path: Path, value: Any) -> None:
    """Replace a JSON file atomically so a crash never leaves half a plan."""

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    partial.replace(path)


def write_secret(path: Path, value: Any) -> None:
    """Like write_json, but the file is readable by its owner only from the first byte."""

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as file:
        json.dump(value, file)
    partial.replace(path)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Project:
    root: Path

    @property
    def id(self) -> str:
        return self.root.name

    @property
    def meta(self) -> dict[str, Any]:
        return read_json(self.root / "project.json", {})

    def update_meta(self, **changes: Any) -> dict[str, Any]:
        meta = {**self.meta, **changes}
        write_json(self.root / "project.json", meta)
        return meta

    @property
    def novel(self) -> str:
        path = self.root / "novel.txt"
        return path.read_text(encoding="utf-8") if path.is_file() else ""

    def save_novel(self, text: str) -> None:
        partial = self.root / f".novel.txt.{os.getpid()}.tmp"
        partial.write_text(text, encoding="utf-8")
        partial.replace(self.root / "novel.txt")

    @property
    def plan(self) -> dict[str, Any]:
        return read_json(self.root / "plan.json", {})

    def save_plan(self, plan: dict[str, Any], *, note: str = "") -> None:
        """Save the plan; with a note, keep the plan it replaces as a snapshot first."""

        if note and (self.root / "plan.json").is_file():
            self.snapshot(note)
        write_json(self.root / "plan.json", plan)

    def snapshot(self, note: str) -> str:
        """Copy the current plan into history/, so a model edit or a restore can be undone."""

        history = self.root / "history"
        stamp = time.strftime("%Y%m%d-%H%M%S")
        number = sum(1 for path in history.glob(f"{stamp}-*.json")) if history.is_dir() else 0
        name = f"{stamp}-{number:03d}.json"
        write_json(history / name, {"note": note[:120], "time": int(time.time()), "plan": self.plan})
        for old in self.snapshots()[KEEP_SNAPSHOTS:]:
            (history / old["name"]).unlink()
        return name

    def snapshots(self) -> list[dict[str, Any]]:
        """Plan snapshots, newest first, without their plans."""

        history = self.root / "history"
        names = sorted((path.name for path in history.glob("*.json") if SNAPSHOT.fullmatch(path.name)),
                       reverse=True) if history.is_dir() else []
        return [{"name": name, **{key: value for key, value in read_json(history / name, {}).items()
                                   if key != "plan"}} for name in names]

    def restore(self, name: str) -> None:
        if not SNAPSHOT.fullmatch(name) or not (self.root / "history" / name).is_file():
            raise KeyError(name)
        self.save_plan(read_json(self.root / "history" / name)["plan"], note=f"before restoring {name}")

    @property
    def review(self) -> dict[str, Any]:
        return read_json(self.root / "review.json", {})

    def save_review(self, review: dict[str, Any]) -> None:
        write_json(self.root / "review.json", review)

    @property
    def stills(self) -> dict[str, Any]:
        return read_json(self.root / "stills.json", {})

    def save_stills(self, stills: dict[str, Any]) -> None:
        write_json(self.root / "stills.json", stills)

    def shot_state(self, episode: int) -> dict[str, Any]:
        return read_json(self.episode_dir(episode) / "shots.json", {})

    def save_shot_state(self, episode: int, state: dict[str, Any]) -> None:
        write_json(self.episode_dir(episode) / "shots.json", state)

    def episode_dir(self, episode: int) -> Path:
        return self.root / "episodes" / f"{episode:02d}"

    def file(self, relative: str) -> Path:
        """A file inside the project; reject anything that escapes it."""

        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root.resolve()) or not path.is_file():
            raise FileNotFoundError(relative)
        return path

    def log(self, message: str) -> None:
        with (self.root / "log.txt").open("a", encoding="utf-8") as file:
            file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")


class Store:
    def __init__(self, root: Path):
        self.root = root
        (root / "projects").mkdir(parents=True, exist_ok=True)

    def projects(self) -> list[dict[str, Any]]:
        found = []
        for path in sorted((self.root / "projects").iterdir()):
            if path.is_dir() and (path / "project.json").is_file():
                found.append({"id": path.name, **Project(path).meta})
        return found

    def get(self, project_id: str) -> Project:
        if not PROJECT_ID.fullmatch(project_id):
            raise KeyError(project_id)
        root = self.root / "projects" / project_id
        if not (root / "project.json").is_file():
            raise KeyError(project_id)
        return Project(root)

    def create(self, title: str, novel: str, *, flow: str, episodes: int, max_seconds: int,
               idea: str = "") -> Project:
        """A project starts from a novel, or from an idea the story step turns into one."""

        if not novel.strip() and not idea.strip():
            raise ValueError("paste a novel or describe an idea")
        base = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40] or "drama"
        project_id, number = base, 1
        while (self.root / "projects" / project_id).exists():
            number += 1
            project_id = f"{base}-{number}"
        root = self.root / "projects" / project_id
        root.mkdir(parents=True)
        (root / "novel.txt").write_text(novel, encoding="utf-8")
        project = Project(root)
        project.update_meta(title=title, flow=flow, episodes=episodes, max_seconds=max_seconds,
                            created=int(time.time()), **({"idea": idea.strip()} if idea.strip() else {}))
        return project
