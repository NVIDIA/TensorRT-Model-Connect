#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audio diff entrypoint.

Concrete audio comparison behavior is owned by model modules under
``python/tensorrt_model_connect/models/*/diff_audio.py``. This shared tool
only discovers and dispatches to those handlers.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any


def _family_roots() -> tuple[Path, ...]:
    repo_root = Path(__file__).resolve().parents[1]
    return (repo_root / "python/tensorrt_model_connect/models",)


def _family_handler_paths(filename: str) -> list[Path]:
    handlers: dict[str, Path] = {}
    for root in reversed(_family_roots()):
        handlers.update({path.parent.name: path for path in root.glob(f"*/{filename}")})
    return [handlers[family] for family in sorted(handlers)]


@lru_cache(maxsize=1)
def _family_audio_diff_modules() -> tuple[ModuleType, ...]:
    modules: list[ModuleType] = []
    for handler_path in _family_handler_paths("diff_audio.py"):
        module_name = f"_trtmc_diff_audio_{handler_path.parent.name}"
        spec = importlib.util.spec_from_file_location(module_name, handler_path)
        if spec is None or spec.loader is None:
            print(f"[diff_audio] WARN: cannot load family audio diff handler "
                  f"{handler_path}", file=sys.stderr)
            continue
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            print(f"[diff_audio] WARN: failed to import family audio diff handler "
                  f"{handler_path}: {exc}", file=sys.stderr)
            continue
        modules.append(module)
    return tuple(modules)


def _select_family_from_argv(argv: list[str]) -> str | None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--family", default=None)
    ns, _ = parser.parse_known_args(argv)
    return ns.family


def _find_family_audio_diff_handler(argv: list[str] | None = None) -> ModuleType:
    args = list(sys.argv[1:] if argv is None else argv)
    requested_family = _select_family_from_argv(args)
    modules = _family_audio_diff_modules()

    if requested_family:
        for module in modules:
            if Path(str(module.__file__)).parent.name == requested_family:
                return module
        raise SystemExit(f"No audio diff handler for family {requested_family!r}")

    claimants = [
        module for module in modules
        if callable(getattr(module, "handles_audio_diff_args", None))
        and module.handles_audio_diff_args(args)
    ]
    if len(claimants) == 1:
        return claimants[0]
    if len(claimants) > 1:
        names = ", ".join(Path(str(module.__file__)).parent.name
                          for module in claimants)
        raise SystemExit(f"Multiple audio diff handlers matched: {names}")
    if len(modules) == 1:
        return modules[0]

    raise SystemExit("No audio diff handler matched; pass --family <name>")


def _strip_wrapper_args(argv: list[str]) -> list[str]:
    stripped: list[str] = []
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg == "--family":
            skip_next = True
            continue
        if arg.startswith("--family="):
            continue
        stripped.append(arg)
    return stripped


def __getattr__(name: str) -> Any:
    return getattr(_find_family_audio_diff_handler([]), name)


def main() -> None:
    argv = sys.argv[1:]
    handler = _find_family_audio_diff_handler(argv)
    original_argv = sys.argv
    sys.argv = [original_argv[0], *_strip_wrapper_args(argv)]
    try:
        handler.main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
