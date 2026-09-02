# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate a local-only protected CI failure preview."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

from .build import build_failure_artifacts
from .export import ExportContext


def _load_object(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _context_from_object(value: Mapping[str, object]) -> ExportContext:
    return ExportContext(
        repository=value.get("repository"),
        pr_number=value.get("pr_number"),
        head_sha=value.get("head_sha"),
        base_sha=value.get("base_sha"),
        tested_revision=value.get("tested_revision"),
        dispatch_nonce=value.get("dispatch_nonce"),
        run_attempt=value.get("run_attempt"),
        result=value.get("result"),
        generated_at=value.get("generated_at"),
        tested_revision_kind=value.get("tested_revision_kind", "head"),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate local public-failure-v1 JSON and text log. Nothing is uploaded."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    internal = _load_object(args.input)
    context = _context_from_object(_load_object(args.context))
    artifacts = build_failure_artifacts(internal, context)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.html").unlink(missing_ok=True)
    (args.output_dir / "public-failure.json").write_bytes(artifacts.json_bytes)
    (args.output_dir / "public-failure.log").write_bytes(artifacts.log_bytes)
    print(f"Local preview written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
