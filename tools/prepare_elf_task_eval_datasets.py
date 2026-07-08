#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare reproducible ELF task-eval datasets under one dataset root."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_XSUM_TEST_PARQUET = (
    "https://huggingface.co/datasets/EdinburghNLP/xsum/resolve/"
    "refs%2Fconvert%2Fparquet/default/test/0000.parquet"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_xsum(*, output_root: Path, parquet_url: str) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover - data-preparation dependency
        raise RuntimeError("XSum preparation requires pyarrow") from exc
    output_dir = output_root / "XSum"
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / "xsum_test.parquet"
    jsonl_path = output_dir / "xsum_test.jsonl"
    if not parquet_path.exists():
        with urllib.request.urlopen(parquet_url, timeout=120) as response:
            with parquet_path.open("wb") as output_file:
                shutil.copyfileobj(response, output_file)
    table = pq.read_table(parquet_path, columns=["id", "document", "summary"])
    written = 0
    skipped_empty = 0
    with jsonl_path.open("w", encoding="utf-8") as output_file:
        for index, row in enumerate(table.to_pylist()):
            document = str(row["document"]).strip()
            summary = str(row["summary"]).strip()
            if not document or not summary:
                skipped_empty += 1
                continue
            output_file.write(
                json.dumps(
                    {
                        "id": str(row.get("id") or f"xsum_test_{index:06d}"),
                        "input": document,
                        "output": summary,
                        "subset": "test",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1
    return {
        "path": str(jsonl_path),
        "source": parquet_url,
        "samples": written,
        "source_rows": int(table.num_rows),
        "skipped_empty": skipped_empty,
        "sha256": _sha256(jsonl_path),
    }


def prepare_owt_requests(*, output_root: Path, count: int, seed: int) -> dict[str, Any]:
    output_dir = output_root / "OpenWebText"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "elf_owt_generation.json"
    payload = {
        "dataset": "ELF OpenWebText unconditional generation requests",
        "description": (
            "Seed-only requests for the official ELF-B OpenWebText 32-step SDE evaluation; "
            "no corpus examples are consumed by unconditional generation."
        ),
        "requests": [
            {"id": f"elf_owt_{index:06d}", "seed": seed + index, "subject": "unconditional"}
            for index in range(count)
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "path": str(output_path),
        "samples": count,
        "sha256": _sha256(output_path),
    }


def prepare_wmt14(output_root: Path, source_path: Path) -> dict[str, Any]:
    path = output_root / "WMT14_de_en" / "wmt14_de_en_test.jsonl"
    if source_path.is_file() and source_path.resolve() != path.resolve():
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, path)
    if not path.is_file():
        return {
            "path": str(path),
            "source": str(source_path),
            "status": "missing",
        }
    samples = sum(1 for line in path.open("r", encoding="utf-8") if line.strip())
    return {
        "path": str(path),
        "source": str(source_path),
        "status": "present",
        "samples": samples,
        "sha256": _sha256(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("/mnt/data"))
    parser.add_argument(
        "--wmt14-path",
        type=Path,
        default=Path("/mnt/data/WMT14_de_en/wmt14_de_en_test.jsonl"),
    )
    parser.add_argument("--xsum-parquet-url", default=DEFAULT_XSUM_TEST_PARQUET)
    parser.add_argument("--owt-request-count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.owt_request_count < 1:
        parser.error("--owt-request-count must be positive")
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "wmt14_de_en": prepare_wmt14(args.output_root, args.wmt14_path),
        "xsum": prepare_xsum(
            output_root=args.output_root,
            parquet_url=args.xsum_parquet_url,
        ),
        "openwebtext_generation": prepare_owt_requests(
            output_root=args.output_root,
            count=args.owt_request_count,
            seed=args.seed,
        ),
    }
    manifest_path = args.output_root / "elf_task_eval_datasets.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({**manifest, "manifest": str(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
