#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run one online Hugging Face cache mutation in a disposable process."""

from __future__ import annotations

import argparse
import json
import os
import sys

_DEFAULT_ETAG_TIMEOUT_SECONDS = 30
_DEFAULT_DOWNLOAD_TIMEOUT_SECONDS = 60


def _set_request_timeout_defaults() -> None:
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", str(_DEFAULT_ETAG_TIMEOUT_SECONDS))
    os.environ.setdefault(
        "HF_HUB_DOWNLOAD_TIMEOUT",
        str(_DEFAULT_DOWNLOAD_TIMEOUT_SECONDS),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operation", choices=("snapshot", "file"), required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--allow-patterns-json")
    parser.add_argument("--filename")
    return parser.parse_args()


def _download(args: argparse.Namespace) -> str:
    # These defaults and HF_HUB_DISABLE_XET must be present before the Hub
    # package initializes its constants and transfer backend.
    _set_request_timeout_defaults()
    from huggingface_hub import hf_hub_download, snapshot_download

    if args.operation == "snapshot":
        if not args.allow_patterns_json:
            raise ValueError("snapshot operation requires --allow-patterns-json")
        allow_patterns = json.loads(args.allow_patterns_json)
        if not isinstance(allow_patterns, list) or not all(
            isinstance(pattern, str) for pattern in allow_patterns
        ):
            raise ValueError("--allow-patterns-json must encode a list of strings")
        return snapshot_download(args.repo_id, allow_patterns=allow_patterns)

    if not args.filename:
        raise ValueError("file operation requires --filename")
    return hf_hub_download(args.repo_id, filename=args.filename)


def main() -> int:
    args = _parse_args()
    try:
        path = _download(args)
    except Exception as exc:  # noqa: BLE001 - surface the worker failure to the parent
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"path": str(path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
