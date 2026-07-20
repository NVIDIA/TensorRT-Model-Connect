#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from pathlib import Path

from tools.legal_headers import load_exceptions


REPO_ROOT = Path(__file__).resolve().parents[2]
IGNORE_FILE = REPO_ROOT / ".clang-format-ignore"
EXCEPTION_MANIFEST = REPO_ROOT / "tools" / "legal_header_exceptions.toml"
CUDNN_ROOT = Path(
    "python/tensorrt_model_connect/families/wan2_2_ti2v/dit_cuda_plugins/third_party/"
    "cudnn_frontend"
)
INCLUDE_PREFIX = CUDNN_ROOT / "include"
IGNORE_RULE = f"{INCLUDE_PREFIX.as_posix()}/**"
UPSTREAM_COMMIT = "a91f0e04dcea10515f0f776fc5a89535e316a9c8"
UPSTREAM_ROOT = f"https://github.com/NVIDIA/cudnn-frontend/blob/{UPSTREAM_COMMIT}"


def test_clang_format_ignore_is_exactly_the_pinned_cudnn_frontend_include_snapshot() -> None:
    rules = [
        line.strip()
        for line in IGNORE_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert rules == [IGNORE_RULE]

    exceptions = load_exceptions(EXCEPTION_MANIFEST)
    prefix = f"{INCLUDE_PREFIX.as_posix()}/"
    exception_paths = {Path(path) for path in exceptions if path.startswith(prefix)}
    actual_paths = {
        path.relative_to(REPO_ROOT)
        for path in (REPO_ROOT / INCLUDE_PREFIX).rglob("*")
        if path.is_file()
    }
    assert len(actual_paths) == 91
    assert actual_paths == exception_paths

    for path in sorted(actual_paths):
        entry = exceptions[path.as_posix()]
        assert hashlib.sha256((REPO_ROOT / path).read_bytes()).hexdigest() == entry.sha256
        assert entry.license == "MIT"
        assert "Vendored cuDNN frontend v1.22.1 source snapshot" in entry.reason
        assert "preserve" in entry.reason and "unchanged" in entry.reason
        upstream_relative = path.relative_to(CUDNN_ROOT).as_posix()
        assert entry.source == f"{UPSTREAM_ROOT}/{upstream_relative}"
