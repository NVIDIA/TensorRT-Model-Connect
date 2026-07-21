# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Concurrency contracts for the node-local Hugging Face cache lock."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from tools.ci.cache_lock import CacheLock
from tools.ci.context import CiContext
from tools.ci.process import CiError


def _context(tmp_path: Path, *, timeout: str = "2") -> CiContext:
    return CiContext(
        repository=tmp_path,
        env={
            "TRTMC_HF_CACHE_LOCK_FILE": str(tmp_path / "cache.lock"),
            "TRTMC_HF_CACHE_LOCK_TIMEOUT_SECONDS": timeout,
        },
    )


def test_multiple_cache_readers_can_hold_shared_lock(tmp_path: Path) -> None:
    context = _context(tmp_path)
    first = CacheLock(context, shared=True)
    second = CacheLock(context, shared=True)

    with first:
        assert first.lock is not None
        with second:
            assert first.lock is not None
            assert second.lock is not None
        assert second.lock is None
    assert first.lock is None


def test_exclusive_cache_writer_blocks_until_reader_releases(tmp_path: Path) -> None:
    context = _context(tmp_path)
    writer_started = threading.Event()
    writer_acquired = threading.Event()

    def hold_writer() -> None:
        writer = CacheLock(context, shared=False)
        writer_started.set()
        with writer:
            writer_acquired.set()

    with ThreadPoolExecutor(max_workers=1) as executor:
        with CacheLock(context, shared=True):
            future = executor.submit(hold_writer)
            assert writer_started.wait(timeout=1)
            assert not writer_acquired.wait(timeout=0.2)

        assert writer_acquired.wait(timeout=2)
        future.result(timeout=2)


def test_cache_lock_times_out_while_incompatible_lock_is_held(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path, timeout="1")

    with CacheLock(context, shared=False):
        started = time.monotonic()
        with pytest.raises(
            CiError,
            match=r"timed out after 1s waiting for shared HF cache lock",
        ):
            with CacheLock(context, shared=True):
                pytest.fail("incompatible shared lock unexpectedly acquired")
        elapsed = time.monotonic() - started

    assert 0.8 <= elapsed < 3


def test_cache_lock_releases_when_protected_operation_raises(tmp_path: Path) -> None:
    context = _context(tmp_path)

    with pytest.raises(RuntimeError, match="synthetic cache failure"):
        with CacheLock(context, shared=False):
            raise RuntimeError("synthetic cache failure")

    replacement = CacheLock(context, shared=False)
    with replacement:
        assert replacement.lock is not None
    assert replacement.lock is None
