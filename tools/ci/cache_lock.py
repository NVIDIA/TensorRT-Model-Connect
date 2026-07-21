# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coordinate node-local Hugging Face cache readers and the Nightly writer.

Boundary: file-lock coordination only; cache selection and download stay with callers.
"""

from __future__ import annotations

import time
from pathlib import Path

from .context import CiContext
from .gpu_lease import FileLock
from .process import CiError


class CacheLock:
    """Hold a shared reader or exclusive writer flock for one node cache."""

    def __init__(self, context: CiContext, *, shared: bool):
        configured = context.env.get("TRTMC_HF_CACHE_LOCK_FILE", "")
        path = Path(configured)
        if not path.is_absolute() or path == Path("/"):
            raise CiError("TRTMC_HF_CACHE_LOCK_FILE must be a safe absolute path")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink() or path.is_dir():
            raise CiError("TRTMC_HF_CACHE_LOCK_FILE must be a regular lock file")
        timeout_text = context.env.get("TRTMC_HF_CACHE_LOCK_TIMEOUT_SECONDS", "7200")
        if not timeout_text.isdigit() or not 1 <= int(timeout_text) <= 21600:
            raise CiError("TRTMC_HF_CACHE_LOCK_TIMEOUT_SECONDS must be an integer from 1 to 21600")
        self.path = path
        self.shared = shared
        self.timeout = int(timeout_text)
        self.lock: FileLock | None = None

    def __enter__(self) -> CacheLock:
        deadline = time.monotonic() + self.timeout
        lock = FileLock(self.path)
        while time.monotonic() < deadline:
            if lock.try_lock(shared=self.shared):
                mode = "shared" if self.shared else "exclusive"
                print(f"Acquired {mode} Hugging Face cache lock via {self.path}")
                self.lock = lock
                return self
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
        lock.handle.close()
        mode = "shared" if self.shared else "exclusive"
        raise CiError(f"timed out after {self.timeout}s waiting for {mode} HF cache lock")

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        if self.lock:
            self.lock.close()
            self.lock = None
