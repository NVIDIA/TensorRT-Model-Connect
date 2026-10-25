# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cache identity and disk accounting guards, not GPU qualification."""

import hashlib

import numpy as np
import pytest

from families.minimax_h3.tests import vae_cached_video_case as cached
from families.minimax_h3.tests import test_trained_video_decoder as owning


def test_cached_reference_requires_full_exact_file_bytes_and_direct_path(tmp_path):
    path = tmp_path / "reference.bin"
    payload = np.arange(16, dtype=np.float32).tobytes()
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    cached.verify_file(path, len(payload), digest)
    with pytest.raises(AssertionError):
        cached.verify_file(path, len(payload) - 1, digest)
    with pytest.raises(AssertionError):
        cached.verify_file(path, len(payload), "0" * 64)
    alias = tmp_path / "indirect.bin"
    alias.symlink_to(path)
    with pytest.raises(AssertionError):
        cached.verify_file(alias, len(payload), digest)
    with pytest.raises(AssertionError):
        cached.verify_file(tmp_path / "missing.bin", len(payload), digest)


def test_cached_budget_retains_both_plans_four_videos_and_atomic_staging():
    cap = (
        max(size for name, (size, _sha) in cached.CACHE_FILES.items() if name.endswith(".bundle"))
        + cached.PLAN_MARGIN
    )
    budget = cached.storage_budget(cap)
    assert cached.PIXEL_BYTES == int(np.prod(owning._SHAPE)) * 4
    assert budget["bundle_cap"] > cap
    assert budget["staging_peak"] == 3 * budget["bundle_cap"]
    assert (
        budget["final_payload"]
        >= 2 * budget["bundle_cap"] + 4 * cached.PIXEL_BYTES + cached.ROWS_BYTES
    )
    assert (
        budget["required_free_bytes"]
        == max(budget["staging_peak"], budget["final_payload"]) + cached.DISK_RESERVE
    )
    interleaved_peak = 3 * budget["bundle_cap"] + 2 * cached.PIXEL_BYTES
    assert max(budget["staging_peak"], budget["final_payload"]) < interleaved_peak


def test_cached_path_is_an_explicit_opt_in_to_the_same_owning_case(monkeypatch, tmp_path):
    # Dispatch only: neither branch executes inference in this host guard.
    seen = []
    monkeypatch.setenv("TRTMC_MINIMAX_H3_VIDEO_REFERENCE_CACHE", str(tmp_path / "exact-cache"))

    def record(*_args):
        pass

    monkeypatch.setattr(cached, "run_cached_case", lambda *args: seen.append(args))
    owning.test_trained_video_decoder_full_portrait_matches_native_memory_presets(tmp_path, record)
    assert seen == [(tmp_path, record, tmp_path / "exact-cache")]
