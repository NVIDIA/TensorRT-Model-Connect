# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Failure and reuse behavior of the owning E2E's bundle construction."""

import pytest

from families.ideogram4.tests import test_e2e


def test_bundle_cache_is_published_only_after_success(monkeypatch, tmp_path, tmp_path_factory):
    monkeypatch.setattr(test_e2e, "_BUNDLE", None)
    attempts = []

    def build(request):
        attempts.append(request.output_path)
        if len(attempts) == 1:
            request.output_path.write_bytes(b"partial")
            raise RuntimeError("injected engine build failure")
        request.output_path.write_bytes(b"complete")

    monkeypatch.setattr(test_e2e, "build", build)
    with pytest.raises(RuntimeError, match="injected engine build failure"):
        test_e2e._build_bundle(tmp_path, tmp_path_factory)
    assert test_e2e._BUNDLE is None
    bundle = test_e2e._build_bundle(tmp_path, tmp_path_factory)
    assert len(attempts) == 2 and attempts[0] != bundle
    assert bundle.read_bytes() == b"complete"
    assert test_e2e._build_bundle(tmp_path, tmp_path_factory) == bundle
    assert len(attempts) == 2
