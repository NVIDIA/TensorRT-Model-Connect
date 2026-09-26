# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from qualification_tests.benchmark_qualification.catalog import QualificationError
from qualification_tests.benchmark_qualification.performance import qualification


def test_bundle_retention_defaults_to_retain(monkeypatch):
    monkeypatch.delenv("TRTMC_QUALIFICATION_BUNDLE_RETENTION", raising=False)

    assert qualification._bundle_retention() == "retain"


@pytest.mark.parametrize("retention", ["retain", "delete_on_pass", "delete_always"])
def test_bundle_retention_accepts_matrix_policies(monkeypatch, retention):
    monkeypatch.setenv("TRTMC_QUALIFICATION_BUNDLE_RETENTION", retention)

    assert qualification._bundle_retention() == retention


def test_bundle_retention_rejects_unknown_policies(monkeypatch):
    monkeypatch.setenv("TRTMC_QUALIFICATION_BUNDLE_RETENTION", "sometimes")

    with pytest.raises(QualificationError):
        qualification._bundle_retention()
