# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run repository-wide ready-model contracts in every builder lane."""

from tests.tools.test_family_source_isolation import (
    test_family_imports_resolve_without_sibling_or_unapproved_shared_modules as _check_family_isolation,
)
from tests.tools.test_family_specialization import (
    test_repository_registers_all_current_families as _check_family_inventory,
)
from tests.tools.test_perf_matrix import (
    test_release_suite_covers_every_non_l0_ready_model_profile as _check_release_coverage,
)
from tests.tools.test_trtmc_validate import (
    test_model_workload_catalog_covers_every_ready_model as _check_validation_coverage,
)


def test_ready_models_have_validation_workloads() -> None:
    _check_validation_coverage()


def test_ready_models_have_release_performance_coverage_or_exclusion() -> None:
    _check_release_coverage()


def test_family_sources_remain_isolated() -> None:
    _check_family_isolation()


def test_repository_family_inventory_is_current() -> None:
    _check_family_inventory()
