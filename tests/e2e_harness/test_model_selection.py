# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from tests.e2e_harness.model_selection import (
    read_e2e_models_file,
    select_cases_from_models_file,
)


@dataclass(frozen=True)
class _Case:
    name: str


def test_read_e2e_models_file_supports_names_node_ids_and_groups(tmp_path):
    models_file = tmp_path / "models.txt"
    models_file.write_text(
        "model-a\n"
        "tests/e2e/models/f/test_f_e2e.py::test_model_e2e[model-b]\n"
        "tests/e2e/models/f/test_f_e2e.py::test_model_e2e["
        "bundle:model-c+model-d]\n"
        "# ignored\n",
        encoding="utf-8",
    )

    assert read_e2e_models_file(models_file) == {
        "model-a",
        "model-b",
        "model-c",
        "model-d",
    }


def test_select_cases_from_models_file_matches_only_case_name(tmp_path):
    models_file = tmp_path / "models.txt"
    models_file.write_text("base\n", encoding="utf-8")
    cases = [_Case("base"), _Case("base-probe01")]

    assert select_cases_from_models_file(cases, models_file) == [_Case("base")]
