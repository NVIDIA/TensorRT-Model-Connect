# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from tools.reporting_html import (
    ReportFilter,
    render_report_filter_script,
    render_report_filters,
    task_type_label,
)


def test_report_filters_share_stable_ids_and_escape_values() -> None:
    document = render_report_filters(
        row_count=2,
        search_placeholder='Model "name"',
        filters=(
            ReportFilter("model-type", "Model type", ("qwen", "vision&language")),
            ReportFilter(
                "status",
                "Status",
                (("green", "Green"), ("white", "Not run")),
            ),
        ),
    )

    assert 'id="report-filter-search"' in document
    assert 'placeholder="Model &quot;name&quot;"' in document
    assert 'id="report-filter-model-type"' in document
    assert 'id="report-filter-status"' in document
    assert ">vision&amp;language</option>" in document
    assert 'id="report-filter-count">Showing 2 of 2 rows<' in document
    assert "data-report-filter" in document
    assert "row.hidden = !matches;" in render_report_filter_script()


def test_task_type_contract_precedes_strategy_fallback() -> None:
    assert (
        task_type_label(
            user_contract="code_completion",
            task_strategy="text_generation_causal",
            operation="generate",
        )
        == "Text → Code"
    )


def test_task_type_distinguishes_media_requests() -> None:
    common = {
        "task_strategy": "diffusion_media_generation",
        "operation": "generate_image",
    }

    assert task_type_label(**common, request={"media_type": "image"}) == "Text → Image"
    assert task_type_label(**common, request={"media_type": "video"}) == "Text → Video"
    assert (
        task_type_label(
            **common, request={"media_type": "image", "image_path": "input.png"}
        )
        == "Image + Text → Image"
    )


def test_task_type_labels_image_feature_extraction() -> None:
    assert (
        task_type_label(
            user_contract="representation_parity",
            task_strategy="image_feature_extraction",
            operation="extract_features",
        )
        == "Image → Features"
    )


def test_task_type_labels_monocular_geometry() -> None:
    assert (
        task_type_label(user_contract="metric_monocular_geometry")
        == "Image → Metric Geometry"
    )
    assert (
        task_type_label(task_strategy="monocular_geometry")
        == "Image → Metric Geometry"
    )
