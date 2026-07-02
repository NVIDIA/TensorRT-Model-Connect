# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the pure helpers in tools/check_cyclomatic_complexity.py.

Intent:
    The CCN gate is required for C++ runtime changes. A silent regression in
    ``parse_csv`` or ``evaluate_gate`` would
    let cyclomatic-complexity violations slip through unnoticed. This module
    covers those two pure helpers with in-memory inputs so that the tests do
    NOT shell out to the ``lizard`` binary and do NOT require that dependency
    to be installed in the test environment.

Preconditions:
    ``tools.check_cyclomatic_complexity`` is importable (no third-party deps
    beyond the Python standard library).

Postconditions:
    ``parse_csv`` and ``evaluate_gate`` behave as documented on the happy path
    and on the malformed/empty-input edge cases that the CCN gate must
    tolerate at runtime.

Trace IDs:
    - ARCH-CI-QUALITY-GATES (cyclomatic-complexity gate for C++ runtime)
    - UD-TOOLS-CCN-HELPERS (pure helpers: parse_csv, evaluate_gate)
    - UT-TOOLS-CCN-PARSE-CSV, UT-TOOLS-CCN-EVALUATE-GATE
"""

from tools import check_cyclomatic_complexity as ccm


# ---------------------------------------------------------------------------
# parse_csv
# ---------------------------------------------------------------------------


def test_parse_csv_empty_input_returns_empty_list() -> None:
    # An empty CSV buffer is what run_lizard() produces when lizard finds no
    # functions; parse_csv must tolerate it without raising.
    assert ccm.parse_csv("") == []


def test_parse_csv_whitespace_only_input_returns_empty_list() -> None:
    # Only blank lines -> csv.reader yields empty rows with len(row) < 11 and
    # they must be skipped rather than trigger IndexError.
    assert ccm.parse_csv("\n\n  \n") == []


def test_parse_csv_well_formed_row_produces_matching_function_metric() -> None:
    # Exactly 11 columns matching the FunctionMetric dataclass order:
    #   nloc, ccn, token, param, length, location, file, function, signature,
    #   start_line, end_line
    row = "12,3,45,2,20,src/foo.cpp:10,src/foo.cpp,do_work,do_work(int a),10,29"
    metrics = ccm.parse_csv(row)

    assert len(metrics) == 1
    m = metrics[0]
    assert isinstance(m, ccm.FunctionMetric)
    assert m.nloc == 12
    assert m.ccn == 3
    assert m.token == 45
    assert m.param == 2
    assert m.length == 20
    assert m.location == "src/foo.cpp:10"
    assert m.file == "src/foo.cpp"
    assert m.function == "do_work"
    assert m.signature == "do_work(int a)"
    assert m.start_line == 10
    assert m.end_line == 29


def test_parse_csv_skips_row_with_non_integer_nloc() -> None:
    # The numeric columns (nloc/ccn/token/param/length/start_line/end_line)
    # are int()-coerced; a non-integer must trigger the ValueError guard and
    # cause the row to be skipped silently, not bubble up as an exception.
    good = "7,2,30,1,15,src/ok.cpp:1,src/ok.cpp,ok_fn,ok_fn(),1,15"
    bad = "NOT_AN_INT,2,30,1,15,src/bad.cpp:1,src/bad.cpp,bad_fn,bad_fn(),1,15"
    csv_text = "\n".join([bad, good])

    metrics = ccm.parse_csv(csv_text)

    # Only the well-formed row survives; the malformed one is dropped silently.
    assert len(metrics) == 1
    assert metrics[0].function == "ok_fn"


def test_parse_csv_skips_row_with_fewer_than_eleven_columns() -> None:
    # len(row) < 11 short-circuits before any int() call; the row is simply
    # skipped. This defends against truncated lizard output and CSV header
    # banners that don't match the function-metric schema.
    short_row = "1,2,3,4,5"  # only 5 columns
    good = "7,2,30,1,15,src/ok.cpp:1,src/ok.cpp,ok_fn,ok_fn(),1,15"
    csv_text = "\n".join([short_row, good])

    metrics = ccm.parse_csv(csv_text)

    assert len(metrics) == 1
    assert metrics[0].function == "ok_fn"


def test_filter_excluded_drops_files_under_excluded_directory() -> None:
    metrics = [
        _make_metric(ccn=20, function="cli_fn", file="src/cli/main.cpp"),
        _make_metric(ccn=4, function="runtime_fn", file="src/runtime/core.cpp"),
    ]

    filtered = ccm.filter_excluded(metrics, ["src/cli"])

    assert [metric.function for metric in filtered] == ["runtime_fn"]


# ---------------------------------------------------------------------------
# evaluate_gate
# ---------------------------------------------------------------------------


def _make_metric(
    ccn: int,
    nloc: int = 10,
    function: str = "fn",
    file: str = "src/x.cpp",
) -> ccm.FunctionMetric:
    """Build a FunctionMetric with the minimum fields needed for the gate.

    evaluate_gate only inspects ``ccn``; all other fields are filler so the
    dataclass constructs cleanly.
    """
    return ccm.FunctionMetric(
        nloc=nloc,
        ccn=ccn,
        token=0,
        param=0,
        length=nloc,
        location=f"{file}:1",
        file=file,
        function=function,
        signature=f"{function}()",
        start_line=1,
        end_line=nloc,
    )


def test_evaluate_gate_empty_metrics_returns_no_failures() -> None:
    # No metrics means nothing to compare; must short-circuit to [] even when
    # thresholds are set, otherwise max() on an empty sequence would raise.
    assert ccm.evaluate_gate([], max_ccn=10, ccn_threshold=20, max_count_at_or_above=5) == []


def test_evaluate_gate_flags_max_ccn_exceeded_with_observed_and_limit_in_message() -> None:
    # One function at CCN=15 with max_ccn=10 must produce a single failure
    # whose message cites both the observed max (15) and the configured limit
    # (10), matching the "max CCN {max_seen} exceeds allowed {max_ccn}" format
    # so operators can read the gate log and see what exceeded what.
    metrics = [_make_metric(ccn=5), _make_metric(ccn=15, function="hotspot")]

    failures = ccm.evaluate_gate(
        metrics,
        max_ccn=10,
        ccn_threshold=20,
        max_count_at_or_above=None,
    )

    assert len(failures) == 1
    msg = failures[0]
    assert "15" in msg
    assert "10" in msg
    assert "max CCN" in msg


def test_evaluate_gate_flags_count_at_or_above_threshold_exceeding_limit() -> None:
    # Three functions at CCN >= 20 (the threshold), but max_count_at_or_above=2
    # means 3 > 2 must fail. The message must cite the observed count (3) and
    # the configured limit (2) per the
    # "count(CCN >= {ccn_threshold}) {count} exceeds allowed {max_count_at_or_above}"
    # format, so operators can diagnose without re-running lizard.
    metrics = [
        _make_metric(ccn=25),
        _make_metric(ccn=20),
        _make_metric(ccn=30),
        _make_metric(ccn=5),  # below threshold, should not be counted
    ]

    failures = ccm.evaluate_gate(
        metrics,
        max_ccn=None,
        ccn_threshold=20,
        max_count_at_or_above=2,
    )

    assert len(failures) == 1
    msg = failures[0]
    # Observed count (3) and limit (2) and threshold (20) must all appear.
    assert "3" in msg
    assert "2" in msg
    assert "20" in msg
    assert "count(CCN >=" in msg


def test_evaluate_gate_returns_empty_when_metrics_within_both_limits() -> None:
    # Max CCN = 9 <= max_ccn=10, and count(CCN >= 20) = 0 <= 1. No failures.
    metrics = [_make_metric(ccn=9), _make_metric(ccn=7), _make_metric(ccn=3)]

    failures = ccm.evaluate_gate(
        metrics,
        max_ccn=10,
        ccn_threshold=20,
        max_count_at_or_above=1,
    )

    assert failures == []


def test_evaluate_gate_boundary_not_flagged_when_equal_to_max_ccn() -> None:
    # The max-CCN gate uses a strict ``>`` comparison: a function at exactly
    # max_ccn is allowed. This locks down the inclusive/exclusive boundary so
    # a regression to ``>=`` would be caught here.
    metrics = [_make_metric(ccn=10)]

    failures = ccm.evaluate_gate(
        metrics,
        max_ccn=10,
        ccn_threshold=20,
        max_count_at_or_above=None,
    )

    assert failures == []


def test_evaluate_gate_count_equal_to_limit_is_not_flagged() -> None:
    # The count gate also uses a strict ``>`` comparison: count == limit is
    # still within bounds. Two functions at CCN >= threshold with a limit of 2
    # must NOT fail.
    metrics = [_make_metric(ccn=25), _make_metric(ccn=22)]

    failures = ccm.evaluate_gate(
        metrics,
        max_ccn=None,
        ccn_threshold=20,
        max_count_at_or_above=2,
    )

    assert failures == []


def test_evaluate_gate_reports_both_failures_when_both_gates_exceeded() -> None:
    # Independently exceeding both thresholds must surface both failure
    # messages — the report is additive so operators see every reason the
    # gate rejected the run.
    metrics = [
        _make_metric(ccn=50),  # exceeds max_ccn=10
        _make_metric(ccn=25),  # together with the 50 -> 2 functions >= 20
    ]

    failures = ccm.evaluate_gate(
        metrics,
        max_ccn=10,
        ccn_threshold=20,
        max_count_at_or_above=1,
    )

    assert len(failures) == 2
    joined = "\n".join(failures)
    assert "max CCN" in joined
    assert "count(CCN >=" in joined
