"""Tests for E2E result schema serialization."""

from __future__ import annotations

from tests.e2e_harness.contracts import E2EResult, E2EStatus, OracleLevel
from tests.e2e_harness.result_schema import deserialize_result, serialize_result


def test_weak_validation_reason_round_trips() -> None:
    result = E2EResult(
        case_name="invariant-only-model",
        status=E2EStatus.PASS.value,
        oracle_level=OracleLevel.L4_INVARIANTS.value,
        weak_validation_reason="oracle_level is L4_invariants",
    )

    data = serialize_result(result)
    restored = deserialize_result(data)

    assert data["weak_validation_reason"] == "oracle_level is L4_invariants"
    assert restored.weak_validation_reason == "oracle_level is L4_invariants"


def test_blank_weak_validation_reason_is_omitted() -> None:
    result = E2EResult(case_name="strong-model")

    data = serialize_result(result)

    assert "weak_validation_reason" not in data
