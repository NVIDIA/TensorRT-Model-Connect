"""Serialization and deserialization for E2EResult and its nested types.

Converts between the dataclass domain model and plain JSON-serializable
dicts, enabling result.json persistence and cross-process communication.

All serialization is explicit: no pickle, no custom JSON encoders.
"""

from __future__ import annotations

from typing import Any, Dict

from .contracts import CompareResult, E2EResult, MetricResult, StageStatus


def _serialize_metric_result(metric_result: MetricResult) -> Dict[str, Any]:
    """Serialize a MetricResult to a JSON-safe dict."""
    d: Dict[str, Any] = {
        "value": metric_result.value,
        "threshold": metric_result.threshold,
        "operator": metric_result.operator,
        "passed": metric_result.passed,
    }
    if metric_result.note:
        d["note"] = metric_result.note
    return d


def _deserialize_metric_result(data: Dict[str, Any]) -> MetricResult:
    """Reconstruct a MetricResult from a dict."""
    return MetricResult(
        value=data.get("value", 0.0),
        threshold=data.get("threshold"),
        operator=data.get("operator", ">="),
        passed=data.get("passed", True),
        note=data.get("note", ""),
    )


def _serialize_compare_result(cr: CompareResult) -> Dict[str, Any]:
    """Serialize a CompareResult to a JSON-safe dict."""
    d: Dict[str, Any] = {
        "status": cr.status,
        "metrics": {
            name: _serialize_metric_result(metric_result)
            for name, metric_result in cr.metrics.items()
        },
        "message": cr.message,
    }
    if cr.composite_rule:
        d["composite_rule"] = cr.composite_rule
    return d


def _deserialize_compare_result(data: Dict[str, Any]) -> CompareResult:
    """Reconstruct a CompareResult from a dict.

    Handles both the new MetricResult-based format and the legacy format
    with separate ``metrics`` (float dict) + ``per_metric_pass`` (bool dict).
    """
    # Detect legacy format: metrics values are plain floats, not dicts
    raw_metrics = data.get("metrics", {})
    metrics: Dict[str, MetricResult] = {}

    if raw_metrics and isinstance(next(iter(raw_metrics.values())), dict):
        # New format: each metric is a MetricResult dict
        for name, mr_data in raw_metrics.items():
            metrics[name] = _deserialize_metric_result(mr_data)
    else:
        # Legacy format: metrics = {name: float}, per_metric_pass = {name: bool}
        per_metric_pass = data.get("per_metric_pass", {})
        for name, value in raw_metrics.items():
            metrics[name] = MetricResult(
                value=float(value) if value is not None else 0.0,
                passed=per_metric_pass.get(name, True),
            )

    # Map legacy passed/status fields
    if "status" in data:
        status = data["status"]
    elif "passed" in data:
        status = (StageStatus.PASSED.value if data["passed"]
                  else StageStatus.FAILED.value)
    else:
        status = StageStatus.FAILED.value

    # Map legacy gate_details to composite_rule
    composite_rule = data.get("composite_rule", "")
    if not composite_rule and "gate_details" in data:
        composite_rule = "; ".join(data["gate_details"])

    return CompareResult(
        stage_name=data.get("stage_name", ""),
        status=status,
        metrics=metrics,
        composite_rule=composite_rule,
        message=data.get("message", ""),
    )


def serialize_result(result: E2EResult) -> Dict[str, Any]:
    """Serialize an E2EResult to a JSON-serializable dict.

    Produces the consolidated result.json schema with all information
    in a single file.
    """
    d: Dict[str, Any] = {
        "case_name": result.case_name,
        "status": result.status,
        "failure_type": result.failure_type,
        "oracle_level": result.oracle_level,
        "timestamp": result.timestamp,
    }

    if result.case_config:
        d["case_config"] = result.case_config

    if result.env_fingerprint:
        d["env_fingerprint"] = dict(result.env_fingerprint)

    d["stages"] = {
        name: _serialize_compare_result(cr)
        for name, cr in result.stages.items()
    }

    if result.stage_outputs:
        d["stage_outputs"] = result.stage_outputs

    if result.timing:
        d["timing"] = dict(result.timing)

    if result.detailed_timing:
        d["detailed_timing"] = dict(result.detailed_timing)

    if result.commands:
        d["commands"] = result.commands

    if result.repro_commands:
        d["repro_commands"] = dict(result.repro_commands)

    if result.artifacts:
        d["artifacts"] = result.artifacts

    if result.log_file:
        d["log_file"] = result.log_file

    if result.determinism:
        d["determinism"] = dict(result.determinism)

    return d


def deserialize_result(data: Dict[str, Any]) -> E2EResult:
    """Reconstruct an E2EResult from a dict (e.g. loaded from result.json).

    Handles both the new consolidated format and the legacy format.
    Missing fields are filled with defaults. Unknown fields are ignored.
    """
    stages_raw = data.get("stages", {})
    stages: Dict[str, CompareResult] = {}
    for stage_name, cr_data in stages_raw.items():
        stages[stage_name] = _deserialize_compare_result(cr_data)

    return E2EResult(
        case_name=data.get("case_name", ""),
        status=data.get("status", "error"),
        failure_type=data.get("failure_type"),
        oracle_level=data.get("oracle_level", ""),
        stages=stages,
        determinism=data.get("determinism", {}),
        timing=data.get("timing", {}),
        detailed_timing=data.get("detailed_timing", {}),
        env_fingerprint=data.get("env_fingerprint", {}),
        timestamp=data.get("timestamp", ""),
        repro_commands=data.get("repro_commands", {}),
        case_config=data.get("case_config", {}),
        commands=data.get("commands", []),
        stage_outputs=data.get("stage_outputs", {}),
        artifacts=data.get("artifacts", {}),
        log_file=data.get("log_file", ""),
    )
