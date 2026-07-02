# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PatchTSMixer-owned torch reference backend."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .. import save_full_stderr
from ..contracts import E2ECase, RunContext, StageOutput, StageSpec


PROJECT_DIR = Path(__file__).resolve().parents[6]


class TorchReference:
    """Run the PatchTSMixer HF reference in an isolated subprocess."""

    @property
    def backend_name(self) -> str:
        return "torch_reference"

    def run_stage(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        if stage.name != "full_inference":
            return StageOutput(
                stage_name=stage.name,
                data={"error": f"Unsupported neural_operator stage: {stage.name}"},
            )
        return _run_reference_subprocess(case, stage, ctx)


def _run_reference_subprocess(
    case: E2ECase, stage: StageSpec, ctx: RunContext
) -> StageOutput:
    python = ctx.reference_python_path() or sys.executable
    payload = {
        "name": case.name,
        "hf_id": case.hf_id,
        "family": case.family,
        "runtime_strategy": case.runtime_strategy,
        "task_strategy": case.task_strategy,
        "inputs": case.inputs,
    }
    script = f"""
import json
import sys

sys.path.insert(0, {str(PROJECT_DIR)!r})

from tests.e2e_harness.contracts import E2ECase
from {__name__} import _resolve_model_path, _run_forward

payload = json.loads({json.dumps(json.dumps(payload))})
case = E2ECase(
    name=payload["name"],
    hf_id=payload["hf_id"],
    family=payload["family"],
    runtime_strategy=payload["runtime_strategy"],
    task_strategy=payload["task_strategy"],
    inputs=payload["inputs"],
)
tensor, output_name = _run_forward(case)
import torch
field = tensor.detach().cpu().to(dtype=torch.float32)
result = {{
    "output_field": field.reshape(-1).tolist(),
    "output_shape": list(field.shape),
    "reference_output_name": output_name,
    "model_path": _resolve_model_path(case.hf_id),
}}
print(json.dumps(result))
"""
    t0 = time.monotonic()
    result = subprocess.run(
        [python, "-c", script],
        capture_output=True,
        text=True,
        timeout=600,
    )
    elapsed = time.monotonic() - t0

    stderr_truncated, stderr_log = save_full_stderr(
        result.stderr or "", ctx.artifacts_dir or "", "torch_time_series", case.name
    )
    if result.returncode != 0:
        return StageOutput(
            stage_name=stage.name,
            data={"error": f"Time-series reference failed: {stderr_truncated}"},
            timing_s=elapsed,
            metadata={
                "backend": "torch_reference",
                "returncode": result.returncode,
                "stderr_log": stderr_log,
            },
        )

    try:
        parsed = json.loads(result.stdout.strip())
    except Exception as exc:
        return StageOutput(
            stage_name=stage.name,
            data={"error": f"Time-series reference emitted invalid JSON: {exc}"},
            timing_s=elapsed,
            metadata={"backend": "torch_reference", "stdout": result.stdout},
        )

    if stderr_log:
        parsed["stderr_log"] = stderr_log
    return StageOutput(
        stage_name=stage.name,
        data=parsed,
        timing_s=elapsed,
        metadata={"backend": "torch_reference"},
    )


def _resolve_model_path(model_id_or_path: str) -> str:
    path = Path(model_id_or_path)
    if path.is_absolute():
        return str(path)

    candidate = PROJECT_DIR / model_id_or_path
    if candidate.exists():
        return str(candidate)
    return model_id_or_path


def _coerce_numeric_sequence(raw: Any, *, key: str) -> list[float]:
    if raw is None:
        raise ValueError(f"Missing required numeric input: {key}")
    if isinstance(raw, (list, tuple)):
        return [float(v) for v in raw]
    if isinstance(raw, (int, float)):
        return [float(raw)]
    if isinstance(raw, str):
        value = raw.strip()
        if not value:
            raise ValueError(f"Numeric input {key} is empty")
        return [float(tok) for tok in value.replace(",", " ").split()]
    raise TypeError(f"Unsupported numeric input type for {key}: {type(raw).__name__}")


def _align_window(values: list[float], expected_len: int, fill_value: float) -> list[float]:
    if expected_len <= 0:
        return []
    out = [float(fill_value)] * expected_len
    if not values:
        return out
    copy_len = min(len(values), expected_len)
    out[-copy_len:] = [float(v) for v in values[-copy_len:]]
    return out


def _infer_task(config: Any) -> str:
    raw = str(getattr(config, "task_type", "") or "").lower()
    if "regress" in raw:
        return "regression"
    if "class" in raw:
        return "classification"
    if "pretrain" in raw:
        return "pretraining"
    if "forecast" in raw or "predict" in raw:
        return "prediction"

    architectures = getattr(config, "architectures", []) or []
    if isinstance(architectures, str):
        architectures = [architectures]
    for arch in architectures:
        raw = str(arch).lower()
        if "regress" in raw:
            return "regression"
        if "class" in raw:
            return "classification"
        if "pretrain" in raw:
            return "pretraining"
        if "predict" in raw:
            return "prediction"
    return "prediction"


def _run_forward(case: E2ECase):
    import torch
    import transformers

    model_path = _resolve_model_path(case.hf_id)
    config = transformers.AutoConfig.from_pretrained(model_path)
    task = _infer_task(config)
    model_cls = {
        "prediction": transformers.PatchTSMixerForPrediction,
        "pretraining": transformers.PatchTSMixerForPretraining,
        "regression": transformers.PatchTSMixerForRegression,
        "classification": transformers.PatchTSMixerForTimeSeriesClassification,
    }[task]
    if not hasattr(model_cls, "all_tied_weights_keys"):
        model_cls.all_tied_weights_keys = {}
    model = model_cls.from_pretrained(model_path, torch_dtype=torch.float32).eval()

    context_length = max(int(getattr(config, "context_length", 1) or 1), 1)
    channels = max(int(getattr(config, "num_input_channels", 1) or 1), 1)
    expected_len = context_length * channels
    raw_values = _coerce_numeric_sequence(case.inputs.get("field_input"), key="field_input")
    raw_mask = case.inputs.get("trunk_input")
    values = _align_window(raw_values, expected_len, 0.0)
    observed_mask = _align_window(
        _coerce_numeric_sequence(raw_mask, key="trunk_input") if raw_mask is not None else [],
        expected_len,
        1.0,
    )

    past_values = torch.tensor(values, dtype=torch.float32).reshape(1, context_length, channels)
    observed_mask_t = torch.tensor(observed_mask, dtype=torch.float32).reshape(
        1, context_length, channels
    )

    with torch.no_grad():
        if task in ("prediction", "pretraining"):
            outputs = model(
                past_values=past_values * observed_mask_t,
                observed_mask=observed_mask_t,
                return_loss=False,
                return_dict=True,
            )
        else:
            outputs = model(
                past_values=past_values * observed_mask_t,
                return_loss=False,
                return_dict=True,
            )

    for output_name in (
        "prediction_outputs",
        "regression_outputs",
        "classification_outputs",
        "logits",
    ):
        tensor = getattr(outputs, output_name, None)
        if tensor is not None:
            return tensor.to(torch.float32), output_name
    raise RuntimeError("PatchTSMixer reference produced no tensor output")


plugin = TorchReference()
