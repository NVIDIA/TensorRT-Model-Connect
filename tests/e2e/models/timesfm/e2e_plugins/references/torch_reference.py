"""TimesFM-owned torch reference backend."""

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
    """Run the TimesFM HF reference in an isolated subprocess."""

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


def _run_forward(case: E2ECase):
    import torch
    import transformers

    model_path = _resolve_model_path(case.hf_id)
    model = transformers.TimesFmModelForPrediction.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
    ).eval()
    context_length = max(int(getattr(model.config, "context_length", 1) or 1), 1)

    raw_series = _coerce_numeric_sequence(case.inputs.get("branch_input"), key="branch_input")
    series = _align_window(raw_series, context_length, 0.0)

    trunk = case.inputs.get("trunk_input")
    if trunk is None:
        freq_value = 0
    else:
        trunk_values = _coerce_numeric_sequence(trunk, key="trunk_input")
        freq_value = int(round(trunk_values[0])) if trunk_values else 0

    padding = [1] * context_length
    valid_len = min(len(raw_series), context_length)
    if valid_len > 0:
        padding[-valid_len:] = [0] * valid_len

    series_t = torch.tensor(series, dtype=torch.float32).reshape(1, context_length)
    padding_t = torch.tensor(padding, dtype=torch.int32).reshape(1, context_length)
    freq_t = torch.tensor([freq_value], dtype=torch.int32)
    with torch.no_grad():
        decoder_output = model.decoder(
            past_values=series_t,
            past_values_padding=padding_t,
            freq=freq_t.reshape(-1, 1).to(torch.long),
            output_attentions=False,
            output_hidden_states=False,
        )
        fprop_outputs = model._postprocess_output(
            decoder_output.last_hidden_state,
            (decoder_output.loc, decoder_output.scale),
        )

    output_patch_len = int(model.config.horizon_length)
    full_outputs = fprop_outputs[:, -1, :output_patch_len, :]
    full_outputs = full_outputs[:, : model.config.horizon_length, :]
    mean_outputs = full_outputs[:, :, 0]
    return mean_outputs.to(torch.float32), "mean_predictions"


plugin = TorchReference()
