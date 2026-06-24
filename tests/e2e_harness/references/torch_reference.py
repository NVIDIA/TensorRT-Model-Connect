"""Torch reference backend for custom models and golden snapshots.

Provides reference outputs for models that do not have a standard HF pipeline:
- PersonaPlex: loads reference tokens from .npy files
- Custom torch models: executes user-provided reference scripts
- Golden snapshots: loads pre-saved reference outputs

Used primarily for speech_to_speech (PersonaPlex) and models without
HF Transformers/Diffusers integration.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .. import save_full_stderr
from ..contracts import E2ECase, RunContext, StageOutput, StageSpec

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parents[3]


class TorchReference:
    """Reference backend using custom torch code or golden .npy snapshots."""

    @property
    def backend_name(self) -> str:
        return "torch_reference"

    def run_stage(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        # Dispatch based on task_strategy + stage
        task = case.task_strategy
        if task == "speech_to_speech":
            return self._run_speech_to_speech_ref(case, stage, ctx)
        if task == "text_to_audio":
            return self._run_text_to_audio_ref(case, stage, ctx)
        if task == "speech_to_text":
            return self._run_speech_to_text_ref(case, stage, ctx)
        if task == "neural_operator" and _is_supported_time_series_case(case):
            return self._run_time_series_ref(case, stage, ctx)
        # Generic: try to load golden snapshot
        return self._run_golden_snapshot(case, stage, ctx)

    def _run_speech_to_speech_ref(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Load reference tokens from .npy file for PersonaPlex-style models."""
        ref_tokens_path = case.inputs.get(
            "speech_reference_tokens",
            case.metadata.get("speech_reference_tokens", ""),
        )
        if not ref_tokens_path:
            return StageOutput(
                stage_name=stage.name,
                data={"error": "No speech_reference_tokens path in manifest"},
            )

        # Resolve relative path against project/tests/e2e directory
        if not os.path.isabs(ref_tokens_path):
            e2e_dir = Path(__file__).resolve().parents[2] / "e2e"
            ref_tokens_path = str(e2e_dir / ref_tokens_path)

        if not os.path.exists(ref_tokens_path):
            return StageOutput(
                stage_name=stage.name,
                data={"error": f"Reference tokens file not found: {ref_tokens_path}"},
            )

        try:
            import numpy as np
            ref_tokens = np.load(ref_tokens_path)
        except Exception as e:
            return StageOutput(
                stage_name=stage.name,
                data={"error": f"Failed to load reference tokens: {e}"},
            )

        return StageOutput(
            stage_name=stage.name,
            data={
                "reference_tokens": ref_tokens,
                "num_frames": ref_tokens.shape[0] if ref_tokens.ndim >= 1 else 0,
                "token_shape": list(ref_tokens.shape),
                "source_path": ref_tokens_path,
            },
            metadata={"backend": "torch_reference"},
        )

    def _run_text_to_audio_ref(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Run HF Bark pipeline as reference for text-to-audio."""
        model_id = case.hf_id
        prompt = case.inputs.get("prompt", "Hello, this is a test.")
        python = ctx.reference_python_path() or sys.executable

        script = f"""
import torch
import numpy as np
import json

from transformers import AutoProcessor, BarkModel

processor = AutoProcessor.from_pretrained({model_id!r})
model = BarkModel.from_pretrained({model_id!r})
model = model.to("cuda")

inputs = processor({prompt!r})
inputs = {{k: v.to("cuda") for k, v in inputs.items()}}

with torch.no_grad():
    output = model.generate(**inputs, do_sample=False)

audio = output.cpu().numpy().flatten()
result = {{
    "num_samples": len(audio),
    "rms": float(np.sqrt(np.mean(audio ** 2))),
    "duration_s": len(audio) / 24000.0,
    "sample_rate": 24000,
}}
print(json.dumps(result))

np.save("/tmp/hf_bark_audio.npy", audio)
"""
        t0 = time.monotonic()
        result = subprocess.run(
            [python, "-c", script],
            capture_output=True, text=True, timeout=600)
        elapsed = time.monotonic() - t0

        stderr_truncated, stderr_log = save_full_stderr(
            result.stderr or "", ctx.artifacts_dir or "",
            "torch_text_to_audio", case.name)
        data: dict = {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": stderr_truncated,
        }
        if stderr_log:
            data["stderr_log"] = stderr_log

        # Parse JSON output
        try:
            import json as json_mod
            parsed = json_mod.loads(result.stdout.strip())
            data.update(parsed)
        except Exception:
            pass

        # Load audio array
        npy_path = "/tmp/hf_bark_audio.npy"
        if os.path.exists(npy_path):
            try:
                import numpy as np
                data["audio_samples"] = np.load(npy_path)
            except Exception:
                pass

        return StageOutput(
            stage_name=stage.name,
            data=data,
            timing_s=elapsed,
            metadata={"backend": "torch_reference"},
        )

    def _run_speech_to_text_ref(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Run HF Whisper pipeline as reference for speech-to-text."""
        model_id = case.hf_id
        python = ctx.reference_python_path() or sys.executable

        # Resolve audio input
        audio_input = case.inputs.get("audio_path", "")
        if not audio_input:
            audio_input = case.metadata.get("test_input_audio", "")
        if audio_input and not os.path.isabs(audio_input):
            e2e_dir = Path(__file__).resolve().parents[2] / "e2e"
            audio_input = str(e2e_dir / audio_input)

        script = f"""
import torch
import numpy as np
import json

from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa

audio, sr = librosa.load({audio_input!r}, sr=16000)
processor = WhisperProcessor.from_pretrained({model_id!r})
model = WhisperForConditionalGeneration.from_pretrained({model_id!r})
model = model.to("cuda")

input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
input_features = input_features.to("cuda")

with torch.no_grad():
    predicted_ids = model.generate(input_features)

transcript = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
token_ids = predicted_ids[0].cpu().tolist()

result = {{
    "transcript": transcript,
    "token_ids": token_ids,
    "num_tokens": len(token_ids),
}}
print(json.dumps(result))
"""
        t0 = time.monotonic()
        result = subprocess.run(
            [python, "-c", script],
            capture_output=True, text=True, timeout=600)
        elapsed = time.monotonic() - t0

        stderr_truncated, stderr_log = save_full_stderr(
            result.stderr or "", ctx.artifacts_dir or "",
            "torch_speech_to_text", case.name)
        data: dict = {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": stderr_truncated,
        }
        if stderr_log:
            data["stderr_log"] = stderr_log

        try:
            import json as json_mod
            parsed = json_mod.loads(result.stdout.strip())
            data.update(parsed)
        except Exception:
            pass

        return StageOutput(
            stage_name=stage.name,
            data=data,
            text=data.get("transcript", ""),
            timing_s=elapsed,
            metadata={"backend": "torch_reference"},
        )

    def _run_golden_snapshot(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Load pre-saved golden reference outputs from .npy files."""
        golden_dir = case.inputs.get("golden_dir", "")
        if not golden_dir:
            golden_dir = os.path.join(ctx.engine_dir, f"{case.name}_golden")

        if not os.path.isabs(golden_dir):
            e2e_dir = Path(__file__).resolve().parents[2] / "e2e"
            golden_dir = str(e2e_dir / golden_dir)

        golden_file = os.path.join(golden_dir, f"{stage.name}.npy")
        if not os.path.exists(golden_file):
            return StageOutput(
                stage_name=stage.name,
                data={"error": f"Golden snapshot not found: {golden_file}"},
            )

        try:
            import numpy as np
            golden_data = np.load(golden_file, allow_pickle=True)
            return StageOutput(
                stage_name=stage.name,
                data={"golden_data": golden_data, "source_path": golden_file},
                metadata={"backend": "torch_reference"},
            )
        except Exception as e:
            return StageOutput(
                stage_name=stage.name,
                data={"error": f"Failed to load golden snapshot: {e}"},
            )

    def _run_time_series_ref(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        if stage.name != "full_inference":
            return StageOutput(
                stage_name=stage.name,
                data={"error": f"Unsupported neural_operator stage: {stage.name}"},
            )

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
from pathlib import Path

sys.path.insert(0, {str(PROJECT_DIR)!r})

from tests.e2e_harness.contracts import E2ECase
from tests.e2e_harness.references.torch_reference import (
    _resolve_model_path,
    _run_time_series_forward,
)

payload = json.loads({json.dumps(json.dumps(payload))})
case = E2ECase(
    name=payload["name"],
    hf_id=payload["hf_id"],
    family=payload["family"],
    runtime_strategy=payload["runtime_strategy"],
    task_strategy=payload["task_strategy"],
    inputs=payload["inputs"],
)
tensor, output_name = _run_time_series_forward(case)
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
            result.stderr or "", ctx.artifacts_dir or "",
            "torch_time_series", case.name)
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


def _torch():
    import torch

    return torch


def _transformers():
    import transformers

    return transformers


def _is_supported_time_series_case(case: E2ECase) -> bool:
    family = str(case.family or "").lower()
    runtime = str(case.runtime_strategy or "").lower()
    return family in {"patchtst", "patchtsmixer", "timesfm", "chronos_bolt"} or runtime in {
        "patchtst_trt",
        "patchtsmixer_trt",
        "timesfm_trt",
        "chronos_bolt_trt",
    }


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


def _infer_patchtst_task(config: Any) -> str:
    for attr in ("patchtst_task", "task_type", "problem_type"):
        raw = str(getattr(config, attr, "") or "").lower()
        if "class" in raw:
            return "classification"
        if "regress" in raw:
            return "regression"
        if "forecast" in raw or "predict" in raw:
            return "forecast"

    architectures = getattr(config, "architectures", []) or []
    if isinstance(architectures, str):
        architectures = [architectures]
    for arch in architectures:
        raw = str(arch).lower()
        if "class" in raw:
            return "classification"
        if "regress" in raw:
            return "regression"
        if "forecast" in raw or "predict" in raw:
            return "forecast"
    return "forecast"


def _run_time_series_forward(case: E2ECase):
    family = str(case.family or "").lower()
    if family == "patchtst":
        return _run_patchtst_forward(case)
    if family == "patchtsmixer":
        return _run_patchtsmixer_forward(case)
    if family == "timesfm":
        return _run_timesfm_forward(case)
    if family == "chronos_bolt":
        return _run_chronos_bolt_forward(case)
    raise ValueError(f"Unsupported time-series family for torch_reference: {case.family!r}")


def _run_patchtst_forward(case: E2ECase):
    torch = _torch()
    transformers = _transformers()

    model_path = _resolve_model_path(case.hf_id)
    config = transformers.AutoConfig.from_pretrained(model_path)
    task = _infer_patchtst_task(config)
    model_cls = {
        "classification": transformers.PatchTSTForClassification,
        "regression": transformers.PatchTSTForRegression,
        "forecast": transformers.PatchTSTForPrediction,
    }[task]
    output_name = {
        "classification": "prediction_logits",
        "regression": "regression_outputs",
        "forecast": "prediction_outputs",
    }[task]
    model = model_cls.from_pretrained(model_path, torch_dtype=torch.float32).eval()

    channels = max(int(getattr(config, "num_input_channels", 1) or 1), 1)
    context_length = max(int(getattr(config, "context_length", 1) or 1), 1)
    expected_len = context_length * channels
    raw_values = _coerce_numeric_sequence(case.inputs.get("field_input"), key="field_input")
    values = _align_window(raw_values, expected_len, 0.0)
    observed_mask = _align_window([1.0] * len(raw_values), expected_len, 0.0)

    past_values = torch.tensor(values, dtype=torch.float32).reshape(1, context_length, channels)
    past_observed_mask = torch.tensor(observed_mask, dtype=torch.float32).reshape(
        1, context_length, channels
    ).gt(0.5)
    with torch.no_grad():
        outputs = model(
            past_values=past_values,
            past_observed_mask=past_observed_mask,
            return_dict=True,
        )
    value = getattr(outputs, output_name)
    if isinstance(value, (tuple, list)) and value and all(torch.is_tensor(item) for item in value):
        value = torch.stack(list(value), dim=-1)
    return value.to(torch.float32), output_name


def _infer_patchtsmixer_task(config: Any) -> str:
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


def _run_patchtsmixer_forward(case: E2ECase):
    torch = _torch()
    transformers = _transformers()

    model_path = _resolve_model_path(case.hf_id)
    config = transformers.AutoConfig.from_pretrained(model_path)
    task = _infer_patchtsmixer_task(config)
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


def _run_timesfm_forward(case: E2ECase):
    torch = _torch()
    transformers = _transformers()

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


def _run_chronos_bolt_forward(case: E2ECase):
    torch = _torch()

    try:
        from chronos import ChronosBoltPipeline
    except ImportError as exc:
        raise RuntimeError(
            "chronos-forecasting is required for Chronos-Bolt reference parity"
        ) from exc

    model_path = _resolve_model_path(case.hf_id)
    pipe = ChronosBoltPipeline.from_pretrained(
        model_path,
        device_map="cpu",
        dtype=torch.float32,
    )

    raw_series = _coerce_numeric_sequence(case.inputs.get("branch_input"), key="branch_input")
    context = torch.tensor(raw_series, dtype=torch.float32)
    with torch.no_grad():
        quantiles = pipe.predict(
            context,
            prediction_length=pipe.model_prediction_length,
            limit_prediction_length=True,
        )
    return quantiles.to(torch.float32), "quantile_preds"


plugin = TorchReference()
