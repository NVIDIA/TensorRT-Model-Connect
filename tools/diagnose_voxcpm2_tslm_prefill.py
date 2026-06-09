#!/usr/bin/env python3
"""Compare VoxCPM2 TSLM full-prefill eager variants against HF tensor dumps.

This diagnostic replays the saved ``tslm_prefill`` rows emitted by the HF
reference tensor dump hook. It is intentionally narrower than the full audio E2E
flow: it proves whether the upstream MiniCPM eager path and the export-patched
MiniCPM eager path still match the HF reference before ONNX/TensorRT lowering.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))


_PREFILL_INPUTS = (
    "local_text_features",
    "text_tokens",
    "text_mask",
    "audio_mask",
)

_FULL_PREFILL_MODE = "full_prefill"
_STEP_LOOP_MODE = "step_loop"


def _load_manifest(dump_dir: Path) -> list[dict[str, Any]]:
    manifest_path = dump_dir / "manifest.jsonl"
    records: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            record["_line"] = line_no
            records.append(record)
    return records


def _record_key(record: dict[str, Any]) -> tuple[int, str, str]:
    return (
        int(record["step"]),
        str(record["direction"]),
        str(record["name"]),
    )


def _prefill_records_by_key(
    records: list[dict[str, Any]],
) -> dict[tuple[int, str, str], dict[str, Any]]:
    return {
        _record_key(record): record
        for record in records
        if record.get("phase") == "tslm_prefill"
    }


def _prefill_steps(records: list[dict[str, Any]]) -> list[int]:
    steps = sorted(
        {
            int(record["step"])
            for record in records
            if record.get("phase") == "tslm_prefill"
            and record.get("direction") == "output"
            and record.get("name") == "semantic_lm_states"
        }
    )
    if not steps:
        raise ValueError("No tslm_prefill semantic_lm_states outputs found")
    return steps


def _load_raw_tensor(record: dict[str, Any], torch_module: Any) -> Any:
    dtype = str(record["dtype"])
    shape = tuple(int(dim) for dim in record.get("shape", []))
    raw = bytearray(Path(record["path"]).read_bytes())
    if dtype == "bfloat16":
        tensor = torch_module.frombuffer(raw, dtype=torch_module.uint8).clone()
        return tensor.view(torch_module.bfloat16).reshape(shape)
    if dtype == "int32":
        return (
            torch_module.frombuffer(raw, dtype=torch_module.int32)
            .clone()
            .reshape(shape)
        )
    if dtype == "float32":
        return (
            torch_module.frombuffer(raw, dtype=torch_module.float32)
            .clone()
            .reshape(shape)
        )
    raise TypeError(f"Unsupported VoxCPM2 diagnostic tensor dtype {dtype!r}")


def _stack_prefill_tensor(
    by_key: dict[tuple[int, str, str], dict[str, Any]],
    steps: list[int],
    *,
    direction: str,
    name: str,
    torch_module: Any,
) -> Any:
    rows = []
    for step in steps:
        try:
            record = by_key[(step, direction, name)]
        except KeyError as exc:
            raise KeyError(
                f"Missing tslm_prefill {direction} tensor {name!r} at step {step}"
            ) from exc
        tensor = _load_raw_tensor(record, torch_module)
        if tensor.ndim > 0 and int(tensor.shape[0]) == 1:
            tensor = tensor.squeeze(0)
        rows.append(tensor)
    return torch_module.stack(rows, dim=0).contiguous()


def _first_mismatch(expected: Any, actual: Any) -> dict[str, Any]:
    if list(expected.shape) != list(actual.shape):
        return {
            "matched": False,
            "shape_mismatch": {
                "expected": [int(dim) for dim in expected.shape],
                "actual": [int(dim) for dim in actual.shape],
            },
        }
    diff = expected != actual
    if not bool(diff.any()):
        return {
            "matched": True,
            "shape": [int(dim) for dim in expected.shape],
            "first_different_element": None,
        }
    first = int(diff.flatten().nonzero()[0].item())
    expected_flat = expected.flatten()
    actual_flat = actual.flatten()
    return {
        "matched": False,
        "shape": [int(dim) for dim in expected.shape],
        "first_different_element": first,
        "expected_value": float(expected_flat[first].float().item()),
        "actual_value": float(actual_flat[first].float().item()),
        "expected_bits": _tensor_bits(expected_flat[first]),
        "actual_bits": _tensor_bits(actual_flat[first]),
    }


def _tensor_bits(value: Any) -> str:
    # The diagnostic compares BF16 semantic rows; keep this helper small and
    # explicit so mismatches can be correlated with raw tensor dump bytes.
    import torch

    if value.dtype != torch.bfloat16:
        return ""
    return hex(int(value.detach().cpu().view(torch.uint16).item()))


def _row_prefix(tensor: Any, count: int = 8) -> list[float]:
    return [float(value) for value in tensor[0, :count].float().detach().cpu()]


def _trt_dtype_to_torch(trt_dtype: Any, torch_module: Any) -> Any:
    dtype_text = str(trt_dtype)
    if "BF16" in dtype_text or "bfloat16" in dtype_text:
        return torch_module.bfloat16
    if "FLOAT" in dtype_text or "float32" in dtype_text:
        return torch_module.float32
    if "HALF" in dtype_text or "float16" in dtype_text:
        return torch_module.float16
    if "INT32" in dtype_text or "int32" in dtype_text:
        return torch_module.int32
    raise TypeError(f"Unsupported TensorRT dtype {dtype_text!r}")


def _pad_first_dim_to_shape(
    tensor: Any,
    shape: tuple[int, ...],
    *,
    torch_module: Any,
    name: str,
) -> Any:
    if any(dim < 0 for dim in shape):
        raise ValueError(f"TensorRT input {name!r} has dynamic shape {shape}")
    if tuple(int(dim) for dim in tensor.shape) == shape:
        return tensor.contiguous()
    if tensor.ndim != len(shape):
        raise ValueError(
            f"TensorRT input {name!r} rank {tensor.ndim} does not match "
            f"engine shape {shape}"
        )
    if tuple(int(dim) for dim in tensor.shape[1:]) != shape[1:]:
        raise ValueError(
            f"TensorRT input {name!r} trailing shape "
            f"{tuple(int(dim) for dim in tensor.shape[1:])} does not match "
            f"engine shape {shape}"
        )
    rows = int(tensor.shape[0])
    if rows > shape[0]:
        raise ValueError(
            f"TensorRT input {name!r} has {rows} rows, engine accepts {shape[0]}"
        )
    padded = torch_module.zeros(
        shape,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    padded[:rows] = tensor
    return padded.contiguous()


def _select_semantic_output_name(
    io_records: list[dict[str, Any]],
    expected_shape: tuple[int, ...],
) -> str:
    outputs = [record for record in io_records if record["mode"] == "output"]
    for record in outputs:
        if record["name"] == "semantic_lm_states":
            return str(record["name"])

    exact = [
        record
        for record in outputs
        if tuple(int(dim) for dim in record["shape"]) == expected_shape
    ]
    if len(exact) == 1:
        return str(exact[0]["name"])

    compatible = [
        record
        for record in outputs
        if len(record["shape"]) == len(expected_shape)
        and int(record["shape"][0]) >= expected_shape[0]
        and tuple(int(dim) for dim in record["shape"][1:]) == expected_shape[1:]
    ]
    if len(compatible) == 1:
        return str(compatible[0]["name"])

    raise ValueError(
        "Could not identify TensorRT semantic_lm_states output from "
        f"{[record['name'] for record in outputs]}"
    )


def _run_trt_prefill_plan(
    *,
    plan_path: Path,
    inputs: dict[str, Any],
    expected: Any,
    device: str,
) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("TensorRT prefill-plan diagnostic requires CUDA")

    from tensorrt_model_connect import trt_compat

    trt = trt_compat.get_trt()
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan_path.read_bytes())
    if engine is None:
        raise RuntimeError(f"Failed to deserialize TensorRT plan {plan_path}")
    context = engine.create_execution_context()

    io_records: list[dict[str, Any]] = []
    buffers: dict[str, Any] = {}
    cuda_device = torch.device(device if str(device).startswith("cuda") else "cuda")
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        dtype = engine.get_tensor_dtype(name)
        shape = tuple(int(dim) for dim in engine.get_tensor_shape(name))
        mode = engine.get_tensor_mode(name)
        mode_name = "input" if mode == trt.TensorIOMode.INPUT else "output"
        io_records.append(
            {
                "name": name,
                "dtype": str(dtype),
                "shape": [int(dim) for dim in shape],
                "mode": mode_name,
            }
        )
        torch_dtype = _trt_dtype_to_torch(dtype, torch)
        if mode_name == "input":
            if name not in inputs:
                raise KeyError(f"TensorRT plan input {name!r} is not in HF dump")
            source = inputs[name].to(device=cuda_device, dtype=torch_dtype)
            buffers[name] = _pad_first_dim_to_shape(
                source,
                shape,
                torch_module=torch,
                name=name,
            )
        else:
            buffers[name] = torch.empty(shape, device=cuda_device, dtype=torch_dtype)
        context.set_tensor_address(name, buffers[name].data_ptr())

    stream = torch.cuda.Stream(device=cuda_device)
    with torch.cuda.stream(stream):
        ok = context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    if not ok:
        raise RuntimeError(f"TensorRT plan execution failed for {plan_path}")

    expected_shape = tuple(int(dim) for dim in expected.shape)
    semantic_output_name = _select_semantic_output_name(io_records, expected_shape)
    actual = buffers[semantic_output_name].detach().cpu()
    if tuple(int(dim) for dim in actual.shape) != expected_shape:
        actual = actual[: expected_shape[0]]
    actual = actual.to(dtype=expected.dtype).contiguous()
    mismatch = _first_mismatch(expected, actual)
    mismatch.update(
        {
            "label": "trt_prefill_plan",
            "prefill_mode": "trt_plan",
            "plan_path": str(plan_path),
            "semantic_output_name": semantic_output_name,
            "engine_io": io_records,
            "compared_rows": expected_shape[0],
            "row0_expected_first8": _row_prefix(expected),
            "row0_actual_first8": _row_prefix(actual),
        }
    )
    return mismatch


def _tactic_label(tactic_sources: tuple[str, ...]) -> str:
    if not tactic_sources:
        return "default"
    return "+".join(tactic_sources).lower()


def _parse_tactic_source_sets(values: list[str] | None) -> list[tuple[str, ...]]:
    source_sets: list[tuple[str, ...]] = []
    for value in values or []:
        sources = tuple(
            part.strip().upper() for part in value.split(",") if part.strip()
        )
        if sources:
            source_sets.append(sources)
    return source_sets


def _set_tactic_sources(
    config: Any,
    trt_module: Any,
    tactic_sources: tuple[str, ...],
) -> None:
    if not tactic_sources:
        return
    if not hasattr(config, "set_tactic_sources"):
        raise RuntimeError("TensorRT builder config does not support tactic sources")
    tactic_source = getattr(trt_module, "TacticSource", None)
    if tactic_source is None:
        raise RuntimeError("TensorRT module does not expose TacticSource")
    mask = 0
    for source in tactic_sources:
        if not hasattr(tactic_source, source):
            raise ValueError(f"Unsupported TensorRT tactic source {source!r}")
        mask |= 1 << int(getattr(tactic_source, source))
    config.set_tactic_sources(mask)


def _run_trt_linear_projection(
    *,
    linear_module: Any,
    input_tensor: Any,
    expected: Any,
    device: str,
    label: str,
    tactic_sources: tuple[str, ...],
) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("TensorRT down-proj diagnostic requires CUDA")

    from tensorrt_model_connect import trt_compat

    class LinearProjectionWrapper(torch.nn.Module):
        def __init__(self, module: Any) -> None:
            super().__init__()
            self.module = module

        def forward(self, down_proj_input: Any) -> Any:
            return self.module(down_proj_input)

    wrapper = LinearProjectionWrapper(linear_module).eval()
    cuda_device = torch.device(device if str(device).startswith("cuda") else "cuda")
    trt = trt_compat.get_trt()
    logger = trt.Logger(trt.Logger.WARNING)

    with tempfile.TemporaryDirectory(prefix="trtmc_down_proj_") as tmpdir:
        onnx_path = Path(tmpdir) / "model.onnx"
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                (input_tensor,),
                str(onnx_path),
                opset_version=18,
                dynamo=False,
                external_data=True,
                input_names=["down_proj_input"],
                output_names=["down_proj_output"],
                dynamic_axes=None,
            )

        builder = trt.Builder(logger)
        network = builder.create_network(
            trt_compat.network_creation_flags(
                explicit_batch=True,
                strongly_typed=True,
            )
        )
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(onnx_path)):
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError("ONNX parsing failed:\n" + "\n".join(errors))

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        if hasattr(config, "clear_flag") and hasattr(trt, "BuilderFlag"):
            config.clear_flag(trt.BuilderFlag.TF32)
        _set_tactic_sources(config, trt, tactic_sources)

        plan = builder.build_serialized_network(network, config)
        if plan is None:
            raise RuntimeError("TensorRT down-proj engine build failed")
        plan_bytes = bytes(plan)

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan_bytes)
    if engine is None:
        raise RuntimeError("Failed to deserialize TensorRT down-proj engine")
    context = engine.create_execution_context()

    io_records: list[dict[str, Any]] = []
    buffers: dict[str, Any] = {}
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        dtype = engine.get_tensor_dtype(name)
        shape = tuple(int(dim) for dim in engine.get_tensor_shape(name))
        mode = engine.get_tensor_mode(name)
        mode_name = "input" if mode == trt.TensorIOMode.INPUT else "output"
        io_records.append(
            {
                "name": name,
                "dtype": str(dtype),
                "shape": [int(dim) for dim in shape],
                "mode": mode_name,
            }
        )
        torch_dtype = _trt_dtype_to_torch(dtype, torch)
        if mode_name == "input":
            buffers[name] = input_tensor.to(
                device=cuda_device,
                dtype=torch_dtype,
            ).contiguous()
        else:
            buffers[name] = torch.empty(
                shape,
                device=cuda_device,
                dtype=torch_dtype,
            )
        context.set_tensor_address(name, buffers[name].data_ptr())

    stream = torch.cuda.Stream(device=cuda_device)
    with torch.cuda.stream(stream):
        ok = context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    if not ok:
        raise RuntimeError("TensorRT down-proj plan execution failed")

    actual = next(
        buffers[record["name"]]
        for record in io_records
        if record["mode"] == "output"
    ).detach().cpu()
    if actual.ndim == 3 and int(actual.shape[0]) == 1:
        actual = actual.squeeze(0)
    actual = actual.to(dtype=expected.dtype).contiguous()
    mismatch = _first_mismatch(expected, actual)
    mismatch.update(
        {
            "label": label,
            "prefill_mode": "trt_down_proj",
            "tactic_sources": list(tactic_sources),
            "engine_io": io_records,
            "plan_bytes": len(plan_bytes),
            "row0_expected_first8": _row_prefix(expected),
            "row0_actual_first8": _row_prefix(actual),
        }
    )
    return mismatch


def _run_down_proj_trt_probe(
    *,
    config: dict[str, Any],
    state: dict[str, Any],
    inputs: dict[str, Any],
    device: str,
    layer_index: int,
    tactic_source_sets: list[tuple[str, ...]] | None = None,
) -> list[dict[str, Any]]:
    import torch
    from tensorrt_model_connect.families.voxcpm2 import component_builders
    from voxcpm.modules.layers import ScalarQuantizationLayer
    from voxcpm.modules.minicpm4 import MiniCPM4Config, MiniCPMModel

    component_builders._patch_minicpm_attention_gqa_for_torch_trt(torch)

    lm_config = dict(config["lm_config"])
    hidden_size = int(lm_config["hidden_size"])
    compute_dtype = torch.bfloat16

    base_lm = MiniCPMModel(MiniCPM4Config(**lm_config))
    base_lm.load_state_dict(
        _prefixed_state(state, "base_lm.", dtype=compute_dtype),
        strict=True,
    )
    base_lm.to(device=device, dtype=compute_dtype).eval()

    fsq_layer = ScalarQuantizationLayer(
        hidden_size,
        hidden_size,
        int(config.get("scalar_quantization_latent_dim", 512)),
        int(config.get("scalar_quantization_scale", 9)),
    )
    fsq_layer.load_state_dict(
        _prefixed_state(state, "fsq_layer.", dtype=compute_dtype),
        strict=True,
    )
    fsq_layer.to(device=device, dtype=compute_dtype).eval()

    if layer_index < 0 or layer_index >= len(base_lm.layers):
        raise ValueError(
            f"VoxCPM2 TSLM layer index {layer_index} is outside "
            f"0..{len(base_lm.layers) - 1}"
        )

    scale_emb = float(lm_config.get("scale_emb", 1.0))
    if not bool(lm_config.get("use_mup", False)):
        scale_emb = 1.0

    with torch.inference_mode():
        local_text_features = inputs["local_text_features"].to(
            device=device,
            dtype=compute_dtype,
        )
        text_tokens = inputs["text_tokens"].to(device=device, dtype=torch.long)
        text_mask = inputs["text_mask"].to(device=device, dtype=compute_dtype)
        audio_mask = inputs["audio_mask"].to(device=device, dtype=compute_dtype)

        text_embed = base_lm.embed_tokens(text_tokens.unsqueeze(0)) * scale_emb
        combined_embed = text_mask.unsqueeze(0).unsqueeze(-1) * text_embed
        combined_embed = combined_embed + (
            audio_mask.unsqueeze(0).unsqueeze(-1)
            * local_text_features.unsqueeze(0)
        )
        _, traces = _run_full_prefill(
            base_lm=base_lm,
            fsq_layer=fsq_layer,
            combined_embed=combined_embed,
            text_mask=text_mask,
            audio_mask=audio_mask,
            compute_dtype=compute_dtype,
            include_layer_traces=True,
            trace_layer_index=layer_index,
        )

    input_stage = _trace_layer_stage(layer_index, "mlp.down_proj_input")
    output_stage = _trace_layer_stage(layer_index, "mlp.down_proj")
    down_proj_input = traces[input_stage].to(
        device=device,
        dtype=compute_dtype,
    ).contiguous().clone()
    expected = traces[output_stage].to(
        device="cpu",
        dtype=compute_dtype,
    ).contiguous().clone()

    source_sets = [()]
    for sources in tactic_source_sets or []:
        if sources not in source_sets:
            source_sets.append(sources)

    results = []
    linear_module = base_lm.layers[layer_index].mlp.down_proj
    for tactic_sources in source_sets:
        label = (
            f"layer_{layer_index:02d}.mlp.down_proj."
            f"trt_{_tactic_label(tactic_sources)}"
        )
        result = _run_trt_linear_projection(
            linear_module=linear_module,
            input_tensor=down_proj_input,
            expected=expected,
            device=device,
            label=label,
            tactic_sources=tactic_sources,
        )
        result["layer_index"] = layer_index
        results.append(result)

    del base_lm, fsq_layer, traces, down_proj_input, expected
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    gc.collect()
    return results


def _record_trace_matrix(
    traces: dict[str, Any],
    *,
    stage: str,
    tensor: Any,
    dtype: Any,
) -> None:
    if tensor.ndim == 3 and int(tensor.shape[0]) == 1:
        tensor = tensor.squeeze(0)
    traces[stage] = tensor.to(dtype=dtype).detach().cpu().contiguous()


def _record_trace_row(
    trace_rows: dict[str, list[Any]],
    *,
    stage: str,
    tensor: Any,
    dtype: Any,
) -> None:
    if tensor.ndim == 2 and int(tensor.shape[0]) == 1:
        tensor = tensor.squeeze(0)
    trace_rows.setdefault(stage, []).append(
        tensor.to(dtype=dtype).detach().cpu().contiguous()
    )


def _run_mlp_with_optional_traces(
    *,
    mlp: Any,
    hidden_states: Any,
    layer_index: int,
    should_trace: bool,
    compute_dtype: Any,
    traces: dict[str, Any] | None = None,
    trace_rows: dict[str, list[Any]] | None = None,
) -> Any:
    if not should_trace:
        return mlp(hidden_states)

    handles = []

    def record(stage_suffix: str, tensor: Any) -> None:
        if not hasattr(tensor, "detach"):
            return
        stage = _trace_layer_stage(layer_index, stage_suffix)
        if trace_rows is not None:
            _record_trace_row(
                trace_rows,
                stage=stage,
                tensor=tensor,
                dtype=compute_dtype,
            )
            return
        if traces is not None:
            _record_trace_matrix(
                traces,
                stage=stage,
                tensor=tensor,
                dtype=compute_dtype,
            )

    def add_forward_hook(module_name: str, stage_suffix: str) -> None:
        module = getattr(mlp, module_name, None)
        if not hasattr(module, "register_forward_hook"):
            return

        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            if isinstance(output, (tuple, list)) and output:
                record(stage_suffix, output[0])
            else:
                record(stage_suffix, output)

        handles.append(module.register_forward_hook(hook))

    down_proj = getattr(mlp, "down_proj", None)
    if hasattr(down_proj, "register_forward_pre_hook"):

        def down_input_hook(_module: Any, inputs: Any) -> None:
            if inputs:
                record("mlp.down_proj_input", inputs[0])

        handles.append(down_proj.register_forward_pre_hook(down_input_hook))

    add_forward_hook("gate_proj", "mlp.gate_proj")
    add_forward_hook("up_proj", "mlp.up_proj")
    add_forward_hook("down_proj", "mlp.down_proj")

    try:
        return mlp(hidden_states)
    finally:
        for handle in handles:
            handle.remove()


def _trace_summary(
    expected_traces: dict[str, Any],
    actual_traces: dict[str, Any],
) -> dict[str, Any]:
    stages = []
    first_divergent_stage = None
    for stage, expected in expected_traces.items():
        actual = actual_traces.get(stage)
        if actual is None:
            mismatch = {
                "stage": stage,
                "matched": False,
                "missing_actual_stage": True,
            }
        else:
            mismatch = _first_mismatch(expected, actual)
            mismatch["stage"] = stage
        if first_divergent_stage is None and not bool(mismatch.get("matched")):
            first_divergent_stage = stage
        stages.append(mismatch)

    for stage in actual_traces:
        if stage not in expected_traces:
            mismatch = {
                "stage": stage,
                "matched": False,
                "extra_actual_stage": True,
            }
            if first_divergent_stage is None:
                first_divergent_stage = stage
            stages.append(mismatch)

    return {
        "first_divergent_stage": first_divergent_stage,
        "stages": stages,
    }


def _trace_layer_stage(layer_index: int, stage: str) -> str:
    return f"layer_{layer_index:02d}.{stage}"


def _apply_layer_residual(
    decoder_layer: Any,
    residual: Any,
    hidden_states: Any,
) -> Any:
    if bool(getattr(decoder_layer, "use_mup", False)):
        return residual + hidden_states * (
            decoder_layer.scale_depth / math.sqrt(decoder_layer.num_hidden_layers)
        )
    return residual + hidden_states


def _copy_layer_cache(
    layer_cache: tuple[Any, Any],
    updated_cache: tuple[Any, Any],
) -> None:
    layer_cache[0].copy_(updated_cache[0])
    layer_cache[1].copy_(updated_cache[1])


def _uses_explicit_trtmc_casts(decoder_layer: Any) -> bool:
    return bool(
        getattr(
            getattr(decoder_layer.self_attn, "__class__", object),
            "_trtmc_explicit_gqa_patch",
            False,
        )
    )


def _maybe_cast(tensor: Any, *, dtype: Any, enabled: bool) -> Any:
    if enabled:
        return tensor.to(dtype=dtype)
    return tensor


def _run_full_decoder_layer(
    *,
    decoder_layer: Any,
    hidden_states: Any,
    position_emb: Any,
    is_causal: bool,
    layer_index: int,
    trace_layer_index: int | None,
    traces: dict[str, Any],
    compute_dtype: Any,
) -> tuple[Any, Any]:
    should_trace = trace_layer_index == layer_index
    explicit_casts = _uses_explicit_trtmc_casts(decoder_layer)
    if should_trace:
        _record_trace_matrix(
            traces,
            stage=_trace_layer_stage(layer_index, "input"),
            tensor=hidden_states,
            dtype=compute_dtype,
        )

    residual = hidden_states
    hidden_states = decoder_layer.input_layernorm(hidden_states)
    hidden_states = _maybe_cast(
        hidden_states,
        dtype=compute_dtype,
        enabled=explicit_casts,
    )
    if should_trace:
        _record_trace_matrix(
            traces,
            stage=_trace_layer_stage(layer_index, "input_norm"),
            tensor=hidden_states,
            dtype=compute_dtype,
        )

    hidden_states, present_key_value = decoder_layer.self_attn(
        hidden_states=hidden_states,
        position_emb=position_emb,
        is_causal=is_causal,
    )
    hidden_states = _maybe_cast(
        hidden_states,
        dtype=compute_dtype,
        enabled=explicit_casts,
    )
    if should_trace:
        _record_trace_matrix(
            traces,
            stage=_trace_layer_stage(layer_index, "attention"),
            tensor=hidden_states,
            dtype=compute_dtype,
        )

    hidden_states = _apply_layer_residual(decoder_layer, residual, hidden_states)
    hidden_states = _maybe_cast(
        hidden_states,
        dtype=compute_dtype,
        enabled=explicit_casts,
    )
    if should_trace:
        _record_trace_matrix(
            traces,
            stage=_trace_layer_stage(layer_index, "attention_residual"),
            tensor=hidden_states,
            dtype=compute_dtype,
        )

    residual = hidden_states
    hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
    hidden_states = _maybe_cast(
        hidden_states,
        dtype=compute_dtype,
        enabled=explicit_casts,
    )
    if should_trace:
        _record_trace_matrix(
            traces,
            stage=_trace_layer_stage(layer_index, "post_attention_norm"),
            tensor=hidden_states,
            dtype=compute_dtype,
        )

    hidden_states = _run_mlp_with_optional_traces(
        mlp=decoder_layer.mlp,
        hidden_states=hidden_states,
        layer_index=layer_index,
        should_trace=should_trace,
        compute_dtype=compute_dtype,
        traces=traces,
    )
    hidden_states = _maybe_cast(
        hidden_states,
        dtype=compute_dtype,
        enabled=explicit_casts,
    )
    if should_trace:
        _record_trace_matrix(
            traces,
            stage=_trace_layer_stage(layer_index, "mlp"),
            tensor=hidden_states,
            dtype=compute_dtype,
        )

    hidden_states = _apply_layer_residual(decoder_layer, residual, hidden_states)
    hidden_states = _maybe_cast(
        hidden_states,
        dtype=compute_dtype,
        enabled=explicit_casts,
    )
    if should_trace:
        _record_trace_matrix(
            traces,
            stage=_trace_layer_stage(layer_index, "output"),
            tensor=hidden_states,
            dtype=compute_dtype,
        )

    return hidden_states, present_key_value


def _run_step_decoder_layer(
    *,
    decoder_layer: Any,
    hidden_states: Any,
    position_emb: Any,
    position_id: Any,
    layer_cache: tuple[Any, Any],
    layer_index: int,
    trace_layer_index: int | None,
    trace_rows: dict[str, list[Any]],
    compute_dtype: Any,
) -> Any:
    should_trace = trace_layer_index == layer_index
    explicit_casts = _uses_explicit_trtmc_casts(decoder_layer)
    if should_trace:
        _record_trace_row(
            trace_rows,
            stage=_trace_layer_stage(layer_index, "input"),
            tensor=hidden_states,
            dtype=compute_dtype,
        )

    residual = hidden_states
    hidden_states = decoder_layer.input_layernorm(hidden_states)
    hidden_states = _maybe_cast(
        hidden_states,
        dtype=compute_dtype,
        enabled=explicit_casts,
    )
    if should_trace:
        _record_trace_row(
            trace_rows,
            stage=_trace_layer_stage(layer_index, "input_norm"),
            tensor=hidden_states,
            dtype=compute_dtype,
        )

    attention_output = decoder_layer.self_attn.forward_step(
        hidden_states=hidden_states,
        position_emb=position_emb,
        position_id=position_id,
        kv_cache=layer_cache,
    )
    if isinstance(attention_output, tuple):
        hidden_states, updated_cache = attention_output
        _copy_layer_cache(layer_cache, updated_cache)
    else:
        hidden_states = attention_output
    hidden_states = _maybe_cast(
        hidden_states,
        dtype=compute_dtype,
        enabled=explicit_casts,
    )
    if should_trace:
        _record_trace_row(
            trace_rows,
            stage=_trace_layer_stage(layer_index, "attention"),
            tensor=hidden_states,
            dtype=compute_dtype,
        )

    hidden_states = _apply_layer_residual(decoder_layer, residual, hidden_states)
    hidden_states = _maybe_cast(
        hidden_states,
        dtype=compute_dtype,
        enabled=explicit_casts,
    )
    if should_trace:
        _record_trace_row(
            trace_rows,
            stage=_trace_layer_stage(layer_index, "attention_residual"),
            tensor=hidden_states,
            dtype=compute_dtype,
        )

    residual = hidden_states
    hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
    hidden_states = _maybe_cast(
        hidden_states,
        dtype=compute_dtype,
        enabled=explicit_casts,
    )
    if should_trace:
        _record_trace_row(
            trace_rows,
            stage=_trace_layer_stage(layer_index, "post_attention_norm"),
            tensor=hidden_states,
            dtype=compute_dtype,
        )

    hidden_states = _run_mlp_with_optional_traces(
        mlp=decoder_layer.mlp,
        hidden_states=hidden_states,
        layer_index=layer_index,
        should_trace=should_trace,
        compute_dtype=compute_dtype,
        trace_rows=trace_rows,
    )
    hidden_states = _maybe_cast(
        hidden_states,
        dtype=compute_dtype,
        enabled=explicit_casts,
    )
    if should_trace:
        _record_trace_row(
            trace_rows,
            stage=_trace_layer_stage(layer_index, "mlp"),
            tensor=hidden_states,
            dtype=compute_dtype,
        )

    hidden_states = _apply_layer_residual(decoder_layer, residual, hidden_states)
    hidden_states = _maybe_cast(
        hidden_states,
        dtype=compute_dtype,
        enabled=explicit_casts,
    )
    if should_trace:
        _record_trace_row(
            trace_rows,
            stage=_trace_layer_stage(layer_index, "output"),
            tensor=hidden_states,
            dtype=compute_dtype,
        )

    return hidden_states


def _load_tslm_state(model_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    from safetensors.torch import load_file

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    state = load_file(str(model_dir / "model.safetensors"), device="cpu")
    return config, state


def _selected_variant_runs(
    *,
    include_upstream: bool,
    include_patched: bool,
    include_step_loop: bool,
) -> list[tuple[str, bool, str]]:
    runs: list[tuple[str, bool, str]] = []
    if include_upstream:
        runs.append(("upstream_full_prefill", False, _FULL_PREFILL_MODE))
        if include_step_loop:
            runs.append(("upstream_step_loop", False, _STEP_LOOP_MODE))
    if include_patched:
        runs.append(("patched_export_full_prefill", True, _FULL_PREFILL_MODE))
        if include_step_loop:
            runs.append(("patched_export_step_loop", True, _STEP_LOOP_MODE))
    return runs


def _prefixed_state(
    state: dict[str, Any],
    prefix: str,
    *,
    dtype: Any,
) -> dict[str, Any]:
    return {
        key[len(prefix) :]: value.to(dtype=dtype)
        for key, value in state.items()
        if key.startswith(prefix)
    }


def _store_layer_cache(
    base_lm: Any,
    layer_index: int,
    updated_cache: tuple[Any, Any],
) -> None:
    kv_cache_obj = getattr(base_lm, "kv_cache", None)
    cache_tensor = getattr(kv_cache_obj, "kv_cache", None)
    if cache_tensor is None:
        return
    key_cache, value_cache = updated_cache
    cache_tensor[0, layer_index].copy_(key_cache)
    cache_tensor[1, layer_index].copy_(value_cache)


def _run_full_prefill(
    *,
    base_lm: Any,
    fsq_layer: Any,
    combined_embed: Any,
    text_mask: Any,
    audio_mask: Any,
    compute_dtype: Any,
    include_layer_traces: bool,
    trace_layer_index: int | None,
) -> tuple[Any, dict[str, Any]]:
    import torch

    traces: dict[str, Any] = {}
    if base_lm.rope_emb is not None:
        position_ids = torch.arange(
            0,
            combined_embed.size(1),
            dtype=torch.long,
            device=combined_embed.device,
        )
        position_emb = base_lm.rope_emb(position_ids)
    else:
        position_emb = None

    hidden_states = combined_embed
    if include_layer_traces:
        _record_trace_matrix(
            traces,
            stage="embedding",
            tensor=hidden_states,
            dtype=compute_dtype,
        )

    for layer_index, decoder_layer in enumerate(base_lm.layers):
        if trace_layer_index == layer_index:
            hidden_states, _ = _run_full_decoder_layer(
                decoder_layer=decoder_layer,
                hidden_states=hidden_states,
                position_emb=position_emb,
                is_causal=True,
                layer_index=layer_index,
                trace_layer_index=trace_layer_index,
                traces=traces,
                compute_dtype=compute_dtype,
            )
        else:
            layer_output = decoder_layer(hidden_states, position_emb, True)
            if isinstance(layer_output, tuple):
                hidden_states = layer_output[0]
            else:
                hidden_states = layer_output
        hidden_states = hidden_states.to(dtype=compute_dtype)
        if include_layer_traces:
            _record_trace_matrix(
                traces,
                stage=f"layer_{layer_index:02d}",
                tensor=hidden_states,
                dtype=compute_dtype,
            )

    raw_hidden = base_lm.norm(hidden_states).to(dtype=compute_dtype)
    if include_layer_traces:
        _record_trace_matrix(
            traces,
            stage="final_norm",
            tensor=raw_hidden,
            dtype=compute_dtype,
        )

    semantic = fsq_layer(raw_hidden) * audio_mask.unsqueeze(0).unsqueeze(-1)
    semantic = semantic + raw_hidden * text_mask.unsqueeze(0).unsqueeze(-1)
    semantic = semantic.squeeze(0).to(dtype=compute_dtype).detach().cpu().contiguous()
    if include_layer_traces:
        traces["semantic"] = semantic
    return semantic, traces


def _run_step_loop(
    *,
    base_lm: Any,
    fsq_layer: Any,
    combined_embed: Any,
    text_mask: Any,
    audio_mask: Any,
    device: str,
    compute_dtype: Any,
    include_layer_traces: bool,
    trace_layer_index: int | None,
) -> tuple[Any, dict[str, Any]]:
    import torch

    base_lm.setup_cache(1, int(combined_embed.shape[1]), device, compute_dtype)
    trace_rows: dict[str, list[Any]] = {}
    rows = []
    for position in range(int(combined_embed.shape[1])):
        position_id = torch.tensor(
            [position],
            dtype=torch.long,
            device=device,
        )
        if base_lm.rope_emb is not None:
            position_emb = base_lm.rope_emb(position_id)
        else:
            position_emb = None

        hidden_states = combined_embed[:, position, :]
        if include_layer_traces:
            _record_trace_row(
                trace_rows,
                stage="embedding",
                tensor=hidden_states,
                dtype=compute_dtype,
            )

        for layer_index, decoder_layer in enumerate(base_lm.layers):
            layer_cache = base_lm.kv_cache.get_layer_cache(layer_index)
            if trace_layer_index == layer_index:
                hidden_states = _run_step_decoder_layer(
                    decoder_layer=decoder_layer,
                    hidden_states=hidden_states,
                    position_emb=position_emb,
                    position_id=position_id,
                    layer_cache=layer_cache,
                    layer_index=layer_index,
                    trace_layer_index=trace_layer_index,
                    trace_rows=trace_rows,
                    compute_dtype=compute_dtype,
                )
            else:
                layer_output = decoder_layer.forward_step(
                    hidden_states,
                    position_emb,
                    position_id,
                    layer_cache,
                )
                if isinstance(layer_output, tuple):
                    hidden_states, updated_cache = layer_output
                    _store_layer_cache(base_lm, layer_index, updated_cache)
                else:
                    hidden_states = layer_output
            hidden_states = hidden_states.to(dtype=compute_dtype)
            if include_layer_traces:
                _record_trace_row(
                    trace_rows,
                    stage=f"layer_{layer_index:02d}",
                    tensor=hidden_states,
                    dtype=compute_dtype,
                )

        raw_hidden = base_lm.norm(hidden_states).to(dtype=compute_dtype)
        if include_layer_traces:
            _record_trace_row(
                trace_rows,
                stage="final_norm",
                tensor=raw_hidden,
                dtype=compute_dtype,
            )

        semantic_row = fsq_layer(raw_hidden) * audio_mask[position].reshape(1, 1)
        semantic_row = semantic_row + raw_hidden * text_mask[position].reshape(1, 1)
        semantic_row = semantic_row.to(dtype=compute_dtype)
        if include_layer_traces:
            _record_trace_row(
                trace_rows,
                stage="semantic",
                tensor=semantic_row,
                dtype=compute_dtype,
            )
        rows.append(semantic_row.squeeze(0).to(dtype=compute_dtype))

    semantic = torch.stack(rows, dim=0).detach().cpu().contiguous()
    traces = {
        stage: torch.stack(stage_rows, dim=0).contiguous()
        for stage, stage_rows in trace_rows.items()
    }
    return semantic, traces


def _run_variant(
    *,
    label: str,
    apply_export_patch: bool,
    prefill_mode: str,
    config: dict[str, Any],
    state: dict[str, Any],
    inputs: dict[str, Any],
    expected: Any,
    device: str,
    include_layer_traces: bool,
    trace_layer_index: int | None,
) -> dict[str, Any]:
    import torch
    from tensorrt_model_connect.families.voxcpm2 import component_builders
    from voxcpm.modules.layers import ScalarQuantizationLayer
    from voxcpm.modules.minicpm4 import MiniCPM4Config, MiniCPMModel

    if apply_export_patch:
        component_builders._patch_minicpm_attention_gqa_for_torch_trt(torch)

    lm_config = dict(config["lm_config"])
    hidden_size = int(lm_config["hidden_size"])
    compute_dtype = torch.bfloat16

    base_lm = MiniCPMModel(MiniCPM4Config(**lm_config))
    base_lm.load_state_dict(
        _prefixed_state(state, "base_lm.", dtype=compute_dtype),
        strict=True,
    )
    base_lm.to(device=device, dtype=compute_dtype).eval()

    fsq_layer = ScalarQuantizationLayer(
        hidden_size,
        hidden_size,
        int(config.get("scalar_quantization_latent_dim", 512)),
        int(config.get("scalar_quantization_scale", 9)),
    )
    fsq_layer.load_state_dict(
        _prefixed_state(state, "fsq_layer.", dtype=compute_dtype),
        strict=True,
    )
    fsq_layer.to(device=device, dtype=compute_dtype).eval()

    scale_emb = float(lm_config.get("scale_emb", 1.0))
    if not bool(lm_config.get("use_mup", False)):
        scale_emb = 1.0

    with torch.inference_mode():
        local_text_features = inputs["local_text_features"].to(
            device=device,
            dtype=compute_dtype,
        )
        text_tokens = inputs["text_tokens"].to(device=device, dtype=torch.long)
        text_mask = inputs["text_mask"].to(device=device, dtype=compute_dtype)
        audio_mask = inputs["audio_mask"].to(device=device, dtype=compute_dtype)

        text_embed = base_lm.embed_tokens(text_tokens.unsqueeze(0)) * scale_emb
        combined_embed = text_mask.unsqueeze(0).unsqueeze(-1) * text_embed
        combined_embed = combined_embed + (
            audio_mask.unsqueeze(0).unsqueeze(-1)
            * local_text_features.unsqueeze(0)
        )
        record_traces = include_layer_traces or trace_layer_index is not None
        if prefill_mode == _FULL_PREFILL_MODE:
            semantic, _ = _run_full_prefill(
                base_lm=base_lm,
                fsq_layer=fsq_layer,
                combined_embed=combined_embed,
                text_mask=text_mask,
                audio_mask=audio_mask,
                compute_dtype=compute_dtype,
                include_layer_traces=False,
                trace_layer_index=None,
            )
        elif prefill_mode == _STEP_LOOP_MODE:
            full_traces: dict[str, Any] = {}
            if record_traces:
                _, full_traces = _run_full_prefill(
                    base_lm=base_lm,
                    fsq_layer=fsq_layer,
                    combined_embed=combined_embed,
                    text_mask=text_mask,
                    audio_mask=audio_mask,
                    compute_dtype=compute_dtype,
                    include_layer_traces=record_traces,
                    trace_layer_index=trace_layer_index,
                )
            semantic, step_traces = _run_step_loop(
                base_lm=base_lm,
                fsq_layer=fsq_layer,
                combined_embed=combined_embed,
                text_mask=text_mask,
                audio_mask=audio_mask,
                device=device,
                compute_dtype=compute_dtype,
                include_layer_traces=record_traces,
                trace_layer_index=trace_layer_index,
            )
        else:
            raise ValueError(f"Unsupported VoxCPM2 TSLM prefill mode {prefill_mode!r}")

    mismatch = _first_mismatch(expected, semantic)
    mismatch.update(
        {
            "label": label,
            "prefill_mode": prefill_mode,
            "row0_expected_first8": _row_prefix(expected),
            "row0_actual_first8": _row_prefix(semantic),
        }
    )
    if prefill_mode == _STEP_LOOP_MODE and record_traces:
        mismatch["full_vs_step_trace"] = _trace_summary(full_traces, step_traces)

    del base_lm, fsq_layer, semantic
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    gc.collect()
    return mismatch


def diagnose(
    *,
    model_dir: Path,
    hf_dump_dir: Path,
    device: str,
    include_upstream: bool = True,
    include_patched: bool = True,
    include_step_loop: bool = False,
    include_layer_traces: bool = False,
    trace_layer_index: int | None = None,
    trt_prefill_plan: Path | None = None,
    trt_down_proj_layer: int | None = None,
    trt_down_proj_tactic_source_sets: list[tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    import torch

    records = _load_manifest(hf_dump_dir)
    by_key = _prefill_records_by_key(records)
    steps = _prefill_steps(records)
    inputs = {
        name: _stack_prefill_tensor(
            by_key,
            steps,
            direction="input",
            name=name,
            torch_module=torch,
        )
        for name in _PREFILL_INPUTS
    }
    expected = _stack_prefill_tensor(
        by_key,
        steps,
        direction="output",
        name="semantic_lm_states",
        torch_module=torch,
    )

    results = []
    needs_tslm_state = (
        include_upstream or include_patched or trt_down_proj_layer is not None
    )
    config: dict[str, Any] | None = None
    state: dict[str, Any] | None = None
    if needs_tslm_state:
        config, state = _load_tslm_state(model_dir)

    if include_upstream or include_patched:
        if config is None or state is None:
            raise RuntimeError("VoxCPM2 TSLM state was not loaded")
        for label, apply_export_patch, prefill_mode in _selected_variant_runs(
            include_upstream=include_upstream,
            include_patched=include_patched,
            include_step_loop=include_step_loop,
        ):
            results.append(
                _run_variant(
                    label=label,
                    apply_export_patch=apply_export_patch,
                    prefill_mode=prefill_mode,
                    config=config,
                    state=state,
                    inputs=inputs,
                    expected=expected,
                    device=device,
                    include_layer_traces=include_layer_traces,
                    trace_layer_index=trace_layer_index,
                )
            )
    if trt_prefill_plan is not None:
        results.append(
            _run_trt_prefill_plan(
                plan_path=trt_prefill_plan,
                inputs=inputs,
                expected=expected,
                device=device,
            )
        )
    if trt_down_proj_layer is not None:
        if config is None or state is None:
            raise RuntimeError("VoxCPM2 TSLM state was not loaded")
        results.extend(
            _run_down_proj_trt_probe(
                config=config,
                state=state,
                inputs=inputs,
                device=device,
                layer_index=trt_down_proj_layer,
                tactic_source_sets=trt_down_proj_tactic_source_sets,
            )
        )

    return {
        "model_dir": str(model_dir),
        "hf_dump_dir": str(hf_dump_dir),
        "device": device,
        "text_steps": len(steps),
        "include_layer_traces": include_layer_traces,
        "trace_layer_index": trace_layer_index,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--hf-dump-dir", required=True, type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--variant",
        choices=("both", "upstream", "patched", "none"),
        default="both",
        help=(
            "Eager variant to run. 'both' must run upstream before applying "
            "the export patch; 'none' runs only the TensorRT plan diagnostic."
        ),
    )
    parser.add_argument(
        "--include-step-loop",
        action="store_true",
        help=(
            "Also replay the same rows through MiniCPM forward_step so TSLM "
            "refresh-path drift can be distinguished from full-prefill drift."
        ),
    )
    parser.add_argument(
        "--include-layer-traces",
        action="store_true",
        help=(
            "For step-loop variants, also compare MiniCPM layer-boundary "
            "hidden states against the full-prefill replay."
        ),
    )
    parser.add_argument(
        "--trace-layer-substeps",
        type=int,
        metavar="LAYER_INDEX",
        help=(
            "For step-loop variants, trace RMSNorm, attention, residual, and "
            "MLP substeps inside this MiniCPM layer. Implies layer tracing."
        ),
    )
    parser.add_argument(
        "--trt-prefill-plan",
        type=Path,
        help=(
            "Optional serialized TensorRT TSLM prefill plan to run against "
            "the stacked HF prefill rows."
        ),
    )
    parser.add_argument(
        "--trt-down-proj-layer",
        type=int,
        metavar="LAYER_INDEX",
        help=(
            "Build and run an isolated TensorRT plan for the traced "
            "layer_N.mlp.down_proj projection from the HF full-prefill rows."
        ),
    )
    parser.add_argument(
        "--trt-down-proj-tactic-sources",
        action="append",
        default=[],
        metavar="SOURCE[,SOURCE...]",
        help=(
            "Optional TensorRT tactic-source set for the isolated down-proj "
            "probe, e.g. CUBLAS_LT or CUBLAS,CUBLAS_LT. Can be repeated. "
            "The default no-source-restriction probe always runs too."
        ),
    )
    args = parser.parse_args()

    result = diagnose(
        model_dir=args.model_dir,
        hf_dump_dir=args.hf_dump_dir,
        device=args.device,
        include_upstream=args.variant in {"both", "upstream"},
        include_patched=args.variant in {"both", "patched"},
        include_step_loop=args.include_step_loop,
        include_layer_traces=args.include_layer_traces,
        trace_layer_index=args.trace_layer_substeps,
        trt_prefill_plan=args.trt_prefill_plan,
        trt_down_proj_layer=args.trt_down_proj_layer,
        trt_down_proj_tactic_source_sets=_parse_tactic_source_sets(
            args.trt_down_proj_tactic_sources
        ),
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
