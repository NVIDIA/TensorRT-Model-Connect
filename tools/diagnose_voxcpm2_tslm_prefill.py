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
import sys
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
        if prefill_mode == _FULL_PREFILL_MODE:
            raw_hidden, _ = base_lm(inputs_embeds=combined_embed, is_causal=True)
            raw_hidden = raw_hidden.to(dtype=compute_dtype)
            semantic = fsq_layer(raw_hidden) * audio_mask.unsqueeze(0).unsqueeze(-1)
            semantic = semantic + raw_hidden * text_mask.unsqueeze(0).unsqueeze(-1)
            semantic = (
                semantic.squeeze(0)
                .to(dtype=compute_dtype)
                .detach()
                .cpu()
                .contiguous()
            )
        elif prefill_mode == _STEP_LOOP_MODE:
            base_lm.setup_cache(
                1,
                int(combined_embed.shape[1]),
                device,
                compute_dtype,
            )
            rows = []
            for position in range(int(combined_embed.shape[1])):
                position_id = torch.tensor(
                    [position],
                    dtype=torch.long,
                    device=device,
                )
                step_output = base_lm.forward_step(
                    combined_embed[:, position, :],
                    position_id,
                )
                if isinstance(step_output, tuple):
                    raw_hidden = step_output[0]
                else:
                    raw_hidden = step_output
                raw_hidden = raw_hidden.to(dtype=compute_dtype)
                semantic_row = fsq_layer(raw_hidden) * audio_mask[position].reshape(
                    1,
                    1,
                )
                semantic_row = semantic_row + raw_hidden * text_mask[position].reshape(
                    1,
                    1,
                )
                rows.append(semantic_row.squeeze(0).to(dtype=compute_dtype))
            semantic = torch.stack(rows, dim=0).detach().cpu().contiguous()
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

    config, state = _load_tslm_state(model_dir)
    results = []
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
            )
        )

    return {
        "model_dir": str(model_dir),
        "hf_dump_dir": str(hf_dump_dir),
        "device": device,
        "text_steps": len(steps),
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
        choices=("both", "upstream", "patched"),
        default="both",
        help=(
            "Eager variant to run. 'both' must run upstream before applying "
            "the export patch."
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
    args = parser.parse_args()

    result = diagnose(
        model_dir=args.model_dir,
        hf_dump_dir=args.hf_dump_dir,
        device=args.device,
        include_upstream=args.variant in {"both", "upstream"},
        include_patched=args.variant in {"both", "patched"},
        include_step_loop=args.include_step_loop,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
