from __future__ import annotations

import importlib.util
import json
import struct
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "tools" / "diagnose_voxcpm2_tslm_prefill.py"


def _load_tool() -> Any:
    spec = importlib.util.spec_from_file_location(
        "diagnose_voxcpm2_tslm_prefill",
        TOOL_PATH,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_record(
    root: Path,
    manifest: Path,
    *,
    step: int,
    direction: str,
    name: str,
    dtype: str,
    shape: list[int],
    raw: bytes,
) -> None:
    raw_path = root / f"tslm_prefill_{step:06d}_{direction}_{name}.raw"
    raw_path.write_bytes(raw)
    record = {
        "stage": "tslm",
        "engine_section": "hf_reference",
        "phase": "tslm_prefill",
        "step": step,
        "direction": direction,
        "name": name,
        "dtype": dtype,
        "shape": shape,
        "nbytes": len(raw),
        "path": str(raw_path),
    }
    with manifest.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def test_voxcpm2_tslm_prefill_probe_loads_manifest_rows(tmp_path: Path) -> None:
    tool = _load_tool()
    manifest = tmp_path / "manifest.jsonl"

    for step, token in enumerate((101, 202)):
        _write_record(
            tmp_path,
            manifest,
            step=step,
            direction="input",
            name="text_tokens",
            dtype="int32",
            shape=[1],
            raw=struct.pack("<i", token),
        )
        _write_record(
            tmp_path,
            manifest,
            step=step,
            direction="output",
            name="semantic_lm_states",
            dtype="bfloat16",
            shape=[1, 2],
            raw=struct.pack("<HH", 0x3F80 + step, 0x4000 + step),
        )

    records = tool._load_manifest(tmp_path)
    by_key = tool._prefill_records_by_key(records)
    steps = tool._prefill_steps(records)
    tokens = tool._stack_prefill_tensor(
        by_key,
        steps,
        direction="input",
        name="text_tokens",
        torch_module=torch,
    )
    semantic = tool._stack_prefill_tensor(
        by_key,
        steps,
        direction="output",
        name="semantic_lm_states",
        torch_module=torch,
    )

    assert steps == [0, 1]
    assert tokens.tolist() == [101, 202]
    assert semantic.shape == (2, 2)
    assert semantic.dtype == torch.bfloat16


def test_voxcpm2_tslm_prefill_probe_reports_first_bf16_mismatch() -> None:
    tool = _load_tool()
    expected = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    actual = torch.tensor([1.0, 4.0], dtype=torch.bfloat16)

    mismatch = tool._first_mismatch(expected, actual)

    assert mismatch["matched"] is False
    assert mismatch["first_different_element"] == 1
    assert mismatch["expected_value"] == 2.0
    assert mismatch["actual_value"] == 4.0
    assert mismatch["expected_bits"] == "0x4000"
    assert mismatch["actual_bits"] == "0x4080"
