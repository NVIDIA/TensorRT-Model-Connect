from __future__ import annotations

import importlib.util
import json
import struct
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "tools" / "diagnose_voxcpm2_zero_prefill_table.py"


def _load_tool() -> Any:
    spec = importlib.util.spec_from_file_location(
        "diagnose_voxcpm2_zero_prefill_table",
        TOOL_PATH,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _bf16_raw(bits: list[int]) -> bytes:
    return b"".join(struct.pack("<H", value) for value in bits)


def _write_hf_row(root: Path, step: int, raw: bytes) -> None:
    raw_path = root / f"tslm_prefill_{step:06d}_input_local_text_features.raw"
    raw_path.write_bytes(raw)
    record = {
        "stage": "tslm",
        "engine_section": "hf_reference",
        "phase": "tslm_prefill",
        "step": step,
        "direction": "input",
        "name": "local_text_features",
        "dtype": "bfloat16",
        "shape": [1, len(raw) // 2],
        "nbytes": len(raw),
        "path": str(raw_path),
    }
    with (root / "manifest.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _table(rows: dict[int, bytes], hidden_size: int) -> bytes:
    payload = bytearray(struct.pack("<III", 1, len(rows), hidden_size))
    for text_steps, row in sorted(rows.items()):
        payload.extend(struct.pack("<I", text_steps))
        payload.extend(row)
    return bytes(payload)


def test_zero_prefill_table_diagnostic_reports_missing_table_row(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tool = _load_tool()
    row = _bf16_raw([0x3F80, 0x4000])
    _write_hf_row(tmp_path, 0, row)
    _write_hf_row(tmp_path, 1, row)
    monkeypatch.setattr(tool, "_load_bundle_section", lambda *_args: _table({1: row}, 2))

    result = tool.diagnose_zero_prefill_table(
        bundle_path=tmp_path / "bundle.trtfb",
        hf_dump_dir=tmp_path,
        text_steps=2,
    )

    assert result["matched"] is False
    assert result["first_mismatch"]["missing_table_row"] == 2


def test_zero_prefill_table_diagnostic_reports_first_bf16_mismatch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tool = _load_tool()
    expected = _bf16_raw([0x3F80, 0x4000, 0x4040])
    actual = _bf16_raw([0x3F80, 0x4080, 0x4040])
    _write_hf_row(tmp_path, 0, expected)

    monkeypatch.setattr(tool, "_load_bundle_section", lambda *_args: _table({1: actual}, 3))

    result = tool.diagnose_zero_prefill_table(
        bundle_path=tmp_path / "bundle.trtfb",
        hf_dump_dir=tmp_path,
    )

    assert result["matched"] is False
    assert result["first_mismatch"]["step"] == 0
    assert result["first_mismatch"]["first_different_element"] == 1
    assert result["first_mismatch"]["hf_bits"] == "0x4000"
    assert result["first_mismatch"]["bundle_bits"] == "0x4080"
    assert result["hf_row0_first8"] == [1.0, 2.0, 3.0]
    assert result["table_row_first8"] == [1.0, 4.0, 3.0]


def test_zero_prefill_table_diagnostic_accepts_matching_bundle_section(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tool = _load_tool()
    row = _bf16_raw([0x3F80, 0x4000])
    _write_hf_row(tmp_path, 0, row)
    _write_hf_row(tmp_path, 1, row)
    monkeypatch.setattr(tool, "_load_bundle_section", lambda *_args: _table({2: row}, 2))

    result = tool.diagnose_zero_prefill_table(
        bundle_path=tmp_path / "bundle.trtfb",
        hf_dump_dir=tmp_path,
    )

    assert result["matched"] is True
    assert result["first_mismatch"] is None
