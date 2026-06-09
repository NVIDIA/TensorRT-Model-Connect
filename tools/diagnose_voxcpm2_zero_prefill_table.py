#!/usr/bin/env python3
"""Compare a VoxCPM2 zero-prefill table against HF TSLM prefill inputs.

The native runtime may package ``voxcpm2_zero_prefill_local_text_features_bf16``
so it can skip running LocEnc over all-zero audio features during text prefill.
This diagnostic checks that the serialized row for the active text length is
the same BF16 row observed at the Hugging Face TSLM prefill boundary.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))


ZERO_PREFILL_SECTION = "voxcpm2_zero_prefill_local_text_features_bf16"


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


def _load_bundle_section(bundle_path: Path, section_name: str) -> bytes:
    from tensorrt_model_connect.engine_defs.torch_trt.bundle_reader import (
        read_bundle_section,
    )

    return read_bundle_section(bundle_path, section_name)


def _parse_zero_prefill_table(section: bytes) -> dict[int, bytes]:
    if len(section) < 12:
        raise ValueError("VoxCPM2 zero-prefill section is shorter than its header")
    version, row_count, hidden_size = struct.unpack_from("<III", section, 0)
    if version != 1:
        raise ValueError(
            f"Unsupported VoxCPM2 zero-prefill table version {version}"
        )
    if hidden_size <= 0:
        raise ValueError("VoxCPM2 zero-prefill table has zero hidden size")

    offset = 12
    row_bytes = hidden_size * 2
    rows: dict[int, bytes] = {}
    for _ in range(row_count):
        if offset + 4 > len(section):
            raise ValueError("VoxCPM2 zero-prefill table is truncated in row header")
        text_steps = struct.unpack_from("<I", section, offset)[0]
        offset += 4
        if offset + row_bytes > len(section):
            raise ValueError("VoxCPM2 zero-prefill table is truncated in row data")
        rows[int(text_steps)] = section[offset : offset + row_bytes]
        offset += row_bytes
    if offset != len(section):
        raise ValueError("VoxCPM2 zero-prefill table has trailing bytes")
    return rows


def _hf_prefill_feature_rows(hf_dump_dir: Path) -> list[tuple[int, bytes]]:
    rows: list[tuple[int, bytes]] = []
    for record in _load_manifest(hf_dump_dir):
        if (
            record.get("phase") == "tslm_prefill"
            and record.get("direction") == "input"
            and record.get("name") == "local_text_features"
        ):
            dtype = str(record.get("dtype"))
            if dtype != "bfloat16":
                raise ValueError(
                    "VoxCPM2 zero-prefill diagnostic expected HF "
                    f"local_text_features as bfloat16, got {dtype!r}"
                )
            rows.append((int(record["step"]), Path(record["path"]).read_bytes()))
    rows.sort(key=lambda item: item[0])
    if not rows:
        raise ValueError(
            "No HF tslm_prefill input local_text_features rows found"
        )
    return rows


def _bf16_to_float(raw: bytes, element_index: int) -> float:
    bits = struct.unpack_from("<H", raw, element_index * 2)[0]
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def _bf16_bits(raw: bytes, element_index: int) -> str:
    return hex(struct.unpack_from("<H", raw, element_index * 2)[0])


def _first_row_mismatch(
    expected_rows: list[tuple[int, bytes]],
    actual_row: bytes,
) -> dict[str, Any] | None:
    for step, expected in expected_rows:
        if len(expected) != len(actual_row):
            return {
                "step": step,
                "metadata_differences": [
                    {
                        "field": "nbytes",
                        "hf": len(expected),
                        "bundle": len(actual_row),
                    }
                ],
            }
        if expected == actual_row:
            continue
        element_count = len(expected) // 2
        for element in range(element_count):
            byte_offset = element * 2
            if expected[byte_offset : byte_offset + 2] == actual_row[
                byte_offset : byte_offset + 2
            ]:
                continue
            return {
                "step": step,
                "first_different_element": element,
                "hf_value": _bf16_to_float(expected, element),
                "bundle_value": _bf16_to_float(actual_row, element),
                "hf_bits": _bf16_bits(expected, element),
                "bundle_bits": _bf16_bits(actual_row, element),
            }
    return None


def diagnose_zero_prefill_table(
    *,
    bundle_path: Path,
    hf_dump_dir: Path,
    text_steps: int | None = None,
) -> dict[str, Any]:
    hf_rows = _hf_prefill_feature_rows(hf_dump_dir)
    active_text_steps = text_steps if text_steps is not None else len(hf_rows)
    section = _load_bundle_section(bundle_path, ZERO_PREFILL_SECTION)
    table_rows = _parse_zero_prefill_table(section)
    actual_row = table_rows.get(active_text_steps)

    result: dict[str, Any] = {
        "bundle": str(bundle_path),
        "hf_dump_dir": str(hf_dump_dir),
        "section": ZERO_PREFILL_SECTION,
        "text_steps": active_text_steps,
        "hf_rows_checked": len(hf_rows),
        "table_row_count": len(table_rows),
        "matched": False,
        "first_mismatch": None,
    }
    if actual_row is None:
        result["first_mismatch"] = {
            "missing_table_row": active_text_steps,
            "available_rows": sorted(table_rows)[:10],
        }
        return result

    mismatch = _first_row_mismatch(hf_rows, actual_row)
    result["matched"] = mismatch is None
    result["first_mismatch"] = mismatch
    if hf_rows:
        result["hf_row0_first8"] = [
            _bf16_to_float(hf_rows[0][1], i) for i in range(min(8, len(hf_rows[0][1]) // 2))
        ]
        result["table_row_first8"] = [
            _bf16_to_float(actual_row, i) for i in range(min(8, len(actual_row) // 2))
        ]
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--hf-dump-dir", required=True, type=Path)
    parser.add_argument(
        "--text-steps",
        type=int,
        help="Text-step row to check; defaults to the number of HF prefill rows.",
    )
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)

    result = diagnose_zero_prefill_table(
        bundle_path=args.bundle,
        hf_dump_dir=args.hf_dump_dir,
        text_steps=args.text_steps,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if result["matched"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
