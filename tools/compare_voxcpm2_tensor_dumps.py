#!/usr/bin/env python3
"""Compare VoxCPM2 HF/TRT tensor dump manifests and raw tensor payloads."""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path
from typing import Any


TensorKey = tuple[str, int, str, str]


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            record["_line"] = line_no
            records.append(record)
    return records


def _record_key(record: dict[str, Any]) -> TensorKey:
    return (
        str(record["phase"]),
        int(record["step"]),
        str(record["direction"]),
        str(record["name"]),
    )


def _shape(record: dict[str, Any]) -> list[int]:
    return [int(dim) for dim in record.get("shape", [])]


def _read_raw(record: dict[str, Any]) -> bytes:
    return Path(record["path"]).read_bytes()


def _bf16_to_float(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", (bits & 0xFFFF) << 16))[0]


def _float16_to_float(bits: int) -> float:
    sign = (bits >> 15) & 0x1
    exponent = (bits >> 10) & 0x1F
    fraction = bits & 0x3FF
    if exponent == 0:
        value = 0.0 if fraction == 0 else (fraction / 1024.0) * (2.0 ** -14)
    elif exponent == 0x1F:
        value = math.inf if fraction == 0 else math.nan
    else:
        value = (1.0 + fraction / 1024.0) * (2.0 ** (exponent - 15))
    return -value if sign else value


def _decode_element(raw: bytes, dtype: str, element_index: int) -> int | float | None:
    if dtype == "bfloat16":
        offset = element_index * 2
        return _bf16_to_float(struct.unpack_from("<H", raw, offset)[0])
    if dtype == "float16":
        offset = element_index * 2
        return _float16_to_float(struct.unpack_from("<H", raw, offset)[0])
    if dtype == "float32":
        offset = element_index * 4
        return float(struct.unpack_from("<f", raw, offset)[0])
    if dtype == "int32":
        offset = element_index * 4
        return int(struct.unpack_from("<i", raw, offset)[0])
    if dtype == "int64":
        offset = element_index * 8
        return int(struct.unpack_from("<q", raw, offset)[0])
    if dtype == "int8":
        return int(struct.unpack_from("<b", raw, element_index)[0])
    if dtype == "uint8":
        return int(raw[element_index])
    return None


def _element_size(dtype: str) -> int:
    if dtype in {"bfloat16", "float16"}:
        return 2
    if dtype in {"float32", "int32"}:
        return 4
    if dtype == "int64":
        return 8
    if dtype in {"int8", "uint8"}:
        return 1
    return 1


def _is_float_dtype(dtype: str) -> bool:
    return dtype in {"bfloat16", "float16", "float32"}


def _element_count(shape: list[int]) -> int:
    count = 1
    for dim in shape:
        count *= dim
    return count


def _first_byte_difference(left: bytes, right: bytes) -> int | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def _first_value_difference(
    hf_raw: bytes,
    hf_dtype: str,
    trt_raw: bytes,
    trt_dtype: str,
    count: int,
) -> int | None:
    for index in range(count):
        hf_value = _decode_element(hf_raw, hf_dtype, index)
        trt_value = _decode_element(trt_raw, trt_dtype, index)
        if isinstance(hf_value, float) and isinstance(trt_value, float):
            if math.isnan(hf_value) and math.isnan(trt_value):
                continue
        if hf_value != trt_value:
            return index
    return None


def _tensor_mismatch(
    hf_record: dict[str, Any],
    trt_record: dict[str, Any],
    key: TensorKey,
) -> dict[str, Any] | None:
    metadata_differences: list[dict[str, Any]] = []
    hf_dtype = str(hf_record.get("dtype", ""))
    trt_dtype = str(trt_record.get("dtype", ""))
    hf_shape = _shape(hf_record)
    trt_shape = _shape(trt_record)
    hf_nbytes = int(hf_record.get("nbytes", -1))
    trt_nbytes = int(trt_record.get("nbytes", -1))

    compatible_float_dtype_mismatch = (
        hf_dtype != trt_dtype
        and _is_float_dtype(hf_dtype)
        and _is_float_dtype(trt_dtype)
        and hf_shape == trt_shape
    )

    if hf_dtype != trt_dtype and not compatible_float_dtype_mismatch:
        metadata_differences.append({"field": "dtype", "hf": hf_dtype, "trt": trt_dtype})
    if hf_shape != trt_shape:
        metadata_differences.append({"field": "shape", "hf": hf_shape, "trt": trt_shape})
    if hf_nbytes != trt_nbytes and not compatible_float_dtype_mismatch:
        metadata_differences.append({"field": "nbytes", "hf": hf_nbytes, "trt": trt_nbytes})

    if metadata_differences:
        return {
            "key": list(key),
            "hf_line": hf_record["_line"],
            "trt_line": trt_record["_line"],
            "hf_engine_section": str(hf_record.get("engine_section", "")),
            "trt_engine_section": str(trt_record.get("engine_section", "")),
            "metadata_differences": metadata_differences,
        }

    hf_raw = _read_raw(hf_record)
    trt_raw = _read_raw(trt_record)
    if compatible_float_dtype_mismatch:
        element_index = _first_value_difference(
            hf_raw,
            hf_dtype,
            trt_raw,
            trt_dtype,
            _element_count(hf_shape),
        )
        if element_index is None:
            return None
        return {
            "key": list(key),
            "hf_line": hf_record["_line"],
            "trt_line": trt_record["_line"],
            "hf_engine_section": str(hf_record.get("engine_section", "")),
            "trt_engine_section": str(trt_record.get("engine_section", "")),
            "hf_dtype": hf_dtype,
            "trt_dtype": trt_dtype,
            "shape": hf_shape,
            "first_different_element": element_index,
            "hf_value": _decode_element(hf_raw, hf_dtype, element_index),
            "trt_value": _decode_element(trt_raw, trt_dtype, element_index),
        }

    if hf_raw == trt_raw:
        return None

    byte_offset = _first_byte_difference(hf_raw, trt_raw)
    element_size = _element_size(hf_dtype)
    element_index = None if byte_offset is None else byte_offset // element_size
    mismatch: dict[str, Any] = {
        "key": list(key),
        "hf_line": hf_record["_line"],
        "trt_line": trt_record["_line"],
        "hf_engine_section": str(hf_record.get("engine_section", "")),
        "trt_engine_section": str(trt_record.get("engine_section", "")),
        "dtype": hf_dtype,
        "shape": hf_shape,
        "nbytes": hf_nbytes,
        "first_different_byte": byte_offset,
        "first_different_element": element_index,
    }
    if element_index is not None:
        mismatch["hf_value"] = _decode_element(hf_raw, hf_dtype, element_index)
        mismatch["trt_value"] = _decode_element(trt_raw, hf_dtype, element_index)
    return mismatch


def compare_tensor_dumps(
    hf_manifest: Path,
    trt_manifest: Path,
    *,
    strict_extra: bool = False,
    max_mismatches: int = 5,
) -> dict[str, Any]:
    hf_records = _load_manifest(hf_manifest)
    trt_records = _load_manifest(trt_manifest)
    hf_by_key = {_record_key(record): record for record in hf_records}
    trt_by_key = {_record_key(record): record for record in trt_records}

    common_keys = [
        _record_key(record)
        for record in hf_records
        if _record_key(record) in trt_by_key
    ]
    missing_from_trt = [
        _record_key(record)
        for record in hf_records
        if _record_key(record) not in trt_by_key
    ]
    extra_trt = [
        _record_key(record)
        for record in trt_records
        if _record_key(record) not in hf_by_key
    ]

    first_mismatch = None
    first_mismatches: list[dict[str, Any]] = []
    mismatch_counts_by_phase: dict[str, int] = {}
    mismatch_counts_by_trt_engine_section: dict[str, int] = {}
    mismatch_count = 0
    for key in common_keys:
        mismatch = _tensor_mismatch(hf_by_key[key], trt_by_key[key], key)
        if mismatch is None:
            continue
        mismatch_count += 1
        phase = key[0]
        trt_engine_section = str(mismatch.get("trt_engine_section", ""))
        mismatch_counts_by_phase[phase] = mismatch_counts_by_phase.get(phase, 0) + 1
        mismatch_counts_by_trt_engine_section[trt_engine_section] = (
            mismatch_counts_by_trt_engine_section.get(trt_engine_section, 0) + 1
        )
        if first_mismatch is None:
            first_mismatch = mismatch
        if len(first_mismatches) < max(0, max_mismatches):
            first_mismatches.append(mismatch)

    passed = (
        not missing_from_trt
        and mismatch_count == 0
        and (not strict_extra or not extra_trt)
    )
    return {
        "passed": passed,
        "hf_manifest": str(hf_manifest),
        "trt_manifest": str(trt_manifest),
        "record_counts": {
            "hf": len(hf_records),
            "trt": len(trt_records),
            "common": len(common_keys),
            "missing_from_trt": len(missing_from_trt),
            "extra_trt": len(extra_trt),
            "common_mismatches": mismatch_count,
        },
        "first_missing_from_trt": [list(key) for key in missing_from_trt[:5]],
        "first_extra_trt": [list(key) for key in extra_trt[:5]],
        "first_common_mismatch": first_mismatch,
        "first_common_mismatches": first_mismatches,
        "mismatch_counts_by_phase": mismatch_counts_by_phase,
        "mismatch_counts_by_trt_engine_section": mismatch_counts_by_trt_engine_section,
        "strict_extra": strict_extra,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hf_manifest", type=Path)
    parser.add_argument("trt_manifest", type=Path)
    parser.add_argument(
        "--strict-extra",
        action="store_true",
        help="fail if TRT emits records that are not present in the HF manifest",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="write the comparison result JSON to this path as well as stdout",
    )
    parser.add_argument(
        "--max-mismatches",
        type=int,
        default=5,
        help="number of early common tensor mismatches to preserve in result JSON",
    )
    args = parser.parse_args(argv)

    result = compare_tensor_dumps(
        args.hf_manifest,
        args.trt_manifest,
        strict_extra=args.strict_extra,
        max_mismatches=args.max_mismatches,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
