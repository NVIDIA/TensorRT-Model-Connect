from __future__ import annotations

import importlib.util
import json
import struct
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "tools" / "compare_voxcpm2_tensor_dumps.py"


def _load_tool() -> Any:
    spec = importlib.util.spec_from_file_location("compare_voxcpm2_tensor_dumps", TOOL_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_record(
    root: Path,
    manifest: Path,
    *,
    phase: str,
    step: int,
    direction: str,
    name: str,
    dtype: str,
    shape: list[int],
    raw: bytes,
    engine_section: str = "test",
) -> None:
    raw_path = root / f"{phase}_{step:06d}_{direction}_{name}.raw"
    raw_path.write_bytes(raw)
    record = {
        "stage": phase.split("_", 1)[0],
        "engine_section": engine_section,
        "phase": phase,
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


def test_compare_voxcpm2_tensor_dumps_allows_extra_trt_records(tmp_path: Path) -> None:
    tool = _load_tool()
    hf_dir = tmp_path / "hf"
    trt_dir = tmp_path / "trt"
    hf_dir.mkdir()
    trt_dir.mkdir()
    hf_manifest = hf_dir / "manifest.jsonl"
    trt_manifest = trt_dir / "manifest.jsonl"
    raw = struct.pack("<i", 42)

    _write_record(
        hf_dir,
        hf_manifest,
        phase="tslm_prefill",
        step=0,
        direction="input",
        name="text_tokens",
        dtype="int32",
        shape=[1],
        raw=raw,
    )
    _write_record(
        trt_dir,
        trt_manifest,
        phase="tslm_prefill",
        step=0,
        direction="input",
        name="text_tokens",
        dtype="int32",
        shape=[1],
        raw=raw,
    )
    _write_record(
        trt_dir,
        trt_manifest,
        phase="tslm_prefill",
        step=0,
        direction="output",
        name="stop_logits",
        dtype="bfloat16",
        shape=[1, 2],
        raw=b"\x00\x00\x00\x00",
    )

    result = tool.compare_tensor_dumps(hf_manifest, trt_manifest)

    assert result["passed"] is True
    assert result["record_counts"]["extra_trt"] == 1


def test_compare_voxcpm2_tensor_dumps_reports_first_common_mismatch(tmp_path: Path) -> None:
    tool = _load_tool()
    hf_dir = tmp_path / "hf"
    trt_dir = tmp_path / "trt"
    hf_dir.mkdir()
    trt_dir.mkdir()
    hf_manifest = hf_dir / "manifest.jsonl"
    trt_manifest = trt_dir / "manifest.jsonl"

    _write_record(
        hf_dir,
        hf_manifest,
        phase="tslm_prefill",
        step=0,
        direction="output",
        name="semantic_lm_states",
        dtype="bfloat16",
        shape=[1, 2],
        raw=struct.pack("<HH", 0x3F80, 0x4000),
    )
    _write_record(
        trt_dir,
        trt_manifest,
        phase="tslm_prefill",
        step=0,
        direction="output",
        name="semantic_lm_states",
        dtype="bfloat16",
        shape=[1, 2],
        raw=struct.pack("<HH", 0x3F80, 0x4080),
    )

    result = tool.compare_tensor_dumps(hf_manifest, trt_manifest)

    assert result["passed"] is False
    assert result["record_counts"]["common_mismatches"] == 1
    mismatch = result["first_common_mismatch"]
    assert mismatch["key"] == [
        "tslm_prefill",
        0,
        "output",
        "semantic_lm_states",
    ]
    assert mismatch["hf_engine_section"] == "test"
    assert mismatch["trt_engine_section"] == "test"
    assert mismatch["first_different_element"] == 1
    assert mismatch["first_different_coordinate"] == [0, 1]
    assert mismatch["different_elements"] == 1
    assert mismatch["total_elements"] == 2
    assert mismatch["max_abs_diff"] == 2.0
    assert mismatch["hf_value"] == 2.0
    assert mismatch["trt_value"] == 4.0
    assert mismatch["bf16_bit_delta_counts"] == {"+128": 1}
    assert mismatch["bf16_mismatch_examples"] == [
        {
            "element": 1,
            "bit_delta": 128,
            "expected_bits": "0x4000",
            "actual_bits": "0x4080",
            "expected_value": 2.0,
            "actual_value": 4.0,
        }
    ]


def test_compare_voxcpm2_tensor_dumps_reports_mismatch_list_and_counts(
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    hf_dir = tmp_path / "hf"
    trt_dir = tmp_path / "trt"
    hf_dir.mkdir()
    trt_dir.mkdir()
    hf_manifest = hf_dir / "manifest.jsonl"
    trt_manifest = trt_dir / "manifest.jsonl"

    _write_record(
        hf_dir,
        hf_manifest,
        phase="tslm_prefill",
        step=0,
        direction="output",
        name="semantic_lm_states",
        dtype="bfloat16",
        shape=[1],
        raw=struct.pack("<H", 0x3F80),
        engine_section="hf_reference",
    )
    _write_record(
        trt_dir,
        trt_manifest,
        phase="tslm_prefill",
        step=0,
        direction="output",
        name="semantic_lm_states",
        dtype="bfloat16",
        shape=[1],
        raw=struct.pack("<H", 0x4000),
        engine_section="tslm_prefill_engine_plan",
    )
    _write_record(
        hf_dir,
        hf_manifest,
        phase="ralm_prefill",
        step=0,
        direction="output",
        name="residual_hidden",
        dtype="bfloat16",
        shape=[1],
        raw=struct.pack("<H", 0x4000),
        engine_section="hf_reference",
    )
    _write_record(
        trt_dir,
        trt_manifest,
        phase="ralm_prefill",
        step=0,
        direction="output",
        name="residual_hidden",
        dtype="bfloat16",
        shape=[1],
        raw=struct.pack("<H", 0x4080),
        engine_section="ralm_prefill_engine_plan",
    )

    result = tool.compare_tensor_dumps(hf_manifest, trt_manifest, max_mismatches=1)

    assert result["record_counts"]["common_mismatches"] == 2
    assert len(result["first_common_mismatches"]) == 1
    assert result["first_common_mismatches"][0] == result["first_common_mismatch"]
    assert result["mismatch_counts_by_phase"] == {
        "ralm_prefill": 1,
        "tslm_prefill": 1,
    }
    assert result["mismatch_counts_by_trt_engine_section"] == {
        "ralm_prefill_engine_plan": 1,
        "tslm_prefill_engine_plan": 1,
    }


def test_compare_voxcpm2_tensor_dumps_reports_bf16_location_summary(
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    hf_dir = tmp_path / "hf"
    trt_dir = tmp_path / "trt"
    hf_dir.mkdir()
    trt_dir.mkdir()
    hf_manifest = hf_dir / "manifest.jsonl"
    trt_manifest = trt_dir / "manifest.jsonl"

    _write_record(
        hf_dir,
        hf_manifest,
        phase="tslm_prefill",
        step=0,
        direction="output",
        name="semantic_lm_states",
        dtype="bfloat16",
        shape=[3, 4],
        raw=struct.pack(
            "<" + "H" * 12,
            0x3F80,
            0x4000,
            0x4080,
            0x4100,
            0x3F80,
            0x4000,
            0x4080,
            0x4100,
            0x3F80,
            0x4000,
            0x4080,
            0x4100,
        ),
    )
    _write_record(
        trt_dir,
        trt_manifest,
        phase="tslm_prefill",
        step=0,
        direction="output",
        name="semantic_lm_states",
        dtype="bfloat16",
        shape=[3, 4],
        raw=struct.pack(
            "<" + "H" * 12,
            0x3F80,
            0x4001,
            0x4080,
            0x4100,
            0x3F80,
            0x4000,
            0x407F,
            0x4100,
            0x3F80,
            0x4002,
            0x4080,
            0x4100,
        ),
    )

    result = tool.compare_tensor_dumps(hf_manifest, trt_manifest)

    mismatch = result["first_common_mismatch"]
    assert mismatch["first_different_element"] == 1
    assert mismatch["first_different_coordinate"] == [0, 1]
    assert mismatch["different_elements"] == 3
    assert mismatch["total_elements"] == 12
    assert mismatch["bf16_adjacent_ulp_mismatches"] == 2
    assert mismatch["bf16_bit_delta_counts"] == {"+1": 1, "+2": 1, "-1": 1}
    assert mismatch["mismatch_rows_with_differences"] == 3
    assert mismatch["mismatch_cols_with_differences"] == 2
    assert mismatch["top_mismatch_rows"] == [
        {"row": 0, "count": 1},
        {"row": 1, "count": 1},
        {"row": 2, "count": 1},
    ]
    assert mismatch["top_mismatch_cols"] == [
        {"column": 1, "count": 2},
        {"column": 2, "count": 1},
    ]


def test_compare_voxcpm2_tensor_dumps_compares_matching_float_values_across_dtypes(
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    hf_dir = tmp_path / "hf"
    trt_dir = tmp_path / "trt"
    hf_dir.mkdir()
    trt_dir.mkdir()
    hf_manifest = hf_dir / "manifest.jsonl"
    trt_manifest = trt_dir / "manifest.jsonl"

    _write_record(
        hf_dir,
        hf_manifest,
        phase="tslm_prefill",
        step=0,
        direction="input",
        name="text_mask",
        dtype="bfloat16",
        shape=[2],
        raw=struct.pack("<HH", 0x3F80, 0x0000),
    )
    _write_record(
        trt_dir,
        trt_manifest,
        phase="tslm_prefill",
        step=0,
        direction="input",
        name="text_mask",
        dtype="float32",
        shape=[2],
        raw=struct.pack("<ff", 1.0, 0.0),
    )

    result = tool.compare_tensor_dumps(hf_manifest, trt_manifest)

    assert result["passed"] is True
    assert result["record_counts"]["common_mismatches"] == 0


def test_compare_voxcpm2_tensor_dumps_reports_float_value_mismatch_across_dtypes(
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    hf_dir = tmp_path / "hf"
    trt_dir = tmp_path / "trt"
    hf_dir.mkdir()
    trt_dir.mkdir()
    hf_manifest = hf_dir / "manifest.jsonl"
    trt_manifest = trt_dir / "manifest.jsonl"

    _write_record(
        hf_dir,
        hf_manifest,
        phase="tslm_prefill",
        step=0,
        direction="input",
        name="local_text_features",
        dtype="bfloat16",
        shape=[2],
        raw=struct.pack("<HH", 0x3F80, 0x4000),
    )
    _write_record(
        trt_dir,
        trt_manifest,
        phase="tslm_prefill",
        step=0,
        direction="input",
        name="local_text_features",
        dtype="float32",
        shape=[2],
        raw=struct.pack("<ff", 1.0, 4.0),
    )

    result = tool.compare_tensor_dumps(hf_manifest, trt_manifest)

    assert result["passed"] is False
    mismatch = result["first_common_mismatch"]
    assert mismatch["key"] == [
        "tslm_prefill",
        0,
        "input",
        "local_text_features",
    ]
    assert mismatch["hf_engine_section"] == "test"
    assert mismatch["trt_engine_section"] == "test"
    assert mismatch["hf_dtype"] == "bfloat16"
    assert mismatch["trt_dtype"] == "float32"
    assert mismatch["first_different_element"] == 1
    assert mismatch["hf_value"] == 2.0
    assert mismatch["trt_value"] == 4.0


def test_compare_voxcpm2_tensor_dumps_main_writes_output_json(
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    hf_dir = tmp_path / "hf"
    trt_dir = tmp_path / "trt"
    hf_dir.mkdir()
    trt_dir.mkdir()
    hf_manifest = hf_dir / "manifest.jsonl"
    trt_manifest = trt_dir / "manifest.jsonl"
    output_json = tmp_path / "compare.json"

    _write_record(
        hf_dir,
        hf_manifest,
        phase="tslm_prefill",
        step=0,
        direction="output",
        name="semantic_lm_states",
        dtype="bfloat16",
        shape=[1],
        raw=struct.pack("<H", 0x3F80),
    )
    _write_record(
        trt_dir,
        trt_manifest,
        phase="tslm_prefill",
        step=0,
        direction="output",
        name="semantic_lm_states",
        dtype="bfloat16",
        shape=[1],
        raw=struct.pack("<H", 0x4000),
    )

    stdout = StringIO()
    with redirect_stdout(stdout):
        rc = tool.main(
            [
                str(hf_manifest),
                str(trt_manifest),
                "--max-mismatches",
                "1",
                "--output-json",
                str(output_json),
            ]
        )

    assert rc == 1
    written = json.loads(output_json.read_text(encoding="utf-8"))
    printed = json.loads(stdout.getvalue())
    assert written == printed
    assert written["first_common_mismatch"]["trt_engine_section"] == "test"
    assert len(written["first_common_mismatches"]) == 1
