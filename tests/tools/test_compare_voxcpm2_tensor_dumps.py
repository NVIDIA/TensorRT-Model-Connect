from __future__ import annotations

import importlib.util
import json
import struct
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
) -> None:
    raw_path = root / f"{phase}_{step:06d}_{direction}_{name}.raw"
    raw_path.write_bytes(raw)
    record = {
        "stage": phase.split("_", 1)[0],
        "engine_section": "test",
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
    assert mismatch["first_different_element"] == 1
    assert mismatch["hf_value"] == 2.0
    assert mismatch["trt_value"] == 4.0
